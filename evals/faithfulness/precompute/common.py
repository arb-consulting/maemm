"""The only module shared across evals/faithfulness: config, model loading, injection, reading, scoring, io.

Everything that has to match the pipeline lives here once, so rollouts, GCG and the
reconstruction scripts cannot drift on the objective. Each convention carries the file:line in the upstream
repo it was copied from (verified against the upstream master on 2026-09-15); where we deliberately
differ, the divergence is named in the comment.

Torch, transformers and peft are imported lazily inside the functions that need them: the local
unit smoke and the CPU `check` product must import this module without a GPU stack present in the
call path, and the Modal image provides them at call time anyway.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# constants that are part of the protocol
# ---------------------------------------------------------------------------------------------

VOL = "/vol"
HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
CONFIG_PATH = PAPER_EVALS / "config.yaml"


# ---------------------------------------------------------------------------------------------
# where a LOCAL reader may write: one place, and it is outside the tree Modal mounts
# ---------------------------------------------------------------------------------------------
#
# NOTHING A READER WRITES MAY LAND UNDER `evals/faithfulness/`, AND THAT IS NOT A STYLE RULE.
# `precompute/modal_app.py` mounts the whole `evals/faithfulness/` tree into every image with
# `add_local_dir(..., copy=True)`, and Modal hashes the tree as it builds. Any file appearing or
# changing under it mid-build kills the launch with `<file> was modified during build process`.
# That is how the 2026-09-23 L14 scoring job was lost, at the cost of a relaunch: not a concurrent
# session editing code -- the diagnosis SMOKES.md recorded for the same failure on 2026-09-16 --
# but a READER'S OWN FETCH MIRROR, `results/data/`, filling up beside it. Running
# readers next to paid Modal jobs makes this a live, recurring, expensive defect, so the defaults
# live here, in ONE place, and `mirror_dir`/`out_dir` refuse a path under the mount whatever it
# came from.
#
# Two kinds, because they are different things and want different homes:
#   * the MIRROR is a pure cache of volume bytes, reproducible by re-fetching and shared by every
#     checkout of every branch (the volume is one volume), so it belongs in the user cache;
#   * the OUTPUT is a work product a person opens -- tables, CSVs, figures -- so it belongs beside
#     the repo, where it can be found, and NOT in a cache directory that a cleaner may empty.
# An explicit `--data` / `--out` still overrides either, and the committed product directories
# (`results/ood/`, `results/faithfulness/`, `results/patchscopes/`, `results/tierb/`) are still
# written by naming them on the command line -- deliberately, at a moment the operator chose,
# which is exactly what a DEFAULT cannot be.

MIRROR_ENV = "MAEM_MIRROR"
OUT_ENV = "MAEM_OUT"


def _outside_the_mount(p: Path, what: str, how: str) -> Path:
    """`p`, resolved, or an assertion naming what would have broken."""
    p = Path(p).expanduser().resolve()
    assert not p.is_relative_to(PAPER_EVALS), (
        f"{what} resolves to {p}, which is INSIDE {PAPER_EVALS} -- the tree "
        f"`precompute/modal_app.py` mounts into every image with copy=True. A reader writing "
        f"there kills any Modal launch racing it with `was modified during build process`. "
        f"{how}"
    )
    return p


def mirror_dir(root: str = "") -> Path:
    """The default local mirror of the Modal volume, for volume-relative prefix `root`.

    `${MAEM_MIRROR}/<root>` when that is set -- which is how several worktrees share one cache --
    otherwise `$XDG_CACHE_HOME/maem-faithfulness/mirror/<root>` (`~/.cache` when XDG is unset),
    the path `results/patchscopes.py` took first on 2026-09-23. `root` is the volume-relative
    prefix the reader was given, slashes flattened; empty means the volume root.
    """
    slug = root.strip("/").replace("/", "_") or "vol"
    base = os.environ.get(MIRROR_ENV) or (
        Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "maem-faithfulness"
        / "mirror"
    )
    return _outside_the_mount(
        Path(base) / slug, f"the default mirror for root {root or '(volume root)'!r}",
        f"Point ${MIRROR_ENV} somewhere else, or pass the mirror explicitly.",
    )


def out_dir(tool: str) -> Path:
    """The default output directory for reader `tool` (`faithfulness`, `ood`, ...).

    `${MAEM_OUT}/<tool>` when that is set, otherwise `<repo>/_out/<tool>` -- beside
    `evals/faithfulness/`, never inside it, and gitignored at the repo root. A committed product
    directory is reached by naming it: `--out results/ood`.
    """
    assert tool and "/" not in tool, f"out_dir takes a bare tool name, not {tool!r}"
    base = Path(os.environ.get(OUT_ENV) or (PAPER_EVALS.parent / "_out"))
    return _outside_the_mount(
        base / tool, f"the default output directory for {tool!r}",
        f"Point ${OUT_ENV} somewhere else, or pass `--out` explicitly.",
    )

# evals/heldout/eval_universal.py:138 -- the re-encode truncation. Checklist item 7: it must leave room for
# the whole rollout plus the prepended sink, i.e. max_length >= rollouts.max_new + 1; asserted in
# load_config() so a later max_new bump cannot silently truncate the scored window.
SCORE_MAX_LENGTH = 95
# evals/heldout/eval_universal.py:129 sbatch. ONE fixed scoring chunk for every product (checklist item 11:
# right-padding chunk size measurably shifts per-row cosine), stated in every scores README.
SCORE_CHUNK = 32
# Column 0 of every scored array is the BOS sink, never a candidate token; width is fixed so that
# chunks of different token lengths concatenate into one rectangular array.
SCORE_WIDTH = SCORE_MAX_LENGTH + 1

MARKER = " ?"  # maem/prompts.py:4

# Input-amplitude conventions of the `nla` verbalizer (`precompute/rollouts_nla.py`, which aliases
# this tuple and documents what each one does). It lives here because load_config validates
# `nla.amp` and must not import a product module to do it.
AMP_MODES = ("exact", "mu", "raw")

# --- the centring vocabulary (2026-09-21, branch `evals/pipeline`) -----------------------------
# A MU IS A FILE. Wherever a centring mean appears -- `maems.<k>.mu`, `bases.<base>.whiten_mu`,
# `--mu` -- the value is null (subtract nothing), or a path to a [d] `.f32` / `.npy` file, or the
# string `unknown` (a checkpoint's own `mu:` only). There is no enum and no registry of mean
# NAMES: a checkpoint trained on a new mean is a new path in a config entry, and nothing else
# changes. That is what lets a new SAE / MAEM land as a config-only edit.
#
# TWO AXES SINCE 2026-09-23 (M0a). `maems.<k>.mu` (`input_mu`) is the INJECTION convention -- what
# a checkpoint was trained to receive. `bases.<base>.whiten_mu` (`score_mu`) is the SCORING
# CONSTANT, the mean both arguments of every centred cosine are taken about, a property of the
# base and of no run. The per-set `mu_stored` / `family_mu` layer that used to name a third thing
# is deleted: a stored unit direction cannot be re-centred, so such a set has no centred reading.
FAMILY_KINDS = ("activation", "synthetic", "dictionary", "subspace")
STORAGE_KINDS = ("raw", "unit", "dirs_only")
# Products that WRITE a held-out set. They must be told which by name -- D6. An omitted --set
# used to resolve to `default_heldout(cfg)`, the LIVE set every table is built on, and `--force`
# would then rmtree it. There is no safe default for "where do I write a new set".
#
# HERE, not in modal_app, because `features/spawn.py` enforces the same guard and bypasses
# modal_app entirely -- which is how the hazard reached the volume in the first place. Two copies
# of this tuple had already drifted apart by 2026-09-21: modal_app knew about `heldout_v3` and
# spawn did not, and neither knew about `draw_sae131k`.
SET_WRITERS = ("targets", "draw_dict2m", "draw_sae131k", "heldout_v3")
# "considered, not established" -- legal in a checkpoint's own `maems.<k>.mu` only. Every run of
# such a checkpoint must be told the convention with --mu, recorded as a choice, not a reading.
MU_UNKNOWN = "unknown"
# Accepted on-disk forms of a mean. Anything else is a typo, not a format.
MU_SUFFIXES = (".f32", ".npy")
# OUR 64/16-window read-layer mean, as a path rather than a name -- root-relative and
# base-templated, so a smoke gets its own. It is ONE mean among several, not "the" one; nothing
# centres on it unless a `mu:` / `--mu` says so.
STATS_MU = "base/{base}/stats/mu.f32"
# The file a `targets`-written set carries its own storage contract in. It is an ordinary product
# file (listed in index.json like any other), NOT a second sidecar: `common.set_storage` reads it
# when present and falls back to config.yaml's `heldout.<set>` block for every set drawn before it
# existed, and for a `--dirs-from` directory that is neither.
STORAGE_FILE = "storage.json"

# The held-out set name `targets.py --import-run1` writes. It is NOT a config.yaml `heldout` entry:
# it is not a draw of ours at all but a 16-row slice of run1's archived eval cache, and it exists
# only so reconstruction/repro_run1.py can compare our pipeline against the archived numbers.
IMPORT_RUN1_SET = "2026-09-03_run1-archive16"

# Corpus scan geometry (checklist item 57: block size and stride are results-affecting, so they are
# named once here and quoted in every README). The old retrieval baseline used BLOCK 64 / STRIDE 32
# (eval/corpus_retrieval.py:89-91); ours is stride 16, i.e. 4x the coverage per token.
# THE window geometry of every scan in this pipeline. It is a CONSTANT, not a default: eleven
# `windows_of(` call sites across stats, scan, top1_act, sae_self, build and gcg reconstruct the
# same window ids to join on, and they all take it from here. `corpora:` DECLARES a per-corpus
# block/stride so a corpus built elsewhere (the train_parity_10m, 32/8) is described honestly --
# but declaring is not threading, and `assert_corpus_geometry` REFUSES such a corpus rather than
# scanning it at 64/16 and writing a README that says 32/8. See H7 in CHANGES-pipeline.md.
SCAN_BLOCK = 64
SCAN_STRIDE = 16

# maem/prompts.py:8-15, verbatim except the read layer, which the template names in words and which
# therefore differs per base ("layer-27" for the 8B, "layer-42" for the 27B).
_INSTR_TEMPLATE = (
    "You are an interpretability research tool. Researchers have injected a single linear probe "
    "direction from this model's own layer-{read_layer} residual stream. To help them read off "
    "what that direction represents, write one short text "
    "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as possible. "
    "Output only the snippet itself, with no explanation, preamble, or quotation marks. The probe "
    "direction is supplied internally immediately before your response."
)


# ---------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    """Parse and validate config.yaml. Every cross-reference is checked here, not at use site."""
    import yaml

    path = Path(path) if path else CONFIG_PATH
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    for key in (
        "bases",
        "corpora",
        "family_kinds",
        "saes",
        "maems",
        "heldout",
        "corpus",
        "rollouts",
        "modal",
    ):
        assert key in cfg, f"config {path} is missing the top-level key {key!r}"

    max_new = cfg["rollouts"]["max_new"]
    assert SCORE_MAX_LENGTH >= max_new + 1, (
        f"scoring truncation SCORE_MAX_LENGTH={SCORE_MAX_LENGTH} must be >= rollouts.max_new + 1 "
        f"= {max_new + 1}, else the tail of a full-length rollout is never scored"
    )

    for base, spec in cfg["bases"].items():
        for key in ("hf", "read_layer", "d", "gpu", "n_layers"):
            assert key in spec, f"base {base!r} is missing {key!r}"
        assert spec["read_layer"] < spec["n_layers"], (
            f"base {base!r}: read_layer {spec['read_layer']} must be < n_layers {spec['n_layers']}"
        )
        if "whiten_mu" in spec:
            _check_mu_value(spec["whiten_mu"], f"bases[{base!r}].whiten_mu", allow_unknown=False)

    # The OOD arm corpora are `corpora:` entries like any other -- declared ONCE, in `ood_arms:`,
    # because that is where the arm's source, ladder and licence already live. Synthesising them
    # here rather than writing 23 more blocks by hand keeps one source for the ladder: a corpus
    # built at [1, 4] and declared at [1, 4, 16] somewhere else is exactly the silent mismatch
    # `assert_corpus_geometry` and the `dirs` uniqueness check below exist to catch.
    # Geometry is the pipeline's 64/16 (common.SCAN_BLOCK/SCAN_STRIDE), which is also the design's
    # (infra/2026-09-18_ood-eval-design.md §1: "64-token windows at stride 16"), so no OOD scan
    # differs from the English one in anything but the text.
    for arm, aspec in (cfg.get("ood_arms") or {}).items():
        key = f"ood_{arm}"
        assert key not in cfg["corpora"], (
            f"corpora[{key!r}] is declared by hand AND synthesised from ood_arms[{arm!r}]; one of "
            f"the two ladders would silently win"
        )
        cfg["corpora"][key] = {
            "dir": arm,
            "sizes": list(aspec["sizes"]),
            "block": SCAN_BLOCK,
            "stride": SCAN_STRIDE,
            "dataset": aspec["dataset"],
            "note": (
                f"OOD arm {arm!r} (family {aspec['family']}), its own in-domain corpus, built by "
                f"`--product corpus --arm {arm}` from the same permuted row stream as the arm's "
                f"targets and disjoint from them by construction (design §2, §4)."
            ),
        }

    for key, cspec in cfg["corpora"].items():
        assert isinstance(cspec, dict), f"corpora[{key!r}] must be a mapping, got {cspec!r}"
        for field in ("dir", "sizes", "block", "stride"):
            assert field in cspec, f"corpora[{key!r}] is missing {field!r}"
        assert isinstance(cspec["dir"], str) and cspec["dir"] and "/" not in cspec["dir"], (
            f"corpora[{key!r}].dir must be a single directory name ('corpus' for the original "
            f"base/<base>/corpus/, anything else for corpora/<dir>/), got {cspec['dir']!r}"
        )
        sizes = cspec["sizes"]
        assert isinstance(sizes, list) and sizes and sizes == sorted(sizes), (
            f"corpora[{key!r}].sizes must be an ascending list of nested sizes in millions of "
            f"tokens, got {sizes!r}"
        )
        for field in ("block", "stride"):
            assert isinstance(cspec[field], int) and cspec[field] > 0, (
                f"corpora[{key!r}].{field} must be a positive int, got {cspec[field]!r}"
            )
        assert cspec["stride"] <= cspec["block"], (
            f"corpora[{key!r}]: stride {cspec['stride']} > block {cspec['block']} would leave gaps "
            f"between windows, so part of the corpus would never be scanned"
        )
    dirs = [c["dir"] for c in cfg["corpora"].values()]
    assert len(set(dirs)) == len(dirs), (
        f"two `corpora:` keys point at the same directory {sorted(dirs)}: a different ladder or "
        f"window geometry is a DIFFERENT corpus, never a second name for one"
    )

    for fam, fspec in cfg["family_kinds"].items():
        assert isinstance(fspec, dict) and sorted(fspec) == ["centrable", "kind"], (
            f"family_kinds[{fam!r}] must carry exactly centrable and kind, got {sorted(fspec)}"
        )
        assert isinstance(fspec["centrable"], bool), (
            f"family_kinds[{fam!r}]: centrable must be a bool, got {fspec['centrable']!r}"
        )
        assert fspec["kind"] in FAMILY_KINDS, (
            f"family_kinds[{fam!r}]: kind must be one of {list(FAMILY_KINDS)}, got {fspec['kind']!r}"
        )
        assert fspec["centrable"] == (fspec["kind"] == "activation"), (
            f"family_kinds[{fam!r}]: only an `activation` family has a mean to subtract, so "
            f"centrable and kind == 'activation' must agree; got {fspec}"
        )

    for key, spec in cfg["saes"].items():
        base, _ = split_key(key, "sae")
        assert base in cfg["bases"], f"sae {key!r} names base {base!r}, which is not in config bases"
        for field in ("hf", "file", "layer"):
            assert field in spec, f"sae {key!r} is missing {field!r}"
        assert spec["layer"] == cfg["bases"][base]["read_layer"], (
            f"sae {key!r} is at layer {spec['layer']} but base {base!r} reads at "
            f"{cfg['bases'][base]['read_layer']}; the SAE must live at the read layer"
        )
        ma = spec.get("max_acts")
        if ma is not None:
            for field in ("file", "windows", "sink_first"):
                assert field in ma, f"sae {key!r}: max_acts is missing {field!r}"
            assert isinstance(ma["sink_first"], bool), (
                f"sae {key!r}: max_acts.sink_first must be a bool (whether EVERY shipped window "
                f"begins with the tokenizer's sink token), got {ma['sink_first']!r}"
            )
            assert ma.get("repo_type", "model") in ("model", "dataset"), (
                f"sae {key!r}: max_acts.repo_type must be 'model' or 'dataset', got {ma['repo_type']!r}"
            )

    for key, spec in cfg["maems"].items():
        base, _ = split_key(key, "maem")
        assert base in cfg["bases"], f"maem {key!r} names base {base!r}, which is not in config bases"
        # `base` is the UNTRAINED-BASE CONTROL: no MAEM weights at all, the clean base run
        # through the identical prompt / marker / injection / sampling path so the tables have a
        # "what does the untrained model reach" row (2026-09-16_base-control).
        # `nla` is the activation-verbalizer BASELINE (NLA training): served exactly like a `full`
        # model, injected at the same block-1 output with the same norm-matched add, but with its
        # own prompt, marker and output format -- so only `rollouts_nla` generates for it.
        assert spec.get("type") in ("lora", "full", "base", "nla"), (
            f"maem {key!r}: type must be 'lora', 'full', 'base' or 'nla', got {spec.get('type')!r}"
        )
        if spec.get("type") == "base":
            assert spec.get("hf") == cfg["bases"][base]["hf"], (
                f"maem {key!r}: type 'base' is the untrained-base control, so its `hf` must be "
                f"the base's own repo {cfg['bases'][base]['hf']!r}, got {spec.get('hf')!r} -- "
                f"anything else is a TRAINED checkpoint wearing the control's label"
            )
        assert spec.get("role") in (None, "control"), (
            f"maem {key!r}: `role`, when given, must be 'control' (reconstruction/stats.py shows "
            f"it between the primary and the secondaries), got {spec.get('role')!r}"
        )
        assert ("hf" in spec) != ("src" in spec), (
            f"maem {key!r}: give exactly one of hf (repo id) / src (volume path), got {sorted(spec)}"
        )
        if spec.get("type") == "nla":
            # The NLA verbalizer builds its OWN prompt from the checkpoint's sidecar (its marker
            # is not our MARKER and is not the last prompt token), so the `prompt in PROMPTS`
            # assert below does not apply to it and `nla:` is validated instead.
            _check_nla(key, spec, cfg["rollouts"])
        else:
            assert spec.get("prompt") in PROMPTS, (
                f"maem {key!r}: prompt {spec.get('prompt')!r} is not one of {sorted(PROMPTS)}"
            )
        inject = spec.get("inject", {})
        assert "layer" in inject and "coef" in inject, (
            f"maem {key!r}: inject needs both 'layer' and 'coef', got {inject}"
        )
        assert isinstance(spec.get("compute", True), bool), (
            f"maem {key!r}: `compute` must be a bool (default true), got {spec.get('compute')!r}"
        )
        if "mu" in spec:
            # `unknown` IS legal on a checkpoint (2026-09-21): a THIRD state between "no key"
            # (nobody has considered it) and a path/null (established). It says the training
            # convention is on the agenda and not on the record, so every run must be told with
            # --mu; `mu_for` refuses to pick one.
            _check_mu_value(spec["mu"], f"maems[{key!r}].mu", allow_unknown=True)

    # `ood_arms:` (design infra/2026-09-18_ood-eval-design.md §2). Optional: a config without it
    # is the pre-2026-09-18 pipeline and every check below is skipped.
    for arm, spec in (cfg.get("ood_arms") or {}).items():
        assert "/" not in arm and arm, f"ood arm {arm!r} is a DIRECTORY name under corpora/"
        for field in ("family", "reader", "dataset", "files", "text", "sizes", "script", "unspaced"):
            assert field in spec, f"ood arm {arm!r} is missing {field!r}"
        assert spec["family"] in ("lang", "ctrl", "code", "math", "diag"), (
            f"ood arm {arm!r}: unknown family {spec['family']!r}"
        )
        assert spec["reader"] in ("parquet", "jsonl", "jsonl_zst", "formulas"), (
            f"ood arm {arm!r}: unknown reader {spec['reader']!r}"
        )
        assert isinstance(spec["files"], list) and spec["files"], f"ood arm {arm!r}: files must be a list"
        sizes = spec["sizes"]
        assert isinstance(sizes, list) and sizes == sorted(sizes) and sizes[0] > 0, (
            f"ood arm {arm!r}: sizes must be an ascending list of MILLIONS of tokens, got {sizes}"
        )
        assert isinstance(spec["unspaced"], bool), f"ood arm {arm!r}: unspaced must be a bool"
        parts = SCRIPT_ALIASES.get(spec["script"], (spec["script"],))
        for name in parts:
            assert name in SCRIPT_RANGES, (
                f"ood arm {arm!r}: script {spec['script']!r} has no range table "
                f"(common.SCRIPT_RANGES knows {sorted(SCRIPT_RANGES)})"
            )

    for set_name, spec in cfg["heldout"].items():
        if spec.get("kind") == "ood":
            assert cfg.get("ood_arms"), f"heldout {set_name!r} is an ood set but config has no ood_arms"
            assert int(spec.get("n_per_arm", 0)) > 0, f"heldout {set_name!r}: n_per_arm must be > 0"
            want = spec.get("arms", "all")
            assert want == "all" or (isinstance(want, list) and want), (
                f"heldout {set_name!r}: `arms` must be `all` or a non-empty list, got {want!r}"
            )
            if isinstance(want, list):
                for a in want:
                    assert a in cfg["ood_arms"], f"heldout {set_name!r}: unknown arm {a!r}"
            base_set = spec.get("variant_of")
            assert base_set is None or base_set in cfg["heldout"], (
                f"heldout {set_name!r}: variant_of {base_set!r} is not a held-out set"
            )
        assert isinstance(spec.get("families"), dict), (
            f"heldout {set_name!r}: families must be a name -> spec MAPPING (checklist item 38: "
            f"families are keyed by name, never by list position), got {type(spec.get('families'))}"
        )
        for fam, fspec in spec["families"].items():
            assert "n" in fspec, f"heldout {set_name!r} family {fam!r} is missing 'n'"
            assert fam in cfg["family_kinds"], (
                f"heldout {set_name!r} names family {fam!r}, which has no `family_kinds:` entry; "
                f"nothing can then say whether a mean may be subtracted from its rows "
                f"(have {sorted(cfg['family_kinds'])})"
            )
            for b in fspec.get("bases", []):
                assert b in cfg["bases"], (
                    f"heldout {set_name!r} family {fam!r} restricted to unknown base {b!r}"
                )
        _check_heldout_storage(cfg, set_name, spec)
    return cfg


# Exactly the keys a `type: nla` entry's `nla:` block carries. A closed set: a typo
# (`max_nev: 96`) in a block whose every field steers an H200 run would otherwise be read as
# "the field is absent", and every field here is required, so "absent" has no safe meaning.
#
# `sampling` LEFT the block on 2026-09-21: temperature / top_p /
# top_k / min_new are now the shared `rollouts:` values every MAEM arm generates under, so the
# NLA arm cannot carry its own. This tuple and config.yaml were on opposite sides of that change
# when the branches met -- main's config had already dropped the key while main's `NLA_KEYS` still
# demanded it, so `load_config` raised on EVERY command, NLA or not. Reconciled here in main's
# direction.
NLA_KEYS = (
    "marker",
    "marker_id",
    "left_id",
    "right_id",
    "template",
    "max_new",
    "score_max_tokens",
    "card_max_new",
    "n",
    "amp",
    "amp_r",
)
# Sampling keys the `nla:` block may OVERRIDE per-MAEM, falling back to the shared `rollouts:`
# block when absent. Only `min_new`: the verbalizer's stop comes well before the shared 16, and
# editing the shared block instead would re-point every rollout product in the pipeline.
NLA_OPTIONAL_KEYS = ("min_new",)
# What the SHARED `rollouts:` block must carry for the NLA arm to generate under it.
NLA_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_new")


def _check_nla(key: str, spec: dict, rollouts: dict) -> None:
    """Validate one `type: nla` maems entry. Called from load_config, never at use site.

    Everything here is a fact about the CHECKPOINT (the NLA checkpoint's nla_meta.yaml
    and generation_config.json) rather than a choice of ours, except `max_new`, `n`, `amp` and
    `amp_r`. `rollouts_nla.check_sidecar` asserts the checkpoint's shipped files still agree with
    the values below before it generates anything; this function only checks the config's shape,
    which is what the CPU `check` gate can do without the weights.
    """
    assert "hf" in spec, f"maem {key!r}: a `type: nla` entry is fetched from HF, so it needs `hf`"
    rev = spec.get("revision")
    assert isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (
        f"maem {key!r}: `revision` must be the 40-hex HF commit sha the snapshot directory is "
        f"named after (rollouts_nla asserts the resolved path against it), got {rev!r}"
    )
    assert "prompt" not in spec, (
        f"maem {key!r}: a `type: nla` entry must NOT name a `prompt` -- the verbalizer builds its "
        f"own from `nla.template` and the marker in `nla.marker`, and common.PROMPTS' marker is "
        f"neither that character nor at that position"
    )
    nla = spec.get("nla")
    assert isinstance(nla, dict), f"maem {key!r}: a `type: nla` entry needs an `nla:` block, got {nla!r}"
    allowed = set(NLA_KEYS) | set(NLA_OPTIONAL_KEYS)
    missing, extra = sorted(set(NLA_KEYS) - set(nla)), sorted(set(nla) - allowed)
    assert not missing and not extra, (
        f"maem {key!r}: `nla:` must carry exactly {list(NLA_KEYS)} (optionally "
        f"{list(NLA_OPTIONAL_KEYS)}) -- missing {missing}, unexpected {extra}"
    )
    for field in NLA_OPTIONAL_KEYS:
        if field in nla:
            assert isinstance(nla[field], int) and not isinstance(nla[field], bool) and nla[field] >= 0, (
                f"maem {key!r}: nla.{field} overrides rollouts.{field} and must be a "
                f"non-negative int, got {nla[field]!r}"
            )
    for field in ("marker", "template", "amp"):
        assert isinstance(nla[field], str) and nla[field], (
            f"maem {key!r}: nla.{field} must be a non-empty string, got {nla[field]!r}"
        )
    for field in ("marker_id", "left_id", "right_id", "max_new", "score_max_tokens", "card_max_new", "n"):
        assert isinstance(nla[field], int) and not isinstance(nla[field], bool) and nla[field] > 0, (
            f"maem {key!r}: nla.{field} must be a positive int, got {nla[field]!r}"
        )
    assert "{injection_char}" in nla["template"], (
        f"maem {key!r}: nla.template must carry the sidecar's `{{injection_char}}` placeholder -- "
        f"that is where the marker token, and so the injected direction, goes"
    )
    # The arm generates under the SHARED block, so that is what has to be well-formed for it.
    # The guard did not go away when `nla.sampling` did -- it moved to the block that replaced it.
    missing_s = sorted(set(NLA_SAMPLING_KEYS) - set(rollouts))
    assert not missing_s, (
        f"maem {key!r} is a `type: nla` arm and generates under the shared `rollouts:` block, "
        f"which is missing {missing_s} -- it must carry {list(NLA_SAMPLING_KEYS)}"
    )
    assert float(rollouts["temperature"]) > 0 and 0 < float(rollouts["top_p"]) <= 1, (
        f"maem {key!r}: rollouts.temperature must be > 0 and top_p in (0, 1], got {rollouts}"
    )
    assert int(rollouts["top_k"]) >= 0 and int(rollouts["min_new"]) >= 0, (
        f"maem {key!r}: rollouts.top_k and min_new must be >= 0, got {rollouts}"
    )
    assert nla["amp"] in AMP_MODES, f"maem {key!r}: nla.amp {nla['amp']!r} is not one of {list(AMP_MODES)}"
    amp_r = nla["amp_r"]
    numeric_r = isinstance(amp_r, int | float) and not isinstance(amp_r, bool) and amp_r > 0
    assert amp_r == "median" or numeric_r, (
        f"maem {key!r}: nla.amp_r must be 'median' (layer read_layer's q[0.5] of "
        f"stats/resid_norm_quantiles.json) or a positive number, got {amp_r!r}"
    )
    # The scorer's window is the binding constraint, exactly as SCORE_MAX_LENGTH is for
    # `rollouts.max_new` -- but this arm brings its OWN window. `nla.score_max_tokens` is the
    # re-encode truncation `score` uses for it (carried there on the rollouts summary), so the
    # bound is against that rather than against the protocol's 95: a generation longer than the
    # window it will be scored in has a tail nothing ever reads.
    assert nla["max_new"] <= nla["score_max_tokens"] - 1, (
        f"maem {key!r}: nla.max_new {nla['max_new']} exceeds nla.score_max_tokens "
        f"{nla['score_max_tokens']} - 1 -- the scoring window must leave room for the whole "
        f"generation plus the sink at column 0, or the tail of a full-length rollout is never "
        f"scored (the same rule SCORE_MAX_LENGTH={SCORE_MAX_LENGTH} enforces on rollouts.max_new "
        f"= {rollouts['max_new']} for every other arm)"
    )
    # Never NARROWER than the protocol: this key exists to widen the window for a model whose
    # native output is long, not to cut an arm's text short and call it a protocol.
    assert nla["score_max_tokens"] >= SCORE_MAX_LENGTH, (
        f"maem {key!r}: nla.score_max_tokens {nla['score_max_tokens']} is below the pipeline's "
        f"SCORE_MAX_LENGTH={SCORE_MAX_LENGTH}; this key may only WIDEN the scoring window"
    )
    assert nla["card_max_new"] >= nla["max_new"], (
        f"maem {key!r}: nla.card_max_new {nla['card_max_new']} is the budget the model card's "
        f"reference script uses and must be >= the nla.max_new {nla['max_new']} we generate at"
    )


def _check_mu_value(val, where: str, allow_unknown: bool) -> None:
    """A mu value is null, a path to a [d] .f32/.npy file, or (stored rows only) `unknown`.

    Checked at LOAD, not at use: a typo in a path that steers an H200 run should cost a CPU second.
    The file's existence is NOT checked here -- config.yaml is read on a laptop with no volume
    mounted -- `load_mu` asserts that, loudly, at the point it needs the bytes.
    """
    if val is None:
        return
    if val == MU_UNKNOWN:
        assert allow_unknown, (
            f"{where}: {MU_UNKNOWN!r} says \"the mean is not on the record\", which is a statement "
            f"about stored rows or about a checkpoint's training convention -- not something a "
            f"run can be performed under. Give a path, or null."
        )
        return
    assert isinstance(val, str) and val, f"{where}: a mu is null, a path or {MU_UNKNOWN!r}, got {val!r}"
    assert val.endswith(MU_SUFFIXES), (
        f"{where}: {val!r} is not a {' or '.join(MU_SUFFIXES)} file. A mu is a [d] array on the "
        f"volume; a path starting with / is absolute, anything else is relative to --root, and "
        f"`{{base}}` expands to the base key."
    )


def resolve_mu_path(mu: str, base: str, root: str = VOL) -> str:
    """A config/CLI mu value -> the path to read. `{base}` expands; a relative path takes --root.

    Relative-to-root is what makes a smoke self-contained: a run under /vol/runs/<date>_smoke gets
    that root's own `base/<base>/stats/mu.f32` rather than the production one, without editing
    config. An absolute path (the archived whiten_mu) is the same file for every run.
    """
    assert mu and mu != MU_UNKNOWN, f"{mu!r} is not a loadable mu path"
    path = mu.format(base=base)
    return path if path.startswith("/") else f"{root.rstrip('/')}/{path}"


_MU_CACHE: dict[tuple, object] = {}


def load_mu(cfg: dict, base: str, mu, root: str = VOL):
    """The centring mean [d] as a float32 numpy array, or None when `mu` is null.

    Loud rather than optional: a product that needs a mean and cannot find the file has to stop,
    because silently falling back to a locally computed mean is the drift this module exists to
    prevent.
    """
    import numpy as np

    if mu is None:
        return None
    assert mu != MU_UNKNOWN, (
        f"mu {MU_UNKNOWN!r} cannot be loaded: it is the label for \"centred on a mean nobody here "
        f"holds\", which is a statement about stored directions and not a file"
    )
    path = resolve_mu_path(mu, base, root)
    key = (base, path)
    if key in _MU_CACHE:
        return _MU_CACHE[key]
    d = cfg["bases"][base]["d"]
    assert os.path.exists(path), (
        f"no {path}: this run centres on that file (config `mu:` / `--mu` = {mu!r}). For a "
        f"stats mean it means the `stats` product has not run on this --root; nothing recomputes "
        f"its own."
    )
    arr = np.load(path) if path.endswith(".npy") else read_array(path, "float32", (d,))
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    assert arr.shape == (d,) and np.isfinite(arr).all(), (
        f"{path}: expected {d} finite float32 values for base {base}, got shape {arr.shape}"
    )
    _MU_CACHE[key] = arr
    return arr


def mu_label(mu, base: str = "", root: str = VOL) -> str:
    """How a mu is spelled in a README line: the resolved path, or `none` / `unknown`."""
    if mu is None:
        return "none"
    if mu == MU_UNKNOWN:
        return MU_UNKNOWN
    return resolve_mu_path(mu, base, root) if base else str(mu)


def input_mu(cfg: dict, maem_key: str):
    """The mean `maem_key` was TRAINED to receive: a path, or None. Refuses rather than defaulting.

    Half the products in this repo have no MAEM in scope at all (scan, gcg, patchscopes,
    repo_examples) and the other half would silently re-point every number in SMOKES.md if this
    guessed -- so an entry with no `mu:` key is a hard stop with the ask named. An explicit
    `mu: null` is a statement; an absent key is a gap.
    """
    assert maem_key in cfg["maems"], f"unknown maem {maem_key!r}, want one of {sorted(cfg['maems'])}"
    spec = cfg["maems"][maem_key]
    # Three states, and they are different: no key at all (nobody has considered it), `unknown`
    # (considered, not established -- every run must be told), a path or null (established).
    assert "mu" in spec, (
        f"maem {maem_key!r} has no `mu:` key in config.yaml, so what it was trained to receive is "
        f"not recorded anywhere. Establish it from the checkpoint's training chain and declare it "
        f"(`mu: null` for a raw unit activation, or the path of the mean); nothing here will guess. "
        f"To run against a convention you are CHOOSING rather than reading, pass --mu -- it is "
        f"recorded as a deviation."
    )
    return spec["mu"]


def score_mu(cfg: dict, base: str) -> str:
    """THE SCORING CONSTANT: the one mean BOTH arguments of every centred cosine are taken about.

    It is `bases.<base>.whiten_mu` -- a key that already existed as the base's archived centring
    mean -- and it is a property of the BASE, not of any MAEM, any run or any set. Read it here
    and nowhere else, so that `score`, `scan` and `gcg` centre on one vector and their numbers are
    comparable by construction.

    It is deliberately DECOUPLED from a MAEM's `mu:` key (`input_mu`), which says what that
    checkpoint was trained to RECEIVE at its marker token and remains the injection convention.
    Before 2026-09-23 the reported cosine's mean was whatever the run's injection convention was,
    which meant: the old primary (`mu: null`) got no centred cosine at all, the base control got
    none, the NLA arm got none, and any two arms trained on different conventions could not be
    differenced. A difference of two cosines taken about two different means is not a difference.
    """
    spec = cfg["bases"][base]
    mu = spec.get("whiten_mu")
    assert isinstance(mu, str) and mu, (
        f"base {base!r} has no `whiten_mu:` in config.yaml, so there is no scoring constant for it "
        f"and no centred cosine can be reported on it. Declare the base's mean file."
    )
    _check_mu_value(mu, f"bases[{base!r}].whiten_mu", allow_unknown=False)
    return mu


def _check_heldout_storage(cfg: dict, set_name: str, spec: dict) -> None:
    """Validate one held-out set's storage contract. See config.yaml's `heldout:` header.

    Required on every declared set, including an `imported: true` one: the contract is the only
    thing that says what `vecs.f16` holds, and a set whose contract is unstated is a set nothing
    may centre or refuse to centre with any honesty.
    """
    storage = spec.get("storage")
    assert storage in STORAGE_KINDS, (
        f"heldout {set_name!r}: `storage` must be one of {list(STORAGE_KINDS)}, got {storage!r} -- "
        f"a set with no declared storage contract cannot be served at any centring"
    )
    sk = spec.get("sae_key")
    assert sk is None or sk in cfg["saes"], (
        f"heldout {set_name!r}: sae_key {sk!r} is not an SAE in config.yaml ({sorted(cfg['saes'])})"
    )
    for dead in ("mu_stored", "family_mu"):
        assert dead not in spec, (
            f"heldout {set_name!r} still declares `{dead}:`. The stored-convention layer was "
            f"deleted on 2026-09-23 (M0a): a `storage: unit` set's rows are served EXACTLY as the "
            f"producer shipped them and simply have no centred cosine, because unit(act) and mu do "
            f"not give unit(act - mu) without ||act||. The mean of every reported centred number "
            f"is now the base's own scoring constant, `common.score_mu`."
        )


def family_centrable(cfg: dict, family: str) -> bool:
    """Can a mean be subtracted from this family's rows at all? `family_kinds:` decides.

    An encoder column, a Gaussian draw and a subspace basis have no mean of their own, so a
    `cos(h - mu, v)` against one is a ONE-SIDED number: the activation moved and the target did
    not. Since 2026-09-23 `score` REPORTS that number for such a row -- the residual centred
    against the stored direction -- and labels it `centred_sided: 1` in `per_target.jsonl`, rather
    than writing NaN as it did before. It is comparable across such rows and not against a
    `centrable` family's two-sided number; see `score._load_dirs`.
    """
    assert family in cfg["family_kinds"], (
        f"family {family!r} has no `family_kinds:` entry, so nothing can say whether a mean may be "
        f"subtracted from its rows (have {sorted(cfg['family_kinds'])})"
    )
    return bool(cfg["family_kinds"][family]["centrable"])


def set_storage(cfg: dict, set_dir: str, root: str = VOL) -> dict:
    """The storage contract of the set at `set_dir`: {storage, source}.

    WHAT `vecs.f16` HOLDS, and nothing about a mean: `raw` (act.f32 is there and every direction is
    derived from it at read time), `unit` (a stored direction, as the producer shipped it, which
    cannot be re-centred) or `dirs_only`. The `mu_stored` / `family_mu` fields the pre-2026-09-23
    contract carried are IGNORED where an old storage.json still has them -- see
    `_check_heldout_storage`.

    Resolution order, because three kinds of directory reach this function:
      1. `<set_dir>/storage.json`, which every set drawn after 2026-09-21 writes;
      2. config.yaml's `heldout.<basename>` block, for the sets drawn before that;
      3. refuse -- a `--dirs-from` directory that is neither has no stated contract, and guessing
         one is how a centred and an uncentred direction become the same file to a reader.
    """
    path = f"{set_dir.rstrip('/')}/{STORAGE_FILE}"
    if os.path.exists(path):
        with open(path) as fh:
            rec = json.load(fh)
        assert "storage" in rec, f"{path} is missing 'storage'"
        assert rec["storage"] in STORAGE_KINDS, f"{path}: storage {rec['storage']!r} is not a kind"
        return {"storage": rec["storage"], "source": path}
    name = os.path.basename(set_dir.rstrip("/"))
    assert name in cfg["heldout"], (
        f"{set_dir} carries no {STORAGE_FILE} and {name!r} is not a set declared in config.yaml, so "
        f"nothing states what its vecs.f16 holds. Declare it under `heldout:` (which needs "
        f"only `storage:`) or re-draw the set, which writes the contract itself."
    )
    return {"storage": cfg["heldout"][name]["storage"], "source": f"config.yaml heldout.{name}"}


def storage_record(cfg: dict, set_name: str, families, sae_key: str = "") -> dict:
    """The `storage.json` a freshly drawn `storage: raw` set writes. See `set_storage`."""
    return {
        "storage": "raw",
        # Which dictionary this set's SAE feature ids index. Every row of a set drawn now also
        # carries its own `sae_key`, so this is belt and braces -- but it is what
        # `common.declared_sae_key` reads, and a set that loses its config entry keeps it.
        "sae_key": sae_key,
        "families": {f: cfg["family_kinds"][f]["kind"] for f in families},
        "note": (
            "RAW STORAGE: act.f32 [N, d] holds the row's own vector before any mean was subtracted "
            "and vecs.f16 is unit(act) -- UNCENTRED. Every centred direction is derived at read "
            "time by common.dirs_for(..., centering=<mu name>). For a family that is not "
            "`centrable` (family_kinds), the act.f32 row IS the stored unit direction, so "
            "unit(act) == vecs.f16 there and no centring ever applies to it."
        ),
    }


def mu_for(cfg: dict, base: str, set_dir: str, args: dict, maem_key: str = "",
           root: str = VOL, notes=None):
    """(the mean this run centres on, where it came from). THE convention is never inferred silently.

    Returns a mu VALUE -- None, or a path as config spells it -- plus a provenance string.

    Order, and nothing else:

      1. `--mu <file>` -- explicit, and when it disagrees with the checkpoint's own `mu:` it is
         recorded as a DEVIATION in the product README, not accepted quietly. `--mu none` is the
         explicit way to say "subtract nothing";
      2. the MAEM's `mu:`, for the products that have a `--maem` in scope;
      3. refuse. A set read by a product with no MAEM (scan, gcg, patchscopes, repo_examples) has
         no convention anywhere in scope, and defaulting one would silently re-point the corpus
         search baseline and the GCG ceiling at a different target vector than every stored
         number. That is the one failure this whole layer exists to stop.

    THE SET'S OWN STORED CONVENTION IS NO LONGER A SOURCE (2026-09-23, M0a). `mu_stored` /
    `family_mu` are deleted: a `storage: unit` set's rows are served exactly as the producer
    shipped them and have no centred reading at all, so there was nothing for the third branch to
    resolve. Note that this is the INJECTION convention; the mean every centred cosine is REPORTED
    about is `score_mu`, the base's constant, and is not resolved here.
    """
    say = notes if notes is not None else []
    want = (args.get("mu") or "").strip()
    if want:
        got = None if want.lower() in ("none", "null") else want
        _check_mu_value(got, "--mu", allow_unknown=False)
        src = "--mu (explicit)"
        if maem_key:
            own = input_mu(cfg, maem_key)
            if own == MU_UNKNOWN:
                line = (
                    f"{maem_key} declares `mu: {MU_UNKNOWN}` (training convention not on the "
                    f"record); this run was TOLD {mu_label(got, base, root)} by --mu. That is a "
                    f"choice being made here, not a fact being read."
                )
                print(f"[mu] {line}", flush=True)
                say.append(line)
                say.append(f"mu={mu_label(got, base, root)} from --mu (a CHOICE, not the record)")
                return got, "--mu (checkpoint's own mu is `unknown`)"
            if own != got:
                line = (
                    f"DEVIATION: --mu {mu_label(got, base, root)} overrides {maem_key}'s own "
                    f"trained input convention {mu_label(own, base, root)} (config.yaml "
                    f"maems.{maem_key}.mu). Every number in this directory is read under the "
                    f"former, not under what the checkpoint was trained on."
                )
                print(f"[mu] {line}", flush=True)
                say.append(line)
                src = "--mu (OVERRIDE of the checkpoint's own mu)"
        say.append(f"mu={mu_label(got, base, root)} from {src}")
        return got, src
    if maem_key:
        own = input_mu(cfg, maem_key)
        assert own != MU_UNKNOWN, (
            f"maem {maem_key!r} declares `mu: {MU_UNKNOWN}`: its training convention is on the "
            f"agenda and NOT on the record, so nothing here will pick one for it. Pass --mu "
            f"explicitly (a path, or `none`) and the choice is recorded as a deviation in the "
            f"product README. That is what the two-arm reconciliation in SMOKES.md settles."
        )
        say.append(f"mu={mu_label(own, base, root)} from config.yaml maems.{maem_key}.mu")
        return own, f"maems.{maem_key}.mu"
    contract = set_storage(cfg, set_dir, root)
    raise AssertionError(
        f"{set_dir} is `storage: {contract['storage']}` ({contract['source']}) and this product "
        f"has no --maem to take an injection convention from. Pass --mu <file> (or --mu none); "
        f"`scan` also takes --centre, which is the base's own scoring constant on both sides. "
        f"Defaulting it would silently move this product's target vector away from every stored "
        f"number."
    )


def note_convention(od, notes) -> None:
    """Put the centring lines `mu_for` / `dirs_for` collected into a product's README.

    Every product that reads a direction calls this. A README that does not say which mean its
    numbers were read under is a README nobody can compare to another one.
    """
    for line in notes or []:
        od.note(f"CENTRING: {line}")


def dirs_for(cfg: dict, base: str, set_dir: str, mu, root: str = VOL, notes=None):
    """The direction every row of this set carries under `mu` -- the WHOLE [N, d] array.

        storage: raw        -> unit(act - mu) for a centrable family, unit(act) for every other row
                               (there is no mean to subtract from an encoder column)
        storage: unit       -> the stored row AS THE PRODUCER SHIPPED IT, unchanged, whatever `mu`
                               says: unit(act) and mu do not give unit(act - mu) without ||act||,
                               so such a set has no TWO-SIDED centred reading at any mean; `score`
                               reports the ONE-SIDED cos(h - mu, unit(d)) against the stored row
                               and marks it `centred_sided: 1` (M0a 2026-09-23, amended 09-23)
        storage: dirs_only  -> the stored row (no family in such a set is centrable)

    `mu` is None or a path, and is always EXPLICIT: the four products with no MAEM in scope (scan,
    gcg, patchscopes, repo_examples) would otherwise silently re-point the corpus search baseline
    and the GCG ceiling away from every number measured between 09-16 and 09-21.

    Returns the whole array and leaves ROW SELECTION at each call site, because `rows` means three
    different things across the readers -- global in score/rollouts_*, family-local in gcg
    (`--family sae --rows 0-7` is global rows 1024-1031), absent in scan/centred/repo_examples.

    `notes`, when a list is passed, receives one human-readable line per thing a reader of the
    product's README has to know (the mean used, and every labelled family).
    """
    import numpy as np

    d = cfg["bases"][base]["d"]
    set_dir = set_dir.rstrip("/")
    rows = read_jsonl(f"{set_dir}/ids.jsonl")
    n = len(rows)
    assert n, f"{set_dir}/ids.jsonl is empty"
    for i, r in enumerate(rows):
        assert r["row"] == i, f"{set_dir}/ids.jsonl line {i} has row={r['row']}: rows must be 0..N-1"
    contract = set_storage(cfg, set_dir, root)
    storage = contract["storage"]
    _check_mu_value(mu, "dirs_for(mu=)", allow_unknown=False)
    fams = [r["family"] for r in rows]
    say = notes if notes is not None else []
    label = mu_label(mu, base, root)

    if storage == "raw":
        apath = f"{set_dir}/act.f32"
        assert os.path.exists(apath), (
            f"{set_dir} declares `storage: raw` ({contract['source']}) but has no act.f32; a raw "
            f"set derives every direction from it. Re-draw the set."
        )
        act = read_array(apath, "float32", (n, d)).astype(np.float32)
        arr = load_mu(cfg, base, mu, root)
        out = act.copy()
        if arr is not None:
            cen = np.array([family_centrable(cfg, f) for f in fams], dtype=bool)
            out[cen] -= arr[None, :]
            # THE ROWS A MEAN IS APPLIED TO ARE EXACTLY THE ROWS THAT HAVE A RAW ACTIVATION.
            # `2026-09-21_v3_ctrl` is the set that makes this a live question: it mixes `random`
            # draws and 131k encoder columns in one `storage: raw` directory, so `act.f32` holds
            # rows that are NOT activations. Subtracting a residual-stream mean from an encoder
            # column or a Gaussian draw gives cos(h - mu, v): the activation moved and the target
            # stood still, a one-sided number that looks like a centred one. `family_kinds:` is
            # the only thing that separates them, so the separation is asserted here and not
            # merely performed -- and it is asserted as a POST-CONDITION on the array, not as a
            # restatement of the line above, so a future `out[...] -=` elsewhere in this branch
            # trips it too.
            # Read straight off `cfg["family_kinds"]` rather than through `family_centrable`,
            # which is what computed `cen` above: a post-condition that called the same function
            # as the code it guards would agree with it by construction and guard nothing.
            moved = np.abs(out - act).max(axis=1) > 0
            bad = [
                (i, fams[i]) for i in range(n)
                if bool(moved[i]) and not cfg["family_kinds"][fams[i]]["centrable"]
            ]
            assert not bad, (
                f"{set_dir}: a mean was subtracted from rows {bad[:8]} whose family is not "
                f"`centrable` (config.yaml family_kinds). An encoder column, a Gaussian draw and "
                f"a subspace basis have no mean; cos(h - mu, v) against one is one-sided and is "
                f"not a centred number."
            )
            say.append(
                f"directions derived from {apath} at mu={label}: unit(act - mu) on "
                f"{int(cen.sum())} centrable rows "
                f"({sorted({f for f, c in zip(fams, cen, strict=True) if c})}), unit(act) on the "
                f"other {int((~cen).sum())} (family_kinds says they have no mean to subtract)"
            )
        else:
            say.append(f"directions derived from {apath} at mu=none: unit(act), all {n} rows")
        return _unit_rows(out)

    v = read_array(f"{set_dir}/vecs.f16", "float16", (n, d)).astype(np.float32)
    if storage == "dirs_only":
        say.append(
            f"{set_dir} is `storage: dirs_only` ({contract['source']}): the stored vecs.f16 rows "
            f"are returned unchanged and mu={label} does not apply to any of them"
        )
        return _unit_rows(v)

    # storage: unit -- the stored row is a direction under the PRODUCER's convention and cannot be
    # moved to another one: unit(act) and mu do not give unit(act - mu) without ||act||. Until
    # 2026-09-23 this branch carried a `mu_stored` / `family_mu` contract that named that
    # convention per family, asserted it against the run's `mu` and labelled an `unknown` one. The
    # whole layer is gone (M0a): the rows come back as shipped, and the centred cosine such a set
    # has is NONE -- `score` writes NaN for every one of its rows rather than a number whose mean
    # nobody can state. Reading it is still exact for the UNCENTRED cosine, which is what every
    # number measured between 2026-09-16 and 2026-09-21 was.
    say.append(
        f"{set_dir} is `storage: unit` ({contract['source']}): the stored vecs.f16 rows are "
        f"returned UNCHANGED, under whatever convention the producer used, and mu={label} does "
        f"not apply to any of them. There is no centred number for this set."
    )
    return _unit_rows(v)


def _unit_rows(v):
    """Row-wise L2 normalisation in fp32 with the numpy eps convention (see centred.py:63)."""
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def is_nla(cfg: dict, maem_key: str) -> bool:
    """True for the activation-verbalizer baseline, whose ONLY generator is `rollouts_nla`."""
    return cfg["maems"][maem_key].get("type") == "nla"


def default_heldout(cfg: dict) -> str:
    """The held-out set a product takes when `--set` is omitted: the latest NON-imported one.

    `sorted(cfg["heldout"])[-1]` was that rule until a set drawn elsewhere had to be REGISTERED
    here so the entrypoint would accept its name (`2026-09-20_dict2m_2k`, written by
    features/draw_dict2m.py through features/spawn.py, which bypasses modal_app.main's assert).
    Registering it under the old rule would have silently moved every default-set product off
    `2026-09-16_v1`, which is the set the paper's tables are built on. `imported: true` marks a
    set as "nameable, never the default".
    """
    own = sorted(k for k, s in cfg["heldout"].items() if not s.get("imported"))
    assert own, (
        f"config.yaml declares no non-imported held-out set: {sorted(cfg['heldout'])} are all "
        f"`imported: true`, so there is no default for a product called without --set"
    )
    return own[-1]


def split_key(key: str, what: str) -> tuple[str, str]:
    """'qwen36-27b/l42-1b' -> ('qwen36-27b', 'l42-1b'). Config keys ARE volume directory names."""
    parts = key.split("/")
    assert len(parts) == 2 and all(parts), f"{what} key {key!r} must be exactly '<base>/<name>'"
    return parts[0], parts[1]


def families_for(cfg: dict, set_name: str, base: str) -> dict[str, dict]:
    """The families of a held-out set that apply to `base`, keyed by name, empty slots included."""
    fams = cfg["heldout"][set_name]["families"]
    return {f: s for f, s in fams.items() if base in s.get("bases", [base])}


def maems_for(cfg: dict, base: str = "", computable_only: bool = True) -> list[str]:
    """The MAEM keys of `base` (or of every base when base is ""), in config order.

    `compute: false` entries are DECLARED but nothing is generated for them: they exist so the
    paper's model table, the prompt inventory and `check` know about them. Every "all MAEMs"
    iteration that would spend GPU on a MAEM must therefore filter them out, which is what the
    default does; `check` passes computable_only=False and resolves them too (leniently -- an
    entry nobody computes need not be in the HF cache yet).
    """
    keys = [k for k in cfg["maems"] if not base or split_key(k, "maem")[0] == base]
    if computable_only:
        keys = [k for k in keys if cfg["maems"][k].get("compute", True)]
    return keys


def sae_key_for_rows(cfg: dict, base: str, rows, want: str = "") -> str:
    """`sae_key_for`, but "" when NO row of this set belongs to an SAE family.

    A base with two dictionaries makes `sae_key_for` refuse without `--sae`, which is right when
    the product is about to look feature ids up in one of them and wrong when the set has no
    feature ids at all. The OOD sets have no `sae` family (`config.yaml`'s `heldout.*.families` is
    empty for them and every row's family is its arm's), so demanding `--sae` there makes the
    operator name a dictionary the run never reads -- and that name then lands in the product
    README as though it meant something.

    Products that read a SET and may meet one without SAE rows call this; products that are ABOUT
    a dictionary (`draw_dict2m`, `sae_self`, `build`, `repo_examples`) still call `sae_key_for`
    directly, because for them an absent dictionary is a bad command line, not a shape of set.
    """
    # `rows` is a LIST of row dicts everywhere it is passed from (`read_jsonl` of ids.jsonl), but
    # a couple of readers keep the same rows in a {row index: row} mapping. Accept either rather
    # than make the caller remember which it is holding -- getting that wrong fails only inside
    # the container, after the base model has been loaded.
    if isinstance(rows, dict):
        rows = rows.values()
    return "" if not any(r.get("family") in SAE_FAMILIES for r in rows) else sae_key_for(
        cfg, base, want
    )


def sae_key_for(cfg: dict, base: str, want: str = "") -> str:
    """WHICH SAE of `base`: `want` when given, else the single one -- asserting when there are two.

    `qwen36-27b` has carried two SAEs since `dict2m` landed (`l42-1b` at 131k and `dict2m` at 2^21),
    and every "the base's SAE" site in this repo was written as `assert len(keys) == 1`. That
    assert is right when nothing says which, and wrong as a way of choosing, so the choice is
    made here once and the message names the options rather than the count.
    """
    keys = [k for k in cfg["saes"] if split_key(k, "sae")[0] == base]
    assert keys, f"base {base!r} has no SAE in config.yaml"
    want = (want or "").strip()
    if want:
        assert want in keys, f"--sae {want!r} is not one of base {base}'s SAEs {sorted(keys)}"
        return want
    assert len(keys) == 1, (
        f"base {base} has {len(keys)} SAEs in config ({sorted(keys)}), so nothing can pick one "
        f"for you: pass --sae <key>"
    )
    return keys[0]


# The family labels whose target IS an SAE feature (row["id"] is a feature index). `sae` is what
# every draw writes since 2026-09-21; `dict2m_enc` is the label the 2,000-row 2026-09-20 set carries
# and is accepted for it rather than rewritten in place.
SAE_FAMILIES = ("sae", "dict2m_enc")


def check_set_on_disk(cfg, base, set_name, fams, root):
    """A configured held-out set, opened rather than named. Absent is fine; WRONG is not.

    `check` used to print the `heldout:` entry's family list and stop there, so a set whose rows
    on the volume disagreed with its declaration -- the failure that made `2026-09-20_dict2m_2k`'s
    `families:` line wrong for a day -- was invisible to the cheap gate and surfaced in a GPU
    product instead. Each set is either NOT on this root (skipped, because a smoke root carries
    two sets and the config declares twelve) or checked against what it declares.
    """
    import json
    import os

    d = heldout_dir(base, set_name, root)
    out = {"dir": d, "status": "absent", "detail": "not on this root"}
    if not os.path.isdir(d):
        return out
    try:
        contract = set_storage(cfg, d, root)            # refuses a set that states none
        rows = [json.loads(ln) for ln in open(f"{d}/ids.jsonl", encoding="utf-8")]
        assert rows, f"{d}/ids.jsonl is empty"
        assert [r["row"] for r in rows] == list(range(len(rows))), (
            f"{d}/ids.jsonl rows are not 0..{len(rows) - 1}")
        # Every family present must be declared, or `family_centrable` / `dirs_for` refuse later.
        present: dict[str, int] = {}
        for r in rows:
            present[r["family"]] = present.get(r["family"], 0) + 1
            family_centrable(cfg, r["family"])
        want = {f: int(s["n"]) for f, s in fams.items() if s.get("status") != "empty"}
        assert present == want, (
            f"{d}: ids.jsonl carries {present} but config.yaml `heldout.{set_name}.families` "
            f"declares {want}. One of the two is wrong, and every family-keyed product reads "
            f"the config one.")
        # The storage contract, against the files that have to exist under it.
        has_act = os.path.exists(f"{d}/act.f32")
        assert has_act == (contract["storage"] == "raw"), (
            f"{d} is `storage: {contract['storage']}` ({contract['source']}) and act.f32 is "
            f"{'present' if has_act else 'absent'}: a raw set derives every direction from it, "
            f"and nothing else may carry one.")
        idx_path = f"{d}/index.json"
        if os.path.exists(idx_path):
            idx = json.loads(open(idx_path, encoding="utf-8").read())
            for name in ("vecs.f16",) + (("act.f32",) if has_act else ()):
                shape = (idx.get(name) or {}).get("shape")
                assert shape == [len(rows), int(cfg["bases"][base]["d"])], (
                    f"{d}/{name} is {shape}, expected {[len(rows), cfg['bases'][base]['d']]}")
        # An SAE row's dictionary must be nameable: per row, or declared for the set (H1).
        sae_rows = [r for r in rows if r["family"] in SAE_FAMILIES]
        if sae_rows and not all(r.get("sae_key") for r in sae_rows):
            declared = contract.get("sae_key") or cfg["heldout"][set_name].get("sae_key")
            assert declared, (
                f"{d} has {sum(1 for r in sae_rows if not r.get('sae_key'))} SAE rows with no "
                f"`sae_key` and neither storage.json nor the `heldout:` entry declares one; "
                f"`common.sae_rows_of` refuses them, and every 131k id is also a valid 2M id")
        out.update(status="ok", detail=(f"{len(rows)} rows {present}, storage "
                                        f"{contract['storage']} ({contract['source']})"))
    except (AssertionError, KeyError, OSError, ValueError) as e:
        out.update(status="FAILED", detail=f"{type(e).__name__}: {e}")
        raise
    return out


def sae_rows_of(rows, sae_key: str, families=SAE_FAMILIES, side: str = "", declared=None,
                where: str = ""):
    """The rows of `rows` whose target is a feature of dictionary `sae_key`.

    THE FAMILY LABEL IS NOT ENOUGH. A set may carry two dictionaries under one `family: sae` label,
    told apart by the per-row `sae_key` that features/draw_dict2m.py and targets.py write -- and a
    feature index is meaningless without it: every id below 131,072 is a valid index into a 2^21
    encoder, so selecting on the family alone looks up the 131k block's ids in the 2M dictionary
    and scores wrong features with nothing raising.

    AND A MISSING `sae_key` IS NOT A LICENCE. The first version of this function read
    `r.get("sae_key", sae_key) == sae_key`, which defaults each unkeyed row to match WHATEVER was
    typed -- so on the two sets that predate the field (2026-09-16_v1, 2026-09-20_dict2m_2k, which
    carry it on no row at all) the guard was vacuous, and `--sae qwen36-27b/dict2m --set
    2026-09-16_v1` selected all 512 of the 131k rows, every id a valid 2^21 index: exactly the
    silent failure the guard is for, on the paper's own set. MEASURED against the real ids
    2026-09-21.

    So an unkeyed row is selectable only against a DECLARED dictionary: `declared` is the one SAE
    the set says its feature ids index (`common.declared_sae_key`, from the set's storage.json or
    its `heldout:` entry), and it must equal `sae_key`. An undeclared set refuses, naming the row
    count, rather than answering a question nobody can check.

    `side` filters the encoder/decoder axis (`sae_side`, NOT draw_dict2m's `side`, which is the
    fit/report split of OUR analysis and a different axis entirely). A row with no `sae_side`
    predates decoder rows and counts as `enc`.
    """
    out, unkeyed = [], 0
    for r in rows:
        if r["family"] not in families:
            continue
        if side and r.get("sae_side", "enc") != side:
            continue
        own = r.get("sae_key")
        if own is None:
            unkeyed += 1
            continue
        if own == sae_key:
            out.append(r)
    if unkeyed:
        assert declared, (
            f"{where or 'this set'} has {unkeyed} SAE rows with no `sae_key` field and declares no "
            f"dictionary for them, so which SAE their feature ids index is not recorded anywhere. "
            f"Nothing here will assume it is --sae {sae_key!r}: every id below 131,072 is a valid "
            f"index into a 2^21 encoder, so a wrong guess scores wrong features silently. Declare "
            f"`sae_key:` on the set's `heldout:` entry (or re-draw it -- targets and draw_dict2m "
            f"stamp it per row)."
        )
        assert declared == sae_key, (
            f"{where or 'this set'} declares its {unkeyed} unkeyed SAE rows are features of "
            f"{declared!r}, but this run asked for --sae {sae_key!r}. Refusing rather than "
            f"selecting them: their ids index {declared!r} and mean something else in {sae_key!r}."
        )
        out.extend(
            r for r in rows
            if r["family"] in families
            and r.get("sae_key") is None
            and not (side and r.get("sae_side", "enc") != side)
        )
        out.sort(key=lambda r: r["row"])
    return out


def declared_sae_key(cfg: dict, set_dir: str, root: str = VOL):
    """The ONE dictionary a set says its unkeyed SAE feature ids index, or None.

    `storage.json`'s `sae_key` first (what a set drawn after 2026-09-21 carries), then the set's
    `heldout:` entry. It exists for the sets whose rows predate the per-row `sae_key` field; a set
    whose rows carry their own needs none of this.
    """
    path = f"{set_dir.rstrip('/')}/{STORAGE_FILE}"
    if os.path.exists(path):
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("sae_key"):
            return rec["sae_key"]
    name = os.path.basename(set_dir.rstrip("/"))
    return (cfg["heldout"].get(name) or {}).get("sae_key")


def stats_mu(cfg: dict, base: str, root: str = VOL):
    """`stats/mu.f32` [d] as a float32 numpy array -- OUR 64/16-window read-layer mean.

    `load_mu(cfg, base, STATS_MU, root)` under its historical name, kept because a dozen call sites
    spell it this way. It is no longer "the ONE centring mean": since 2026-09-21 a run names the
    FILE it centres on (`mu:` in config.yaml, `--mu` on the command line) and this is one file
    among several. Still loud rather than optional -- a product that needs a mean and finds no
    `stats/` has to stop, because silently computing its own is the drift this file exists to
    prevent.
    """
    return load_mu(cfg, base, STATS_MU, root)


# ---------------------------------------------------------------------------------------------
# volume paths (infra/precompute-layout.md §1)
#
# Every base product takes a `root` so a smoke can mirror the whole relative layout under
# <root>/base/<base>/... (default /vol). maems/, gcg/ and the archive are not root-relative: they
# are never written by a smoke.
# ---------------------------------------------------------------------------------------------


def base_dir(base: str, root: str = VOL) -> str:
    return f"{root}/base/{base}"


def corpus_key_name(cfg: dict, key: str) -> str:
    """A `corpora:` KEY -> the directory name `corpus_dir` / `--corpus-name` takes ("" = corpus/).

    Products name a corpus by key, not by directory: the key carries the ladder, the window
    geometry and the provenance sentence that says whether a search win on it is evidence about
    UNSEEN text (heldout16m) or about text the model may have memorised (train10m). A
    directory name carries none of that, and the two corpora are not comparable.
    """
    assert key in cfg["corpora"], (
        f"unknown --corpus {key!r}; config.yaml declares {sorted(cfg['corpora'])}"
    )
    d = cfg["corpora"][key]["dir"]
    return "" if d == "corpus" else d


def corpus_geometry(cfg: dict, key: str) -> tuple[int, int]:
    """(block, stride) of a `corpora:` key. A different geometry is a different corpus."""
    spec = cfg["corpora"][key]
    return int(spec["block"]), int(spec["stride"])


def corpus_key_of_dir(cfg: dict, corpus_name: str) -> str:
    """The `corpora:` key whose `dir` is `corpus_name` ("" = the original corpus/), or ""."""
    want = corpus_name or "corpus"
    for key, spec in cfg["corpora"].items():
        if spec["dir"] == want:
            return key
    return ""


def assert_corpus_geometry(cfg: dict, corpus_name: str) -> tuple[int, int]:
    """Refuse a corpus whose declared window geometry this pipeline does not actually use.

    H7. `corpora:` declares `block`/`stride` per corpus and `load_config` type-checks them, but
    nothing threads them: all eleven `windows_of(` sites take SCAN_BLOCK/SCAN_STRIDE. Scanning
    the `train_parity_10m` (declared 32/8, and its own meta.json says 32/8) would therefore cut
    64/16 windows while `scan` wrote a README asserting 64/16 -- two artefacts on the volume
    contradicting each other, and a number silently not the one the config promises.

    Threading it is a real change (every consumer reconstructs window ids to join on and would
    have to read the producer's geometry rather than the constant). Until that is done the honest
    behaviour is to STOP, which is what this does. Returns the pair when it is safe.
    """
    key = corpus_key_of_dir(cfg, corpus_name)
    if not key:
        return SCAN_BLOCK, SCAN_STRIDE
    block, stride = corpus_geometry(cfg, key)
    assert (block, stride) == (SCAN_BLOCK, SCAN_STRIDE), (
        f"corpus {key!r} declares window {block}/{stride} but this pipeline cuts windows at "
        f"{SCAN_BLOCK}/{SCAN_STRIDE} everywhere (common.SCAN_BLOCK; eleven windows_of call sites "
        f"take it). Scanning it anyway would write a README claiming {SCAN_BLOCK}/{SCAN_STRIDE} "
        f"over {block}/{stride} data and produce a number that is not the one config.yaml "
        f"promises. Thread the geometry through every windows_of site first, or scan a corpus "
        f"whose geometry matches."
    )
    return block, stride


def corpus_dir(base: str, root: str = VOL, name: str = "") -> str:
    """`corpus/` by default; `corpora/<name>/` when a name is given.

    A new size ladder or window geometry is a NEW corpus, never an edit of the existing
    one: docs.jsonl carries each document's size_tag and every stored scan window and
    top-k list indexes into that exact tokens.i32, so rebuilding in place silently
    invalidates all of them.
    """
    return f"{base_dir(base, root)}/corpora/{name}" if name else f"{base_dir(base, root)}/corpus"


def heldout_dir(base: str, set_name: str, root: str = VOL) -> str:
    return f"{base_dir(base, root)}/heldout/{set_name}"


def stats_dir(base: str, root: str = VOL) -> str:
    return f"{base_dir(base, root)}/stats"


def scan_dir(base: str, set_name: str, root: str = VOL, corpus_name: str = "") -> str:
    """`scan/<set>`, or `scan/<set>__<corpus>` when the scan is not over the default corpus.

    H5: a scan is (set x corpus), and until 2026-09-21 the corpus axis did not exist so keying by
    set alone was complete. This branch introduced `corpora:` and `--corpus`, and the eval plan
    scans ONE set over TWO corpora (§2.5's train search baseline, §3.4's 16M autointerp
    scan). Both resolved here to one path: the second refuses without --force and destroys the
    first with it -- ~$6 and ~$14 of GPU, and the paper's comparison anchor. Empty resolves to
    today's path, so nothing already on the volume moves.
    """
    suffix = f"__{corpus_name}" if corpus_name else ""
    return f"{base_dir(base, root)}/scan/{set_name}{suffix}"


def nll_dir(base: str, set_name: str, root: str = VOL) -> str:
    """`<root>/base/<base>/nll/<set>/` -- the base's own per-token NLL on each target's window."""
    return f"{base_dir(base, root)}/nll/{set_name}"


def sae_dir(sae_key: str, root: str = VOL) -> str:
    base, name = split_key(sae_key, "sae")
    return f"{base_dir(base, root)}/sae/{name}"


def sae_examples_dir(sae_key: str, set_name: str, root: str = VOL, write: bool = False,
                     corpus_name: str = "") -> str:
    """`<root>/base/<base>/sae/<sae>/examples/<set>` -- `scan`'s per-feature activation windows.

    KEYED BY SET since 2026-09-21 (B9). It used to be `examples/` keyed by the SAE alone, so a
    second `scan` of the same dictionary against a different held-out set refused without --force
    and destroyed the first set's examples with it; the eval plan runs three scans on `dict2m`.

    A READER (`write=False`) used to fall back to the legacy unkeyed directory when the keyed one
    was absent, with a stdout note and nothing else. IT NOW REFUSES (2026-09-23). The fallback was
    silent in every way that matters -- a `--set 2026-09-21_v3_ctrl` build whose scan had landed
    under the `--with-set` bank's name found no keyed directory and read the September
    `2026-09-16_v1` scan of a different set instead, producing a correct-looking C16 arm over the
    wrong features. A legacy directory records neither the set nor the corpus it was scanned
    against, so nothing here can check that it is the right one; naming the key that WAS expected
    is the only honest answer. A caller that genuinely wants the legacy product passes its path
    explicitly. A WRITER always writes the set-keyed path.

    Absent keyed AND absent legacy returns the keyed path unchanged, so a caller that tolerates a
    missing `examples/` (autointerp's `build`, which falls back to `examples_4m`) still sees it
    missing rather than an exception.
    """
    # The corpus axis too (H5): the examples of a feature are the windows it fires on IN A GIVEN
    # CORPUS, so two corpora give two different answers for one (set, sae) and must not share a
    # directory. Empty resolves to today's path.
    keyed = f"{sae_dir(sae_key, root)}/examples/{set_name}" + (f"__{corpus_name}" if corpus_name else "")
    if write:
        return keyed
    legacy = f"{sae_dir(sae_key, root)}/examples"
    assert os.path.exists(keyed) or not os.path.exists(f"{legacy}/tested.json"), (
        f"{keyed} is absent and the LEGACY unkeyed {legacy} is there. Reading it is refused: it "
        f"was written before 2026-09-21, when `examples/` gained a set component, and its path "
        f"records neither the set nor the corpus it was scanned against -- so it may be any set's "
        f"scan and nothing here can tell. The expected key is "
        f"set={set_name!r}, corpus_key={corpus_name or '(none)'!r}. Run `--product scan --set "
        f"{set_name}` at that corpus and tag, or -- if the scan exists under another bank's name "
        f"from a `scan --with-set` call -- address it by that directory."
    )
    return keyed


def repo_examples_dir(sae_key: str, set_name: str, root: str = VOL) -> str:
    """`<root>/base/<base>/sae/<sae>/repo_examples/<set>` -- the SAE repo's own shipped windows,
    scored for the features the held-out set `<set>` actually tests."""
    return f"{sae_dir(sae_key, root)}/repo_examples/{set_name}"


def gcg_dir(base: str, set_name: str, family: str, arm: str, root: str = VOL) -> str:
    """`<root>/base/<base>/gcg/<set>/<family>/<arm>` with arm = `<mode>-<init>`.

    Root-relative like every other base product, so a smoke writes its whole gcg/ subtree under
    /vol/runs/<date>_.... The FAMILY level is explicit (2026-09-16): the same four arms run on
    `realact` and on `sae` targets, and the two are different objects -- a realact direction is
    unit(X[p] - mu) and an sae direction is an encoder column -- so they never share a directory.
    """
    return f"{base_dir(base, root)}/gcg/{set_name}/{family}/{arm}"


def maem_dir(maem_key: str, root: str = VOL) -> str:
    """`<root>/maems/<base>/<name>`. Root-relative since step 3: a rollouts/score smoke writes the
    whole maems/ subtree under /vol/runs/<date>_faithfulness-smoke, exactly as base/ does."""
    base, name = split_key(maem_key, "maem")
    return f"{root}/maems/{base}/{name}"


ENGINES = ("hf", "vllm")


def rollout_stem(set_name: str, engine: str = "hf", tag: str = "") -> str:
    """The rollouts/ file stem of one (set, engine, tag) triple.

    The HF stem is the bare set name, so every step-3 file keeps its path; the vLLM stem is
    suffixed. Both engines write into the SAME accumulating rollouts/ directory and `score` picks
    one with `--engine`, so an HF and a vLLM run of the same set never overwrite each other and the
    paired comparison has both files side by side.

    `tag` (`--run-tag`) is the THIRD axis, added 2026-09-21 for the same reason `scan`'s examples/
    gained a set component: two runs of ONE checkpoint on ONE set that differ only in `--mu` are
    different experiments, and without a tag the second silently replaces the first -- the whole
    file, mid-comparison, with nothing raising. It is empty for every run that does not need it,
    so no existing path moves.
    """
    assert engine in ENGINES, f"unknown engine {engine!r}, want one of {list(ENGINES)}"
    tag = (tag or "").strip()
    assert "/" not in tag and " " not in tag, f"--run-tag {tag!r} must be a bare file-name suffix"
    stem = set_name if engine == "hf" else f"{set_name}__{engine}"
    return f"{stem}__{tag}" if tag else stem


def rollouts_path(maem_key: str, set_name: str, root: str = VOL, engine: str = "hf",
                  tag: str = "") -> str:
    return f"{maem_dir(maem_key, root)}/rollouts/{rollout_stem(set_name, engine, tag)}.jsonl"


def rollouts_dir(maem_key: str, root: str = VOL) -> str:
    return f"{maem_dir(maem_key, root)}/rollouts"


ROWS_MARK = "__rows"


def rollout_chunk_stem(stem: str, rows_spec: str = "") -> str:
    """The file stem of ONE `--rows` chunk of the rollouts product `stem`.

    A run over the whole set keeps the bare `rollout_stem` spelling, so every product written
    before 2026-09-23 and every full-set run after it is the same path it always was. A run given
    `--rows` writes `<stem>__rows<spec>.jsonl` instead, and the chunks of one (set, engine, tag)
    live SIDE BY SIDE in the one accumulating `rollouts/` directory -- which is only possible
    because the directory write is additive (OutDir). `read_rollouts` then reads all of them as
    ONE product, so `score` scores one stem and `results.common.discover_sources` sees one source:
    a per-chunk `--run-tag` would have made every chunk a separate arm in both OOD readers.

    The spec is spelled into the name rather than reduced to (lo, hi) because `--rows 3,5,9-11` is
    not an interval and a name that pretended it was would collide with `--rows 3-11`.
    """
    spec = (rows_spec or "").strip().replace(" ", "")
    if not spec:
        return stem
    assert all(c in "0123456789,-" for c in spec), f"--rows {rows_spec!r} is not a row spec"
    return f"{stem}{ROWS_MARK}{spec.replace(',', '_')}"


def rollout_chunk_paths(out_dir: str, stem: str) -> list[str]:
    """Every `--rows` chunk file of `stem` in `out_dir`, sorted by name. [] when there are none."""
    import glob as _glob

    return sorted(_glob.glob(f"{out_dir.rstrip('/')}/{stem}{ROWS_MARK}*.jsonl"))


# Summary fields that every chunk of one product must agree on: they describe the EXPERIMENT, and
# two chunks that disagree on one of them are two experiments wearing one stem.
_CHUNK_INVARIANT = (
    "maem", "base", "set", "engine", "kind", "n", "bo", "seed", "max_new", "min_new",
    "prompt", "prompt_tokens", "marker_pos", "inject_layer", "inject_coef",
    "temperature", "top_p", "top_k", "weight_sha256", "score_max_length",
)


def read_rollouts(out_dir: str, stem: str):
    """(rows, summary, sources) for the rollouts product `stem` -- whole, or as `--rows` chunks.

    One of the two shapes, never both (both is a refusal: a full-set file and a chunk of the same
    stem are two runs claiming one product, and silently preferring either is how a partial gets
    scored as if it were complete):

      * `<stem>.jsonl` + `<stem>.summary.json` -- one run over the whole set, the only shape any
        product written before 2026-09-23 has;
      * `<stem>__rows<spec>.jsonl` + summaries -- N chunks of ONE product under ONE `--run-tag`,
        concatenated here. The chunks must cover disjoint target rows and agree on every field of
        `_CHUNK_INVARIANT`; the merged summary carries the union of `rows` and a `chunks` list.
    """
    out_dir = out_dir.rstrip("/")
    whole = f"{out_dir}/{stem}.jsonl"
    chunks = rollout_chunk_paths(out_dir, stem)
    if os.path.exists(whole) and chunks:
        raise AssertionError(
            f"{whole} and {len(chunks)} `{ROWS_MARK}` chunk(s) of the same stem are both in "
            f"{out_dir} ({[os.path.basename(p) for p in chunks]}): that is a whole-set run and a "
            f"chunked run claiming one product. Keep one and move the other aside."
        )
    if os.path.exists(whole):
        with open(f"{out_dir}/{stem}.summary.json") as fh:
            return read_jsonl(whole), json.load(fh), [whole]
    assert chunks, (
        f"no rollouts at {whole} and no {stem}{ROWS_MARK}*.jsonl chunk beside it: run "
        f"`--product rollouts_* --set ...` first"
    )
    rows: list[dict] = []
    summary: dict = {}
    seen: dict[int, str] = {}
    for path in chunks:
        spath = path[: -len(".jsonl")] + ".summary.json"
        assert os.path.exists(spath), f"chunk {path} has no {os.path.basename(spath)} beside it"
        with open(spath) as fh:
            s = json.load(fh)
        if not summary:
            summary = dict(s)
        else:
            bad = {
                k: (summary.get(k), s.get(k))
                for k in _CHUNK_INVARIANT
                if summary.get(k) != s.get(k)
            }
            assert not bad, (
                f"{os.path.basename(path)} disagrees with {os.path.basename(chunks[0])} on "
                f"{bad}: the chunks of one product must be one experiment"
            )
        for r in s["rows"]:
            assert int(r) not in seen, (
                f"target row {r} is in both {seen[int(r)]} and {os.path.basename(path)}: the "
                f"chunks of one product must cover DISJOINT rows"
            )
            seen[int(r)] = os.path.basename(path)
        rows += read_jsonl(path)
    summary["rows"] = sorted(seen)
    summary["n_targets"] = len(seen)
    summary["chunks"] = [os.path.basename(p) for p in chunks]
    print(
        f"[rollouts] {stem}: {len(chunks)} `{ROWS_MARK}` chunk(s), {len(seen)} target rows, "
        f"{len(rows)} rollout rows",
        flush=True,
    )
    return rows, summary, chunks


def nla_variant_dir(maem_key: str, set_name: str, amp: str, root: str = VOL) -> str:
    """`<root>/maems/<base>/<nla>/variants/<set>__amp-<amp>` -- a NON-default `rollouts_nla --amp`.

    Its own one-shot directory in the `score --rollouts-dir` layout (`rollouts.jsonl` +
    `rollouts.summary.json` + `scores/`), deliberately NOT the accumulating `rollouts/`: an amp
    sweep is a different INPUT to the same model, and putting it under the set's own stem there
    would make it indistinguishable from the headline run in `index.json`.
    """
    assert amp and "/" not in amp and " " not in amp, f"amp {amp!r} must be a bare directory suffix"
    return f"{maem_dir(maem_key, root)}/variants/{set_name}__amp-{amp}"


def scores_dir(maem_key: str, set_name: str, root: str = VOL, engine: str = "hf",
               tag: str = "", write: bool = False) -> str:
    """`<root>/maems/<base>/<maem>/scores/<set>[__<engine>][__<tag>]` -- `rollout_stem`'s layout.

    `tag` IS IN THE SAME POSITION AS `rollout_stem`'s, which is the whole point of it existing
    here (C7, infra/2026-09-22_inventory-alignment.md). This function took no tag, so a tagged
    score run could only name itself through `--score-name <set>__<tag>` -- and `rollout_stem`
    then treated that whole string as the set and appended `__<engine>` AFTER the tag. On the
    volume:

        rollouts/ 2026-09-21_v3_ctrl__vllm__mu-none.jsonl     <- engine, then tag
        scores/   2026-09-21_v3_ctrl__mu-none__vllm/          <- tag, then engine

    One pair of products, two orders. `results/common.parse_scores_dir` was already patched to
    read the engine part wherever it sits, and its docstring records what the first version cost:
    six of eval 1's arms went into the paper's own CSV as HF when they were vLLM. Any future tool
    joining a rollout to its score by tag inherits the same trap. Writers are canonical from here.

    `tag` IS TWO AXES JOINED, not one -- build it with `score_tag_of(args)`, never by hand. Both
    `evals/pipeline` and `evals/pipeline-ood` gave this function a `tag` in this position within a
    day of each other and meant DIFFERENT things by it; see `score_tag_of`.

    A READER (the default) falls back to the legacy `<set>__<tag>__<engine>` spelling when the
    canonical path is absent and the legacy one is there, exactly as `sae_examples_dir` does for
    its own rename, so the products already on the volume stay readable. Only `engine != "hf"`
    can differ: at `hf` the two spellings are the same string.
    """
    tag = (tag or "").strip()
    canonical = f"{maem_dir(maem_key, root)}/scores/{rollout_stem(set_name, engine, tag)}"
    if write or not tag or engine == "hf":
        return canonical
    legacy = f"{maem_dir(maem_key, root)}/scores/{rollout_stem(f'{set_name}__{tag}', engine)}"
    if not os.path.exists(canonical) and os.path.exists(legacy):
        print(
            f"[scores] {canonical} is absent; reading the LEGACY {legacy} (written before "
            f"2026-09-21, when scores/ took its run tag through --score-name and so spelled it "
            f"after the engine instead of before). Same product, older name.",
            flush=True,
        )
        return legacy
    return canonical


def score_tag_of(args: dict) -> str:
    """The tag component of a scores directory: the run tag, the score tag, or both.

    TWO ORTHOGONAL AXES, and BOTH have to reach the name or one of them silently overwrites the
    other's product:

      `--run-tag`   selects which rollouts FILE is scored (`rollout_stem`'s third component).
                    Two run tags scored under one convention are two different inputs.
      `--score-tag` names only the OUTPUT: the same rollouts file scored again under a second
                    convention, a second mean, or for a column the first run did not have --
                    `cos_asym` is why it exists. Pointing `--run-tag` at a re-score instead sends
                    `score` looking for a `<set>__<engine>__<tag>.jsonl` that is not there, which
                    is how the mu-stats arm failed on 2026-09-21.

    Joined RUN FIRST, because the rollouts file is the outer object: every score of one rollouts
    file sorts together. `results.common.parse_scores_dir` returns the joined string as one tag,
    which is what it did before either axis existed and is all any reader has ever needed.

    `evals/pipeline` and `evals/pipeline-ood` each gave `scores_dir` a `tag` parameter in the same
    position, with the same name, meaning these two different things; the rebase that met them
    could have kept either one alone and lost the other's products with no error anywhere.
    """
    run = (args.get("run_tag") or "").strip()
    score = (args.get("score_tag") or "").strip()
    return "__".join(t for t in (run, score) if t)


# ---------------------------------------------------------------------------------------------
# corpus geometry and access (shared by stats.py pass A and scan.py pass B, which MUST agree on
# the window cut: the same window ids appear in topk.jsonl and in the SAE examples)
# ---------------------------------------------------------------------------------------------


def windows_of(n_tok: int, block: int = SCAN_BLOCK, stride: int = SCAN_STRIDE) -> list[tuple[int, int]]:
    """[(start, len)] of the scan windows of a document of `n_tok` tokens. Windows never cross docs.

    Rule, stated in every README that depends on it:
      * a document of <= `block` tokens is ONE window of its whole length;
      * otherwise full `block`-token windows start at 0, `stride`, 2*`stride`, ... for as long as
        the whole window fits;
      * if the last full window still leaves a tail uncovered, ONE partial window is appended at
        `last_start + stride` and runs to the end of the document (its length is in
        (block - stride, block), so it is never shorter than 49 tokens at 64/16).
    Every token of the document is therefore in at least one window.
    """
    assert n_tok > 0, f"empty document: n_tok={n_tok}"
    assert 0 < stride <= block, f"need 0 < stride <= block, got stride={stride} block={block}"
    if n_tok <= block:
        return [(0, n_tok)]
    out = []
    s = 0
    while s + block <= n_tok:
        out.append((s, block))
        s += stride
    last = out[-1][0]
    if last + block < n_tok:
        out.append((last + stride, n_tok - last - stride))
    return out


def size_tag_of(cum_before: int, n_tok: int, sizes: list[int]) -> int:
    """The nested-subset tag of a document: the smallest size (in MILLIONS of tokens) whose budget
    the document still fits under, given `cum_before` tokens already tagged.

    Documents are tagged in the PERMUTED corpus order, so subset k is exactly the prefix of the
    corpus with size_tag <= k and every nested subset is a contiguous prefix of tokens.i32.
    A document that overruns the largest size (possible only for the single document that crosses
    the total budget) is clamped to the largest size; corpus.py counts and reports those.
    """
    for s in sizes:
        if cum_before + n_tok <= round(s * 1_000_000):
            return s
    return sizes[-1]


def load_corpus(base: str, root: str = VOL, name: str = ""):
    """(tokens, docs) for a built corpus: a read-only int32 memmap and the docs.jsonl rows.

    The memmap is never randomly indexed across the whole file by the passes -- they walk documents
    in stored order (checklist item 76: volume random reads crawl).

    `name` selects a named corpus (`corpora/<name>/`, e.g. an OOD arm's in-domain corpus)
    instead of the base's own English one.
    """
    import numpy as np

    d = corpus_dir(base, root, name)
    docs = read_jsonl(f"{d}/docs.jsonl")
    toks = np.memmap(f"{d}/tokens.i32", dtype=np.int32, mode="r")
    assert docs, f"{d}/docs.jsonl is empty"
    end = docs[-1]["offset"] + docs[-1]["len"]
    assert end == len(toks), f"{d}: docs.jsonl ends at token {end} but tokens.i32 holds {len(toks)} tokens"
    return toks, docs


def corpus_sizes(docs: list[dict]) -> list[float]:
    """The nested sizes actually present in a built corpus, ascending (a smoke has fewer).

    Sizes are MILLIONS of tokens and may be FRACTIONAL: the 2026-09-20 ladder is
    1.25 / 2.5 / 5 / 10, for parity with training, which saw 9-10M activations. Integers
    stay integral so an existing 1/2/4/8/16 corpus reads back unchanged.
    """
    out = sorted({float(r["size_tag"]) for r in docs})
    return [int(s) if float(s).is_integer() else s for s in out]


def quantiles_from_hist(counts, qs, lo: float = -1.0, hi: float = 1.0):
    """Quantiles of a value binned into a uniform histogram over [lo, hi].

    `counts` is [..., n_bins]; returns [..., len(qs)] of bin UPPER edges, so every value is exact to
    one bin width ((hi - lo) / n_bins). An all-empty row returns `lo`.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    n_bins = counts.shape[-1]
    total = counts.sum(-1)  # [...]
    cum = np.cumsum(counts, axis=-1)  # [..., n_bins]
    q = np.asarray(qs, dtype=np.float64)  # [Q]
    targets = np.maximum(np.ceil(q * total[..., None]), 1).astype(np.int64)  # [..., Q]
    # first bin whose cumulative count reaches the target
    idx = (cum[..., None, :] >= targets[..., :, None]).argmax(-1)  # [..., Q]
    edges = lo + (idx + 1) * (hi - lo) / n_bins
    return np.where(total[..., None] > 0, edges, lo)


# ---------------------------------------------------------------------------------------------
# HF cache resolution (modal/maem_modal.py:_snapshot)
# ---------------------------------------------------------------------------------------------


def hf_home(cfg: dict) -> str:
    return cfg["modal"]["hf_home"]


def snapshot(cfg: dict, repo_id: str, repo_type: str = "model") -> str:
    """The one snapshot dir of an HF-cache repo on the volume (hash resolved at call time).

    Asserting a SINGLE snapshot is deliberate: two revisions in the cache means the weights a run
    used are ambiguous, and that has to be resolved by hand rather than by picking the newest.

    `repo_type` picks the cache prefix: huggingface_hub stores a dataset repo under `datasets--`,
    not `models--` (adamkarvonen's max-activating windows are a dataset repo).
    """
    import glob

    assert repo_type in ("model", "dataset"), f"repo_type must be 'model' or 'dataset', got {repo_type!r}"
    prefix = "models--" if repo_type == "model" else "datasets--"
    cache = prefix + repo_id.replace("/", "--")
    snaps = sorted(glob.glob(f"{hf_home(cfg)}/hub/{cache}/snapshots/*"))
    assert len(snaps) == 1, (
        f"expected exactly one snapshot under {cache} in {hf_home(cfg)}/hub, found {snaps} -- "
        f"fetch it with infra/fetch_hf.py, or delete the stale revision"
    )
    return snaps[0]


def maem_weights_path(cfg: dict, maem_key: str) -> str:
    """Directory holding the MAEM's weights: adapter dir (lora) or full snapshot (full).

    The subdir is JOINED onto the snapshot rather than passed as PeftModel(subfolder=...) because
    that is what modal/maem_modal.py:1842 does and because the resulting path is what the README
    and the weight sha have to name.
    """
    spec = cfg["maems"][maem_key]
    path = spec["src"] if "src" in spec else snapshot(cfg, spec["hf"])
    if spec.get("subdir"):
        path = os.path.join(path, spec["subdir"])
    # lora: the adapter dir. full / base: a model dir -- for `base` that is the base snapshot
    # itself, which is exactly the point of the control.
    marker = "adapter_config.json" if spec["type"] == "lora" else "config.json"
    assert os.path.exists(os.path.join(path, marker)), (
        f"maem {maem_key!r}: no {marker} under {path} (type={spec['type']})"
    )
    return path


def sae_path(cfg: dict, sae_key: str) -> str:
    spec = cfg["saes"][sae_key]
    path = os.path.join(snapshot(cfg, spec["hf"]), spec["file"])
    assert os.path.exists(path), f"sae {sae_key!r}: no checkpoint at {path}"
    return path


def max_acts_path(cfg: dict, sae_key: str) -> str:
    """The SAE's shipped max-activating-window file (`repo_examples`).

    `max_acts.repo` names a DIFFERENT repo when the windows are not in the SAE repo itself -- the
    8B's come from adamkarvonen's `sae_max_acts` dataset, not from his SAE repo -- and defaults to
    the SAE's own `hf` otherwise.
    """
    spec = cfg["saes"][sae_key]
    ma = spec.get("max_acts")
    assert ma, f"sae {sae_key!r} has no `max_acts` entry in config.yaml; `repo_examples` needs one"
    snap = snapshot(cfg, ma.get("repo", spec["hf"]), ma.get("repo_type", "model"))
    path = os.path.join(snap, ma["file"])
    assert os.path.exists(path), f"sae {sae_key!r}: no max-acts file at {path}"
    return path


# ---------------------------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------------------------


def load_base(cfg: dict, base: str, device: str = "cuda"):
    """(model, tok) for the clean base. attn_implementation='sdpa' as evals/heldout/eval_ckpt_daemon.py:341."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = snapshot(cfg, cfg["bases"][base]["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    )
    model.eval()
    print(f"[load] base {base} from {path} in {time.time() - t0:.0f}s", flush=True)
    return model, tok


def load_maem(cfg: dict, base: str, maem_key: str, device: str = "cuda"):
    """(model, tok, kind) for the GENERATING model. kind is the config `type`: 'lora', 'full' or 'base'.

    lora: base + PeftModel; scoring runs on the same object with the adapter disabled.
    full: the tuned model IS the generator and has no adapter to switch off, so the caller must
    load a separate clean base for scoring (evals/heldout/eval_ckpt_daemon.py:333-390).
    base: the UNTRAINED-BASE CONTROL -- no MAEM weights anywhere; `maem_weights_path` resolves to
    the base's own snapshot, so this loads exactly what `load_base` would and takes the same
    no-adapter path as `full`. The kind is returned verbatim so every caller can tell the control
    apart from a trained full-parameter MAEM (their marker-norm expectations are OPPOSITE).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key_base, _ = split_key(maem_key, "maem")
    assert key_base == base, f"maem {maem_key!r} is not on base {base!r}"
    spec = cfg["maems"][maem_key]
    path = maem_weights_path(cfg, maem_key)
    if spec["type"] == "lora":
        import torch
        from peft import PeftModel

        # `parent:` -- a LoRA whose base is ANOTHER MAEM rather than the clean base. Needed the
        # moment you train an adapter on top of a full-parameter checkpoint (2026-09-23: the
        # rare-feature LoRA over rl-final): without it the adapter loads onto the UNTRAINED base,
        # every product runs, and the tables report a model nobody trained. The parent is resolved
        # through this same function, so a parent may itself be `full`, `lora` or `base`.
        parent = spec.get("parent")
        if parent:
            assert parent in cfg["maems"], f"unknown parent {parent!r} of {maem_key!r}"
            assert parent != maem_key, f"maem {maem_key!r} is its own parent"
            model, tok, pkind = load_maem(cfg, base, parent, device=device)
            print(f"[load] parent {parent} ({pkind}) under adapter {path}", flush=True)
            model = PeftModel.from_pretrained(model, path, is_trainable=False,
                                              torch_dtype=torch.bfloat16)
            model.eval()
            return model, tok, "lora"
        model, tok = load_base(cfg, base, device=device)
        t0 = time.time()
        model = PeftModel.from_pretrained(model, path, is_trainable=False, torch_dtype=torch.bfloat16)
        model.eval()
        print(f"[load] lora adapter {path} in {time.time() - t0:.0f}s", flush=True)
        return model, tok, "lora"

    import torch

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    )
    model.eval()
    print(f"[load] {spec['type']} model {path} in {time.time() - t0:.0f}s", flush=True)
    return model, tok, spec["type"]


def clean_adapter(model):
    """Context manager putting `model` in clean-base mode for scoring.

    A PeftModel disables its adapter; a plain model (full-parameter MAEM protocol, where the
    scoring model is a separately loaded clean base) is already clean, so this is a no-op rather
    than an error -- the caller decides which object to hand in.
    """
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------
# prompts (maem/prompts.py)
# ---------------------------------------------------------------------------------------------


def _chat_ids(tok, content, add_gen):
    """maem/prompts.py:18-24, verbatim."""
    out = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=add_gen,
        enable_thinking=False,
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    while isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def marker_positions(tok, ids):
    """maem/prompts.py:27-32, verbatim."""
    mid = tok.encode(MARKER, add_special_tokens=False)
    assert len(mid) == 1, f"marker not single-token: {mid}"
    pos = [i for i, t in enumerate(ids) if t == mid[0]]
    assert len(pos) == 1, f"expected exactly one marker, got {len(pos)}"
    return pos


def _maem_prompt(tok, read_layer: int):
    """maem/prompts.py:35-40: the marker goes AFTER the chat template's generation prefix."""
    instr = _INSTR_TEMPLATE.format(read_layer=read_layer)
    ids = _chat_ids(tok, instr, add_gen=True) + tok.encode(MARKER, add_special_tokens=False)
    return ids, len(ids) - 1


# ours8b and maem27b are today the SAME template at read layer 27 vs 42 (this repo's
# maem/prompts.py substitutes READ_LAYER=27 into the upstream text). Both names are kept so that a
# future MAEM trained on a different prompt costs one config entry and one function here.
PROMPTS = {
    "ours8b": _maem_prompt,
    "maem27b": _maem_prompt,
}


def prompt_ids(tok, prompt_name: str, read_layer: int):
    """(ids, marker_pos). Asserts the marker is one token occurring exactly once."""
    assert prompt_name in PROMPTS, f"unknown prompt {prompt_name!r}, want one of {sorted(PROMPTS)}"
    ids, pos = PROMPTS[prompt_name](tok, read_layer)
    found = marker_positions(tok, ids)
    assert found == [pos], f"marker at {found} but the prompt builder reported {pos}"
    return ids, pos


# ---------------------------------------------------------------------------------------------
# injection and reading (maem/inject.py)
# ---------------------------------------------------------------------------------------------


def get_layer(model, layer: int):
    """maem/inject.py:10-14, verbatim: the decoder block at `layer`, unwrapping DDP + PEFT."""
    m = model.module if hasattr(model, "module") else model
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.model.layers[layer]


def make_inject_hook(vecs, positions, coeff, device, dtype):
    """maem/inject.py:24-57 with mode='add' only (the only mode any MAEM was trained with).

    vecs: list of [k_i, d] directions, one row per marker position of batch row i.
    The `h.shape[1] <= 1` guard is the decode step under the KV cache: the marker was injected at
    prefill and re-injecting during decode would corrupt it.
    """
    import torch

    if len(vecs) != len(positions):
        raise ValueError(f"{len(vecs)} vector rows != {len(positions)} position rows")
    counts = [len(p) for p in positions]
    if any(v.shape[0] != n for v, n in zip(vecs, counts, strict=True)):
        raise ValueError("each vector row must have one vector per marker position")
    normed = torch.nn.functional.normalize(torch.cat(vecs).to(device, dtype), dim=-1)
    rows = torch.repeat_interleave(
        torch.arange(len(vecs), device=device), torch.tensor(counts, device=device)
    )
    cols = torch.tensor([p for row in positions for p in row], device=device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        if h.shape[0] != len(vecs):
            raise RuntimeError(f"inject batch {h.shape[0]} != {len(vecs)} vector rows")
        base = h[rows, cols]
        scale = base.norm(dim=-1, keepdim=True) * coeff
        h[rows, cols] = base + (normed * scale).to(h.dtype).detach()
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


@contextlib.contextmanager
def hooked(module, hook):
    """maem/inject.py:129-135, verbatim."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class _Stop(Exception):
    """Raised by the read hook to abort the forward once the read layer has been captured."""


# The same exception under a public name: gcg/gcg.py runs its OWN grad-carrying read pass (it needs
# the graph, which read_resid's @no_grad discards) and must catch exactly this to stop the forward.
StopForward = _Stop


def read_layer_hook(captured: dict):
    """maem/inject.py:147-149: capture the block OUTPUT (never output_hidden_states) and stop."""

    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).float()
        raise _Stop

    return cap


def read_resid(model, layer, batch, pool="all"):
    """maem/inject.py:142-166, verbatim. Layer-`layer` residual for a tokenized batch, no injection.

    The forward is aborted at the read layer, so nothing above it is computed.
    """
    import torch

    captured: dict = {}
    h = None
    handle = get_layer(model, layer).register_forward_hook(read_layer_hook(captured))
    try:
        with torch.no_grad():
            model(**batch)
    except _Stop:
        h = captured["h"]
    finally:
        handle.remove()
    assert h is not None, f"read hook at layer {layer} never fired (forward returned without it)"
    mask = batch["attention_mask"].bool()
    if pool == "all":
        return h, mask
    if pool == "last":
        idx = mask.sum(1) - 1
        return h[torch.arange(h.shape[0]), idx]
    assert pool == "mean", f"pool must be 'all' | 'last' | 'mean', got {pool!r}"
    summed = (h * mask.unsqueeze(-1)).sum(1)
    return summed / mask.sum(1, keepdim=True).clamp(min=1)


def marker_norm(model, prompt_ids_, pos: int, inject_layer: int, adapter: bool = True) -> float:
    """train/rl/rl.py:170-192: ||h|| at the marker of INJECT_LAYER's output for the shared prompt.

    This is the exact scalar make_inject_hook multiplies unit(dir) by. Checklist item 22: every
    rollout run logs it under BOTH the served model and the clean base and asserts they differ --
    adapter-on ~98.5 vs base ~14.06 for 8B run1; equality is the silent-adapter-off signature.
    """
    import torch

    cap: dict = {}

    def grab(_m, _i, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out

    submodule = get_layer(model, inject_layer)
    handle = submodule.register_forward_hook(grab)
    try:
        ids = torch.as_tensor(
            prompt_ids_, dtype=torch.long, device=next(model.parameters()).device
        ).unsqueeze(0)
        with torch.no_grad():
            if adapter:
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
            else:
                with clean_adapter(model):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()
    assert "h" in cap, f"inject-layer hook at layer {inject_layer} never fired"
    return cap["h"][0, pos].norm().float().item()


# ---------------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------------


def sink_token_id(tok) -> int:
    """The token prepended at column 0 of every scored / scanned window and then dropped.

    evals/heldout/eval_universal.py:139 uses bos, falling back to eos: the Qwen3 tokenizers have no bos, so
    in practice this is the eos id. Named once so the scorer, the corpus passes and the SAE
    examples all prepend the SAME token (a different sink shifts every position's context).
    """
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert sink is not None, "tokenizer has neither a bos nor an eos token to use as the sink"
    return int(sink)


def strip_repo_sink(ids, acts, sink: int, sink_first: bool):
    """Align an SAE repo's shipped max-activating windows with our scoring protocol.

    A repo whose scan re-encoded each window the way `evals/heldout/eval_universal.py:_reencode` does has a
    SINK TOKEN at position 0 of every window (Qwen3 has no bos, so it is the tokenizer's eos).
    `common.score_tokens` prepends its own sink, so that column must be dropped before scoring --
    from the ids AND from the per-token activations, or the repo's activations no longer line up
    with the tokens they were measured on.

    `sink_first` is the config's declaration for this file; it is ASSERTED against the data rather
    than inferred, because a file that is 99% sink-prefixed and 1% not would otherwise strip the
    wrong column on some rows and nothing would say so.

    Returns (ids, acts, frac_sink_first) with the same leading dimensions and one column fewer when
    the sink was stripped. Pure: no globals, no device assumptions.
    """
    assert ids.shape == acts.shape, (
        f"ids {tuple(ids.shape)} and acts {tuple(acts.shape)} must have the SAME shape: every "
        f"shipped activation belongs to exactly one shipped token"
    )
    assert ids.shape[-1] >= 2, f"a window of {ids.shape[-1]} tokens has nothing left after a strip"
    first = ids[..., 0] == sink
    n = int(first.numel())
    frac = float(first.sum()) / n
    if sink_first:
        assert frac == 1.0, (
            f"config says this max-acts file prepends the sink token {sink} to EVERY window, but "
            f"only {int(first.sum())} of {n} windows start with it (frac {frac:.6f}); stripping "
            f"column 0 would drop a real token on the rest and shift every activation by one"
        )
        return ids[..., 1:], acts[..., 1:], frac
    assert frac < 0.5, (
        f"config says this max-acts file has NO sink prefix, but {frac:.6f} of its {n} windows "
        f"start with token {sink}: it looks sink-prefixed and the strip was skipped, which would "
        f"give score_tokens two sinks in a row"
    )
    return ids, acts, frac


def eos_ids(tok, model) -> set[int]:
    """train/rl/rl.py:61-75: tokenizer eos UNION generation_config eos (checklist item 85).

    The upstream version wraps the generation_config read in a bare `except Exception: pass`; here a model
    without a generation_config simply contributes nothing, and any other failure propagates.
    """
    ids: set[int] = set()

    def add(e):
        if isinstance(e, list | tuple):
            for x in e:
                ids.add(int(x))
        elif e is not None:
            ids.add(int(e))

    add(tok.eos_token_id)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        add(getattr(gen_cfg, "eos_token_id", None))
    assert ids, "no eos id found on either the tokenizer or the generation config"
    return ids


def trim_at_stop(g, stop_ids):
    """train/rl/rl.py:82-90: keep tokens up to and INCLUDING the first stop token; drop any pad tail."""
    trimmed = []
    for t in g:
        trimmed.append(t)
        if t in stop_ids:
            break
    return trimmed if trimmed else list(g)


def vllm_finish_ids(token_ids, finish_reason: str, stop_reason, stop_ids, eos_fallback: int):
    """vLLM's returned ids -> the ids rollouts_hf would have stored. Returns (ids, appended).

    vLLM drops the stop token from `token_ids` when it stopped on one of `stop_token_ids`
    (train/rl/rl.py:310-313 re-appends it, because train/rl/rl.py:82-90's trimming KEEPS the stop token and the
    two engines' rows must be comparable). `stop_reason` carries the id it stopped on; a
    non-integer stop_reason falls back to `eos_fallback` (the tokenizer eos).

    Only a `finish_reason == "stop"` row whose last id is not already a stop token is appended to;
    a "length" row is returned as is. The result is then trimmed exactly as the HF path trims.
    """
    ids = [int(t) for t in token_ids]
    appended = False
    if finish_reason == "stop" and (not ids or ids[-1] not in stop_ids):
        ids.append(int(stop_reason) if isinstance(stop_reason, int) else int(eos_fallback))
        appended = True
    return trim_at_stop(ids, stop_ids), appended


# train/rl/rl.py:192-211 -- the adapter module rename vLLM needs for the 27B, which it serves as
# Qwen3_5ForConditionalGeneration. vLLM validates adapter module names by SUFFIX ONLY, so a
# CausalLM-named adapter passes validation and is then SILENTLY ignored (measured: 1.47 nats of
# logprob gap). The 8B is a plain CausalLM and needs no rename.
VLLM_LORA_RENAME = {"qwen36-27b": ("model.layers.", "model.language_model.layers.")}


def rename_lora_keys(state_dict_keys, base: str) -> dict[str, str]:
    """{old key -> new key} for `base`'s vLLM naming. Identity when the base needs no rename.

    A key that already mentions `language_model` is left alone (train/rl/rl.py:205), and the prefix is
    replaced ONCE, at the front, so a module literally called `model.layers.` deeper in a name
    cannot be rewritten by accident.
    """
    rule = VLLM_LORA_RENAME.get(base)
    out = {}
    for k in state_dict_keys:
        if rule is None or "language_model" in k:
            out[k] = k
            continue
        old, new = rule
        out[k] = k.replace(old, new, 1)
    assert len(set(out.values())) == len(out), (
        f"the vLLM adapter rename collapsed two distinct module names into one for base {base!r}"
    )
    return out


def parse_rows(spec: str, n_rows: int) -> list[int]:
    """`--rows` -> a sorted list of target row indices. "" means every row.

    Accepts comma-separated singletons and inclusive `a-b` ranges: "0-7", "0-15", "3,5,9-11".
    Duplicates collapse; every index must be in range, so a typo cannot silently score fewer rows.
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_rows))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            assert lo <= hi, f"--rows range {part!r} runs backwards ({lo} > {hi})"
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if not 0 <= i < n_rows)
    assert not bad, f"--rows {spec!r} names rows {bad} outside the set's 0..{n_rows - 1}"
    assert out, f"--rows {spec!r} selected no rows"
    return sorted(out)


def gen_seed_for(base_seed: int, row: int, k: int, n: int) -> int:
    """The per-generate-call seed of the call whose FIRST row is rollout `k` of target `row`.

    Rule (stated in every rollouts README): the (target, rollout) grid is flattened in target-major
    order, `flat = row * n + k`, cut into contiguous chunks of `gen_rows`, and the chunk is seeded
    `rollouts.seed * 1000 + flat_of_its_first_row` right before `generate`. HF `generate` takes no
    `torch.Generator` on every version, so the seed is set with `torch.manual_seed` /
    `torch.cuda.manual_seed_all` instead and recorded on every row it produced. The flat index is
    GLOBAL (the set's own row numbering), so `--rows 4-7` reproduces the rows `--rows 0-15` made,
    as long as the chunking still lands on the same boundaries.
    """
    assert n > 0 and 0 <= k < n, f"rollout index k={k} must be in [0, n) with n={n}"
    return int(base_seed) * 1000 + row * n + k


# ---------------------------------------------------------------------------------------------
# scoring: the one re-encode protocol (evals/heldout/eval_universal.py:129-162)
# ---------------------------------------------------------------------------------------------


def encode_for_score(tok, texts, max_length: int = SCORE_MAX_LENGTH):
    """The TOKENIZATION half of the scoring protocol, on its own: -> a list of id lists.

    evals/heldout/eval_universal.py:136-138 -- an all-whitespace rollout tokenizes to zero tokens, so `" "`
    is substituted to keep the row scoreable; then add_special_tokens=False and truncation at
    `max_length`. Nothing is padded here and no sink is prepended: `score_ids` owns both, because a
    caller that already HAS ids (gcg/gcg.py optimises ids, never text) must reach the SAME forward
    without passing through a tokenizer at all.

    `max_length` defaults to SCORE_MAX_LENGTH, which is the protocol and what every arm but one
    uses. It is a PARAMETER only because the NLA arm generates at the checkpoint's native 200
    tokens and a 95-token cut would score less than half of each text (README, "The NLA arm").
    A run that widens it records the value it used in its own outputs -- `score` writes
    `score_max_length` into `rows.json` and `score_width_of` reads it back -- so a stored array's
    width is never inferred from a constant that has since changed.
    """
    assert max_length >= 1, f"max_length must be at least one token, got {max_length}"
    chunk = [t if t.strip() else " " for t in texts]
    prev = tok.padding_side
    tok.padding_side = "right"  # the protocol's side (eval_universal.py:133); nothing pads here
    try:
        enc = tok(chunk, add_special_tokens=False, truncation=True, max_length=max_length)
    finally:
        tok.padding_side = prev
    return [list(x) for x in enc["input_ids"]]


def score_width_of(sdir: str | Path) -> int:
    """The stored width T of a `scores/<set>/` directory's [N, n, T] arrays.

    `score` writes `score_max_length` into `rows.json`; a directory written before that field
    existed, or by a run at the protocol's own truncation, has none and is SCORE_WIDTH. Anything
    that reshapes a stored `cos.f16` must go through this rather than through the constant, or it
    silently misreads an arm scored at a different width as a different number of rows.
    """
    path = Path(sdir) / "rows.json"
    assert path.exists(), f"no {path}: a scores/ directory always carries rows.json"
    with open(path) as fh:
        rows_json = json.load(fh)
    max_length = int(rows_json.get("score_max_length", SCORE_MAX_LENGTH))
    assert max_length >= 1, f"{path}: score_max_length {max_length} is not a token count"
    return max_length + 1


def score_ids(
    model,
    tok,
    id_lists,
    dirs,
    read_layer,
    sbatch: int = SCORE_CHUNK,
    device="cuda",
    on_chunk=None,
    max_length: int = SCORE_MAX_LENGTH,
    *,
    dirs_centred=None,
    mu=None,
):
    """Per-token cosine and residual norm of each ID LIST on the CLEAN base at `read_layer`.

    THE scoring forward of the whole pipeline. `score_tokens` is this function with
    `encode_for_score` in front of it, so rollouts, scan references, SAE repo windows and the GCG
    loop cannot drift on the objective: they are the same code path by construction, not by
    agreement (checklist item 12).

    Protocol, all of it from evals/heldout/eval_universal.py:_reencode: right padding, a BOS sink (or eos if
    the tokenizer has no bos) prepended at column 0 and excluded from `keep`, cosine in fp32.

    DIVERGENCE (2026-09-15): the upstream 10x-nanmedian norm filter (eval_universal.py:71,145-147) is
    NOT applied. The per-token norm is stored so reconstruction/stats.py can apply it as an option.

    TWO COSINES FROM ONE FORWARD (2026-09-21). `cos` is the uncentred number this function has
    always produced. Passing BOTH `dirs_centred` and `mu` adds `cos_centred` to the output:

        cos          = einsum(normalize(h),      normalize(dirs))
        cos_centred  = einsum(normalize(h - mu), normalize(dirs_centred))
        cos_asym     = einsum(normalize(h),      normalize(dirs_centred))   # the scan's convention

    -- the same residual, a second einsum, ~0 extra GPU time, and the two sides of the centred
    number are centred by the SAME mean. This function CANNOT derive `dirs_centred` itself: it is
    handed unit vectors, and unit(act) together with mu does not give unit(act - mu) without
    ||act||, which lives in the set's ids.jsonl. So the caller (score.py, which owns rows_meta)
    builds both tensors and puts NaN rows where the family is not centrable -- an encoder column
    has no mean, and cos(h - mu, encoder column) is a one-sided number, not a centred one. NaN
    propagates through the normalize and the einsum on its own; nothing special-cases it.

    A caller that passes neither keyword gets bit-identical output to before: gcg.py:332 and
    sae_self.py:246 call with keywords only and are untouched, which is the one-scoring-path
    invariant gcg.py:19-21 names.

    Every OTHER cosine in the pipeline is uncentred while a legacy realact target direction is
    unit(act - mu) (data/build_universal_bank.py:26,310). That asymmetry is known; it is what
    `cos_centred` exists to measure against rather than to replace.

    `model` must already be the scoring model: a PeftModel (its adapter is disabled here) or a
    separately loaded clean base (full-parameter MAEMs have no adapter to switch off).

    Returns a dict of [N, max_length + 1] tensors on the cpu -- cos (f32), norm (f32), keep (bool),
    ids (i64), and cos_centred (f32) when asked for -- where column 0 is the sink, cos/norm are NaN
    outside `keep`, and ids is -1 there.
    Rows are rectangular across chunks of different token lengths, so the arrays concatenate.
    `max_length` defaults to SCORE_MAX_LENGTH and every caller but the NLA arm leaves it there;
    see `encode_for_score` for why it is a parameter and who records the value used.

    `on_chunk(s, h, cos, keep, ids)` is called once per chunk with that chunk's fp32 read-layer
    residual still on the device, so a caller needing more than cos/norm (score.py wants the
    residual at the argmax token and the SAE features there) gets it from THIS forward instead of
    running a second, divergent one.
    """
    import torch
    import torch.nn.functional as F

    n = len(id_lists)
    assert len(dirs) == n, f"{n} id lists but {len(dirs)} directions: the scorer pairs them by row"
    assert max_length >= 1, f"max_length must be at least one token, got {max_length}"
    assert (dirs_centred is None) == (mu is None), (
        "score_ids takes dirs_centred and mu TOGETHER or neither: the centred cosine centres both "
        "sides by the same mean, and one without the other is the asymmetric number this keyword "
        "pair exists to replace"
    )
    want_centred = dirs_centred is not None
    if want_centred:
        assert len(dirs_centred) == n, (
            f"{n} id lists but {len(dirs_centred)} centred directions"
        )
        mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device).reshape(-1)
        assert mu_t.shape[0] == int(dirs.shape[-1]), (
            f"mu is [{mu_t.shape[0]}] but the directions are [.., {int(dirs.shape[-1])}]"
        )
    # +1 for the sink at column 0. This is SCORE_WIDTH whenever max_length is the protocol's own
    # SCORE_MAX_LENGTH, which is every caller but the NLA arm.
    score_width = max_length + 1
    out = {
        "cos": torch.full((n, score_width), float("nan")),
        "norm": torch.full((n, score_width), float("nan")),
        "keep": torch.zeros((n, score_width), dtype=torch.bool),
        "ids": torch.full((n, score_width), -1, dtype=torch.long),
    }
    if want_centred:
        out["cos_centred"] = torch.full((n, score_width), float("nan"))
        # THE ASYMMETRIC COSINE (2026-09-21): the scorer's side UNCENTRED against the
        # CENTRED target. It is the convention `scan` uses for every corpus window
        # (`normalize(h) @ unit(act - mu)`, precompute/scan.py) and the one the paper's bo64
        # 0.569 and corpus 0.351 are both stated in, so it is the only one of the three that can
        # be differenced against a corpus search. The legacy path produced it for free, because a
        # `storage: unit` set's stored rows ARE unit(act - mu) and `dirs` was already the centred
        # target; on a `storage: raw` set nothing did until this column.
        out["cos_asym"] = torch.full((n, score_width), float("nan"))
    sink = sink_token_id(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else sink
    with torch.no_grad():
        for s in range(0, n, sbatch):
            chunk = [list(x) for x in id_lists[s : s + sbatch]]
            b = len(chunk)
            lens = [len(x) for x in chunk]
            assert min(lens) >= 1, (
                f"row {s + lens.index(min(lens))} has no tokens at all; the caller must substitute "
                f"something scoreable (encode_for_score puts a single space)"
            )
            assert max(lens) <= max_length, (
                f"row {s + lens.index(max(lens))} carries {max(lens)} ids, above the re-encode "
                f"truncation at max_length={max_length}: it must be truncated BEFORE scoring"
            )
            width = max(lens)
            ids = torch.full((b, width + 1), pad, dtype=torch.long)
            am = torch.zeros((b, width + 1), dtype=torch.long)
            ids[:, 0], am[:, 0] = sink, 1
            for i, x in enumerate(chunk):
                ids[i, 1 : 1 + len(x)] = torch.as_tensor(x, dtype=torch.long)
                am[i, 1 : 1 + len(x)] = 1
            ids, am = ids.to(device), am.to(device)
            with clean_adapter(model):
                h, mask = read_resid(
                    model, read_layer, {"input_ids": ids, "attention_mask": am}, pool="all"
                )
            keep = mask.clone()
            keep[:, 0] = False  # the sink is never a candidate token
            d = F.normalize(dirs[s : s + b].to(device).float(), dim=-1)
            cos = torch.einsum("btd,bd->bt", F.normalize(h.float(), dim=-1), d)
            nrm = h.float().norm(dim=-1)
            cos_c = None
            if want_centred:
                dc = F.normalize(dirs_centred[s : s + b].to(device).float(), dim=-1)
                cos_c = torch.einsum(
                    "btd,bd->bt", F.normalize(h.float() - mu_t, dim=-1), dc
                )
                cos_a = torch.einsum("btd,bd->bt", F.normalize(h.float(), dim=-1), dc)
            t = ids.shape[1]
            assert t <= score_width, (
                f"chunk width {t} exceeds this run's width {score_width}: truncation at "
                f"max_length={max_length} did not hold"
            )
            if on_chunk is not None:
                on_chunk(s, h, cos, keep, ids)
            out["keep"][s : s + b, :t] = keep.cpu()
            out["ids"][s : s + b, :t] = torch.where(mask, ids, torch.full_like(ids, -1)).cpu()
            out["cos"][s : s + b, :t] = torch.where(keep, cos, torch.full_like(cos, float("nan"))).cpu()
            out["norm"][s : s + b, :t] = torch.where(keep, nrm, torch.full_like(nrm, float("nan"))).cpu()
            if want_centred:
                out["cos_centred"][s : s + b, :t] = torch.where(
                    keep, cos_c, torch.full_like(cos_c, float("nan"))
                ).cpu()
                out["cos_asym"][s : s + b, :t] = torch.where(
                    keep, cos_a, torch.full_like(cos_a, float("nan"))
                ).cpu()
    return out


def score_tokens(
    model,
    tok,
    texts,
    dirs,
    read_layer,
    sbatch: int = SCORE_CHUNK,
    device="cuda",
    on_chunk=None,
    max_length: int = SCORE_MAX_LENGTH,
    *,
    dirs_centred=None,
    mu=None,
):
    """`score_ids` with the tokenizer in front: see `encode_for_score` and `score_ids`.

    This is a two-line wrapper on purpose. Before 2026-09-16 it held the scoring forward itself and
    GCG (which optimises IDS, not text) would have needed its own copy; now there is one forward and
    the text path is the one with an extra step, not the other way round. `unit_smoke`'s
    `check_score_ids_is_score_tokens` asserts the two agree bit for bit on independently built ids.
    """
    return score_ids(
        model,
        tok,
        encode_for_score(tok, texts, max_length),
        dirs,
        read_layer,
        sbatch=sbatch,
        device=device,
        on_chunk=on_chunk,
        max_length=max_length,
        # score.py reaches score_ids through THIS wrapper, not directly, so the centred pair has to
        # be forwarded here or the second cosine never leaves the caller.
        dirs_centred=dirs_centred,
        mu=mu,
    )


def agg(cos, keep):
    """(best, argmax) over kept tokens; eval_universal.py:161's masked_fill(-1).max(1).

    A row with no kept token (an empty rollout) scores -1.0 at column 0 rather than raising: that
    is a real, reportable outcome, and stats.py separates it by keep.sum(1) == 0.
    """
    import torch

    masked = torch.where(keep, cos, torch.full_like(cos, -1.0))
    best, arg = masked.max(dim=1)
    return best, arg


def bo_ladder(vals, ks) -> dict[int, float]:
    """{k: UNBIASED best-of-k} over `vals`, the n per-rollout scores of one target.

        E[max of k draws] = sum_{i=1..n} x_(i) * C(i-1, k-1) / C(n, k)      (x sorted ASCENDING)

    The i-th smallest of the n observed scores is the maximum of a k-subset exactly when the other
    k-1 members come from the i-1 below it, and every k-subset is equally likely. Unbiased for any
    k <= n and using ALL n rollouts. k > n is SKIPPED, never clamped, so a summary never claims a
    bo-k it could not compute.

    ONE ESTIMATOR IN THE PIPELINE (2026-09-23, M0a). This replaced `best_of_k_means`, the
    disjoint-group mean -- floor(n/k) consecutive groups of k, each group's max, averaged -- which
    `score` stored while `reconstruction/stats.py` printed the unbiased one, so the same quantity
    had two values depending on which file a reader opened and they agreed only at k = n.

    It is DELIBERATELY duplicated in `results/common.bo_unbiased`: this module is what the Modal
    container ships and `results/` is a standalone local script layer that imports nothing from
    it, exactly as `read_array` is duplicated. `results/selftest.check_one_bo_estimator` asserts
    the two agree to floating point on a random draw, so the duplicate is checked, not trusted.
    """
    import math as _math

    vals = sorted(float(v) for v in vals)
    n = len(vals)
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        assert k >= 1, f"best-of-k needs k >= 1, got {k}"
        if k > n:
            continue
        denom = _math.comb(n, k)
        out[k] = sum(vals[i - 1] * _math.comb(i - 1, k - 1) for i in range(1, n + 1)) / denom
    return out


# ---------------------------------------------------------------------------------------------
# SAE (maem/sae.py:20-52 + evals/heldout/eval_universal.py:77-93 for the gate)
# ---------------------------------------------------------------------------------------------


class BatchTopKSAE:
    """W_enc [d, F], W_dec [F, d], b_enc [F], b_dec [d], threshold: the learned BatchTopK gate."""

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, col_of=None, d_sae_full=0):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.threshold = threshold
        self.d_in, self.n_cols = W_enc.shape
        # A COLUMN SLICE carries the map from the dictionary's feature id to its column here, and
        # still reports the dictionary's true size as `d_sae` -- a slice that called itself a
        # 16-feature SAE would make every "2097152 features" line a lie. `col_of is None` is the
        # whole dictionary, where the two indices coincide.
        self.col_of = col_of
        self.d_sae = d_sae_full or self.n_cols


def load_sae(path: str, d_model: int, device: str = "cpu", dtype=None, need_decoder: bool = True):
    """maem/sae.py:39-52 plus the `threshold` buffer the upstream eval reads through sae_gate().

    The checkpoint is a dictionary_learning nn.Linear state dict, so both weight matrices are
    stored [out, in] and are transposed here.

    `need_decoder=False` skips W_dec entirely -- it is never moved to the device and its
    unit-norm check is skipped. At 2^21 features W_dec is 43 GB in fp32, which is the
    difference between fitting an H200 beside the 27B and not: the `stats` pass reads only
    b_dec, W_enc, b_enc and threshold, so it pays 43 GB for a matrix it never touches.
    A caller that then reaches for sae.W_dec gets a clear AttributeError, not a wrong number.
    """
    import torch

    dtype = dtype or torch.float32
    params = torch.load(path, map_location="cpu", weights_only=False)
    key_map = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "b_dec": "b_dec",
    }  # dictionary_learning aliases for b_dec
    wanted = set(key_map.values()) if need_decoder else set(key_map.values()) - {"W_dec"}
    t = {key_map[k]: v.to(dtype) for k, v in params.items()
         if k in key_map and key_map[k] in wanted}
    missing = ({"W_enc", "W_dec", "b_enc", "b_dec"} if need_decoder
               else {"W_enc", "b_enc", "b_dec"}) - set(t)
    assert not missing, f"SAE {path}: missing {sorted(missing)} (checkpoint keys: {sorted(params)})"
    # evals/heldout/eval_universal.py:77-84: "fired" is act > this learned threshold (~1.654 for the 131k
    # 27B SAE), not the older arbitrary raw-act > 1.0 cut.
    raw_thr = params.get("threshold")
    assert raw_thr is not None, (
        f"SAE {path} has no 'threshold' buffer; the fire gate would be undefined "
        f"(checkpoint keys: {sorted(params)})"
    )
    threshold = float(raw_thr.item() if hasattr(raw_thr, "item") else raw_thr)
    assert threshold > 0, f"SAE {path}: threshold {threshold} must be > 0"
    sae = BatchTopKSAE(
        t["W_enc"].T.contiguous().to(device),  # nn.Linear stores [out, in]
        t["W_dec"].T.contiguous().to(device) if need_decoder else None,
        t["b_enc"].to(device),
        t["b_dec"].to(device),
        threshold,
    )
    assert sae.d_in == d_model, f"SAE d_in {sae.d_in} != base d_model {d_model}"
    if need_decoder:
        nrm = sae.W_dec.norm(dim=1)
        assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-2), (
            f"decoder rows must be unit norm; got min {nrm.min():.4f} max {nrm.max():.4f}"
        )
    return sae


def _sae_cols(sae: BatchTopKSAE, feature_ids) -> list[int]:
    """Dictionary feature ids -> column numbers of THIS object's W_enc.

    Identity for a full dictionary; the slice's own map for one from `load_sae_columns`, which
    refuses an id it does not hold rather than returning some other feature's column. Shared by
    `sae_encode` and `sae_dirs` so the two cannot come to mean different things by one id.
    """
    ids = [int(f) for f in feature_ids]
    if sae.col_of is None:
        return ids
    missing = sorted({f for f in ids if f not in sae.col_of})
    assert not missing, (
        f"this SAE is a {sae.n_cols}-column slice of a {sae.d_sae}-feature dictionary and does "
        f"not hold feature(s) {missing[:8]}; it holds {sorted(sae.col_of)[:8]}"
        f"{'...' if len(sae.col_of) > 8 else ''}. Load the columns you mean -- nothing here will "
        f"reinterpret a feature id as a column number."
    )
    return [sae.col_of[f] for f in ids]


def sae_encode(sae: BatchTopKSAE, h, feature_ids):
    """maem/sae.py:27-31: pre-topk post-ReLU activations relu((x - b_dec) @ W_enc[:,f] + b_enc[f]).

    `feature_ids` are always the DICTIONARY's ids, whether `sae` is the whole dictionary or a
    column slice from `load_sae_columns`. The slice translates them through its own `col_of` and
    refuses an id it does not hold, so a caller cannot get a different feature's activation by
    handing a local index to one object and a global id to the other -- which is the only way this
    optimisation could have gone wrong silently.
    """
    import torch

    ids = _sae_cols(sae, feature_ids)
    idx = torch.as_tensor(ids, device=sae.W_enc.device)
    return torch.relu((h - sae.b_dec) @ sae.W_enc[:, idx] + sae.b_enc[idx])


def load_sae_columns(path: str, d_model: int, feature_ids, device: str = "cpu", dtype=None):
    """The SAE restricted to `feature_ids`: everything `sae_encode` reads, nothing else.

    `relu((h - b_dec) @ W_enc[:, f] + b_enc[f])` needs one column of W_enc per feature, one entry
    of b_enc, all of b_dec, and the gate. A caller that wants the activation of SIXTEEN features
    of a 2^21 dictionary does not need the other 2,097,136 columns and certainly does not need
    W_dec -- which is 43 GB in fp32 at that width, and is what made `gcg --mode epo --sae
    qwen36-27b/dict2m` OOM an H200 at setup (MEASURED 2026-09-21: the fp32 unembedding's 4.74 GiB
    could not be allocated with 135.55 GiB already in use).

    The full encoder is read on the CPU and only the slice is moved, so the device never holds the
    dictionary. The returned object is an ordinary BatchTopKSAE carrying `col_of`, so `sae_encode`
    still takes dictionary ids and `d_sae` still reports the dictionary's true width.
    """
    import torch

    ids = [int(f) for f in feature_ids]
    assert ids, "load_sae_columns needs at least one feature id"
    dup = sorted({f for f in ids if ids.count(f) > 1})
    assert not dup, f"duplicate feature ids {dup[:8]} -- the column map would be ambiguous"
    full = load_sae(path, d_model, device="cpu", dtype=dtype, need_decoder=False)
    bad = sorted({f for f in ids if not 0 <= f < full.d_sae})
    assert not bad, f"feature ids {bad[:8]} are outside the {full.d_sae}-feature dictionary {path}"
    idx = torch.as_tensor(ids)
    return BatchTopKSAE(
        full.W_enc[:, idx].contiguous().to(device),
        None,
        full.b_enc[idx].contiguous().to(device),
        full.b_dec.to(device),
        full.threshold,
        col_of={f: i for i, f in enumerate(ids)},
        d_sae_full=full.d_sae,
    )


def sae_dirs(sae: BatchTopKSAE, feature_ids):
    """maem/sae.py:33-36: the `sae` family target is the UNIT ENCODER COLUMN unit(W_enc[:, f]).

    Sibling of `sae_encode` and indexes W_enc the same way, so it takes DICTIONARY ids on a column
    slice too. Without this the two functions would disagree about what an id means on the same
    object, which is worse than either convention.
    """
    import torch

    ids = _sae_cols(sae, feature_ids)
    idx = torch.as_tensor(ids, device=sae.W_enc.device)
    return torch.nn.functional.normalize(sae.W_enc[:, idx].T, dim=-1)


# ---------------------------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows) -> int:
    """One json object per line. Returns the number of rows written."""
    n = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_array(path: str | Path, arr, dtype: str):
    """Raw little-endian array, no header. The shape lives in the README and index.json.

    Returns (dtype, shape, bytes). torch tensors are accepted and moved to the cpu first.
    """
    import numpy as np

    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(arr).astype(dtype))
    assert arr.dtype.byteorder in ("=", "|"), f"{path}: expected native byte order, got {arr.dtype}"
    with open(path, "wb") as fh:
        fh.write(arr.tobytes())
    return str(arr.dtype), list(arr.shape), int(arr.nbytes)


def read_array(path: str | Path, dtype: str, shape):
    import numpy as np

    return np.fromfile(path, dtype=dtype).reshape(shape)


def sha256_of_weights(path: str | Path) -> dict:
    """Streamed sha256 over a checkpoint directory's weight files, for the MAEM README.

    Hashes every *.safetensors / *.bin / *.pt in name order, plus a combined digest over the
    per-file digests, so a sibling checkpoint cannot be mistaken for this one (checklist item 74).
    """
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt")
        )
    assert files, f"no weight files (*.safetensors|*.bin|*.pt) under {path}"
    per_file = {}
    combined = hashlib.sha256()
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda fh=fh: fh.read(8 << 20), b""):
                h.update(block)
        per_file[f.name] = {"sha256": h.hexdigest(), "bytes": f.stat().st_size}
        combined.update(f.name.encode())
        combined.update(h.digest())
    return {"path": str(path), "files": per_file, "sha256": combined.hexdigest()}


def sha256_of_index(path: str | Path) -> dict:
    """Cheap weight identity for a SHARDED full model: sha256 of `model.safetensors.index.json`
    plus every shard's name and byte size -- no shard content is read.

    MEASURED constraint: the 27B full MAEM is ~52 GiB across 13 files on the FUSE-mounted volume;
    a streamed sha256 of that is minutes of wall on a $4.54/h GPU for a field nobody diffs. The
    index.json pins the parameter->shard map and every tensor name, and the sizes pin the shard
    bytes, so two different checkpoints of the same architecture differ here only if they happen to
    have byte-identical shard sizes AND an identical index. That is weaker than sha256_of_weights
    and the README says so in those words (checklist item 74).
    """
    path = Path(path)
    idx = path / "model.safetensors.index.json"
    assert idx.exists(), f"no model.safetensors.index.json under {path}: not a sharded checkpoint"
    h = hashlib.sha256()
    with open(idx, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    shards = {
        f.name: f.stat().st_size for f in sorted(path.iterdir()) if f.is_file() and f.suffix == ".safetensors"
    }
    assert shards, f"no *.safetensors shards under {path}"
    combined = hashlib.sha256()
    combined.update(h.digest())
    for name, size in shards.items():
        combined.update(f"{name}:{size}".encode())
    return {
        "path": str(path),
        "kind": "index+sizes (shard CONTENT is not hashed -- see common.sha256_of_index)",
        "index_sha256": h.hexdigest(),
        "shards": shards,
        "bytes": sum(shards.values()),
        "sha256": combined.hexdigest(),
    }


def repo_commit(repo: str | Path | None = None) -> str:
    """`git rev-parse HEAD` of the checkout this file lives in. Local side only: call it at launch
    and pass the string to the remote function, which has no git checkout."""
    repo = Path(repo) if repo else HERE
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def dir_size(path: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024.0
    raise AssertionError("unreachable")


def _read_index(path: Path) -> dict[str, dict]:
    """A product directory's `index.json`, or {} when it is absent or unreadable.

    Never raises: the index is README metadata (no consumer in `evals/faithfulness/` reads it), and a run
    that has just produced a rollout does not fail because a concurrent writer was mid-replace.
    """
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as e:
        if os.path.exists(path):
            print(f"[outdir] could not read {path} ({e}); the README file table will be partial",
                  flush=True)
        return {}
    if not isinstance(rec, dict):
        return {}
    return {k: v for k, v in rec.items() if k not in ("README.md", "index.json")}


INDEX_LOCK = ".index.lock"


@contextlib.contextmanager
def _index_lock(product: Path, timeout: float = 60.0, stale: float = 900.0):
    """Hold `<product>/.index.lock` while index.json and README.md are read-merged-written.

    O_CREAT|O_EXCL, spun on with jitter, a stale lock broken after `stale` seconds, and -- after
    `timeout` -- the merge proceeds UNLOCKED with a warning rather than failing a finished
    rollout: the files themselves are already in place by then and the worst an unlocked merge
    costs is a row of the README's file table (nothing in `evals/faithfulness/` reads index.json).

    The lock is a dotfile inside the product directory and is removed on release, so the
    directory's committed contents are byte-identical to what the pre-2026-09-23 write produced.
    """
    lock = product / INDEX_LOCK
    t0, fd = time.time(), None
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock).st_mtime
            except OSError:
                continue
            if age > stale:
                print(f"[outdir] breaking a stale {lock} ({age:.0f}s old)", flush=True)
                with contextlib.suppress(OSError):
                    os.unlink(lock)
                continue
            if time.time() - t0 > timeout:
                print(f"[outdir] WARNING: {lock} held for {timeout:.0f}s; merging index.json "
                      f"WITHOUT the lock (the product files are already in place)", flush=True)
                break
            time.sleep(0.005 + 0.02 * random.random())
        except OSError as e:  # a filesystem with no O_EXCL: best effort, say so
            print(f"[outdir] WARNING: cannot take {lock} ({e}); merging index.json unlocked",
                  flush=True)
            break
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(lock)


def _replace_atomically(path: Path, text: str) -> None:
    """Write `text` to `path` through a sibling temp + os.replace, so no reader sees it half-written."""
    tmp = path.with_name(f".{path.name}.tmp-{_tmp_stamp()}")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _tmp_stamp() -> str:
    """The unique component of an ADDITIVE product's staging directory.

    It keeps the `.tmp-<date>` prefix that `reconstruction/stats.py:344-359` and
    `gcg/modal_app.py:169` filter on, and appends pid + a random nibble so that two concurrent
    writers into ONE accumulating product directory never share a staging directory. That sharing
    is the whole of the silent loss recorded at `SMOKES.md:4349-4356`.
    """
    return f"{time.strftime('%Y-%m-%d')}-{os.getpid()}-{random.randrange(16**6):06x}"


class OutDir:
    """Temp-and-rename output directory with a README and an array index (infra/design.md §1).

    ONE-SHOT products (the default) write to `<name>.tmp-<date>/` and rename on completion; an
    existing `<name>/` is never overwritten without force=True. On an exception the temp directory
    is LEFT IN PLACE and its path printed, so a failed run is inspectable and never half-renamed.

    ACCUMULATING products (`keep_existing=True`: `rollouts/`, `stats/`, `sae/<name>/`, the scores
    directory) are ADDITIVE since 2026-09-23. The old behaviour copytree'd the whole existing
    directory into `<name>.tmp-<date>/` and, on commit, rmtree'd the original and renamed the temp
    over it; two concurrent runs shared the date-stamped temp name and the later rename silently
    discarded the earlier run's file (`SMOKES.md:4349-4356`). Now:

      * `__enter__` creates `<name>/` if it is missing and stages this run's files in a temp
        directory unique to the process; nothing that already exists is read, copied or removed;
      * `__exit__` MOVES only the files this run wrote into `<name>/`, then merges its index
        entries into `<name>/index.json` and rewrites `README.md`. Both are written through a
        temp-and-`os.replace`, so a concurrent reader never sees a half-written one.

    The on-disk result is byte-identical to what the copytree path produced for a single writer:
    the same files, the same `index.json` mapping and the same README layout. What changes is only
    that a second concurrent writer's file survives.

    The README is the only metadata (command line, date, repo commit, inputs, sizes, status,
    provenance). The one allowed sidecar is `index.json`: file -> {dtype, shape, bytes} for the raw
    arrays, because a raw array's length is otherwise inferred from its file size (checklist
    items 29, 73).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        force: bool = False,
        argv: list[str] | None = None,
        commit: str = "",
        inputs: dict | None = None,
        provenance: dict | None = None,
        status: str = "ok",
        on_commit=None,
        gpu: str = "",
        usd_per_s: float = 0.0,
        t0: float = 0.0,
        keep_existing: bool = False,
    ):
        self.path = Path(path)
        self.force = force
        # keep_existing: an ACCUMULATING directory (rollouts/, which gains one <set>.jsonl per run)
        # rather than a one-shot product. This run's files are staged in a temp dir of its own and
        # MOVED in one at a time on commit; nothing already in the directory is copied or removed,
        # so two concurrent writers of disjoint files both survive. The caller's per-file overwrite
        # rule is still its own (rollouts_vllm asserts the stem is free unless --force).
        self.keep_existing = keep_existing
        self.gpu = gpu
        self.usd_per_s = usd_per_s
        self.argv = argv if argv is not None else list(sys.argv)
        self.repo_commit = commit
        self.inputs = dict(inputs or {})
        self.provenance = dict(provenance or {})
        self.status = status
        self.on_commit = on_commit  # e.g. modal.Volume.commit, called after the rename
        self.index: dict[str, dict] = {}
        # the entries already in the product directory when an ADDITIVE run started: never written
        # by this run, carried only so the README's file table lists the whole directory.
        self.existing: dict[str, dict] = {}
        self.notes: list[str] = []
        self.sections: list[tuple[str, list[str]]] = []
        # A one-shot product keeps the dated name: `gcg --resume-from` and the "re-run is the
        # resume" convention both address `<name>.tmp-<date>` by that exact spelling. An
        # accumulating product gets a per-process name in __enter__ instead.
        self.tmp = self.path.with_name(
            f"{self.path.name}.tmp-"
            + (_tmp_stamp() if keep_existing else time.strftime("%Y-%m-%d"))
        )
        # the product's own start, so `wall` and `cost` cover the whole call (model load included);
        # 0 means "measure from __enter__"
        self._t0 = t0

    def __enter__(self):
        if self.keep_existing:
            # ADDITIVE. Nothing existing is read for correctness, copied or removed -- the only
            # read is index.json, for the README's file table, and a failure to read it costs a
            # table row and no data.
            assert not self.tmp.exists(), f"staging dir {self.tmp} already exists"
            self.tmp.mkdir(parents=True)
            self.path.mkdir(parents=True, exist_ok=True)
            self.existing = _read_index(self.path / "index.json")
            if not self._t0:
                self._t0 = time.time()
            print(
                f"[outdir] ADDITIVE into {self.path} ({len(self.existing)} entries already there); "
                f"staging in {self.tmp}",
                flush=True,
            )
            return self
        if self.path.exists():
            assert self.force, (
                f"{self.path} already exists; refusing to overwrite without --force "
                f"(infra/design.md §1: old products are never rewritten in place)"
            )
            print(f"[outdir] --force: removing the existing {self.path}", flush=True)
            shutil.rmtree(self.path)
        if self.tmp.exists():
            print(f"[outdir] removing a leftover temp dir {self.tmp}", flush=True)
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        if not self._t0:
            self._t0 = time.time()
        return self

    def file(self, name: str) -> Path:
        return self.tmp / name

    def write_jsonl(self, name: str, rows) -> int:
        n = write_jsonl(self.file(name), rows)
        self.index[name] = {"kind": "jsonl", "rows": n, "bytes": self.file(name).stat().st_size}
        return n

    def write_json(self, name: str, obj) -> None:
        with open(self.file(name), "w") as fh:
            json.dump(obj, fh, indent=1)
        self.index[name] = {"kind": "json", "bytes": self.file(name).stat().st_size}

    def write_array(self, name: str, arr, dtype: str) -> None:
        dt, shape, nbytes = write_array(self.file(name), arr, dtype)
        self.index[name] = {"kind": "array", "dtype": dt, "shape": shape, "bytes": nbytes}

    def note(self, line: str) -> None:
        """A line that goes into the README body. Use it for anything a reader must know."""
        self.notes.append(line)

    def section(self, title: str, lines: list[str]) -> None:
        """A `## <title>` section of the README, before Notes. For a recipe a reader must be able
        to find by heading rather than by scanning bullets (e.g. the realact draw in `targets`)."""
        self.sections.append((title, list(lines)))

    def wall(self) -> float:
        return time.time() - self._t0

    def cost_usd(self) -> float:
        """This product's own share of the container cost. Checklist item 84: `modal app logs`
        replays stale output under a timeout, so the README's cost field is the one to trust."""
        return self.wall() * self.usd_per_s

    def _readme(self) -> str:
        lines = [f"# {self.path.name}", ""]
        lines += [
            f"- date: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
            f"- command: `{' '.join(self.argv)}`",
            f"- repo commit: {self.repo_commit or 'UNKNOWN (not passed in)'}",
            f"- wall: {self.wall():.1f}s (whole product call)",
            f"- gpu: {self.gpu or 'n/a'}",
            f"- cost: ${self.cost_usd():.4f} (product wall x {self.usd_per_s * 3600:.2f} $/h)",
            f"- status: {self.status}",
            "",
        ]
        if self.inputs:
            lines += ["## Inputs", ""]
            lines += [f"- {k}: {v}" for k, v in self.inputs.items()] + [""]
        if self.provenance:
            lines += ["## Provenance", ""]
            lines += [f"- {k}: {v}" for k, v in self.provenance.items()] + [""]
        for title, body in self.sections:
            lines += [f"## {title}", ""] + list(body) + [""]
        merged = self._merged_index()
        if merged:
            lines += ["## Files", "", "| file | kind | dtype | shape | size |", "|---|---|---|---|---|"]
            for name, meta in merged.items():
                lines.append(
                    f"| `{name}` | {meta['kind']} | {meta.get('dtype', '')} | "
                    f"{meta.get('shape', meta.get('rows', ''))} | {human(meta['bytes'])} |"
                )
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def _merged_index(self) -> dict[str, dict]:
        """What the whole product directory holds: what was there, plus what this run wrote."""
        return {**self.existing, **self.index}

    def _commit_additive(self) -> None:
        """Move this run's files into the product directory, then merge index.json and README.md.

        Per file, `os.replace` within one filesystem: atomic, and a concurrent writer of a
        DIFFERENT file is untouched. `index.json` is a read-merge-write and so is racy in the
        window between the read and the replace; it is re-read immediately before the write to
        keep that window at a few milliseconds, and it carries no data -- every consumer in
        `evals/faithfulness/` reads the product's files directly and none reads index.json (grep says
        so), so a lost entry costs a README row, never a rollout.
        """
        staged = sorted(self.tmp.iterdir(), key=lambda p: p.name)
        wrote = [p.name for p in staged]
        for p in staged:
            if p.is_dir():
                # A subdirectory is not a product file; move it whole and refuse to merge into an
                # existing one rather than half-overwriting somebody else's subtree.
                assert not (self.path / p.name).exists(), (
                    f"{self.path / p.name} already exists; an additive product never merges into "
                    f"an existing subdirectory"
                )
                shutil.move(str(p), str(self.path / p.name))
            else:
                os.replace(p, self.path / p.name)
        idx = self.path / "index.json"
        # The index merge is a read-modify-write and is the ONE part of the commit two writers
        # share, so it is the one part that takes a lock. Everything above this line is already
        # safe: each writer moved only its own files.
        with _index_lock(self.path):
            self.existing = {**_read_index(idx), **self.existing}
            merged = self._merged_index()
            for stale in ("README.md", "index.json"):
                merged.pop(stale, None)
            _replace_atomically(idx, json.dumps(merged, indent=1))
            self.index["index.json"] = {"kind": "json", "bytes": idx.stat().st_size}
            _replace_atomically(self.path / "README.md", self._readme())
        self.tmp.rmdir()
        print(
            f"[outdir] ADDITIVE: moved {len(wrote)} file(s) into {self.path} "
            f"({', '.join(wrote)}); the directory now holds {len(merged)} "
            f"wall={self.wall():.1f}s cost=${self.cost_usd():.4f}",
            flush=True,
        )

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"[outdir] FAILED: {exc_type.__name__}; temp dir kept at {self.tmp}", flush=True)
            return False  # never swallow
        if self.keep_existing:
            self._commit_additive()
            if self.on_commit is not None:
                self.on_commit()
            return False
        self.write_json("index.json", self.index)
        with open(self.tmp / "README.md", "w") as fh:
            fh.write(self._readme())
        self.tmp.rename(self.path)
        print(
            f"[outdir] wrote {self.path} ({human(dir_size(self.path))}) "
            f"wall={self.wall():.1f}s cost=${self.cost_usd():.4f}",
            flush=True,
        )
        if self.on_commit is not None:
            self.on_commit()
        return False


def outdir(path: str | Path, args: dict, **kw) -> OutDir:
    """OutDir wired from the dispatch args, so every product records the same provenance fields.

    `args` is the dict modal_app.main built (argv, repo_commit, force) plus the fields
    modal_app._run injected container-side (gpu, usd_per_s, on_commit = the volume commit).
    """
    return OutDir(
        path,
        force=bool(args.get("force")),
        argv=args.get("argv") or list(sys.argv),
        commit=args.get("repo_commit", ""),
        gpu=args.get("gpu", ""),
        usd_per_s=float(args.get("usd_per_s", 0.0)),
        on_commit=args.get("on_commit"),
        t0=float(args.get("t0", 0.0)),
        **kw,
    )


def hard_exit(code: int = 0):
    """os._exit after flushing: an HF `datasets` streaming worker can abort the interpreter at
    finalisation and turn a finished run into a failed one (checklist item 59). Call it only AFTER
    the volume commit."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---------------------------------------------------------------------------------------------
# The OOD generalisation evaluation (design infra/2026-09-18_ood-eval-design.md)
#
# Everything here is CPU and tokenizer-only. It is shared by `corpus --arm`, `targets` on an `ood`
# held-out set, `nll`, and (through an import of this module) reconstruction/stats_ood.py, so the
# arm table, the script table and the code-like rule have ONE definition each.
# ---------------------------------------------------------------------------------------------


def ood_arms(cfg: dict) -> dict:
    """The `ood_arms:` table, keyed by arm id, in config order. Empty when the key is absent."""
    return dict(cfg.get("ood_arms") or {})


def ood_arm(cfg: dict, arm: str) -> dict:
    arms = ood_arms(cfg)
    assert arm in arms, f"unknown ood arm {arm!r}; config.yaml has {sorted(arms)}"
    spec = arms[arm]
    for field in ("family", "reader", "dataset", "files", "text", "sizes", "script", "unspaced"):
        assert field in spec, f"ood arm {arm!r} is missing {field!r}"
    return spec


def is_ood_set(cfg: dict, set_name: str) -> bool:
    return (cfg["heldout"].get(set_name) or {}).get("kind") == "ood"


def ood_set_arms(cfg: dict, set_name: str) -> list[str]:
    """The arms an `ood` held-out set draws, in config order."""
    spec = cfg["heldout"][set_name]
    assert spec.get("kind") == "ood", f"held-out set {set_name!r} is not an ood set"
    want = spec.get("arms", "all")
    if want == "all":
        return list(ood_arms(cfg))
    assert isinstance(want, list) and want, f"heldout {set_name!r}: `arms` must be `all` or a list"
    for a in want:
        ood_arm(cfg, a)
    return list(want)


def arm_perm(arm: str, n_rows: int, seed: int):
    """The arm's ONE row permutation: `default_rng(seed ^ crc32(arm)).permutation(n_rows)`.

    Design §2. `corpus --arm` consumes it from the front until its token budget is met and records
    how far it got in `stream.json`; `targets` regenerates the identical permutation and continues
    from that position, which is what makes corpus and target documents disjoint BY CONSTRUCTION
    rather than by a check. crc32 (not python's `hash`) because `hash` of a str is salted per
    process and would not reproduce.
    """
    import numpy as np

    return np.random.default_rng(int(seed) ^ zlib.crc32(arm.encode("utf-8"))).permutation(int(n_rows))


# --- the byte-level BPE pieces behind every tokenisation covariate ----------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable-unicode table, written out rather than imported.

    `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode` is the same table; it is copied
    here because a covariate that decides the paper's `byte_piece` rate must not move when a
    transformers internal does. `check_token_bytes` verifies the mapping against the real
    tokenizer before any arm is drawn, so a base whose tokenizer is NOT byte-level GPT-2 style
    fails loudly instead of producing a plausible-looking wrong rate.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}
WS_BYTES = frozenset(b" \t\n\r\x0b\x0c")


def token_bytes(tok, ids) -> list[bytes | None]:
    """The raw bytes of each token id, or None for a piece that is not byte-level (a special token).

    `convert_ids_to_tokens` returns the byte-level strings ('Ġthe', 'ä¸Ń'); BYTE_DECODER maps them
    back to the bytes the text was made of, which is the only representation in which "this token
    is half a character" and "this token starts a whitespace unit" are well defined. `tok.decode`
    cannot answer either question: it replaces a partial character with U+FFFD and loses the bytes.
    """
    pieces = tok.convert_ids_to_tokens([int(i) for i in ids])
    out: list[bytes | None] = []
    for pc in pieces:
        try:
            out.append(bytes(BYTE_DECODER[c] for c in pc))
        except KeyError:
            out.append(None)  # a special token (<|endoftext|> and friends): not byte-level
    return out


def check_token_bytes(tok, ids) -> str:
    """Assert that `token_bytes` reconstructs exactly what the tokenizer decodes. Returns a note.

    Run once per arm in `targets` and in the unit smoke. A mismatch means the byte table above is
    not this tokenizer's, and every `byte_piece` / `tok_class` number would be quietly wrong.
    """
    pb = token_bytes(tok, ids)
    n_special = sum(1 for b in pb if b is None)
    joined = b"".join(b for b in pb if b is not None)
    ours = joined.decode("utf-8", errors="replace")
    theirs = tok.decode([int(i) for i in ids])
    assert n_special == 0 and ours == theirs, (
        f"token_bytes does not reconstruct the tokenizer's own decode: {n_special} non-byte-level "
        f"piece(s), and the two strings differ at "
        f"{next((k for k in range(min(len(ours), len(theirs))) if ours[k] != theirs[k]), len(ours))} "
        f"(lengths {len(ours)} vs {len(theirs)}). The byte-level BPE assumption behind every "
        f"tokenisation covariate does not hold for this tokenizer."
    )
    return f"token_bytes verified against tok.decode on {len(pb)} tokens ({len(joined)} bytes)"


def _char_boundaries(pb: list[bytes | None]):
    """(boundary, first_char) per token, from one incremental UTF-8 decode of the byte stream.

    boundary[i]   the byte prefix through token i ends on a character boundary (nothing pending)
    first_char[i] the first character COMPLETED inside token i, or None when token i completes none
    """
    import codecs

    dec = codecs.getincrementaldecoder("utf-8")("replace")
    boundary, first_char = [], []
    for b in pb:
        s = dec.decode(b if b is not None else b"")
        boundary.append(dec.getstate()[0] == b"")
        first_char.append(s[0] if s else None)
    return boundary, first_char


# Unicode script ranges, one entry per script an arm can be in (config `script:`). Coarse on
# purpose: the covariate asks "is this character in the arm's script", not "which of 160 scripts".
SCRIPT_RANGES = {
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    "Cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "Greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Thai": ((0x0E00, 0x0E7F),),
    "Han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
    "Kana": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    "Hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)),
}
# A composite arm script: Japanese text is Han + both kana, and a Han-only test would call every
# hiragana particle "other script".
SCRIPT_ALIASES = {"Jpan": ("Han", "Kana")}


def in_script(ch: str, script: str) -> bool:
    """Is `ch` a character of `script` (config `script:` of an arm)? Unknown scripts raise."""
    parts = SCRIPT_ALIASES.get(script, (script,))
    o = ord(ch)
    for name in parts:
        assert name in SCRIPT_RANGES, f"unknown script {name!r}; known: {sorted(SCRIPT_RANGES)}"
        if any(lo <= o <= hi for lo, hi in SCRIPT_RANGES[name]):
            return True
    return False


def is_letter(ch: str) -> bool:
    """A letter for the script covariates: a Unicode letter OR a combining mark.

    `str.isalpha()` is False for Mn/Mc, which would make every Thai vowel sign, every Devanagari
    matra and every Arabic diacritic `punct` -- exactly the characters an abugida arm is made of.
    """
    import unicodedata

    return unicodedata.category(ch) in ("Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me")


def script_fraction(text: str, script: str) -> float:
    """Fraction of the LETTERS of `text` that are in `script` (design §5, H's CPU classifier).

    Non-letters (digits, punctuation, whitespace, symbols) are not counted on either side, so a
    rollout of pure LaTeX has an undefined -- returned as nan -- script fraction rather than 0.
    """
    letters = [c for c in text if is_letter(c)]
    if not letters:
        return float("nan")
    return sum(1 for c in letters if in_script(c, script)) / len(letters)


# The one code-like rule, used twice (review R3 on rollouts, R7 on the training corpus). Written
# out here and PRINTED by both callers' READMEs, so the paper can state it in a sentence.
CODE_LIKE_MARKERS = ("def ", "{", "};", "import ", "#include", "</", "=>", "->", "();")
CODE_LIKE_MIN = 3  # distinct markers (indentation runs count as one) per 512-token window


def code_like(text: str) -> bool:
    """>= CODE_LIKE_MIN of {the markers above} + `an indentation run` occur in `text`."""
    import re

    hits = sum(1 for m in CODE_LIKE_MARKERS if m in text)
    if re.search(r"\n[ \t]{2,}\S", text):
        hits += 1
    return hits >= CODE_LIKE_MIN


UNIT_CAP = 16  # `n_subtokens` is capped here (design §3)
UNITEND_MAX_SHIFT = 8  # how far `_unitend` may move p forward (design §3)


def token_covariates(tok, ids, p: int, script: str, unspaced: bool) -> dict:
    """The design §3 + §11 R5 covariates of the token at position `p` of a token window.

    All of it from the tokenizer alone, at draw time, on the CPU:

      tok_class    `unspaced` on an unspaced arm; otherwise `word` (the token both starts and ends
                   a whitespace-delimited unit), `first`, `mid` or `last`
      n_subtokens  the length of that unit, capped at UNIT_CAP; None on an unspaced arm
      unit_start / unit_end   the unit's token indices (None on an unspaced arm)
      byte_piece   the token's bytes are not valid UTF-8 on their own -- a partial character under
                   Qwen's byte-level BPE (R5)
      whole_char   the token is exactly one character; multi_char: two or more (R5)
      char_type    of the character the token's first byte belongs to: `letter_arm` (a letter of
                   the arm's script), `letter_other`, `digit`, `punct`, `space`
      unitend_p    where the `_unitend` variant would move p: the unit's last token on a spaced
                   arm (wordend), the end of the character on an unspaced one (charend)
      unitend_rule `wordend` | `charend` | `none` (p already ends its unit / character)

    A whitespace-delimited unit is a maximal run of tokens with no whitespace byte between them:
    token i starts a unit iff its first byte is whitespace, or the previous token's bytes carry
    whitespace after their first byte, or i is the first token of the window.
    """
    n = len(ids)
    assert 0 <= p < n, f"p={p} outside the {n}-token window"
    pb = token_bytes(tok, ids)
    boundary, first_char = _char_boundaries(pb)

    def bts(i) -> bytes:
        return pb[i] if pb[i] is not None else b""

    def lead_ws(i) -> bool:
        b = bts(i)
        return bool(b) and b[0] in WS_BYTES

    def inner_ws(i) -> bool:
        return any(c in WS_BYTES for c in bts(i)[1:])

    def starts(i) -> bool:
        return i == 0 or lead_ws(i) or inner_ws(i - 1)

    own = bts(p)
    try:
        dec_own = own.decode("utf-8")
        is_byte_piece = False
    except UnicodeDecodeError:
        dec_own = ""
        is_byte_piece = True

    # the character this token's first byte belongs to = the first character completed at or after p
    ch = next((first_char[j] for j in range(p, n) if first_char[j] is not None), None)

    def _type(c) -> str:
        if c is None:
            return "partial"
        if c.isspace():
            return "space"
        if c.isdigit():
            return "digit"
        if is_letter(c):
            return "letter_arm" if in_script(c, script) else "letter_other"
        return "punct"

    char_type = _type(ch)
    # `char_type` is the design's: the type of the token's FIRST character. Under a byte-level BPE
    # a spaced script's tokens carry their leading space, so that is `space` for about half of them
    # (MEASURED 2026-09-18: 11 of 21 tokens of a Czech sentence, 13 of 25 of a Python snippet) and
    # the stratum says little about the token's content. `char_type_body` is the same rule applied
    # to the token's first NON-space character, which is the one a reader means; both are stored
    # and the design's field keeps its name and its definition.
    body = own.decode("utf-8", errors="ignore").lstrip()
    char_type_body = _type(body[0]) if body else char_type

    out = {
        "byte_piece": is_byte_piece,
        "whole_char": (not is_byte_piece) and len(dec_own) == 1,
        "multi_char": (not is_byte_piece) and len(dec_own) >= 2,
        "char_type": char_type,
        "char_type_body": char_type_body,
    }
    if unspaced:
        out.update(tok_class="unspaced", n_subtokens=None, unit_start=None, unit_end=None)
        j = next((k for k in range(p, n) if boundary[k]), n - 1)
        out["unitend_p"] = min(j, p + UNITEND_MAX_SHIFT, n - 1)
        out["unitend_rule"] = "charend" if out["unitend_p"] > p else "none"
        return out

    s = p
    while s > 0 and not starts(s):
        s -= 1
    e = p
    while e + 1 < n and not starts(e + 1):
        e += 1
    at_start, at_end = s == p, e == p
    out["tok_class"] = (
        "word" if at_start and at_end else "first" if at_start else "last" if at_end else "mid"
    )
    out["n_subtokens"] = min(e - s + 1, UNIT_CAP)
    out["unit_start"], out["unit_end"] = s, e
    out["unitend_p"] = min(e, p + UNITEND_MAX_SHIFT, n - 1)
    out["unitend_rule"] = "wordend" if out["unitend_p"] > p else "none"
    return out


def arm_rng(arm: str, seed: int, stream: int = 1):
    """A per-arm rng INDEPENDENT of `arm_perm`'s (which is `stream` 0 in all but name).

    `default_rng` takes a sequence of ints as entropy, so (seed, crc32(arm), stream) gives each arm
    its own reproducible stream for the p / L / norm-presample draws without disturbing the row
    permutation the corpus and the pool were cut from.
    """
    import numpy as np

    return np.random.default_rng([int(seed), zlib.crc32(arm.encode("utf-8")), int(stream)])
