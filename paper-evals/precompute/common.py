"""The only module shared across paper-evals: config, model loading, injection, reading, scoring, io.

Everything that has to match Celeste's pipeline lives here once, so rollouts, GCG and the
reconstruction scripts cannot drift on the objective. Each convention carries the file:line in her
repo it was copied from (verified against master 09d4a01 on 2026-09-15); where we deliberately
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
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# constants that are part of the protocol
# ---------------------------------------------------------------------------------------------

VOL = "/vol"
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "config.yaml"

# eval/eval_universal.py:138 -- the re-encode truncation. Checklist item 7: it must leave room for
# the whole rollout plus the prepended sink, i.e. max_length >= rollouts.max_new + 1; asserted in
# load_config() so a later max_new bump cannot silently truncate the scored window.
SCORE_MAX_LENGTH = 95
# eval/eval_universal.py:129 sbatch. ONE fixed scoring chunk for every product (checklist item 11:
# right-padding chunk size measurably shifts per-row cosine), stated in every scores README.
SCORE_CHUNK = 32
# Column 0 of every scored array is the BOS sink, never a candidate token; width is fixed so that
# chunks of different token lengths concatenate into one rectangular array.
SCORE_WIDTH = SCORE_MAX_LENGTH + 1

MARKER = " ?"  # mxf/prompts.py:4

# Input-amplitude conventions of the `nla` verbalizer (`precompute/rollouts_nla.py`, which aliases
# this tuple and documents what each one does). It lives here because load_config validates
# `nla.amp` and must not import a product module to do it.
AMP_MODES = ("exact", "mu", "raw")

# --- the centring vocabulary (2026-09-21, branch `evals/conventions`) --------------------------
# One word per concept, used by config.yaml, every `--centering` flag and every product README.
# `none` and `unknown` are RESERVED and can never be the name of a mean in `mus:`.
MU_SOURCES = ("product", "set", "archive", "unknown")
FAMILY_KINDS = ("activation", "synthetic", "dictionary", "subspace")
# The closed key set of a `maemms.<key>.input:` block, shaped like `nla:` above: one key today, and
# a typo in it steers an H200 run, so "absent" must have no safe meaning.
INPUT_KEYS = ("centering",)
STORAGE_KINDS = ("raw", "unit", "dirs_only")
# "subtract nothing" -- a legal value of `input.centering`, `--centering` and `family_mu`.
NO_CENTRING = "none"
# "this arrived centred on a mean nobody here can name" -- legal in `family_mu` / `mu_stored` only.
MU_UNKNOWN = "unknown"
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
SCAN_BLOCK = 64
SCAN_STRIDE = 16

# mxf/prompts.py:8-15, verbatim except the read layer, which the template names in words and which
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
        "mus",
        "family_kinds",
        "saes",
        "maemms",
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
        assert base in cfg["mus"], (
            f"base {base!r} has no `mus:` block; every base must declare its named centring means "
            f"(at least `stats_mu`), because `input.centering` and `--centering` name one"
        )

    for base, block in cfg["mus"].items():
        assert base in cfg["bases"], f"mus names base {base!r}, which is not in config bases"
        assert isinstance(block, dict) and block, f"mus[{base!r}] must be a non-empty name -> spec map"
        for name, mspec in block.items():
            assert name != "none", (
                f"mus[{base!r}] declares a mean called 'none'; that word is RESERVED for \"subtract "
                f"nothing\" in `input.centering` / `--centering` and cannot also be a file"
            )
            assert isinstance(mspec, dict) and mspec.get("source") in MU_SOURCES, (
                f"mus[{base!r}][{name!r}]: `source` must be one of {list(MU_SOURCES)}, got {mspec!r}"
            )
            has_path = "path" in mspec
            assert has_path == (mspec["source"] != "unknown"), (
                f"mus[{base!r}][{name!r}]: source {mspec['source']!r} "
                + ("must NOT carry a `path` (that is what 'unknown' means)" if not has_path
                   else "needs a `path`")
            )
            extra = sorted(set(mspec) - {"source", "path"})
            assert not extra, f"mus[{base!r}][{name!r}]: unexpected keys {extra}"

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

    for key, spec in cfg["maemms"].items():
        base, _ = split_key(key, "maemm")
        assert base in cfg["bases"], f"maemm {key!r} names base {base!r}, which is not in config bases"
        # `base` is the UNTRAINED-BASE CONTROL: no MAEMM weights at all, the clean base run
        # through the identical prompt / marker / injection / sampling path so the tables have a
        # "what does the untrained model reach" row (2026-09-16_base-control).
        # `nla` is the activation-verbalizer BASELINE (EasyNLA): served exactly like a `full`
        # model, injected at the same block-1 output with the same norm-matched add, but with its
        # own prompt, marker and output format -- so only `rollouts_nla` generates for it.
        assert spec.get("type") in ("lora", "full", "base", "nla"), (
            f"maemm {key!r}: type must be 'lora', 'full', 'base' or 'nla', got {spec.get('type')!r}"
        )
        if spec.get("type") == "base":
            assert spec.get("hf") == cfg["bases"][base]["hf"], (
                f"maemm {key!r}: type 'base' is the untrained-base control, so its `hf` must be "
                f"the base's own repo {cfg['bases'][base]['hf']!r}, got {spec.get('hf')!r} -- "
                f"anything else is a TRAINED checkpoint wearing the control's label"
            )
        assert spec.get("role") in (None, "control"), (
            f"maemm {key!r}: `role`, when given, must be 'control' (reconstruction/stats.py shows "
            f"it between the primary and the secondaries), got {spec.get('role')!r}"
        )
        assert ("hf" in spec) != ("src" in spec), (
            f"maemm {key!r}: give exactly one of hf (repo id) / src (volume path), got {sorted(spec)}"
        )
        if spec.get("type") == "nla":
            # The NLA verbalizer builds its OWN prompt from the checkpoint's sidecar (its marker
            # is not our MARKER and is not the last prompt token), so the `prompt in PROMPTS`
            # assert below does not apply to it and `nla:` is validated instead.
            _check_nla(key, spec, max_new)
        else:
            assert spec.get("prompt") in PROMPTS, (
                f"maemm {key!r}: prompt {spec.get('prompt')!r} is not one of {sorted(PROMPTS)}"
            )
        inject = spec.get("inject", {})
        assert "layer" in inject and "coef" in inject, (
            f"maemm {key!r}: inject needs both 'layer' and 'coef', got {inject}"
        )
        assert isinstance(spec.get("compute", True), bool), (
            f"maemm {key!r}: `compute` must be a bool (default true), got {spec.get('compute')!r}"
        )
        if "input" in spec:
            _check_input(cfg, key, base, spec["input"])

    for set_name, spec in cfg["heldout"].items():
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


# Exactly the keys a `type: nla` entry's `nla:` block carries, and the keys of its `sampling:`
# sub-block. Both are closed sets: a typo (`max_nev: 96`) in a block whose every field steers an
# H200 run would otherwise be read as "the field is absent", and every field here is required, so
# "absent" has no safe meaning.
NLA_KEYS = (
    "marker",
    "marker_id",
    "left_id",
    "right_id",
    "template",
    "sampling",
    "max_new",
    "score_max_tokens",
    "card_max_new",
    "n",
    "amp",
    "amp_r",
)
NLA_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_new")


def _check_nla(key: str, spec: dict, rollouts_max_new: int) -> None:
    """Validate one `type: nla` maemms entry. Called from load_config, never at use site.

    Everything here is a fact about the CHECKPOINT (ceselder/qwen3.6-27b-nla-av's nla_meta.yaml
    and generation_config.json) rather than a choice of ours, except `max_new`, `n`, `amp` and
    `amp_r`. `rollouts_nla.check_sidecar` asserts the checkpoint's shipped files still agree with
    the values below before it generates anything; this function only checks the config's shape,
    which is what the CPU `check` gate can do without the weights.
    """
    assert "hf" in spec, f"maemm {key!r}: a `type: nla` entry is fetched from HF, so it needs `hf`"
    rev = spec.get("revision")
    assert isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (
        f"maemm {key!r}: `revision` must be the 40-hex HF commit sha the snapshot directory is "
        f"named after (rollouts_nla asserts the resolved path against it), got {rev!r}"
    )
    assert "prompt" not in spec, (
        f"maemm {key!r}: a `type: nla` entry must NOT name a `prompt` -- the verbalizer builds its "
        f"own from `nla.template` and the marker in `nla.marker`, and common.PROMPTS' marker is "
        f"neither that character nor at that position"
    )
    nla = spec.get("nla")
    assert isinstance(nla, dict), f"maemm {key!r}: a `type: nla` entry needs an `nla:` block, got {nla!r}"
    missing, extra = sorted(set(NLA_KEYS) - set(nla)), sorted(set(nla) - set(NLA_KEYS))
    assert not missing and not extra, (
        f"maemm {key!r}: `nla:` must carry exactly {list(NLA_KEYS)} -- missing {missing}, unexpected {extra}"
    )
    for field in ("marker", "template", "amp"):
        assert isinstance(nla[field], str) and nla[field], (
            f"maemm {key!r}: nla.{field} must be a non-empty string, got {nla[field]!r}"
        )
    for field in ("marker_id", "left_id", "right_id", "max_new", "score_max_tokens", "card_max_new", "n"):
        assert isinstance(nla[field], int) and not isinstance(nla[field], bool) and nla[field] > 0, (
            f"maemm {key!r}: nla.{field} must be a positive int, got {nla[field]!r}"
        )
    assert "{injection_char}" in nla["template"], (
        f"maemm {key!r}: nla.template must carry the sidecar's `{{injection_char}}` placeholder -- "
        f"that is where the marker token, and so the injected direction, goes"
    )
    samp = nla["sampling"]
    assert isinstance(samp, dict) and sorted(samp) == sorted(NLA_SAMPLING_KEYS), (
        f"maemm {key!r}: nla.sampling must carry exactly {list(NLA_SAMPLING_KEYS)}, got {sorted(samp)}"
    )
    assert float(samp["temperature"]) > 0 and 0 < float(samp["top_p"]) <= 1, (
        f"maemm {key!r}: nla.sampling temperature must be > 0 and top_p in (0, 1], got {samp}"
    )
    assert int(samp["top_k"]) >= 0 and int(samp["min_new"]) >= 0, (
        f"maemm {key!r}: nla.sampling top_k and min_new must be >= 0, got {samp}"
    )
    assert nla["amp"] in AMP_MODES, f"maemm {key!r}: nla.amp {nla['amp']!r} is not one of {list(AMP_MODES)}"
    amp_r = nla["amp_r"]
    numeric_r = isinstance(amp_r, int | float) and not isinstance(amp_r, bool) and amp_r > 0
    assert amp_r == "median" or numeric_r, (
        f"maemm {key!r}: nla.amp_r must be 'median' (layer read_layer's q[0.5] of "
        f"stats/resid_norm_quantiles.json) or a positive number, got {amp_r!r}"
    )
    # The scorer's window is the binding constraint, exactly as SCORE_MAX_LENGTH is for
    # `rollouts.max_new` -- but this arm brings its OWN window. `nla.score_max_tokens` is the
    # re-encode truncation `score` uses for it (carried there on the rollouts summary), so the
    # bound is against that rather than against the protocol's 95: a generation longer than the
    # window it will be scored in has a tail nothing ever reads.
    assert nla["max_new"] <= nla["score_max_tokens"] - 1, (
        f"maemm {key!r}: nla.max_new {nla['max_new']} exceeds nla.score_max_tokens "
        f"{nla['score_max_tokens']} - 1 -- the scoring window must leave room for the whole "
        f"generation plus the sink at column 0, or the tail of a full-length rollout is never "
        f"scored (the same rule SCORE_MAX_LENGTH={SCORE_MAX_LENGTH} enforces on rollouts.max_new "
        f"= {rollouts_max_new} for every other arm)"
    )
    # Never NARROWER than the protocol: this key exists to widen the window for a model whose
    # native output is long, not to cut an arm's text short and call it a protocol.
    assert nla["score_max_tokens"] >= SCORE_MAX_LENGTH, (
        f"maemm {key!r}: nla.score_max_tokens {nla['score_max_tokens']} is below the pipeline's "
        f"SCORE_MAX_LENGTH={SCORE_MAX_LENGTH}; this key may only WIDEN the scoring window"
    )
    assert nla["card_max_new"] >= nla["max_new"], (
        f"maemm {key!r}: nla.card_max_new {nla['card_max_new']} is the budget the model card's "
        f"reference script uses and must be >= the nla.max_new {nla['max_new']} we generate at"
    )


def _check_input(cfg: dict, key: str, base: str, block) -> None:
    """Validate one `maemms.<key>.input:` block. Called from load_config, never at use site.

    `input.centering` is what the CHECKPOINT was trained to receive, so it is a fact about the
    training chain and not a knob: a product may override it with `--centering`, and then it
    records the override as a deviation, but it may never infer one.
    """
    assert isinstance(block, dict), f"maemm {key!r}: `input:` must be a mapping, got {block!r}"
    missing, extra = sorted(set(INPUT_KEYS) - set(block)), sorted(set(block) - set(INPUT_KEYS))
    assert not missing and not extra, (
        f"maemm {key!r}: `input:` must carry exactly {list(INPUT_KEYS)} -- missing {missing}, "
        f"unexpected {extra}"
    )
    cen = block["centering"]
    known = sorted(cfg["mus"].get(base, {}))
    assert isinstance(cen, str) and (cen == NO_CENTRING or cen in cfg["mus"].get(base, {})), (
        f"maemm {key!r}: input.centering {cen!r} is neither {NO_CENTRING!r} nor one of base "
        f"{base}'s declared means {known}"
    )


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
    assert "mu_stored" in spec, (
        f"heldout {set_name!r}: `mu_stored` must be present (null when the set is not centred as a "
        f"whole); an absent key and an explicit null are the same thing to yaml and must not be"
    )
    all_mus = {name for block in cfg["mus"].values() for name in block}

    def _ok(name, where):
        assert name is None or name in (NO_CENTRING, MU_UNKNOWN) or name in all_mus, (
            f"heldout {set_name!r}: {where} names {name!r}, which is not null, {NO_CENTRING!r}, "
            f"{MU_UNKNOWN!r} or a declared mean ({sorted(all_mus)})"
        )

    _ok(spec["mu_stored"], "mu_stored")
    fam_mu = spec.get("family_mu") or {}
    assert isinstance(fam_mu, dict), f"heldout {set_name!r}: family_mu must be a mapping"
    assert not fam_mu or storage == "unit", (
        f"heldout {set_name!r}: family_mu only means something on a `storage: unit` set (this one "
        f"is {storage!r}); a raw set derives every direction and a dirs_only set never centred one"
    )
    for fam, name in fam_mu.items():
        assert fam in spec["families"], (
            f"heldout {set_name!r}: family_mu names family {fam!r}, which the set does not draw "
            f"({sorted(spec['families'])})"
        )
        _ok(name, f"family_mu[{fam!r}]")
    if storage == "unit":
        assert spec["mu_stored"] is not None or set(fam_mu) >= set(spec["families"]), (
            f"heldout {set_name!r}: a `storage: unit` set stores directions under SOME mean, so it "
            f"must say which -- either one `mu_stored` for the set or a `family_mu` entry for "
            f"every family (missing {sorted(set(spec['families']) - set(fam_mu))})"
        )


def mu_of_family(cfg: dict, set_name: str, family: str):
    """The mean `family`'s rows of the CONFIGURED set `set_name` were stored under.

    `family_mu` first, then the set-wide `mu_stored`, then -- for a family that cannot be centred
    at all -- `none`. Returns a `mus:` name, `none` or `unknown`.
    """
    spec = cfg["heldout"][set_name]
    fam_mu = spec.get("family_mu") or {}
    if family in fam_mu:
        return fam_mu[family]
    if spec.get("mu_stored") is not None:
        return spec["mu_stored"]
    return NO_CENTRING


def family_centrable(cfg: dict, family: str) -> bool:
    """Can a mean be subtracted from this family's rows at all? `family_kinds:` decides.

    An encoder column, a Gaussian draw and a subspace basis have no mean of their own, so a
    `cos(h - mu, v)` against one is a ONE-SIDED number: the activation moved and the target did
    not. Every such row carries NaN in `cos_centred` rather than that number (score.py).
    """
    assert family in cfg["family_kinds"], (
        f"family {family!r} has no `family_kinds:` entry, so nothing can say whether a mean may be "
        f"subtracted from its rows (have {sorted(cfg['family_kinds'])})"
    )
    return bool(cfg["family_kinds"][family]["centrable"])


def input_centering(cfg: dict, maemm_key: str) -> str:
    """The centring convention `maemm_key` was TRAINED to receive: a `mus:` name or `none`.

    Refuses rather than defaulting. Half the products in this repo have no MAEMM in scope at all
    (scan, gcg, patchscopes, repo_examples) and the other half would silently re-point every
    number in SMOKES.md if this guessed -- so an entry with no `input:` block is a hard stop with
    the ask named, not a `none`.
    """
    assert maemm_key in cfg["maemms"], f"unknown maemm {maemm_key!r}, want one of {sorted(cfg['maemms'])}"
    spec = cfg["maemms"][maemm_key]
    block = spec.get("input")
    assert isinstance(block, dict) and "centering" in block, (
        f"maemm {maemm_key!r} has no `input: {{centering: ...}}` block in config.yaml, so what it "
        f"was trained to receive is not recorded anywhere. Establish it from the checkpoint's "
        f"training chain and declare it; nothing here will guess (see config.yaml's `maemms:` "
        f"header). To run against a convention you are choosing rather than reading, pass "
        f"--centering explicitly -- it is recorded as a deviation."
    )
    return str(block["centering"])


_MU_CACHE: dict[tuple, object] = {}


def mu_named(cfg: dict, base: str, name: str, root: str = VOL, set_dir: str = ""):
    """The named centring mean [d] as a float32 numpy array, or None for `none`.

    A mean is referred to BY NAME everywhere and no product ever names a path: `mus:` in
    config.yaml is the one registry (it replaced stats.ARCHIVE_MU). `source: set` means the file
    lives in a held-out set's own directory, so those need `set_dir`.

    Loud rather than optional, exactly as `stats_mu` was: a product that needs a mean and cannot
    find it has to stop, because silently falling back to a locally computed mean is the drift
    this module exists to prevent.
    """
    import numpy as np

    if name == NO_CENTRING:
        return None
    assert name != MU_UNKNOWN, (
        f"mu {MU_UNKNOWN!r} cannot be loaded: it is the label for \"centred on a mean nobody here "
        f"can name\", which is a statement about a stored direction and not a file"
    )
    block = cfg["mus"].get(base, {})
    assert name in block, f"base {base!r} declares no mean {name!r}; it has {sorted(block)}"
    spec = block[name]
    src = spec["source"]
    assert src != "unknown", (
        f"mus[{base!r}][{name!r}] has source `unknown`: the mean is not a file we hold. A direction "
        f"stored under it can be RETURNED with a label (common.dirs_for) but never re-derived."
    )
    if src == "product":
        path = f"{base_dir(base, root)}/{spec['path']}"
    elif src == "set":
        assert set_dir, (
            f"mu {name!r} has source `set` ({spec['path']}), so it lives in a held-out set's own "
            f"directory: mu_named needs set_dir="
        )
        path = f"{set_dir.rstrip('/')}/{spec['path']}"
    else:
        path = os.path.join(cfg["modal"]["archive"], spec["path"])
    key = (base, name, path)
    if key in _MU_CACHE:
        return _MU_CACHE[key]
    d = cfg["bases"][base]["d"]
    assert os.path.exists(path), (
        f"no {path}: mean {name!r} of base {base} is declared in config.yaml's `mus:` but is not on "
        f"disk. For `stats_mu` that means the `stats` product has not run; nothing recomputes its own."
    )
    mu = np.load(path) if path.endswith(".npy") else read_array(path, "float32", (d,))
    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    assert mu.shape == (d,) and np.isfinite(mu).all(), (
        f"{path}: expected {d} finite float32 values for mean {name!r}, got shape {mu.shape}"
    )
    _MU_CACHE[key] = mu
    return mu


def set_storage(cfg: dict, set_dir: str, root: str = VOL) -> dict:
    """The storage contract of the set at `set_dir`: {storage, mu_stored, family_mu, source}.

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
        for field in ("storage", "mu_stored"):
            assert field in rec, f"{path} is missing {field!r}"
        assert rec["storage"] in STORAGE_KINDS, f"{path}: storage {rec['storage']!r} is not a kind"
        return {
            "storage": rec["storage"],
            "mu_stored": rec["mu_stored"],
            "family_mu": dict(rec.get("family_mu") or {}),
            "source": path,
        }
    name = os.path.basename(set_dir.rstrip("/"))
    assert name in cfg["heldout"], (
        f"{set_dir} carries no {STORAGE_FILE} and {name!r} is not a set declared in config.yaml, so "
        f"nothing states whether its vecs.f16 is centred. Declare it under `heldout:` (storage / "
        f"mu_stored / family_mu) or re-draw the set, which writes the contract itself."
    )
    spec = cfg["heldout"][name]
    return {
        "storage": spec["storage"],
        "mu_stored": spec["mu_stored"],
        "family_mu": dict(spec.get("family_mu") or {}),
        "source": f"config.yaml heldout.{name}",
    }


def storage_record(cfg: dict, set_name: str, families) -> dict:
    """The `storage.json` a freshly drawn `storage: raw` set writes. See `set_storage`."""
    return {
        "storage": "raw",
        "mu_stored": None,
        "family_mu": {},
        "families": {f: cfg["family_kinds"][f]["kind"] for f in families},
        "note": (
            "RAW STORAGE: act.f32 [N, d] holds the row's own vector before any mean was subtracted "
            "and vecs.f16 is unit(act) -- UNCENTRED. Every centred direction is derived at read "
            "time by common.dirs_for(..., centering=<mu name>). For a family that is not "
            "`centrable` (family_kinds), the act.f32 row IS the stored unit direction, so "
            "unit(act) == vecs.f16 there and no centring ever applies to it."
        ),
    }


def dirs_for(cfg: dict, base: str, set_dir: str, centering: str, root: str = VOL, notes=None):
    """The direction every row of this set carries under `centering` -- the WHOLE [N, d] array.

        storage: raw        -> unit(act - mu_named(centering)) for a centrable family,
                               unit(act) for every other row (there is no mean to subtract)
        storage: unit       -> the stored row, ASSERTING the family's own mean == `centering`;
                               a family whose mean is `unknown` is returned with a warning and a
                               label instead of a refusal (infra/2026-09-21_evals-plan-main.md §1.4)
        storage: dirs_only  -> the stored row (no family in such a set is centrable)

    `centering` is a `mus:` name or `none` and is always EXPLICIT: the four products with no MAEMM
    in scope (scan, gcg, patchscopes, repo_examples) would otherwise silently re-point the corpus
    search baseline and the GCG ceiling away from every number measured between 09-16 and 09-21.

    Returns the whole array and leaves ROW SELECTION at each call site, because `rows` means three
    different things across the readers -- global in score/rollouts_*, family-local in gcg
    (`--family sae --rows 0-7` is global rows 1024-1031), absent in scan/centred/repo_examples.

    `notes`, when a list is passed, receives one human-readable line per thing a reader of the
    product's README has to know (the convention used, and every labelled family).
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
    assert centering == NO_CENTRING or centering in cfg["mus"].get(base, {}), (
        f"--centering {centering!r} is neither {NO_CENTRING!r} nor one of base {base}'s declared "
        f"means {sorted(cfg['mus'].get(base, {}))}"
    )
    fams = [r["family"] for r in rows]
    say = notes if notes is not None else []

    if storage == "raw":
        apath = f"{set_dir}/act.f32"
        assert os.path.exists(apath), (
            f"{set_dir} declares `storage: raw` ({contract['source']}) but has no act.f32; a raw "
            f"set derives every direction from it. Re-draw, or `--product targets --re-derive`."
        )
        act = read_array(apath, "float32", (n, d)).astype(np.float32)
        mu = mu_named(cfg, base, centering, root, set_dir=set_dir)
        out = act.copy()
        if mu is not None:
            cen = np.array([family_centrable(cfg, f) for f in fams], dtype=bool)
            out[cen] -= mu[None, :]
            say.append(
                f"directions derived from {apath} at centering={centering!r}: "
                f"unit(act - {centering}) on {int(cen.sum())} centrable rows "
                f"({sorted({f for f, c in zip(fams, cen, strict=True) if c})}), unit(act) on the "
                f"other {int((~cen).sum())} (family_kinds says they have no mean to subtract)"
            )
        else:
            say.append(f"directions derived from {apath} at centering='none': unit(act), all {n} rows")
        return _unit_rows(out)

    v = read_array(f"{set_dir}/vecs.f16", "float16", (n, d)).astype(np.float32)
    if storage == "dirs_only":
        say.append(
            f"{set_dir} is `storage: dirs_only` ({contract['source']}): the stored vecs.f16 rows are "
            f"returned unchanged and centering={centering!r} does not apply to any of them"
        )
        return _unit_rows(v)

    # storage: unit -- the stored row is a direction under SOME mean, and without act.f32 (or a raw
    # act_norm to solve with, rollouts_nla.build_inputs) it cannot be moved to another one.
    fam_mu = contract["family_mu"]
    stored = contract["mu_stored"]
    labelled, mismatched = [], []
    for fam in sorted(set(fams)):
        own = fam_mu.get(fam, stored if stored is not None else NO_CENTRING)
        if own == MU_UNKNOWN:
            labelled.append(fam)
            continue
        if not family_centrable(cfg, fam):
            continue  # an encoder column is the same object at every centring
        if own != centering:
            mismatched.append((fam, own))
    assert not mismatched, (
        f"{set_dir} is `storage: unit` ({contract['source']}) and its "
        + ", ".join(f"{f!r} rows are stored under mean {m!r}" for f, m in mismatched)
        + f", but this run asks for centering={centering!r}. A stored unit direction cannot be "
        f"re-centred -- unit(act) and mu do not give unit(act - mu) without ||act||. Re-derive the "
        f"set at `storage: raw` (`--product targets --re-derive <set>`), or run at the mean it was "
        f"built with and say so."
    )
    if labelled:
        msg = (
            f"{set_dir}: families {labelled} arrived already centred on a mean nobody here can name "
            f"(family_mu: {MU_UNKNOWN}). Their rows are returned AS SHIPPED, under the producer's "
            f"convention, NOT at centering={centering!r}; every number read off them is labelled."
        )
        print(f"[dirs_for] WARNING: {msg}", flush=True)
        say.append(msg)
    say.append(
        f"{set_dir} is `storage: unit` ({contract['source']}): stored directions returned as they "
        f"are, which matches centering={centering!r} for every centrable family that names a mean"
    )
    return _unit_rows(v)


def _unit_rows(v):
    """Row-wise L2 normalisation in fp32 with the numpy eps convention (see centred.py:63)."""
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def is_nla(cfg: dict, maemm_key: str) -> bool:
    """True for the activation-verbalizer baseline, whose ONLY generator is `rollouts_nla`."""
    return cfg["maemms"][maemm_key].get("type") == "nla"


def default_heldout(cfg: dict) -> str:
    """The held-out set a product takes when `--set` is omitted: the latest NON-imported one.

    `sorted(cfg["heldout"])[-1]` was that rule until a set drawn elsewhere had to be REGISTERED
    here so the entrypoint would accept its name (`2026-09-20_sae2m_2k`, written by
    features/draw_sae2m.py through features/spawn.py, which bypasses modal_app.main's assert).
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


def maemms_for(cfg: dict, base: str = "", computable_only: bool = True) -> list[str]:
    """The MAEMM keys of `base` (or of every base when base is ""), in config order.

    `compute: false` entries are DECLARED but nothing is generated for them: they exist so the
    paper's model table, the prompt inventory and `check` know about them. Every "all MAEMMs"
    iteration that would spend GPU on a MAEMM must therefore filter them out, which is what the
    default does; `check` passes computable_only=False and resolves them too (leniently -- an
    entry nobody computes need not be in the HF cache yet).
    """
    keys = [k for k in cfg["maemms"] if not base or split_key(k, "maemm")[0] == base]
    if computable_only:
        keys = [k for k in keys if cfg["maemms"][k].get("compute", True)]
    return keys


def sae_key_for(cfg: dict, base: str, want: str = "") -> str:
    """WHICH SAE of `base`: `want` when given, else the single one -- asserting when there are two.

    `qwen36-27b` has carried two SAEs since `sae2m` landed (`l42-1b` at 131k and `sae2m` at 2^21),
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


def stats_mu(cfg: dict, base: str, root: str = VOL):
    """`stats/mu.f32` [d] as a float32 numpy array -- OUR 64/16-window read-layer mean.

    `mu_named(cfg, base, "stats_mu", root)` under its historical name, kept because a dozen call
    sites spell it this way. It is no longer "the ONE centring mean": since 2026-09-21 a mean is
    chosen by name per run (`input.centering` / `--centering`), and `stats_mu` is one of several
    (`mus:` in config.yaml). Still loud rather than optional -- a product that needs a mean and
    finds no `stats/` has to stop, because silently computing its own is the drift this file exists
    to prevent.
    """
    return mu_named(cfg, base, "stats_mu", root)


# ---------------------------------------------------------------------------------------------
# volume paths (infra/precompute-layout.md §1)
#
# Every base product takes a `root` so a smoke can mirror the whole relative layout under
# <root>/base/<base>/... (default /vol). maemms/, gcg/ and the archive are not root-relative: they
# are never written by a smoke.
# ---------------------------------------------------------------------------------------------


def base_dir(base: str, root: str = VOL) -> str:
    return f"{root}/base/{base}"


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


def scan_dir(base: str, set_name: str, root: str = VOL) -> str:
    return f"{base_dir(base, root)}/scan/{set_name}"


def sae_dir(sae_key: str, root: str = VOL) -> str:
    base, name = split_key(sae_key, "sae")
    return f"{base_dir(base, root)}/sae/{name}"


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


def maemm_dir(maemm_key: str, root: str = VOL) -> str:
    """`<root>/maemms/<base>/<name>`. Root-relative since step 3: a rollouts/score smoke writes the
    whole maemms/ subtree under /vol/runs/<date>_paper-evals-smoke, exactly as base/ does."""
    base, name = split_key(maemm_key, "maemm")
    return f"{root}/maemms/{base}/{name}"


ENGINES = ("hf", "vllm")


def rollout_stem(set_name: str, engine: str = "hf") -> str:
    """The rollouts/ file stem of one (set, engine) pair.

    The HF stem is the bare set name, so every step-3 file keeps its path; the vLLM stem is
    suffixed. Both engines write into the SAME accumulating rollouts/ directory and `score` picks
    one with `--engine`, so an HF and a vLLM run of the same set never overwrite each other and the
    paired comparison has both files side by side.
    """
    assert engine in ENGINES, f"unknown engine {engine!r}, want one of {list(ENGINES)}"
    return set_name if engine == "hf" else f"{set_name}__{engine}"


def rollouts_path(maemm_key: str, set_name: str, root: str = VOL, engine: str = "hf") -> str:
    return f"{maemm_dir(maemm_key, root)}/rollouts/{rollout_stem(set_name, engine)}.jsonl"


def rollouts_dir(maemm_key: str, root: str = VOL) -> str:
    return f"{maemm_dir(maemm_key, root)}/rollouts"


def nla_variant_dir(maemm_key: str, set_name: str, amp: str, root: str = VOL) -> str:
    """`<root>/maemms/<base>/<nla>/variants/<set>__amp-<amp>` -- a NON-default `rollouts_nla --amp`.

    Its own one-shot directory in the `score --rollouts-dir` layout (`rollouts.jsonl` +
    `rollouts.summary.json` + `scores/`), deliberately NOT the accumulating `rollouts/`: an amp
    sweep is a different INPUT to the same model, and putting it under the set's own stem there
    would make it indistinguishable from the headline run in `index.json`.
    """
    assert amp and "/" not in amp and " " not in amp, f"amp {amp!r} must be a bare directory suffix"
    return f"{maemm_dir(maemm_key, root)}/variants/{set_name}__amp-{amp}"


def scores_dir(maemm_key: str, set_name: str, root: str = VOL, engine: str = "hf") -> str:
    return f"{maemm_dir(maemm_key, root)}/scores/{rollout_stem(set_name, engine)}"


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
# HF cache resolution (modal/maemm_modal.py:_snapshot)
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


def maemm_weights_path(cfg: dict, maemm_key: str) -> str:
    """Directory holding the MAEMM's weights: adapter dir (lora) or full snapshot (full).

    The subdir is JOINED onto the snapshot rather than passed as PeftModel(subfolder=...) because
    that is what modal/maemm_modal.py:1842 does and because the resulting path is what the README
    and the weight sha have to name.
    """
    spec = cfg["maemms"][maemm_key]
    path = spec["src"] if "src" in spec else snapshot(cfg, spec["hf"])
    if spec.get("subdir"):
        path = os.path.join(path, spec["subdir"])
    # lora: the adapter dir. full / base: a model dir -- for `base` that is the base snapshot
    # itself, which is exactly the point of the control.
    marker = "adapter_config.json" if spec["type"] == "lora" else "config.json"
    assert os.path.exists(os.path.join(path, marker)), (
        f"maemm {maemm_key!r}: no {marker} under {path} (type={spec['type']})"
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
    """(model, tok) for the clean base. attn_implementation='sdpa' as eval/eval_ckpt_daemon.py:341."""
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


def load_maemm(cfg: dict, base: str, maemm_key: str, device: str = "cuda"):
    """(model, tok, kind) for the GENERATING model. kind is the config `type`: 'lora', 'full' or 'base'.

    lora: base + PeftModel; scoring runs on the same object with the adapter disabled.
    full: the tuned model IS the generator and has no adapter to switch off, so the caller must
    load a separate clean base for scoring (eval/eval_ckpt_daemon.py:333-390).
    base: the UNTRAINED-BASE CONTROL -- no MAEMM weights anywhere; `maemm_weights_path` resolves to
    the base's own snapshot, so this loads exactly what `load_base` would and takes the same
    no-adapter path as `full`. The kind is returned verbatim so every caller can tell the control
    apart from a trained full-parameter MAEMM (their marker-norm expectations are OPPOSITE).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key_base, _ = split_key(maemm_key, "maemm")
    assert key_base == base, f"maemm {maemm_key!r} is not on base {base!r}"
    spec = cfg["maemms"][maemm_key]
    path = maemm_weights_path(cfg, maemm_key)
    if spec["type"] == "lora":
        import torch
        from peft import PeftModel

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

    A PeftModel disables its adapter; a plain model (full-parameter MAEMM protocol, where the
    scoring model is a separately loaded clean base) is already clean, so this is a no-op rather
    than an error -- the caller decides which object to hand in.
    """
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------
# prompts (mxf/prompts.py)
# ---------------------------------------------------------------------------------------------


def _chat_ids(tok, content, add_gen):
    """mxf/prompts.py:18-24, verbatim."""
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
    """mxf/prompts.py:27-32, verbatim."""
    mid = tok.encode(MARKER, add_special_tokens=False)
    assert len(mid) == 1, f"marker not single-token: {mid}"
    pos = [i for i, t in enumerate(ids) if t == mid[0]]
    assert len(pos) == 1, f"expected exactly one marker, got {len(pos)}"
    return pos


def _maemm_prompt(tok, read_layer: int):
    """mxf/prompts.py:35-40: the marker goes AFTER the chat template's generation prefix."""
    instr = _INSTR_TEMPLATE.format(read_layer=read_layer)
    ids = _chat_ids(tok, instr, add_gen=True) + tok.encode(MARKER, add_special_tokens=False)
    return ids, len(ids) - 1


# ours8b and celeste27b are today the SAME template at read layer 27 vs 42 (our fork's
# repo-maemm/mxf/prompts.py substitutes READ_LAYER=27 into her text). Both names are kept so that a
# future MAEMM trained on a different prompt costs one config entry and one function here.
PROMPTS = {
    "ours8b": _maemm_prompt,
    "celeste27b": _maemm_prompt,
}


def prompt_ids(tok, prompt_name: str, read_layer: int):
    """(ids, marker_pos). Asserts the marker is one token occurring exactly once."""
    assert prompt_name in PROMPTS, f"unknown prompt {prompt_name!r}, want one of {sorted(PROMPTS)}"
    ids, pos = PROMPTS[prompt_name](tok, read_layer)
    found = marker_positions(tok, ids)
    assert found == [pos], f"marker at {found} but the prompt builder reported {pos}"
    return ids, pos


# ---------------------------------------------------------------------------------------------
# injection and reading (mxf/inject.py)
# ---------------------------------------------------------------------------------------------


def get_layer(model, layer: int):
    """mxf/inject.py:10-14, verbatim: the decoder block at `layer`, unwrapping DDP + PEFT."""
    m = model.module if hasattr(model, "module") else model
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.model.layers[layer]


def make_inject_hook(vecs, positions, coeff, device, dtype):
    """mxf/inject.py:24-57 with mode='add' only (the only mode any MAEMM was trained with).

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
    """mxf/inject.py:129-135, verbatim."""
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
    """mxf/inject.py:147-149: capture the block OUTPUT (never output_hidden_states) and stop."""

    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).float()
        raise _Stop

    return cap


def read_resid(model, layer, batch, pool="all"):
    """mxf/inject.py:142-166, verbatim. Layer-`layer` residual for a tokenized batch, no injection.

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
    """rl/rl.py:170-192: ||h|| at the marker of INJECT_LAYER's output for the shared prompt.

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

    eval/eval_universal.py:139 uses bos, falling back to eos: the Qwen3 tokenizers have no bos, so
    in practice this is the eos id. Named once so the scorer, the corpus passes and the SAE
    examples all prepend the SAME token (a different sink shifts every position's context).
    """
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert sink is not None, "tokenizer has neither a bos nor an eos token to use as the sink"
    return int(sink)


def strip_repo_sink(ids, acts, sink: int, sink_first: bool):
    """Align an SAE repo's shipped max-activating windows with our scoring protocol.

    A repo whose scan re-encoded each window the way `eval/eval_universal.py:_reencode` does has a
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
    """rl/rl.py:61-75: tokenizer eos UNION generation_config eos (checklist item 85).

    Her version wraps the generation_config read in a bare `except Exception: pass`; here a model
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
    """rl/rl.py:82-90: keep tokens up to and INCLUDING the first stop token; drop any pad tail."""
    trimmed = []
    for t in g:
        trimmed.append(t)
        if t in stop_ids:
            break
    return trimmed if trimmed else list(g)


def vllm_finish_ids(token_ids, finish_reason: str, stop_reason, stop_ids, eos_fallback: int):
    """vLLM's returned ids -> the ids rollouts_hf would have stored. Returns (ids, appended).

    vLLM drops the stop token from `token_ids` when it stopped on one of `stop_token_ids`
    (rl/rl.py:310-313 re-appends it, because rl/rl.py:82-90's trimming KEEPS the stop token and the
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


# rl/rl.py:192-211 -- the adapter module rename vLLM needs for the 27B, which it serves as
# Qwen3_5ForConditionalGeneration. vLLM validates adapter module names by SUFFIX ONLY, so a
# CausalLM-named adapter passes validation and is then SILENTLY ignored (measured: 1.47 nats of
# logprob gap). The 8B is a plain CausalLM and needs no rename.
VLLM_LORA_RENAME = {"qwen36-27b": ("model.layers.", "model.language_model.layers.")}


def rename_lora_keys(state_dict_keys, base: str) -> dict[str, str]:
    """{old key -> new key} for `base`'s vLLM naming. Identity when the base needs no rename.

    A key that already mentions `language_model` is left alone (rl/rl.py:205), and the prefix is
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
# scoring: the one re-encode protocol (eval/eval_universal.py:129-162)
# ---------------------------------------------------------------------------------------------


def encode_for_score(tok, texts, max_length: int = SCORE_MAX_LENGTH):
    """The TOKENIZATION half of the scoring protocol, on its own: -> a list of id lists.

    eval/eval_universal.py:136-138 -- an all-whitespace rollout tokenizes to zero tokens, so `" "`
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
):
    """Per-token cosine and residual norm of each ID LIST on the CLEAN base at `read_layer`.

    THE scoring forward of the whole pipeline. `score_tokens` is this function with
    `encode_for_score` in front of it, so rollouts, scan references, SAE repo windows and the GCG
    loop cannot drift on the objective: they are the same code path by construction, not by
    agreement (checklist item 12).

    Protocol, all of it from eval/eval_universal.py:_reencode: right padding, a BOS sink (or eos if
    the tokenizer has no bos) prepended at column 0 and excluded from `keep`, cosine in fp32.

    DIVERGENCE (Tomáš, 2026-09-15): her 10x-nanmedian norm filter (eval_universal.py:71,145-147) is
    NOT applied. The per-token norm is stored so reconstruction/stats.py can apply it as an option.

    The cosine is UNCENTRED while realact target directions are unit(act - mu)
    (data/build_universal_bank.py:26,310). That asymmetry is Celeste's and is kept.

    `model` must already be the scoring model: a PeftModel (its adapter is disabled here) or a
    separately loaded clean base (full-parameter MAEMMs have no adapter to switch off).

    Returns a dict of [N, max_length + 1] tensors on the cpu -- cos (f32), norm (f32), keep (bool),
    ids (i64) -- where column 0 is the sink, cos/norm are NaN outside `keep`, and ids is -1 there.
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
    # +1 for the sink at column 0. This is SCORE_WIDTH whenever max_length is the protocol's own
    # SCORE_MAX_LENGTH, which is every caller but the NLA arm.
    score_width = max_length + 1
    out = {
        "cos": torch.full((n, score_width), float("nan")),
        "norm": torch.full((n, score_width), float("nan")),
        "keep": torch.zeros((n, score_width), dtype=torch.bool),
        "ids": torch.full((n, score_width), -1, dtype=torch.long),
    }
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


def best_of_k_means(vals, ks) -> dict[int, float]:
    """Best-of-k means by DISJOINT groups: split `vals` into floor(n/k) consecutive groups of k,
    take each group's max, average them.

    This is the plain subsample estimator, not the unbiased order-statistic one (that belongs to
    reconstruction/stats.py): it uses only floor(n/k)*k of the n rollouts and its variance at
    k = n is the variance of a single best-of-n draw. k values above n are skipped rather than
    silently clamped, so a summary never claims a bo-k it could not compute.
    """
    vals = [float(v) for v in vals]
    n = len(vals)
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        assert k >= 1, f"best-of-k needs k >= 1, got {k}"
        if k > n:
            continue
        g = n // k
        out[k] = sum(max(vals[i * k : (i + 1) * k]) for i in range(g)) / g
    return out


# ---------------------------------------------------------------------------------------------
# SAE (mxf/sae.py:20-52 + eval/eval_universal.py:77-93 for the gate)
# ---------------------------------------------------------------------------------------------


class BatchTopKSAE:
    """W_enc [d, F], W_dec [F, d], b_enc [F], b_dec [d], threshold: the learned BatchTopK gate."""

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.threshold = threshold
        self.d_in, self.d_sae = W_enc.shape


def load_sae(path: str, d_model: int, device: str = "cpu", dtype=None, need_decoder: bool = True):
    """mxf/sae.py:39-52 plus the `threshold` buffer her eval reads through sae_gate().

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
    # eval/eval_universal.py:77-84: "fired" is act > this learned threshold (~1.654 for the 131k
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


def sae_encode(sae: BatchTopKSAE, h, feature_ids):
    """mxf/sae.py:27-31: pre-topk post-ReLU activations relu((x - b_dec) @ W_enc[:,f] + b_enc[f])."""
    import torch

    idx = torch.as_tensor(feature_ids, device=sae.W_enc.device)
    return torch.relu((h - sae.b_dec) @ sae.W_enc[:, idx] + sae.b_enc[idx])


def sae_dirs(sae: BatchTopKSAE, feature_ids):
    """mxf/sae.py:33-36: the `sae` family target is the UNIT ENCODER COLUMN unit(W_enc[:, f])."""
    import torch

    idx = torch.as_tensor(feature_ids, device=sae.W_enc.device)
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
    """Streamed sha256 over a checkpoint directory's weight files, for the MAEMM README.

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

    MEASURED constraint: the 27B full MAEMM is ~52 GiB across 13 files on the FUSE-mounted volume;
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


class OutDir:
    """Temp-and-rename output directory with a README and an array index (infra/design.md §1).

    Writers write to `<name>.tmp-<date>/` and rename on completion; an existing `<name>/` is never
    overwritten without force=True. On an exception the temp directory is LEFT IN PLACE and its
    path printed, so a failed run is inspectable and never half-renamed.

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
        # rather than a one-shot product. The existing directory is copied into the temp dir first,
        # so the rename is still atomic and the caller's per-file overwrite rule is its own.
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
        self.notes: list[str] = []
        self.sections: list[tuple[str, list[str]]] = []
        self.tmp = self.path.with_name(f"{self.path.name}.tmp-{time.strftime('%Y-%m-%d')}")
        # the product's own start, so `wall` and `cost` cover the whole call (model load included);
        # 0 means "measure from __enter__"
        self._t0 = t0

    def __enter__(self):
        if self.path.exists() and self.keep_existing:
            if self.tmp.exists():
                print(f"[outdir] removing a leftover temp dir {self.tmp}", flush=True)
                shutil.rmtree(self.tmp)
            shutil.copytree(self.path, self.tmp)
            existing = self.tmp / "index.json"
            if existing.exists():
                with open(existing) as fh:
                    self.index.update(json.load(fh))
            for stale in ("README.md", "index.json"):
                (self.tmp / stale).unlink(missing_ok=True)
                self.index.pop(stale, None)
            if not self._t0:
                self._t0 = time.time()
            print(f"[outdir] keeping the {len(self.index)} entries already in {self.path}", flush=True)
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
        if self.index:
            lines += ["## Files", "", "| file | kind | dtype | shape | size |", "|---|---|---|---|---|"]
            for name, meta in self.index.items():
                lines.append(
                    f"| `{name}` | {meta['kind']} | {meta.get('dtype', '')} | "
                    f"{meta.get('shape', meta.get('rows', ''))} | {human(meta['bytes'])} |"
                )
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"[outdir] FAILED: {exc_type.__name__}; temp dir kept at {self.tmp}", flush=True)
            return False  # never swallow
        self.write_json("index.json", self.index)
        with open(self.tmp / "README.md", "w") as fh:
            fh.write(self._readme())
        if self.keep_existing and self.path.exists():
            shutil.rmtree(self.path)
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
