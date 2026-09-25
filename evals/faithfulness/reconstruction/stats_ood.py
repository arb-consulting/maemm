#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""The OOD generalisation evaluation's analysis layer (design infra/2026-09-18_ood-eval-design.md §6, §11).

Local, CPU, no GPU and no model. Like `reconstruction/stats.py` it fetches only small files off the
volume into `common.mirror_dir(<root-tag>)` and writes into
`common.out_dir("reconstruction")/<root-tag>/`, both outside `evals/faithfulness/` by default (the old
`reconstruction/data/<root-tag>/` and `reconstruction/out/<root-tag>/` are legacy);
`paper/inversion-eval/` is FROZEN and nothing here writes into it.

    cd <repo>
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=<your-profile>; \\
     uv run evals/faithfulness/reconstruction/stats_ood.py tables --root-tag full)

Commands:

    tables        the per-arm paired comparison, the strata, the chance levels and the examples
    en-ref        review R1 alone: the English reference recomputed from the 2026-09-16_v1 scan
                  with own-document windows excluded (needs only the local mirror)
    train-share   review R7: what share of the inverter's training text is code-like / non-English
    selfcheck     every code path above on synthetic inputs, in seconds, before any launch

Products (design §6): `ood_arms.csv`, `ood_per_target.csv`, `ood_strata.csv`, `ood_examples.md`.

Estimators. Best-of-k is the UNBIASED order statistic (`results.common.bo_unbiased`, the one the
paper's tables use). The per-arm paired difference is bootstrapped by resampling TARGETS (10,000
percentile resamples), which is the unit of analysis; the outcome is three-state (review R9):
`exceeds` (the CI is above zero), `inconclusive` (it covers zero -- a failure to reject, NOT
"does not generalise") and `reversed` (below zero).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
import yaml
from rich.console import Console
from rich.table import Table as RichTable

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
CONFIG = PAPER_EVALS / "config.yaml"
sys.path.insert(0, str(PAPER_EVALS))

import precompute.common as C  # noqa: E402  (the ONE script table, code-like rule and arm table)
from reconstruction.stats import ROOTS, Scores, Vol, default_out, mirror_dir  # noqa: E402

# THE best-of-k estimator of the pipeline, one definition for the whole results layer
# (M0a, 2026-09-23). `reconstruction/stats.py` has its own copy of the same order
# statistic; `results/selftest.check_one_bo_estimator` holds the layers to one number.
from results.common import bo_unbiased  # noqa: E402

OOD_SET = "2026-09-18_ood_v1"
BASE = "qwen36-27b"
EN_SET = "2026-09-16_v1"
BOOT = 10_000
BOOT_SEED = 20260918
ALPHA = 0.05
BO_KS = (1, 4, 64)
# The nested corpus sizes the per-target and per-arm columns are built for, in MILLIONS. It is a
# UNION over arms, not a ladder any one arm has: `config.yaml`'s `ood_arms:` gives 1/4/10 to every
# arm, 16 to the four that reach it (`tha_Thai ufw_en python owm`) and stops `shell` at 4 and
# `formulas` at 1. A size an arm does not carry simply has no scan, and the column stays null --
# which is why this is one constant read in three places rather than three literals that drifted
# apart: before 2026-09-23 all three said (1, 4, 16) and no 10M column could reach the table.
CORPUS_SIZES = (1, 4, 10, 16)
# The arms of the level-1 conjunction: everything except the `diag` formula arm and the `ufw_en`
# pipeline check (design §0: 21 arms = lang 8, code 8, math 4, ufw_zh).
CONJUNCTION_EXCLUDE = ("formulas", "ufw_en")

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------------


def boot_ci(d: np.ndarray, n: int = BOOT, alpha: float = ALPHA, seed: int = BOOT_SEED):
    """(mean, lo, hi) of a percentile bootstrap over the TARGETS of one arm."""
    d = np.asarray(d, dtype=float)
    assert d.ndim == 1 and d.size > 1, f"need >1 paired difference, got {d.shape}"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def outcome(lo: float, hi: float) -> str:
    """The three-state per-arm verdict (review R9)."""
    if lo > 0:
        return "exceeds"
    if hi < 0:
        return "reversed"
    return "inconclusive"


from results.common import resolve_mu  # noqa: E402

def declared_mu(cfg: dict, key: str, base: str) -> str:
    """The mean `config.yaml` DECLARES this checkpoint was trained to receive, as one string.

    `precompute.common.input_mu` refuses an entry with no `mu:` key at all rather than guessing. A
    bare checkpoint name is prefixed with the base, the way `tables` accepts `--maemm`.
    """
    full = key if "/" in key else f"{base}/{key}"
    return resolve_mu(C.input_mu(cfg, full), base)


def assert_control_paired(cfg: dict, base: str, maemm_key: str, control_key: str) -> None:
    """SPEC §7 R4: a control may only be differenced against a MAEMM that DECLARES the same mean.

    LOUD, because the failure it guards is silent. Every `maemm_vs_control` cell below is
    `bo64_maemm - bo64_control` on the same targets, which is a control only when both arms were
    injected at the same centring; at two means it is the difference of two angles to two
    different vectors, and this file had **no mu, `centred` or `whiten` reference anywhere** before
    today -- `--control` resolved into whatever `--stem` named and `tables` checked only that the
    key was in `cfg["maemms"]`.

    As of 2026-09-23 the base control `qwen36-27b/2026-09-16_base-control` declares the 27B
    `whiten_mu` path in `config.yaml` (it was `mu: null` before, an explicit statement that it took
    a RAW activation), so the pairing HOLDS for the intended pair -- rl-last16 against that
    control. It does not hold for the old primary, which declares `base/{base}/stats/mu.f32`.
    """
    a, b = declared_mu(cfg, maemm_key, base), declared_mu(cfg, control_key, base)
    assert a == b and a != "unknown", (
        f"CONTROL NOT PAIRED (spec §7 R4): maemm {maemm_key!r} declares mu={a!r} and control "
        f"{control_key!r} declares mu={b!r} in config.yaml. The control is 'the MAEMM with base "
        f"weights', so it takes the MAEMM's input convention; at two means it is a control for a "
        f"different experiment and `maemm_vs_control` is not a difference. As of 2026-09-23 "
        f"`qwen36-27b/2026-09-16_base-control` declares the 27B whiten_mu path (it was `null` "
        f"before), so the intended pair -- rl-last16 against that control -- does pair; the old "
        f"primary `2026-09-10_rl-8x2048-full` declares `base/{{base}}/stats/mu.f32` and does not. "
        f"Fix the config or pass a control drawn at this MAEMM's mean; nothing here will guess."
    )


def derangement(n: int, seed: int = BOOT_SEED) -> np.ndarray:
    """A fixed permutation with no fixed point -- the R4 shuffled-target control's pairing."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p
    raise AssertionError(f"no derangement of {n} found in 1000 draws")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without scipy: Pearson on the ranks (ties averaged)."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average the ranks of ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    a, b = rank(x), rank(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------------------------
# R1: the English reference, own-document windows excluded
# ---------------------------------------------------------------------------------------------


def english_reference(vol: Vol, base: str = BASE, set_name: str = EN_SET) -> dict | None:
    """Corpus top-1 per nested size on the English realact targets, with and without own-document
    windows (review R1).

    The paper's frozen table counts the target's OWN document -- at 4M the top-1 window IS the
    target's document for 114 of 512 targets -- while the OOD corpora contain no target document at
    all. The like-for-like reference is therefore the no-own-document one, which is what every OOD
    comparison in `ood_arms.csv` is read against.

    A target whose whole stored top-64 is own-document has NO non-own candidate and is EXCLUDED
    from the no-own mean rather than scored -1; that exclusion is what makes the recomputation
    reproduce 0.314 / 0.351 / 0.385 at 1/4/16M.
    """
    topk = vol.jsonl(f"base/{base}/scan/{set_name}/topk.jsonl")
    ids = vol.jsonl(f"base/{base}/heldout/{set_name}/ids.jsonl")
    if topk is None or ids is None:
        return None
    doc_of = {int(r["row"]): r.get("doc") for r in ids if r["family"] == "realact"}
    out = {}
    for r in topk:
        if r["family"] != "realact" or not r["top"]:
            continue
        own = doc_of[int(r["row"])]
        size = int(r["size"])
        non = [t[3] for t in r["top"] if t[0] != own]
        rec = out.setdefault(size, {"all": [], "noown": [], "own_top1": 0, "no_candidate": 0})
        rec["all"].append(r["top"][0][3])
        rec["own_top1"] += int(r["top"][0][0] == own)
        if non:
            rec["noown"].append(max(non))
        else:
            rec["no_candidate"] += 1
    return {
        size: {
            "n": len(v["all"]),
            "top1_all": float(np.mean(v["all"])),
            "top1_noown": float(np.mean(v["noown"])) if v["noown"] else float("nan"),
            "n_noown": len(v["noown"]),
            "own_is_top1": v["own_top1"],
            "no_noown_candidate": v["no_candidate"],
        }
        for size, v in sorted(out.items())
    }


# ---------------------------------------------------------------------------------------------
# loading the OOD products
# ---------------------------------------------------------------------------------------------


def load_ood_ids(vol: Vol, base: str, set_name: str) -> list[dict] | None:
    return vol.jsonl(f"base/{base}/heldout/{set_name}/ids.jsonl")


def load_scans(vol: Vol, base: str, set_name: str) -> dict[str, dict]:
    """{scan subdirectory -> {(set, set_row, size) -> top-1 cos}} for every scan of this set.

    An OOD scan writes `scan/<set>/<corpus>[-<M>m]/`; the pre-2026-09-18 English scan writes
    `scan/<set>/` directly and is read by `english_reference`, never here.
    """
    # TWO LAYOUTS. The pre-rebase OOD branch nested the corpus under the set --
    # `scan/<set>/<corpus>[-<M>m]/` -- and the sets drawn before 2026-09-21 still carry it. The
    # pipeline keys a scan by (set, corpus) as SIBLINGS: `scan/<set>__<corpus>[__<M>m]/`. Reading
    # only the first would leave the in-domain column of every post-rebase set EMPTY rather than
    # wrong, which is the failure that looks like a result.
    subs = [(sub, f"base/{base}/scan/{set_name}/{sub}") for sub in vol.ls(f"base/{base}/scan/{set_name}")]
    for d in vol.ls(f"base/{base}/scan"):
        if d.startswith(set_name + "__"):
            subs.append((d[len(set_name) + 2 :], f"base/{base}/scan/{d}"))
    out = {}
    for sub, rel in subs:
        rows = vol.jsonl(f"{rel}/topk.jsonl")
        if rows is None:
            continue
        top1: dict = {}
        for r in rows:
            if not r["top"]:
                continue
            key = (r.get("set", set_name), int(r.get("set_row", r["row"])), int(r["size"]))
            top1[key] = r["top"][0][3]
        assert sub not in out, (
            f"two scan directories resolve to the corpus label {sub!r} for set {set_name}: the "
            f"nested and the sibling layouts both carry it, and they are different products"
        )
        out[sub] = top1
        console.print(f"[dim]scan {sub}: {len(top1)} (target, size) cells[/dim]")
    return out


def load_quantiles(vol: Vol, base: str, set_name: str, sub: str):
    idx = vol.json(f"base/{base}/scan/{set_name}/{sub}/index.json")
    if not idx or "quantiles.f16" not in idx:
        return None
    shape = tuple(idx["quantiles.f16"]["shape"])
    arr = vol.array(f"base/{base}/scan/{set_name}/{sub}/quantiles.f16", "float16", shape)
    return None if arr is None else arr.astype(np.float32)


def load_nll(vol: Vol, base: str, set_name: str) -> dict[int, dict]:
    rows = vol.jsonl(f"base/{base}/nll/{set_name}/per_target.jsonl")
    return {int(r["row"]): r for r in rows} if rows else {}


def wall_seconds(vol: Vol, rel: str) -> float | None:
    """The `- wall: N s` line of a product's own README (review R2's GPU-seconds column).

    The README is the number to trust: `modal app logs` replays stale output under a timeout
    (checklist item 84), so the per-product wall recorded at write time is the only honest one.
    """
    p = vol.get(f"{rel}/README.md")
    if p is None:
        return None
    m = re.search(r"^- wall: ([0-9.]+)s", p.read_text(), re.M)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------
# R3: language id and the code-like classifier on the rollouts
# ---------------------------------------------------------------------------------------------


LID_REPO = "facebook/fasttext-language-identification"
LID_FILE = "model.bin"


def load_lid(path: Path | None = None):
    """fastText **lid218e** (`facebook/fasttext-language-identification`), or None.

    2026-09-18: the official Facebook repo rather than a community re-upload of `lid.176`,
    because its FLORES-200 labels ARE our arm ids for seven of the eight language arms (`tha_Thai`,
    `ces_Latn`, ...); Chinese is `zho_Hans`/`zho_Hant` there, which config.yaml's `lid:` list
    carries. `--lid-model` overrides with a local file; otherwise the HF cache is used and the
    file's sha256 is printed so the paper can pin it.
    """
    try:
        import fasttext  # noqa: PLC0415
    except ImportError:
        console.print("[yellow]fasttext is not installed: `uv run --with fasttext ...`[/yellow]")
        return None
    if path is not None and path.exists():
        console.print(f"[dim]lid model {path}[/dim]")
        return fasttext.load_model(str(path))
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url  # noqa: PLC0415

        meta = get_hf_file_metadata(hf_hub_url(LID_REPO, LID_FILE))
        p = hf_hub_download(LID_REPO, LID_FILE)
    except Exception as e:  # noqa: BLE001 -- offline or gated: the lid columns are optional
        console.print(f"[yellow]no lid model ({type(e).__name__}): the lid columns are skipped[/yellow]")
        return None
    console.print(f"[dim]lid218e {LID_REPO}/{LID_FILE} sha256 {meta.etag} commit {meta.commit_hash}[/dim]")
    return fasttext.load_model(p)


def lid_label(model, text: str) -> tuple[str, float]:
    """(label, probability) of the top language, through fastText's C++ predict.

    `_FastText.predict` ends in `np.array(probs, copy=False)`, which numpy >= 2 REFUSES
    (MEASURED 2026-09-18: `ValueError: Unable to avoid copy while creating an array as requested`
    on fasttext 0.9.3), and this script is pinned to numpy >= 2 like the rest of reconstruction/.
    The underlying `model.f.predict` returns plain python lists and is what the wrapper calls, so
    it is used directly where it exists.
    """
    t = " ".join(text.split())
    if not t:
        return "", 0.0
    inner = getattr(model, "f", None)
    if inner is not None:
        # MEASURED on fasttext 0.9.3: `f.predict` returns [(prob, "__label__x"), ...] for a single
        # string -- the python wrapper is what unzips it and then numpy-wraps the probabilities.
        pairs = inner.predict(t, 1, 0.0, "strict")
        if not pairs:
            return "", 0.0
        prob, label = pairs[0]
        return label.removeprefix("__label__"), float(prob)
    labels, probs = model.predict(t, k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


# `(.+)$` and NOT `(\S+)$`: `score` writes ONE `- rollouts:` line naming EVERY file it read, and a
# product `--rows` split into chunks names all of them, comma-separated. The end-anchored
# non-space group matched nothing at all on such a line, so `rollouts_rels_from_readme` returned
# None and the R3 language-id column went silently empty on every chunked product -- the exact
# failure this rule was extracted to prevent, arriving through a shape it did not know about
# (M5 full-scale run, 2026-09-23: 22 chunks, `lid` never ran).
ROLLOUTS_RE = re.compile(r"^- rollouts: (.+)$", re.M)


def _root_relative(rel: str) -> str:
    """A path as the README spells it (`/vol/...`) as a volume-root-relative one."""
    rel = rel.strip()
    for pre in ("/vol/", "vol/", "/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


def rollouts_rels_from_readme(vol, scores_rel: str) -> list[str]:
    """EVERY rollouts file a scores directory actually READ, from its own README. Root-relative.

    THE ONE RULE, in the lower layer, because both readers need it and only one of them had it.
    `results/ood.lid_rates` learned it on 2026-09-21 ("the language-id column was
    silently empty"): a scores directory's name does NOT determine its rollouts file. `--score-tag`
    makes them differ on purpose -- `scores/<set>__vllm__asym` scores `rollouts/<set>__vllm.jsonl`
    -- and after the 2026-09-22 rebase `--run-tag` and `--score-tag` are BOTH in the scores name
    (common.score_tag_of), so there are now two ways for a rebuilt stem to be wrong instead of one.

    `stats_ood.rollout_texts` still rebuilt it from `--stem` and so still had the defect its
    sibling had been fixed for: `--stem <set>__vllm__asym` finds the scores, looks for a rollouts
    file that was never written, gets nothing, and the R3 language-id column comes out EMPTY with
    no error -- which is worse than no column, because an empty one reads as "the answers were not
    in the arm's language".

    A LIST, not one path, since 2026-09-23: `--rows` splits one rollouts product across N
    `<stem>__rows<a>-<b>.jsonl` chunks that live side by side and are ONE product
    (`precompute.common.read_rollouts`), and `score` records all of them on the one line. Returning
    the first would score the language column on 512 of 11264 targets and say nothing about it.
    [] when the README is absent or names no rollouts, which the caller reports rather than
    treating as an empty product.

    `vol` is duck-typed: `reconstruction.stats.Vol` and `results.common.Vol` both have `.get` and
    `.local`.
    """
    p = vol.get(f"{scores_rel}/README.md")
    if p is None:
        return []
    m = ROLLOUTS_RE.search(p.read_text())
    if not m:
        return []
    return [_root_relative(x) for x in m.group(1).split(",") if x.strip()]


def read_rollout_rows(vol, rels: list[str]) -> list[dict]:
    """The rollout rows of the ONE product `rels` names -- a whole-set file, or `--rows` chunks.

    The chunked union is NOT concatenated here: it goes through
    `precompute.common.read_rollouts`, which is where "these N files are one product" is defined
    and where the checks live -- the chunks must agree on every field of `_CHUNK_INVARIANT` (one
    experiment) and cover DISJOINT target rows. A local concatenation would read a half-overwritten
    directory as a complete product and nothing would say so.

    `read_rollouts` globs the local directory and needs each chunk's `.summary.json` beside it,
    which the README does not list; both are fetched here from the producer's own list, and the
    files it ended up reading are asserted to be exactly that list -- so a stale chunk of the same
    stem left in the mirror cannot silently join the union.

    A file the README names that is NOT on the volume (or not in the mirror, offline) returns []
    for the WHOLE product, logged in `vol.missing`, never a partial read: `lid` is an optional
    column and its absence is a note, while a rate computed over 512 of 11264 targets and printed
    as the arm's rate is a wrong number. The assertions above stay assertions -- they are about
    two runs wearing one stem, which is a defect and not an absence.
    """
    if not rels:
        return []
    if len(rels) == 1 and C.ROWS_MARK not in Path(rels[0]).name:
        return vol.jsonl(rels[0]) or []
    dirs = sorted({str(Path(r).parent) for r in rels})
    assert len(dirs) == 1, (
        f"the scores README names rollout chunks in {len(dirs)} directories ({dirs}); the chunks "
        f"of one product live side by side in one `rollouts/` directory"
    )
    stems = sorted({Path(r).name.split(C.ROWS_MARK)[0] for r in rels})
    assert len(stems) == 1, (
        f"the scores README names chunks of {len(stems)} different stems ({stems}); that is two "
        f"rollouts products claiming one scores directory"
    )
    absent: list[str] = []
    for rel in rels:
        srel = (rel[: -len(".jsonl")] if rel.endswith(".jsonl") else rel) + ".summary.json"
        for want in (rel, srel):
            if vol.get(want) is None:
                absent.append(want)
    if absent:
        print(
            f"[stats_ood] {len(absent)} of {2 * len(rels)} files of the chunked rollouts product "
            f"`{stems[0]}` are not available ({absent[0]} ...): the product is NOT read, and "
            f"nothing downstream gets a partial one",
            flush=True,
        )
        return []
    rows, summary, _ = C.read_rollouts(str(Path(vol.local) / dirs[0]), stems[0])
    got, want = sorted(summary.get("chunks") or []), sorted(Path(r).name for r in rels)
    assert got == want, (
        f"the union read {got} but the scores README names {want}: the mirror holds chunk files "
        f"of this stem that this product did not score, and reading them would mix two runs"
    )
    return rows


def rollout_texts(vol: Vol, base: str, maemm: str, set_name: str, stem: str) -> dict[int, list[str]]:
    """{target row -> [rollout text] in k order} from the rollouts jsonl (the one big fetch here).

    The paths come from the SCORES README when it names any, and only fall back to the `--stem`
    composition for a product written before READMEs carried the line. See
    `rollouts_rels_from_readme` for why rebuilding it from the directory name is wrong, and
    `read_rollout_rows` for why a `--rows` chunked product is a list of files and still one
    product.
    """
    rels = rollouts_rels_from_readme(vol, f"maemms/{base}/{maemm}/scores/{stem}")
    if not rels:
        rels = [f"maemms/{base}/{maemm}/rollouts/{stem}.jsonl"]
        print(f"[stats_ood] the scores README names no rollouts file; falling back to {rels[0]}",
              flush=True)
    rows = read_rollout_rows(vol, rels)
    if not rows:
        return {}
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(int(r["row"]), []).append(r.get("text", ""))
    return out


# ---------------------------------------------------------------------------------------------
# R7: what the inverter was trained on
# ---------------------------------------------------------------------------------------------


TRAIN_WINDOW = 512
TRAIN_N = 10_000


def _train_share_corpus(vol: Vol, base: str, tok, n: int, seed: int):
    """`--source corpus`: n random 512-token windows of OUR English corpus (UFW en p0009-0010)."""
    p = vol.get(f"base/{base}/corpus/tokens.i32")
    if p is None:
        return None, "base/<base>/corpus/tokens.i32 is not mirrored locally"
    toks = np.memmap(p, dtype=np.int32, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(toks) - TRAIN_WINDOW, size=n)
    return ((None, tok.decode([int(x) for x in toks[s : s + TRAIN_WINDOW]])) for s in starts), (
        f"{n} random {TRAIN_WINDOW}-token windows of base/{base}/corpus/tokens.i32 "
        f"(Ultra-FineWeb en p0009-0010), seed {seed}"
    )


DOMAIN_DOCS = 150  # documents per domain in the stratified R7 draw (2026-09-18)
HEAD_BYTES = 12 << 20  # how much of each domain's first file is read: ~12 MB gives >= 150 documents


def _card_domain_tokens(dataset: str) -> tuple[dict[str, float], str]:
    """{domain -> total tokens} from the dataset card's own "Data Statistics" table.

    Read from the card through the HF API rather than hardcoded, so the weights and the sample come
    from the same revision. Rows look like `| aerospace | 5.77B | ... | 6.34B | ... |`; the `Total
    Tokens` column (the 4th) is taken.
    """
    from huggingface_hub import hf_hub_download

    card = Path(hf_hub_download(dataset, "README.md", repo_type="dataset")).read_text()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    out: dict[str, float] = {}
    for line in card.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not re.fullmatch(r"[a-z_]+", cells[0]):
            continue
        m = re.fullmatch(r"([0-9.]+)([KMBT])", cells[4])
        if m:
            out[cells[0]] = float(m.group(1)) * mult[m.group(2)]
    assert out, f"no Data Statistics rows parsed from {dataset}'s card"
    return out, f"{dataset} card 'Data Statistics', Total Tokens column, {len(out)} domains"


def _head_lines(url: str, nbytes: int):
    """Complete json lines from the first `nbytes` of a URL (the transfer is cut, not ranged).

    A FineFineWeb domain file is ~317 MB and we want 150 documents of it, so the read stops after
    the head; the last, partial line is dropped.
    """
    import urllib.request

    with urllib.request.urlopen(url) as fh:  # noqa: S310 -- an https hub URL built by hf_hub_url
        buf = fh.read(nbytes)
    return buf.decode("utf-8", errors="replace").split("\n")[:-1]


def _train_share_hf(dataset: str, tok, n: int, stratify: bool = True):
    """`--source hf:<id>`: the inverter's activation corpus, sampled over its DOMAINS.

    `m-a-p/FineFineWeb` is the primary checkpoint's activation corpus
    (`infra/2026-09-18_ood-eval-training-data.md`, 2026-09-18). It is laid out as
    `<domain>/<domain>_NNNNNN.jsonl` over 67 domains and 66,103 files of ~317 MB, so reading it "in
    file order" -- what the upstream collectors do -- samples ONE domain: the first 10k documents are all
    `aerospace` (MEASURED 2026-09-18). The default here is therefore STRATIFIED (2026-09-18):
    the head of the first file of every domain, `DOMAIN_DOCS` documents each, with the per-domain
    rate reported and the aggregate weighted by the card's own token counts. `--no-stratify` gives
    the file-order draw back.

    No revision is pinned by the checkpoint's config, so the repo revision and the fetch date are
    recorded instead.
    """
    import datetime
    import json as _json

    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

    api = HfApi()
    rev = api.repo_info(dataset, repo_type="dataset").sha
    files = sorted(f for f in api.list_repo_files(dataset, repo_type="dataset") if f.endswith(".jsonl"))
    assert files, f"{dataset} has no .jsonl files"
    today = datetime.date.today().isoformat()
    read: list[str] = []

    if not stratify:
        def gen():
            got = 0
            for f in files:
                read.append(f)
                path = hf_hub_download(dataset, f, repo_type="dataset")
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        text = (_json.loads(line).get("text") or "")
                        ids = tok(text, add_special_tokens=False)["input_ids"] if text else []
                        if len(ids) < TRAIN_WINDOW:
                            continue
                        got += 1
                        yield None, tok.decode(ids[:TRAIN_WINDOW])
                        if got >= n:
                            return

        return gen(), read, (
            f"{n} documents of {dataset} at revision {rev[:12]}, FILE ORDER over {len(files)} "
            f".jsonl files, one {TRAIN_WINDOW}-token window each, fetched {today}. NOTE: file "
            f"order samples the first domain directory alphabetically, not the corpus"
        )

    first_of: dict[str, str] = {}
    for f in files:
        first_of.setdefault(f.split("/")[0], f)
    domains = sorted(first_of)

    def gen():
        for dom in domains:
            f = first_of[dom]
            read.append(f)
            got = 0
            try:
                lines = _head_lines(hf_hub_url(dataset, f, repo_type="dataset"), HEAD_BYTES)
            except Exception as e:  # noqa: BLE001 -- one unreadable domain must not sink the pass
                console.print(f"[yellow]{f}: {type(e).__name__}, skipped[/yellow]")
                continue
            for line in lines:
                if got >= DOMAIN_DOCS:
                    break
                try:
                    text = _json.loads(line).get("text") or ""
                except _json.JSONDecodeError:
                    continue
                if not text:
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) < TRAIN_WINDOW:
                    continue
                got += 1
                yield dom, tok.decode(ids[:TRAIN_WINDOW])
            console.print(f"[dim]{dom}: {got} documents[/dim]")

    return gen(), read, (
        f"{DOMAIN_DOCS} documents from the head of the first file of EACH of {len(domains)} "
        f"domains of {dataset} at revision {rev[:12]} (stratified; the first {HEAD_BYTES >> 20} MB "
        f"of each file are read), one {TRAIN_WINDOW}-token window each, fetched {today}. No "
        f"revision is pinned by the checkpoint's own config, so this is the share as of that date"
    )


# ---------------------------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------------------------


def build_tables(vol: Vol, cfg: dict, out_dir: Path, args: dict) -> dict:
    base, set_name = args["base"], args["set"]
    maemm, control = args["maemm"], args["control"]
    stem = args["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # SPEC §7 R4: THE BASE CONTROL IS RESOLVED FROM CONFIG WITH ITS CENTRING DECLARED. `--control`
    # is otherwise resolved into whatever `--stem` names, and a control at the wrong mean produces
    # a `maemm_vs_control` column that looks exactly like a good one. FIRST, before a single
    # volume read: an unpaired pair is a config fact, and finding it out after the fetches is
    # finding it out late.
    if maemm and control:
        assert_control_paired(cfg, base, maemm, control)
        notes.append(
            f"control `{control}` and maemm `{maemm}` both declare "
            f"mu={declared_mu(cfg, control, base)} (spec §7 R4, checked)"
        )

    ids = load_ood_ids(vol, base, set_name)
    assert ids, f"no held-out set at base/{base}/heldout/{set_name}/ids.jsonl"
    arms = C.ood_arms(cfg)
    by_row = {int(r["row"]): r for r in ids}

    en_ref = english_reference(vol)
    if en_ref is None:
        notes.append("R1 English reference: the 2026-09-16_v1 scan is not available")

    scans = load_scans(vol, base, set_name)
    if not scans:
        notes.append(f"no scan of {set_name}: every corpus column is empty")
    nll = load_nll(vol, base, set_name)
    if not nll:
        notes.append(f"no nll product for {set_name}: bpb_ctx / nll_ctx columns are empty")

    sides = {}
    for label, key in (("maemm", maemm), ("control", control)):
        if not key:
            continue
        s = Scores(vol, base, key, set_name, by_row, f"maemms/{base}/{key}/scores/{stem}")
        if getattr(s, "ok", False):
            sides[label] = s
        else:
            notes.append(f"no scores for {key} on {stem}")

    # ---- per target -------------------------------------------------------------------------
    rows = []
    for r in ids:
        row = int(r["row"])
        arm = r["arm"]
        spec = arms.get(arm, {})
        rec = {
            "row": row,
            "arm": arm,
            "family": r["family"],
            "tok_class": r.get("tok_class"),
            "n_subtokens": r.get("n_subtokens"),
            "byte_piece": bool(r.get("byte_piece")),
            "whole_char": bool(r.get("whole_char")),
            "multi_char": bool(r.get("multi_char")),
            "char_type": r.get("char_type"),
            "char_type_body": r.get("char_type_body"),
            "p": r.get("p"),
            "L": r.get("L"),
            "unspaced": bool(spec.get("unspaced")),
        }
        for label, s in sides.items():
            i = s.rows.index(row) if row in s.rows else None
            for k in BO_KS:
                rec[f"bo{k}_{label}"] = (
                    float(bo_unbiased(s.best[i : i + 1], k)[0]) if i is not None else None
                )
        for sub, top1 in scans.items():
            for size in CORPUS_SIZES:
                v = top1.get((set_name, row, size))
                if v is not None:
                    rec[f"corpus_{sub}_{size}m"] = v
        if row in nll:
            rec["nll_ctx"] = nll[row]["nll_ctx"]
            rec["bpb_ctx"] = nll[row]["bpb_ctx"]
            rec["nll_p"] = nll[row]["nll_p"]
        rows.append(rec)
    per_target = pl.DataFrame(rows, infer_schema_length=None)

    # In-domain corpus column: the scan of the arm's OWN corpus at nested size `size`. The scan
    # SUBDIRECTORY carries the bound the whole scan ran at (`tha_Thai-4m`), which is not the size
    # being read, so the arm's subdirectory is matched by prefix rather than by name.
    def in_domain(rec, size):
        arm = rec["arm"]
        for sub in sorted(scans):
            if sub == arm or sub.startswith(f"{arm}-"):
                col = f"corpus_{sub}_{size}m"
                if col in per_target.columns:
                    return col
        return None

    # ---- per arm ----------------------------------------------------------------------------
    arm_rows, strata_rows = [], []
    lid = load_lid(args["lid_model"]) if args["lid"] else None
    texts = rollout_texts(vol, base, maemm, set_name, stem) if (lid or args["lid"]) else {}
    gpu = {
        "rollouts": wall_seconds(vol, f"maemms/{base}/{maemm}/rollouts"),
        "score": wall_seconds(vol, f"maemms/{base}/{maemm}/scores/{stem}"),
    }

    for arm, spec in arms.items():
        sel = per_target.filter(pl.col("arm") == arm)
        if sel.height == 0:
            continue
        rec = {
            "arm": arm,
            "family": spec["family"],
            "n": sel.height,
            "script": spec["script"],
            "unspaced": bool(spec["unspaced"]),
            "sizes": "/".join(str(s) for s in spec["sizes"]),
            "licence": spec.get("licence"),
        }
        for label in sides:
            for k in BO_KS:
                col = f"bo{k}_{label}"
                if col in sel.columns and sel[col].null_count() < sel.height:
                    rec[col] = round(float(sel[col].mean()), 4)
        for size in CORPUS_SIZES:
            col = in_domain({"arm": arm}, size)
            if col and sel[col].null_count() < sel.height:
                rec[f"corpus_{size}m"] = round(float(sel[col].mean()), 4)
        for size in CORPUS_SIZES:
            col = f"corpus_corpus-4m_{size}m"
            if col not in sel.columns:
                col = f"corpus_corpus_{size}m"
            if col in sel.columns and sel[col].null_count() < sel.height:
                rec[f"english_{size}m"] = round(float(sel[col].mean()), 4)
        for c in ("nll_ctx", "bpb_ctx"):
            if c in sel.columns and sel[c].null_count() < sel.height:
                rec[c] = round(float(sel[c].mean()), 4)
        rec["byte_piece_rate"] = round(float(sel["byte_piece"].mean()), 4)
        rec["whole_char_rate"] = round(float(sel["whole_char"].mean()), 4)
        for cls in ("word", "first", "mid", "last", "unspaced"):
            rec[f"cls_{cls}"] = int((sel["tok_class"] == cls).sum())

        # the four comparisons of design §6
        comps = {
            "bo64_vs_4m": ("bo64_maemm", in_domain({"arm": arm}, 4)),
            "bo4_vs_4m": ("bo4_maemm", in_domain({"arm": arm}, 4)),
            "bo1_vs_1m": ("bo1_maemm", in_domain({"arm": arm}, 1)),
            "bo64_vs_en16m": ("bo64_maemm", "corpus_corpus_16m"),
            "maemm_vs_control": ("bo64_maemm", "bo64_control"),
        }
        for name, (a, b) in comps.items():
            if not a or not b or a not in sel.columns or b not in sel.columns:
                continue
            if sel[a].null_count() or sel[b].null_count():
                continue
            d = (sel[a] - sel[b]).to_numpy()
            mean, lo, hi = boot_ci(d)
            rec[f"{name}_delta"] = round(mean, 4)
            rec[f"{name}_lo"] = round(lo, 4)
            rec[f"{name}_hi"] = round(hi, 4)
            rec[f"{name}_win"] = round(float((d > 0).mean()), 4)
            if name == "bo64_vs_4m":
                rec["outcome"] = outcome(lo, hi)
                rec["in_conjunction"] = arm not in CONJUNCTION_EXCLUDE

        # R4 chance levels
        v = vol.array(
            f"base/{base}/heldout/{set_name}/vecs.f16", "float16",
            (len(ids), cfg["bases"][base]["d"]),
        )
        if v is not None:
            idx = sel["row"].to_numpy()
            vv = v[idx].astype(np.float32)
            vv /= np.linalg.norm(vv, axis=1, keepdims=True)
            g = vv @ vv.T
            rec["chance_pairwise_cos"] = round(float(g[np.triu_indices(len(vv), 1)].mean()), 4)
        q = load_quantiles(vol, base, set_name, arm)
        if q is None:
            q = load_quantiles(vol, base, set_name, f"{arm}-4m")
        if q is not None:
            sizes_here = [s for s in spec["sizes"]]
            si = sizes_here.index(4) if 4 in sizes_here else len(sizes_here) - 1
            idx = sel["row"].to_numpy()
            rec["chance_scan_median"] = round(float(q[idx, si, 0].mean()), 4)
            rec["chance_scan_p99"] = round(float(q[idx, si, 2].mean()), 4)

        # R3 language id / code-like on the top-1 and top-4 rollouts
        if "maemm" in sides and texts:
            s = sides["maemm"]
            hits1, hits4, code1, n_seen = 0, 0, 0, 0
            for row in sel["row"].to_numpy():
                if row not in texts or row not in s.rows:
                    continue
                order = np.argsort(-s.best[s.rows.index(row)])
                tops = [texts[row][k] for k in order[:4] if k < len(texts[row])]
                if not tops:
                    continue
                n_seen += 1
                code1 += int(C.code_like(tops[0]))
                if lid is not None and spec.get("lid"):
                    want = set(spec["lid"])
                    l1, _ = lid_label(lid, tops[0])
                    hits1 += int(l1 in want)
                    hits4 += int(any(lid_label(lid, t)[0] in want for t in tops))
            if n_seen:
                rec["code_like_top1_rate"] = round(code1 / n_seen, 4)
                if lid is not None and spec.get("lid"):
                    rec["lid_top1_rate"] = round(hits1 / n_seen, 4)
                    rec["lid_top4_rate"] = round(hits4 / n_seen, 4)

        # R2 GPU-seconds per target
        if gpu["rollouts"] and len(ids):
            rec["gpu_s_maemm"] = round((gpu["rollouts"] + (gpu["score"] or 0)) / len(ids), 3)
        scan_w = wall_seconds(vol, f"base/{base}/scan/{set_name}/{arm}")
        if scan_w:
            rec["gpu_s_scan_4m"] = round(scan_w / len(ids), 3)

        # within-arm Spearman of bo64 against the base's own bits per byte (R6)
        if "bo64_maemm" in sel.columns and "bpb_ctx" in sel.columns:
            if not sel["bo64_maemm"].null_count() and not sel["bpb_ctx"].null_count():
                rec["spearman_bo64_bpb"] = round(
                    spearman(sel["bo64_maemm"].to_numpy(), sel["bpb_ctx"].to_numpy()), 4
                )
        arm_rows.append(rec)

        # strata
        dcol, ccol = "bo64_maemm", in_domain({"arm": arm}, 4)
        for key in ("tok_class", "byte_piece", "char_type", "char_type_body"):
            for val in sel[key].unique().sort():
                sub = sel.filter(pl.col(key) == val)
                srec = {
                    "arm": arm, "family": spec["family"], "stratum": key,
                    "value": str(val), "n": sub.height,
                }
                if dcol in sub.columns and not sub[dcol].null_count():
                    srec["bo64"] = round(float(sub[dcol].mean()), 4)
                if ccol and ccol in sub.columns and not sub[ccol].null_count():
                    srec["corpus_4m"] = round(float(sub[ccol].mean()), 4)
                    if dcol in sub.columns and not sub[dcol].null_count():
                        srec["delta"] = round(float((sub[dcol] - sub[ccol]).mean()), 4)
                strata_rows.append(srec)

    arms_df = pl.DataFrame(arm_rows, infer_schema_length=None) if arm_rows else pl.DataFrame()
    strata_df = pl.DataFrame(strata_rows, infer_schema_length=None) if strata_rows else pl.DataFrame()

    # the en_ref row, so the reference and the arms come out of ONE script (R1)
    if en_ref and not arms_df.is_empty():
        ref = {
            "arm": "en_ref",
            "family": "reference",
            "n": en_ref[max(en_ref)]["n_noown"],
            "script": "Latin",
            "unspaced": False,
            "sizes": "/".join(str(s) for s in sorted(en_ref)),
            "licence": "apache-2.0",
        }
        for size, v in en_ref.items():
            if size in (1, 4, 16):
                ref[f"corpus_{size}m"] = round(v["top1_noown"], 4)
        arms_df = pl.concat([arms_df, pl.DataFrame([ref])], how="diagonal_relaxed")

    if not arms_df.is_empty():
        arms_df.write_csv(out_dir / "ood_arms.csv")
    per_target.write_csv(out_dir / "ood_per_target.csv")
    if not strata_df.is_empty():
        strata_df.write_csv(out_dir / "ood_strata.csv")

    # R8 examples: the median-delta target of each arm, with its licence
    lines = ["# OOD examples: the median-delta target of each arm (review R8)", ""]
    lines += [
        "One target per arm, the one whose `bo64 - in-domain top-1 at 4M` is the arm's MEDIAN, with",
        "the licence of its source beside it. Proof-Pile-2 (`arxiv`) declares no licence on the HF",
        "repo, so no example is printed from it.",
        "",
    ]
    for rec in arm_rows:
        arm = rec["arm"]
        if arm == "arxiv":
            lines += [f"## {arm}", "", "*no example printed: the source declares no licence*", ""]
            continue
        sel = per_target.filter(pl.col("arm") == arm)
        ccol = in_domain({"arm": arm}, 4)
        if "bo64_maemm" not in sel.columns or not ccol or sel["bo64_maemm"].null_count():
            continue
        d = (sel["bo64_maemm"] - sel[ccol]).to_numpy()
        row = int(sel["row"].to_numpy()[int(np.argsort(d)[len(d) // 2])])
        src = by_row[row]
        lines += [
            f"## {arm} (row {row}, delta {d[int(np.argsort(d)[len(d) // 2])]:+.4f})",
            "",
            f"- source: `{src['src_dataset']}` @ `{src['src_revision'][:12]}`, rows "
            f"{src['src_rows']}, licence **{src.get('licence')}**",
            f"- p {src['p']}, L {src['L']}, tok_class `{src.get('tok_class')}`, "
            f"byte_piece {src.get('byte_piece')}, char_type `{src.get('char_type')}`",
            "",
            "```",
            src["span_text"],
            "```",
            "",
        ]
    (out_dir / "ood_examples.md").write_text("\n".join(lines) + "\n")

    if not arms_df.is_empty():
        want = ("arm", "family", "n", "bo64_maemm", "corpus_4m", "bo64_vs_4m_delta",
                "bo64_vs_4m_lo", "bo64_vs_4m_hi", "outcome", "byte_piece_rate")
        show = [c for c in want if c in arms_df.columns]
        t = RichTable(title="OOD arms", header_style="bold")
        for c in show:
            t.add_column(c)
        for r in arms_df.select(show).iter_rows():
            t.add_row(*["" if v is None else str(v) for v in r])
        console.print(t)
    for n in notes:
        console.print(f"[yellow]note: {n}[/yellow]")
    (out_dir / "ood_notes.md").write_text("\n".join(f"- {n}" for n in notes) + "\n")
    return {
        "arms": arms_df.height if not arms_df.is_empty() else 0,
        "targets": per_target.height,
        "strata": strata_df.height if not strata_df.is_empty() else 0,
        "notes": notes,
        "en_ref": en_ref,
    }


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------


def _vol(root_tag, fetch, refetch, quiet, modal_cmd, data_dir):
    return Vol(
        root_tag, data_dir or mirror_dir(ROOTS[root_tag]), modal_cmd, refetch, quiet,
        offline=not fetch
    )


@app.command()
def tables(
    root_tag: Annotated[str, typer.Option(help="smoke | full")] = "full",
    set_name: Annotated[str, typer.Option("--set", help="the OOD held-out set")] = OOD_SET,
    base: Annotated[str, typer.Option()] = BASE,
    # REQUIRED, and resolved through `cfg["maemms"]` below. They were Python defaults naming one
    # checkpoint generation; the eval now runs BOTH (the old primary and `rl-last16`), so a default
    # here would silently label one generation's numbers with the other's name (eval plan §4.3.4).
    maemm: Annotated[
        str, typer.Option(help="the MAEMM to tabulate, e.g. 2026-09-18_rl-last16-lr5e-7")
    ] = ...,
    control: Annotated[str, typer.Option(help="the untrained-base control")] = ...,
    stem: Annotated[str, typer.Option(help="the scores subdirectory")] = "",
    lid: Annotated[bool, typer.Option(help="run fastText lid.176 on the rollouts (review R3)")] = True,
    lid_model: Annotated[Path | None, typer.Option(help="lid.176.bin")] = None,
    fetch: Annotated[bool, typer.Option()] = True,
    refetch: Annotated[bool, typer.Option()] = False,
    quiet: Annotated[bool, typer.Option()] = False,
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
    data_dir: Annotated[Path | None, typer.Option()] = None,
    out_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """The per-arm tables, strata, chance levels and examples."""
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    for label, key in (("--maemm", maemm), ("--control", control)):
        full = key if "/" in key else f"{base}/{key}"
        assert full in cfg["maemms"], (
            f"{label} {key!r} is not a checkpoint in config.yaml; the names under base {base!r} "
            f"are {sorted(k.split('/', 1)[1] for k in cfg['maemms'] if k.startswith(base + '/'))}"
        )
    vol = _vol(root_tag, fetch, refetch, quiet, modal_cmd, data_dir)
    res = build_tables(
        vol,
        cfg,
        out_dir or (default_out("reconstruction") / root_tag / "ood"),
        {
            "base": base,
            "set": set_name,
            "maemm": maemm,
            "control": control,
            "stem": stem or f"{set_name}__vllm",
            "lid": lid,
            "lid_model": lid_model,
        },
    )
    console.print(res)


@app.command("en-ref")
def en_ref_cmd(
    root_tag: Annotated[str, typer.Option(help="smoke | full")] = "full",
    fetch: Annotated[bool, typer.Option()] = True,
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
    data_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Review R1 alone: the English corpus top-1 with and without own-document windows."""
    vol = _vol(root_tag, fetch, False, False, modal_cmd, data_dir)
    ref = english_reference(vol)
    assert ref, "the 2026-09-16_v1 scan and ids are not available on this root"
    t = RichTable(title="R1: English realact corpus top-1 (2026-09-16_v1, 27B)", header_style="bold")
    cols = ("size", "n", "top1 (all windows)", "top1 (no own document)", "own is top-1",
            "no non-own candidate")
    for c in cols:
        t.add_column(c)
    for size, v in ref.items():
        t.add_row(
            f"{size}M", str(v["n"]), f"{v['top1_all']:.4f}", f"{v['top1_noown']:.4f}",
            f"{v['own_is_top1']}", f"{v['no_noown_candidate']}",
        )
    console.print(t)
    console.print(json.dumps(ref, indent=1))


@app.command("train-share")
def train_share(
    source: Annotated[str, typer.Option(help="hf:<dataset id> | corpus")] = "hf:m-a-p/FineFineWeb",
    stratify: Annotated[bool, typer.Option(help="hf source: one file per DOMAIN, not file order")] = True,
    n: Annotated[int, typer.Option()] = TRAIN_N,
    seed: Annotated[int, typer.Option()] = BOOT_SEED,
    base: Annotated[str, typer.Option()] = BASE,
    root_tag: Annotated[str, typer.Option()] = "full",
    tokenizer: Annotated[str, typer.Option(help="the 27B tokenizer's HF id or path")] = "Qwen/Qwen3.6-27B",
    lid_model: Annotated[Path | None, typer.Option()] = None,
    fetch: Annotated[bool, typer.Option()] = True,
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
    data_dir: Annotated[Path | None, typer.Option()] = None,
    out_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Review R7: the code-like and non-English share of the inverter's training text.

    `--source hf:m-a-p/FineFineWeb` is the primary checkpoint's ACTUAL activation corpus
    (`infra/2026-09-18_ood-eval-training-data.md`, 2026-09-18); `--source corpus` measures OUR
    English eval corpus (Ultra-FineWeb en p0009-0010) the same way, and both are reported.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    vol = _vol(root_tag, fetch, False, False, modal_cmd, data_dir)
    read: list[str] = []
    if source == "corpus":
        gen, prov = _train_share_corpus(vol, base, tok, n, seed)
        assert gen is not None, prov
    else:
        assert source.startswith("hf:"), f"--source must be `corpus` or `hf:<id>`, got {source!r}"
        gen, read, prov = _train_share_hf(source.removeprefix("hf:"), tok, n, stratify)
    lid = load_lid(lid_model)
    n_code, n_noneng, seen = 0, 0, 0
    per_dom: dict[str, list[int]] = {}
    for dom, text in gen:
        seen += 1
        c = int(C.code_like(text))
        n_code += c
        if dom is not None:
            d = per_dom.setdefault(dom, [0, 0, 0])
            d[0] += 1
            d[1] += c
        if lid is not None:
            label, p = lid_label(lid, text)
            ne = int(not (label == "eng_Latn" and p > 0.5))
            n_noneng += ne
            if dom is not None:
                per_dom[dom][2] += ne
    rec = {
        "source": source,
        "provenance": prov,
        "windows": seen,
        "files_read": read,
        "code_like": n_code,
        "code_like_share": round(n_code / max(seen, 1), 5),
        "non_english": n_noneng if lid is not None else None,
        "non_english_share": round(n_noneng / max(seen, 1), 5) if lid is not None else None,
        "code_like_rule": (
            f">= {C.CODE_LIKE_MIN} of {list(C.CODE_LIKE_MARKERS)} + an indentation run "
            f"(regex \\n[ \\t]{{2,}}\\S) per {TRAIN_WINDOW}-token window"
        ),
        "lid_rule": (
            "fastText lid218e top label != 'eng_Latn' or p <= 0.5" if lid is not None else "not run"
        ),
        "verdict": (
            "unseen" if n_code / max(seen, 1) < 0.01 else "under-represented"
        ),
    }
    if per_dom:
        weights, wprov = _card_domain_tokens(source.removeprefix("hf:"))
        tot = sum(weights.get(d, 0.0) for d in per_dom)
        rec["domains"] = {
            d: {
                "n": v[0],
                "code_like": v[1],
                "code_like_share": round(v[1] / max(v[0], 1), 4),
                "non_english_share": round(v[2] / max(v[0], 1), 4) if lid is not None else None,
                "card_tokens": weights.get(d),
            }
            for d, v in sorted(per_dom.items())
        }
        rec["weights_provenance"] = wprov
        rec["missing_from_card"] = sorted(d for d in per_dom if d not in weights)
        if tot > 0:
            rec["code_like_share_token_weighted"] = round(
                sum(weights.get(d, 0.0) * v[1] / max(v[0], 1) for d, v in per_dom.items()) / tot, 5
            )
            if lid is not None:
                rec["non_english_share_token_weighted"] = round(
                    sum(weights.get(d, 0.0) * v[2] / max(v[0], 1) for d, v in per_dom.items()) / tot, 5
                )
            rec["verdict"] = (
                "unseen" if rec["code_like_share_token_weighted"] < 0.01 else "under-represented"
            )
    console.print(json.dumps(rec, indent=1))
    d = out_dir or (default_out("reconstruction") / root_tag / "ood")
    d.mkdir(parents=True, exist_ok=True)
    tag = "corpus" if source == "corpus" else source.removeprefix("hf:").replace("/", "_")
    (d / f"train_share_{tag}.json").write_text(json.dumps(rec, indent=1) + "\n")


@app.command()
def selfcheck() -> None:
    """Every code path above on synthetic inputs, in seconds, before any launch."""
    import shutil
    import tempfile

    ok = []

    # --- estimators --------------------------------------------------------------------------
    d = np.full(64, 0.2)
    m, lo, hi = boot_ci(d)
    assert abs(m - 0.2) < 1e-12 and abs(lo - 0.2) < 1e-9 and abs(hi - 0.2) < 1e-9, (m, lo, hi)
    assert outcome(lo, hi) == "exceeds"
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.1, 64)
    m, lo, hi = boot_ci(z)
    assert lo < 0 < hi and outcome(lo, hi) == "inconclusive", (m, lo, hi)
    m, lo, hi = boot_ci(z - 0.5)
    assert outcome(lo, hi) == "reversed", (m, lo, hi)
    # the CI half-width the design predicts at sd 0.104, n 64: ~0.025
    hw = np.mean([
        (lambda t: (t[2] - t[1]) / 2)(boot_ci(rng.normal(0.1, 0.104, 64), n=2000, seed=s))
        for s in range(5)
    ])
    assert 0.018 < hw < 0.033, f"CI half-width {hw:.4f} is not the design's ~0.025"
    p = derangement(64)
    assert len(set(p.tolist())) == 64 and not (p == np.arange(64)).any()
    assert (derangement(64) == p).all(), "the derangement must be fixed"
    assert abs(spearman(np.arange(10), np.arange(10)) - 1.0) < 1e-12
    assert abs(spearman(np.arange(10), -np.arange(10)) + 1.0) < 1e-12
    ok.append("estimators")

    # --- english_reference on a synthetic scan -------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="stats_ood_selfcheck_"))
    try:
        (tmp / f"base/{BASE}/scan/{EN_SET}").mkdir(parents=True)
        (tmp / f"base/{BASE}/heldout/{EN_SET}").mkdir(parents=True)
        ids = [{"row": 0, "family": "realact", "doc": 7}, {"row": 1, "family": "realact", "doc": 9}]
        (tmp / f"base/{BASE}/heldout/{EN_SET}/ids.jsonl").write_text(
            "\n".join(json.dumps(r) for r in ids) + "\n"
        )
        topk = [
            {"row": 0, "family": "realact", "size": 4, "top": [[7, 0, 0, 0.90], [3, 0, 0, 0.40]]},
            {"row": 1, "family": "realact", "size": 4, "top": [[9, 0, 0, 0.80]]},  # own only
        ]
        (tmp / f"base/{BASE}/scan/{EN_SET}/topk.jsonl").write_text(
            "\n".join(json.dumps(r) for r in topk) + "\n"
        )
        vol = Vol("full", tmp, "uvx modal", False, True, offline=True)
        ref = english_reference(vol)
        assert abs(ref[4]["top1_all"] - 0.85) < 1e-9, ref
        assert abs(ref[4]["top1_noown"] - 0.40) < 1e-9, ref  # row 1 has no non-own candidate
        assert ref[4]["n_noown"] == 1 and ref[4]["no_noown_candidate"] == 1, ref
        assert ref[4]["own_is_top1"] == 2, ref
        ok.append("english_reference (including the no-non-own-candidate exclusion)")

        # --- build_tables end to end on a synthetic OOD set -----------------------------------
        cfg = yaml.safe_load(CONFIG.read_text())
        arms = {"tha_Thai": cfg["ood_arms"]["tha_Thai"], "python": cfg["ood_arms"]["python"]}
        cfg["ood_arms"] = arms
        n_arm, d_model = 8, cfg["bases"][BASE]["d"]
        hd = tmp / f"base/{BASE}/heldout/{OOD_SET}"
        hd.mkdir(parents=True)
        ids, rowi = [], 0
        for arm, spec in arms.items():
            for j in range(n_arm):
                ids.append({
                    "row": rowi, "family": spec["family"], "arm": arm, "id": f"{arm}:{j}",
                    "pool_i": j, "src_rows": [j], "src_dataset": spec["dataset"],
                    "src_revision": "0" * 40, "src_split": spec.get("split"),
                    "src_files": spec["files"], "licence": spec.get("licence"),
                    "p": 100 + j, "L": 32, "act_norm": 1.0, "span_text": f"span {arm} {j}",
                    "byte_piece": j % 3 == 0, "whole_char": j % 3 == 1, "multi_char": j % 3 == 2,
                    "char_type": "letter_arm", "char_type_body": "letter_arm",
                    "tok_class": "unspaced" if spec["unspaced"] else "word",
                    "n_subtokens": None if spec["unspaced"] else 1,
                })
                rowi += 1
        (hd / "ids.jsonl").write_text("\n".join(json.dumps(r) for r in ids) + "\n")
        rs = np.random.default_rng(1)
        vecs = rs.normal(size=(len(ids), d_model)).astype(np.float16)
        vecs.tofile(hd / "vecs.f16")
        for arm in arms:
            sd = tmp / f"base/{BASE}/scan/{OOD_SET}/{arm}"
            sd.mkdir(parents=True)
            lines = []
            for r in ids:
                if r["arm"] != arm:
                    continue
                for size in (1, 4, 10):   # 10 so the fixture exercises the full-run column
                    lines.append({
                        "row": r["row"], "set": OOD_SET, "set_row": r["row"], "family": r["family"],
                        "arm": arm, "corpus": arm, "size": size,
                        "top": [[0, 0, 0, 0.30 + 0.01 * size]],
                    })
            (sd / "topk.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
            q = np.tile(np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float16), (len(ids), 2, 1))
            q.tofile(sd / "quantiles.f16")
            (sd / "index.json").write_text(json.dumps({"quantiles.f16": {"shape": [len(ids), 2, 5]}}))
        nd = tmp / f"base/{BASE}/nll/{OOD_SET}"
        nd.mkdir(parents=True)
        (nd / "per_target.jsonl").write_text(
            "\n".join(
                json.dumps({"row": r["row"], "arm": r["arm"], "nll_ctx": 2.0 + 0.01 * r["row"],
                            "bpb_ctx": 1.0 + 0.01 * r["row"], "nll_p": 3.0})
                for r in ids
            ) + "\n"
        )
        # scores: 64 rollouts per target, the MAEMM above the corpus and the control below it.
        # THE PAIR IS rl-last16 AGAINST THE BASE CONTROL, not the old primary: since 2026-09-23 the
        # control declares the 27B whiten_mu path while the old primary declares stats/mu.f32, so
        # `assert_control_paired` (spec §7 R4) refuses that pair -- which is exactly what it is for.
        shapes = (("2026-09-18_rl-last16-lr5e-7", 0.30, 0.70), ("2026-09-16_base-control", 0.05, 0.20))
        for label, lo_, hi_ in shapes:
            sd = tmp / f"maemms/{BASE}/{label}/scores/{OOD_SET}__vllm"
            sd.mkdir(parents=True)
            n_t, n_k, width = len(ids), 64, 96
            cos = np.full((n_t, n_k, width), np.nan, dtype=np.float16)
            cos[:, :, 1:5] = rs.uniform(lo_, hi_, size=(n_t, n_k, 4)).astype(np.float16)
            cos.tofile(sd / "cos.f16")
            np.zeros((n_t, n_k), dtype=np.int16).tofile(sd / "argmax.i16")
            (sd / "index.json").write_text(json.dumps({"cos.f16": {"shape": [n_t, n_k, width]}}))
            (sd / "rows.json").write_text(json.dumps({"rows": [r["row"] for r in ids], "n": n_k}))
            best = np.nanmax(cos.astype(np.float32), axis=2)
            (sd / "per_target.jsonl").write_text(
                "\n".join(
                    json.dumps({"row": r["row"], "mean_cos": float(best[i].mean()), "seed": 1234})
                    for i, r in enumerate(ids)
                ) + "\n"
            )
        vol = Vol("full", tmp, "uvx modal", False, True, offline=True)
        out = tmp / "out"
        res = build_tables(vol, cfg, out, {
            "base": BASE, "set": OOD_SET, "maemm": "2026-09-18_rl-last16-lr5e-7",
            "control": "2026-09-16_base-control", "stem": f"{OOD_SET}__vllm",
            "lid": False, "lid_model": None,
        })
        assert res["targets"] == len(ids), res
        adf = pl.read_csv(out / "ood_arms.csv")
        assert set(adf["arm"]) == {"tha_Thai", "python", "en_ref"}, adf["arm"].to_list()
        arm_rows = adf.filter(pl.col("arm") != "en_ref")
        assert (arm_rows["outcome"] == "exceeds").all(), arm_rows.select(["arm", "outcome"]).to_dicts()
        assert (arm_rows["bo64_vs_4m_lo"] > 0).all()
        assert (arm_rows["maemm_vs_control_delta"] > 0).all()
        assert arm_rows["chance_pairwise_cos"].null_count() == 0
        assert arm_rows["chance_scan_median"].null_count() == 0
        assert set(arm_rows["in_conjunction"]) == {True}
        pt = pl.read_csv(out / "ood_per_target.csv")
        assert pt.height == len(ids) and "bpb_ctx" in pt.columns
        st = pl.read_csv(out / "ood_strata.csv")
        assert st.height > 0 and "delta" in st.columns
        ex = (out / "ood_examples.md").read_text()
        assert "## tha_Thai" in ex and "## python" in ex and "span " in ex
        ok.append("build_tables end to end (arms, per-target, strata, examples)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- the rollouts path comes from the README, not from the stem ----------------------------
    # R3's language-id column is the one that says whether a cosine above the corpus means the
    # output is actually in the arm's language, so an EMPTY one is worse than no column -- it
    # reads as "the answers were not in the arm's language". It goes empty whenever the rollouts
    # file is looked for under a name nobody wrote, and a scores directory's name is not its
    # rollouts file's: `--score-tag` makes them differ on purpose, and since 2026-09-22 both
    # `--run-tag` and `--score-tag` are in the scores name. `results/ood` was fixed for this
    # earlier; this half was not, and kept rebuilding the path from `--stem`.
    tmp = Path(tempfile.mkdtemp(prefix="stats_ood_rollpath_"))
    try:
        sd = tmp / "maemms/b/m/scores/s__vllm__asym"
        sd.mkdir(parents=True)
        (sd / "README.md").write_text(
            "# scores\n\n## Inputs\n\n- rollouts: /vol/maemms/b/m/rollouts/s__vllm.jsonl\n")
        rd = tmp / "maemms/b/m/rollouts"
        rd.mkdir(parents=True)
        (rd / "s__vllm.jsonl").write_text(
            '{"row": 0, "k": 0, "text": "ahoj"}\n{"row": 0, "k": 1, "text": "svete"}\n')
        vol = Vol("full", tmp, "uvx modal", False, True, offline=True)

        rels = rollouts_rels_from_readme(vol, "maemms/b/m/scores/s__vllm__asym")
        assert rels == ["maemms/b/m/rollouts/s__vllm.jsonl"], (
            f"the README's own `- rollouts:` line was not read back root-relative: {rels!r}")
        texts = rollout_texts(vol, "b", "m", "s", "s__vllm__asym")
        assert texts == {0: ["ahoj", "svete"]}, (
            f"rollout_texts rebuilt the path from --stem instead of reading the scores README, so "
            f"the R3 language-id column would be silently EMPTY on every --score-tag source: "
            f"{texts!r}")
        # ...and a product whose README names no rollouts file still resolves, by falling back
        (sd / "README.md").write_text("# scores\n\nno inputs section\n")
        assert rollouts_rels_from_readme(vol, "maemms/b/m/scores/s__vllm__asym") == []
        assert rollout_texts(vol, "b", "m", "s", "s__vllm") == {0: ["ahoj", "svete"]}, (
            "the pre-README fallback to the --stem composition stopped working")
        ok.append("rollouts path from the scores README, with the pre-README fallback")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- THE CHUNKED PRODUCT: `score` over a `--rows` split rollouts product --------------------
    #
    # The shape the M5 full-scale run has and the shape the end-anchored `(\S+)$` could not match:
    # ONE `- rollouts:` line naming 22 `<stem>__rows<a>-<b>.jsonl` chunks, comma-separated. The
    # reader returned None, `lid_rates` reported "does not name its rollouts file", and the
    # language-id column, `ex.lid`, `corp10.lid` and both classifier ceilings were all empty on a
    # product that was complete (runs/2026-09-23_ledger.md, M5 gap 1). The union is read through
    # `precompute.common.read_rollouts`, so the chunk invariants and the disjointness are checked
    # where they are defined rather than re-implemented here.
    tmp = Path(tempfile.mkdtemp(prefix="stats_ood_rollchunks_"))
    try:
        sd = tmp / "maemms/b/m/scores/s__vllm__tag"
        sd.mkdir(parents=True)
        rd = tmp / "maemms/b/m/rollouts"
        rd.mkdir(parents=True)
        stem, chunks = "s__vllm__tag", ["0-1", "2-3"]
        inv = {"maemm": "b/m", "base": "b", "set": "s", "engine": "vllm", "kind": "ood", "n": 2,
               "bo": 2, "seed": 1, "max_new": 64, "min_new": 0, "prompt": "p", "prompt_tokens": 3,
               "marker_pos": 1, "inject_layer": 42, "inject_coef": 1.0, "temperature": 1.0,
               "top_p": 1.0, "top_k": 0, "weight_sha256": "0" * 64, "score_max_length": 95}
        for ci, spec in enumerate(chunks):
            rows_here = [2 * ci, 2 * ci + 1]
            (rd / f"{stem}{C.ROWS_MARK}{spec}.jsonl").write_text("\n".join(
                json.dumps({"row": r, "k": k, "text": f"t{r}{k}"})
                for r in rows_here for k in range(2)) + "\n")
            (rd / f"{stem}{C.ROWS_MARK}{spec}.summary.json").write_text(
                json.dumps({**inv, "rows": rows_here}))
        (sd / "README.md").write_text(
            "# scores\n\n## Inputs\n\n- rollouts: "
            + ", ".join(f"/vol/maemms/b/m/rollouts/{stem}{C.ROWS_MARK}{c}.jsonl" for c in chunks)
            + "\n- engine: vllm\n")
        vol = Vol("full", tmp, "uvx modal", False, True, offline=True)
        rels = rollouts_rels_from_readme(vol, "maemms/b/m/scores/s__vllm__tag")
        assert len(rels) == 2 and all(C.ROWS_MARK in r for r in rels), (
            f"the comma-separated chunk list on the `- rollouts:` line was not parsed: {rels!r}")
        texts = rollout_texts(vol, "b", "m", "s", "s__vllm__tag")
        assert texts == {0: ["t00", "t01"], 1: ["t10", "t11"],
                         2: ["t20", "t21"], 3: ["t30", "t31"]}, (
            f"the chunks of one rollouts product were not read as ONE product: {texts!r}")
        # A chunk of the same stem that the README does NOT name must stop the read, not join it:
        # a partial rerun left beside the scored chunks is two runs under one name.
        (rd / f"{stem}{C.ROWS_MARK}4-5.jsonl").write_text(
            json.dumps({"row": 4, "k": 0, "text": "stale"}) + "\n")
        (rd / f"{stem}{C.ROWS_MARK}4-5.summary.json").write_text(
            json.dumps({**inv, "rows": [4, 5]}))
        try:
            rollout_texts(vol, "b", "m", "s", "s__vllm__tag")
        except AssertionError as exc:
            assert "did not score" in str(exc), exc
        else:
            raise AssertionError(
                "a chunk file the scores README does not name joined the union silently")
        # A NAMED FILE THAT IS NOT THERE IS AN ABSENCE, NOT A DEFECT: the whole product is
        # withheld and the caller reports "lid not run". A partial read would print a rate over a
        # fraction of the targets as if it were the arm's rate -- which is a wrong number, where
        # an absent column is only an absent column.
        (rd / f"{stem}{C.ROWS_MARK}4-5.jsonl").unlink()
        (rd / f"{stem}{C.ROWS_MARK}4-5.summary.json").unlink()
        (rd / f"{stem}{C.ROWS_MARK}0-1.jsonl").unlink()
        assert rollout_texts(vol, "b", "m", "s", "s__vllm__tag") == {}, (
            "a chunked product missing one of its files was read PARTIALLY")
        ok.append("the chunked rollouts product is read as one product, and only its own chunks")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- spec §7 R4: the control pairing, in both readers ---------------------------------------
    # BOTH readers' assertions are exercised here because `results/ood.py` has NO self-test entry
    # point of its own -- it is one `main` command and nothing else. `results/selftest.py` does
    # carry three ood checks (`check_ood_arm_table`, `check_ood_lid_ranking`, `check_ood_scan_key`),
    # but that file is not this module's to edit and `check_ood_arm_table` still INJECTS `"asym"`
    # into `arm_rows`, so nothing over there covers change 3, change 5 or the recomputed bo8.
    import importlib  # noqa: PLC0415

    cfg = yaml.safe_load(CONFIG.read_text())
    R_ood = importlib.import_module("results.ood")

    # the two mu normalisers, held to one rule
    for probe in (None, "none", "null", "", "base/{base}/stats/mu.f32",
                  "/vol/checkpoints/data/qwen3.6-27b/whiten_mu.npy",
                  "vol/base/x/mu.f32", "/base/x/mu.f32", "unknown"):
        assert resolve_mu(probe, BASE) == R_ood.C_resolve_mu(probe, BASE), (
            f"stats_ood.resolve_mu and results/ood.C_resolve_mu disagree on {probe!r}: "
            f"{resolve_mu(probe, BASE)!r} vs {R_ood.C_resolve_mu(probe, BASE)!r}")

    PAIR = ("2026-09-18_rl-last16-lr5e-7", "2026-09-16_base-control")
    # (a) the intended pair passes on the REAL config, in both readers
    assert_control_paired(cfg, BASE, *PAIR)
    R_ood.assert_control_paired(cfg, BASE, *PAIR)
    declared = declared_mu(cfg, PAIR[1], BASE)
    assert declared == "checkpoints/data/qwen3.6-27b/whiten_mu.npy", (
        f"the base control no longer declares the 27B whiten_mu path: {declared!r}. The pairing "
        f"assertion is only a gate while config.yaml:255 carries it; `mu: null` would make every "
        f"pair below fail and this check is what says so out loud.")

    # (b) a MISMATCHED pair must raise, in both readers. The mutation is asserted to have
    # APPLIED first -- a gate that has never been red is not a gate.
    bad = {**cfg, "maemms": dict(cfg["maemms"])}
    ck = f"{BASE}/{PAIR[1]}"
    bad["maemms"][ck] = {**bad["maemms"][ck], "mu": "base/{base}/stats/mu.f32"}
    assert declared_mu(bad, PAIR[1], BASE) != declared_mu(cfg, PAIR[1], BASE), (
        "the mutation did not apply: the mismatched-pair check would pass vacuously")
    for name, fn in (("stats_ood", assert_control_paired), ("results/ood", R_ood.assert_control_paired)):
        try:
            fn(bad, BASE, *PAIR)
        except AssertionError as e:
            msg = str(e)
            assert PAIR[0] in msg and PAIR[1] in msg, f"{name}: the keys are not in the message: {msg}"
            assert "whiten_mu.npy" in msg and "stats/mu.f32" in msg, (
                f"{name}: both mu values must be in the message: {msg}")
        else:
            raise AssertionError(f"{name}.assert_control_paired accepted a control at another mean")

    # (c) the REAL old primary against the REAL control is a mismatch, and stays one
    try:
        assert_control_paired(cfg, BASE, "2026-09-10_rl-8x2048-full", PAIR[1])
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the old primary declares stats/mu.f32 and the control declares whiten_mu; the "
            "pairing assertion must refuse it")

    # (d) a checkpoint with no `mu:` key at all is refused by input_mu, not defaulted
    gap = {**cfg, "maemms": dict(cfg["maemms"])}
    gap["maemms"][ck] = {k: v for k, v in gap["maemms"][ck].items() if k != "mu"}
    assert "mu" not in gap["maemms"][ck], "the mutation did not apply"
    try:
        declared_mu(gap, PAIR[1], BASE)
    except AssertionError as e:
        assert "no `mu:` key" in str(e), str(e)
    else:
        raise AssertionError("a `maemms:` entry with no mu: key must be refused, not defaulted")
    # (e) and `build_tables` itself must refuse it, BEFORE any volume read -- the helper being
    # correct is worth nothing if the driver never calls it.
    tmp_r4 = Path(tempfile.mkdtemp(prefix="stats_ood_r4_"))
    try:
        vol_r4 = Vol("full", tmp_r4, "uvx modal", False, True, offline=True)
        try:
            build_tables(vol_r4, cfg, tmp_r4 / "out", {
                "base": BASE, "set": OOD_SET, "maemm": "2026-09-10_rl-8x2048-full",
                "control": PAIR[1], "stem": f"{OOD_SET}__vllm", "lid": False, "lid_model": None,
            })
        except AssertionError as e:
            assert "CONTROL NOT PAIRED" in str(e), (
                f"build_tables failed for another reason, so the pairing is not what it checked "
                f"first: {e}")
        else:
            raise AssertionError("build_tables tabulated a control declared at another mean")
    finally:
        shutil.rmtree(tmp_r4, ignore_errors=True)
    ok.append("control pairing on declared mu (both readers; passing pair, mutated pair, real "
              "old primary, missing key, and build_tables refusing before any fetch)")

    # --- panel c's bo8, recomputed from cos_centred.f16 against a hand-worked number -------------
    # Spec §2 names `bo_c_8` and NO such column exists anywhere in evals/faithfulness/: `score` stores the
    # centred ladder at k = 64 only. `results/ood.centred_bo_k` therefore recomputes it from the
    # array through `results.common.bo_unbiased`, and this is the only test of that path.
    tmp = Path(tempfile.mkdtemp(prefix="stats_ood_bo8_"))
    try:
        rel = "maemms/b/m/scores/s__vllm"
        sd = tmp / rel
        sd.mkdir(parents=True)
        n_t, n_k, width = 3, 10, 4
        arr = np.full((n_t, n_k, width), np.nan, dtype=np.float16)
        # row 0: all 10 rollouts finite, per-rollout best (r+1)/32
        for r in range(10):
            arr[0, r, 1] = np.float16((r + 1) / 32)
            arr[0, r, 2] = np.float16((r + 1) / 32 - 1 / 32)   # not the max, so the nanmax matters
        # row 1: rollouts 0 and 1 have NO kept token at all -> 8 finite draws
        for r in range(2, 10):
            arr[1, r, 1] = np.float16((r - 1) / 32)
        # row 2: only 7 finite draws -> NO bo8 at all, never a clamped k
        for r in range(7):
            arr[2, r, 1] = np.float16((r + 1) / 32)
        arr.tofile(sd / "cos_centred.f16")
        # a DECOY raw cosine in the same directory: reading `cos.f16` would give 0.9 everywhere
        np.full((n_t, n_k, width), np.float16(0.9), dtype=np.float16).tofile(sd / "cos.f16")
        (sd / "index.json").write_text(json.dumps({
            "cos_centred.f16": {"shape": [n_t, n_k, width]},
            "cos.f16": {"shape": [n_t, n_k, width]},
        }))
        vol_r = R_ood.R.Vol("", tmp, offline=True)
        src = R_ood.R.Source(maemm="b/m", base="b", engine="vllm", run_tag="",
                             scores_rel=rel, rollouts_rel="", role="primary",
                             rows_meta={"rows": [10, 11, 12], "n": n_k})
        got, note = R_ood.centred_bo_k(vol_r, src, 8)

        # HAND-WORKED. bo8 of n = 10 draws is sum_i x_(i) C(i-1,7)/C(10,8); C(10,8) = 45 and the
        # only non-zero weights are i = 8, 9, 10 with C(7,7) = 1, C(8,7) = 8, C(9,7) = 36:
        #   (1*8/32 + 8*9/32 + 36*10/32) / 45 = 440/1440
        want0 = 440 / 1440
        assert abs(got[10] - want0) < 1e-6, f"row 0: {got[10]!r} != {want0!r} ({note})"
        # row 1: the 2 NaN draws are DROPPED, so it is bo8 of 8 draws = the max, 8/32.
        assert abs(got[11] - 8 / 32) < 1e-6, f"row 1: {got[11]!r} != 0.25 ({note})"
        # and dropping is not filling: filling the 2 NaNs with 0 gives 350/1440, a different number
        filled = np.concatenate([np.zeros(2), np.arange(1, 9) / 32])
        assert abs(float(R_ood.R.bo_unbiased(filled, 8)) - 350 / 1440) < 1e-9
        assert abs(got[11] - 350 / 1440) > 1e-3, (
            "the NaN draws were FILLED rather than dropped: the bo8 is the 10-draw estimator's")
        # row 2 has 7 finite draws: absent, never clamped to bo7
        assert 12 not in got, f"a row with 7 finite draws must carry no bo8, got {got.get(12)!r}"
        assert "fewer than 8 finite draws" in note, note
        # the decoy: reading cos.f16 would have made every row 0.9
        assert max(got.values()) < 0.4, f"centred_bo_k read the wrong array: {got!r}"

        # a source with no centred cosine at all is a skip WITH A REASON, not an empty column
        sd2 = tmp / "maemms/b/m/scores/none"
        sd2.mkdir(parents=True)
        (sd2 / "index.json").write_text(json.dumps({"cos.f16": {"shape": [1, 1, 1]}}))
        got2, note2 = R_ood.centred_bo_k(
            vol_r, R_ood.R.Source(maemm="b/m", base="b", engine="vllm", run_tag="",
                                  scores_rel="maemms/b/m/scores/none", rollouts_rel=""), 8)
        assert got2 == {} and "not on the volume" in note2, (got2, note2)
        # AND IT MUST REACH THE TABLE. A correct estimator that no column carries is the
        # "silently absent" failure spec §2 is about, so `arm_rows` is called with the maps and
        # the record is read back. `a_none` has no bo8 for any of its rows and must carry None --
        # which `R.num` prints as an em dash -- rather than a zero or a partial mean.
        ids8, pt8, top8 = [], {}, {}
        bo8_map, ctrl8_map = {}, {}
        for j, arm in enumerate(("a_has", "a_none")):
            for i in range(4):
                rw = 10 * j + i
                ids8.append({"row": rw, "arm": arm, "family": "lang", "doc": 900 + rw})
                pt8[rw] = {"bo_64": 0.60, "bo_c_64": 0.70, "bo_a_64": 0.65}
                top8[(rw, 1.0)] = 0.50
                if arm == "a_has":
                    bo8_map[rw] = 0.40 + 0.01 * i      # mean 0.415
                    ctrl8_map[rw] = 0.10 + 0.01 * i    # mean 0.115
        src8 = R_ood.R.Source(maemm="m", base="b", engine="vllm", run_tag="",
                              scores_rel="", rollouts_rel="")
        src8.per_target = pt8
        recs8, _ = R_ood.arm_rows(
            ids8, src8, {"a_has": top8, "a_none": top8}, {"a_has": 1.0, "a_none": 1.0}, "centred",
            lambda d: (float(np.mean(d)), float(np.mean(d)) - 0.01, float(np.mean(d)) + 0.01),
            outcome, None, bo8=bo8_map, control_bo8=ctrl8_map)
        by8 = {r["arm"]: r for r in recs8}
        assert abs(by8["a_has"]["bo8_centred"] - 0.415) < 1e-9, by8["a_has"]
        assert abs(by8["a_has"]["control_bo8"] - 0.115) < 1e-9, by8["a_has"]
        assert by8["a_none"]["bo8_centred"] is None and by8["a_none"]["control_bo8"] is None, (
            "an arm with no recomputed bo8 must carry None (an em dash), never a zero or a "
            "partial mean")
        assert R_ood.R.num(by8["a_none"]["bo8_centred"]) == "—"
        ok.append("panel c's bo8 recomputed from cos_centred.f16 (hand-worked 440/1440, NaN "
                  "dropped not filled, short row absent, absent array named) and carried into "
                  "arm_rows' record")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- results/ood.py reads `centred`, and nothing reads `asym` by omission --------------------
    import inspect  # noqa: PLC0415

    body = inspect.getsource(R_ood.main)
    assert '"asym"' not in body, (
        "results/ood.main still passes the literal \"asym\": the arms table and the language-id "
        "column would be read on `cos_asym`, which spec §2 says is printed nowhere")
    assert body.count('"centred"') >= 2, "both call sites must pass `centred`"
    assert inspect.signature(R_ood.lid_rates).parameters["which"].default == "centred", (
        "lid_rates still defaults to `asym`, so a caller that omits `which` reads the wrong array")
    assert R_ood.BO64["centred"] == "bo_c_64" and R_ood.SCORE_ARRAY["centred"] == "cos_centred.f16"
    assert "bo_a_64" in R_ood.BO64.values(), (
        "the asym column map is kept on purpose -- cos_asym stays on disk, read by no driver")
    ok.append("results/ood reads `centred` at both call sites and by lid_rates' default")

    # --- the shared classifiers ----------------------------------------------------------------
    assert C.code_like("def f(x):\n    return {1: 2};\nimport os\n")
    assert not C.code_like("Dnes je krasne pocasi a jdu ven do parku.")
    assert C.script_fraction("hello", "Latin") == 1.0
    ok.append("code_like / script_fraction (common.py, one definition for both layers)")

    for name in ok:
        console.print(f"[green]ok[/green] {name}")
    console.print(f"[bold green]{len(ok)}/{len(ok)} stats_ood selfchecks passed[/bold green]")


if __name__ == "__main__":
    app()
