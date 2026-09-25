#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6"]
# ///
"""The four-arm CASE STUDY page: the same features seen through C16, M-top16, M-cos16 and M-jac16.

    cd <repo>
    uv run results/autointerp_cases.py --run 2026-09-24_autointerp-512 \\
      --out ../../<repo>/evals/2026-09-24_autointerp-cases.md

ONE Markdown file, written for a reader who wants to see WHAT THE FOUR SELECTION ARMS ACTUALLY
SHOW, not what they score on average. `results/autointerp.py` has the block-level contrast with its
paired bootstrap and `results/feature_page.py` has the full 512-feature debugging page; this is
neither. It is 16 features chosen to populate a 2 x 2 x 2 design -- Exemplifier wins / loses,
MAEM cosine high / low, corpus cosine high / low -- with every arm's explainer input and output
side by side, and per-arm diversity statistics computed with THE SELECTION RULES' OWN FUNCTIONS.

WHY THE SELECTION RULES' OWN FUNCTIONS AND NOT A REIMPLEMENTATION. The mean pairwise Jaccard
printed per arm is `autointerp.build.jaccard` over `autointerp.build.content_words`, which IS the
distance `M-jac16` maximises the minimum of; the mean pairwise residual cosine is
`autointerp.build.cosine_distances` over `autointerp.build._centred_residuals`, which IS
`M-cos16`'s. A second tokenisation or a second centring would give this page two numbers that look
like the arms' own and are not, which is exactly the trap `build.content_words`'s docstring
records. Nothing here re-derives a selection; the shown sets are read from the build.

PER-FEATURE DIFFERENCES CARRY NO INTERVAL, and the page says so where it prints one. The margin
that puts a feature in the `win` or `lose` half is two stored balanced accuracies subtracted, on
twenty positives judged once. It is a SELECTION KEY for a case study, never evidence about a
feature (spec section 3: no anecdotal-wins subsection).

WHAT IS READ, and from where:

  * the scores, explanations and the build directory the run read -- `results/feature_page.py`'s
    loaders, unchanged, so this page and the debugging page cannot disagree about a number;
  * the shown examples and their provenance -- the build's `<feature>.jsonl`, whose `block` field
    IS the explainer's user message (`autointerp/run.py:1205` passes it as `user`). With
    `--verify-cache N` that identity is CHECKED against the run's own call cache by rebuilding the
    request body and its sha256 key, so "verbatim" is a tested claim and not an assertion;
  * the Exemplifier's per-rollout cosine to the feature's SAE direction -- `cos_centred.f16` of the
    rl-final `score` product, through `results.common.best_per_rollout`, the same estimator
    `results/corpus_search.exemplifier_bok` reads it with;
  * the residuals `M-cos16` selects on -- `best_act.f16` of the same product, memory-mapped;
  * the corpus-side cosine -- `results/corpus_search.read_top1` on the centred 10M scan, plus a
    second pass over the same `topk.jsonl` for the per-window join, cross-checked against it;
  * the per-band recall -- `results/autointerp.load_band_index` joined to the scorer's
    `batches.jsonl`, the same join `results/autointerp.band_rows` makes for its cells.

THE MODEL-GENERATED TEXT IS ON THIS PAGE AND NOWHERE ELSE, the same rule `feature_page.py` states:
explanations and example blocks are the thing being shown, and they do not reach a log entry, a
commit message or `cells.csv`.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autointerp.build as B  # noqa: E402
import autointerp.sae_self as SS  # noqa: E402
import results.autointerp as A  # noqa: E402
import results.common as R  # noqa: E402
import results.corpus_search as CS  # noqa: E402
import results.feature_page as FP  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The four arms this page is about, in the order they are printed, under the names the SPEC uses.
# The run's `scores.jsonl` calls the top-16 rollout arm `M` (that is `ARM_SPECS`'s key); the paper
# and this page call it `M-top16`, because `M` beside `M-cos16` reads as a family and not as an
# arm. The mapping is printed on the page rather than left for a reader to infer.
ARMS = ("C16", "M", "M-jac16", "M-cos16")
DISPLAY = {"C16": "C16", "M": "M-top16", "M-jac16": "M-jac16", "M-cos16": "M-cos16"}
# The contrast the `win` / `lose` axis is: the Exemplifier's top-16 against the corpus arm, on
# detection, per feature, under the DROPPED convention (a feature enters only if BOTH arms have a
# balanced accuracy -- a refusal leaves no score row and is never imputed at chance).
CASE_ARM, REF_ARM, MAIN_SCORER = "M", "C16", "detection"
SCORERS = ("detection", "fuzzing")
# The corpus size of the scan ladder this page reads. The scan snapshots its running top-k at
# every size boundary, so 10 is the whole `train_parity_10m` corpus and 1.25 / 2.5 / 5 are its
# nested prefixes. A page that read a prefix would be reporting a smaller search.
SCAN_SIZE = 10.0
# The `score` product whose rollout cosines and residuals the M arms are measured with. It is the
# run tag, not the whole path: the rest is `precompute/common.scores_dir`'s layout, resolved from
# the build's own `maem` / `set` / `engine` so that this page cannot be pointed at another
# checkpoint's numbers by a stale constant.
SCORE_TAG = "paper0923"
# Activation bands, lowest first, as `results/autointerp.BAND_SLOT` spells them.
BANDS = ("q0", "q1", "q2", "q3", "top")
PLACES = FP.PLACES


# ---------------------------------------------------------------------------------------------
# the block: one stored string, n example windows
# ---------------------------------------------------------------------------------------------


EX_RE = re.compile(r"(?m)^Example (\d+):  ")


@dataclass
class Shown:
    """One example as the explainer saw it, split back out of the arm's stored `block`."""

    marked: str       # exactly as it appears in the block, `<<...>>` marks included
    plain: str        # the same text with the marks removed -- what `content_words` is taken over
    acts_line: str    # the `Activations: ...` line, or "" when the block carried none
    meta: dict        # the build's own row for this example (src, k / doc+start, max_act, ...)

    @property
    def n_folds(self) -> int:
        """Newlines INSIDE the window and its `Activations:` line: what one-lining them costs.

        Both can carry them. A corpus window contains newline tokens like any other token, and a
        marked newline token appears in the `Activations:` pair list as itself.
        """
        return self.marked.count("\n") + self.acts_line.count("\n")


def split_block(block: str, examples: list[dict]) -> list[Shown]:
    """`exemplar_block`'s output back into its examples, with the `Activations:` line separated.

    THE SPLIT HAS TO BE EXACT, because the Jaccard printed per arm is taken over these texts and
    the `Activations:` line repeats every marked token: leaving it in would add the marked tokens a
    second time to every example's content-word set and inflate the overlap. So the line is removed
    on the build's own evidence -- `exemplar_block` emits it iff the example had a marked token,
    which the stored row records as `n_marked` -- and the two are asserted to agree rather than the
    line being guessed at by its prefix.

    A corpus window CONTAINS newlines (they are ordinary tokens), so an example is not a line and
    cannot be found by splitting on one. The anchor is `^Example <n>:  ` at a line start with the
    two-space gap `exemplar_block` writes, taken only when `<n>` is the NEXT expected index -- a
    window whose own text carries such a line keeps it, and a block with one boundary too many is
    refused by name rather than being read as a different set of examples.
    """
    # SEQUENTIAL boundaries only. A corpus window is ordinary text and can contain a line that
    # reads `Example 9:  ` -- `exemplar_block` numbers its own 1..n in order, so a match whose
    # number is not the next expected one belongs to a window's text and is left in it. What the
    # split may NOT do is stop early, so the line that would have been example n+1 is asserted
    # absent: a block holding more examples than the build stored is a join failure, and reading
    # the extra one as part of example n's text would silently change every statistic on this page.
    starts, want = [], 1
    for m in EX_RE.finditer(block):
        if int(m.group(1)) == want:
            starts.append(m)
            want += 1
    assert len(starts) == len(examples), (
        f"the block carries {len(starts)} sequential `Example n:  ` boundaries and the build "
        f"stored {len(examples)} examples")
    assert not re.search(rf"(?m)^Example {len(examples) + 1}:  ", block), (
        f"the block carries an `Example {len(examples) + 1}:  ` line and the build stored only "
        f"{len(examples)} examples: this is not the block that belongs to these rows")
    chunks = [block[m.end():(starts[i + 1].start() if i + 1 < len(starts) else len(block))]
              for i, m in enumerate(starts)]
    assert not starts or starts[0].start() == 0, (
        f"the block does not start with `Example 1:  `: it starts {block[:40]!r}")
    out: list[Shown] = []
    for j, (chunk, meta) in enumerate(zip(chunks, examples, strict=True)):
        # `"\n".join` put a newline between examples; it belongs to the separator, not to the text.
        if j < len(chunks) - 1:
            assert chunk.endswith("\n"), f"example {j + 1} does not end at a line break"
            chunk = chunk[:-1]
        acts = ""
        if int(meta.get("n_marked") or 0) > 0:
            # THE LAST `\nActivations: `, not the last LINE. A marked token can BE a newline --
            # `exemplar_block` writes the token piece into the pair list verbatim -- so the
            # `Activations:` line is not always one line, and `rpartition("\n")` returned the tail
            # of the pair list on the first corpus feature this was run on.
            i = chunk.rfind("\nActivations: ")
            assert i >= 0, (
                f"example {j + 1} has {meta.get('n_marked')} marked tokens, so `exemplar_block` "
                f"wrote an `Activations:` line, and the chunk carries none: {chunk[-60:]!r}")
            chunk, acts = chunk[:i], chunk[i + 1:]
        # `marked_text` inserts only the two literal strings `<<` and `>>`, so deleting them
        # recovers `render_example`'s `text` -- which is what `content_words` was taken over when
        # `M-jac16` selected. It is exact unless the SOURCE text contained `<<` or `>>` itself;
        # the count of each is reported per arm so that case is visible rather than assumed away.
        out.append(Shown(marked=chunk, plain=chunk.replace("<<", "").replace(">>", ""),
                         acts_line=acts, meta=meta))
    return out


def one_line(text: str) -> str:
    """A window on ONE line: an interior newline printed as the two characters `\\n`.

    The only departure from the byte-for-byte block anywhere on this page, made because a corpus
    window with six line breaks in it is unreadable beside a rollout that has none. It is
    reversible, it is counted per arm, and the page says so above every block.
    """
    return text.replace("\\", "\\\\").replace("\n", "\\n")


# ---------------------------------------------------------------------------------------------
# the per-arm statistics -- every one of them through the selection rule's own function
# ---------------------------------------------------------------------------------------------


def mean_pairwise_jaccard(shown: list[Shown]) -> float | None:
    """`M-jac16`'s own distance, averaged over the pairs of a SHOWN set (not of the pool).

    `build.jaccard(build.content_words(a), build.content_words(b))` -- the similarity, so a higher
    number is a less diverse set. `M-jac16` maximises the minimum of `1 - ` this over its picks, so
    the arm is expected to sit BELOW `M-top16` here and the number is the check on that, not a
    restatement of it.
    """
    sets = [B.content_words(s.plain) for s in shown]
    if len(sets) < 2:
        return None
    vals = [B.jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals))


def mean_pairwise_residual_cosine(shown: list[Shown], best_act, row_ix: int | None, mu):
    """`M-cos16`'s own distance, averaged over the pairs: (mean cosine, n zero-norm rows) or None.

    None for a set with a CORPUS window in it. `best_act.f16` is a per-ROLLOUT product -- the
    read-layer residual at each rollout's cosine argmax (`precompute/score.py:624`) -- and there is
    no such vector for a corpus window anywhere on the volume, so the arm's own geometry simply
    cannot be reported for `C16` and the page says "rollouts only" rather than substituting the
    window's SAE activation and calling it a cosine.
    """
    if any(str(s.meta.get("src")) != "rollout" for s in shown) or row_ix is None:
        return None
    pool = [{"k": int(s.meta["k"])} for s in shown]
    vecs = B._centred_residuals(pool, best_act, row_ix, mu)
    if len(vecs) < 2:
        return None
    n_zero = int((np.linalg.norm(np.asarray(vecs, dtype=np.float64), axis=1) == 0).sum())
    d = B.cosine_distances(vecs)
    iu = np.triu_indices(len(vecs), k=1)
    return float(np.mean(1.0 - d[iu])), n_zero


def act_range(shown: list[Shown]) -> dict:
    """The shown set's activation span: min/max `max_act`, and the same as a fraction of the peak.

    `peak_frac` is the build's own field (this block's peak over the feature's corpus peak), not
    recomputed here, so a rollout above the corpus peak reads > 1 exactly as it does in the build's
    `n_shown_exceeding_corpus_peak` count.
    """
    acts = [float(s.meta.get("max_act") or 0.0) for s in shown]
    fr = [s.meta.get("peak_frac") for s in shown]
    fr = [float(x) for x in fr if x is not None]
    return {"act_min": min(acts) if acts else None, "act_max": max(acts) if acts else None,
            "frac_min": min(fr) if fr else None, "frac_max": max(fr) if fr else None}


# ---------------------------------------------------------------------------------------------
# the volume products this page adds to `feature_page`'s
# ---------------------------------------------------------------------------------------------


@dataclass
class Geometry:
    """Everything read out of the `score` product and the corpus scan, per heldout row."""

    scores_rel: str = ""
    scan_dir: str = ""
    row_ix: dict[int, int] = field(default_factory=dict)      # heldout row -> index in the product
    best_cos: dict[int, np.ndarray] = field(default_factory=dict)   # heldout row -> [n] per rollout
    best_act: object = None                                   # memmap [N, n, d] or None
    mu: object = None
    mu_path: str = ""
    corpus_top1: dict[int, float] = field(default_factory=dict)     # heldout row -> top-1 cosine
    window_cos: dict[int, dict[tuple[int, int], float]] = field(default_factory=dict)
    centred_sided: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def resolve_scores_rel(vol: R.Vol, maem: str, set_name: str, engine: str, tag: str) -> str:
    """`precompute/common.scores_dir`'s two spellings, the canonical one first.

    The writer's rename put the tag before the engine (`<set>__<tag>__<engine>`) and the products
    already on the volume carry the older `<set>__<engine>__<tag>`. Both are tried HERE rather than
    a constant being edited when a future run writes the other one, and the one that exists is
    reported on the page.
    """
    stem = f"maems/{maem}/scores"
    cands = [f"{stem}/{set_name}__{tag}__{engine}", f"{stem}/{set_name}__{engine}__{tag}"]
    for rel in cands:
        if vol.exists(f"{rel}/index.json"):
            return rel
    return ""


def memmap(path: Path, dtype: str, shape) -> np.ndarray:
    """`best_act.f16` WITHOUT reading it: 671 MB at the paper's scale, and 16 rows are wanted.

    `results.common.read_array` is `np.fromfile`, which would put the whole product in RAM for the
    sake of sixteen [64, 5120] slices. The dtype and the shape still come from the product's own
    `index.json`, so this is the same contract read a different way.
    """
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(int(v) for v in shape))


def load_geometry(vol: R.Vol, build: dict, tag: str, scan_dir: str, scan_size: float) -> Geometry:
    """The `score` product's cosines and residuals, and the corpus scan's top-k, by heldout row."""
    g = Geometry(scan_dir=scan_dir)
    maem, set_name = str(build.get("maem") or ""), str(build.get("set") or "")
    base, engine = str(build.get("base") or ""), str(build.get("engine") or "hf")
    g.scores_rel = resolve_scores_rel(vol, maem, set_name, engine, tag)
    assert g.scores_rel, (
        f"no `score` product for maem {maem!r}, set {set_name!r}, engine {engine!r}, tag {tag!r} "
        f"under `maems/{maem}/scores/` in either spelling: the M arms' cosines and the residuals "
        f"`M-cos16` selected on live there, and nothing on this page recomputes them")
    rows = vol.json(f"{g.scores_rel}/rows.json") or {}
    idx = vol.json(f"{g.scores_rel}/index.json") or {}
    g.row_ix = {int(r): i for i, r in enumerate(rows.get("rows") or [])}

    shape = tuple(int(v) for v in (idx.get("cos_centred.f16") or {}).get("shape") or ())
    assert len(shape) == 3, (
        f"{g.scores_rel}/index.json carries no `cos_centred.f16` shape: the centred rollout cosine "
        f"is the quantity the MAEM-cosine axis splits on and there is no second source for it")
    cos = vol.array(f"{g.scores_rel}/cos_centred.f16", "float16", shape)
    assert cos is not None, f"{g.scores_rel}/cos_centred.f16 is not in the mirror or on the volume"
    for row, i in g.row_ix.items():
        # NaN for a rollout with no kept centred token, exactly as `exemplifier_bok` reads it:
        # `score` DROPS such a rollout from its centred aggregates rather than scoring it -1, so
        # filling it here would invent a draw.
        g.best_cos[row] = R.best_per_rollout(cos[i].astype(np.float32), empty=float("nan"))

    for rec in vol.jsonl(f"{g.scores_rel}/per_target.jsonl") or []:
        g.centred_sided[int(rec["row"])] = int(rec.get("centred_sided") or 0)

    ba = idx.get("best_act.f16") or {}
    p = vol.get(f"{g.scores_rel}/best_act.f16") if ba.get("shape") else None
    if p is None:
        g.notes.append(
            f"`{g.scores_rel}/best_act.f16` is not in the mirror ({(ba.get('bytes') or 0) / 1e6:.0f}"
            f" MB): the per-arm mean pairwise RESIDUAL cosine -- `M-cos16`'s own distance -- is "
            f"blank below. Every other number is unaffected.")
    else:
        g.best_act = memmap(p, "float16", ba["shape"])
    mu_rel = str(rows.get("mu") or "").removeprefix("/vol/")
    cfg_mu = str(((R.load_config().get("bases") or {}).get(base) or {}).get("whiten_mu") or "")
    assert not cfg_mu or cfg_mu.removeprefix("/vol/") == mu_rel, (
        f"the score product centred on {mu_rel!r} and config.yaml's base {base!r} says "
        f"{cfg_mu!r}: `M-cos16`'s residuals were centred on the product's mean and this page must "
        f"subtract that same one, so the disagreement is refused rather than resolved silently")
    if mu_rel and g.best_act is not None:
        mp = vol.get(mu_rel)
        assert mp is not None, (
            f"the score product centred on `{mu_rel}` and it is not in the mirror: raw layer-42 "
            f"residuals have mutual cosines in a narrow band near 1, so an uncentred number here "
            f"would not be `M-cos16`'s distance at all")
        g.mu, g.mu_path = np.asarray(np.load(mp), dtype=np.float32).reshape(-1), mu_rel

    # The corpus side. `read_top1` is the reused loader and gives the headline top-1 per row; the
    # second pass gives the per-WINDOW cosines the C16 join needs, and the two are checked against
    # each other so a change in either reader shows up as a failure and not as a quiet difference.
    t = CS.read_top1(vol, base, scan_dir, set_name, family="sae", apply_exclusions=False)
    assert scan_size in t.by_size, (
        f"scan {scan_dir} has sizes {t.sizes} and not {scan_size}: the page reports the FULL "
        f"corpus search, and a prefix would be a smaller one under the same label")
    g.corpus_top1 = dict(t.by_size[scan_size])
    recs = vol.jsonl(f"base/{base}/scan/{scan_dir}/topk.jsonl") or []
    for r in recs:
        if r.get("set", set_name) != set_name or float(r["size"]) != scan_size:
            continue
        if (r.get("family") or "") != "sae":
            continue
        row = int(r.get("set_row", r["row"]))
        g.window_cos[row] = {(int(e[0]), int(e[1])): float(e[3]) for e in (r.get("top") or [])}
        if r.get("top"):
            assert abs(float(r["top"][0][3]) - g.corpus_top1[row]) < 1e-9, (
                f"row {row}: `read_top1` says {g.corpus_top1[row]} and this pass says "
                f"{r['top'][0][3]} for the top-1 cosine at {scan_size}M -- two readers of one file "
                f"disagree")
    topk = max((len(v) for v in g.window_cos.values()), default=0)
    g.notes.append(
        f"corpus side: `base/{base}/scan/{scan_dir}/topk.jsonl` at size {scan_size:g}M, "
        f"{len(g.corpus_top1)} rows of set `{set_name}` family `sae`, top-{topk} windows per row")
    return g


def band_tpr(vol: R.Vol, run_dir: str, build_rel: str, ri: FP.RunIn, feats: list[int],
             arms: list[str], scorer: str) -> tuple[dict, list[str]]:
    """{(feature, arm, band): (n correct, n items)} for the POSITIVE half, and what was dropped.

    The same join `results/autointerp.band_rows` makes for its cells and by the same rules: the
    band is the BUILD's (`load_band_index`), the answer is the scorer's `batches.jsonl`, an
    unparsed batch is dropped everywhere, and the draw comes from the score row rather than from
    the arm's name. What differs is only the aggregation -- per feature here, bootstrapped over
    features there -- so a band column on this page and a `ai.*.tpr.b*` cell are the same events
    counted at two grains.
    """
    notes: list[str] = []
    idx, absent = A.load_band_index(vol, build_rel, feats)
    if absent:
        notes.append(f"{len(absent)} of {len(feats)} build row files are absent, so those features "
                     f"carry no per-band recall: {', '.join(str(a) for a in absent[:5])}")
    batches = vol.jsonl(f"runs/{run_dir}/{scorer}/batches.jsonl")
    if batches is None:
        return {}, [*notes, f"`runs/{run_dir}/{scorer}/batches.jsonl` is not there: no per-band "
                            f"recall on this page"]
    want, acc = set(feats), {}
    n_unparsed = n_unjoined = 0
    for b in batches:
        feat, arm = int(b["feature"]), str(b["arm"])
        if feat not in want or arm not in arms:
            continue
        if not b.get("parsed"):
            n_unparsed += 1
            continue
        row = ri.scores.get((arm, scorer, feat)) or {}
        items = (idx.get(feat) or {}).get("test2" if int(row.get("draw") or 1) == 2 else "test") or {}
        for i, lab, pred in zip(b["items"], b["labels"], b["preds"], strict=True):
            hit = items.get(int(i))
            if hit is None:
                n_unjoined += 1
                continue
            blab, band = hit
            assert blab == int(lab), (
                f"item {i} of feature {feat} is label {lab} in the run and {blab} in the build: "
                f"the build being joined is not the one this run scored")
            if int(lab) != 1 or band not in BANDS:
                continue
            tally = acc.setdefault((feat, arm, band), [0, 0])
            tally[1] += 1
            tally[0] += int(int(pred) == 1)
    notes.append(f"per-band recall from `runs/{run_dir}/{scorer}/batches.jsonl` joined to "
                 f"`{build_rel}`: {n_unparsed} unparsed batch(es) dropped, {n_unjoined} item(s) "
                 f"with no build row")
    return {k: tuple(v) for k, v in acc.items()}, notes


def verify_blocks(vol: R.Vol, run_dir: str, ri: FP.RunIn, pairs: list[tuple[int, str]],
                  model: str, max_tokens: int, shots: int) -> tuple[list[dict], list[str]]:
    """Is the stored `block` byte-for-byte the explainer's user message? Ask the run's own cache.

    `run.Cache.key` hashes the job key AND the request body, and `run.Claude.params` builds that
    body from the system prompt, Delphi's few-shot turns and the block. So rebuilding both here and
    finding the file on the volume proves the identity for that (feature, arm) -- and a changed
    block, a changed prompt or a different model would miss. `Claude.params` touches only `model`
    and `cache_prompt`, so it is called against a stand-in rather than an API client: nothing here
    can reach the network except the volume.

    A MISS IS NOT A FAILURE and is reported as its own outcome: `_submit` does not cache an empty
    answer unless the job was already a retry, so a refused (feature, arm) legitimately has no file
    under its first-call key.
    """
    import autointerp.run as RUN

    stand_in = types.SimpleNamespace(model=model, cache_prompt=True)
    fewshot = RUN.explainer_fewshot(shots)
    out, notes = [], []
    for feat, arm in pairs:
        row = (ri.arms_of.get(feat) or {}).get(arm)
        if row is None:
            continue
        body = RUN.Claude.params(stand_in, RUN.DELPHI_EXPLAINER_SYSTEM, str(row.get("block") or ""),
                                 max_tokens, fewshot)
        key = RUN.Cache.key(f"explain|{feat}|{arm}", body)
        rel = f"runs/{run_dir}/cache/{key[:2]}/{key}.json"
        rec = vol.json(rel)
        e = ri.expl.get((arm, feat)) or {}
        out.append({"feature": feat, "arm": arm, "key": key, "hit": rec is not None,
                    "refused": bool(e.get("refused")), "rel": rel})
    hits = sum(1 for r in out if r["hit"])
    miss_ok = sum(1 for r in out if not r["hit"] and r["refused"])
    tail = ""
    if len(out) - hits:
        rest = len(out) - hits - miss_ok
        tail = (f"; {len(out) - hits} did not hit, of which {miss_ok} are refused calls, whose "
                f"empty answer `run._submit` deliberately does not cache"
                + (f", and {rest} are unexplained" if rest else ""))
    notes.append(
        f"call-cache verification: {hits}/{len(out)} rebuilt request bodies hash to a cache file "
        f"that exists under `runs/{run_dir}/cache/`, which proves the printed block is byte for "
        f"byte the user message that was sent{tail}")
    return out, notes


# ---------------------------------------------------------------------------------------------
# the 2 x 2 x 2 selection
# ---------------------------------------------------------------------------------------------


@dataclass
class Feature:
    """One candidate feature and the three quantities the design splits on."""

    feat: int
    row: int
    margin: float | None = None       # M-top16 minus C16, detection balanced accuracy
    m_cos: float | None = None        # mean centred cosine of the 16 rollouts M-top16 shows
    c_cos: float | None = None        # top-1 centred corpus cosine at the full corpus size
    eligible: bool = False            # has all three quantities and is not a tie
    explained: tuple[str, ...] = ()   # the arms with a non-refused, non-empty explanation
    why: str = ""

    @property
    def all_explained(self) -> bool:
        return len(self.explained) == len(ARMS)


def collect(ri: FP.RunIn, g: Geometry, feats: list[int]) -> list[Feature]:
    """One `Feature` per scored feature, with the three quantities and why it is or is not usable."""
    out: list[Feature] = []
    for f in feats:
        meta = ri.feats.get(f) or {}
        row = meta.get("row")
        fe = Feature(feat=f, row=int(row) if row is not None else -1)
        fe.margin = FP.detection_delta(ri, CASE_ARM, REF_ARM, f, MAIN_SCORER)
        m_row = (ri.arms_of.get(f) or {}).get(CASE_ARM) or {}
        ks = [int(e["k"]) for e in (m_row.get("examples") or []) if e.get("src") == "rollout"]
        best = g.best_cos.get(fe.row)
        if ks and best is not None:
            vals = [float(best[k]) for k in ks if k < len(best)]
            if vals and not any(math.isnan(v) for v in vals):
                fe.m_cos = float(np.mean(vals))
        fe.c_cos = g.corpus_top1.get(fe.row)
        missing = [n for n, v in (("detection margin", fe.margin), ("MAEM cosine", fe.m_cos),
                                  ("corpus cosine", fe.c_cos)) if v is None]
        # A margin of exactly zero is neither a win nor a loss and is not made into one by the
        # sign convention of a comparison operator: the feature is simply not a case for this axis.
        if fe.margin == 0.0:
            missing.append("a non-zero detection margin (the two arms tie)")
        # A missing explanation is a PREFERENCE and not a bar (the brief's word): a feature with
        # one refused arm still shows three arms' inputs and outputs side by side, and in a cell
        # that has no fully-explained feature at all it is better evidence than an empty cell.
        # `select` sorts fully-explained features first, so it is reached only as a last resort.
        fe.explained = tuple(a for a in ARMS
                             if ((ri.expl.get((a, f)) or {}).get("explanation") or "")
                             and not (ri.expl.get((a, f)) or {}).get("refused"))
        fe.eligible, fe.why = not missing, "; ".join(missing)
        out.append(fe)
    return out


def cell_of(fe: Feature, m_med: float, c_med: float) -> tuple[str, str, str]:
    return (("win" if (fe.margin or 0.0) > 0 else "lose"),
            ("mcos-hi" if (fe.m_cos or 0.0) >= m_med else "mcos-lo"),
            ("ccos-hi" if (fe.c_cos or 0.0) >= c_med else "ccos-lo"))


def select(cands: list[Feature], m_med: float, c_med: float, n_per_cell: int,
           margin: float) -> tuple[list[tuple[tuple[str, str, str], Feature]], list[dict]]:
    """`n_per_cell` features per cell by largest |margin|, relaxing the threshold CELL BY CELL.

    The rule and its relaxation are both recorded, because the eight cells are not equally
    populated and pretending otherwise would be the padding this refuses to do: the Exemplifier
    loses on most features, so the four `win` cells are thin and reach their quota only below the
    threshold. A cell that cannot be filled even at a threshold of zero is reported SHORT.
    """
    by_cell: dict[tuple[str, str, str], list[Feature]] = {}
    for fe in cands:
        by_cell.setdefault(cell_of(fe, m_med, c_med), []).append(fe)
    picked: list[tuple[tuple[str, str, str], Feature]] = []
    report: list[dict] = []
    for half in ("win", "lose"):
        for mc in ("mcos-hi", "mcos-lo"):
            for cc in ("ccos-hi", "ccos-lo"):
                cell = (half, mc, cc)
                # FULLY EXPLAINED FIRST, then by |margin|: the design wants a case a reader can
                # compare across all four arms, and only falls back to a partly-refused feature
                # when the cell holds nothing better.
                pool = sorted(by_cell.get(cell, []),
                              key=lambda x: (not x.all_explained, -abs(x.margin or 0.0)))
                at_thr = [x for x in pool if abs(x.margin or 0.0) >= margin and x.all_explained]
                take = at_thr[:n_per_cell]
                relaxed = False
                if len(take) < n_per_cell:
                    take, relaxed = pool[:n_per_cell], True
                picked += [(cell, x) for x in take]
                report.append({
                    "cell": cell, "n_at_threshold": len(at_thr), "n_total": len(pool),
                    "n_all_explained": sum(1 for x in pool if x.all_explained),
                    "taken": [x.feat for x in take], "relaxed": relaxed,
                    "min_abs_margin": min((abs(x.margin or 0.0) for x in take), default=None),
                    "short": len(take) < n_per_cell,
                    "partial": [x.feat for x in take if not x.all_explained],
                })
    return picked, report


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------


def num(v, places: int = PLACES) -> str:
    return R.num(v, places)


def signed(v, places: int = PLACES) -> str:
    return FP.signed(v, places)


def frac(t) -> str:
    """`k/n` and its rate, or an em dash: a band with no item is not a recall of zero."""
    if not t or not t[1]:
        return "—"
    return f"{t[0]}/{t[1]} ({t[0] / t[1]:.2f})"


def arm_stats(ri: FP.RunIn, g: Geometry, feat: int, arm: str) -> dict:
    """Everything this page says about ONE arm's shown set on ONE feature."""
    row = (ri.arms_of.get(feat) or {}).get(arm) or {}
    ex = list(row.get("examples") or [])
    shown = split_block(str(row.get("block") or ""), ex) if ex else []
    meta = ri.feats.get(feat) or {}
    hrow = int(meta.get("row") or -1)
    st = {"arm": arm, "n": len(shown), "shown": shown,
          "jaccard": mean_pairwise_jaccard(shown), **act_range(shown)}
    rc = mean_pairwise_residual_cosine(shown, g.best_act, g.row_ix.get(hrow), g.mu) \
        if g.best_act is not None else None
    st["res_cos"], st["res_zero"] = (rc if rc else (None, 0))
    best = g.best_cos.get(hrow)
    wc = g.window_cos.get(hrow) or {}
    probe, kind, n_joined = [], "", 0
    for s in shown:
        if str(s.meta.get("src")) == "rollout" and best is not None:
            k = int(s.meta["k"])
            v = float(best[k]) if k < len(best) else float("nan")
            probe.append(None if math.isnan(v) else v)
            kind = "per-rollout centred cosine (`cos_centred.f16`)"
        else:
            v = wc.get((int(s.meta.get("doc", -1)), int(s.meta.get("start", -1))))
            probe.append(v)
            n_joined += int(v is not None)
            kind = "per-window centred cosine, joined on (doc, start) into the scan's top-k"
    have = [v for v in probe if v is not None]
    st["probe_kind"] = kind
    # For a corpus arm the denominator is the JOIN's coverage: a shown window that is not in the
    # scan's top-k has no stored cosine, and a fraction below 1 here is that and not a defect.
    st["probe_joined"] = f"{len(have)}/{len(shown)}"
    st["probe"] = probe
    st["probe_min"] = min(have) if have else None
    st["probe_med"] = statistics.median(have) if have else None
    st["probe_max"] = max(have) if have else None
    st["folds"] = sum(s.n_folds for s in shown)
    st["marks_in_source"] = sum(s.plain.count("<<") + s.plain.count(">>") for s in shown)
    return st


def score_bits(ri: FP.RunIn, arm: str, feat: int, scorer: str) -> list[str]:
    r = ri.scores.get((arm, scorer, feat))
    if r is None:
        return ["—", "—", "—"]
    return [num(r.get("bal_acc")), num(r.get("tpr")), num(r.get("tnr"))]


def feature_block(ri: FP.RunIn, g: Geometry, fe: Feature, cell: tuple[str, str, str],
                  bands: dict, stats: dict[str, dict]) -> list[str]:
    """One feature: the header, the per-arm numbers, the example statistics, then the four arms."""
    meta = ri.feats.get(fe.feat) or {}
    out = [f"## Feature {fe.feat} — {'/'.join(cell)}", ""]
    out += ["- " + " · ".join([
        f"heldout row **{fe.row}**", f"stratum **{meta.get('stratum', '—')}**",
        f"corpus peak **{num(meta.get('corpus_peak'))}**", f"gate {num(meta.get('gate'))}",
        f"density {num(meta.get('density'), 6)}",
        f"detection margin M-top16 − C16 **{signed(fe.margin)}**",
        f"Exemplifier cosine (mean over M-top16's 16) **{num(fe.m_cos, 5)}**",
        f"corpus top-1 cosine at {SCAN_SIZE:g}M **{num(fe.c_cos, 5)}**",
    ]), ""]
    out += ["*The margin is two stored balanced accuracies subtracted, on twenty positives judged "
            "once. It is the selection key for this page and carries no interval.*", ""]

    hdr = ["arm", "det bal_acc", "det TPR", "det TNR", "fuzz bal_acc", "fuzz TPR", "fuzz TNR",
           *[f"TPR {b}" for b in BANDS]]
    rows = []
    for arm in ARMS:
        rows.append([DISPLAY[arm], *score_bits(ri, arm, fe.feat, "detection"),
                     *score_bits(ri, arm, fe.feat, "fuzzing"),
                     *[frac(bands.get((fe.feat, arm, b))) for b in BANDS]])
    out += ["### Scores", "",
            f"*Detection and fuzzing are never pooled. Chance is {A.CHANCE}. The per-band columns "
            f"are recall on the build's four equal-width bins of (0, corpus peak] plus the "
            f"top-beyond-shown fallback tier, over the items of parsed batches only.*", ""]
    out += FP.md_table(hdr, rows)

    out += ["### The shown sets", "",
            "*`mean pairwise Jaccard` is `M-jac16`'s own distance (content-word Jaccard, "
            "similarity not distance, so lower is more diverse); `mean pairwise residual cosine` "
            "is `M-cos16`'s, over the centred `best_act` residuals, which exist for rollouts only; "
            "`cosine to the SAE direction` is per rollout from `cos_centred.f16` and per corpus "
            "window from the scan's top-k where the window is in it.*", ""]
    rows = []
    for arm in ARMS:
        st = stats[arm]
        rows.append([
            DISPLAY[arm], st["n"], num(st["jaccard"], 4),
            num(st["res_cos"], 4) if st["res_cos"] is not None else "rollouts only",
            f"{num(st['probe_min'], 4)} / {num(st['probe_med'], 4)} / {num(st['probe_max'], 4)}",
            st["probe_joined"],
            f"{num(st['act_min'], 3)} – {num(st['act_max'], 3)}",
            f"{num(st['frac_min'], 3)} – {num(st['frac_max'], 3)}",
        ])
    out += FP.md_table(
        ["arm", "n", "mean pairwise Jaccard", "mean pairwise residual cosine",
         "cosine to SAE direction min / median / max", "cosine coverage",
         "max_act range", "fraction of corpus peak"], rows)

    for arm in ARMS:
        out += arm_section(ri, fe.feat, arm, stats[arm])
    return out


def arm_section(ri: FP.RunIn, feat: int, arm: str, st: dict) -> list[str]:
    """One arm on one feature: what it showed, then what the explainer wrote about it."""
    out = [f"### `{DISPLAY[arm]}` — feature {feat}", ""]
    src = "corpus windows" if any(s.meta.get("src") == "corpus" for s in st["shown"]) \
        else "Exemplifier rollouts"
    note = (f"{st['n']} {src}, verbatim from the build's `block` (the explainer's user message), "
            f"with {st['folds']} interior newline(s) printed as `\\n` so that one window is one "
            f"line")
    if st["marks_in_source"]:
        note += (f"; {st['marks_in_source']} literal `<<`/`>>` occurrence(s) are in the source text "
                 f"itself, so the unmarked text used for the Jaccard is approximate on this arm")
    out += [f"*{note}.*", ""]
    body = []
    for j, s in enumerate(st["shown"]):
        body.append(f"Example {j + 1}:  {one_line(s.marked)}")
        if s.acts_line:
            body.append(one_line(s.acts_line))
    text = "\n".join(body)
    out += FP.quote(text) if text else ["*(no block stored)*", ""]
    out += [""]
    e = ri.expl.get((arm, feat)) or {}
    if e.get("refused"):
        out += [f"Explanation: **REFUSED** (`stop_reason` `{e.get('stop_reason')}`).", ""]
    elif not e.get("explanation"):
        out += [f"Explanation: *empty* (`stop_reason` `{e.get('stop_reason')}`).", ""]
    else:
        out += ["Explanation (verbatim, `explain/explanations.jsonl`):", ""]
        out += FP.quote(str(e["explanation"])) + [""]
    return out


def render(ri: FP.RunIn, g: Geometry, picked, report, bands, stats, cells_note: dict,
           verify, notes: list[str], argv: list[str], vol: R.Vol) -> str:
    """The whole document, frontmatter included."""
    b = ri.build
    L = ["---", "privacy: private", "---", "",
         "# SAE autointerp case studies — the four example-selection arms", ""]
    L += [
        f"- generated {time.strftime('%Y-%m-%d %H:%M')} by `{' '.join(argv)}`",
        f"- run `runs/{ri.run_dir}`, build `{ri.build_rel}`, dictionary `{ri.sae}`, set "
        f"`{b.get('set', '—')}`, base `{b.get('base', '—')}`, maem `{b.get('maem', '—')}`, "
        f"judge `claude-sonnet-5`, N = {b.get('n_examples', '—')} examples per arm, gate "
        f"{b.get('gate', '—')}",
        f"- score product `{g.scores_rel}`; corpus scan `base/{b.get('base')}/scan/{g.scan_dir}`; "
        f"mirror `{vol.local}`",
        f"- {len(picked)} features in a 2 x 2 x 2 design, {cells_note['n_per_cell']} per cell; "
        f"every number below is read from a product, and each derived one says so",
        "- **a per-feature difference carries no interval.** The block-level contrast with its "
        "paired bootstrap is `results/autointerp.py`'s table; one feature is not evidence for a "
        "method.",
        "", "## What each arm is", "",
    ]
    L += arm_definitions(b, g)
    L += [f"## How the {len(picked)} features were chosen", ""]
    L += [
        f"Three binary axes, each split independently over the {cells_note['n_scored']} scored "
        f"features of this run:", "",
        f"1. **Exemplifier wins / loses** — per-feature detection balanced accuracy of `M-top16` "
        f"minus `C16`, under the dropped convention (a feature enters only if BOTH arms have a "
        f"number; a refusal leaves no score row and is never imputed at chance). "
        f"{cells_note['n_margin']} features have a margin. The threshold is "
        f"|margin| >= {cells_note['margin']:.2f}.",
        f"2. **MAEM cosine high / low** — the mean over the 16 rollouts `M-top16` shows of that "
        f"rollout's best centred cosine to the feature's SAE direction "
        f"(`{g.scores_rel}/cos_centred.f16` through `results.common.best_per_rollout`, the "
        f"estimator `corpus_search.exemplifier_bok` uses). It is a MEAN OVER THE SHOWN 16, not a "
        f"best-of-k: the product's own `bo_c_*` ladder is over all 64 rollouts and would not be "
        f"about the set the explainer saw. Median over "
        f"{cells_note['n_mcos']} features: **{cells_note['m_med']:.5f}**.",
        f"3. **Corpus cosine high / low** — the top-1 centred cosine of any corpus window to the "
        f"same direction at {SCAN_SIZE:g}M (`read_top1` on the scan's `topk.jsonl`). Median over "
        f"{cells_note['n_ccos']} features: **{cells_note['c_med']:.5f}**.", "",
        f"A feature is a candidate if all three quantities exist and the two contrasted arms do "
        f"not tie: {cells_note['n_eligible']} of {cells_note['n_scored']} qualify, "
        f"{cells_note['n_all_explained']} of them with a non-refused, non-empty explanation on all "
        f"four arms. Fully-explained features are preferred inside every cell and a partly-refused "
        f"one is reached only where the cell holds nothing else.", "",
        "**Both cosines are ONE-SIDED.** The SAE rows of this set are not `centrable` — there is "
        "no raw activation to subtract a mean from — so in both products the SCORER side (the "
        "rollout's or the window's residual) is centred on the base's `whiten_mu` and the TARGET "
        "side is the stored unit SAE direction. `per_target.jsonl` records this as "
        "`centred_sided: 1` and the scan's README states it for the window side. The two "
        "quantities therefore share a convention and are comparable with each other; neither is a "
        "two-sided centred cosine and neither should be reported as one.", "",
    ]
    L += FP.md_table(
        ["cell", "candidates in cell", "all four arms explained",
         "of which |margin| >= threshold", "features taken", "smallest |margin| taken",
         "threshold relaxed?"],
        [["/".join(r["cell"]), r["n_total"], r["n_all_explained"], r["n_at_threshold"],
          ", ".join(str(x) for x in r["taken"]) or "**none**",
          num(r["min_abs_margin"], 4), "**yes**" if r["relaxed"] else "no"] for r in report])
    if any(r["relaxed"] for r in report):
        L += ["*A relaxed cell took its largest-|margin| features regardless of the threshold, "
              "because the cell held fewer than the quota above it. Nothing was padded from "
              "another cell.*", ""]
    if any(r["short"] for r in report):
        short = [r for r in report if r["short"]]
        L += ["*Cells " + ", ".join("`" + "/".join(r["cell"]) + "`" for r in short)
              + " are SHORT: the run has "
              + ", ".join(f"{r['n_total']} candidate(s) in `{'/'.join(r['cell'])}`" for r in short)
              + ", so the quota cannot be met and nothing was borrowed from a neighbouring cell.*",
              ""]
    if any(r["partial"] for r in report):
        L += ["*A feature marked below as missing an arm's explanation was taken because its cell "
              "held no fully-explained candidate; the refusing arm's section says so in place of "
              "an explanation.*", ""]

    L += [f"## Summary over the {len(picked)} features", "",
          "*`Δ` is `M-top16` − `C16` on detection. `J` is the mean pairwise content-word Jaccard "
          "of the arm's own shown set and `cos` the mean pairwise centred residual cosine of it "
          "(rollout arms only). Derived columns: `Δ` and the two means.*", ""]
    hdr = ["feature", "cell", "Δ det", "MAEM cos", "corpus cos",
           *[f"{DISPLAY[a]} det" for a in ARMS], *[f"J {DISPLAY[a]}" for a in ARMS],
           *[f"cos {DISPLAY[a]}" for a in ARMS if a != "C16"]]
    rows = []
    for cell, fe in picked:
        st = stats[fe.feat]
        rows.append([
            fe.feat, "/".join(cell), signed(fe.margin, 3), num(fe.m_cos, 4), num(fe.c_cos, 4),
            *[num((ri.scores.get((a, MAIN_SCORER, fe.feat)) or {}).get("bal_acc"), 3)
              for a in ARMS],
            *[num(st[a]["jaccard"], 3) for a in ARMS],
            *[num(st[a]["res_cos"], 3) for a in ARMS if a != "C16"],
        ])
    L += FP.md_table(hdr, rows)

    pooled = pooled_bands(bands, [fe.feat for _c, fe in picked])
    if pooled:
        L += [f"### Pooled per-band recall over these {len(picked)} features", "",
              "*Items pooled across the 16 features, not a mean of per-feature rates: the bins are "
              "equal-width and per-feature, so a feature contributes what it has.*", ""]
        L += FP.md_table(["arm", *[f"TPR {b}" for b in BANDS]],
                         [[DISPLAY[a], *[frac(pooled.get((a, b))) for b in BANDS]] for a in ARMS])

    for cell, fe in picked:
        L += feature_block(ri, g, fe, cell, bands, stats[fe.feat])

    L += ["## Provenance and what was not read", ""]
    for n in [*g.notes, *notes, *ri.notes]:
        L += [f"- {n}"]
    if verify:
        L += ["- verified (feature, arm) pairs: "
              + ", ".join(f"{v['feature']}/{DISPLAY[v['arm']]} "
                          f"{'hit' if v['hit'] else ('miss, refused' if v['refused'] else 'MISS')}"
                          for v in verify)]
    if vol.missing:
        L += [f"- {len(vol.missing)} file(s) were not in the mirror or on the volume, first: "
              + ", ".join(f"`{m}`" for m in vol.missing[:5])]
    L += ["- the explanations and example blocks on this page are model-generated text and stay "
          "here: they do not go into a log entry, a commit message or `cells.csv`.", ""]
    return "\n".join(L) + "\n"


def pooled_bands(bands: dict, feats: list[int]) -> dict:
    out: dict[tuple[str, str], list[int]] = {}
    for (feat, arm, band), (k, n) in bands.items():
        if feat not in feats:
            continue
        t = out.setdefault((arm, band), [0, 0])
        t[0] += k
        t[1] += n
    return {k: tuple(v) for k, v in out.items()}


def arm_definitions(build: dict, g: Geometry) -> list[str]:
    """What the four arms DO, read off `ARM_SPECS` and the selection functions, with the paths.

    Written out rather than generated from the tuple, because the tuple says `("docmax", 16, None,
    0)` and the reader's question is what a document-ranked pool is. The `ARM_SPECS` row is printed
    beside each one so the two cannot drift apart unnoticed.
    """
    specs = build.get("arms") or {k: list(B.ARM_SPECS[k]) for k in ARMS}
    return [
        "All four arms show **N = 16** examples, are explained by the same model with Delphi's "
        "explainer prompt, and are scored by the same detection and fuzzing judges on the **same "
        "test items**. Only WHICH 16 examples differ. The run's `scores.jsonl` spells the top-16 "
        "rollout arm `M`; this page calls it `M-top16`.", "",
        f"**C16** — `ARM_SPECS[\"C16\"] = {tuple(specs['C16'])}`, i.e. 16 windows from the "
        f"`docmax` corpus pool. That pool is `examples_docmax` on the SHOWN corpus "
        f"(`{build.get('shown_corpus', '—')}`, {build.get('examples_docmax', '—')}): for each "
        f"feature it holds the single best-activating 64-token window of each of the top "
        f"`EXDOC_TOP = {SS.EXDOC_TOP}` DOCUMENTS, ranked by that window's pre-gate SAE activation "
        f"(`autointerp/sae_self.py:1189`, `run_examples_docmax`). The build sorts those rows by "
        f"descending `max_act`, drops any window overlapping one already kept (`build.dedup` / "
        f"`_overlaps` — a no-op here, since the pool is already one window per document) and takes "
        f"the first 16 (`build.py:1793-1800`). So C16 is **one window per document, document-"
        f"ranked, top 16 by activation** — not a window ranking, which would put sixteen "
        f"overlapping cuts of one passage in front of the explainer.", "",
        f"**The pool the three M arms share** — the Exemplifier's "
        f"{build.get('rollout_source', 'maem')} rollouts for that feature, 64 per feature, each "
        f"rendered by `build.render_example` with its per-token activations from the `sae_self` "
        f"product, exact-text deduplicated, and sorted by **descending per-rollout peak SAE "
        f"activation** (the max over the rollout's scored tokens of the feature's pre-gate "
        f"activation, `build.py:1697-1700`). All three M arms draw from this one pool; none of "
        f"them re-reads the rollouts or touches a GPU.", "",
        f"**M-top16** — `ARM_SPECS[\"M\"] = {tuple(specs['M'])}`: the **first 16 of that pool**, "
        f"i.e. the 16 rollouts with the highest SAE-activation peak. Ranking is by peak activation "
        f"and by nothing else.", "",
        f"**M-jac16** — `ARM_SPECS[\"M-jac16\"] = {tuple(specs['M-jac16'])}`: **greedy "
        f"farthest-point** over the same 64 rollouts (`build.farthest_point`), seeded at "
        f"`pool[0]` — the top-activation rollout, so M-jac16's first example is M-top16's first "
        f"example — with each further pick maximising the MINIMUM distance to what is already "
        f"chosen. The distance is `1 − ` the **content-word Jaccard of the two rendered example "
        f"texts** (`build.jaccard_distances` over `build.content_words`): the Jaccard is over "
        f"**unigram token SETS**, the lower-cased `[a-z0-9']+` tokens longer than two characters "
        f"with a {len(B.CONTENT_STOP)}-word stoplist removed — not over n-grams. (The older "
        f"`M-div` arm used word-TRIGRAM Jaccard plus quantile sampling; `M-jac16` is the single "
        f"greedy rule and is a different arm.) `np.argmax` takes the first maximum, so ties fall "
        f"back to the activation order and the selection is deterministic.", "",
        f"**M-cos16** — `ARM_SPECS[\"M-cos16\"] = {tuple(specs['M-cos16'])}`: the same greedy "
        f"farthest-point over the same 64 rollouts and the same seed, with the distance `1 − ` the "
        f"**mutual cosine of the stored `best_act.f16` residuals after subtracting the base's "
        f"`whiten_mu`** (`build.cosine_distances` over `build._centred_residuals`; mu "
        f"`{g.mu_path or '(not loaded)'}`). `best_act.f16` is `[N targets, 64 rollouts, d]` and "
        f"holds, per rollout, the read-layer residual **at the token of maximum UNCENTRED cosine "
        f"to the target direction** (`precompute/score.py:78-84`, `_Extra.__call__`) — which is "
        f"NOT the SAE-activation peak token that M-top16 ranks by. M-cos16 is therefore a "
        f"**diversity control, not a matched-token comparison**. The centring is not cosmetic: raw "
        f"layer-42 residuals share a large mean and their mutual cosines sit in a narrow band near "
        f"1, which would rank almost nothing.", "",
        "Two consequences worth keeping in view while reading the blocks below: (i) all three M "
        "arms show the same Example 1, because the farthest-point seed is the pool's top-activation "
        "rollout; (ii) M-cos16 maximises spread in a space picked out by the cosine argmax token, "
        "so its set can look textually repetitive while being geometrically spread, and M-jac16's "
        "can look textually varied at similar activations.", "",
        "Source files: `evals/faithfulness/autointerp/build.py` (`ARM_SPECS`, `farthest_point`, "
        "`jaccard_distances`, `cosine_distances`, `_centred_residuals`, `content_words`, `dedup`, "
        "`render_example`, `exemplar_block`), `evals/faithfulness/autointerp/sae_self.py` "
        "(`run_examples_docmax`, `EXDOC_TOP`), `evals/faithfulness/precompute/score.py` (`best_act.f16`, "
        "`cos_centred.f16`), `evals/faithfulness/autointerp/run.py` (the explainer call and its cache), "
        "`evals/faithfulness/results/feature_page.py` and `evals/faithfulness/results/autointerp.py` (the "
        "loaders and the per-band join reused here).", "",
    ]


# ---------------------------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    run: Annotated[str, typer.Option("--run", help="the run directory under `runs/`")]
    = "2026-09-24_autointerp-512",
    build: Annotated[str, typer.Option(
        help="the build directory (volume-relative), for a run whose summary README does not name "
             "the build it read")] = "",
    out: Annotated[Path | None, typer.Option(help="the Markdown FILE to write")] = None,
    n_per_cell: Annotated[int, typer.Option(help="features per cell of the 2 x 2 x 2 design")] = 2,
    margin: Annotated[float, typer.Option(
        help="the |detection margin| a cell's features must clear before the threshold is relaxed "
             "for that cell")] = 0.10,
    score_tag: Annotated[str, typer.Option(
        help="the run tag of the `score` product carrying the rollout cosines and residuals")]
    = SCORE_TAG,
    scan_dir: Annotated[str, typer.Option(
        help="the corpus scan directory; default: the bank the build's shown-corpus `examples` "
             "came from")] = "",
    scan_size: Annotated[float, typer.Option(help="the corpus size, in millions of tokens")]
    = SCAN_SIZE,
    verify_cache: Annotated[int, typer.Option(
        help="rebuild this many explainer request bodies and check their sha256 against the run's "
             "own call cache; 0 skips it")] = 8,
    root: Annotated[str, typer.Option(help="volume-relative root the run was written under")] = "",
    data: Annotated[Path | None, typer.Option(help="local mirror of the volume")] = None,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what the mirror has")] = False,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
) -> None:
    mirror = Path(data) if data else R.mirror_dir(root)
    vol = R.Vol(root, mirror, modal_cmd, refetch, quiet, offline=not fetch)
    ri = FP.load_run(vol, "run", run, build)
    assert ri.scores, ("no readable `summary/scores.jsonl`: " + "; ".join(ri.notes))
    assert ri.build_rel, ("the run's summary README names no build directory and no --build was "
                          "given: the shown examples are the point of this page")
    feats = sorted({f for _a, _s, f in ri.scores})
    FP.load_feature_blocks(vol, ri, feats)
    present = sorted({a for a, _s, _f in ri.scores})
    missing = [a for a in ARMS if a not in present]
    assert not missing, (f"this run has no arm(s) {missing}: it carries {present}, and the page is "
                         f"the four-way comparison of {list(ARMS)}")

    scan = scan_dir or Path(str(ri.build.get("examples") or "")).name
    assert scan, ("no --scan-dir and the build records no shown-corpus `examples` bank to take it "
                  "from: the corpus-cosine axis has no source")
    g = load_geometry(vol, ri.build, score_tag, scan, scan_size)

    cands = collect(ri, g, feats)
    ok = [c for c in cands if c.eligible]
    m_vals = [c.m_cos for c in cands if c.m_cos is not None]
    c_vals = [c.c_cos for c in cands if c.c_cos is not None]
    assert m_vals and c_vals, "neither cosine axis has a value on any feature"
    m_med, c_med = statistics.median(m_vals), statistics.median(c_vals)
    picked, report = select(ok, m_med, c_med, n_per_cell, margin)
    chosen = [fe.feat for _c, fe in picked]
    print(f"[cases] {len(ok)}/{len(feats)} eligible, {len(chosen)} features picked: {chosen}",
          flush=True)

    bands, band_notes = band_tpr(vol, run, ri.build_rel, ri, chosen, list(ARMS), MAIN_SCORER)
    stats = {fe.feat: {a: arm_stats(ri, g, fe.feat, a) for a in ARMS} for _c, fe in picked}
    verify: list[dict] = []
    vnotes: list[str] = []
    if verify_cache:
        cfg_ai = R.load_config().get("autointerp") or {}
        pairs = [(fe.feat, a) for _c, fe in picked for a in ARMS][:verify_cache]
        verify, vnotes = verify_blocks(
            vol, run, ri, pairs, str(cfg_ai.get("explainer_model") or "claude-sonnet-5"),
            int(cfg_ai.get("explainer_max_tokens") or 600), 1)

    cells_note = {
        "n_per_cell": n_per_cell, "margin": margin, "n_scored": len(feats),
        "n_margin": sum(1 for c in cands if c.margin is not None),
        "n_mcos": len(m_vals), "n_ccos": len(c_vals), "n_eligible": len(ok),
        "n_all_explained": sum(1 for c in ok if c.all_explained),
        "m_med": m_med, "c_med": c_med,
    }
    doc = render(ri, g, picked, report, bands, stats, cells_note, verify,
                 [*band_notes, *vnotes], sys.argv, vol)
    path = Path(out) if out else R.out_dir("autointerp_cases") / "autointerp_cases.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    print(f"[cases] wrote {path} ({path.stat().st_size / 1e3:.1f} kB)", flush=True)
    if vol.missing:
        print(f"[cases] {len(vol.missing)} files absent, first: {vol.missing[:3]}", flush=True)


if __name__ == "__main__":
    app()
