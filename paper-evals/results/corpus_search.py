#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "rich>=13"]
# ///
"""Eval 1's corpus-search baseline: panel a row 3, the corpus-size curve, and their cells.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-m2/paper-evals/results/corpus_search.py table \\
       --scan-dir 2026-09-21_v3_realact__train_parity_10m__paper0923 \\
       --set 2026-09-21_v3_realact)

Local, CPU, no GPU, no model. Every number is READ from what `scan` (and, for the paired
difference, `score`) wrote; this file recomputes nothing but the aggregation, its SE and the fit.

WHAT IT READS, AND WHY NOT `results/faithfulness.py`. The corpus-search baseline is a COSINE and
its product is the scan's `topk.jsonl` (`precompute/scan.py`), one line per (target row, corpus
size) with `top` = up to 64 entries `[doc, window start, argmax within the window, cos]`, rank 0
being the best cosine over corpus windows at that size. `faithfulness.py`'s `corpus_peak` is a
peak ACTIVATION from `sae_self` and is a different quantity on a different axis; anything citing
it as panel a row 3's source is wrong (survey §9.3g). `results/ood.py` has a sibling `scan_top1`
for the OOD arms table; this file is the eval-1 half, kept separate because the OOD one is M5's.

THE FOUR SIZES ARE ONE PASS, NOT FOUR RUNS. `scan` snapshots its running top-k at every size
boundary inside one walk, so `1.25 / 2.5 / 5 / 10M` are nested prefixes of a single corpus and
every target is measured at all four. That is an assumption this file GATES rather than trusts:
a row missing a size is a refusal, because a curve fitted over a ragged design is not a curve.

THE OWN-DOCUMENT EXCLUSION IS ROW-DROPPING HERE, NOT WINDOW MASKING. `scan` masks the windows of
a target's own document only where the scanned corpus IS that target's corpus, which for the
eval-1 headline block is never: her realact rows carry a `doc` into HER v2 collection and no
`p`/`L` at all, and the training corpus is built from disjoint document ranges (spec §1.4). What
stands in its place, and what the paper's methods state, is the n-gram exclusion recorded in the
set's own `exclusions.json` -- 26 of her 512 rows at coverage >= 0.05 on 7-grams, headline
n = 486. This file applies it BY DEFAULT and refuses to report an unexcluded headline silently.

STATISTICS. The unit is the target; the SE is the document-clustered bootstrap of
`results.common.cluster_bootstrap` (her block puts up to `doc_n_targets` targets in one document,
and two targets of one document are not two draws). The curve's slope per doubling is fitted PER
TARGET over log2(size) and averaged, which on a complete design is exactly the slope of the mean
curve -- asserted here, so the two readings can never drift -- and unlike the fit on four means it
carries an SE.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results.common as R  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The corpus sizes, in millions of tokens, that each `arm` slot of the key grammar names
# (paper/2026-09-22_writing-plan.md §2: `corp1p25 corp2p5 corp5 corp10`). Written here as the ONE
# mapping from a size to its key fragment, so a new ladder is one edit and not a regex.
SIZE_SLOT = {1.25: "corp1p25", 2.5: "corp2p5", 5.0: "corp5", 10.0: "corp10", 16.0: "corp16"}


# ---------------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------------


def read_exclusions(vol: R.Vol, base: str, set_name: str) -> tuple[set[int], str]:
    """(the set's excluded row indices, a provenance sentence) from its `exclusions.json`.

    The indices are into THAT set's own 512 and must never be applied to another block: her 26
    come from `ngram_overlap --side hers --n 7` over her v2 text, our 6 from a different
    instrument at n = 13, and the two lists are not interchangeable (the sets' own READMEs say so
    in those words). Hence the set name is a required argument and never defaulted.

    An absent file is NOT an empty exclusion. A block whose exclusions have not been computed and
    one whose exclusions are empty are different states, and only the second may be reported as a
    headline, so the caller is handed None and decides.
    """
    rec = vol.json(f"base/{base}/heldout/{set_name}/exclusions.json")
    if rec is None:
        return set(), ""
    rows = {int(r) for r in rec.get("excluded_rows") or []}
    why = (
        f"{len(rows)} of {rec.get('rows_total', '?')} rows excluded, headline n = "
        f"{rec.get('n_headline', '?')}, criterion {rec.get('criterion')!r} at n-gram "
        f"{rec.get('n_gram')}, from {rec.get('source')}"
    )
    return rows, why


def read_ids(vol: R.Vol, base: str, set_name: str) -> dict[int, dict]:
    """{set row -> the row's ids.jsonl record}. Raises on an absent set: a corpus number about a
    set whose rows cannot be read is a number about nothing."""
    rows = vol.jsonl(f"base/{base}/heldout/{set_name}/ids.jsonl")
    assert rows, f"no ids.jsonl for set {set_name!r} under base/{base}/heldout/"
    out = {int(r["row"]): r for r in rows}
    assert len(out) == len(rows), f"{set_name}/ids.jsonl has duplicate `row` values"
    return out


def cluster_of(row: dict) -> object:
    """The bootstrap cluster of one target: its document where it has one, else itself.

    A `random` or `sae` draw has no document and becomes its own cluster, which makes the
    clustered bootstrap reduce EXACTLY to the ordinary one for those families rather than to a
    silently different estimator (`results.common.cluster_bootstrap` says the same).
    """
    doc = row.get("doc")
    return ("doc", int(doc)) if doc is not None else ("row", int(row["row"]))


@dataclass
class Top1:
    """One scan's corpus-search top-1, per (set row, size), with what was dropped and why."""

    by_size: dict[float, dict[int, float]]
    sizes: list[float]
    corpus: str
    set_name: str
    scan_dir: str
    n_rows_seen: int
    empty: dict[float, list[int]]        # rows with no unmasked window at that size
    excluded: set[int]
    exclusion_note: str

    @property
    def rows(self) -> list[int]:
        """The set rows present at EVERY size -- the only rows a curve may be fitted over."""
        if not self.sizes:
            return []
        common = set(self.by_size[self.sizes[0]])
        for s in self.sizes[1:]:
            common &= set(self.by_size[s])
        return sorted(common)


def read_top1(
    vol: R.Vol,
    base: str,
    scan_dir: str,
    set_name: str,
    *,
    family: str | None = "realact",
    apply_exclusions: bool = True,
    sae_key: str | None = None,
) -> Top1:
    """The per-direction top-1 cosine per corpus size, for one scan directory and one set.

    `scan --with-set` puts several sets' rows in one `topk.jsonl` and re-indexes `row` to the
    position WITHIN the scan, so the join key is (`set`, `set_row`) and never `row`: reading
    `row` here would silently read her realact block's numbers off whichever bank happened to be
    appended at that offset. Rows of other sets and other families are skipped, not merged.

    A row whose `top` is EMPTY at a size had no unmasked window there. It is recorded in `empty`
    and left out at that size rather than scored 0, and `rows` then excludes it from the curve on
    every size -- a target that is missing one point of its own curve cannot contribute a slope.
    """
    recs = vol.jsonl(f"base/{base}/scan/{scan_dir}/topk.jsonl")
    assert recs, f"no topk.jsonl at base/{base}/scan/{scan_dir}/ (is the scan still running?)"
    excluded, why = (set(), "")
    if apply_exclusions:
        excluded, why = read_exclusions(vol, base, set_name)
        assert why, (
            f"--apply-exclusions was asked for but base/{base}/heldout/{set_name}/exclusions.json "
            f"is not on the volume. An absent exclusion file is not an empty exclusion: pass "
            f"--no-apply-exclusions deliberately if this block genuinely has none."
        )
    ids = read_ids(vol, base, set_name)

    by_size: dict[float, dict[int, float]] = {}
    empty: dict[float, list[int]] = {}
    seen: set[int] = set()
    corpus = ""
    for r in recs:
        if r.get("set", set_name) != set_name:
            continue
        row = int(r.get("set_row", r["row"]))
        fam = r.get("family") or ids.get(row, {}).get("family")
        if family is not None and fam != family:
            continue
        if sae_key is not None and (ids.get(row, {}).get("sae_key") != sae_key):
            continue
        if row in excluded:
            continue
        size = float(r["size"])
        seen.add(row)
        corpus = corpus or str(r.get("corpus") or "")
        if not r.get("top"):
            empty.setdefault(size, []).append(row)
            continue
        by_size.setdefault(size, {})[row] = float(r["top"][0][3])
    assert seen, (
        f"scan {scan_dir} carries no rows of set {set_name!r}"
        + (f" family {family!r}" if family else "")
        + (" after exclusions" if excluded else "")
    )
    return Top1(
        by_size=by_size,
        sizes=sorted(by_size),
        corpus=corpus,
        set_name=set_name,
        scan_dir=scan_dir,
        n_rows_seen=len(seen),
        empty=empty,
        excluded=excluded,
        exclusion_note=why,
    )


def assert_complete(t: Top1) -> None:
    """THE GATE the plan names: every selected row carries every size.

    The ladder is nested inside ONE pass, so a row that has a top-1 at 10M and none at 2.5M means
    the snapshots and the rows disagree -- a product defect, not a small-sample nuisance -- and
    every mean below it would be over a different set of targets per column. Refuse, naming rows.
    """
    per_size = {s: set(t.by_size[s]) for s in t.sizes}
    everywhere = t.rows
    ragged = sorted(set().union(*per_size.values()) - set(everywhere)) if per_size else []
    assert not ragged, (
        f"{len(ragged)} rows of {t.set_name} are absent at some size of {t.sizes} in {t.scan_dir}: "
        f"{ragged[:8]}{'...' if len(ragged) > 8 else ''}. The sizes are nested snapshots of one "
        f"pass, so this is a disagreement between the product and the set, and a per-size mean "
        f"over different targets is not a curve."
    )


# ---------------------------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------------------------


@dataclass
class SizeStat:
    size: float
    mean: float
    se: float          # document-clustered bootstrap
    se_iid: float      # std/sqrt(n), carried so the size of the clustering correction is visible
    n: int
    n_clusters: int


def size_table(t: Top1, ids: dict[int, dict]) -> list[SizeStat]:
    """Mean top-1 with its clustered SE at every size, over the rows present at ALL sizes."""
    assert_complete(t)
    rows = t.rows
    clusters = [cluster_of(ids[r]) for r in rows]
    out = []
    for s in t.sizes:
        vals = [t.by_size[s][r] for r in rows]
        mean, se, n, nc = R.cluster_bootstrap(vals, clusters)
        out.append(SizeStat(s, mean, se, R.se_iid(vals), n, nc))
    return out


@dataclass
class Slope:
    per_doubling: float
    se: float
    n: int
    n_clusters: int
    from_mean_curve: float   # the same quantity read the other way; asserted equal


def curve_slope(t: Top1, ids: dict[int, dict]) -> Slope:
    """The fitted change in top-1 cosine per DOUBLING of corpus size.

    Fitted per target by ordinary least squares of its four top-1 values on log2(size), then
    averaged over targets with the document-clustered bootstrap for the SE. OLS is linear in the
    responses, so on a COMPLETE design (which `assert_complete` has just guaranteed) the mean of
    the per-target slopes is identically the slope of the mean curve; both are computed and
    asserted equal, so "the slope" cannot come to mean two things in two places.
    """
    assert_complete(t)
    rows = t.rows
    assert len(t.sizes) >= 2, f"a slope needs at least two sizes, got {t.sizes}"
    x = np.log2(np.asarray(t.sizes, dtype=np.float64))
    xc = x - x.mean()
    denom = float((xc**2).sum())
    y = np.asarray([[t.by_size[s][r] for s in t.sizes] for r in rows], dtype=np.float64)  # [R, S]
    per_target = (y @ xc) / denom
    clusters = [cluster_of(ids[r]) for r in rows]
    mean, se, n, nc = R.cluster_bootstrap(per_target, clusters)
    mean_curve = float((y.mean(axis=0) @ xc) / denom)
    assert math.isclose(mean, mean_curve, rel_tol=1e-9, abs_tol=1e-12), (
        f"the mean of the per-target slopes ({mean!r}) and the slope of the mean curve "
        f"({mean_curve!r}) disagree, which on a complete design is arithmetically impossible: "
        f"the design is not complete, or the rows are not aligned across sizes"
    )
    return Slope(mean, se, n, nc, mean_curve)


@dataclass
class Paired:
    mean: float
    se: float
    lo: float
    hi: float
    n: int
    n_clusters: int
    win: float       # fraction of targets on which the Exemplifier beats the corpus top-1


def exemplifier_bok(vol: R.Vol, scores_rel: str, k: int, which: str = "centred"):
    """[N] unbiased best-of-k per target from a `score` product, or None when it is not there.

    `score` stores the bo ladder at k = 64 only (`bo_c_64`), so a bo8 is computed here from
    `cos_centred.f16` [N, n, T] through the pipeline's ONE estimator
    (`results.common.best_per_rollout` then `bo_unbiased`, M0a's `bo_unbiased`) rather than read
    off a column that does not exist. `score` drops a rollout with no kept CENTRED token rather
    than scoring it -1, so the centred path takes NaN for "no draw" and rows with any NaN are
    returned as NaN -- the estimator's k-subsets are not equally likely with a missing draw.
    """
    name = {"centred": "cos_centred.f16", "raw": "cos.f16", "asym": "cos_asym.f16"}[which]
    idx = vol.json(f"{scores_rel}/index.json") or {}
    meta = idx.get(name)
    if not meta or "shape" not in meta:
        return None
    shape = tuple(int(v) for v in meta["shape"])
    arr = vol.array(f"{scores_rel}/{name}", "float16", shape)
    if arr is None:
        return None
    empty = float("nan") if which == "centred" else -1.0
    best = np.stack([R.best_per_rollout(arr[i].astype(np.float32), empty=empty)
                     for i in range(shape[0])])          # [N, n]
    out = np.full(shape[0], np.nan)
    ok = ~np.isnan(best).any(axis=1)
    if ok.any():
        out[ok] = R.bo_unbiased(best[ok], k)
    return out


def paired_difference(
    t: Top1, ids: dict[int, dict], bok: np.ndarray, size: float, *, n_boot: int = R.N_BOOT
) -> Paired:
    """Exemplifier bo-k MINUS the corpus top-1 at `size`, PAIRED on the target.

    The pairing is on the SET ROW, which is the index `score`'s arrays are in and the `set_row`
    the scan wrote -- the two products index the same held-out set, and that is the only reason
    they may be differenced at all. A row missing on either side is dropped from both, never
    zero-filled; the count comes back so the caller can print it beside the number.
    """
    assert size in t.by_size, f"no corpus top-1 at size {size}M; the scan carries {t.sizes}"
    corp = t.by_size[size]
    rows = [r for r in sorted(corp) if r < len(bok) and np.isfinite(bok[r])]
    assert rows, f"no target has both a corpus top-1 at {size}M and a finite best-of-k"
    d = np.asarray([float(bok[r]) - corp[r] for r in rows], dtype=np.float64)
    clusters = [cluster_of(ids[r]) for r in rows]
    mean, se, n, nc = R.cluster_bootstrap(d, clusters, n_boot=n_boot)
    # The percentile interval over the SAME clusters, so the CI and the SE are one resampling
    # scheme and not two. Re-resampled here rather than returned by cluster_bootstrap, which
    # gives an SE only.
    order: dict = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    groups = [np.asarray(v, dtype=np.int64) for v in order.values()]
    if len(groups) < 2:
        lo = hi = float("nan")
    else:
        rng = np.random.default_rng(R.BOOT_SEED)
        sums = np.array([d[g].sum() for g in groups])
        counts = np.array([g.size for g in groups], dtype=np.float64)
        pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return Paired(mean, se, lo, hi, n, nc, float((d > 0).mean()))


# ---------------------------------------------------------------------------------------------
# cells.csv
# ---------------------------------------------------------------------------------------------


def _fmt(v: float | None, places: int = 4) -> str:
    return "" if v is None or not math.isfinite(float(v)) else f"{float(v):.{places}f}"


def cells_rows(
    stats: list[SizeStat],
    slope: Slope,
    *,
    eval_slot: str = "fid",
    set_slot: str = "ra",
    run: str = "R2",
    source: str,
    date: str,
    note: str,
) -> list[dict]:
    """The `paper/numbers/cells.csv` rows M2 owns, in the schema's column order.

    Keys are `<eval>.<set>.<arm>.<metric>` of the writing plan §2 grammar with the corpus size in
    the ARM slot (`corp1p25 corp2p5 corp5 corp10`) -- a corpus cell never carries a `bo` slot,
    because a corpus search has no rollout budget. `value` is written with the digits exactly as
    they should print; `se` is the document-clustered one.
    """
    rows = []
    for s in stats:
        slot = SIZE_SLOT.get(s.size)
        assert slot, (
            f"corpus size {s.size}M has no arm slot in the key grammar (writing plan §2 lists "
            f"{sorted(SIZE_SLOT.values())}); add one there before adding a cell for it"
        )
        rows.append({
            "key": f"{eval_slot}.{set_slot}.{slot}.cos",
            "value": _fmt(s.mean), "se": _fmt(s.se), "lo": "", "hi": "", "n": str(s.n),
            "status": "final", "run": run, "source": source, "date": date,
            "note": f"{note}; document-clustered SE over {s.n_clusters} clusters",
        })
    rows.append({
        "key": f"{eval_slot}.{set_slot}.corp10.slope",
        "value": _fmt(slope.per_doubling), "se": _fmt(slope.se), "lo": "", "hi": "",
        "n": str(slope.n), "status": "final", "run": run, "source": source, "date": date,
        "note": f"{note}; change in top-1 cosine per DOUBLING of corpus size, per-target OLS "
                f"over log2(size), document-clustered SE over {slope.n_clusters} clusters",
    })
    return rows


CELLS_COLUMNS = ["key", "value", "se", "lo", "hi", "n", "status", "run", "source", "date", "note"]


def write_cells(rows: list[dict], path: Path) -> None:
    """Append rows to a cells CSV fragment (never to `cells.csv` in place: the writer merges)."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CELLS_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


@app.command()
def table(
    scan_dir: Annotated[str, typer.Option("--scan-dir", help="directory under base/<base>/scan/")],
    set_name: Annotated[str, typer.Option("--set", help="the held-out set whose rows to read")],
    base: str = "qwen36-27b",
    family: str = "realact",
    root: str = "",
    data_dir: str = "",
    scores_rel: Annotated[str, typer.Option(help="a `score` product to difference against")] = "",
    bo_k: int = 8,
    diff_size: float = 10.0,
    apply_exclusions: bool = True,
    no_fetch: bool = False,
    cells_out: str = "",
    date: str = "2026-09-23",
) -> None:
    """Print the per-size table, the slope, the optional paired difference, and write the cells."""
    vol = R.Vol(root, Path(data_dir) if data_dir else R.mirror_dir(root), offline=no_fetch)
    t = read_top1(vol, base, scan_dir, set_name, family=family,
                  apply_exclusions=apply_exclusions)
    ids = read_ids(vol, base, set_name)
    stats = size_table(t, ids)
    sl = curve_slope(t, ids)

    print(f"scan {scan_dir}  corpus {t.corpus!r}  set {set_name}  family {family}")
    if t.exclusion_note:
        print(f"exclusions: {t.exclusion_note}")
    for s, rowlist in sorted(t.empty.items()):
        print(f"  NOTE {len(rowlist)} rows had no unmasked window at {s}M: {rowlist[:8]}")
    print(f"{'size (M)':>10} {'mean top-1':>12} {'clustered SE':>13} {'iid SE':>9} {'n':>5} {'docs':>6}")
    for s in stats:
        print(f"{s.size:>10} {s.mean:>12.4f} {s.se:>13.4f} {s.se_iid:>9.4f} {s.n:>5} {s.n_clusters:>6}")
    print(f"slope per doubling: {sl.per_doubling:+.4f} ± {sl.se:.4f} "
          f"(n = {sl.n}, {sl.n_clusters} documents)")

    if scores_rel:
        bok = exemplifier_bok(vol, scores_rel, bo_k)
        if bok is None:
            print(f"  no cos_centred.f16 under {scores_rel}: no paired difference")
        else:
            p = paired_difference(t, ids, bok, diff_size)
            print(f"paired bo{bo_k} - corpus@{diff_size}M: {p.mean:+.4f} ± {p.se:.4f} "
                  f"[{p.lo:+.4f}, {p.hi:+.4f}]  n = {p.n}  win = {p.win:.3f}")

    if cells_out:
        rows = cells_rows(
            stats, sl,
            source=f"base/{base}/scan/{scan_dir}/topk.jsonl", date=date,
            note=f"corpus search on {t.corpus or scan_dir}, centred both sides (scan --centre); "
                 f"{set_name} {family} after the set's n-gram exclusions",
        )
        write_cells(rows, Path(cells_out))
        print(f"wrote {len(rows)} cells rows to {cells_out}")


@app.command()
def fired(
    top1_rel: Annotated[str, typer.Option("--top1-dir", help="a `top1_act` product directory, "
                                          "volume-relative")],
    data_dir: str = "",
    root: str = "",
    no_fetch: bool = False,
) -> None:
    """Panel b's corpus comparator: does the COSINE top-1 corpus window make the feature fire?

    Read from `top1_act`'s own rows, per corpus-frequency quartile. `stratum` is the draw's own
    rarity band with 0 the RAREST (the density column on the same row is what it was cut on), so
    it maps to the writing plan's `q1..q4` rarest-first in that order -- printed beside the
    median density so the mapping is checkable and not merely asserted.

    `fired` is the fraction whose top-1 window peaks above the checkpoint's own learned BatchTopK
    gate -- `top1_act`'s `passes_gate`, the same gate `stats`, `score` and `scan` use. The median
    peak is the raw pre-gate activation, not a ratio: panel b's ratio has the corpus peak as its
    DENOMINATOR and is 1 by construction there (writing plan §2), which is why this reports the
    fired fraction and the level rather than a ratio of 1.
    """
    vol = R.Vol(root, Path(data_dir) if data_dir else R.mirror_dir(root), offline=no_fetch)
    recs = vol.jsonl(f"{top1_rel.rstrip('/')}/top1_act.jsonl")
    assert recs, f"no top1_act.jsonl under {top1_rel}"
    gate = recs[0].get("gate")
    by_q: dict[object, list[dict]] = {}
    for r in recs:
        by_q.setdefault(r.get("stratum"), []).append(r)
    print(f"{top1_rel}  n = {len(recs)}  gate = {gate}")
    print(f"{'stratum':>8} {'key':>4} {'n':>5} {'fired':>7} {'median peak':>12} "
          f"{'median cos':>11} {'median density':>15}")
    for st in sorted(by_q, key=lambda v: (v is None, v)):
        rs = by_q[st]
        f = float(np.mean([bool(r["passes_gate"]) for r in rs]))
        pk = float(np.median([float(r["act_max"]) for r in rs]))
        cs = float(np.median([float(r["top1_cos"]) for r in rs]))
        de = [r.get("density") for r in rs if r.get("density") is not None]
        dv = float(np.median(de)) if de else float("nan")
        key = f"q{int(st) + 1}" if isinstance(st, int) else "-"
        print(f"{str(st):>8} {key:>4} {len(rs):>5} {f:>7.4f} {pk:>12.4f} {cs:>11.4f} {dv:>15.3e}")
    allf = float(np.mean([bool(r["passes_gate"]) for r in recs]))
    allp = float(np.median([float(r["act_max"]) for r in recs]))
    print(f"{'pooled':>8} {'-':>4} {len(recs):>5} {allf:>7.4f} {allp:>12.4f}")


# ---------------------------------------------------------------------------------------------
# selftest -- a synthetic scan directory, and every gate deliberately broken
# ---------------------------------------------------------------------------------------------


def _synth(dirpath: Path, base: str, set_name: str, scan_dir: str, *, n: int = 8,
           sizes=(1.25, 2.5, 5.0, 10.0), excluded=(1,), seed: int = 7) -> None:
    """A scan directory and a held-out set whose numbers are known by construction.

    Row i's top-1 at size s is `0.30 + 0.02 * log2(s / 1.25) + 0.01 * i`, so the slope per
    doubling is 0.02 for EVERY row exactly, the mean at 1.25M is 0.30 + 0.005 * (n - 1) over the
    kept rows, and any drift in the estimator shows up as a departure from a closed form rather
    than as a plausible-looking number.
    """
    hd = dirpath / "base" / base / "heldout" / set_name
    sd = dirpath / "base" / base / "scan" / scan_dir
    hd.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)
    ids = [
        # two targets per document, so the clustered bootstrap has something to cluster
        {"row": i, "family": "realact", "id": f"doc{i // 2}:pos{i}", "doc": i // 2,
         "source": "hers", "act_norm": 90.0 + i}
        for i in range(n)
    ]
    (hd / "ids.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ids), encoding="utf-8")
    (hd / "exclusions.json").write_text(json.dumps({
        "block": "realact", "rows_total": n, "excluded_rows": list(excluded),
        "n_headline": n - len(excluded), "criterion": 0.05, "n_gram": 7,
        "source": "synthetic (results/corpus_search.py selftest)",
    }), encoding="utf-8")
    rng = np.random.default_rng(seed)
    lines = []
    for s in sizes:
        for i in range(n):
            cos = 0.30 + 0.02 * math.log2(s / sizes[0]) + 0.01 * i
            lines.append({
                "row": 100 + i,                 # deliberately NOT the set row: the join is on set_row
                "set": set_name, "set_row": i, "family": "realact", "arm": None,
                "corpus": "train_parity_10m", "size": s,
                "top": [[int(rng.integers(0, 1000)), 0, 3, round(cos, 5)],
                        [int(rng.integers(0, 1000)), 16, 1, round(cos - 0.05, 5)]],
            })
        # a row of ANOTHER set and another family at every size: both must be skipped
        lines.append({"row": 900, "set": "other_set", "set_row": 0, "family": "realact",
                      "corpus": "train_parity_10m", "size": s, "top": [[0, 0, 0, 0.99]]})
        lines.append({"row": 901, "set": set_name, "set_row": 999, "family": "random",
                      "corpus": "train_parity_10m", "size": s, "top": [[0, 0, 0, 0.98]]})
    (sd / "topk.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def _mutate(path: Path, fn) -> None:
    """Rewrite a jsonl through `fn`, asserting the mutation actually changed the file.

    A gate that has never failed has never been shown to be a gate, and a mutation that did not
    apply proves nothing about the gate it was meant to trip.
    """
    before = path.read_text(encoding="utf-8")
    recs = [json.loads(ln) for ln in before.splitlines() if ln.strip()]
    out = [r for r in (fn(dict(r)) for r in recs) if r is not None]
    after = "".join(json.dumps(r) + "\n" for r in out)
    assert after != before, f"the mutation did not change {path}: it tests nothing"
    path.write_text(after, encoding="utf-8")


@app.command()
def selftest() -> None:  # noqa: C901 -- one check per paragraph, kept in one place on purpose
    """Every reader and every gate, against a synthetic scan directory. CPU, offline, $0."""
    base, set_name, scan_dir = "qwen36-27b", "2026-09-21_v3_synth", "2026-09-21_v3_synth__synth"
    tmp = Path(tempfile.mkdtemp(prefix="corpus-search-selftest-"))
    checks = mutations = 0
    try:
        _synth(tmp, base, set_name, scan_dir)
        vol = R.Vol("", tmp, offline=True)
        ids = read_ids(vol, base, set_name)

        # --- the reader -------------------------------------------------------------------
        t = read_top1(vol, base, scan_dir, set_name)
        assert t.sizes == [1.25, 2.5, 5.0, 10.0], t.sizes
        checks += 1
        assert t.excluded == {1}, t.excluded
        checks += 1
        assert set(t.rows) == {0, 2, 3, 4, 5, 6, 7}, t.rows           # row 1 excluded
        checks += 1
        assert t.corpus == "train_parity_10m", t.corpus
        checks += 1
        # the join is (set, set_row): the other set's 0.99 and the `random` row's 0.98 are absent
        assert max(t.by_size[10.0].values()) < 0.5, t.by_size[10.0]
        checks += 1
        # closed form: row i at size s is 0.30 + 0.02*log2(s/1.25) + 0.01*i
        assert math.isclose(t.by_size[5.0][3], 0.30 + 0.02 * 2 + 0.03, abs_tol=1e-9)
        checks += 1
        # unexcluded reading differs, which is what makes the exclusion a decision and not a no-op
        t_all = read_top1(vol, base, scan_dir, set_name, apply_exclusions=False)
        assert set(t_all.rows) == set(range(8)), t_all.rows
        checks += 1

        # --- the per-size table ------------------------------------------------------------
        stats = size_table(t, ids)
        assert [s.size for s in stats] == [1.25, 2.5, 5.0, 10.0]
        checks += 1
        kept = [0, 2, 3, 4, 5, 6, 7]
        want = 0.30 + 0.01 * float(np.mean(kept))
        assert math.isclose(stats[0].mean, want, abs_tol=1e-9), (stats[0].mean, want)
        checks += 1
        # every size shifts by exactly one doubling's worth
        assert math.isclose(stats[3].mean - stats[0].mean, 0.02 * 3, abs_tol=1e-9)
        checks += 1
        # clustering is real: 7 rows over 4 documents, and the clustered SE is not the iid one
        assert stats[0].n == 7 and stats[0].n_clusters == 4, (stats[0].n, stats[0].n_clusters)
        checks += 1
        assert not math.isclose(stats[0].se, stats[0].se_iid, rel_tol=1e-3), (
            stats[0].se, stats[0].se_iid)
        checks += 1

        # --- the slope ----------------------------------------------------------------------
        sl = curve_slope(t, ids)
        assert math.isclose(sl.per_doubling, 0.02, abs_tol=1e-9), sl.per_doubling
        checks += 1
        assert math.isclose(sl.se, 0.0, abs_tol=1e-12), sl.se   # every row has the same slope
        checks += 1
        assert math.isclose(sl.from_mean_curve, sl.per_doubling, abs_tol=1e-12)
        checks += 1

        # --- the paired difference ----------------------------------------------------------
        # bo-k is handed in directly here; `exemplifier_bok`'s own reading of cos_centred.f16 is
        # exercised below against a synthetic [N, n, T] array.
        bok = np.array([0.5 + 0.01 * i for i in range(8)])
        p = paired_difference(t, ids, bok, 10.0)
        # d_i = 0.5 + 0.01 i - (0.30 + 0.06 + 0.01 i) = 0.14 for every kept row
        assert math.isclose(p.mean, 0.14, abs_tol=1e-9), p.mean
        checks += 1
        assert p.n == 7 and p.n_clusters == 4 and p.win == 1.0, (p.n, p.n_clusters, p.win)
        checks += 1
        assert math.isclose(p.se, 0.0, abs_tol=1e-12) and math.isclose(p.lo, 0.14, abs_tol=1e-9)
        checks += 1
        # a NaN row (a rollout with no kept centred token) is dropped from both sides, not filled
        bok_nan = bok.copy()
        bok_nan[4] = np.nan
        assert paired_difference(t, ids, bok_nan, 10.0).n == 6
        checks += 1

        # --- exemplifier_bok, against a synthetic score product ------------------------------
        sc = tmp / "maemms" / "m" / "scores" / "s"
        sc.mkdir(parents=True, exist_ok=True)
        arr = np.full((3, 4, 5), np.nan, dtype=np.float16)
        for i in range(3):
            for j in range(4):
                arr[i, j, : 3] = np.float16(0.1 * i + 0.05 * j)
        arr.tofile(sc / "cos_centred.f16")
        (sc / "index.json").write_text(json.dumps(
            {"cos_centred.f16": {"shape": [3, 4, 5], "dtype": "float16"}}), encoding="utf-8")
        got = exemplifier_bok(vol, "maemms/m/scores/s", 4)
        assert got is not None and got.shape == (3,), got
        checks += 1
        # at k = n the unbiased best-of-k IS the max over the four rollouts
        assert math.isclose(float(got[1]), 0.1 + 0.15, abs_tol=2e-3), got
        checks += 1
        assert exemplifier_bok(vol, "maemms/m/scores/absent", 4) is None
        checks += 1

        # --- cells rows -----------------------------------------------------------------------
        rows = cells_rows(stats, sl, source="synthetic", date="2026-09-23", note="selftest")
        assert [r["key"] for r in rows] == [
            "fid.ra.corp1p25.cos", "fid.ra.corp2p5.cos", "fid.ra.corp5.cos",
            "fid.ra.corp10.cos", "fid.ra.corp10.slope"], [r["key"] for r in rows]
        checks += 1
        assert all(set(r) == set(CELLS_COLUMNS) for r in rows)
        checks += 1
        assert rows[0]["value"] == f"{stats[0].mean:.4f}" and rows[0]["n"] == "7"
        checks += 1
        assert all(r["status"] == "final" and not r["lo"] and not r["hi"] for r in rows)
        checks += 1

        # ============ MUTATIONS: every gate deliberately broken ==============================
        topk = tmp / "base" / base / "scan" / scan_dir / "topk.jsonl"
        keep = topk.read_text(encoding="utf-8")

        # (1) a row missing one size -> the completeness gate refuses, naming it
        _mutate(topk, lambda r: None if (r.get("set_row") == 3 and r.get("size") == 2.5) else r)
        try:
            size_table(read_top1(vol, base, scan_dir, set_name), ids)
        except AssertionError as e:
            assert "absent at some size" in str(e) and " 3" in str(e).replace("[", " "), str(e)
            mutations += 1
        else:
            raise AssertionError("the completeness gate did NOT fire on a ragged design")
        topk.write_text(keep, encoding="utf-8")

        # (2) an empty top list -> the row is recorded and left out, never scored 0
        _mutate(topk, lambda r: ({**r, "top": []}
                                 if (r.get("set_row") == 5 and r.get("size") == 10.0) else r))
        t2 = read_top1(vol, base, scan_dir, set_name)
        assert t2.empty.get(10.0) == [5] and 5 not in t2.by_size[10.0], (t2.empty, t2.by_size[10.0])
        assert 0.0 not in t2.by_size[10.0].values()
        mutations += 1
        try:
            size_table(t2, ids)
        except AssertionError as e:
            assert "absent at some size" in str(e)
            mutations += 1
        else:
            raise AssertionError("a row with an empty top at one size passed the completeness gate")
        topk.write_text(keep, encoding="utf-8")

        # (3) the join key: rewrite `set_row` to the within-scan `row` and the set goes missing
        _mutate(topk, lambda r: {**r, "set_row": r["row"]} if r.get("set") == set_name else r)
        t3 = read_top1(vol, base, scan_dir, set_name, apply_exclusions=False)
        assert set(t3.rows) == {100 + i for i in range(8)}, t3.rows
        assert not (set(t3.rows) & set(range(8))), t3.rows
        mutations += 1
        topk.write_text(keep, encoding="utf-8")

        # (4) a size boundary silently changed -> the slope moves, i.e. the fit reads `size`
        _mutate(topk, lambda r: {**r, "size": 20.0} if r.get("size") == 10.0 else r)
        sl4 = curve_slope(read_top1(vol, base, scan_dir, set_name), ids)
        assert not math.isclose(sl4.per_doubling, 0.02, abs_tol=1e-6), sl4.per_doubling
        mutations += 1
        topk.write_text(keep, encoding="utf-8")

        # (5) the exclusion file removed -> an exclusion that was asked for is not silently skipped
        excl = tmp / "base" / base / "heldout" / set_name / "exclusions.json"
        body = excl.read_text(encoding="utf-8")
        excl.unlink()
        try:
            read_top1(R.Vol("", tmp, offline=True), base, scan_dir, set_name)
        except AssertionError as e:
            assert "is not an empty exclusion" in str(e), str(e)
            mutations += 1
        else:
            raise AssertionError("a missing exclusions.json was treated as no exclusions")
        excl.write_text(body, encoding="utf-8")

        # (6) an excluded row changed -> the headline moves, so the exclusion is load-bearing
        excl.write_text(json.dumps({**json.loads(body), "excluded_rows": [7]}), encoding="utf-8")
        t6 = read_top1(R.Vol("", tmp, offline=True), base, scan_dir, set_name)
        s6 = size_table(t6, ids)
        assert not math.isclose(s6[0].mean, stats[0].mean, abs_tol=1e-9), (s6[0].mean, stats[0].mean)
        mutations += 1
        excl.write_text(body, encoding="utf-8")

        # (7) the paired difference must not pair across a row shift
        shifted = np.concatenate([[np.nan], bok[:-1]])
        p7 = paired_difference(t, ids, shifted, 10.0)
        assert not math.isclose(p7.mean, 0.14, abs_tol=1e-9), p7.mean
        mutations += 1

        # (8) an unknown corpus size has no key slot and must refuse rather than invent one
        try:
            cells_rows([SizeStat(3.0, 0.4, 0.01, 0.01, 7, 4)], sl, source="x", date="d", note="n")
        except AssertionError as e:
            assert "has no arm slot" in str(e), str(e)
            mutations += 1
        else:
            raise AssertionError("a size with no key slot produced a cell anyway")

        # (9) the scan carries no rows of this set -> refuse, never an empty table
        try:
            read_top1(vol, base, scan_dir, "2026-09-21_v3_absent", apply_exclusions=False)
        except AssertionError as e:
            assert "ids.jsonl" in str(e) or "carries no rows" in str(e), str(e)
            mutations += 1
        else:
            raise AssertionError("an absent set produced a table")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"corpus_search selftest OK: {checks} checks, {mutations} mutation gates")


if __name__ == "__main__":
    app()
