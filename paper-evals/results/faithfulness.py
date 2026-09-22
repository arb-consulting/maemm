#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9"]
# ///
"""Eval 1's tables and figures, from the products already on the volume.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm/paper-evals/results/faithfulness.py --set 2026-09-21_v1raw)

Local, CPU, no GPU, no model. Every number is READ from what `score` and `sae_self` wrote; this
file recomputes nothing except the aggregation across rows and the standard errors (and, as a
CHECK, the per-rollout reductions the products already did -- see "reader check" below).

WHAT IT BUILDS, for every (family x source x run-tag) present on the volume for `--set`:

  cosines (every family whose `family_kinds.<f>.kind` is not `dictionary`)
      bo1 (the plain mean over rollouts), bo8 and bo64 of `cos_centred` -- the headline where the
      run centred on something -- and of `cos_raw`, each with a standard error CLUSTERED BY
      DOCUMENT and bootstrapped over documents. One row per source per cosine, carrying n, the
      number of rows and the number of distinct documents behind them.

  cells    the `paper/numbers/cells.csv` rows of module M1 -- panel a's Exemplifier, base-control,
      NLA, random-floor and paired cells, and panel b's per-quartile ratio and fired cells on the
      131k and the 2M blocks. Built on every run, PRINTED on every run, and written only under
      `--cells <path>`, in place, touching no key this module does not own. A cell that cannot be
      built is listed with its reason and not written: the placeholder row already in the CSV says
      "expected, not yet measured", which is true, and a zero would not be.

  SAE families (`sae` rows, split by the dictionary in the row's own `sae_key` and by `sae_side`)
      median and mean of `peak / corpus peak`, the denominator chosen by `--corpus-peak` and its
      provenance carried into every cell and caption (`stored` is our 16M `max_act`, the default;
      `top1_act:<dir>` is M2's scan of the 10M training corpus, which spec §1.2 asks for), at
      bo1/bo8/bo64 where the rollout count allows, the fraction of draws above the learned gate
      and the fraction of features that fire at all, whole-family and per stratum. NO SAE-TARGET
      COSINE reaches a markdown table (plan §2.3): the per-row cosines go to `sae_cosines.csv`,
      where nothing invites them to be read beside an activation ratio.

  sanity  the gates of plan §2.3, from `results/sanity.yaml`, which the USER edits: her card's
      numbers, the `sae_smoke64.md` medians and the old primary's recorded values, each with its
      own stated tolerance and a pass / FLAG / absent verdict per line.

  figures per-family bo-k curves per source; per-stratum firing and ratio bars per SAE family.
      PDF + PNG. (The two-arm row-by-row figure went with the old primary's `mu-none` /
      `mu-stats` arms on 2026-09-23: the old primary is dropped from the paper and survives only
      as a never-printed pipeline sanity check, and no other checkpoint carries two run tags on
      one set.)

NOTHING IS KEYED ON A CHECKPOINT OR A DICTIONARY NAME. Sources come from iterating `config.yaml`'s
`maemms:` against the volume's scores directories, families from the rows' own fields. A new SAE
or a new MAEMM is a config entry plus its products, and this file does not change.

A source that is absent is SKIPPED and LISTED, never zero-filled: a checkpoint nobody ran and a
checkpoint that scored zero are opposite findings and must not print the same.

READER CHECK. The per-rollout reductions are recomputed from `cos.f16` / `sae_self.f16` and
compared with the products' own `per_target.jsonl` / `sae_self.json`. These must agree to f16
rounding -- both are the same reduction of the same array -- so a mismatch is a defect in this
reader or in that product, not a result. `cos.f16` is [N, n, T] and reaches ~63 MB per arm at the
paper's full scale, so the cosine half is bounded by `--check-arrays-max-mb` and says when it
skipped rather than quietly passing.
"""

from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.common as R  # noqa: E402

# THE CORPUS-SEARCH COMPARATOR IS M2's, AND THIS FILE DOES NOT READ `topk.jsonl`. Panel a's
# paired rows compare the Exemplifier against the corpus search's per-target top-1 at each nested
# size, and that top-1 lives in `scan/<set>/topk.jsonl`. `results/corpus_search.py` (module M2) is
# the ONE reader of that file; a second reader here would be a second answer to "what was the
# corpus top-1 for row r", differing by whatever the two disagreed about (own-document exclusion,
# which size line, rank-0 vs best-over-ranks) and agreeing loudly in the normal case.
#
# Until that module lands the import fails and every cell that needs it is SKIPPED AND LISTED by
# `corpus_top1_missing`. It is never zero-filled: "the corpus scored 0" and "nobody ran the
# corpus scan" are opposite findings, and a paired difference against a zero-filled comparator is
# the Exemplifier's own number wearing a comparison's name.
try:  # noqa: SIM105 -- the `else` branch is the provenance string, not a pass
    import results.corpus_search as CORPUS_SEARCH  # noqa: E402
except ImportError:
    CORPUS_SEARCH = None

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The cosine columns, and the `per_target.jsonl` prefix each one's best-of-k lives under.
COSINES = {"cos_raw": "bo_", "cos_centred": "bo_c_"}
# How far a recomputation from a stored f16 array may sit from the product's own aggregate before
# it is called a mismatch. `reconstruction/sae_smoke64.py`'s rule, unchanged and for its reason:
# both `score` and `sae_self` reduce a float32 buffer into `per_target` and only then cast the
# per-token array to f16, so this module -- which has nothing but the f16 file -- can land one f16
# ulp away on a healthy product. The bound is therefore RELATIVE (float16 keeps 11 mantissa bits),
# plus the 4-decimal rounding both sides apply. An ABSOLUTE bound is wrong here and was the first
# version of this check: at the 2M dictionary's activation scale (peaks in the tens) it reported
# three mismatches on products `sae_smoke64` had already read as clean.
F16_EPS = 2.0**-11
ROUND_EPS = 1e-4


def reader_tol(a: float, b: float) -> float:
    return F16_EPS * max(abs(float(a)), abs(float(b))) + 2 * ROUND_EPS


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------


def _clusters_for(rows: list[int], ids: dict[int, dict]) -> list:
    """The bootstrap cluster of each row: its document where it has one, else itself.

    A `realact` row carries `doc` and shares it with other rows (122 of 512 on `2026-09-16_v1`),
    so those are one cluster. A `random` or `sae` row has no document and becomes a singleton,
    which makes the clustered bootstrap reduce exactly to the ordinary one for those families.
    """
    return [ids[r].get("doc", None) if ids[r].get("doc") is not None else f"row{r}" for r in rows]


def centred_bok(vol: R.Vol, src: R.Source, max_mb: float) -> tuple[dict[int, dict[int, float]], dict]:
    """({row: {k: bo-k of `cos_centred`}}, how it went) -- computed from `cos_centred.f16`.

    THE PRODUCT DOES NOT CARRY THIS. `score` writes `bo_c_<k>` only when EVERY one of the n
    rollouts has a finite centred best (`score.py:521`, `if len(vals_c) == n`); on the paper's own
    products some rollout of some row always lacks a kept centred token, so the whole ladder is
    absent and `per_target.jsonl` has only `mean_cos_centred` / `max_cos_centred` / `n_centred`.
    That is SMOKES.md's "three things the results run must not get wrong", item 1: the centred bo-k
    columns of plan §2.3 have to come from the array.

    THE SAME ESTIMATOR AS THE STORED RAW LADDER: `results.common.bo_unbiased`, the unbiased
    order statistic over all n draws, which is what `precompute/common.bo_ladder` stores since
    2026-09-23 (M0a). k > n is skipped, never clamped. The one thing this has to decide that
    `score` never did is what a NaN rollout does, and the answer is the one that reduces exactly
    to `score`'s when there are none: the NaN draws are DROPPED and the estimator is applied to
    the m finite ones at the same k, which is the unbiased best-of-k of the draws that exist. A
    row with fewer than k finite draws has no bo-k and is skipped rather than clamped, and a row
    with none at all is absent -- never a one-sided number.

    Where `score` DID store the ladder (a row whose rollouts were all finite) the two are compared,
    and a disagreement is a defect in this reader, not a result.
    """
    entry = src.index.get("cos_centred.f16")
    if entry is None:
        return {}, {"skipped": f"{src.scores_rel} has no cos_centred.f16 (the run centred on nothing)"}
    mb = float(entry.get("bytes", 0)) / 1e6
    if mb > max_mb:
        # LOUD, and never a silent fall-back to the stored ladder that is not there: a centred
        # bo64 column quietly absent at the paper's scale is exactly the failure this guards.
        return {}, {"skipped": f"cos_centred.f16 is {mb:.1f} MB, over --centred-bok-max-mb "
                               f"{max_mb:g}; the centred bo-k columns would be EMPTY, so raise "
                               f"the bound rather than reading the table without them"}
    arr = vol.array(f"{src.scores_rel}/cos_centred.f16", "float16", tuple(entry["shape"]))
    if arr is None:
        return {}, {"skipped": f"{src.scores_rel}/cos_centred.f16 is declared but not on the volume"}
    order = [int(r) for r in src.rows_meta.get("rows", [])]
    out: dict[int, dict[int, float]] = {}
    n_nan_rollouts = 0
    rows_with_nan = 0
    checked = mism = 0
    worst = -math.inf
    for i, row in enumerate(order):
        best = R.best_per_rollout(arr[i].astype(np.float32), empty=float("nan"))
        finite = np.isfinite(best)
        if not finite.any():
            continue                       # a non-centrable row: absent, never a one-sided number
        if not finite.all():
            rows_with_nan += 1
            n_nan_rollouts += int((~finite).sum())
        n = int(best.size)
        live = best[finite]
        cells = {k: v for k, v in R.bo_ladder(live, R.BO_KS_ALL).items() if k <= n}
        if cells:
            out[row] = cells
        pt = src.per_target.get(row) or {}
        for k, v in cells.items():
            stored = pt.get(f"bo_c_{k}")
            if stored is None:
                continue
            d = abs(v - float(stored))
            worst = max(worst, d - reader_tol(v, float(stored)))
            mism += int(d > reader_tol(v, float(stored)))
            checked += 1
    info = {"rows": len(out), "rows_with_nan_rollouts": rows_with_nan,
            "nan_rollouts": n_nan_rollouts, "stored_comparisons": checked,
            "n_mismatches": mism, "worst_excess": round(worst, 6) if checked else None,
            "product": f"{src.scores_rel}/cos_centred.f16", "mb": round(mb, 2)}
    return out, info


def cosine_cells(fam: R.Family, rows: list[int], src: R.Source, ids: dict[int, dict],
                 boot: int, seed: int, centred: dict[int, dict[int, float]] | None = None) -> list[dict]:
    """One dict per (family, source, cosine): the bo-k means with clustered SEs, or [].

    `centred` is `centred_bok`'s per-row ladder. Where it is given it is the ONLY source of the
    centred numbers -- one estimator, one array -- and the product's own `bo_c_<k>`, on the rows
    that happen to carry it, is a cross-check made inside `centred_bok` rather than a second
    supply. Where it is absent (no `cos_centred.f16`, or the size bound refused it) the stored
    ladder is read as before, so a run that never needed the array behaves exactly as it did.
    """
    present = [r for r in rows if r in src.per_target]
    if not present:
        return []
    clusters = _clusters_for(present, ids)
    out = []
    for cosine, prefix in COSINES.items():
        derived = centred if (cosine == "cos_centred" and centred) else None
        cells: dict[int, dict] = {}
        for k in R.BO_KS_ALL:
            key = f"{prefix}{k}"
            vals, cl = [], []
            for r, c in zip(present, clusters, strict=True):
                v = derived.get(r, {}).get(k) if derived is not None else src.per_target[r].get(key)
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
                    cl.append(c)
            if not vals:
                continue
            mean, se, n_items, n_clusters = R.cluster_bootstrap(vals, cl, boot, seed)
            cells[k] = {"mean": mean, "se": se, "se_iid": R.se_iid(vals),
                        "n_rows": n_items, "n_clusters": n_clusters}
        if not cells:
            continue
        out.append({
            "family": fam.label, "source": src.label, "cosine": cosine,
            "n": src.n, "engine": src.engine, "run_tag": src.run_tag, "maemm": src.maemm,
            "mu": src.mu, "bo": cells,
            # WHERE THE NUMBERS CAME FROM, carried into the CSV and the caption. The centred
            # ladder is `cos_centred.f16` recomputed here; the raw one is what `score` stored.
            "bo_source": "cos_centred.f16 (recomputed)" if derived is not None else "per_target.jsonl",
            "n_rows": max(c["n_rows"] for c in cells.values()),
            "n_clusters": max(c["n_clusters"] for c in cells.values()),
        })
    return out


# ---------------------------------------------------------------------------------------------
# the corpus-peak denominator, as a PARAMETER
# ---------------------------------------------------------------------------------------------
#
# Panel b's quantity is a feature's peak activation on its own rollouts OVER THAT FEATURE'S OWN
# CORPUS PEAK, and which corpus the denominator is taken on is most of what the ratio means. Until
# 2026-09-22 there was one answer and it was wired in: `sae_self` records `corpus_peak` per feature
# from `base/<base>/sae/<sae>/max_act.f16`, our 16M HELD-OUT scan (`autointerp/sae_self.py:494`),
# and `sae_cells` divided by that. Spec §1.2 and §1.4 and writing plan §2 move the denominator to
# the feature's peak on the 10M TRAINING corpus (`celeste-train10m`), which is a different number
# on a different text, produced by a different product -- M2's `top1_act` scan -- and which does
# not exist on the volume yet.
#
# So the denominator is a PARAMETER carrying a provenance string, and that string travels into
# every ratio cell, every caption and every CSV row it reaches. A ratio whose table does not say
# which corpus its denominator came from is unreadable a month later, and the two answers are not
# close: the 16M held-out peak and the 10M training peak are maxima over different texts.
#
# ASKED FOR AND ABSENT RAISES. It does not fall back to `stored`. A ratio against the wrong
# denominator prints as a perfectly ordinary number, is wrong by whatever the two corpora differ
# by, and nothing downstream can tell -- which is exactly the defect this parameter exists to
# prevent. An absent product is a stopped run; a silently substituted one is a wrong paper.

CORPUS_PEAK_STORED = "stored"
CORPUS_PEAK_TOP1 = "top1_act:"
# What `sae_self` itself recorded, named once so the provenance string is not written twice.
STORED_PEAK_PROVENANCE = (
    "sae_self's own `corpus_peak` -- our 16M HELD-OUT scan's `sae/<sae>/max_act.f16` "
    "(autointerp/sae_self.py:494), NOT the 10M training corpus the spec asks for"
)


class CorpusPeaks:
    """The per-feature ratio denominator and where it came from, resolved once per run.

    `of(row, stored)` answers for one feature row and returns `(peak, provenance)`. Under
    `stored` it hands back what `sae_self` recorded, unchanged, so a run that asks for nothing
    behaves exactly as this file did before. Under a `top1_act` source it hands back that
    product's number for the row -- and, for a row the product does not carry, NaN plus a note in
    `missing_rows`, so the feature drops out of the median with a count rather than quietly
    keeping a denominator from the other corpus.

    WHAT `top1_act` ACTUALLY MEASURES, and why this is flagged rather than assumed: `act_max` is
    the pre-gate activation of the feature on the COSINE top-1 corpus window (`top1_act.py:1-9`),
    which is a LOWER BOUND on the feature's peak over that corpus -- the window the cosine picked
    need not be the window the activation peaks on, and `top1_act`'s own summary reports that most
    cosine top-1 windows are not activation examples of their own feature at all. The feature's
    true peak on a corpus is a scan's `max_act.f16` over that corpus. The spec names `top1_act` as
    the producer (plan §2 M2) and the writing plan names "the feature's peak on the 10M training
    corpus" as the quantity; those are two different numbers and the provenance string says which
    one a cell was built from, so the table cannot be misread whichever M2 lands.
    """

    def __init__(self, spec: str, by_row: dict[int, float] | None, provenance: str, rel: str = "",
                 set_name: str = "", sae_key: str = ""):
        self.spec = spec
        self.by_row = by_row
        self.provenance = provenance
        self.rel = rel
        # WHAT THE ROW NUMBERS MEAN, read off the product's own `summary.json`. `by_row` is keyed
        # by the SET ROW of the set `top1_act` ran on and a row number means nothing outside it.
        self.set_name = set_name
        self.sae_key = sae_key
        self.missing_rows: list[int] = []

    @property
    def is_stored(self) -> bool:
        return self.by_row is None

    def applies_to(self, set_name: str, sae_key: str) -> tuple[bool, str]:
        """(may this product answer for that family, why not) -- the SET and the DICTIONARY.

        THE ROW JOIN IS NOT SELF-DESCRIBING. `2026-09-21_v3_ctrl`'s 131k features sit at rows
        512-1023 and `2026-09-21_v3_sae2m`'s decoder half sits at exactly the same numbers, so a
        `top1_act` product of the first answers every question the second asks -- and answers all
        512 of them with ANOTHER DICTIONARY's feature, silently, in a cell that prints as a
        perfectly ordinary ratio. `discover_families` already refuses to guess a dictionary from a
        feature id for this exact reason ("every 131k id is also a valid 2M index"); the
        denominator is the same join and gets the same refusal.

        `stored` needs no check: `sae_self` records each feature's `corpus_peak` in the family's
        own product, so it cannot be another set's number. A product whose `summary.json` states
        neither field is REFUSED rather than trusted -- an absent check is not a passed one.
        """
        if self.is_stored:
            return True, ""
        if not self.set_name or not self.sae_key:
            return False, (
                f"`{self.rel}` states no `set`/`sae` in its summary.json, so which set its row "
                f"numbers index is unstated and a row join onto `{set_name}` cannot be checked")
        if self.set_name != set_name or self.sae_key != sae_key:
            return False, (
                f"`{self.rel}` is indexed by the rows of set `{self.set_name}`, dictionary "
                f"`{self.sae_key}`; this family is set `{set_name}`, dictionary `{sae_key}`. The "
                f"row numbers overlap and mean different features, so the denominator is ABSENT "
                f"here rather than taken from the other product")
        return True, ""

    def of(self, row: int, stored: float) -> tuple[float, str]:
        if self.by_row is None:
            return float(stored), self.provenance
        v = self.by_row.get(int(row))
        if v is None:
            self.missing_rows.append(int(row))
            return float("nan"), self.provenance
        return float(v), self.provenance


def corpus_peaks(vol: R.Vol, spec: str) -> CorpusPeaks:
    """Resolve `--corpus-peak` into a `CorpusPeaks`, or RAISE naming the path that is not there.

    Two forms, and no third:

      `stored`                     what `sae_self` recorded, the 16M held-out `max_act` -- today's
                                   behaviour, kept as the default so nothing already written moves
      `top1_act:<volume path>`     a `top1_act` product directory, volume-relative, e.g.
                                   `base/qwen36-27b/sae/l42-1b/top1_act/2026-09-21_v3_ctrl`;
                                   its `top1_act.jsonl` is read and each row's `act_max` becomes
                                   that feature's denominator

    A bare path (no prefix) is accepted as the `top1_act` form, because that is the only path form
    there is and refusing it would be a spelling trap rather than a safety one.
    """
    spec = (spec or CORPUS_PEAK_STORED).strip()
    if spec == CORPUS_PEAK_STORED:
        return CorpusPeaks(spec, None, STORED_PEAK_PROVENANCE)
    rel = (spec[len(CORPUS_PEAK_TOP1):] if spec.startswith(CORPUS_PEAK_TOP1) else spec).strip("/")
    assert rel, (
        f"--corpus-peak {spec!r} names no path. Give `{CORPUS_PEAK_STORED}` or "
        f"`{CORPUS_PEAK_TOP1}<volume-relative top1_act directory>`"
    )
    recs = vol.jsonl(f"{rel}/top1_act.jsonl")
    # LOUD, and never a fall-back to `stored`: see the section comment above. The path is named in
    # full because the usual cause is a product that has not landed yet and the next question is
    # always "under which directory was M2 told to write it".
    assert recs is not None, (
        f"--corpus-peak asked for `{rel}` and {vol.prefix or '/'}/{rel}/top1_act.jsonl is not on "
        f"the volume. That product is M2's `top1_act` scan on `celeste-train10m`; until it lands "
        f"there is no 10M denominator, and this run REFUSES to divide by the 16M held-out peak "
        f"under a 10M label. Run it, or pass --corpus-peak {CORPUS_PEAK_STORED} and accept that "
        f"every ratio cell will be labelled as the 16M held-out number it is"
    )
    by_row: dict[int, float] = {}
    for r in recs:
        v = r.get("act_max")
        if v is None:
            continue
        by_row[int(r["row"])] = float(v)
    assert by_row, f"{rel}/top1_act.jsonl carries {len(recs)} rows and not one `act_max`"
    summary = vol.json(f"{rel}/summary.json") or {}
    size = summary.get("corpus_size_m")
    prod_set, prod_sae = str(summary.get("set") or ""), str(summary.get("sae") or "")
    prov = (
        f"`{rel}` top1_act.jsonl `act_max`: the pre-gate activation of the COSINE top-1 corpus "
        f"window of each feature"
        + (f" at {size}M" if size is not None else "")
        + (f", dictionary `{summary['sae']}`" if summary.get("sae") else "")
        + (f", set `{summary['set']}`" if summary.get("set") else "")
        + f" ({len(by_row)} features). A lower bound on the feature's peak over that corpus, not "
          f"a scan max_act"
    )
    return CorpusPeaks(spec, by_row, prov, rel, prod_set, prod_sae)


def sae_cells(fam: R.Family, rows: list[int], src: R.Source, ids: dict[int, dict],
              vol: R.Vol, peak_src: CorpusPeaks | None = None,
              set_name: str = "") -> tuple[list[dict], list[dict], dict]:
    """(aggregate rows incl. per stratum, per-feature rows, the reader check) for one SAE family.

    Per feature, `peaks_of` gives one peak activation per rollout; `bo_ladder` turns those
    into the same disjoint-group best-of-k the cosine side reports, and every ratio divides by
    that feature's own corpus peak -- WHICHEVER corpus `peak_src` names (`corpus_peaks` above; the
    default is `sae_self`'s own 16M `max_act`, which is what this did before the denominator
    became a parameter). The provenance string comes back on every aggregate, every per-feature
    row and the reader check, because a ratio without its denominator's corpus is not a number.

    WHICH PRODUCT: `sae_self` writes the encoder half to `sae_self/` (its name since the stage
    existed) and any other side to `sae_self__<side>/`, because both halves of a `--sides enc,dec`
    set live in ONE scores directory. So a `sae_side: dec` family is read from `sae_self__dec`
    and, if that is absent, from `sae_self` -- where the row filter below finds none of its rows
    and the family is reported missing. The fallback therefore cannot report the encoder block's
    numbers under a decoder label; it exists so a set with no side field at all still reads.
    """
    rel = f"{src.scores_rel}/sae_self"
    meta, act = (None, "")
    if fam.sae_side and fam.sae_side != "enc":
        rel = f"{src.scores_rel}/sae_self__{fam.sae_side}"
        meta, act = R.load_sae_self(vol, rel)
        if meta is None:
            rel = f"{src.scores_rel}/sae_self"
            meta, act = (None, "")
    if meta is None:
        meta, act = R.load_sae_self(vol, rel)
    if meta is None or isinstance(act, str):
        return [], [], {"absent": act if isinstance(act, str) else rel}
    gate = float(meta["gate"])
    denom = peak_src or CorpusPeaks(CORPUS_PEAK_STORED, None, STORED_PEAK_PROVENANCE)
    # A `top1_act` product of ANOTHER set indexes rows this family also has, so it is refused per
    # family and the ratios become ABSENT rather than another dictionary's number. Every cell
    # without a denominator (the ratios) is then skipped and listed; the fired cells have no
    # denominator and are unaffected, which is the split the 2M block needs.
    applies, why = denom.applies_to(set_name, fam.sae_key)
    if not applies:
        denom = CorpusPeaks(denom.spec, {}, f"NO DENOMINATOR -- {why}", denom.rel,
                            denom.set_name, denom.sae_key)
    stored = {int(p["row"]): p for p in meta.get("per_target", [])}
    want = set(rows)
    feats: list[dict] = []
    mism = 0
    worst = -math.inf
    for i, row in enumerate(meta["rows"]):
        row = int(row)
        if row not in want:
            continue
        peaks = R.peaks_of(act[i])
        cp, cp_from = denom.of(row, stored.get(row, {}).get("corpus_peak", 0.0))
        bo = R.bo_ladder(peaks, R.BO_KS_ALL)
        # THE FIRED INDICATOR THROUGH THE SAME ESTIMATOR (M0a). "fired at best-of-k" is
        # P(at least one of k draws is above the gate), and for a 0/1 vector the unbiased
        # order-statistic best-of-k IS that probability: 1 - C(n - m, k)/C(n, k) with m the
        # number of draws above the gate. At k = 1 it is the plain rate this used to report and
        # at k = n it is `fired_any`, so nothing that was right before moved.
        fired = R.bo_ladder((peaks > gate).astype(np.float64), R.BO_KS_ALL)
        # The reader check of sae_smoke64, kept: our reduction of the array against the product's
        # own per_target block. Both are the same reduction of the same numbers.
        for ours, key in ((float(peaks.mean()), "mean_peak_act"), (float(peaks.max()), "max_peak_act")):
            theirs = stored.get(row, {}).get(key)
            if theirs is None:
                continue
            d = abs(ours - float(theirs))
            worst = max(worst, d - reader_tol(ours, float(theirs)))
            mism += int(d > reader_tol(ours, float(theirs)))
        feats.append({
            "row": row, "feature": int(meta["features"][i]), "stratum": ids[row].get("stratum"),
            # The stratum STATISTIC, carried from the row that the draw stamped it on. The 131k
            # draw's strata are quartiles of log10 POOL PEAK ACTIVATION (`draw_sae131k.py:35,91`)
            # and the 2M draw's are quartiles of log10 FIRE COUNT (`draw_sae2m._stratified_draw`),
            # while spec §1.2 and panel b both say "corpus-frequency quartile". Those are not the
            # same cut. The name travels with every cell so the caption can say which one it is
            # rather than the reader assuming the spec's.
            "stratum_stat": ids[row].get("stratum_stat"),
            "corpus_peak": cp, "corpus_peak_from": cp_from, "n": int(meta["n"]),
            "ratio": {k: (v / cp if (cp > 0 and math.isfinite(cp)) else float("nan"))
                      for k, v in bo.items()},
            "bo": bo,
            "fired": fired,
            "item_fired": fired.get(1, float("nan")),
            "fired_any": bool(peaks.max() > gate),
        })
    check = {"rows": len(feats), "n_mismatches": mism, "worst_excess": round(worst, 6),
             "gate": gate, "product": rel, "corpus_peak_source": denom.provenance,
             "n_no_corpus_peak": sum(1 for f in feats
                                     if not (f["corpus_peak"] > 0
                                             and math.isfinite(f["corpus_peak"])))}
    if not feats:
        return [], [], {"absent": f"{rel} carries none of this family's rows"}

    def agg(sub: list[dict], stratum) -> dict:
        ks = sorted({k for f in sub for k in f["ratio"]})
        per_k = {}
        for k in ks:
            vals = [f["ratio"][k] for f in sub if k in f["ratio"] and math.isfinite(f["ratio"][k])]
            if vals:
                per_k[k] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                            "se_iid": R.se_iid(vals), "n": len(vals)}
        return {
            "family": fam.label, "source": src.label, "stratum": stratum,
            "n": src.n, "engine": src.engine, "run_tag": src.run_tag, "maemm": src.maemm,
            "product": rel,
            "sae_key": fam.sae_key, "sae_side": fam.sae_side,
            "n_features": len(sub), "gate": gate, "per_k": per_k,
            # WHERE THE DENOMINATOR CAME FROM, on every aggregate: the ratio cells this feeds
            # print it in `cells.csv`'s `note` and in the table caption, and `n_no_denominator`
            # is how many features of this stratum had no peak in that source at all -- absent
            # from the median rather than divided by the other corpus's number.
            "corpus_peak_source": denom.provenance,
            "corpus_peak_spec": denom.spec,
            "n_no_denominator": sum(1 for f in sub
                                    if not (f["corpus_peak"] > 0
                                            and math.isfinite(f["corpus_peak"]))),
            "stratum_stat": next((f["stratum_stat"] for f in sub if f.get("stratum_stat")), None),
            # The fired LADDER, averaged over the features of this stratum: one cell per k, which
            # is what panel b's `sae.l131k.ex.fired.bo<k>.q<q>` keys print. `item_fired` is its
            # k = 1 cell and `feat_firing` its k = n cell, kept under their old names so nothing
            # reading this dict had to move.
            "fired_k": {
                k: float(np.mean([f["fired"][k] for f in sub if k in f["fired"]]))
                for k in sorted({k for f in sub for k in f["fired"]})
            },
            "item_fired": float(np.mean([f["item_fired"] for f in sub])),
            "feat_firing": float(np.mean([f["fired_any"] for f in sub])),
        }

    out = [agg(feats, None)]
    strata = sorted({f["stratum"] for f in feats if f["stratum"] is not None})
    for s in strata:
        out.append(agg([f for f in feats if f["stratum"] == s], s))
    return out, feats, check


def cosine_reader_check(vol: R.Vol, src: R.Source, max_mb: float) -> dict:
    """Recompute `mean_cos` (and the centred one) from `cos.f16` and compare with per_target.

    Bounded by size: the array is [N, n, T] and at the paper's scale each arm's is ~63 MB, which
    is not worth downloading to re-derive a number the product already stored. Over the bound the
    check SKIPS and says so -- it is never relaxed into a vacuous pass.
    """
    entry = src.index.get("cos.f16")
    if entry is None:
        return {"skipped": f"{src.scores_rel}/index.json declares no cos.f16"}
    mb = float(entry.get("bytes", 0)) / 1e6
    if mb > max_mb:
        return {"skipped": f"cos.f16 is {mb:.1f} MB, over --check-arrays-max-mb {max_mb:g}"}
    order = [int(r) for r in src.rows_meta.get("rows", [])]
    shape = tuple(entry["shape"])
    checked = 0
    mism = 0
    worst = -math.inf
    for name, agg_key, empty in (("cos.f16", "mean_cos", -1.0),
                                 ("cos_centred.f16", "mean_cos_centred", float("nan"))):
        if name not in src.index:
            continue
        arr = vol.array(f"{src.scores_rel}/{name}", "float16",
                        tuple(src.index[name]["shape"]) if name in src.index else shape)
        if arr is None:
            continue
        for i, row in enumerate(order):
            pt = src.per_target.get(row)
            if pt is None or agg_key not in pt:
                continue
            best = R.best_per_rollout(arr[i].astype(np.float32), empty=empty)
            best = best[np.isfinite(best)]
            if not len(best):
                continue
            d = abs(float(best.mean()) - float(pt[agg_key]))
            worst = max(worst, d - reader_tol(float(best.mean()), float(pt[agg_key])))
            mism += int(d > reader_tol(float(best.mean()), float(pt[agg_key])))
            checked += 1
    if not checked:
        return {"skipped": f"{src.scores_rel}: no (array, aggregate) pair to compare"}
    return {"rows": checked, "n_mismatches": mism, "worst_excess": round(worst, 6),
            "product": src.scores_rel}


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------


def load_exclusions(vol: R.Vol, base: str, set_name: str) -> dict:
    """The set's own `exclusions.json`, or an empty record when it has none.

    A v3 block that has one carries it as a first-class product of the set, written at freeze time
    and never recomputed: `_realact`'s 26 rows are Ari's `ngram_overlap --side hers` list at
    coverage >= 0.05, n = 7, over HER training parquets, and `_ours`'s 6 are the rows of our own
    draw that `check_v2_targets_overlap` found fully reproduced in her v2 text at n = 13. They are
    DIFFERENT INSTRUMENTS over different corpora and the file says so; the two lists must never be
    applied to each other's block, which is exactly why they live beside the rows they index
    rather than in this file.

    The rows are KEPT in the set so every arm pairs row for row; dropping them is the reader's job
    and this is the reader.
    """
    rec = vol.json(f"base/{base}/heldout/{set_name}/exclusions.json")
    if rec is None:
        return {"rows": set(), "present": False,
                "why": f"base/{base}/heldout/{set_name} carries no exclusions.json"}
    rows = {int(r) for r in rec.get("excluded_rows", [])}
    return {"rows": rows, "present": True, "n_total": int(rec.get("rows_total", 0)),
            "n_headline": int(rec.get("n_headline", 0)), "block": rec.get("block"),
            "criterion": rec.get("criterion"), "n_gram": rec.get("n_gram"),
            "source": rec.get("source"), "computed_over": rec.get("computed_over")}


def analyse(vol: R.Vol, cfg: dict, set_name: str, sources: str, boot: int, seed: int,
            check_arrays: bool, check_arrays_max_mb: float,
            centred_bok_max_mb: float = 128.0, apply_exclusions: bool = True,
            corpus_peak: str = CORPUS_PEAK_STORED) -> dict:
    """Everything the tables and figures are built from. Never raises on a missing source.

    The one thing it DOES raise on is `corpus_peak` naming a source that is not on the volume
    (`corpus_peaks`): a missing arm is a normal outcome and a missing ratio denominator is not,
    because the second one has a wrong answer available and the first one does not.
    """
    entry = (cfg.get("heldout") or {}).get(set_name)
    assert entry is not None, (
        f"--set {set_name!r} is not declared in config.yaml `heldout:`. Declare it (the products "
        f"cannot be read without its families and its sae_key) or pass a declared set: "
        f"{sorted(cfg.get('heldout') or {})}"
    )
    # Which base a set was drawn on is not a `heldout:` key -- the same set name can exist under
    # two bases -- so it is resolved by asking the volume which `base/<b>/heldout/<set>` is there.
    # An explicit `base:` in the entry wins; more than one match is refused rather than picked.
    bases = sorted(cfg.get("bases") or {})
    if entry.get("base"):
        base = str(entry["base"])
    else:
        hits = [b for b in bases if vol.exists(f"base/{b}/heldout/{set_name}/ids.jsonl")]
        assert len(hits) == 1, (
            f"`{set_name}` is under {len(hits)} of the declared bases ({hits or bases}) at root "
            f"{vol.prefix or '/'}: name the base with `heldout.{set_name}.base` in config.yaml")
        base = hits[0]
    ids, declared_sae = R.load_ids(vol, base, set_name, cfg)
    excl = load_exclusions(vol, base, set_name)
    # The excluded rows are dropped from the FAMILY MAP, which is the single place row membership
    # is decided -- so they leave the tables, the figures, the sanity registry, the SE clusters
    # and the CSVs together, and there is no path by which one of those keeps them. They stay in
    # `ids`, because the products are still indexed by the set's own row numbers.
    drop = excl["rows"] if apply_exclusions else set()
    fams = {f: [r for r in rows if r not in drop]
            for f, rows in R.families_of(ids, cfg, declared_sae).items()}
    fams = {f: rows for f, rows in fams.items() if rows}

    # Resolved BEFORE any product is read, so a run that asked for a denominator it cannot have
    # stops before it has printed anything, rather than after a table is already on disk.
    peak_src = corpus_peaks(vol, corpus_peak)

    found, absent = R.discover_sources(vol, cfg, base, set_name)
    colours = R.colour_map(found)
    missing = [f"`{k}`: no scores directory for `{set_name}` under `maemms/{k}/scores/`"
               for k in absent]
    wanted = [s.strip() for s in sources.split(",") if s.strip()]
    usable: list[R.Source] = []
    for src in found:
        if wanted and not any(w in src.label for w in wanted):
            continue
        why = R.load_source(vol, src)
        if why:
            missing.append(f"`{src.label}`: {why}")
            continue
        usable.append(src)

    # The centred best-of-k ladder, per SOURCE rather than per family: it is one read of one
    # array and every family of that arm cuts rows out of it.
    centred: dict[str, dict[int, dict[int, float]]] = {}
    cos_rows: list[dict] = []
    sae_rows: list[dict] = []
    sae_feats: list[dict] = []
    checks: list[dict] = []
    notes: list[str] = []
    # A set none of whose families is centrable (`_ctrl`, `_sae2m`, `_subspace`) may still have a
    # `cos_centred.f16` on some arms. Whether it is worth reading depends on WHEN the product was
    # scored, not on the config: before 2026-09-23 such a row was NaN there (absent from the
    # centred aggregates), and since then it carries the ONE-SIDED cos(h - score_mu, unit(d)) and
    # `per_target.jsonl` says so per row with `centred_sided`. So the decision is taken from the
    # PRODUCT -- an old one is skipped and ~12 MB per arm is not fetched to produce nothing, a new
    # one is read -- and never from `family_kinds:` alone, which no longer decides it.
    kinds = cfg.get("family_kinds") or {}
    centrable = sorted({f.family for f in fams if (kinds.get(f.family) or {}).get("centrable")})
    for src in usable:
        one_sided_scored = any(r.get("centred_sided") == 1 for r in src.per_target.values())
        if not centrable and not one_sided_scored:
            checks.append({"kind": "cos_centred bo-k", "source": src.label, "family": "(all)",
                           "skipped": f"no family of `{set_name}` is centrable "
                                      f"({sorted({f.family for f in fams})}) and `{src.label}`'s "
                                      f"per_target.jsonl declares no `centred_sided: 1` row, so it "
                                      f"was scored before 2026-09-23 and its centred cosine is NaN "
                                      f"on every row -- the array is not read"})
            centred[src.label] = {}
            continue
        ladder, info = centred_bok(vol, src, centred_bok_max_mb)
        centred[src.label] = ladder
        checks.append({"kind": "cos_centred bo-k", "source": src.label, "family": "(all)", **info})
    for fam, rows in fams.items():
        dict_family = R.is_dictionary(fam, cfg)
        if dict_family and not fam.sae_key:
            notes.append(
                f"family `{fam.label}`: its rows carry no `sae_key` and the set declares none, so "
                f"which dictionary the feature ids index is unstated -- activation metrics skipped "
                f"(every 131k id is also a valid 2M index; guessing is silently wrong)")
            continue
        for src in usable:
            if dict_family:
                aggs, feats, chk = sae_cells(fam, rows, src, ids, vol, peak_src, set_name)
                sae_rows += aggs
                for f in feats:
                    sae_feats.append({**f, "family": fam.label, "source": src.label})
                if "absent" in chk:
                    missing.append(f"`{src.label}` / `{fam.label}`: {chk['absent']}")
                else:
                    checks.append({"kind": "sae_self", "source": src.label, "family": fam.label, **chk})
            else:
                cells = cosine_cells(fam, rows, src, ids, boot, seed, centred.get(src.label))
                if not cells:
                    missing.append(
                        f"`{src.label}` / `{fam.label}`: scored none of this family's "
                        f"{len(rows)} rows")
                cos_rows += cells
    if check_arrays:
        for src in usable:
            chk = cosine_reader_check(vol, src, check_arrays_max_mb)
            checks.append({"kind": "cos", "source": src.label, "family": "(all)", **chk})

    return {
        "set": set_name, "base": base, "root": vol.prefix or "/vol",
        "ids": ids, "families": fams, "sources": usable, "all_sources": found,
        "colours": colours, "cos": cos_rows, "sae": sae_rows, "sae_feats": sae_feats,
        "checks": checks, "missing": missing, "notes": notes,
        "declared_sae_key": declared_sae, "boot": boot, "seed": seed,
        "exclusions": {**excl, "rows": sorted(excl["rows"]), "applied": bool(drop)},
        # The PER-ROW centred ladders, kept rather than dropped after the aggregates were taken.
        # Panel a's paired cells (`fid.ra.diff.cos.bo8`, `fid.ra.ex.win.bo8`, `fid.ra.nla.dex`)
        # are comparisons on identical targets, so they need the row values and not the means --
        # spec §1.2, "paired, not per-row: per-row SEs do not carry it". This is the same object
        # `cosine_cells` aggregated, not a second computation of it.
        "centred": centred,
        "corpus_peak": {"spec": peak_src.spec, "provenance": peak_src.provenance,
                        "product": peak_src.rel,
                        "rows_without_a_peak": sorted(set(peak_src.missing_rows))},
    }


def stat_registry(res: dict) -> dict[tuple, float]:
    """{(family, source, stratum, metric): value} -- what `sanity.yaml` selects against.

    Flat on purpose: a gate names a family, a source, an optional stratum and a metric, and the
    YAML is then readable by whoever edits it without knowing this file's data structures.
    """
    out: dict[tuple, float] = {}
    for c in res["cos"]:
        for k, cell in c["bo"].items():
            out[(c["family"], c["source"], None, f"{c['cosine']}.bo{k}")] = cell["mean"]
    for a in res["sae"]:
        for k, cell in a["per_k"].items():
            out[(a["family"], a["source"], a["stratum"], f"ratio.bo{k}.median")] = cell["median"]
            out[(a["family"], a["source"], a["stratum"], f"ratio.bo{k}.mean")] = cell["mean"]
        for k, v in a.get("fired_k", {}).items():
            out[(a["family"], a["source"], a["stratum"], f"fired.bo{k}")] = v
        out[(a["family"], a["source"], a["stratum"], "fired.item")] = a["item_fired"]
        out[(a["family"], a["source"], a["stratum"], "fired.feature")] = a["feat_firing"]
    return out


def support_registry(res: dict) -> dict[tuple, int]:
    """{(family, source, stratum): how many rows or features are behind its statistics}.

    Carried into the sanity table because a gate written against a 512-row draw and answered by a
    6-row smoke is not a disagreement about the pipeline, and the reader has to see which it is.
    """
    out: dict[tuple, int] = {}
    for c in res["cos"]:
        out[(c["family"], c["source"], None)] = c["n_rows"]
    for a in res["sae"]:
        out[(a["family"], a["source"], a["stratum"])] = a["n_features"]
    return out


def cross_set(vol: R.Vol | None, cfg: dict | None, res: dict, chk: dict) -> dict:
    """One `kind: cross_set` gate: THIS set's arm against ANOTHER set's, on the rows they share.

    The re-derive case is what this exists for. `2026-09-21_v1raw` is `2026-09-16_v1` re-forwarded
    row for row, so a row index means the same target in both -- but only for a family whose
    stored direction is the same in both. For a NON-CENTRABLE family (`sae`, `random`) it is: the
    old set stores `unit(W_enc[:, f])` and the new one stores the identical vector, so `cos_raw`
    is the same statistic and a difference is a real difference between the two runs. For
    `realact` it is NOT: `2026-09-16_v1` is `storage: unit` with its realact rows already centred
    on `stats/mu.f32`, so its `cos` has a centred TARGET and a raw scorer while a raw set's `cos`
    has neither side centred. Write that one with `compare: false`, not with a wide tolerance.

    Only cosine metrics are resolvable: the SAE activation metrics would need the other set's
    `sae_self` product, which is a second fetch this gate does not make.
    """
    if vol is None or cfg is None:
        return {"why": "cross-set gates need the volume; this run has none"}
    metric = str(chk.get("metric", ""))
    if not metric.startswith(("cos_raw.bo", "cos_centred.bo")):
        return {"why": f"`{metric}` is not a cosine metric; a cross-set gate resolves only those"}
    cosine, _, kpart = metric.partition(".")
    k = int(kpart[2:])
    other_set = str(chk.get("from_set", ""))
    if not other_set:
        return {"why": "a cross-set gate needs `from_set`"}
    entry = (cfg.get("heldout") or {}).get(other_set)
    if entry is None:
        return {"why": f"`from_set: {other_set}` is not declared in config.yaml `heldout:`"}
    base = res["base"]
    other_ids = vol.jsonl(f"base/{base}/heldout/{other_set}/ids.jsonl")
    if other_ids is None:
        return {"why": f"base/{base}/heldout/{other_set}/ids.jsonl is not on the volume"}
    other_ids = {int(r["row"]): r for r in other_ids}
    found, _absent = R.discover_sources(vol, cfg, base, other_set)
    want = str(chk.get("from_source", chk.get("source", "")))
    hits = [x for x in found if want in x.label]
    if len(hits) != 1:
        return {"why": f"`from_source: {want}` matched {len(hits)} arm(s) of `{other_set}` "
                       f"({', '.join(x.label for x in found) or 'none'})"}
    other = hits[0]
    why = R.load_source(vol, other)
    if why:
        return {"why": why}

    mine = next((x for x in res["sources"] if str(chk.get("source", "")) in x.label), None)
    if mine is None:
        return {"why": f"`source: {chk.get('source')}` matched none of this run's arms"}
    fam = str(chk.get("family"))
    prefix = COSINES[cosine]
    rows = [r for r in sorted(set(mine.per_target) & set(other.per_target))
            if R.family_of(res["ids"][r], res["declared_sae_key"]).label == fam
            and r in other_ids
            and f"{prefix}{k}" in mine.per_target[r] and f"{prefix}{k}" in other.per_target[r]]
    if not rows:
        return {"why": f"`{fam}` has no row carrying `{prefix}{k}` in BOTH "
                       f"`{mine.label}` on `{res['set']}` and `{other.label}` on `{other_set}`"}
    ours = float(np.mean([mine.per_target[r][f"{prefix}{k}"] for r in rows]))
    theirs = float(np.mean([other.per_target[r][f"{prefix}{k}"] for r in rows]))
    return {"ours": ours, "theirs": theirs, "n": len(rows),
            "label": f"{other.label} on {other_set}"}


def run_sanity(res: dict, path: Path, vol: R.Vol | None = None,
               cfg: dict | None = None) -> list[dict]:
    """Resolve every check in the YAML against the computed statistics. Flags, never stops."""
    with open(path) as fh:
        spec = yaml.safe_load(fh) or {}
    reg = stat_registry(res)
    support = support_registry(res)
    labels = [s.label for s in res["sources"]]
    out = []
    for chk in spec.get("checks") or []:
        name = str(chk.get("name", "(unnamed)"))
        rec = {"name": name, "expect": chk.get("expect"), "tol": chk.get("tol"),
               "provenance": " ".join(str(chk.get("provenance", "")).split()),
               "compare": chk.get("compare", True)}
        # `set:` (optional) restricts a gate to ONE block. Without it a gate resolves on every
        # block that happens to carry its family and its source -- which is how the v1raw
        # four-row smoke's 0.7518 came to be compared against the 512-row `2026-09-21_v3_realact`
        # number and FLAG: two different populations, one gate, and the flag said nothing about
        # the mu. A gate that names a block is `absent` on every other one, which is the correct
        # and quiet outcome.
        want_set = str(chk.get("set", ""))
        if want_set and want_set != res["set"]:
            rec["verdict"] = "absent"
            rec["family"] = str(chk.get("family", ""))
            rec["source"] = str(chk.get("source", ""))
            rec["metric"] = str(chk.get("metric", ""))
            rec["why"] = f"`set: {want_set}` — this run is `{res['set']}`"
            out.append(rec)
            continue
        if str(chk.get("kind", "")) == "cross_set":
            got = cross_set(vol, cfg, res, chk)
            rec["family"] = str(chk.get("family"))
            rec["source"] = str(chk.get("source", ""))
            rec["metric"] = str(chk.get("metric", ""))
            if "why" in got:
                rec["verdict"] = "absent"
                rec["why"] = got["why"]
                out.append(rec)
                continue
            rec.update(ours=got["ours"], expect=got["theirs"], n=got["n"])
            rec["diff"] = got["ours"] - got["theirs"]
            if chk.get("compare") is False:
                rec["verdict"] = "no verdict"
                rec["why"] = (f"vs {got['label']} on {got['n']} shared rows "
                              f"(declared not the same statistic)")
            else:
                tol = float(chk.get("tol", 0.01))
                rec["tol"] = tol
                rec["verdict"] = "pass" if abs(rec["diff"]) <= tol else "FLAG"
                rec["why"] = (f"vs {got['label']} on {got['n']} shared rows: "
                              f"|{got['ours']:.4f} − {got['theirs']:.4f}| = {abs(rec['diff']):.4f} "
                              f"vs tol {tol:g}")
            out.append(rec)
            continue
        hits = [x for x in labels if str(chk.get("source", "")) in x]
        if len(hits) != 1:
            rec["verdict"] = "absent"
            rec["why"] = (f"`source: {chk.get('source')}` matched {len(hits)} of the sources "
                          f"present ({', '.join(labels) or 'none'})")
            out.append(rec)
            continue
        key = (str(chk.get("family")), hits[0], chk.get("stratum"), str(chk.get("metric")))
        rec["source"], rec["family"], rec["metric"] = hits[0], key[0], key[3]
        if key not in reg:
            rec["verdict"] = "absent"
            rec["why"] = (f"no `{key[3]}` for family `{key[0]}` on `{hits[0]}` in this run "
                          f"(the family or the best-of-k is not present on this set)")
            out.append(rec)
            continue
        ours = reg[key]
        rec["ours"] = ours
        rec["n"] = support.get((key[0], key[1], key[2]))
        exp = float(chk["expect"])
        rec["diff"] = ours - exp
        if chk.get("compare") is False:
            rec["verdict"] = "no verdict"
            rec["why"] = "declared not the same statistic (`compare: false`)"
        else:
            tol = float(chk.get("tol", 0.05))
            if chk.get("tol_rel"):
                tol *= abs(exp)
            rec["tol"] = tol
            rec["verdict"] = "pass" if abs(ours - exp) <= tol else "FLAG"
            rec["why"] = f"|{ours:.4f} − {exp:.4f}| = {abs(ours - exp):.4f} vs tol {tol:g}"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------


def _bo_n(src_n: int) -> int | None:
    """The largest best-of-k a run of n rollouts can report, from score's own BO_KS."""
    ks = [k for k in R.BO_KS_ALL if k <= src_n]
    return max(ks) if ks else None


def _exclusion_line(res: dict) -> str:
    """One line saying which rows left this block's tables, and on whose instrument."""
    e = res["exclusions"]
    if not e.get("present"):
        return f"none — {e.get('why', 'the set declares none')}"
    n = len(e["rows"])
    if not e.get("applied"):
        return (f"**NOT APPLIED** (`--no-exclusions`): the set declares {n} rows "
                f"({e.get('criterion')}), and they ARE in every number below")
    return (f"{n} rows dropped, n = {e.get('n_headline')} of {e.get('n_total')} — "
            f"criterion `{e.get('criterion')}`"
            + (f" at n-gram {e['n_gram']}" if e.get("n_gram") else "")
            + f", from `{e.get('source')}`, computed over `{e.get('computed_over')}`. "
              f"Rows: {', '.join(str(r) for r in e['rows'])}")


def render(res: dict, out: R.Out, sanity: list[dict], figures: list[str]) -> Path:
    src_by_label = {s.label: s for s in res["sources"]}

    # --- cosines, one table per family -----------------------------------------------------
    by_family: dict[str, list[dict]] = {}
    for c in res["cos"]:
        by_family.setdefault(c["family"], []).append(c)
    for family in sorted(by_family):
        head = ["source", "run tag", "n", "rows", "docs", "cosine",
                *[f"bo{k}" for k in R.BO_KS_REPORT], "bo_n"]
        csv_head = ["family", "source", "maemm", "engine", "run_tag", "mu", "n", "n_rows",
                    "n_clusters", "cosine", "bo_source", "k", "mean", "se_cluster", "se_iid"]
        rows, csv_rows = [], []
        for c in sorted(by_family[family], key=lambda c: (c["source"], c["cosine"])):
            n = c["n"]
            bn = _bo_n(n)
            cells = c["bo"]
            rows.append([
                c["source"], c["run_tag"] or "—", n, c["n_rows"], c["n_clusters"], c["cosine"],
                *[R.pm(cells[k]["mean"], cells[k]["se"]) if k in cells else "—"
                  for k in R.BO_KS_REPORT],
                (f"{R.pm(cells[bn]['mean'], cells[bn]['se'])} (k={bn})"
                 if bn in cells else "—"),
            ])
            for k in sorted(cells):
                cell = cells[k]
                csv_rows.append([family, c["source"], c["maemm"], c["engine"], c["run_tag"],
                                 c["mu"], n, cell["n_rows"], cell["n_clusters"], c["cosine"],
                                 c.get("bo_source", ""), k,
                                 round(cell["mean"], 6), round(cell["se"], 6),
                                 round(cell["se_iid"], 6)])
        centred = [c for c in by_family[family] if c["cosine"] == "cos_centred"]
        recomputed = sorted({c["source"] for c in centred
                             if c.get("bo_source", "").startswith("cos_centred.f16")})
        out.table(
            f"cos_{family.replace('/', '_')}", f"Cosine — family `{family}`",
            (f"set `{res['set']}`, base `{res['base']}`, root `{res['root']}`. "
             f"**THE ESTIMATOR, on both cosines: bo-k is the UNBIASED order statistic** — "
             f"sum_i x_(i) C(i-1, k-1) / C(n, k) over all n draws of a row, k > n skipped and "
             f"never clamped (`results.common.bo_unbiased`, the same estimator "
             f"`precompute/common.bo_ladder` stores since 2026-09-23). A `bo_<k>` written before "
             f"that date is the disjoint-group mean and is a different number at every k < n. "
             f"Row values are then averaged over the family's "
             f"rows; ± is a bootstrap SE over {res['boot']} resamples of the DOCUMENT clusters "
             f"(seed {res['seed']}). `cos_centred` is present only for a run that centred on "
             f"something — {len(centred)} of {len(by_family[family])} source-rows here. The raw "
             f"ladder is the product's own `bo_<k>`; the CENTRED ladder is RECOMPUTED here from "
             f"`cos_centred.f16` by that same estimator, because `score` writes `bo_c_<k>` only "
             f"for a row whose every rollout kept a centred token and the paper's products have "
             f"none such — the NaN draws are dropped and the estimator is applied to the finite "
             f"ones at the same k, which reduces exactly to `score`'s when nothing is NaN. "
             f"Recomputed for: "
             f"{', '.join(recomputed) or '(no source on this set)'}. Per-row `bo_source` is in the "
             f"CSV, and the agreement with the stored `bo_c_<k>` wherever it exists is in the "
             f"reader-check table."),
            head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- SAE families ----------------------------------------------------------------------
    by_sae: dict[str, list[dict]] = {}
    for a in res["sae"]:
        by_sae.setdefault(a["family"], []).append(a)
    for family in sorted(by_sae):
        head = ["source", "run tag", "stratum", "features", "n",
                *[f"bo{k} med" for k in R.BO_KS_REPORT], *[f"bo{k} mean" for k in R.BO_KS_REPORT],
                "item fired", "feat firing"]
        csv_head = ["family", "sae_key", "sae_side", "source", "maemm", "engine", "run_tag",
                    "stratum", "n", "n_features", "gate", "k", "ratio_median", "ratio_mean",
                    "ratio_se_iid", "item_fired", "feat_firing"]
        rows, csv_rows = [], []
        for a in sorted(by_sae[family], key=lambda a: (a["source"], a["stratum"] is not None,
                                                       str(a["stratum"]))):
            pk = a["per_k"]
            rows.append([
                a["source"], a["run_tag"] or "—",
                "all" if a["stratum"] is None else a["stratum"], a["n_features"], a["n"],
                *[R.num(pk[k]["median"]) if k in pk else "—" for k in R.BO_KS_REPORT],
                *[R.num(pk[k]["mean"]) if k in pk else "—" for k in R.BO_KS_REPORT],
                R.num(a["item_fired"]), R.num(a["feat_firing"]),
            ])
            for k in sorted(pk):
                csv_rows.append([family, a["sae_key"], a["sae_side"], a["source"], a["maemm"],
                                 a["engine"], a["run_tag"], a["stratum"], a["n"], a["n_features"],
                                 round(a["gate"], 6), k, round(pk[k]["median"], 6),
                                 round(pk[k]["mean"], 6), round(pk[k]["se_iid"], 6),
                                 round(a["item_fired"], 6), round(a["feat_firing"], 6)])
        denoms = sorted({a["corpus_peak_source"] for a in by_sae[family]})
        stats = sorted({str(a.get("stratum_stat")) for a in by_sae[family] if a.get("stratum_stat")})
        n_missing = sum(a["n_no_denominator"] for a in by_sae[family] if a["stratum"] is None)
        out.table(
            f"act_{family.replace('/', '_')}", f"SAE activation — family `{family}`",
            (f"median and mean over features of `peak / corpus peak`. **THE DENOMINATOR IS A "
             f"PARAMETER** (`--corpus-peak`) and this run's is: {'; '.join(denoms)}. It is "
             f"printed rather than assumed because the 16M held-out peak and the 10M training "
             f"peak spec §1.2 asks for are maxima over different texts and a ratio against the "
             f"wrong one prints as an ordinary number. A feature with no peak in that source is "
             f"ABSENT from the median, never divided by the other corpus's number "
             f"({n_missing} pooled). `item fired` is the mean over features of the fraction of "
             f"that source's own draws above the learned gate; `feat firing` is the fraction of "
             f"features that fire at all; the per-k `fired` ladder is the unbiased best-of-k of "
             f"the same 0/1 indicator and is in `results.json`. Strata are numbered 0..3 "
             f"ASCENDING in the draw's own statistic"
             + (f" — {', '.join(stats)}, which is NOT corpus frequency; spec §1.2 and panel b "
                f"both say 'corpus-frequency quartile' and the draws cut on something else"
                if stats else "")
             + ". NO SAE-target cosine appears here (plan §2.3); the per-row cosines are in "
               "`sae_cosines.csv`."),
            head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- the SAE cosines, CSV only ---------------------------------------------------------
    cos_head = ["family", "source", "row", "feature", "stratum", "n", "mean_cos", "max_cos",
                *[f"bo_{k}" for k in R.BO_KS_ALL], "n_sae_gated"]
    cos_rows = []
    for f in res["sae_feats"]:
        src = src_by_label.get(f["source"])
        pt = (src.per_target.get(f["row"]) if src else None) or {}
        cos_rows.append([f["family"], f["source"], f["row"], f["feature"], f["stratum"],
                         f["n"], pt.get("mean_cos"), pt.get("max_cos"),
                         *[pt.get(f"bo_{k}") for k in R.BO_KS_ALL], pt.get("n_sae_gated")])
    R.write_csv(out.dir / "sae_cosines.csv", cos_head, cos_rows)
    out.section(
        "### SAE-target cosines\n\n"
        "*Not tabulated. They answer a different question (does the rollout point the right way) "
        "on a different scale, and a table that put them beside an activation ratio would invite "
        f"the two to be read as one story. {len(cos_rows)} rows in `sae_cosines.csv`.*\n")

    # --- reader check ----------------------------------------------------------------------
    chk_rows = []
    for c in res["checks"]:
        if "skipped" in c:
            chk_rows.append([c["kind"], c["source"], c["family"], "—", "—", "skipped: " + c["skipped"]])
        elif c["kind"] == "cos_centred bo-k":
            # Not a comparison of two stored numbers but a RECOMPUTATION that the product mostly
            # cannot be compared against, so the outcome line says how much of it was compared.
            defect = "" if not c["n_mismatches"] else "  ← POSSIBLE DEFECT"
            how = (f"{c['stored_comparisons']} vs the stored `bo_c_k`, {c['n_mismatches']} "
                   f"mismatches{defect}" if c["stored_comparisons"]
                   else "the product stores NO `bo_c_k` to compare against (score.py:521)")
            chk_rows.append([c["kind"], c["source"], c["family"], c["rows"],
                             R.num(c["worst_excess"], 6),
                             f"{how}; {c['rows_with_nan_rollouts']} rows have a NaN rollout "
                             f"({c['nan_rollouts']} draws)"])
        else:
            defect = "" if not c["n_mismatches"] else "  ← POSSIBLE DEFECT"
            chk_rows.append([c["kind"], c["source"], c["family"], c["rows"],
                             R.num(c["worst_excess"], 6),
                             f"{c['n_mismatches']} mismatches{defect}"])
    out.table("reader_check", "Reader check — our reduction against the product's own",
              ("Both sides are the SAME reduction of the SAME array, so anything outside the "
               "storage's own rounding is a defect in this reader or in that product, never a "
               f"finding. The tolerance is one float16 ulp ({F16_EPS:g} relative) plus the "
               f"4-decimal rounding both sides apply ({2 * ROUND_EPS:g}); `worst excess` is how "
               "far the largest difference sat OUTSIDE its own tolerance, so a negative number "
               "means every comparison was inside it. `cos` recomputes `mean_cos` from "
               "`cos.f16`; `sae_self` recomputes `mean_peak_act` / `max_peak_act` from "
               "`sae_self.f16`."),
              ["kind", "source", "family", "comparisons", "worst excess", "outcome"], chk_rows)

    # --- sanity ----------------------------------------------------------------------------
    sane_rows = []
    for s in sanity:
        sane_rows.append([
            s["name"], s.get("family", "—"), s.get("source", "—"), s.get("metric", "—"),
            s.get("n", "—"), R.num(s.get("ours")), R.num(s.get("expect")),
            R.num(s["tol"]) if isinstance(s.get("tol"), (int, float)) else "—",
            s["verdict"], s.get("why", ""),
        ])
    out.table("sanity", "Sanity gates (plan §2.3)",
              ("From `results/sanity.yaml`, which the user edits. A `FLAG` means the mu or the "
               "prompt may be wrong and the numbers above should not be read until it is "
               "explained; `absent` means the gate's selector does not resolve on this set, which "
               "is a normal outcome for a gate written against another set; `no verdict` means "
               "the YAML declares the two are not the same statistic. `n` is the number of rows "
               "or features behind `ours` — a wide miss at small `n` is a different statement "
               "from a wide miss at the paper's own draw size."),
              ["gate", "family", "source", "metric", "n", "ours", "expected", "tol", "verdict",
               "why"],
              sane_rows)
    prov = [f"- **{s['name']}** — {s['provenance']}" for s in sanity if s.get("provenance")]
    if prov:
        out.section("#### Where the expected numbers come from\n\n" + "\n".join(prov) + "\n")

    for m in res["missing"]:
        out.note(m)
    for n in res["notes"]:
        out.note(n)
    return out.finish(figures)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------


def make_figures(res: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": R.SURFACE, "savefig.facecolor": R.SURFACE,
                         "font.size": 9, "legend.frameon": False})
    colours = res["colours"]
    names: list[str] = []

    # (1) bo-k curves, one figure per cosine family, one panel per cosine, one line per source.
    by_family: dict[str, list[dict]] = {}
    for c in res["cos"]:
        by_family.setdefault(c["family"], []).append(c)
    for family, cells in sorted(by_family.items()):
        fig, axes = plt.subplots(1, len(COSINES), figsize=(9.0, 3.4), sharey=True)
        drew = False
        # Only the k values some source actually reached: a tick at 64 on a run of 4 rollouts is
        # empty axis, and an empty right half reads as a curve that fell off rather than one that
        # was never computed.
        ks_seen = sorted({k for c in cells for k in c["bo"]}) or [1]
        for ax, cosine in zip(np.atleast_1d(axes), COSINES, strict=True):
            R.style_axes(ax, xlabel="k (best-of-k, disjoint groups)",
                         ylabel="mean cosine over rows", title=cosine)
            ax.set_xscale("log", base=2)
            for c in sorted(cells, key=lambda c: c["source"]):
                if c["cosine"] != cosine:
                    continue
                ks = sorted(c["bo"])
                if not ks:
                    continue
                ys = [c["bo"][k]["mean"] for k in ks]
                es = [c["bo"][k]["se"] for k in ks]
                es = [0.0 if not math.isfinite(e) else e for e in es]
                colour = colours.get(c["source"], R.PALETTE[0])
                ax.errorbar(ks, ys, yerr=es, color=colour, linewidth=2.0, marker="o",
                            markersize=5, capsize=2, label=c["source"], zorder=3)
                # Direct label at the last point: identity is never colour alone.
                ax.annotate(c["source"], (ks[-1], ys[-1]), textcoords="offset points",
                            xytext=(6, 0), color=colour, fontsize=7, va="center")
                drew = True
            ax.set_xticks(ks_seen)
            ax.set_xticklabels([str(k) for k in ks_seen])
            ax.set_xlim(ks_seen[0] * 0.8, ks_seen[-1] * 1.9)
        if not drew:
            plt.close(fig)
            continue
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        if len(labels) >= 2:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                       bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"best-of-k, family `{family}` — set {res['set']}", color=R.INK,
                     fontsize=11, x=0.02, ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out_dir, f"bok_{family.replace('/', '_')}"))

    # (2) per-stratum firing and ratio, one figure per SAE family that has strata.
    by_sae: dict[str, list[dict]] = {}
    for a in res["sae"]:
        by_sae.setdefault(a["family"], []).append(a)
    for family, aggs in sorted(by_sae.items()):
        strata = sorted({a["stratum"] for a in aggs if a["stratum"] is not None})
        if not strata:
            continue
        sources = sorted({a["source"] for a in aggs})
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        k0 = min({k for a in aggs for k in a["per_k"]}, default=1)
        width = 0.8 / max(len(sources), 1)
        for ax, (key, ylabel) in zip(axes, (("ratio", f"median peak / corpus peak (bo{k0})"),
                                            ("item_fired", "fraction of draws above gate")),
                                     strict=True):
            R.style_axes(ax, xlabel="stratum (rarity quartile)", ylabel=ylabel)
            for j, s in enumerate(sources):
                xs, ys = [], []
                for i, st in enumerate(strata):
                    hit = [a for a in aggs if a["source"] == s and a["stratum"] == st]
                    if not hit:
                        continue
                    a = hit[0]
                    v = (a["per_k"].get(k0, {}).get("median") if key == "ratio"
                         else a["item_fired"])
                    if v is None or not math.isfinite(float(v)):
                        continue
                    xs.append(i + (j - (len(sources) - 1) / 2) * width)
                    ys.append(float(v))
                # A 2px surface gap between adjacent bars, per the mark spec.
                ax.bar(xs, ys, width=width * 0.88, color=colours.get(s, R.PALETTE[0]),
                       label=s, zorder=3, edgecolor=R.SURFACE, linewidth=1.0)
            ax.set_xticks(range(len(strata)))
            ax.set_xticklabels([str(s) for s in strata])
        handles, labels = axes[0].get_legend_handles_labels()
        if len(labels) >= 2:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                       bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"per stratum, `{family}` — set {res['set']}", color=R.INK, fontsize=11,
                     x=0.02, ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out_dir, f"strata_{family.replace('/', '_')}"))

    # (3) The two-arm row-by-row figure was DELETED on 2026-09-23 (M0a). It compared two run
    # tags of one checkpoint on one set, and the only pair that ever existed was the old
    # primary's `mu-none` against `mu-stats` -- the reconciliation that settled that
    # checkpoint's `mu:` key and is now recorded in config.yaml rather than redrawn every run.
    # The old primary is dropped from the paper; nothing else carries two run tags on one set.

    return names


# ---------------------------------------------------------------------------------------------
# the combined layer: one document over several sets
# ---------------------------------------------------------------------------------------------
#
# `analyse` is per SET, deliberately: a set is a storage contract and a family list, and merging
# two of them into one namespace is how `realact` on `2026-09-21_v3_realact` and `realact` on
# `_ours` become one row that is neither. So the six eval-1 blocks are six independent runs into
# six subdirectories, each with its own tables, CSVs, figures and sanity verdicts, and THIS layer
# writes the document that reads across them: the headline comparison, the merged sanity block,
# and an index. Nothing here recomputes a number -- every cell is lifted from a block's own `res`.


def headline_arms(all_res: list[dict]) -> list[str]:
    """Every source label present on any block, in `discover_sources`' own order.

    NOT A HARDCODED LIST. `discover_sources` already sorts `primary` first and then by checkpoint,
    engine and tag, so the arm order is the config's and a new checkpoint joins the headline by
    having products -- the same promise the rest of this file makes. A label spelled here would be
    the one place a checkpoint name lived in the code.
    """
    out: list[str] = []
    for res in all_res:
        for src in res["sources"]:
            if src.label not in out:
                out.append(src.label)
    return out


def headline_keys(all_res: list[dict]) -> list[tuple[str, str]]:
    """The (block, family) pairs the headline has a column for, in block then family order.

    KEYED ON THE BLOCK AS WELL AS THE FAMILY. `realact` is measured on two of eval 1's blocks --
    `_realact` is Celeste's draw and `_ours` is the draw every v1 table was built on -- and they
    are two different populations. A key that was the family alone would put them in one column
    and show one of the two numbers, silently.
    """
    out: list[tuple[str, str]] = []
    for r in sorted(headline_rows(all_res), key=lambda r: (r["set"], r["family"])):
        if (r["set"], r["family"]) not in out:
            out.append((r["set"], r["family"]))
    return out


def headline_rows(all_res: list[dict]) -> list[dict]:
    """One row per (set, family, source, cosine), with the clustered SE.

    Lifted from each block's `cos` rows -- the `bo` dict is the block's own object, not a copy and
    not a recomputation. A (family, source) the block does not carry is simply absent, which is
    how the old primary's and the NLA arm's missing `realact_long` / `bsf` / `jlens` and every
    non-centrable family's missing centred row say what they are.
    """
    return [{"set": res["set"], "family": c["family"], "arm": c["source"],
             "source": c["source"], "cosine": c["cosine"], "n": c["n"],
             "n_rows": c["n_rows"], "n_clusters": c["n_clusters"],
             "bo": c["bo"], "bo_source": c.get("bo_source", "")}
            for res in all_res for c in res["cos"]]


def render_combined(all_res: list[dict], out: R.Out, sanity: dict[str, list[dict]],
                    figures: list[str], blocks: dict[str, Path]) -> Path:
    """The cross-set document: headline table, merged sanity block, and the index of the blocks."""
    rows, csv_rows = [], []
    head = ["set", "family", "arm", "cosine", "rows", "docs", "n",
            *[f"bo{k}" for k in R.BO_KS_REPORT]]
    csv_head = ["set", "family", "arm", "source", "cosine", "n", "n_rows", "n_clusters",
                "bo_source", "k", "mean", "se_cluster", "se_iid"]
    order = {label: i for i, label in enumerate(headline_arms(all_res))}
    hl = headline_rows(all_res)
    for r in sorted(hl, key=lambda r: (r["set"], r["family"], order[r["arm"]], r["cosine"])):
        cells = r["bo"]
        rows.append([r["set"].replace("2026-09-21_v3_", ""), r["family"], r["arm"], r["cosine"],
                     r["n_rows"], r["n_clusters"], r["n"],
                     *[R.pm(cells[k]["mean"], cells[k]["se"]) if k in cells else "—"
                       for k in R.BO_KS_REPORT]])
        for k in sorted(cells):
            cell = cells[k]
            csv_rows.append([r["set"], r["family"], r["arm"], r["source"], r["cosine"], r["n"],
                             cell["n_rows"], cell["n_clusters"], r["bo_source"], k,
                             round(cell["mean"], 6), round(cell["se"], 6),
                             round(cell["se_iid"], 6)])
    out.table(
        "headline", "Headline — every arm, every cosine family, both cosines",
        ("One row per (block, family, arm, cosine), lifted from the per-block tables and not "
         "recomputed. bo-k is the DISJOINT-GROUP best-of-k mean (floor(n/k) groups of k "
         "consecutive draws, each group's max, averaged; k > n skipped); ± is a bootstrap SE over "
         "resamples of the DOCUMENT clusters, so `rows` and `docs` differ wherever targets share "
         "a document. The RAW ladder is each product's own `bo_<k>`; the CENTRED ladder is "
         "recomputed from `cos_centred.f16` by that same estimator, because `score` stores it "
         "only for a row whose every rollout kept a centred token and none does. A missing row "
         "is a missing PRODUCT: the old primary and the NLA arm were never run on "
         "`realact_long` / `subspace` (no raw directions for them), and a non-centrable family "
         "has no centred row by the NaN rule, never a one-sided number. **`realact_long`'s "
         "centred column is read at `whiten_mu` and not at the `mu_long` its rows were built "
         "under** — a labelled number, not one comparable with the `realact` centred column."),
        head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- the merged sanity block -------------------------------------------------------------
    sane_rows = []
    for set_name, recs in sanity.items():
        for s in recs:
            sane_rows.append([
                set_name.replace("2026-09-21_v3_", ""), s["name"], s.get("family", "—"),
                s.get("source", "—"), s.get("metric", "—"), s.get("n", "—"),
                R.num(s.get("ours")), R.num(s.get("expect")),
                R.num(s["tol"]) if isinstance(s.get("tol"), (int, float)) else "—",
                s["verdict"], s.get("why", ""),
            ])
    counts: dict[str, int] = {}
    for _set_name, recs in sanity.items():
        for s in recs:
            counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
    out.table(
        "sanity_all", "Sanity gates (plan §2.3), every block",
        (f"From `results/sanity.yaml`, resolved independently against each block. "
         f"**{', '.join(f'{v}: {n}' for v, n in sorted(counts.items()))}.** A `FLAG` means the mu "
         f"or the prompt may be wrong and the numbers above should not be read until it is "
         f"explained; `absent` means the gate's selector does not resolve on THAT block, which is "
         f"the normal and expected outcome for most gate-block pairs — a gate written against "
         f"`2026-09-21_v1raw` is absent on all six of these, and a gate on `realact` is absent on "
         f"the four blocks that carry no realact rows. `no verdict` means the YAML declares the "
         f"two are not the same statistic, which on this run covers every gate whose expected "
         f"value uses a different denominator (her 1.0B corpus peak) or a different population "
         f"(SMOKES' exclusion-applied n = 486 against this driver's full 512). `n` is the rows or "
         f"features behind `ours`."),
        ["block", "gate", "family", "source", "metric", "n", "ours", "expected", "tol", "verdict",
         "why"],
        sane_rows)
    prov = {}
    for recs in sanity.values():
        for s in recs:
            if s.get("provenance"):
                prov[s["name"]] = s["provenance"]
    if prov:
        out.section("#### Where the expected numbers come from\n\n"
                    + "\n".join(f"- **{k}** — {v}" for k, v in sorted(prov.items())) + "\n")

    # --- the index of the per-block documents ------------------------------------------------
    idx = ["### The six blocks", "",
           "*Each is an independent `faithfulness.py --set <block>` run with its own tables, "
           "CSVs, figures and sanity verdicts. The numbers above are lifted from them.*", "",
           "| block | sources | cosine rows | SAE rows | document |",
           "|---|---|---|---|---|"]
    for res in all_res:
        rel = blocks[res["set"]]
        idx.append(f"| `{res['set']}` | {len(res['sources'])} | {len(res['cos'])} | "
                   f"{len(res['sae'])} | [`{rel}/tables.md`]({rel}/tables.md) |")
    out.section("\n".join(idx) + "\n")

    for res in all_res:
        for m in res["missing"]:
            out.note(f"`{res['set']}`: {m}")
        for n in res["notes"]:
            out.note(f"`{res['set']}`: {n}")
    return out.finish(figures)


def make_headline_figure(all_res: list[dict], out_dir: Path) -> list[str]:
    """One figure: centred and raw cosine per family, per arm, with the CLUSTERED CI.

    The bars are bo1 (the plain mean over rollouts) with a 95% interval from the document-clustered
    bootstrap, because that is the estimator every SE in this run uses and a figure drawn on a
    different one would not match its own table. Two panels, raw and centred, on a shared axis, so
    the one family where the centred number sits ABOVE the raw one (`realact_long`, read at a mean
    that is not its own) is visible as what it is rather than hidden in a second scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": R.SURFACE, "savefig.facecolor": R.SURFACE,
                         "font.size": 9, "legend.frameon": False})
    hl = headline_rows(all_res)
    if not hl:
        return []
    keys = headline_keys(all_res)
    labels = [f"{f}\n{s.replace('2026-09-21_v3_', '')}" for s, f in keys]
    arms = headline_arms(all_res)
    colours = {a: R.PALETTE[i % len(R.PALETTE)] for i, a in enumerate(arms)}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    width = 0.8 / len(arms)
    for ax, cosine in zip(axes, ("cos_raw", "cos_centred"), strict=True):
        R.style_axes(ax, ylabel="mean cosine over the family's rows (bo1)", title=cosine)
        for j, arm in enumerate(arms):
            xs, ys, es = [], [], []
            for i, k in enumerate(keys):
                hit = [r for r in hl if (r["set"], r["family"]) == k and r["arm"] == arm
                       and r["cosine"] == cosine and 1 in r["bo"]]
                if not hit:
                    continue
                cell = hit[0]["bo"][1]
                xs.append(i + (j - (len(arms) - 1) / 2) * width)
                ys.append(cell["mean"])
                # 95% from the clustered bootstrap SE. NaN where the SE is not estimable (one
                # cluster), drawn as no bar rather than as a zero-width certainty.
                es.append(0.0 if not math.isfinite(cell["se"]) else 1.96 * cell["se"])
            if not xs:
                continue
            ax.bar(xs, ys, width=width * 0.88, color=colours[arm], label=arm, zorder=3,
                   edgecolor=R.SURFACE, linewidth=1.0)
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=R.INK_MUTED, elinewidth=1.0,
                        capsize=2, zorder=4)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=7)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Eval 1 headline — bo1 cosine per family, 95% CI over DOCUMENT clusters",
                 color=R.INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return [R.savefig(fig, out_dir, "headline_cosine")]


# ---------------------------------------------------------------------------------------------
# the paper's cells: `paper/numbers/cells.csv`
# ---------------------------------------------------------------------------------------------
#
# Every number the tex prints is `\N{key}` and comes from one row of `paper/numbers/cells.csv`
# (`paper/numbers/README.md`). This section turns THIS run's aggregates into those rows, and it is
# the only place in eval 1 where a paper key is spelled.
#
# THREE RULES THE WRITER ENFORCES RATHER THAN DOCUMENTS.
#
#   1. A key is rewritten IN PLACE and every other row of the file keeps its exact bytes. The file
#      is shared with six other builders; a rewrite that reflowed quoting or reordered rows would
#      show up as a hundred-line diff with one real change in it, and the real change would be
#      reviewed by nobody.
#   2. A key this module does not OWN is never touched, and a key it owns is never written twice.
#      `--check` treats a duplicate key as an error that blocks the paper build, so a driver that
#      could append a second copy of a row it had already rewritten is a build break waiting for
#      the second run.
#   3. A cell that cannot be built is SKIPPED AND LISTED with the reason, never written empty and
#      never written zero. An empty `value` is legal in the CSV and means "expected, not yet
#      measured", so the existing placeholder row already says the right thing; overwriting it
#      with a zero would replace a true statement with a false one.
#
# The default is READ-ONLY: without `--cells <path>` this builds the rows, prints them and writes
# nothing, so the driver can be run for its tables at any time without touching the paper.

# THE KEYS MODULE M1 OWNS, and the only ones its writer will touch. Plan §2's table: 5 unmeasured
# and 11 measured-to-rewrite in the main list, plus the quartile and 2M appendix rows its own
# section tells it to add. Spelled out rather than discovered from the CSV, because "which rows are
# mine" is a decision about who owns what and not a fact about the file's current contents -- a
# key another builder happened to add would otherwise become M1's on the next run.
M1_KEYS = frozenset({
    # panel a (spec §4): Exemplifier bo1/bo8, the untrained-base control, NLA, the random floor,
    # and the paired appendix block of §1.2 (against the corpus at 10M, and against NLA).
    "fid.ra.ex.cos.bo1", "fid.ra.ex.cos.bo8", "fid.ra.base.cos.bo8", "fid.ra.nla.cos.bo1",
    "fid.rnd.ex.cos.bo1", "fid.ra.diff.cos.bo8", "fid.ra.ex.win.bo8", "fid.ra.nla.dex",
    # panel b, the 131k dictionary. Per quartile at bo1, bo8 and bo64 -- and the three POOLED rows
    # that are already in the file carrying 09-21 old-convention values, rewritten in place with
    # `note` saying they are pooled (spec §1.2: what the panel and the table print is per stratum).
    "sae.l131k.ex.ratio.bo1", "sae.l131k.ex.ratio.bo64", "sae.l131k.ex.fired.bo1",
    *(f"sae.l131k.ex.{m}.bo{k}.q{q}"
      for m in ("ratio", "fired") for k in (1, 8, 64) for q in (1, 2, 3, 4)),
    # the 2M appendix block, enc and dec, bo1 and bo8 (spec §6, run R7).
    *(f"sae.s2m{side}.ex.{m}.bo{k}.q{q}"
      for side in ("enc", "dec") for m in ("ratio", "fired") for k in (1, 8) for q in (1, 2, 3, 4)),
})

CELLS_COLUMNS = ("key", "value", "se", "lo", "hi", "n", "status", "run", "source", "date", "note")
# 4 decimals on cosines, ratios and fired fractions -- the precision the rows already in the file
# carry, and `value` is copied into the tex VERBATIM, so this is the printed precision and not a
# storage choice (`paper/numbers/README.md`, "write the digits exactly as they should print").
CELL_PLACES = 4
# The 131k dictionary's learned BatchTopK threshold, asserted rather than assumed: spec §1.2 and
# panel b both name 1.5846 as THE gate the fired fraction is taken above, and a product carrying a
# different one is a different dictionary or a different checkpoint.
GATE_131K = 1.5846
GATE_PLACES = 4
# 95% interval from the document-clustered bootstrap SE. The NORMAL approximation on that SE, the
# same one `make_headline_figure` draws, and NOT the percentile bootstrap: `R.cluster_bootstrap`
# returns (mean, SE, n, clusters) and not its resample distribution, and writing a percentile
# interval here would mean a second resampler in this file -- two estimators under one name, which
# is the defect M0a's single `bo_unbiased` exists to prevent. Spec §2's panel c says "percentile"
# for `stats_ood`, which is M5's own estimator; this is stated in the caption and in the `note`.
CI_Z = 1.96

# The one place the paper's key vocabulary meets the volume's dictionary ids. `l131k` and `s2m`
# are the writing plan's `<set>` slots (§2); `l42-1b` and `sae2m` are `sae_key` suffixes the rows
# themselves carry. Declarative, so a third dictionary is a line here and no code.
CELL_DICTIONARIES = {
    "l131k": ("l42-1b", ""),
    "s2menc": ("sae2m", "enc"),
    "s2mdec": ("sae2m", "dec"),
}
# What M2's `results/corpus_search.py` ACTUALLY exports, and what this driver calls. `read_top1`
# is its reader of `scan/<dir>/topk.jsonl`, keyed by (set, set_row) and returning a `Top1` whose
# `by_size[size][set_row]` is the rank-0 cosine; `assert_complete` is M2's own gate that every
# selected row carries every nested size. BOTH are required: reading the top-1 without running
# that gate would take a paired difference over whatever rows happened to survive each snapshot.
# A module that is there but is missing one of them is a CONTRACT MISMATCH and raises -- that is a
# different thing from the module not existing, and reporting it as "absent" would send the reader
# looking for a run that has already happened.
CORPUS_SEARCH_EXPORTS = ("read_top1", "assert_complete")
CORPUS_SEARCH_SIGNATURE = (
    "read_top1(vol=..., base=..., scan_dir=..., set_name=..., family=..., apply_exclusions=...) "
    "-> Top1 with .by_size[size][set_row], .corpus, .exclusion_note; assert_complete(Top1)"
)


def corpus_top1_missing(keys: list[str], why: str) -> str:
    """Say which cells are skipped and WHY the corpus comparator did not answer.

    One line, printed and returned, because the two ways this is read are a terminal during the
    run and the `## Skipped and absent` block of the document afterwards. It fills nothing:
    "the corpus scored 0" and "nobody ran the corpus scan" are opposite findings, and a paired
    difference against a zero-filled comparator is the Exemplifier's own number wearing a
    comparison's name.
    """
    line = (
        f"corpus comparator ABSENT -- {len(keys)} cell(s) SKIPPED and not written: "
        f"{', '.join(sorted(keys))}. {why}. The comparator is `results/corpus_search.py` (module "
        f"M2), which reads `scan/<dir>/topk.jsonl` and is the ONE reader of it; this driver does "
        f"not read topk.jsonl itself. Nothing is zero-filled: a corpus that scored 0 and a corpus "
        f"scan nobody ran are opposite findings"
    )
    print(f"   MISSING  {line}", flush=True)
    return line


def corpus_search_module():
    """(M2's module, its provenance prefix) or (None, why it is not there)."""
    if CORPUS_SEARCH is None:
        return None, (
            "`results/corpus_search.py` is not importable (module M2 writes it); "
            f"expected exports: {CORPUS_SEARCH_SIGNATURE}"
        )
    absent = [n for n in CORPUS_SEARCH_EXPORTS if not callable(getattr(CORPUS_SEARCH, n, None))]
    if absent:
        raise AssertionError(
            f"`results/corpus_search.py` is importable but does not export {absent}; it defines "
            f"{sorted(n for n in dir(CORPUS_SEARCH) if not n.startswith('_'))}. That is a "
            f"CONTRACT MISMATCH, not an absent product -- the expected exports are "
            f"{CORPUS_SEARCH_SIGNATURE}"
        )
    return CORPUS_SEARCH, "results.corpus_search.read_top1"


def corpus_scan_dir(base: str, set_name: str, spec: str) -> str:
    """`--corpus-scan` -> the scan DIRECTORY NAME `read_top1` takes.

    `read_top1` addresses the product as `base/<base>/scan/<dir>/topk.jsonl`, so what travels is
    the directory name and not a path. Three spellings are accepted because the run ledger records
    the product in all three, and they resolve to one directory:

      `<set>__<corpus key>`          the directory, as `precompute.common.scan_dir` composed it
      `<corpus key>`                 the corpus key alone, composed here onto this block's set
      `base/<base>/scan/<dir>`       the full volume-relative path, prefix stripped

    The corpus key is not optional and is never defaulted. Her block was scanned against TWO
    corpora under this run tag -- the 10M training corpus and the 16M held-out one -- and their
    top-1s are different numbers about different questions (spec §1.4), so a driver that guessed
    would print one under the other's label and look principled doing it.
    """
    spec = (spec or "").strip().strip("/")
    assert spec, "corpus_scan_dir was handed an empty --corpus-scan; the caller checks first"
    prefix = f"base/{base}/scan/"
    if "/" in spec:
        assert spec.startswith(prefix), (
            f"--corpus-scan {spec!r} looks like a path but does not start with `{prefix}`. Give "
            f"the scan directory `<set>__<corpus key>`, the corpus key alone, or the full "
            f"volume-relative path under that prefix")
        spec = spec[len(prefix):].strip("/")
        assert spec and "/" not in spec, (
            f"--corpus-scan names more than one directory level below `{prefix}`")
    return spec if spec == set_name or spec.startswith(f"{set_name}__") else f"{set_name}__{spec}"


def corpus_top1_for(vol: R.Vol, res: dict, rows: list[int], size: float, scan_spec: str):
    """({row: corpus top-1 at `size`M}, provenance) or (None, why), through M2's reader only.

    The exclusions travel with the call: her block's headline n is the post-`exclusions.json` one
    (486 of 512 at the paper's scale) and `ex_bo8` upstream is already cut to it, so asking M2 for
    the unexcluded corpus side would put a 512-row comparator beside a 486-row mean and leave the
    difference over the intersection with nothing in the output saying which cut produced it.
    `read_top1` REFUSES when the file it is told to apply is not on the volume, which is the gate
    this driver wants: an absent exclusion list is not an empty one.
    """
    mod, prov = corpus_search_module()
    if mod is None:
        return None, prov
    if not (scan_spec or "").strip():
        return None, (
            "`--corpus-scan` is unset, so no scan was named. The comparator is a (set x corpus) "
            f"product and this file will not guess which one: pass the corpus key, e.g. "
            f"`--corpus-scan <corpus key>` -> base/{res['base']}/scan/{res['set']}__<corpus key>")
    scan_dir = corpus_scan_dir(res["base"], res["set"], scan_spec)
    apply_excl = bool(res["exclusions"].get("applied"))
    top1 = mod.read_top1(vol=vol, base=res["base"], scan_dir=scan_dir, set_name=res["set"],
                         family="realact", apply_exclusions=apply_excl)
    mod.assert_complete(top1)
    assert size in top1.by_size, (
        f"`base/{res['base']}/scan/{scan_dir}/topk.jsonl` carries sizes {top1.sizes} and no "
        f"{size:g}M. The paired cells are specified at {size:g}M (spec §1.4); a scan that stopped "
        f"short is a different product, not a smaller one")
    want = {int(r) for r in rows}
    out = {int(r): float(v) for r, v in top1.by_size[size].items()
           if int(r) in want and math.isfinite(float(v))}
    prov = (
        f"{prov} at {size:g}M from `base/{res['base']}/scan/{scan_dir}/topk.jsonl`, corpus "
        f"`{top1.corpus or '(unnamed)'}`, family realact, rank-0 cosine centred both sides; "
        + (top1.exclusion_note or "NO exclusions applied")
        + f" ({len(out)} of {len(want)} targets)"
    )
    return out, prov


# --- building one row --------------------------------------------------------------------------


def _fmt(v, places: int = CELL_PLACES) -> str:
    """One CSV field. An empty string is what an absent number is; NaN never reaches here."""
    if v is None:
        return ""
    f = float(v)
    assert math.isfinite(f), f"a non-finite number reached the cells writer ({v!r})"
    return f"{f:.{places}f}"


def cell(key: str, value, *, se=None, lo=None, hi=None, n=None, status: str = "final",
         run: str = "", source: str = "", date: str = "", note: str = "") -> dict:
    """One `cells.csv` row as a dict of strings, with the file's own invariants asserted here.

    `lo` and `hi` only together: a lone one is an ERROR in `make_numbers.py --check` and blocks
    the paper build, so it is refused where it is built rather than discovered at compile time.
    """
    assert (lo is None) == (hi is None), f"{key}: lo and hi are written together or not at all"
    assert status in ("placeholder", "provisional", "final"), f"{key}: bad status {status!r}"
    assert "\n" not in note, f"{key}: a newline in `note` would break the CSV record"
    return {
        "key": key, "value": _fmt(value), "se": _fmt(se), "lo": _fmt(lo), "hi": _fmt(hi),
        "n": "" if n is None else str(int(n)), "status": status, "run": run,
        "source": source, "date": date, "note": note,
    }


# --- finding the arms and the blocks -------------------------------------------------------------


def hers_realact(all_res: list[dict]) -> tuple[dict | None, str]:
    """(the block whose `realact` rows are HER draw, why not) -- by the ROW's own `source` field.

    NOT by set name. Two of eval 1's blocks carry family `realact` -- `_v3_realact` is Celeste's
    draw and `_v3_ours` is ours -- and the paper's rows are hers (spec §1.1: the test block (ours)
    has one internal consumer and no paper number). `features/heldout_v3._provenance_rows` stamps
    `source: "hers"` on every row it builds from her parquet, so the block says which it is and
    this does not have to know a directory name.
    """
    hits = [r for r in all_res
            if any(str(r["ids"][row].get("source", "")) == "hers"
                   for fam, rws in r["families"].items() if fam.family == "realact" for row in rws)]
    if len(hits) == 1:
        return hits[0], ""
    return None, (
        f"{len(hits)} of the {len(all_res)} block(s) analysed carry `realact` rows stamped "
        f"`source: hers` ({[r['set'] for r in hits]}); panel a's realact rows need exactly one"
    )


def random_floor_block(all_res: list[dict]) -> tuple[dict | None, str]:
    """(the block carrying the `random` family, why not) -- 512 Gaussian directions, spec §1.1."""
    hits = [r for r in all_res if any(f.family == "random" for f in r["families"])]
    if len(hits) == 1:
        return hits[0], ""
    return None, (f"{len(hits)} of the block(s) analysed carry a `random` family "
                  f"({[r['set'] for r in hits]}); the floor needs exactly one")


def surviving_rows(res: dict, family: str) -> list[int]:
    """The rows of `family` that are still in play on this block: the family map, post-exclusion.

    `analyse` drops the excluded rows from the family map and nothing else, which is what makes
    the family map the single row-membership surface. THE PER-ROW LADDERS DO NOT GO THROUGH IT:
    `res["centred"]` is keyed by source and covers every row the product scored, excluded rows
    included, because it is one read of one array before any family is cut out of it. So anything
    that consumes a per-row ladder -- which is every paired cell -- has to intersect with this,
    and a paired difference computed over the raw ladder would silently be the 512-row number
    beside a 486-row mean.
    """
    return [r for fam, rws in res["families"].items() if fam.family == family for r in rws]


def arms_of(res: dict, cfg: dict, exemplifier: str, nla: str = "") -> dict[str, tuple]:
    """{'ex'|'base'|'nla': (the Source, why not)} for one block, by ROLE and TYPE, not by name.

    `base` is `role: control` in `config.yaml` (`2026-09-16_base-control`, carried onto the Source
    by `discover_sources`) and `nla` is `type: nla`, so both follow the config and a second
    control or a second verbalizer joins by being declared.

    `ex` is the one neither can decide. No config field marks WHICH checkpoint the paper calls the
    Exemplifier: `primary: true` is on the OLD primary, which spec §0 item 1 drops from the paper,
    so keying on it would print the dropped checkpoint's numbers under the Exemplifier's rows and
    look principled while doing it. Nor is the name written here -- no checkpoint name is spelled
    in this file, which `selftest.check_combined_layer_lifts_and_never_recomputes` enforces. So:
    `--exemplifier <substring>` names it, and with no flag the MAEMM arms are those that are
    neither `role: control` nor `type: nla`/`base`, which resolves when exactly one MAEMM was
    scored on the block and REFUSES (skip and list) when more than one was. Refusing is the right
    answer there: on the 09-21 volume her block carries the dropped old primary under two run tags
    beside the Exemplifier, and picking among them is a decision, not a lookup.
    """
    out: dict[str, tuple] = {}
    maemms = cfg.get("maemms") or {}

    def _type(s: R.Source) -> str:
        return str((maemms.get(s.maemm) or {}).get("type", ""))

    if exemplifier:
        ex = [s for s in res["sources"] if exemplifier in s.label]
        ex_why = (f"`--exemplifier {exemplifier}` matched {len(ex)} of the arms on "
                  f"`{res['set']}` ({', '.join(s.label for s in res['sources']) or 'none'})")
    else:
        ex = [s for s in res["sources"]
              if s.role != "control" and _type(s) not in ("nla", "base")]
        ex_why = (
            f"no `--exemplifier` was given and {len(ex)} arm(s) on `{res['set']}` are MAEMMs "
            f"(not `role: control`, not `type: nla`/`base`): "
            f"{', '.join(s.label for s in ex) or 'none'}. Which of them the paper calls the "
            f"Exemplifier is a decision no config field records — name it with `--exemplifier`")
    out["ex"] = (ex[0], "") if len(ex) == 1 else (None, ex_why)
    ctrl = [s for s in res["sources"] if s.role == "control"]
    out["base"] = (ctrl[0], "") if len(ctrl) == 1 else (
        None, f"{len(ctrl)} arm(s) on `{res['set']}` are `role: control` in config.yaml; the "
              f"untrained-base control of run R10 has not been run on this block "
              f"(spec §7 R10: no base control on a v3 set exists today)")
    # The NLA arm needs the same discriminator as the Exemplifier for the same reason: the volume
    # accumulates run tags, so one `type: nla` checkpoint can be present as several scored arms
    # (the stored 2026-07 product and spec §7's R6 rerun beside it), and those are DIFFERENT
    # numbers -- the stored directories carry no centred cosine at all and the rerun exists to
    # produce one. `--nla` picks; with no flag an ambiguity is skipped and listed, not guessed.
    nla_arms = [s for s in res["sources"] if _type(s) == "nla" and (not nla or nla in s.label)]
    out["nla"] = (nla_arms[0], "") if len(nla_arms) == 1 else (
        None, f"{len(nla_arms)} arm(s) on `{res['set']}` are `type: nla` in config.yaml"
              + (f" and match `--nla {nla}`" if nla else "")
              + f" ({', '.join(s.label for s in nla_arms) or 'none'})"
              + ("" if nla else " -- name one with `--nla <substring>`"))
    return out


def bo_cell(res: dict, family: str, src: R.Source, k: int, cosines: tuple[str, ...]):
    """(the aggregate for (family, arm, the first cosine column present), which column, why not).

    `cosines` is an ORDERED preference and the answer says which one it landed on, because the
    caller writes that into the `note`. The headline arms declare `("cos_centred",)` alone -- a raw
    number under a key whose grammar says centred (writing plan §2: `cos` IS the centred cosine)
    is the same class of defect as a ratio against the wrong denominator, so it is skipped instead.
    """
    have = [c for c in res["cos"] if c["family"] == family and c["source"] == src.label]
    for cosine in cosines:
        hit = [c for c in have if c["cosine"] == cosine]
        if hit and k in hit[0]["bo"]:
            return hit[0], cosine, ""
    return None, "", (
        f"`{src.label}` / `{family}` on `{res['set']}` has no bo{k} of "
        f"{' or '.join(cosines)} (present: "
        f"{sorted({(c['cosine'], tuple(sorted(c['bo']))) for c in have})})")


def paired(a: dict[int, float], b: dict[int, float], ids: dict[int, dict], boot: int, seed: int):
    """The paired statistics over the targets BOTH sides carry: (difference, win fraction, rows).

    Spec §1.2, "paired, not per-row": "exceeds corpus search at 10M" is a comparison on identical
    targets, so it is computed row by row and only then aggregated, and its interval comes from
    resampling the DOCUMENT clusters of those same rows -- the estimator every other SE in this
    run uses. A row either side is missing is dropped from BOTH, never defaulted, so the
    difference and the win fraction are over one row set and `n` means the same thing in both.
    """
    rows = sorted(set(a) & set(b))
    if not rows:
        return None, None, []
    cl = _clusters_for(rows, ids)
    diffs = [a[r] - b[r] for r in rows]
    wins = [1.0 if a[r] > b[r] else 0.0 for r in rows]
    d_mean, d_se, d_n, d_cl = R.cluster_bootstrap(diffs, cl, boot, seed)
    w_mean, w_se, _, _ = R.cluster_bootstrap(wins, cl, boot, seed)
    diff = {"mean": d_mean, "se": d_se, "n": d_n, "clusters": d_cl,
            "lo": d_mean - CI_Z * d_se, "hi": d_mean + CI_Z * d_se}
    win = {"mean": w_mean, "se": w_se, "n": d_n, "clusters": d_cl}
    return diff, win, rows


# --- panel a ------------------------------------------------------------------------------------


def panel_a_cells(vol: R.Vol, all_res: list[dict], cfg: dict, opts: dict):
    """(the rows for panel a and the appendix's paired block, the reasons the rest were skipped).

    Spec §4 panel a, rows 1, 2, 4, 7 and 8, plus the paired appendix cells of §1.2. Everything
    here reads `cos_centred` and everything here is over HER block minus its exclusions -- both
    applied upstream, in `analyse`, so there is no path by which this layer sees a row the tables
    did not.
    """
    rows: list[dict] = []
    skipped: list[str] = []
    res, why = hers_realact(all_res)
    if res is None:
        skipped.append(f"every panel a realact cell SKIPPED: {why}")
    floor_res, floor_why = random_floor_block(all_res)
    boot, seed, date = opts["boot"], opts["seed"], opts["date"]
    excl_note = ("her v3 realact block minus its own exclusions.json rows; whiten_mu centred; "
                 "doc-clustered bootstrap SE")

    if res is not None:
        arms = arms_of(res, cfg, opts["exemplifier"], opts.get("nla", ""))
        src_by_label = {s.label: s for s in res["sources"]}
        # --- rows 1, 2 and 7: the Exemplifier at bo1 and bo8, the base control at bo8 ---------
        for key, arm, k, run in (("fid.ra.ex.cos.bo1", "ex", 1, "R1"),
                                 ("fid.ra.ex.cos.bo8", "ex", 8, "R1"),
                                 ("fid.ra.base.cos.bo8", "base", 8, "R10"),
                                 ("fid.ra.nla.cos.bo1", "nla", 1, "R6")):
            src, arm_why = arms[arm]
            if src is None:
                skipped.append(f"`{key}` SKIPPED: {arm_why}")
                continue
            got, cosine, cell_why = bo_cell(res, "realact", src, k, ("cos_centred",))
            if got is None:
                skipped.append(f"`{key}` SKIPPED: {cell_why}")
                continue
            c = got["bo"][k]
            rows.append(cell(
                key, c["mean"], se=None if not math.isfinite(c["se"]) else c["se"],
                n=c["n_rows"], run=run, date=date, source=src.scores_rel,
                note=(f"{src.label}; {cosine} from {got.get('bo_source', '')}; {excl_note}, "
                      f"{boot} resamples over {c['n_clusters']} documents, seed {seed}"),
            ))
        # --- the paired cells, against M2's corpus comparator --------------------------------
        ex_src, _ = arms["ex"]
        # INTERSECTED WITH THE FAMILY MAP, which is where the exclusions were applied: see
        # `surviving_rows`. Without this the paired cells would be over the full draw while the
        # means beside them are over the headline n, and nothing in the output would say so.
        keep = set(surviving_rows(res, "realact"))
        ex_rows = res["centred"].get(ex_src.label, {}) if ex_src is not None else {}
        ex_bo8 = {r: v[8] for r, v in ex_rows.items() if 8 in v and r in keep}
        paired_keys = ["fid.ra.diff.cos.bo8", "fid.ra.ex.win.bo8"]
        if not ex_bo8:
            skipped.append(f"`{'`, `'.join(paired_keys)}` and `fid.ra.nla.dex` SKIPPED: the "
                           f"Exemplifier arm has no per-row centred bo8 on `{res['set']}`")
        else:
            corp, corp_prov = corpus_top1_for(vol, res, sorted(ex_bo8), opts["corpus_size"],
                                              opts.get("corpus_scan", ""))
            if corp is None:
                skipped.append(corpus_top1_missing(paired_keys, corp_prov))
            else:
                diff, win, shared = paired(ex_bo8, corp, res["ids"], boot, seed)
                if diff is None:
                    skipped.append(f"`{'`, `'.join(paired_keys)}` SKIPPED: the Exemplifier and "
                                   f"{corp_prov} share no target")
                else:
                    src_rel = src_by_label[ex_src.label].scores_rel
                    # The corpus is named from the PRODUCT (`corp_prov` carries the scan
                    # directory, the corpus label and the exclusion line), never from a literal
                    # here: her block was scanned against two corpora under one run tag.
                    pair_note = (f"paired on {len(shared)} shared targets; Exemplifier bo8 "
                                 f"(cos_centred) minus corpus top-1 at "
                                 f"{opts['corpus_size']:g}M; {corp_prov}; "
                                 f"95% = mean +/- 1.96 x doc-clustered bootstrap SE over "
                                 f"{diff['clusters']} documents, {boot} resamples, seed {seed}")
                    rows.append(cell("fid.ra.diff.cos.bo8", diff["mean"], se=diff["se"],
                                     lo=diff["lo"], hi=diff["hi"], n=diff["n"], run="R1+R2",
                                     date=date, source=src_rel, note=pair_note))
                    rows.append(cell("fid.ra.ex.win.bo8", win["mean"], se=win["se"], n=win["n"],
                                     run="R1+R2", date=date, source=src_rel,
                                     note=(f"fraction of the {len(shared)} shared targets whose "
                                           f"Exemplifier bo8 exceeds the corpus top-1 at "
                                           f"{opts['corpus_size']:g}M; {corp_prov}; SE is the "
                                           f"doc-clustered bootstrap over {win['clusters']} "
                                           f"documents")))
            # --- the Exemplifier against NLA, which is its OWN key (writing plan §2) ---------
            nla_src, nla_why = arms["nla"]
            if nla_src is None:
                skipped.append(f"`fid.ra.nla.dex` SKIPPED: {nla_why}")
            else:
                nla_rows = res["centred"].get(nla_src.label, {})
                nla_bo1 = {r: v[1] for r, v in nla_rows.items() if 1 in v and r in keep}
                if not nla_bo1:
                    skipped.append(
                        f"`fid.ra.nla.dex` SKIPPED: `{nla_src.label}` carries no per-row CENTRED "
                        f"cosine on `{res['set']}` -- the stored NLA directories are `cos_raw` by "
                        f"construction and spec §1.4's R6 rerun produces the centred one")
                else:
                    d, _w, shared = paired(ex_bo8, nla_bo1, res["ids"], boot, seed)
                    if d is None:
                        skipped.append("`fid.ra.nla.dex` SKIPPED: the two arms share no target")
                    else:
                        rows.append(cell(
                            "fid.ra.nla.dex", d["mean"], se=d["se"], lo=d["lo"], hi=d["hi"],
                            n=d["n"], run="R1+R6", date=date, source=nla_src.scores_rel,
                            note=(f"paired on {len(shared)} shared targets: Exemplifier bo8 minus "
                                  f"NLA bo1 (n = 4 rollouts, so bo1 IS the mean of 4), both "
                                  f"cos_centred; 95% = mean +/- 1.96 x doc-clustered bootstrap SE "
                                  f"over {d['clusters']} documents, seed {seed}")))

    # --- row 8: the random floor -----------------------------------------------------------
    if floor_res is None:
        skipped.append(f"`fid.rnd.ex.cos.bo1` SKIPPED: {floor_why}")
    else:
        src, arm_why = arms_of(floor_res, cfg, opts["exemplifier"],
                               opts.get("nla", ""))["ex"]
        if src is None:
            skipped.append(f"`fid.rnd.ex.cos.bo1` SKIPPED: {arm_why}")
        else:
            # THE ONE CELL WITH A FALL-BACK, and it is declared rather than silent. `random` is
            # `centrable: false` in config.yaml:72 -- a Gaussian unit vector is not an activation
            # and nothing centres it -- so `score` writes no `cos_centred` for this family and
            # never will. The key's grammar says `cos` is the centred cosine, the row already in
            # cells.csv carries "cos_raw not centred" in its own note, and panel a's axis IS the
            # centred one: the honest form is to take the raw number, print WHICH cosine it is in
            # the note, and leave the conflict visible for the writer rather than resolve it here.
            got, cosine, cell_why = bo_cell(floor_res, "random", src, 1, ("cos_centred", "cos_raw"))
            if got is None:
                skipped.append(f"`fid.rnd.ex.cos.bo1` SKIPPED: {cell_why}")
            else:
                c = got["bo"][1]
                raw = "" if cosine == "cos_centred" else (
                    " -- NOT CENTRED: family `random` is `centrable: false` (config.yaml:72) so no "
                    "cos_centred exists for it, and panel a's axis is the centred cosine")
                rows.append(cell(
                    "fid.rnd.ex.cos.bo1", c["mean"],
                    se=None if not math.isfinite(c["se"]) else c["se"], n=c["n_rows"], run="R1",
                    date=date, source=src.scores_rel,
                    note=(f"{src.label}; 512 Gaussian unit directions on `{floor_res['set']}`; "
                          f"{cosine}{raw}")))
    return rows, skipped


# --- panel b ------------------------------------------------------------------------------------


def panel_b_cells(all_res: list[dict], cfg: dict, opts: dict):
    """(the rows for panel b and the 2M appendix block, the reasons the rest were skipped).

    Spec §1.2 and §4 panel b. PER STRATUM AND NEVER POOLED is the rule for what the panel and the
    appendix table print; the three pooled keys that already exist in `cells.csv`
    (`sae.l131k.ex.ratio.bo1`, `.ratio.bo64`, `.fired.bo1`) are rewritten in place because they
    carry 09-21 old-convention values, and their `note` says they are pooled so nobody reads one
    as a stratum. `make_numbers --check` reports all three as `defined but unused`, which is what
    a pooled cell should be once the tex prints per-quartile ones.

    The quartile slot is `q<stratum + 1>`: the draws number strata 0..3 ASCENDING in their own
    statistic (`draw_sae2m._stratified_draw`, `searchsorted` over the pool quartile cuts) and the
    writing plan's `q1..q4` are "rarest first", so stratum 0 is q1. The statistic itself is NOT
    corpus frequency on either dictionary -- see `stratum_stat` in the note.
    """
    rows: list[dict] = []
    skipped: list[str] = []
    date = opts["date"]
    seen: set[str] = set()
    # The SAME arm resolution panel a uses, so the two panels cannot end up on different arms: a
    # block's Exemplifier is whichever `arms_of` resolves there, and a block where it does not
    # resolve contributes nothing rather than its other arms' numbers.
    ex_labels = set()
    for res in all_res:
        src, _why = arms_of(res, cfg, opts["exemplifier"], opts.get("nla", ""))["ex"]
        if src is not None:
            ex_labels.add(src.label)
    for key_set, (want_key, want_side) in sorted(CELL_DICTIONARIES.items()):
        aggs = [a for res in all_res for a in res["sae"]
                if str(a["sae_key"]).split("/")[-1] == want_key
                and str(a["sae_side"] or "") == want_side
                and (not opts["exemplifier"] or opts["exemplifier"] in str(a["source"]))
                and str(a["source"]) in ex_labels]
        if not aggs:
            skipped.append(
                f"every `sae.{key_set}.*` cell SKIPPED: no `{want_key}`"
                + (f"/{want_side}" if want_side else "")
                + f" aggregate for an arm matching `--exemplifier {opts['exemplifier']}` in the "
                  f"block(s) analysed ({', '.join(r['set'] for r in all_res)})")
            continue
        sources = {a["source"] for a in aggs}
        assert len(sources) == 1, (
            f"`sae.{key_set}` matched {len(sources)} arms ({sorted(sources)}); narrow "
            f"`--exemplifier` -- two arms' numbers must never land in one key")
        # THE GATE IS READ FROM THE PRODUCT AND ASSERTED, not taken from the spec. Spec §1.2 and
        # panel b both name 1.5846 as the threshold the 131k fired fraction is taken above; a
        # product carrying another one is another dictionary or another checkpoint, and the fired
        # cells would be a different quantity under the same name.
        gates = sorted({round(float(a["gate"]), GATE_PLACES) for a in aggs})
        if key_set == "l131k":
            assert gates == [GATE_131K], (
                f"the 131k block's `sae_self` products carry gate(s) {gates}, and spec §1.2 and "
                f"§4 panel b both name {GATE_131K} to {GATE_PLACES} decimals as THE gate "
                f"`sae.l131k.ex.fired.*` is taken above. Refusing to write a fired fraction above "
                f"a gate the paper does not name")
        denoms = sorted({a["corpus_peak_source"] for a in aggs})
        assert len(denoms) == 1, f"`sae.{key_set}` spans {len(denoms)} ratio denominators: {denoms}"
        denom = denoms[0]
        # EVERY ratio of this dictionary is absent -- say why HERE, once, instead of letting the
        # keys fall into the generic "not built by this run" list at the end. An empty `per_k` on
        # every aggregate means the source `--corpus-peak` names answered for none of these
        # features, and on the real volume that is a product of ANOTHER SET whose row numbers
        # collide with this one's (`CorpusPeaks.applies_to`). The fired cells below are unaffected.
        if all(not a["per_k"] for a in aggs):
            skipped.append(
                f"every `sae.{key_set}.ex.ratio.*` cell SKIPPED and not written: no feature of "
                f"this dictionary has a corpus-peak denominator ("
                f"{sum(a['n_no_denominator'] for a in aggs if a['stratum'] is None)} of "
                f"{sum(a['n_features'] for a in aggs if a['stratum'] is None)} features). The "
                f"denominator source is: {denom}")
        # A RATIO IS ONLY `final` ON THE SPEC'S DENOMINATOR. Spec §1.2 and §1.4 say the ratio's
        # denominator is the feature's peak on the 10M TRAINING corpus (`celeste-train10m`), which
        # M2's scan produces; the default `stored` denominator is `sae_self`'s own 16M held-out
        # `max_act`. The two are different corpora, so a ratio taken against the second is a
        # provisional number wearing the paper's key, and marking it `final` would be the exact
        # thing plan §2 forbids ("a cell is never left carrying a number from a different
        # convention"). The FIRED cells are unaffected -- `peak > gate` has no denominator -- so
        # they stay `final` and the panel's two series can land on different days.
        ratio_status = "final" if denom != STORED_PEAK_PROVENANCE else "provisional"
        ratio_caveat = ("" if ratio_status == "final" else
                        "; PROVISIONAL: this denominator is the 16M held-out max_act, NOT the 10M "
                        "training corpus spec §1.4 names -- rerun with --corpus-peak <M2's 10M "
                        "top1_act path> when that product lands")
        stat = next((a.get("stratum_stat") for a in aggs if a.get("stratum_stat")), None)
        stat_note = f"strata are quartiles of {stat}" if stat else "stratum statistic not recorded"
        product = next((a.get("product", "") for a in aggs), "")
        for a in sorted(aggs, key=lambda a: (a["stratum"] is not None, str(a["stratum"]))):
            pooled = a["stratum"] is None
            slot = "" if pooled else f".q{int(a['stratum']) + 1}"
            where = ("POOLED across strata, not a quartile" if pooled
                     else f"stratum {a['stratum']} of 0..3, {stat_note}")
            miss = ("" if not a["n_no_denominator"]
                    else f"; {a['n_no_denominator']} feature(s) had no peak in that source and are "
                         f"absent from the median, never divided by another corpus's number")
            for k in R.BO_KS_ALL:
                if k in a["per_k"]:
                    key = f"sae.{key_set}.ex.ratio.bo{k}{slot}"
                    if key in opts["owned"] and key not in seen:
                        seen.add(key)
                        rows.append(cell(
                            key, a["per_k"][k]["median"], n=a["per_k"][k]["n"], run="R1",
                            status=ratio_status, date=date, source=product,
                            note=(f"{a['source']}; MEDIAN over features of peak / corpus peak at "
                                  f"bo{k}; denominator: {denom}; {where}; gate "
                                  f"{a['gate']:.{GATE_PLACES}f}{miss}{ratio_caveat}")))
                if k in a.get("fired_k", {}):
                    key = f"sae.{key_set}.ex.fired.bo{k}{slot}"
                    if key in opts["owned"] and key not in seen:
                        seen.add(key)
                        rows.append(cell(
                            key, a["fired_k"][k], n=a["n_features"], run="R1", date=date,
                            source=product,
                            note=(f"{a['source']}; fraction fired above gate "
                                  f"{a['gate']:.{GATE_PLACES}f} at bo{k}, the UNBIASED best-of-k "
                                  f"of the 0/1 fired indicator over the same {a['n']} draws "
                                  f"(1 - C(n-m,k)/C(n,k)); {where}")))
    return rows, skipped


# --- the writer ----------------------------------------------------------------------------------


def _cells_records(path: Path) -> list[tuple[list[str], str]]:
    """[(the parsed record, its EXACT source bytes)] -- so an untouched row is re-emitted verbatim.

    Round-tripping through `csv.writer` would be simpler and would reflow quoting on rows this
    module has no business changing; six builders share this file and a diff has to show one
    change when one thing changed. The reader is fed line by line and the lines each record
    consumed become that record's raw text, which handles the quoted `note` fields that carry
    commas (and would handle an embedded newline, though `cell()` refuses to write one).
    """
    import csv

    # `newline=""` and NOT the default: universal-newline translation would turn this file's CRLF
    # terminators into LF on the way in, the byte-identity assertion below would compare the
    # rewrite against the ALREADY-TRANSLATED text and pass, and the write would reflow all 413
    # lines of a file six builders share. MEASURED 2026-09-23 on the real `cells.csv`, which is
    # CRLF: every line came back changed for one row rewritten.
    with path.open(newline="") as fh:   # `Path.read_text(newline=)` is 3.13+
        text = fh.read()
    lines = text.splitlines(keepends=True)
    seen: list[str] = []

    def feed():
        for ln in lines:
            seen.append(ln)
            yield ln

    out: list[tuple[list[str], str]] = []
    used = 0
    for rec in csv.reader(feed()):
        out.append((rec, "".join(seen[used:])))
        used = len(seen)
    assert "".join(raw for _, raw in out) == text, (
        f"{path}: the record split does not reproduce the file byte for byte, so an in-place "
        f"rewrite cannot be proved not to touch the other builders' rows"
    )
    return out


def _cells_line(row: dict, terminator: str = "\n") -> str:
    """One CSV line, ending with the terminator THE FILE ALREADY USES.

    `cells.csv` is CRLF today. A row written with a bare LF into a CRLF file is a second line
    ending in a file six builders diff, so the terminator is a parameter and `write_cells` reads
    it off the header line rather than assuming either one.
    """
    import csv
    import io

    buf = io.StringIO()
    csv.writer(buf, lineterminator=terminator).writerow([row[c] for c in CELLS_COLUMNS])
    return buf.getvalue()


# `paper/numbers/cells.csv` is NOT inside this repository: the repo is checked out as a
# subdirectory of the paper project, `paper/` is its sibling, and the CSV therefore sits two
# levels above `paper-evals/`. Computed rather than typed so a run from any working directory
# finds it, and asserted before anything is written so a moved checkout fails with the path it
# looked at rather than by creating a new file somewhere harmless-looking.
CELLS_DEFAULT_REL = Path("paper") / "numbers" / "cells.csv"


def resolve_cells_path(spec: str) -> Path | None:
    """`--cells` -> the file to rewrite, or None for the READ-ONLY default.

    "" (the default) writes nothing, which is what keeps this driver safe to run for its tables at
    any moment. The literal `default` resolves to `<paper project>/paper/numbers/cells.csv`.
    Anything else is taken as a path, absolute or relative to the working directory.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec == "default":
        path = R.PAPER_EVALS.parent.parent / CELLS_DEFAULT_REL
        assert path.is_file(), (
            f"--cells default resolved to {path}, which is not a file. The repo is expected to be "
            f"checked out beside the paper project's `paper/` directory; pass the path explicitly "
            f"if this checkout is arranged differently")
        return path
    path = Path(spec)
    assert path.is_file(), f"--cells {spec}: {path.resolve()} is not a file"
    return path


def write_cells(path: Path, rows: list[dict], owned: set[str]) -> dict:
    """Rewrite this module's keys in `cells.csv` IN PLACE; every other byte of the file survives.

    Refuses, loudly and before writing anything:

      * a key that is not in `owned` -- this module writes M1's rows and nobody else's;
      * the same key twice in `rows` -- `--check` calls a duplicate key an error that blocks the
        build, and appending a second copy of a row already rewritten is how one appears;
      * a duplicate key already in the file, which is the same error arriving from elsewhere;
      * a column set that is not `cells.csv`'s, because a reordered header would silently write
        every field into the wrong column.
    """
    keys = [r["key"] for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    assert not dup, f"the cells writer was handed {dup} more than once; one key, one row"
    stray = sorted(set(keys) - set(owned))
    assert not stray, (
        f"the cells writer was handed {stray}, which module M1 does not own. `cells.csv` is "
        f"shared with six other builders and each one writes only its own keys")
    records = _cells_records(path)
    assert records, f"{path} is empty"
    header, header_raw = records[0]
    assert tuple(header) == CELLS_COLUMNS, (
        f"{path} has columns {header}, not {list(CELLS_COLUMNS)}")
    body = records[1:]
    present = [rec[0] for rec, _ in body]
    dup_file = sorted({k for k in present if present.count(k) > 1})
    assert not dup_file, f"{path} already carries {dup_file} more than once"

    pending = {r["key"]: r for r in rows}
    # The terminator the file already uses, taken from its own header line.
    term = "\r\n" if header_raw.endswith("\r\n") else "\n"
    out = [header_raw if header_raw.endswith("\n") else header_raw + term]
    rewritten: list[str] = []
    for rec, raw in body:
        key = rec[0]
        if key in pending:
            out.append(_cells_line(pending.pop(key), term))
            rewritten.append(key)
        else:
            out.append(raw if raw.endswith("\n") else raw + term)
    appended = [r["key"] for r in rows if r["key"] in pending]
    for r in rows:
        if r["key"] in pending:
            out.append(_cells_line(pending.pop(r["key"]), term))
    assert not pending, pending
    with path.open("w", newline="") as fh:
        fh.write("".join(out))
    return {"path": str(path), "rewritten": rewritten, "appended": appended,
            "untouched": len(body) - len(rewritten)}


def paper_cells(vol: R.Vol, all_res: list[dict], cfg: dict, opts: dict):
    """(every cells.csv row this run can build, the reasons the rest were skipped)."""
    a_rows, a_skipped = panel_a_cells(vol, all_res, cfg, opts)
    b_rows, b_skipped = panel_b_cells(all_res, cfg, opts)
    rows = a_rows + b_rows
    skipped = a_skipped + b_skipped
    # ONE RUN, OR SAY SO. `--exemplifier` is a substring and it is resolved per block, so on a
    # volume that has accumulated run tags it can legitimately land on `<ckpt>@vllm` in one block
    # and `<ckpt>@vllm:<tag>` in another -- two scoring runs, one set of paper rows, and nothing
    # in the CSV saying which cell came from which. Not refused (a rerun of one block is normal),
    # but never silent.
    labels = sorted({lbl for res in all_res
                     for lbl in [arms_of(res, cfg, opts["exemplifier"],
                                         opts.get("nla", ""))["ex"][0]]
                     if lbl is not None for lbl in [lbl.label]})
    if len(labels) > 1:
        skipped.append(
            f"the Exemplifier resolved to MORE THAN ONE arm across the blocks ({', '.join(labels)}"
            f"): panel a and panel b cells may come from different scoring runs. Narrow "
            f"`--exemplifier` to one arm if that was not intended")
    built = {r["key"] for r in rows}
    missing = sorted(set(opts["owned"]) - built)
    if missing:
        skipped.append(
            f"{len(missing)} owned key(s) not built by this run and LEFT AS THEY ARE in cells.csv "
            f"(an empty `value` there already means 'expected, not yet measured'): "
            f"{', '.join(missing)}")
    return rows, skipped


def render_cells(rows: list[dict], skipped: list[str], out: R.Out, opts: dict) -> None:
    """The cells block of the combined document: what was written, and what was not, and why."""
    out.table(
        "cells", "Paper cells — `paper/numbers/cells.csv` rows this run builds",
        (f"Module M1's own keys and no others. `value` carries the digits exactly as the tex "
         f"prints them ({CELL_PLACES} decimals), because `\\N{{key}}` copies the field verbatim "
         f"(`paper/numbers/README.md`). `lo`/`hi` is mean ± {CI_Z} × the DOCUMENT-CLUSTERED "
         f"bootstrap SE, the same estimator every ± in this document uses, and not a percentile "
         f"interval — `results.common.cluster_bootstrap` returns an SE and not its resample "
         f"distribution, and a second resampler here would be a second estimator under one name. "
         f"Ratio cells carry the denominator's corpus in their `note`: `--corpus-peak "
         f"{opts['corpus_peak']}`; the two PAIRED cells carry the scan their comparator came "
         f"from: `--corpus-scan {opts.get('corpus_scan') or '(unset — they are skipped)'}`. "
         f"A cell that could not be built is in the list below with its "
         f"reason and is NOT written — the placeholder row already in the CSV says 'expected, not "
         f"yet measured', which is true, and a zero would not be."),
        ["key", "value", "se", "lo", "hi", "n", "status", "run", "source", "date"],
        [[r[c] for c in CELLS_COLUMNS[:-1]] for r in rows],
        csv_header=list(CELLS_COLUMNS), csv_rows=[[r[c] for c in CELLS_COLUMNS] for r in rows])
    if skipped:
        out.section("#### Cells NOT written, and why\n\n"
                    + "\n".join(f"- {s}" for s in skipped) + "\n")
    for s in skipped:
        out.note(s)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    set_: Annotated[str, typer.Option(
        "--set",
        help="the held-out set, as declared in config.yaml; a COMMA-SEPARATED list runs each one "
             "into <out>/<block>/ and writes the cross-set document at <out>/")] = "",
    sources: Annotated[str, typer.Option(help="comma-separated substrings of source labels to keep")] = "",
    out: Annotated[Path | None, typer.Option(
        help="output directory; default $MAEMM_OUT or <repo>/_out/faithfulness")] = None,
    root: Annotated[str, typer.Option(help="volume-relative root the products were written under")] = "",
    data: Annotated[Path | None, typer.Option(
        help="local mirror of the volume; default $MAEMM_MIRROR or "
             "$XDG_CACHE_HOME/maemm-paper-evals/mirror/<root>, NEVER under paper-evals/")] = None,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what the mirror already has")] = False,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    sanity_file: Annotated[Path, typer.Option("--sanity", help="the gates YAML the user edits")]
    = R.HERE / "sanity.yaml",
    boot: Annotated[int, typer.Option(help="bootstrap resamples for the clustered SE")] = R.N_BOOT,
    seed: Annotated[int, typer.Option(help="bootstrap seed")] = R.BOOT_SEED,
    check_arrays: Annotated[bool, typer.Option(help="recompute the aggregates from the arrays")] = True,
    check_arrays_max_mb: Annotated[float, typer.Option(help="skip the cos.f16 check above this size")] = 8.0,
    centred_bok_max_mb: Annotated[float, typer.Option(
        help="refuse to read cos_centred.f16 above this size (the centred bo-k columns need it)")] = 128.0,
    exclusions: Annotated[bool, typer.Option(
        help="drop the rows in the set's own exclusions.json (the paper's n)")] = True,
    corpus_peak: Annotated[str, typer.Option(
        "--corpus-peak",
        help="the panel b ratio DENOMINATOR: `stored` (sae_self's own 16M held-out max_act, the "
             "default and what this file did before) or `top1_act:<volume-relative dir>` (M2's "
             "scan of celeste-train10m, which spec §1.2/§1.4 asks for). An asked-for source that "
             "is not on the volume RAISES and never falls back")] = CORPUS_PEAK_STORED,
    cells: Annotated[str, typer.Option(
        "--cells",
        help="rewrite THIS module's keys in a cells.csv, in place. Empty (the default) writes "
             "NOTHING, so the driver stays read-only; `default` resolves to the paper project's "
             "own paper/numbers/cells.csv; anything else is taken as a path")] = "",
    cells_date: Annotated[str, typer.Option(
        help="the `date` column of every cell written -- the RUN date, not today's")] = "",
    exemplifier: Annotated[str, typer.Option(
        help="substring naming the Exemplifier arm for the paper's cells. No config field marks "
             "it (`primary: true` is the DROPPED old primary) and no checkpoint name is spelled "
             "in this file, so with no flag the MAEMM arms are those that are neither `role: "
             "control` nor `type: nla`/`base` and the cells are SKIPPED when more than one is "
             "present rather than guessed at")] = "",
    nla: Annotated[str, typer.Option(
        help="substring naming the NLA arm, where a block carries more than one `type: nla` "
             "product (the stored 2026-07 directories and spec §7's R6 rerun)")] = "",
    corpus_size: Annotated[float, typer.Option(
        help="corpus size in millions the paired cells compare against (spec §1.4: 10M)")] = 10.0,
    corpus_scan: Annotated[str, typer.Option(
        "--corpus-scan",
        help="the corpus-search product the PAIRED cells (`fid.ra.diff.cos.bo8`, "
             "`fid.ra.ex.win.bo8`) are taken against: the corpus key (`<corpus key>`), the scan "
             "directory (`<set>__<corpus key>`) or its full volume-relative path. Read through "
             "M2's `results.corpus_search.read_top1`. Unset (the default) SKIPS those two cells "
             "and says so -- her block is scanned against two corpora under one run tag and "
             "their top-1s are different numbers")] = "",
    figures: Annotated[bool, typer.Option(help="write figures/ (PDF + PNG)")] = True,
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
) -> None:
    assert set_, "--set <name> is required; config.yaml `heldout:` lists the declared sets"
    # Resolved FIRST, before an hour of fetching: a typo in `--cells` must not be discovered after
    # the run, and the write itself still happens last so a refused write costs the tables nothing.
    out = Path(out) if out else R.out_dir("faithfulness")
    cells_path = resolve_cells_path(cells)
    names = [x.strip() for x in set_.split(",") if x.strip()]
    assert len(names) == len(set(names)), f"--set names a block twice: {names}"
    cfg = R.load_config()
    mirror = Path(data) if data else R.mirror_dir(root)
    vol = R.Vol(root, mirror, modal_cmd, refetch, quiet, offline=not fetch)

    all_res: list[dict] = []
    all_sanity: dict[str, list[dict]] = {}
    blocks: dict[str, Path] = {}
    # ONE BLOCK PER SUBDIRECTORY, even for a single `--set`: the layout is then the same whether
    # one block or six were asked for, so a command line that grows a name does not move files
    # that were already written and cited.
    for name in names:
        res = analyse(vol, cfg, name, sources, boot, seed, check_arrays, check_arrays_max_mb,
                      centred_bok_max_mb, exclusions, corpus_peak)
        sub = Path(out) / name.replace("2026-09-21_v3_", "").replace("/", "_")
        figs = make_figures(res, sub) if figures else []
        sanity = run_sanity(res, sanity_file, vol, cfg)
        preamble = [
            f"- set `{res['set']}`, base `{res['base']}`, volume root `{res['root']}`, "
            f"mirror `{vol.local}`",
            f"- command: `{' '.join(sys.argv)}`",
            f"- sources present: {', '.join(s.label for s in res['sources']) or '(none)'}",
            f"- SEs: bootstrap over document clusters, {boot} resamples, seed {seed}",
            f"- exclusions: {_exclusion_line(res)}",
        ]
        o = R.Out(sub, f"Eval 1 — faithfulness on `{res['set']}`", preamble)
        path = render(res, o, sanity, figs)
        with open(sub / "results.json", "w") as fh:
            json.dump({"set": res["set"], "base": res["base"], "root": res["root"],
                       "cos": res["cos"], "sae": res["sae"], "checks": res["checks"],
                       "sanity": sanity, "missing": res["missing"], "notes": res["notes"]},
                      fh, indent=1, default=str)
        all_res.append(res)
        all_sanity[name] = sanity
        blocks[name] = sub.relative_to(Path(out))

        print(f"\n[faithfulness] {path}")
        print(f"[faithfulness] {len(res['sources'])} sources, {len(res['cos'])} cosine rows, "
              f"{len(res['sae'])} SAE rows, {len(figs)} figures")
        for m in res["missing"]:
            print(f"   MISSING  {m}")
        for n in res["notes"]:
            print(f"   NOTE     {n}")
        for s in sanity:
            if s["verdict"] != "absent":
                print(f"   {s['verdict']:<10} {s['name']}: {s.get('why', '')}")
        for c in res["checks"]:
            if "skipped" in c:
                print(f"   check      {c['kind']}/{c['source']}: skipped ({c['skipped']})")
            elif c.get("n_mismatches"):
                print(f"   check      {c['kind']}/{c['source']}: {c['n_mismatches']} MISMATCHES "
                      f"(worst excess {c['worst_excess']})")

    hfigs = make_headline_figure(all_res, Path(out)) if figures else []
    pre = [
        f"- blocks: {', '.join('`' + r['set'] + '`' for r in all_res)}",
        f"- base `{all_res[0]['base']}`, volume root `{all_res[0]['root']}`, "
        f"mirror `{vol.local}`",
        f"- command: `{' '.join(sys.argv)}`",
        f"- SEs and CIs: bootstrap over document clusters, {boot} resamples, seed {seed}",
        "- every number here is LIFTED from a block's own table; nothing is recomputed at this "
        "level",
        f"- panel b ratio denominator: `--corpus-peak {corpus_peak}` — "
        f"{all_res[0]['corpus_peak']['provenance']}",
        f"- paired-cell corpus comparator: `--corpus-scan "
        f"{corpus_scan or '(unset — the two paired cells are skipped and listed)'}`",
        *[f"- exclusions, `{r['set']}`: {_exclusion_line(r)}" for r in all_res],
    ]
    o = R.Out(out, "Eval 1 — faithfulness, all blocks", pre)
    # The paper's cells are built from the SAME objects the tables were, at this level and not per
    # block, because panel a's rows span two blocks (her realact draw and `_ctrl`'s random floor)
    # and a per-block writer could not see both.
    cell_opts = {"boot": boot, "seed": seed, "date": cells_date or datetime.date.today().isoformat(),
                 "exemplifier": exemplifier, "nla": nla, "corpus_size": corpus_size,
                 "owned": M1_KEYS,
                 "corpus_peak": corpus_peak, "corpus_scan": corpus_scan}
    cell_rows, cell_skipped = paper_cells(vol, all_res, cfg, cell_opts)
    render_cells(cell_rows, cell_skipped, o, cell_opts)
    path = render_combined(all_res, o, all_sanity, hfigs, blocks)
    # THE ONLY WRITE OUTSIDE `--out`, and it happens last: the tables are on disk before the
    # paper's own CSV is touched, so a writer that refuses (a duplicate key, a column set that is
    # not cells.csv's) costs the run nothing and leaves the numbers readable.
    if cells_path is not None:
        rec = write_cells(cells_path, cell_rows, M1_KEYS)
        print(f"[faithfulness] cells {rec['path']}: {len(rec['rewritten'])} rewritten in place, "
              f"{len(rec['appended'])} appended, {rec['untouched']} rows untouched")
    else:
        print(f"[faithfulness] cells: {len(cell_rows)} row(s) built, NOT written "
              f"(pass --cells <path> to rewrite them in paper/numbers/cells.csv)")
    for s in cell_skipped:
        print(f"   CELL SKIPPED  {s}")

    flags = [(k, s) for k, recs in all_sanity.items() for s in recs if s["verdict"] == "FLAG"]
    print(f"\n[faithfulness] COMBINED {path}")
    print(f"[faithfulness] {len(all_res)} blocks, {len(headline_rows(all_res))} headline rows, "
          f"{len(flags)} FLAG(s)")
    for k, s in flags:
        print(f"   FLAG       {k} / {s['name']}: {s.get('why', '')}")


if __name__ == "__main__":
    app()
