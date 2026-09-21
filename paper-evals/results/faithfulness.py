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

  SAE families (`sae` rows, split by the dictionary in the row's own `sae_key` and by `sae_side`)
      median and mean of `peak / corpus_peak` -- our 16M `max_act`, never a 1B-scan peak -- at
      bo1/bo8/bo64 where the rollout count allows, the fraction of draws above the learned gate
      and the fraction of features that fire at all, whole-family and per stratum. NO SAE-TARGET
      COSINE reaches a markdown table (plan §2.3): the per-row cosines go to `sae_cosines.csv`,
      where nothing invites them to be read beside an activation ratio.

  sanity  the gates of plan §2.3, from `results/sanity.yaml`, which the USER edits: her card's
      numbers, the `sae_smoke64.md` medians and the old primary's recorded values, each with its
      own stated tolerance and a pass / FLAG / absent verdict per line.

  figures per-family bo-k curves per source; per-stratum firing and ratio bars per SAE family;
      and, for any checkpoint with two run tags on this set, the two arms compared row by row --
      which on the current products is the old primary's `mu-none` against `mu-stats`. PDF + PNG.

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

    THE SAME ESTIMATOR AS THE STORED RAW LADDER, and the grouping is what makes it the same one:
    `common.best_of_k_means` splits the n per-rollout bests into floor(n/k) DISJOINT groups of k
    CONSECUTIVE rollouts, takes each group's max and averages them; k > n is skipped, never
    clamped. The one thing this has to decide that `score` never did is what a NaN rollout does to
    a group, and the answer is the one that reduces exactly to `score`'s when there are none: the
    group's max is taken over its FINITE entries, and a group with no finite entry is dropped from
    the average rather than counted as a zero. Positions are kept -- rollout j stays in group
    j // k -- because dropping the NaNs first would regroup the survivors and quietly change which
    draws compete with which.

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
        cells: dict[int, float] = {}
        for k in R.BO_KS_ALL:
            if k > n:
                continue
            maxima = []
            for g in range(n // k):
                grp = best[g * k : (g + 1) * k]
                grp = grp[np.isfinite(grp)]
                if grp.size:
                    maxima.append(float(grp.max()))
            if maxima:
                cells[k] = float(np.mean(maxima))
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


def sae_cells(fam: R.Family, rows: list[int], src: R.Source, ids: dict[int, dict],
              vol: R.Vol) -> tuple[list[dict], list[dict], dict]:
    """(aggregate rows incl. per stratum, per-feature rows, the reader check) for one SAE family.

    Per feature, `peaks_of` gives one peak activation per rollout; `best_of_k_means` turns those
    into the same disjoint-group best-of-k the cosine side reports, and every ratio divides by
    that feature's own `corpus_peak` as `sae_self` recorded it (our 16M `max_act`).

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
        cp = float(stored.get(row, {}).get("corpus_peak", 0.0))
        bo = R.best_of_k_means(peaks, R.BO_KS_ALL)
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
            "corpus_peak": cp, "n": int(meta["n"]),
            "ratio": {k: (v / cp if cp > 0 else float("nan")) for k, v in bo.items()},
            "bo": bo,
            "item_fired": float(np.mean(peaks > gate)),
            "fired_any": bool(peaks.max() > gate),
        })
    check = {"rows": len(feats), "n_mismatches": mism, "worst_excess": round(worst, 6),
             "gate": gate, "product": rel}
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
            "sae_key": fam.sae_key, "sae_side": fam.sae_side,
            "n_features": len(sub), "gate": gate, "per_k": per_k,
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
            centred_bok_max_mb: float = 128.0, apply_exclusions: bool = True) -> dict:
    """Everything the tables and figures are built from. Never raises on a missing source."""
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
    # A set none of whose families is centrable (`_ctrl`, `_sae2m`, `_subspace`) still has a
    # `cos_centred.f16` on some arms -- all NaN, by the rule that a non-centrable row is ABSENT
    # from the centred aggregates rather than scored one-sidedly. Reading it would fetch ~12 MB
    # per arm to produce nothing, so the read is skipped and the reason recorded.
    kinds = cfg.get("family_kinds") or {}
    centrable = sorted({f.family for f in fams if (kinds.get(f.family) or {}).get("centrable")})
    for src in usable:
        if not centrable:
            checks.append({"kind": "cos_centred bo-k", "source": src.label, "family": "(all)",
                           "skipped": f"no family of `{set_name}` is centrable "
                                      f"({sorted({f.family for f in fams})}), so a centred cosine "
                                      f"would be NaN on every row -- the array is not read"})
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
                aggs, feats, chk = sae_cells(fam, rows, src, ids, vol)
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
             f"**THE ESTIMATOR, on both cosines: bo-k is the DISJOINT-GROUP best-of-k mean** — the "
             f"n rollouts of a row are split into floor(n/k) groups of k CONSECUTIVE draws, each "
             f"group's max is taken, and those are averaged; k > n is skipped, never clamped "
             f"(`precompute/common.best_of_k_means`). It is not the unbiased order-statistic "
             f"estimator, and the two do not agree. Row values are then averaged over the family's "
             f"rows; ± is a bootstrap SE over {res['boot']} resamples of the DOCUMENT clusters "
             f"(seed {res['seed']}). `cos_centred` is present only for a run that centred on "
             f"something — {len(centred)} of {len(by_family[family])} source-rows here. The raw "
             f"ladder is the product's own `bo_<k>`; the CENTRED ladder is RECOMPUTED here from "
             f"`cos_centred.f16` by that same estimator, because `score` writes `bo_c_<k>` only "
             f"for a row whose every rollout kept a centred token and the paper's products have "
             f"none such — a group's max is over its finite draws and an all-NaN group is dropped, "
             f"which reduces exactly to `score`'s when nothing is NaN. Recomputed for: "
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
        out.table(
            f"act_{family.replace('/', '_')}", f"SAE activation — family `{family}`",
            ("median and mean over features of `peak / corpus_peak`, the denominator being OUR 16M "
             "`max_act` as `sae_self` recorded it per feature — never a 1B-scan peak. `item fired` "
             "is the mean over features of the fraction of that source's own draws above the "
             "learned gate; `feat firing` is the fraction of features that fire at all. NO "
             "SAE-target cosine appears here (plan §2.3); the per-row cosines are in "
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

    # (3) two arms of one checkpoint, row by row. On the current products this is the old
    # primary's `mu-none` against `mu-stats`; the figure is emitted for ANY checkpoint that has
    # two run tags on this set, because that is what a run tag is for.
    by_ckpt: dict[str, list[R.Source]] = {}
    for s in res["sources"]:
        by_ckpt.setdefault(f"{s.maemm}@{s.engine}", []).append(s)
    for ckpt, arms in sorted(by_ckpt.items()):
        if len(arms) < 2:
            continue
        a, b = arms[0], arms[1]
        shared = sorted(set(a.per_target) & set(b.per_target))
        if not shared:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        fams = {res["ids"][r]["family"] for r in shared}
        R.style_axes(axes[0], xlabel=f"{a.label}   mean cos", ylabel=f"{b.label}   mean cos",
                     title="per row")
        lo, hi = 1.0, 0.0
        for i, fam in enumerate(sorted(fams)):
            rows = [r for r in shared if res["ids"][r]["family"] == fam]
            xs = [float(a.per_target[r]["mean_cos"]) for r in rows]
            ys = [float(b.per_target[r]["mean_cos"]) for r in rows]
            lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
            axes[0].scatter(xs, ys, s=42, color=R.PALETTE[i % len(R.PALETTE)], label=fam,
                            zorder=3, edgecolor=R.SURFACE, linewidth=1.0)
        pad = 0.05 * max(hi - lo, 1e-3)
        axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=R.INK_MUTED,
                     linewidth=1.0, linestyle="--", zorder=2)
        axes[0].legend(fontsize=7, loc="upper left")
        R.style_axes(axes[1], ylabel="mean cos over the family's rows", title="family means")
        labels = sorted(fams)
        for j, src in enumerate((a, b)):
            xs = [i + (j - 0.5) * 0.4 for i in range(len(labels))]
            ys = [float(np.mean([src.per_target[r]["mean_cos"] for r in shared
                                 if res["ids"][r]["family"] == f])) for f in labels]
            axes[1].bar(xs, ys, width=0.36, color=colours.get(src.label, R.PALETTE[j]),
                        label=src.label, zorder=3, edgecolor=R.SURFACE, linewidth=1.0)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].legend(fontsize=7)
        fig.suptitle(f"two arms of `{ckpt}` on {len(shared)} shared rows — set {res['set']}",
                     color=R.INK, fontsize=11, x=0.02, ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out_dir, f"arms_{ckpt.replace('/', '_').replace('@', '_')}"))

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
# CLI
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    set_: Annotated[str, typer.Option(
        "--set",
        help="the held-out set, as declared in config.yaml; a COMMA-SEPARATED list runs each one "
             "into <out>/<block>/ and writes the cross-set document at <out>/")] = "",
    sources: Annotated[str, typer.Option(help="comma-separated substrings of source labels to keep")] = "",
    out: Annotated[Path, typer.Option(help="output directory")] = R.HERE / "out" / "faithfulness",
    root: Annotated[str, typer.Option(help="volume-relative root the products were written under")] = "",
    data: Annotated[Path | None, typer.Option(help="local mirror (default results/data/<root>)")] = None,
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
    figures: Annotated[bool, typer.Option(help="write figures/ (PDF + PNG)")] = True,
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
) -> None:
    assert set_, "--set <name> is required; config.yaml `heldout:` lists the declared sets"
    names = [x.strip() for x in set_.split(",") if x.strip()]
    assert len(names) == len(set(names)), f"--set names a block twice: {names}"
    cfg = R.load_config()
    mirror = data or (R.HERE / "data" / (root.replace("/", "_") or "vol"))
    vol = R.Vol(root, mirror, modal_cmd, refetch, quiet, offline=not fetch)

    all_res: list[dict] = []
    all_sanity: dict[str, list[dict]] = {}
    blocks: dict[str, Path] = {}
    # ONE BLOCK PER SUBDIRECTORY, even for a single `--set`: the layout is then the same whether
    # one block or six were asked for, so a command line that grows a name does not move files
    # that were already written and cited.
    for name in names:
        res = analyse(vol, cfg, name, sources, boot, seed, check_arrays, check_arrays_max_mb,
                      centred_bok_max_mb, exclusions)
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
        *[f"- exclusions, `{r['set']}`: {_exclusion_line(r)}" for r in all_res],
    ]
    o = R.Out(out, "Eval 1 — faithfulness, all blocks", pre)
    path = render_combined(all_res, o, all_sanity, hfigs, blocks)
    flags = [(k, s) for k, recs in all_sanity.items() for s in recs if s["verdict"] == "FLAG"]
    print(f"\n[faithfulness] COMBINED {path}")
    print(f"[faithfulness] {len(all_res)} blocks, {len(headline_rows(all_res))} headline rows, "
          f"{len(flags)} FLAG(s)")
    for k, s in flags:
        print(f"   FLAG       {k} / {s['name']}: {s.get('why', '')}")


if __name__ == "__main__":
    app()
