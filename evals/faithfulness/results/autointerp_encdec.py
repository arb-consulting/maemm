#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9", "rich>=13"]
# ///
"""Eval 2, encoder vs decoder: the same 512 features of one SAE explained from the ENCODER column
and from the DECODER row, paired feature by feature.

    cd <repo>
    (export MODAL_PROFILE=<your-profile>; \\
     uv run evals/faithfulness/results/autointerp_encdec.py \\
       --enc rl-final=2026-09-24_autointerp-512 --dec rl-final-dec=2026-09-24_autointerp-512-dec \\
       --out _out/autointerp-dec)

Local, CPU, no GPU, no model, no API. Like `results/autointerp.py` it recomputes nothing but the
aggregation: every per-feature number is READ from the two runs' `scores.jsonl`, the build rows,
the two `score` products and the two corpus scans. The readers, the bootstrap and the per-band join
are the driver's own (`autointerp.load_run`, `load_refusals`, `index_rows`, `values_of`,
`paired_diff`, `boot_ci`, `band_rows`, `reader_check`), and the fidelity side is
`corpus_search.exemplifier_bok` / `read_top1`, the functions the paper's corpus-search cells use --
so a number here and the same number in the paper's tables are one convention, not two.

THE DESIGN THIS READS. Two run directories on the same features, the same C16 pool, the same judge
test items and the same null arms, launched with ONE shared `--cache-dir`. `run.py`'s cache key has
no run or build component, so every call whose request body is identical in the two runs is the
SAME call replayed: C16, its two nulls and `R-shuffled` should come back byte-identical, and only
the MAEM arms -- whose shown examples are rollouts made from a different injected direction --
are new measurements. That is an assumption this file CHECKS (`replay_check`) rather than trusts:
a replayed arm whose rows differ on a feature that was not refused in either run means the two
builds' test items or descriptions differ, and then the encoder and decoder MAEM arms are not
being judged on the same items and the paired contrast is not the contrast it says it is.

WHAT IT BUILDS:

  pairs      per (MAEM arm x scorer): dec - enc per feature over the INTERSECTION of the two
             runs' features (`paired_diff`), its percentile bootstrap CI over features
             (`boot_ci`, the driver's resample count and seed), the win fraction, the same for
             the TPR and TNR halves, and each side's own full-set mean beside the paired one.
  per feat   `enc_vs_dec.csv`: one row per (feature, arm, scorer) over the union of both runs'
             rows and refusals, with the refusal flag of each side. A refused (feature, arm) has
             no score row; it is a row here with an empty bal_acc and `refused_* = 1`.
  replay     per replayed arm: how many features are identical on every stored rate and count,
             which differ, and whether every difference is on a feature refused in either run.
  bands      per (arm x scorer x band): each run's per-band recall (and foil specificity), and
             the paired dec - enc over the features both runs have in that band.
  fidelity   per feature: Exemplifier centred bo1 and bo8 cosine to the encoder direction (the
             encoder set's `score` product) and to the decoder direction (the decoder set's), and
             the 10M corpus top-1 centred cosine of each direction from its scan. Both cosines are
             ONE-SIDED on these rows (the `sae` family is not centrable: scorer centred, target
             the raw unit direction), identically on both sides, so the two are comparable to each
             other and are NOT the two-sided number the realact headline carries.

OTHER PAIRS OF RUNS. Nothing in the autointerp half is about encoders: it pairs ANY two run
directories that share features, test items and the C16 pool through one cache, by feature id.
`--names a,b` / `--titles A,B` relabel the outputs (file stems `<a>_vs_<b>*`, figures
`<a><b>-*-<b>`, `results_<a><b>.json`, headings and CSV columns); the defaults `enc,dec` /
`encoder,decoder` reproduce the M6-dec outputs. The JSON keeps the `enc_*`/`dec_*` keys and
records `names`, so `enc` there means the FIRST run (`--enc`) and `dec` the SECOND (`--dec`).
The fidelity half IS about the two directions of one MAEM and refuses non-default names.
M6-sft (2026-09-24) uses it as `--enc rl-final=... --dec sft-simple2m=... --names rl,sft
--titles RL,SFT --arms M --no-fidelity`: the SFT init's `M` arm against the RL checkpoint's.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.autointerp as A  # noqa: E402
import results.common as R  # noqa: E402
import results.corpus_search as CS  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The fields a replayed row must reproduce EXACTLY. Every one is written by `run.py` from the
# judge's answers on the build's items, so two replays of the same calls agree to the bit; a
# rounding tolerance here would hide a different item set that happened to score alike.
REPLAY_FIELDS = ("bal_acc", "tpr", "tnr", "tnr_zero", "tnr_nearmiss", "acc", "n_items",
                 "n_batches", "n_parsed", "n_pos", "n_neg_nearmiss", "draw")
# The Exemplifier ladder points the paper reports for eval 1 (bo1 = the plain mean, bo8).
FID_KS = (1, 8)
# The scan's per-size key for the corpus top-1 the paper's corpus-search cells use.
FID_SIZE = 10.0
# `score` stores `bo_c_<k>` computed in fp32 and `cos_centred.f16` is its fp16 cast; the recompute
# from the array can sit ~1e-3 from the stored value (fp16 spacing near 0.1 is 6e-5, but the max
# over up to 96 tokens and the mean over 64 rollouts both move with it). Above this the stored and
# recomputed ladders are two statistics -- which is what a product written before M0a's one
# estimator (2026-09-23) carries for k > 1 -- and the check says so rather than failing.
BOK_TOL = 2e-3
# The two sides' labels in the OUTPUTS (see the module docstring, "other pairs of runs").
DEFAULT_NAMES = ("enc", "dec")
DEFAULT_TITLES = ("encoder", "decoder")


def _relabel(text: str, names: tuple[str, str]) -> str:
    """A table header with the whole words `enc` / `dec` replaced by the two side names."""
    return re.sub(r"\b(enc|dec)\b", lambda m: names[0] if m.group(1) == "enc" else names[1],
                  text)


def _col(col: str, names: tuple[str, str]) -> str:
    """A CSV column name relabelled `_`-token-wise: `enc_mean` -> `<a>_mean`, `n_dec` -> `n_<b>`."""
    return "_".join({"enc": names[0], "dec": names[1]}.get(t, t) for t in col.split("_"))


def _parse_one(spec: str, flag: str) -> tuple[str, str]:
    runs = A.parse_runs([spec])
    assert len(runs) == 1, f"{flag} takes ONE `<label>=<run_dir>`, got {spec!r}"
    return next(iter(runs.items()))


def _f(v) -> float | None:
    """A finite float or None -- a null rate is an absent measurement, never 0."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------------------------
# the autointerp side
# ---------------------------------------------------------------------------------------------


def replay_check(enc: dict[int, dict], dec: dict[int, dict], refused: set[int]) -> dict:
    """Are a replayed arm's rows identical in the two runs, except where a call was re-sent?

    Two kinds of call are NOT served from the shared cache and are sent again by the second run,
    so on their features and only there the runs may legitimately differ:

      * an EXPLAINER refusal (`refused`, the union of the two runs' refused features for this
        arm): `run.py` retries it under `<key>|retry`, does not cache the decline, and the second
        run asks again -- one run may then have a row the other lacks;
      * a SCORER batch that came back unparsed (a judge refusal or an unreadable answer), read off
        the rows themselves as `n_parsed < n_batches` in either run: the second run re-sends that
        batch, and a batch answered there changes `n_items` / `n_parsed` / `acc` even when the
        rates happen to come out the same.

    Everything else must be the same calls replayed, bit for bit.
    """
    common = sorted(set(enc) & set(dec))
    differ = [f for f in common
              if any(enc[f].get(k) != dec[f].get(k) for k in REPLAY_FIELDS)]
    bal_differ = [f for f in common if enc[f].get("bal_acc") != dec[f].get("bal_acc")]
    only_enc, only_dec = sorted(set(enc) - set(dec)), sorted(set(dec) - set(enc))

    def short(r: dict) -> bool:
        nb, npd = r.get("n_batches"), r.get("n_parsed")
        return nb is not None and npd is not None and int(npd) < int(nb)

    resent = sorted(f for f in common if short(enc[f]) or short(dec[f]))
    unexplained = sorted(set(differ + only_enc + only_dec) - refused - set(resent))
    return {"n_enc": len(enc), "n_dec": len(dec), "n_common": len(common),
            "n_identical": len(common) - len(differ), "differ": differ,
            "bal_acc_differ": bal_differ, "only_enc": only_enc, "only_dec": only_dec,
            "refused_either": sorted(refused), "scorer_resent": sorted(set(resent) & set(differ)),
            "unexplained": unexplained, "ok": not unexplained}


def _paired(a: dict[int, float], b: dict[int, float], boot: int, seed: int) -> dict:
    """`a - b` over the shared features, with the driver's estimator on both sides and on the
    difference -- so the three intervals are the same resamples of the same features."""
    pd = A.paired_diff(a, b)
    feats = pd["features"]
    mean, lo, hi = A.boot_ci(pd["d"], boot, seed)
    am, alo, ahi = A.boot_ci([a[f] for f in feats], boot, seed)
    bm, blo, bhi = A.boot_ci([b[f] for f in feats], boot, seed)
    n = len(pd["d"])
    return {"mean": mean, "lo": lo, "hi": hi, "n_paired": pd["n_paired"],
            "a_mean": am, "a_lo": alo, "a_hi": ahi, "b_mean": bm, "b_lo": blo, "b_hi": bhi,
            "win_frac": float(np.mean(pd["d"] > 0)) if n else float("nan"),
            "loss_frac": float(np.mean(pd["d"] < 0)) if n else float("nan"),
            "n_zero": int(np.sum(pd["d"] == 0)),
            "only_in_a": pd["only_in_a"], "only_in_b": pd["only_in_b"]}


def compare(vol: R.Vol, enc: tuple[str, str], dec: tuple[str, str], arms: list[str],
            replay_arms: list[str], boot: int, seed: int, bands: bool = True) -> dict:
    """Everything the autointerp half of the tables and figures is built from."""
    (el, ed), (dl, dd) = enc, dec
    assert el != dl, f"--enc and --dec need different labels, both are {el!r}"
    rows: dict[str, list[dict]] = {}
    refusals: dict[str, dict[str, set[int]]] = {}
    checks: list[dict] = []
    notes: list[str] = []
    builds: dict[str, dict] = {}
    for label, run_dir in ((el, ed), (dl, dd)):
        got, build, why = A.load_run(vol, label, run_dir)
        assert not why, f"{label}: {why}"
        rows[label], builds[label] = got, build
        checks.append(A.reader_check(got, label, run_dir))
        ref, why = A.load_refusals(vol, label, run_dir)
        assert not why, (f"{why} -- the per-feature refusal flags are part of this comparison's "
                         f"contract, so an unknown refusal set is a refusal to compare")
        refusals[label] = {k.arm: v for k, v in ref.items()}
    by_cell = A.index_rows(rows[el] + rows[dl])
    scorers = sorted({r["scorer"] for r in rows[el]} | {r["scorer"] for r in rows[dl]})

    def cell(label: str, arm: str, scorer: str) -> dict[int, dict]:
        return by_cell.get((A.Arm(label, arm), scorer)) or {}

    pairs: list[dict] = []
    per_feature: list[dict] = []
    for arm in arms:
        for scorer in scorers:
            ce, cd = cell(el, arm, scorer), cell(dl, arm, scorer)
            re_, rd = refusals[el].get(arm, set()), refusals[dl].get(arm, set())
            if not ce and not cd:
                notes.append(f"`{arm}`/{scorer}: no rows in either run")
                continue
            rec = {"arm": arm, "scorer": scorer, "n_rows_enc": len(ce), "n_rows_dec": len(cd),
                   "n_refused_enc": len(re_), "n_refused_dec": len(rd)}
            for metric in ("bal_acc", "tpr", "tnr"):
                ve, vd = A.values_of(ce, metric), A.values_of(cd, metric)
                if not ve or not vd:
                    continue
                p = _paired(vd, ve, boot, seed)          # dec - enc
                rec[metric] = {**p, "dec_mean": p.pop("a_mean"), "dec_lo": p.pop("a_lo"),
                               "dec_hi": p.pop("a_hi"), "enc_mean": p.pop("b_mean"),
                               "enc_lo": p.pop("b_lo"), "enc_hi": p.pop("b_hi"),
                               "only_dec": p.pop("only_in_a"), "only_enc": p.pop("only_in_b"),
                               "n_enc": len(ve), "n_dec": len(vd)}
                # Each side's OWN full-set mean, the number its own driver table headlines.
                for side, vals in (("enc", ve), ("dec", vd)):
                    m, lo, hi = A.boot_ci(vals.values(), boot, seed)
                    rec[metric][f"{side}_own"] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
            pairs.append(rec)
            for feat in sorted(set(ce) | set(cd) | re_ | rd):
                e, d = ce.get(feat) or {}, cd.get(feat) or {}
                be, bd = _f(e.get("bal_acc")), _f(d.get("bal_acc"))
                per_feature.append({
                    "feature": feat, "arm": arm, "scorer": scorer,
                    "bal_acc_enc": be, "bal_acc_dec": bd,
                    "diff": None if be is None or bd is None else bd - be,
                    "tpr_enc": _f(e.get("tpr")), "tnr_enc": _f(e.get("tnr")),
                    "tpr_dec": _f(d.get("tpr")), "tnr_dec": _f(d.get("tnr")),
                    "refused_enc": int(feat in re_), "refused_dec": int(feat in rd),
                    "stratum": e.get("stratum", d.get("stratum")),
                })

    replay: list[dict] = []
    for arm in replay_arms:
        for scorer in scorers:
            ce, cd = cell(el, arm, scorer), cell(dl, arm, scorer)
            if not ce and not cd:
                continue
            refused = refusals[el].get(arm, set()) | refusals[dl].get(arm, set())
            replay.append({"arm": arm, "scorer": scorer, **replay_check(ce, cd, refused)})

    band_rs: list[dict] = []
    band_pf: dict[tuple, dict[int, float]] = {}
    if bands:
        band_rs, band_pf, band_notes = A.band_rows(vol, {el: ed, dl: dd}, by_cell, scorers,
                                                   boot, seed)
        notes += band_notes
    band_pairs: list[dict] = []
    for arm in [*replay_arms[:1], *arms]:
        for scorer in scorers:
            for slot in A.BAND_ORDER:
                half = "tnr" if slot == A.FOIL_SLOT else "tpr"
                ve = band_pf.get((f"{el}/{arm}", scorer, slot, half))
                vd = band_pf.get((f"{dl}/{arm}", scorer, slot, half))
                if not ve or not vd:
                    continue
                p = _paired(vd, ve, boot, seed)
                own = {}
                for side, vals in (("enc", ve), ("dec", vd)):
                    m, lo, hi = A.boot_ci(vals.values(), boot, seed)
                    own[side] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
                band_pairs.append({"arm": arm, "scorer": scorer, "band": slot, "half": half,
                                   "enc": own["enc"], "dec": own["dec"],
                                   "mean": p["mean"], "lo": p["lo"], "hi": p["hi"],
                                   "n_paired": p["n_paired"]})
    return {"enc": enc, "dec": dec, "arms": arms, "replay_arms": replay_arms, "scorers": scorers,
            "pairs": pairs, "per_feature": per_feature, "replay": replay, "bands": band_rs,
            "band_pairs": band_pairs, "checks": checks, "notes": notes, "builds": builds,
            "refusals": {lab: {a: sorted(v) for a, v in m.items()} for lab, m in refusals.items()},
            "own_cells": {(lab, arm, sc): A.values_of(cell(lab, arm, sc))
                          for lab in (el, dl) for arm in [*replay_arms[:1], *arms]
                          for sc in scorers},
            "boot": boot, "seed": seed}


# ---------------------------------------------------------------------------------------------
# the fidelity side
# ---------------------------------------------------------------------------------------------


def _feature_rows(ids: dict[int, dict], sae: str) -> dict[int, int]:
    """{SAE feature id: set row} for the set's rows of this dictionary. A feature on two rows is
    refused: the pairing below is BY FEATURE, and two rows would make it a choice."""
    out: dict[int, int] = {}
    for row, r in sorted(ids.items()):
        if r.get("family") != "sae" or (r.get("sae_key") and r["sae_key"] != sae):
            continue
        f = int(r["id"])
        assert f not in out, f"feature {f} sits on rows {out[f]} and {row} of one set"
        out[f] = row
    return out


def _unit_rows(vol: R.Vol, base: str, set_name: str) -> np.ndarray | None:
    idx = vol.json(f"base/{base}/heldout/{set_name}/index.json") or {}
    meta = idx.get("act.f32")
    if not meta:
        return None
    a = vol.array(f"base/{base}/heldout/{set_name}/act.f32", "float32", tuple(meta["shape"]))
    if a is None:
        return None
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def fidelity(vol: R.Vol, *, base: str, sae: str, maem: str, enc_set: str, dec_set: str,
             enc_scores: str, dec_scores: str, enc_scan: str, dec_scan: str,
             boot: int, seed: int) -> dict:
    """Per feature: Exemplifier centred bo-k to each direction, and each direction's corpus top-1."""
    ids = {"enc": CS.read_ids(vol, base, enc_set), "dec": CS.read_ids(vol, base, dec_set)}
    rows = {s: _feature_rows(ids[s], sae) for s in ids}
    feats = sorted(set(rows["enc"]) & set(rows["dec"]))
    checks: list[dict] = []
    # THE JOIN: the decoder set's row j carries the encoder set's row `ids_from_row`, and both must
    # name the same feature. Checked per row rather than trusted from the set's README.
    bad = [f for f in feats
           if ids["dec"][rows["dec"][f]].get("ids_from_row") not in (None, rows["enc"][f])]
    checks.append({"kind": "join", "detail": f"{dec_set} -> {enc_set}", "n": len(feats),
                   "only_enc": len(set(rows["enc"]) - set(rows["dec"])),
                   "only_dec": len(set(rows["dec"]) - set(rows["enc"])), "n_bad": len(bad)})
    assert not bad, f"{len(bad)} decoder rows name another encoder row than their feature's: {bad[:8]}"

    scores = {"enc": f"maems/{maem}/scores/{enc_scores}", "dec": f"maems/{maem}/scores/{dec_scores}"}
    bok: dict[tuple[str, int], np.ndarray] = {}
    for side, rel in scores.items():
        pt = {int(r["row"]): r for r in (vol.jsonl(f"{rel}/per_target.jsonl") or [])}
        sided = {int(pt[rows[side][f]].get("centred_sided") or 0) for f in feats
                 if rows[side][f] in pt}
        checks.append({"kind": "centred_sided", "detail": side, "values": sorted(sided)})
        for k in FID_KS:
            arr = CS.exemplifier_bok(vol, rel, k, "centred")
            assert arr is not None, f"{rel}: no cos_centred.f16 to compute the centred bo{k} from"
            bok[(side, k)] = arr
            # The stored ladder beside the recompute: bo1 must agree to the fp16 cast; a larger
            # gap at k > 1 is a product written under the pre-M0a group estimator.
            gaps = [abs(float(arr[rows[side][f]]) - float(pt[rows[side][f]][f"bo_c_{k}"]))
                    for f in feats if rows[side][f] in pt and f"bo_c_{k}" in pt[rows[side][f]]
                    and np.isfinite(arr[rows[side][f]])]
            checks.append({"kind": f"bo_c_{k} stored vs recomputed", "detail": side,
                           "n": len(gaps), "max_abs": max(gaps) if gaps else float("nan"),
                           "ok": bool(gaps) and max(gaps) <= BOK_TOL})
    top1 = {}
    for side, scan, set_name in (("enc", enc_scan, enc_set), ("dec", dec_scan, dec_set)):
        # No `exclusions.json` exists for either SAE set: the n-gram exclusion is a property of
        # the upstream realact rows (a document the corpus might contain), and an SAE direction has no
        # document. Passed as False DELIBERATELY, which `read_top1` requires to be explicit.
        t = CS.read_top1(vol, base, scan, set_name, family="sae", apply_exclusions=False,
                         sae_key=sae)
        CS.assert_complete(t)
        assert FID_SIZE in t.by_size, f"{scan}: no {FID_SIZE}M snapshot (sizes {t.sizes})"
        top1[side] = t.by_size[FID_SIZE]
        checks.append({"kind": "corpus top-1", "detail": f"{side} {scan}",
                       "n": len(t.by_size[FID_SIZE]), "sizes": t.sizes})
    units = {"enc": _unit_rows(vol, base, enc_set), "dec": _unit_rows(vol, base, dec_set)}

    per_feature: list[dict] = []
    for f in feats:
        re_, rd = rows["enc"][f], rows["dec"][f]
        rec = {"feature": f, "enc_row": re_, "dec_row": rd,
               "stratum": ids["enc"][re_].get("stratum"),
               "cos_enc_dec": (float(units["enc"][re_] @ units["dec"][rd])
                               if units["enc"] is not None and units["dec"] is not None else None)}
        for k in FID_KS:
            for side, row in (("enc", re_), ("dec", rd)):
                rec[f"ex_bo{k}_{side}"] = _f(bok[(side, k)][row])
        rec["corpus10_enc"] = _f(top1["enc"].get(re_))
        rec["corpus10_dec"] = _f(top1["dec"].get(rd))
        per_feature.append(rec)

    summary: list[dict] = []
    metrics = [*(f"ex_bo{k}" for k in FID_KS), "corpus10"]
    for m in metrics:
        e = {r["feature"]: r[f"{m}_enc"] for r in per_feature if r[f"{m}_enc"] is not None}
        d = {r["feature"]: r[f"{m}_dec"] for r in per_feature if r[f"{m}_dec"] is not None}
        p = _paired(d, e, boot, seed)
        summary.append({"metric": m, "what": "dec - enc", **p})
    # The paper's own fidelity contrast, Exemplifier bo-k minus the corpus top-1, on each side.
    for k in FID_KS:
        for side in ("enc", "dec"):
            a = {r["feature"]: r[f"ex_bo{k}_{side}"] for r in per_feature
                 if r[f"ex_bo{k}_{side}"] is not None}
            b = {r["feature"]: r[f"corpus10_{side}"] for r in per_feature
                 if r[f"corpus10_{side}"] is not None}
            summary.append({"metric": f"ex_bo{k} - corpus10", "what": side,
                            **_paired(a, b, boot, seed)})
    cos = [r["cos_enc_dec"] for r in per_feature if r["cos_enc_dec"] is not None]
    return {"per_feature": per_feature, "summary": summary, "checks": checks,
            "cos_enc_dec": ({"min": float(np.min(cos)), "median": float(np.median(cos)),
                             "max": float(np.max(cos)), "n": len(cos)} if cos else None),
            "sources": {"enc_scores": scores["enc"], "dec_scores": scores["dec"],
                        "enc_scan": f"base/{base}/scan/{enc_scan}",
                        "dec_scan": f"base/{base}/scan/{dec_scan}",
                        "enc_set": enc_set, "dec_set": dec_set}}


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------


def _ci(m, lo, hi, signed: bool = False) -> str:
    return A.ci(m, lo, hi, 4, signed)


def _md(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(R._cell(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(R._cell(v) for v in r) + " |" for r in rows]
    return out + [""]


def _ids(xs: list[int], cap: int = 12) -> str:
    return ", ".join(str(x) for x in xs[:cap]) + (f" (+{len(xs) - cap})" if len(xs) > cap else "") \
        if xs else "none"


def write_outputs(res: dict, fid: dict | None, out: Path, preamble: list[str],
                  figures: list[str], names: tuple[str, str] = DEFAULT_NAMES,
                  titles: tuple[str, str] = DEFAULT_TITLES) -> Path:
    (el, _ed), (dl, _dd) = res["enc"], res["dec"]
    (na, nb), (ta, tb) = names, titles
    stem = f"{na}_vs_{nb}"

    def H(cols: list[str]) -> list[str]:
        return [_relabel(c, names) for c in cols]

    def C(cols: list[str]) -> list[str]:
        return [_col(c, names) for c in cols]

    L = [f"# Eval 2 — {ta} vs {tb} autointerp, paired per feature", "", *preamble, ""]

    # --- headline ---------------------------------------------------------------------------
    head, head_csv = [], []
    for p in res["pairs"]:
        b = p.get("bal_acc")
        if not b:
            continue
        head.append([p["arm"], p["scorer"], _ci(b["enc_mean"], b["enc_lo"], b["enc_hi"]),
                     _ci(b["dec_mean"], b["dec_lo"], b["dec_hi"]),
                     _ci(b["mean"], b["lo"], b["hi"], True), b["n_paired"],
                     f"{b['win_frac']:.3f} / {b['loss_frac']:.3f} / {b['n_zero']}",
                     _ci(p["tpr"]["mean"], p["tpr"]["lo"], p["tpr"]["hi"], True),
                     _ci(p["tnr"]["mean"], p["tnr"]["lo"], p["tnr"]["hi"], True),
                     f"{b['n_enc']} / {b['n_dec']}", f"{p['n_refused_enc']} / {p['n_refused_dec']}",
                     _ci(b["enc_own"]["mean"], b["enc_own"]["lo"], b["enc_own"]["hi"]),
                     _ci(b["dec_own"]["mean"], b["dec_own"]["lo"], b["dec_own"]["hi"])])
        head_csv.append([p["arm"], p["scorer"], b["enc_mean"], b["enc_lo"], b["enc_hi"],
                         b["dec_mean"], b["dec_lo"], b["dec_hi"], b["mean"], b["lo"], b["hi"],
                         b["n_paired"], b["win_frac"], b["loss_frac"], b["n_zero"],
                         p["tpr"]["mean"], p["tpr"]["lo"], p["tpr"]["hi"],
                         p["tnr"]["mean"], p["tnr"]["lo"], p["tnr"]["hi"],
                         b["n_enc"], b["n_dec"], p["n_refused_enc"], p["n_refused_dec"],
                         b["enc_own"]["mean"], b["enc_own"]["lo"], b["enc_own"]["hi"],
                         b["dec_own"]["mean"], b["dec_own"]["lo"], b["dec_own"]["hi"]])
    L += [f"## Headline: balanced accuracy, {tb} − {ta}, paired per feature", "",
          f"*Per (arm, scorer): the mean over the features BOTH runs scored of each run's "
          f"per-feature `bal_acc`, and of `{nb} − {na}`, each with a percentile bootstrap over those "
          f"features ({res['boot']} resamples, seed {res['seed']}, the driver's `boot_ci`; the "
          f"same resample indices on all three, so the intervals are comparable). `win / loss / "
          f"tie` is the fraction of paired features where the {tb} arm scored higher / lower, "
          f"and the count of exact ties. `Δ TPR` and `Δ TNR` are the same paired contrast on each "
          f"half of the balanced accuracy. `feats` is each run's own feature count, `refused` "
          f"each run's declined explainer calls for the arm (no score row); the last two columns "
          f"are each run's own full-set mean, the number its own driver table headlines.*", ""]
    L += _md(H(["arm", "scorer", "enc (paired)", "dec (paired)", "dec − enc", "n paired",
                "win / loss / tie", "Δ TPR", "Δ TNR", "feats enc / dec", "refused enc / dec",
                "enc own", "dec own"]), head)
    R.write_csv(out / f"{stem}_summary.csv",
                C(["arm", "scorer", "enc_mean", "enc_lo", "enc_hi", "dec_mean", "dec_lo", "dec_hi",
                 "diff_mean", "diff_lo", "diff_hi", "n_paired", "win_frac", "loss_frac", "n_tie",
                 "dtpr_mean", "dtpr_lo", "dtpr_hi", "dtnr_mean", "dtnr_lo", "dtnr_hi",
                 "n_enc", "n_dec", "n_refused_enc", "n_refused_dec", "enc_own_mean",
                 "enc_own_lo", "enc_own_hi", "dec_own_mean", "dec_own_lo", "dec_own_hi"]), head_csv)

    # --- replay -------------------------------------------------------------------------------
    rep = []
    for x in res["replay"]:
        rep.append([x["arm"], x["scorer"], f"{x['n_enc']} / {x['n_dec']}", x["n_common"],
                    x["n_identical"], _ids(x["differ"]), _ids(x["bal_acc_differ"]),
                    _ids(x["only_enc"]), _ids(x["only_dec"]), _ids(x["refused_either"]),
                    _ids(x["scorer_resent"]), "PASS" if x["ok"] else
                    f"FAIL: {_ids(x['unexplained'])}"])
    L += ["## Replay check: the arms both runs share through the cache", "",
          f"*A replayed arm's rows compared field by field ({', '.join(REPLAY_FIELDS)}), exact "
          f"equality. PASS iff every feature that differs, or has a row in one run only, is one "
          f"whose call the second run had to RE-SEND because it is not in the cache: an explainer "
          f"refusal in either run (retried, not cached), or a scorer batch that came back "
          f"unparsed in either run (`n_parsed < n_batches`, e.g. a judge refusal). "
          f"`{res['replay_arms'][0]}` is the one that matters: its test items are the "
          f"items every MAEM arm is judged on.*", ""]
    L += _md(H(["arm", "scorer", "rows enc / dec", "common", "identical", "differ (any field)",
                "differ (bal_acc)", "only enc", "only dec", "explainer refused (either run)",
                "differ, scorer batch re-sent", "verdict"]), rep)
    R.write_csv(out / f"{stem}_replay.csv",
                C(["arm", "scorer", "n_enc", "n_dec", "n_common", "n_identical", "differ",
                   "bal_acc_differ", "only_enc", "only_dec", "refused_either", "scorer_resent",
                   "ok"]),
                [[x["arm"], x["scorer"], x["n_enc"], x["n_dec"], x["n_common"], x["n_identical"],
                  " ".join(map(str, x["differ"])), " ".join(map(str, x["bal_acc_differ"])),
                  " ".join(map(str, x["only_enc"])),
                  " ".join(map(str, x["only_dec"])), " ".join(map(str, x["refused_either"])),
                  " ".join(map(str, x["scorer_resent"])), int(x["ok"])] for x in res["replay"]])

    # --- refusals -----------------------------------------------------------------------------
    L += ["## Refusals (declined explainer calls, after the retry)", ""]
    arms_all = sorted({a for m in res["refusals"].values() for a in m})
    L += _md(["arm", f"`{el}`", f"`{dl}`", "both", f"{na} only", f"{nb} only"],
             [[a, len(res["refusals"][el].get(a, [])), len(res["refusals"][dl].get(a, [])),
               len(set(res["refusals"][el].get(a, [])) & set(res["refusals"][dl].get(a, []))),
               len(set(res["refusals"][el].get(a, [])) - set(res["refusals"][dl].get(a, []))),
               len(set(res["refusals"][dl].get(a, [])) - set(res["refusals"][el].get(a, [])))]
              for a in arms_all])

    # --- bands --------------------------------------------------------------------------------
    brow, bcsv = [], []
    for x in res["band_pairs"]:
        brow.append([x["arm"], x["scorer"], x["band"], x["half"],
                     f"{_ci(x['enc']['mean'], x['enc']['lo'], x['enc']['hi'])} (n={x['enc']['n']})",
                     f"{_ci(x['dec']['mean'], x['dec']['lo'], x['dec']['hi'])} (n={x['dec']['n']})",
                     _ci(x["mean"], x["lo"], x["hi"], True), x["n_paired"]])
        bcsv.append([x["arm"], x["scorer"], x["band"], x["half"], x["enc"]["mean"],
                     x["enc"]["lo"], x["enc"]["hi"], x["enc"]["n"], x["dec"]["mean"],
                     x["dec"]["lo"], x["dec"]["hi"], x["dec"]["n"], x["mean"], x["lo"], x["hi"],
                     x["n_paired"]])
    L += [f"## Per activation band: recall (b1..b4, btop) and foil specificity (b0), {na} vs {nb}", "",
          "*`band_rows`' join of each run's build rows against its scorer's per-item answers. "
          "b1..b4 are the build's four EQUAL-WIDTH bins of (0, corpus peak], lowest first, "
          "per feature (not comparable across features); btop is the top-beyond-shown fallback "
          "tier; b0 is specificity on every non-activating item. Each side's mean is over its "
          f"own features with an item in that band; `{nb} − {na}` is paired over the features both "
          "have there.*", ""]
    L += _md(H(["arm", "scorer", "band", "half", "enc", "dec", "dec − enc", "n paired"]), brow)
    R.write_csv(out / f"{stem}_bands.csv",
                C(["arm", "scorer", "band", "half", "enc_mean", "enc_lo", "enc_hi", "enc_n",
                   "dec_mean", "dec_lo", "dec_hi", "dec_n", "diff_mean", "diff_lo", "diff_hi",
                   "n_paired"]), bcsv)

    # --- per feature --------------------------------------------------------------------------
    pf_cols = ["feature", "arm", "scorer", "bal_acc_enc", "bal_acc_dec", "diff", "tpr_enc",
               "tnr_enc", "tpr_dec", "tnr_dec", "refused_enc", "refused_dec", "stratum"]
    R.write_csv(out / f"{stem}.csv", C(pf_cols),
                [[r[c] for c in pf_cols] for r in res["per_feature"]])

    # --- fidelity -----------------------------------------------------------------------------
    if fid is not None:
        frows = []
        for s in fid["summary"]:
            frows.append([s["metric"], s["what"], _ci(s["b_mean"], s["b_lo"], s["b_hi"]),
                          _ci(s["a_mean"], s["a_lo"], s["a_hi"]),
                          _ci(s["mean"], s["lo"], s["hi"], True), s["n_paired"],
                          f"{s['win_frac']:.3f}"])
        src = fid["sources"]
        L += ["## Fidelity: Exemplifier and corpus search against each direction", "",
              f"*Per feature, centred cosine of the Exemplifier's rollouts to the direction it was "
              f"injected with — the encoder column (`{src['enc_scores']}`) or the decoder row "
              f"(`{src['dec_scores']}`) — as the unbiased best-of-k over 64 rollouts recomputed from "
              f"`cos_centred.f16` (`corpus_search.exemplifier_bok`, i.e. `common.best_per_rollout` "
              f"then `common.bo_unbiased`), and the 10M corpus top-1 centred cosine of the same "
              f"direction (`corpus_search.read_top1`, `{src['enc_scan']}` for the encoder set, "
              f"`{src['dec_scan']}` for the decoder set, family `sae`, no exclusions exist for "
              f"either set). **Both are ONE-SIDED on these rows** — the scorer side is centred about "
              f"the scoring mean, the target is the raw unit direction, because the `sae` family is "
              f"not centrable — identically on both sides, so enc and dec are comparable to each "
              f"other and not to the two-sided realact headline. Rows `dec − enc`: columns are enc, "
              f"dec, paired difference. Rows `ex_bok − corpus10`: columns are corpus, Exemplifier, "
              f"paired difference, on that side. Same percentile bootstrap over features.*", ""]
        L += _md(["metric", "side", "enc | corpus", "dec | Exemplifier", "difference", "n",
                  "win frac"], frows)
        if fid["cos_enc_dec"]:
            c = fid["cos_enc_dec"]
            L += [f"cos(unit encoder column, unit decoder row) over the {c['n']} features: min "
                  f"{c['min']:.4f}, median {c['median']:.4f}, max {c['max']:.4f} (from the two "
                  f"sets' `act.f32`).", ""]
        fcols = ["feature", "enc_row", "dec_row", "stratum", "cos_enc_dec",
                 *(f"ex_bo{k}_{s}" for k in FID_KS for s in ("enc", "dec")),
                 "corpus10_enc", "corpus10_dec"]
        R.write_csv(out / "fidelity_enc_vs_dec.csv", fcols,
                    [[r[c] for c in fcols] for r in fid["per_feature"]])
        R.write_csv(out / "fidelity_summary.csv",
                    ["metric", "side", "first_mean", "first_lo", "first_hi", "second_mean",
                     "second_lo", "second_hi", "diff_mean", "diff_lo", "diff_hi", "n_paired",
                     "win_frac"],
                    [[s["metric"], s["what"], s["b_mean"], s["b_lo"], s["b_hi"], s["a_mean"],
                      s["a_lo"], s["a_hi"], s["mean"], s["lo"], s["hi"], s["n_paired"],
                      s["win_frac"]] for s in fid["summary"]])

    # --- checks -------------------------------------------------------------------------------
    L += ["## Checks", ""]
    for c in res["checks"]:
        L.append(f"- reader `{c['who']}` ({c['detail']}): " + (
            f"skipped, {c['skipped']}" if "skipped" in c else
            f"{c['comparisons']} stored bal_acc recomputed as 0.5·(TPR+TNR), "
            f"{c['n_mismatches']} mismatches"))
    for c in (fid or {}).get("checks", []):
        L.append("- fidelity " + ", ".join(f"{k} {v}" for k, v in c.items()))
    L += [f"- NOTE {n}" for n in res["notes"]] + [""]
    if figures:
        L += ["## Figures", ""] + [f"- `figures/{f}.pdf` / `.png`" for f in figures] + [""]
    path = out / f"{stem}_summary.md"
    path.write_text("\n".join(L))
    return path


def make_figures(res: dict, fid: dict | None, out: Path, names: tuple[str, str] = DEFAULT_NAMES,
                 titles: tuple[str, str] = DEFAULT_TITLES) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (el, _), (dl, _) = res["enc"], res["dec"]
    (na, nb), (ta, tb) = names, titles
    ce, cd = R.PALETTE[0], R.PALETTE[1]
    ref = res["replay_arms"][0]
    names: list[str] = []
    det = "detection"

    # (a) paired scatter of per-feature detection bal_acc, one panel per MAEM arm
    arms = res["arms"]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.3 * len(arms), 3.4), squeeze=False)
    for ax, arm in zip(axes[0], arms, strict=True):
        e, d = res["own_cells"][(el, arm, det)], res["own_cells"][(dl, arm, det)]
        fs = sorted(set(e) & set(d))
        ax.plot([0, 1], [0, 1], color=R.INK_MUTED, lw=0.8, ls="--", zorder=1)
        ax.axhline(A.CHANCE, color=R.GRID, lw=0.8, zorder=0)
        ax.axvline(A.CHANCE, color=R.GRID, lw=0.8, zorder=0)
        ax.scatter([e[f] for f in fs], [d[f] for f in fs], s=9, color=R.PALETTE[5], alpha=0.35,
                   lw=0, zorder=2, label=f"feature (n={len(fs)})")
        me, md = np.mean([e[f] for f in fs]), np.mean([d[f] for f in fs])
        ax.scatter([me], [md], s=70, marker="D", color=R.PALETTE[5], edgecolor=R.SURFACE,
                   lw=1.5, zorder=4, label=f"{arm} mean")
        re_, rd = res["own_cells"][(el, ref, det)], res["own_cells"][(dl, ref, det)]
        if re_ and rd:
            ax.scatter([np.mean(list(re_.values()))], [np.mean(list(rd.values()))], s=110,
                       marker="*", color=R.INK, edgecolor=R.SURFACE, lw=1.0, zorder=4,
                       label=f"{ref} mean (reference)")
        ax.set_xlim(0.2, 1.02)
        ax.set_ylim(0.2, 1.02)
        ax.set_aspect("equal")
        R.style_axes(ax, xlabel=f"{ta} run, bal_acc", ylabel=f"{tb} run, bal_acc", title=arm)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Detection balanced accuracy per feature, {ta} vs {tb}", fontsize=10,
                 color=R.INK, x=0.01, ha="left")
    fig.tight_layout()
    names.append(R.savefig(fig, out, f"{na}{nb}-scatter-{nb}"))

    # (b) mean bal_acc per arm, enc and dec, each run's own features, both scorers
    show = [ref, *arms]
    fig, axes = plt.subplots(1, len(res["scorers"]), figsize=(4.2 * len(res["scorers"]), 3.2),
                             squeeze=False, sharey=True)
    for ax, sc in zip(axes[0], res["scorers"], strict=True):
        for j, (lab, col, side) in enumerate(((el, ce, ta), (dl, cd, tb))):
            xs, ms, los, his = [], [], [], []
            for i, arm in enumerate(show):
                v = res["own_cells"].get((lab, arm, sc)) or {}
                if not v:
                    continue
                m, lo, hi = A.boot_ci(v.values(), res["boot"], res["seed"])
                xs.append(i + (j - 0.5) * 0.28)
                ms.append(m)
                los.append(m - lo)
                his.append(hi - m)
            ax.errorbar(xs, ms, yerr=[los, his], fmt="o", ms=6, color=col, ecolor=col,
                        elinewidth=1.5, capsize=0, label=side, zorder=3)
        ax.axhline(A.CHANCE, color=R.INK_MUTED, lw=0.8, ls="--", zorder=1)
        ax.set_xticks(range(len(show)), show)
        R.style_axes(ax, ylabel="mean bal_acc [95% CI]" if sc == res["scorers"][0] else "",
                     title=sc)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    names.append(R.savefig(fig, out, f"{na}{nb}-means-{nb}"))

    # (c) per-band TPR enc vs dec per arm, detection and fuzzing
    slots = [s for s in A.BAND_ORDER if s != A.FOIL_SLOT]
    bp = {(x["arm"], x["scorer"], x["band"]): x for x in res["band_pairs"]}
    if bp:
        fig, axes = plt.subplots(len(res["scorers"]), len(show),
                                 figsize=(2.7 * len(show), 2.5 * len(res["scorers"])),
                                 squeeze=False, sharey=True)
        for r_i, sc in enumerate(res["scorers"]):
            for c_i, arm in enumerate(show):
                ax = axes[r_i][c_i]
                here = [s for s in slots if (arm, sc, s) in bp]
                for j, (side, col, name) in enumerate((("enc", ce, ta), ("dec", cd, tb))):
                    xs = [i + (j - 0.5) * 0.2 for i in range(len(here))]
                    ms = [bp[(arm, sc, s)][side]["mean"] for s in here]
                    lo = [m - bp[(arm, sc, s)][side]["lo"] for m, s in zip(ms, here, strict=True)]
                    hi = [bp[(arm, sc, s)][side]["hi"] - m for m, s in zip(ms, here, strict=True)]
                    ax.errorbar(xs, ms, yerr=[lo, hi], fmt="o-", ms=4, lw=1.2, color=col,
                                elinewidth=1.2, capsize=0, label=name)
                ax.set_xticks(range(len(here)), here)
                ax.set_ylim(0, 1.02)
                R.style_axes(ax, ylabel=f"{sc} recall" if c_i == 0 else "",
                             title=arm if r_i == 0 else "")
                if r_i == 0 and c_i == 0:
                    ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle("Recall per activation band (b1 lowest .. b4 highest, btop fallback)",
                     fontsize=10, color=R.INK, x=0.01, ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out, f"{na}{nb}-bands-{nb}"))

    # (d) fidelity scatter
    if fid is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), squeeze=False)
        for ax, (m, title) in zip(axes[0], (("ex_bo8", "Exemplifier centred bo8"),
                                            ("corpus10", "corpus top-1 at 10M")), strict=True):
            pts = [(r[f"{m}_enc"], r[f"{m}_dec"]) for r in fid["per_feature"]
                   if r[f"{m}_enc"] is not None and r[f"{m}_dec"] is not None]
            x, y = np.array(pts).T
            lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            pad = 0.03 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=R.INK_MUTED, lw=0.8,
                    ls="--", zorder=1)
            ax.scatter(x, y, s=9, color=R.PALETTE[5], alpha=0.35, lw=0, zorder=2,
                       label=f"feature (n={len(x)})")
            ax.scatter([x.mean()], [y.mean()], s=70, marker="D", color=R.PALETTE[5],
                       edgecolor=R.SURFACE, lw=1.5, zorder=4, label="mean")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            R.style_axes(ax, xlabel="to the encoder direction", ylabel="to the decoder direction",
                         title=title)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle("Centred cosine (one-sided), per feature", fontsize=10, color=R.INK, x=0.01,
                     ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out, "encdec-fidelity-dec"))
    return names


def _commit() -> dict:
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "-uno", "--", "."],
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "why": str(exc)}
    return {"commit": head, "results_dir_dirty": bool(dirty), "dirty_files": dirty.splitlines()}


@app.command()
def main(
    enc: Annotated[str, typer.Option(help="`<label>=<run_dir>` of the ENCODER run")]
    = "rl-final=2026-09-24_autointerp-512",
    dec: Annotated[str, typer.Option(help="`<label>=<run_dir>` of the DECODER run")]
    = "rl-final-dec=2026-09-24_autointerp-512-dec",
    arms: Annotated[str, typer.Option(help="the arms compared, comma-separated")]
    = "M,M-jac16,M-cos16",
    replay: Annotated[str, typer.Option(
        help="arms both runs replay through the shared cache; the FIRST is the reference marker "
             "and the one whose bands are shown")] = "C16,C16-draw2,C16-judge2,R-shuffled",
    sae: Annotated[str, typer.Option(help="the SAE key")] = "qwen36-27b/l42-1b",
    bands: Annotated[bool, typer.Option(help="per-activation-band join (needs both builds)")] = True,
    fid: Annotated[bool, typer.Option("--fidelity/--no-fidelity",
                                      help="the Exemplifier / corpus-search side")] = True,
    base: Annotated[str, typer.Option()] = "qwen36-27b",
    maem: Annotated[str, typer.Option()] = "qwen36-27b/2026-09-18_rl-final",
    enc_set: Annotated[str, typer.Option()] = "2026-09-21_v3_ctrl",
    dec_set: Annotated[str, typer.Option()] = "2026-09-24_v3_ctrl_dec",
    enc_scores: Annotated[str, typer.Option()] = "2026-09-21_v3_ctrl__vllm__paper0923",
    dec_scores: Annotated[str, typer.Option()] = "2026-09-24_v3_ctrl_dec__vllm__paper0923",
    enc_scan: Annotated[str, typer.Option(
        help="the scan carrying the encoder set's rows (via --with-set)")]
    = "2026-09-21_v3_realact__train_parity_10m__paper0923",
    dec_scan: Annotated[str, typer.Option()] = "2026-09-24_v3_ctrl_dec__train_parity_10m__paper0923",
    out: Annotated[Path | None, typer.Option(help="output dir; default <repo>/_out/autointerp-dec")]
    = None,
    root: Annotated[str, typer.Option(help="volume-relative root")] = "",
    data: Annotated[Path | None, typer.Option(help="local mirror of the volume")] = None,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    boot: Annotated[int, typer.Option()] = R.N_BOOT,
    seed: Annotated[int, typer.Option()] = R.BOOT_SEED,
    figures: Annotated[bool, typer.Option(help="write figures/ (PDF + PNG)")] = True,
    names: Annotated[str, typer.Option(
        help="`<first>,<second>`: the two sides' short names in file names, headings and CSV "
             "columns (default reproduces the M6-dec outputs)")] = ",".join(DEFAULT_NAMES),
    titles: Annotated[str, typer.Option(
        help="`<first>,<second>`: the two sides' long names in headings and figures")]
    = ",".join(DEFAULT_TITLES),
) -> None:
    nm = tuple(x.strip() for x in names.split(","))
    tt = tuple(x.strip() for x in titles.split(","))
    if len(nm) != 2 or len(tt) != 2 or not all(nm) or not all(tt) or nm[0] == nm[1]:
        raise typer.BadParameter(f"--names / --titles take two distinct comma-separated labels, "
                                 f"got {names!r} / {titles!r}")
    if not re.fullmatch(r"[A-Za-z0-9-]+", nm[0]) or not re.fullmatch(r"[A-Za-z0-9-]+", nm[1]):
        raise typer.BadParameter(f"--names {names!r}: letters, digits and '-' only (they are file "
                                 f"stems and `_`-separated CSV tokens)")
    if fid and nm != DEFAULT_NAMES:
        raise typer.BadParameter(
            f"--names {names!r} with --fidelity: the fidelity half compares ONE MAEM's rollouts "
            f"against the encoder and decoder directions, which is not a comparison of two "
            f"arbitrary runs; pass --no-fidelity")
    out = Path(out) if out else R.out_dir("autointerp-dec")
    out.mkdir(parents=True, exist_ok=True)
    vol = R.Vol(root, Path(data) if data else R.mirror_dir(root), modal_cmd, False, True,
                offline=not fetch)
    e, d = _parse_one(enc, "--enc"), _parse_one(dec, "--dec")
    arm_l = [a.strip() for a in arms.split(",") if a.strip()]
    rep_l = [a.strip() for a in replay.split(",") if a.strip()]
    res = compare(vol, e, d, arm_l, rep_l, boot, seed, bands)
    fres = (fidelity(vol, base=base, sae=sae, maem=maem, enc_set=enc_set, dec_set=dec_set,
                     enc_scores=enc_scores, dec_scores=dec_scores, enc_scan=enc_scan,
                     dec_scan=dec_scan, boot=boot, seed=seed) if fid else None)
    figs = make_figures(res, fres, out, nm, tt) if figures else []
    prov = {"command": " ".join(sys.argv), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **_commit(), "enc_run": f"/vol/runs/{e[1]}", "dec_run": f"/vol/runs/{d[1]}",
            "builds": {lab: A.resolve_build(vol, run_dir)[0] for lab, run_dir in (e, d)},
            "band_builds": sorted({b["build_dir"] for b in res["bands"]}),
            "fidelity_sources": (fres or {}).get("sources"), "sae": sae, "boot": boot,
            "seed": seed, "mirror": str(vol.local), "figures": figs, "names": list(nm),
            "titles": list(tt)}
    (out / "figures").mkdir(exist_ok=True)
    (out / "figures" / f"provenance-{nm[1]}.json").write_text(json.dumps(prov, indent=1))
    preamble = [
        f"- command: `{prov['command']}`",
        f"- code: `{prov.get('commit')}`" + (" (results/ has uncommitted changes)"
                                              if prov.get("results_dir_dirty") else ""),
        f"- {tt[0]} run `{e[0]}` = `{prov['enc_run']}`; {tt[1]} run `{d[0]}` = `{prov['dec_run']}`",
        f"- per-band builds: {', '.join(f'`{b}`' for b in prov['band_builds']) or 'not joined'}",
        f"- SAE `{sae}`; intervals: percentile bootstrap over features, {boot} resamples, seed {seed}",
        f"- mirror `{vol.local}`",
    ]
    path = write_outputs(res, fres, out, preamble, figs, nm, tt)
    with open(out / f"results_{nm[0]}{nm[1]}.json", "w") as fh:
        json.dump({"names": {"enc": nm[0], "dec": nm[1]},
                   "compare": {k: v for k, v in res.items() if k not in ("own_cells", "builds")},
                   "fidelity": fres, "provenance": prov}, fh, indent=1, default=str)
    print(f"[{nm[0]}{nm[1]}] {path}")
    for x in res["replay"]:
        print(f"   replay {x['arm']}/{x['scorer']}: {'PASS' if x['ok'] else 'FAIL'} "
              f"({x['n_identical']}/{x['n_common']} identical, differ {x['differ'][:8]}, "
              f"bal_acc differ {x['bal_acc_differ'][:8]}, re-sent {x['scorer_resent'][:8]}, "
              f"only enc {x['only_enc'][:8]}, only dec {x['only_dec'][:8]})")
    for c in res["checks"]:
        print(f"   reader {c['who']}: {c.get('n_mismatches', c.get('skipped'))} mismatches")
    for n in res["notes"]:
        print(f"   NOTE   {n}")
    for c in (fres or {}).get("checks", []):
        print(f"   fidelity check {c}")


if __name__ == "__main__":
    app()
