"""Product `centred`: the two SECONDARY per-rollout cosines, added to an existing scores dir.

    <root>/maemms/<base>/<maemm>/scores/<set>/cos_centred_best.f16   [N, n]
    <root>/maemms/<base>/<maemm>/scores/<set>/cos_filtered_best.f16  [N, n]
    <root>/maemms/<base>/<maemm>/scores/<set>/centred.json           the aggregates + drop rates

Both are computed from what `score` already stored -- `best_act.f16`, `cos.f16`, `norm.f16` -- so
this product runs on a CPU container, loads no model and re-scores nothing. It writes INTO the
scores directory (`common.OutDir(keep_existing=True)`): the README is rewritten with the new files
appended to the file table and `index.json` gains the two array entries.

**Primary vs secondary (README "Methods", checklist item 60).** The pipeline's primary number is
the stored, UNCENTRED per-token cosine `cos.f16`, max over kept tokens. There is exactly one
centring in the pipeline and it happens at construction: a `realact` target is `unit(X[p] - mu)`
with `mu = base/<base>/stats/mu.f32`. So the target is already centred and the activation it is
compared against is not -- Celeste's asymmetry, kept deliberately.

`cos_centred_best` is the SECONDARY reading of the same rollout: `cos(best_act[i, k] - mu, v_i)`,
i.e. both sides centred by the same mu, evaluated at the argmax token the primary cosine chose.
It is a different statistic, not a correction:

  * for `realact` it is the "both sides centred" convention (v was built as unit(X[p] - mu));
  * for `sae` and `random` the target was NEVER centred (an encoder column and a Gaussian draw have
    no mean to subtract), so subtracting mu from the activation alone is a one-sided change and the
    number is a diagnostic only. It is still written for every family, because the comparison
    across families at a fixed convention is the point of the table.

`cos_filtered_best` is the primary cosine with Celeste's 10x-nanmedian norm filter applied
(eval_universal.py:71,145-147). Checklist items 4 and 11: the products store per-token values
UNFILTERED and the filter is an option here. Per rollout row, the median is taken over that row's
kept tokens' residual norms, tokens above `10 x median` are dropped, and the max is retaken over
what is left (-1.0 for a row that keeps nothing, matching `common.agg`'s masked_fill(-1)). The
fraction of kept tokens the filter drops is measured and reported -- on a full 27B re-embed it was
0 of 98,304 (checklist item 4), so a non-zero rate here is itself a finding.

The argmax position is NOT recomputed for the centred variant: it stays the primary's argmax,
because `best_act.f16` is the residual at that token and nothing else was stored. A row whose
primary kept no token at all (`argmax.i16 == -1`) is NaN in both outputs, never -1.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

# rows of the [N, n, d] best_act array handled per chunk; d is 4096/5120, so 4096 rows is ~40 MB
CHUNK = 4096
NORM_FILTER_MULT = 10.0  # eval/eval_universal.py:71 -- 10x the row's nanmedian residual norm


def _load_dirs(cfg, args, notes=None):
    """(rows_meta, dirs [N_set, d] fp32 UNIT, the run's mu, source) -- what `score` paired rows with.

    The mu comes back with the directions because BOTH SIDES of the centred cosine must use it.
    Returning only the directions is what made this product compute `cos(best_act - stats_mu,
    unit(act - whiten_mu))` for every `rl-last16` run -- two sides, two means, and the README
    asserting they were the same one.
    """
    base, root, set_name = args["base"], args["root"], args["heldout"]
    d = cfg["bases"][base]["d"]
    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{src}/ids.jsonl")
    mu, _ = C.mu_for(cfg, base, src, args, args.get("maemm") or "", root, notes)
    v = C.dirs_for(cfg, base, src, mu, root, notes)
    assert v.shape == (len(rows), d), f"{src}: dirs_for returned {v.shape} for {len(rows)} rows"
    return rows, np.asarray(v, dtype=np.float32), mu, src


def run(cfg, args):
    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base, "product centred needs --base"
    assert maemm, "product centred needs --maemm (the scores dir it extends)"
    assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    engine = args.get("engine") or "hf"
    d = cfg["bases"][base]["d"]

    sdir = C.scores_dir(maemm, set_name, root, engine, args.get("run_tag") or "")
    assert os.path.exists(sdir), (
        f"no scores at {sdir}: run `--product score --maemm {maemm} --set {set_name} "
        f"--engine {engine}` first (this product only reads what score.py stored)"
    )
    with open(f"{sdir}/index.json") as fh:
        index = json.load(fh)
    for name in ("cos.f16", "norm.f16", "argmax.i16", "best_act.f16"):
        assert name in index, (
            f"{sdir}/index.json has no {name}: this looks like a --rescore-texts variant "
            f"(FLAT arrays, no best_act), which `centred` cannot extend"
        )
    n_t, n, width = index["cos.f16"]["shape"]
    assert index["best_act.f16"]["shape"] == [n_t, n, d], (
        f"{sdir}: best_act.f16 is {index['best_act.f16']['shape']}, expected [{n_t}, {n}, {d}]"
    )
    # The width is the DIRECTORY's, not the module constant's: an arm whose rollouts summary asked
    # for a wider scoring window (the NLA arm, 256) stores wider arrays and records the length in
    # its rows.json. common.score_width_of reads it back, defaulting to SCORE_WIDTH.
    want_width = C.score_width_of(sdir)
    assert width == want_width, (
        f"{sdir}: cos.f16 width {width} != the {want_width} its rows.json score_max_length implies"
    )
    with open(f"{sdir}/rows.json") as fh:
        rows_json = json.load(fh)
    sel = list(rows_json["rows"])
    assert len(sel) == n_t and int(rows_json["n"]) == n, (
        f"{sdir}: rows.json says {len(rows_json['rows'])} rows x n={rows_json['n']} but the arrays "
        f"are [{n_t}, {n}, ...]"
    )

    cen_notes: list[str] = []
    rows_meta, dirs, mu_val, dirs_src = _load_dirs(cfg, args, notes=cen_notes)
    assert max(sel) < len(rows_meta), (
        f"{sdir}/rows.json names row {max(sel)} but {dirs_src}/ids.jsonl has {len(rows_meta)} rows"
    )
    # THE SAME MEAN THE TARGET WAS DERIVED UNDER, not `stats/mu.f32` by name. This line used to
    # hardcode the stats mean while `_load_dirs` above resolved the target through the run's `--mu`
    # / the checkpoint's `mu:`, so for every run whose mean is not the stats one -- every
    # `rl-last16` run -- the two sides of `cos_centred_best` were centred on DIFFERENT means and
    # the README said they were the same. On a raw set at `--mu none` it was worse: the target is
    # then `unit(act)` and the product computed a ONE-SIDED cosine, the statistic its own docstring
    # exists to avoid. reconstruction/stats.py:290 reads this array into the paper tables.
    mu_arr = C.load_mu(cfg, base, mu_val, root)
    assert mu_arr is not None, (
        "this run centres on nothing (mu=none), so `cos_centred_best` would be a ONE-SIDED "
        "cosine: the activation moved and the target did not. That is not a centred number and "
        "this product will not write it under that name. Pass --mu <file>, or read "
        "`cos.f16` / `cos_centred.f16` from the scores directory instead."
    )
    mu = mu_arr.astype(np.float32)
    t0 = time.time()

    # ---- centred cosine at the primary's argmax token ------------------------------------------
    argmax = C.read_array(f"{sdir}/argmax.i16", "int16", (n_t, n)).astype(np.int64)
    best_act = np.memmap(f"{sdir}/best_act.f16", dtype=np.float16, mode="r", shape=(n_t, n, d))
    v_sel = dirs[np.asarray(sel, dtype=np.int64)]  # [n_t, d]
    cos_c = np.full((n_t, n), np.nan, dtype=np.float32)
    tchunk = max(1, CHUNK // n)  # targets per chunk: the fp32 transient is tchunk x n x d floats
    for i0 in range(0, n_t, tchunk):
        a = best_act[i0 : i0 + tchunk].astype(np.float32) - mu  # [t, n, d]
        nrm = np.linalg.norm(a, axis=2)
        num = np.einsum("tnd,td->tn", a, v_sel[i0 : i0 + tchunk])
        cos_c[i0 : i0 + tchunk] = num / np.maximum(nrm, 1e-12)
    empty = argmax < 0
    cos_c[empty] = np.nan  # best_act is all-zero there; (0 - mu) would score -cos(mu, v)
    del best_act

    # ---- the 10x-nanmedian norm filter on the stored per-token arrays ---------------------------
    cos = C.read_array(f"{sdir}/cos.f16", "float16", (n_t, n, width)).astype(np.float32)
    nrm = C.read_array(f"{sdir}/norm.f16", "float16", (n_t, n, width)).astype(np.float32)
    keep = ~np.isnan(cos)
    assert np.array_equal(keep, ~np.isnan(nrm)), (
        f"{sdir}: cos.f16 and norm.f16 disagree on which tokens were kept; they come from one "
        f"forward and must have NaN in exactly the same places"
    )
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(nrm, axis=2, keepdims=True)  # [N, n, 1]; NaN for an all-NaN row
        drop = keep & (nrm > NORM_FILTER_MULT * med)
    keep_f = keep & ~drop
    cos_f = np.where(keep_f, cos, -1.0).max(axis=2).astype(np.float32)
    cos_f[~keep_f.any(axis=2)] = -1.0
    cos_p = np.where(keep, cos, -1.0).max(axis=2).astype(np.float32)  # the PRIMARY, recomputed
    n_keep, n_drop = int(keep.sum()), int(drop.sum())
    frac_dropped = n_drop / max(n_keep, 1)
    rows_touched = int((drop.any(axis=2)).sum())
    elapsed = time.time() - t0

    # ---- aggregates, per family ----------------------------------------------------------------
    fam_of = [rows_meta[r]["family"] for r in sel]
    per_fam = {}
    for fam in sorted(set(fam_of)):
        m = np.array([f == fam for f in fam_of])
        with np.errstate(invalid="ignore"):
            per_fam[fam] = {
                "n_targets": int(m.sum()),
                "bo1_primary": round(float(np.nanmean(cos_p[m])), 6),
                "bo1_centred": round(float(np.nanmean(cos_c[m])), 6),
                "bo1_filtered": round(float(np.nanmean(cos_f[m])), 6),
                f"bo{n}_primary": round(float(np.nanmean(cos_p[m].max(axis=1))), 6),
                f"bo{n}_centred": round(float(np.nanmean(np.nanmax(cos_c[m], axis=1))), 6),
                f"bo{n}_filtered": round(float(np.nanmean(cos_f[m].max(axis=1))), 6),
            }
    summary = {
        "scores_dir": sdir,
        "maemm": maemm,
        "base": base,
        "set": set_name,
        "engine": engine,
        "dirs_from": dirs_src,
        "rows": sel,
        "n_targets": n_t,
        "n": n,
        "bo": n,
        # The mean BOTH sides were centred on, as the path it was read from. Named, not implied:
        # reconstruction/stats.py reads this file and a reader must be able to tell two runs apart.
        "mu": C.mu_label(mu_val, base, root),
        "mu_norm": round(float(np.linalg.norm(mu)), 6),
        "norm_filter_mult": NORM_FILTER_MULT,
        "kept_tokens": n_keep,
        "dropped_tokens": n_drop,
        "frac_tokens_dropped": round(frac_dropped, 8),
        "rollouts_touched_by_filter": rows_touched,
        "rollouts_with_no_kept_token": int(empty.sum()),
        "per_family": per_fam,
        "wall_s": round(elapsed, 1),
    }

    inputs = {
        "scores": sdir,
        "maemm": maemm,
        "engine": engine,
        "dirs": dirs_src,
        "mu": summary["mu"],
        "targets": f"{n_t} rows x n={n}",
    }
    with C.outdir(sdir, args, inputs=inputs, keep_existing=True) as od:
        C.note_convention(od, cen_notes)
        od.write_array("cos_centred_best.f16", cos_c, "float16")
        od.write_array("cos_filtered_best.f16", cos_f, "float16")
        od.write_json("centred.json", summary)
        od.note(
            "`centred` (this run) ADDED cos_centred_best.f16, cos_filtered_best.f16 and "
            "centred.json to a scores directory `score` had already written; every other file "
            "below is from the scoring run, whose own header this README replaced. CPU only: no "
            "model is loaded and nothing is re-scored."
        )
        od.note(
            f"`cos_centred_best.f16` [{n_t}, {n}] = cos(best_act[i, k] - mu, v_i) with mu = "
            f"{summary['mu']} (||mu|| = {summary['mu_norm']}), evaluated at the PRIMARY's argmax "
            f"token (argmax.i16); NaN on the {int(empty.sum())} rollouts that kept no token. This "
            f"is the SECONDARY statistic of checklist item 60 -- the primary number stays the "
            f"stored uncentred cos.f16."
        )
        od.note(
            "two mu conventions, stated so the table is readable: a `realact` target is already "
            "unit(X[p] - mu), so the centred variant is the both-sides-centred reading; `sae` and "
            "`random` targets are never centred, so for them subtracting mu from the activation "
            "alone is one-sided and the column is a DIAGNOSTIC, not a competing metric."
        )
        od.note(
            f"`cos_filtered_best.f16` [{n_t}, {n}] = the primary cosine with the "
            f"{NORM_FILTER_MULT:g}x-nanmedian residual-norm filter (eval_universal.py:71,145-147) "
            f"applied per rollout row, max retaken over the survivors, -1.0 where nothing survives. "
            f"MEASURED here: {n_drop} of {n_keep} kept tokens dropped "
            f"({frac_dropped:.3%}), touching {rows_touched} of {n_t * n} rollouts."
        )
        od.note(
            "both arrays are [N, n] and index the SAME (target, rollout) grid as cos.f16's first "
            "two axes, i.e. rows.json's `rows` in order; reconstruction/stats.py reads them as "
            "table (h) beside the primary."
        )
        od.note(f"wall {elapsed:.1f}s for {n_t * n} rollouts (arrays read off the volume, not re-scored)")
    return {
        "out": sdir,
        "targets": n_t,
        "n": n,
        **{k: summary[k] for k in ("frac_tokens_dropped",)},
        "per_family": per_fam,
    }
