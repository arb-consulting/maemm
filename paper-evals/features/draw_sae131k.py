"""Draw the shared 131k-SAE target set: 2,000 held-out features, 80/20.

    python -m features.spawn --product draw_sae131k --base qwen36-27b \
        --sae qwen36-27b/l42-1b --heldout 2026-09-21_sae131k_2k

The sibling of `draw_sae2m` for the OTHER dictionary. Same shape, same 80/20 train/test
column, same `family: "sae"` tag, so every consumer reads it identically and the only
difference between the two sets is which SAE the feature ids belong to. That is the
point: `id` alone is ambiguous -- feature 4242 of `l42-1b` and of `sae2m` are unrelated
directions -- so every row also carries `sae_key`.

## Why this set exists

The 2M SAE is the standard (Ari, 2026-09-20) and all 131k numbers in the draft are
being discarded. But the two dictionaries behave very differently and the contrast is
itself a result: Tomas's 2026-09-21 smoke has autointerp at chance on 2M features with
the CORPUS reference at chance too, while the same pipeline worked on 131k; SAE cosines
sit barely above the random floor on 2M; and our Patchscopes screen put 2M SAE at
0.016-0.021 against a 0.030 floor. Running both dictionaries through one pipeline, on
sets that differ in nothing but the dictionary, is what turns that from an anecdote
into a measurement.

## Held-out provenance, and how it differs from the 2M set

The 2M set draws from Celeste's seed-2026 `eval` split, which is a partition of feature
IDS and is verified against the training banks. The 131k set has no such split file:
what exists is `heldout/pool_heldout/sae.parquet`, the 13,107 features the earlier
chains' training banks excluded. So this set's held-out claim is inherited from that
pool, not re-derived here, and it is weaker in one specific way -- it was established
against the EARLIER chains, and whether a 131k feature has a near-duplicate among the
1.85M 2M features the v2 checkpoint trained on is NOT established. Recorded per row as
`heldout_kind: "feature_id"` with `heldout_note` saying exactly that, so nobody reads
this set's provenance as equal to the 2M set's.

Strata are quartiles of log10 peak activation over the drawn set, recorded rather than
sampled, as in `draw_sae2m`.

## The decoder twin of an existing set (`--sides dec --dirs-from <set> --rows <spec>`)

    modal run precompute/modal_app.py --product draw_sae131k --base qwen36-27b \
        --sae qwen36-27b/l42-1b --set 2026-09-24_v3_ctrl_dec --sides dec \
        --dirs-from /vol/base/qwen36-27b/heldout/2026-09-21_v3_ctrl --rows 512-1023

NOT a draw. It takes the feature ids of `--rows` of an existing set, in that order, and writes
one row per feature whose vector is `unit(W_dec[f])` -- the decoder direction of the same
feature -- so the encoder and decoder runs pair feature for feature with no re-sampling. Every
copied row must be an ENCODER row of `--sae` (`common.sae_rows_of(..., side="enc")`), and the
source's own vectors are checked against `unit(W_enc[:, f])` re-read from the checkpoint before
anything is written, so a row range that is not this dictionary's encoder columns refuses.

`storage: raw` (not `dirs_only` as the draw above): `act.f32` holds the unit decoder rows and
`vecs.f16` is the same thing in fp16. The family is `sae`, which `family_kinds:` calls
non-centrable, so `common.dirs_for` returns `unit(act)` at every `--mu` -- the injection is the
raw unit decoder direction, exactly as the encoder rows of the source are injected raw -- and
`scan --centre`, which requires `storage: raw` on every bank, can read it. Each row carries
`sae_side: "dec"` (what `common.sae_rows_of`, `sae_self --sae-side dec` and `build --sae-side
dec` select on) and `vector: "dec"`, plus `ids_from_set` / `ids_from_row` naming the source row.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

from .draw_sae2m import _columns, _finish

N_FEATURES = 2_000
FIT_FRACTION = 0.8
DRAW_SEED = 20260921
POOL = "heldout/pool_heldout/sae.parquet"
HELDOUT_NOTE = (
    "held out by the EARLIER chains' feature split (pool_heldout/sae.parquet, 13,107 "
    "features), not by Celeste's seed-2026 2M partition. Whether a 131k feature has a "
    "near-duplicate among the 1.85M 2M features the v2 checkpoint trained on is NOT "
    "established."
)


def build(cfg, args):
    import pandas as pd

    base, root = args["base"], args["root"]
    # ONE --sae syntax in the whole CLI (fd502c1). The inline form this file was written with
    # accepted a bare name where every other product refuses one, and produced the literal string
    # "qwen36-27b/None" rather than asserting when --sae was omitted.
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    spec = cfg["bases"][base]
    set_name = args["heldout"] or time.strftime("%Y-%m-%d") + "_sae131k"
    out_dir = C.heldout_dir(base, set_name, root)
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    pool_path = args.get("subset") or f"{root}/data/celeste-v2-2026-09-17/{POOL}"
    pool = pd.read_parquet(pool_path, columns=["feature", "act"])
    ids = pool["feature"].to_numpy()
    acts = pool["act"].to_numpy(dtype=np.float64)
    assert len(np.unique(ids)) == len(ids), "the held-out pool repeats a feature"
    print(f"[draw131k] pool {len(ids):,} held-out features, act "
          f"min {acts.min():.2f} median {np.median(acts):.2f} max {acts.max():.1f}",
          flush=True)
    # `--n`, as draw_sae2m has always taken it (draw_sae2m.py:243). Hardcoding 2000 meant a
    # rare-feature mining draw could not be made with this product at all; the DEFAULT is
    # unchanged, so `2026-09-21_sae131k_2k` still reproduces byte-for-byte.
    n_feat = int(args.get("n") or N_FEATURES)
    assert n_feat > 0, f"--n must be positive, got {n_feat}"
    assert len(ids) >= n_feat, f"only {len(ids)} features, need {n_feat}"

    rng = np.random.default_rng(int(args.get("seed") or DRAW_SEED))
    pick = np.sort(rng.choice(len(ids), size=n_feat, replace=False))
    drawn, peak = ids[pick], acts[pick]

    cuts = np.quantile(np.log10(np.maximum(peak, 1e-6)), [0.25, 0.5, 0.75])
    stratum = np.searchsorted(cuts, np.log10(np.maximum(peak, 1e-6)), side="right")
    side = np.where(rng.random(n_feat) < FIT_FRACTION, "train", "test")

    peak_by_id = dict(zip(drawn.tolist(), peak.tolist(), strict=True))
    set_name, out_dir, rows, vecs, meta = _finish(
        cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
        # NAME THE POOL THAT WAS ACTUALLY READ. This was the hardcoded literal
        # "pool_heldout/sae.parquet", so a `--subset` draw wrote a README claiming a source it
        # never opened -- the one field whose whole job is to say where the features came from.
        peak_by_id, "log10_pool_peak_act", f"{pool_path} ({len(ids):,})",
        # `peak16` (the 16M-corpus peak) is None here: this draw reads the 1B pool's `act`, and
        # `corpus_peak_16m` is a different quantity on a different corpus. The slot was added to
        # `_finish` on this branch AFTER draw_sae131k was written against the 16-argument
        # signature, which bound `cuts` to `peak16` and raised a TypeError on the meta dict.
        # `sides` is POPPED by `_finish` (draw_sae2m.py:461) and supplied there at :386;
        # draw_sae131k never passed it, so every call raised KeyError before reaching the
        # GPU. This draw is encoder-only -- its rows are unit(W_enc[:, f]) -- so it is the
        # single-side shape, which is what the set on the volume already carries.
        False, None, None, cuts,
        {"pool": len(ids), "eligible": int(len(ids)), "sides": ("enc",),
         "pool_path": pool_path})
    for r in rows:
        # NOT `r["sae_key"] = "l42-1b"`, which is what this loop used to do. `_finish` already
        # stamps the FULL config key, and `common.sae_rows_of` matches on the full key -- a bare
        # name matches nothing, and because the row IS keyed (just wrongly) the unkeyed branch's
        # loud assert never fires: `scan` would simply run with n_feat = 0.
        r["heldout_note"] = (
            f"drawn from --subset {pool_path}; no training-split check has been made"
            if args.get("subset") else HELDOUT_NOTE)
    return set_name, out_dir, rows, vecs, meta


def _paired(args) -> bool:
    """True for the decoder-twin mode, refusing every half-specified form of it.

    `--sides` defaults to `enc` and, without `--dirs-from`, reproduces the 2k draw byte for byte.
    `dec` needs a source set and a row range; `enc` WITH a source is a copy of encoder rows,
    which is `heldout_v3 --block ctrl`'s job, and `enc,dec` would duplicate the source's own
    encoder rows under a second name.
    """
    sides = str(args.get("sides") or "enc").strip()
    src = (args.get("dirs_from") or "").strip()
    if sides == "enc" and not src:
        return False
    assert sides == "dec", (
        f"draw_sae131k --sides {sides!r}: the 131k draw is encoder-only, and the one other form "
        f"is `--sides dec --dirs-from <set dir> --rows <spec>`, the decoder twin of an existing "
        f"set's encoder rows. A copy of encoder rows is `heldout_v3 --block ctrl`.")
    assert src and (args.get("rows") or "").strip(), (
        "draw_sae131k --sides dec copies the FEATURE IDS of an existing set: pass --dirs-from <that "
        "set's directory> and --rows <the range of its encoder rows>. It never draws, because the "
        "point is to pair row for row with the encoder run.")
    return True


def build_paired(cfg, args):
    """(set_name, out_dir, rows, act [n, d] fp32 unit, meta) -- `unit(W_dec[f])` for the source's f.

    The source's rows are copied field by field (stratum, density, max_act, ...) with `row`
    renumbered from 0, `sae_side`/`vector` set to `dec`, and the source named in `ids_from_set` /
    `ids_from_row`. The source's own `src_set`/`src_row` are dropped: they state where the ENCODER
    vector's bytes came from, and these rows carry different bytes.
    """
    base, root = args["base"], args["root"]
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    d = int(cfg["bases"][base]["d"])
    set_name = args["heldout"]
    assert set_name, "draw_sae131k --sides dec writes a held-out set and needs an explicit --set (D6)"
    out_dir = C.heldout_dir(base, set_name, root)
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    src = args["dirs_from"].rstrip("/")
    src_name = os.path.basename(src)
    assert src_name != set_name, f"--dirs-from {src} is the set being written"
    src_rows = C.read_jsonl(f"{src}/ids.jsonl")
    want = C.parse_rows(args["rows"], len(src_rows))
    sel = [src_rows[i] for i in want]
    # EVERY copied row must be an encoder row of THIS dictionary. `sae_rows_of` is the one rule
    # (per-row sae_key, or the set's declared one for unkeyed rows); a range that strays into the
    # `random` block or onto another dictionary's ids refuses here, by row.
    enc = C.sae_rows_of(sel, sae_key, side="enc", declared=C.declared_sae_key(cfg, src, root),
                        where=src)
    stray = sorted({r["row"] for r in sel} - {r["row"] for r in enc})
    assert not stray, (
        f"--rows {args['rows']} of {src_name} include rows {stray[:8]} that are not encoder rows "
        f"of {sae_key!r} (families {sorted({src_rows[i]['family'] for i in stray})}); the decoder "
        f"twin is defined only for encoder rows of the dictionary it reads")
    drawn = np.asarray([int(r["id"]) for r in sel], dtype=np.int64)
    assert len(np.unique(drawn)) == len(drawn), (
        f"--rows {args['rows']} of {src_name} repeat a feature id; the twin set would carry one "
        f"feature twice and every per-feature join downstream would be ambiguous")

    cols, gate, d_sae = _columns(C.sae_path(cfg, sae_key), d, drawn, ("enc", "dec"))
    col_enc = cols["enc"].numpy().astype(np.float64)
    col_dec = cols["dec"].numpy().astype(np.float32)

    # THE SOURCE IS WHAT IT SAYS IT IS: its stored vectors must be this dictionary's encoder
    # columns for exactly these ids. act.f32 when the source is raw (fp32, so ~1e-7 off), else
    # vecs.f16 (~1e-3 off after the fp16 round trip).
    if os.path.exists(f"{src}/act.f32"):
        sv = C.read_array(f"{src}/act.f32", "float32", (len(src_rows), d))[np.asarray(want)]
        src_arr = "act.f32"
    else:
        sv = C.read_array(f"{src}/vecs.f16", "float16", (len(src_rows), d))[np.asarray(want)]
        src_arr = "vecs.f16"
    sv = sv.astype(np.float64)
    sv /= np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-12)
    cos_src = (sv * col_enc).sum(1)
    assert float(cos_src.min()) > 0.999, (
        f"{src_name} rows {want[int(np.argmin(cos_src))]}.. are NOT unit(W_enc[:, f]) of "
        f"{sae_key} for their own ids: min cos {float(cos_src.min()):.6f} over {len(want)} rows "
        f"({src_arr}). The feature ids and the stored vectors disagree, so a decoder twin keyed "
        f"on those ids would pair against something else.")
    cos_ed = (col_enc * col_dec.astype(np.float64)).sum(1)
    norms = np.linalg.norm(col_dec.astype(np.float64), axis=1)
    assert float(np.abs(norms - 1).max()) < 1e-5, f"decoder rows not unit: {norms.min()}..{norms.max()}"

    rows = []
    for j, (i, r) in enumerate(zip(want, sel, strict=True)):
        keep = {k: v for k, v in r.items() if k not in ("row", "src_set", "src_row")}
        rows.append({
            "row": j,
            **keep,
            "family": "sae",
            "sae_key": sae_key,
            # `sae_side` is the SELECTOR (common.sae_rows_of, sae_self/build --sae-side);
            # `vector` says the same thing in the words of the eval plan.
            "sae_side": "dec",
            "vector": "dec",
            "ids_from_set": src_name,
            "ids_from_row": int(i),
        })
    meta = {
        "sae_key": sae_key,
        "sides": ["dec"],
        "d_sae": int(d_sae),
        "gate": gate,
        "ids_from": src,
        "ids_from_rows": args["rows"],
        "n": len(rows),
        "src_vectors_checked": src_arr,
        "src_vs_unit_enc_min_cos": round(float(cos_src.min()), 7),
        "cos_enc_dec": {q: round(float(np.quantile(cos_ed, p)), 4)
                        for q, p in (("min", 0.0), ("q25", 0.25), ("median", 0.5),
                                     ("q75", 0.75), ("max", 1.0))},
    }
    return set_name, out_dir, rows, col_dec, meta


def run_paired(cfg, args):
    set_name, out_dir, rows, act, meta = build_paired(cfg, args)
    inputs = {"sae": args.get("sae"), "ids_from": meta["ids_from"], "rows": meta["ids_from_rows"]}
    with C.outdir(out_dir, args, inputs=inputs) as od:
        od.write_jsonl("ids.jsonl", rows)
        # RAW: act.f32 IS the unit decoder row and vecs.f16 is unit(act) -- the contract
        # `common.dirs_for` reads. Non-centrable family, so no --mu ever moves either.
        od.write_array("act.f32", act, "float32")
        od.write_array("vecs.f16", act, "float16")
        od.write_json(
            "storage.json",
            {
                "storage": "raw",
                "mu_stored": None,
                "family_mu": {},
                "families": {"sae": cfg["family_kinds"]["sae"]["kind"]},
                "sae_key": meta["sae_key"],
                "sae_sides": meta["sides"],
                "ids_from": {"set": os.path.basename(meta["ids_from"]),
                             "rows": meta["ids_from_rows"]},
                "note": (
                    "RAW STORAGE of unit dictionary rows -- unit(W_dec[f]) of the 131k `l42-1b` "
                    "SAE for the feature ids of " + os.path.basename(meta["ids_from"]) + " rows "
                    + meta["ids_from_rows"] + ", in that order. act.f32 is the unit decoder row "
                    "and vecs.f16 is unit(act); family `sae` is not centrable (config.yaml "
                    "family_kinds), so every --mu is a no-op on them"
                ),
            },
        )
        od.note(
            f"DECODER TWIN, not a draw: {len(rows)} rows, row j = unit(W_dec[f]) for the feature f "
            f"of `{os.path.basename(meta['ids_from'])}` row {rows[0]['ids_from_row']} + j "
            f"(`--rows {meta['ids_from_rows']}`), same order, same per-row fields. `sae_side: dec` "
            f"and `vector: dec` on every row; `ids_from_set` / `ids_from_row` name the source row.")
        od.note(
            f"source checked before writing: its {meta['src_vectors_checked']} rows are "
            f"unit(W_enc[:, f]) of {meta['sae_key']} for their own ids, min cos "
            f"{meta['src_vs_unit_enc_min_cos']}. cos(unit enc, unit dec) per feature: "
            f"{meta['cos_enc_dec']}.")
        od.note(f"gate {meta['gate']}; F = {meta['d_sae']:,}. The ACTIVATION of feature f is its "
                f"ENCODER readout whichever direction was injected; `sae_self --sae-side dec` "
                f"reads these rows' stored vectors and skips the encoder-column cross-check.")
        od.note("HELD-OUT PROVENANCE is the source set's; nothing about it changes here.")
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_sae131k", "set": set_name, **meta}


def run(cfg, args):
    if _paired(args):
        return run_paired(cfg, args)
    set_name, out_dir, rows, vecs, meta = build(cfg, args)
    with C.outdir(out_dir, args, inputs={"sae": args.get("sae"), "pool": meta["pool_path"]}) as od:
        od.write_jsonl("ids.jsonl", rows)
        od.write_array("vecs.f16", vecs, "float16")
        # THE STORAGE CONTRACT (H4), the same one `draw_sae2m.run` writes. Without it
        # `common.set_storage` refuses the set outright and somebody has to hand-write a
        # `heldout:` entry -- which is exactly what `2026-09-21_sae131k_2k` needed on the way in.
        # `dirs_only`: these rows are unit encoder columns, never centred and not centrable, so no
        # `--mu` applies to them at all.
        od.write_json(
            "storage.json",
            {
                "storage": "dirs_only",
                "mu_stored": None,
                "family_mu": {},
                "families": {"sae": "dictionary"},
                "sae_key": meta["sae_key"],
                "sae_sides": meta["sides"],
                "note": (
                    "unit dictionary columns -- unit(W_enc[:, f]) of the 131k `l42-1b` SAE: "
                    "never centred, not centrable (config.yaml family_kinds), so every --mu is "
                    "a no-op on them"
                ),
            },
        )
        od.note(f"{len(rows)} features of the 131k SAE, family tag 'sae', sae_key "
                f"{rows[0]['sae_key']!r} -- feature ids are NOT comparable with sae2m's")
        # "BOTH halves are unseen" is a claim about the DEFAULT pool, and the seed is no longer
        # always DRAW_SEED now that --seed exists. Report what this run did, not what the 2k
        # draw did.
        seed_used = int(args.get("seed") or DRAW_SEED)
        od.note(f"{meta['n_fit']} train / {meta['n_report']} test, seed {seed_used}; "
                + ("the split is OURS, for fitting vs reporting; it says nothing about what the "
                   "checkpoint saw (see the provenance note below)"
                   if args.get("subset") else
                   "BOTH halves are unseen -- this splits our analysis, not the training"))
        od.note(f"strata: {meta['stratum_stat']} quartiles over the drawn set, recorded "
                f"not sampled, cuts {meta['cuts']}")
        # HELDOUT_NOTE describes the DEFAULT pool (the earlier chains' 13,107-feature split).
        # A --subset draw reads some other pool, whose rows were NOT held out by that split and
        # may sit in the checkpoint's training set -- so stamping the note there asserts a
        # cleanliness this set has not got. Say what is true instead, and say it loudly.
        if args.get("subset"):
            od.note("HELD-OUT PROVENANCE: NONE ESTABLISHED. Drawn from the --subset pool "
                    f"{meta['pool_path']!r}, not from {POOL}. Nothing here has checked these "
                    "features against any training split: treat them as POSSIBLY SEEN by the "
                    "checkpoint. Fine for a set that is trained ON; NOT usable as an evaluation "
                    "set without a leakage scan first.")
        else:
            od.note("HELD-OUT PROVENANCE IS WEAKER THAN THE 2M SET'S: " + HELDOUT_NOTE)
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_sae131k", "set": set_name, **meta}
