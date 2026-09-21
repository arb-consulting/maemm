"""The eval-1 (faithfulness) target blocks: Celeste's v3 families, and our own controls.

    modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b \
        --block realact --set 2026-09-21_v3_realact

One BLOCK per set directory, because a set directory carries ONE storage contract
(`common.set_storage` returns a single `storage` kind and `common.dirs_for` branches on it once
for all N rows). The blocks of eval 1 do not share one: her realact rows become RAW (see below),
her `realact_long` rows are unit directions centred on a mean nobody holds a file for, her `bsf`
and `jlens` rows are subspace bases that were never centred at all, and the controls are copied
raw. Four contracts, four directories, one config entry group named `2026-09-21_v3_*`.

    --block realact       her 512 realact rows, RECOVERED TO RAW (below)     storage: raw
    --block realact_long  her 512 realact_long rows, as shipped              storage: unit
    --block subspace      her 512 bsf + 512 jlens rows, as shipped           storage: dirs_only
    --block ctrl          rows of an existing set, copied with provenance    storage: raw

The 2M dictionary block is NOT here: it is a draw, and `features/draw_sae2m.py --sides enc,dec`
is the tool that makes it.

WHY HER REALACT ROWS CAN BE RAW (U1, settled 2026-09-21, $0)
------------------------------------------------------------
Her rows ship `direction` = `unit(act - whiten_mu)` and a scalar `pool_act_norm`, and the whole
recovery turns on which norm that scalar is. `‖act‖` makes `act` recoverable by the exact solve
`rollouts_nla.build_inputs` already implements; `‖act - whiten_mu‖` would make it recoverable
even more directly, but would also mean every distribution below is a different quantity. It is
`‖act‖`, on three independent readings:

  * her own mint statistic. `heldout/pool_heldout/build_stats.json` records
    `family_stats.realact.median_norm = 91.2669` over the 200,000-row pool these 512 are drawn
    from; the 512 `pool_act_norm` values have median 90.48.
  * our corpus. `stats`'s layer-42 block-output residual-norm quantiles over 949,557 sampled
    positions of our 16M corpus are q05 76.38 / q50 93.26 / q95 109.26. Her `pool_act_norm` is
    q05 75.21 / q50 90.48 / q95 105.67 -- the same distribution to ~3%. Under the other reading
    the implied `‖act‖` would be q05 87.91 / q50 112.34 / q95 134.54, i.e. her MEDIAN activation
    would sit above the 95th percentile of ours.
  * the geometry. `‖whiten_mu‖` = 67.2647 and `cos(direction, whiten_mu)` has mean -0.0198 over
    the 512 (so `act - mu` is, as it must be, near-orthogonal to `mu`). A mu-orthogonal residual
    at `‖act‖` = 90.48 has `‖act - mu‖` = sqrt(90.48^2 - 67.26^2) = 60.52; the exact solve
    returns a median `t` of 59.95.

So `act = mu + t*u` with `t > 0` solving `‖mu + t*u‖ = pool_act_norm`, which is
`rollouts_nla.build_inputs(amp="exact")`'s root, taken here through the same helper so the two
cannot drift. On these 512 the discriminant is non-negative and `t > 0` on all 512 rows, the
round trip reproduces `pool_act_norm` to 2.8e-14 and her own `direction` to `cos = 1.0000000000`.

THREE ROWS ARE AMBIGUOUS, and are flagged rather than resolved: 26, 32 and 360 have
`pool_act_norm < ‖mu‖` AND `mu.u < 0`, so both roots are positive and two different raw
activations satisfy the constraint. The larger root is taken (`build_inputs`'s rule). It does
NOT affect anything read at her own mean -- `unit(act - whiten_mu)` is her `direction` for
either root, exactly -- but a number read off `act.f32` itself, or at any OTHER mean, is
uncertain on those three rows. `exact_ambiguous: true` is on the row and the ids are in the
README.

EXCLUSIONS are recorded, never applied here. The set keeps all 512 rows in her order so every
arm pairs row for row (plan §2.1); `exclusions.json` and the README carry the list and the
headline n, and the tables drop them.
"""
from __future__ import annotations

import json
import os

import numpy as np

import precompute.common as C

# `activations`, `bundle` and `layout` import pandas, which the CPU unit smoke's environment does
# not carry (it pins torch/numpy/transformers only). They are imported where they are used so
# `recover_raw` -- the one piece of this module with a CPU selftest -- stays importable there.

BLOCKS = ("realact", "realact_long", "subspace", "ctrl")

# Her 7-gram coverage exclusion for the realact block, read off the volume rather than retyped:
# `features/ngram_overlap.py --side hers --n 7` writes it, and the threshold is its own.
EXCLUDE_JSON = "/vol/shared/ngram-overlap/hers_n7.exclude.json"
EXCLUDE_SOURCE = "heldout/eval_directions_v3/realact.parquet"

SUBSPACE_FAMILIES = ("bsf", "jlens")


# --------------------------------------------------------------------------- the recovery


def recover_raw(u: np.ndarray, act_norm: np.ndarray, mu: np.ndarray):
    """(act [n, d] f32, per-row info) for rows stored as `unit(act - mu)` plus `‖act‖`.

    The solve is `rollouts_nla.build_inputs(..., amp="exact")`, called rather than re-derived:
    it owns the root choice, the discriminant fallback and the ambiguity flag, and a second
    implementation of a quadratic is a second place for the sign to be wrong.
    """
    from precompute import rollouts_nla

    u = np.ascontiguousarray(u, dtype=np.float32)
    meta = [{"family": "realact", "act_norm": (None if not np.isfinite(a) else float(a))}
            for a in act_norm]
    x, info = rollouts_nla.build_inputs(u, meta, mu.astype(np.float32), "exact", 1.0)
    fell_back = [i for i, r in enumerate(info) if r["fallback"]]
    assert not fell_back, (
        f"the exact solve fell back on {len(fell_back)} of {len(info)} rows "
        f"({[(i, info[i]['fallback']) for i in fell_back[:8]]}): those rows carry no raw "
        f"activation norm, or none is reachable on the line mu + R*u, so they cannot be "
        f"recovered to raw. Import the block as shipped (`storage: unit`) instead."
    )
    # The two identities that make this a recovery rather than a construction.
    got = np.linalg.norm(x.astype(np.float64), axis=1)
    d_norm = float(np.abs(got - act_norm.astype(np.float64)).max())
    assert d_norm < 1e-3, f"‖act‖ != the stored norm: max |d| {d_norm:.3e}"
    back = x.astype(np.float64) - mu[None, :].astype(np.float64)
    back /= np.maximum(np.linalg.norm(back, axis=1), 1e-12)[:, None]
    cos_back = float((back * u.astype(np.float64)).sum(1).min())
    assert cos_back > 1 - 1e-6, (
        f"unit(act - mu) does not reproduce the shipped direction: min cos {cos_back:.10f}")
    return x, info, {"max_abs_norm_error": d_norm, "min_cos_to_shipped": cos_back}


def _exclusions(od_notes: list[str]) -> dict:
    """Her realact block's exclusion list, read off the volume; `{}` with a note when absent."""
    if not os.path.exists(EXCLUDE_JSON):
        od_notes.append(
            f"NO EXCLUSION LIST: {EXCLUDE_JSON} is not on the volume, so this block is frozen "
            f"with none. Run `modal run features/ngram_overlap.py --n 7 --side hers` and record "
            f"the result before any score is read."
        )
        return {}
    rec = json.loads(open(EXCLUDE_JSON).read())
    assert rec.get("source") == EXCLUDE_SOURCE, (
        f"{EXCLUDE_JSON} was computed over {rec.get('source')!r}, not {EXCLUDE_SOURCE!r}; its row "
        f"indices would not be indices into this block"
    )
    return rec


# --------------------------------------------------------------------------- the four blocks


def _hers(cfg, args, family: str):
    """Her frozen directions for one family, plus whatever provenance the bundle ships."""
    from . import bundle, layout

    t = bundle.load_family(family)
    kind = layout.heldout_kind(family)          # refuses an undeclared family
    assert kind != "in_distribution", f"{family!r} is in_distribution -- not held out"
    C.family_centrable(cfg, family)             # refuses a family with no family_kinds entry
    return t, kind


def _provenance_rows(family: str, t, kind: str) -> list[dict]:
    """One row dict per target, with the bundle's own doc/pos/norm where it ships them."""
    import pandas as pd

    from . import activations

    reg = activations.build()
    sub = reg[reg["family"] == family].reset_index(drop=True)
    rows = []
    for i in range(len(t)):
        row = {
            "row": i,
            "family": family,
            # Her own identifier for the target, not its position in our set: a renumbering of
            # this block must not silently rename its targets.
            "id": (f"doc{int(t.meta['pool_seq'].iloc[i])}:pos{int(t.meta['pool_pos'].iloc[i])}"
                   if "pool_seq" in t.meta.columns else int(i)),
            "source": "hers",
            "stratum": None,
            "heldout_kind": kind,
            "family_row": i,
        }
        if len(sub) == len(t):
            cell = sub.iloc[i]
            if pd.notna(cell.get("norm_stratum")):
                row["stratum"] = int(cell["norm_stratum"])
            for col in ("doc", "pos", "act_norm", "doc_n_targets"):
                if pd.notna(cell.get(col)):
                    row[col] = float(cell[col]) if col == "act_norm" else int(cell[col])
        if "pool_target_text" in t.meta.columns:
            row["span_text"] = str(t.meta["pool_target_text"].iloc[i])
        rows.append(row)
    return rows


def _block_realact(cfg, args, notes: list[str]):
    base = args["base"]
    t, kind = _hers(cfg, args, "realact")
    rows = _provenance_rows("realact", t, kind)
    assert all("act_norm" in r for r in rows), (
        "a realact row carries no act_norm, so it cannot be recovered to raw")

    mu_path = C.resolve_mu_path(cfg["bases"][base]["whiten_mu"], base, args["root"])
    mu = C.load_mu(cfg, base, cfg["bases"][base]["whiten_mu"], args["root"])
    act, info, ident = recover_raw(
        t.directions, np.array([r["act_norm"] for r in rows], dtype=np.float64), mu)

    exc = _exclusions(notes)
    excluded = set(exc.get("excluded_rows", []))
    cov = {int(k): v for k, v in (exc.get("coverage") or {}).items()}
    amb = []
    for i, r in enumerate(rows):
        r["exact_ambiguous"] = bool(info[i]["exact_ambiguous"])
        r["act_norm_recovered"] = float(info[i]["in_norm"])
        r["t_solved"] = float(info[i]["r"])
        r["excluded"] = i in excluded
        if i in excluded:
            r["exclude_reason"] = f"ngram_overlap n=7 coverage {cov.get(i)} >= {exc['threshold']}"
        if r["exact_ambiguous"]:
            amb.append(i)

    n_head = len(rows) - len(excluded)
    extra = {
        "exclusions.json": {
            "block": "realact",
            "rows_total": len(rows),
            "excluded_rows": sorted(excluded),
            "n_headline": n_head,
            "criterion": exc.get("threshold"),
            "n_gram": exc.get("n"),
            "source": EXCLUDE_JSON,
            "computed_over": exc.get("source"),
            "coverage": exc.get("coverage"),
            "note": ("rows are KEPT in her order so every arm pairs; the tables drop these. "
                     "The fully-covered rows (coverage 1.0) are a subset of this list."),
        },
        "recovery.json": {
            "question": "is her pool_act_norm ||act|| or ||act - whiten_mu||?",
            "answer": "||act||",
            "solve": "rollouts_nla.build_inputs(amp='exact'): t>0 with ||mu + t u|| = act_norm",
            "mu": mu_path,
            "mu_norm": round(float(np.linalg.norm(mu)), 6),
            "rows": len(rows),
            "fallbacks": 0,
            "exact_ambiguous_rows": amb,
            **{k: (round(v, 12) if isinstance(v, float) else v) for k, v in ident.items()},
        },
    }
    notes.append(
        f"HER 512 realact rows, RECOVERED TO RAW. `pool_act_norm` is `||act||` (recovery.json "
        f"records the three readings that settle it), so `act = mu + t*u` with t the positive "
        f"root of `||mu + t*u|| = pool_act_norm` under mu = {mu_path}. 0 fallbacks; ||act|| "
        f"reproduced to {ident['max_abs_norm_error']:.2e} and her own `direction` to "
        f"cos = {ident['min_cos_to_shipped']:.10f}. `vecs.f16` here is therefore `unit(act)`, "
        f"UNCENTRED -- her shipped direction is what `dirs_for(..., mu={mu_path})` returns."
    )
    if amb:
        notes.append(
            f"{len(amb)} AMBIGUOUS rows {amb}: `pool_act_norm < ||mu||` and `mu.u < 0`, so both "
            f"roots are positive and two different raw activations meet the constraint. The "
            f"larger is taken (build_inputs' rule). Anything read at her own mean is unaffected "
            f"(both roots give her `direction` exactly); `act.f32` itself, and any OTHER mean, "
            f"are uncertain on these three."
        )
    notes.append(
        f"EXCLUSIONS frozen here, applied downstream: {len(excluded)} of {len(rows)} rows, "
        f"headline n = {n_head}. See exclusions.json."
    )
    return rows, act, C.storage_record(cfg, args["heldout"], ["realact"], ""), extra


def _block_shipped(cfg, args, families, storage: str, notes: list[str]):
    """Her rows imported EXACTLY as shipped -- unit directions, no act.f32, no re-centring."""
    rows, vecs, fam_mu = [], [], {}
    for family in families:
        t, kind = _hers(cfg, args, family)
        block = _provenance_rows(family, t, kind)
        for r in block:
            r["row"] = len(rows)
            rows.append(r)
        vecs.append(t.directions)
        centrable = C.family_centrable(cfg, family)
        fam_mu[family] = C.MU_UNKNOWN if centrable else None
        notes.append(
            f"`{family}`: {len(block)} rows as shipped, "
            + ("centred on a mean this repo holds NO FILE FOR (`unknown`). For `realact_long` "
               "that mean is `mu_long` -- the mean over ALL collected long-context activations, "
               "computed on the fly in `eval/build_ctx_eval.py:47-54` from `MAEMM_ACTS_LONG` "
               "(`/root/pmx/bsf27b/acts_long`) and never written to a file. It is NOT "
               "`whiten_mu`: over these rows `cos(direction, whiten_mu)` has mean -0.0618 and "
               "||mean(direction)|| is 0.1216, against -0.0198 / 0.0615 on `realact`. The rows "
               "are returned AS SHIPPED and every number read off them is labelled."
               if centrable else
               "never centred and not centrable (config.yaml `family_kinds`), so no `--mu` "
               "applies to it.")
        )
    contract = {
        "storage": storage,
        "mu_stored": None,
        "family_mu": fam_mu if storage == "unit" else {},
        "families": {f: cfg["family_kinds"][f]["kind"] for f in families},
        "note": ("imported from Celeste's bundle: stored unit directions, no act.f32, so they "
                 "cannot be moved to another mean. `unknown` means the mean they carry is not "
                 "one this repo holds a file for."),
    }
    return rows, np.concatenate(vecs, axis=0), contract, {}


def _block_ctrl(cfg, args, notes: list[str]):
    """A row range of an EXISTING set, copied with its provenance. Nothing is re-drawn."""
    src = args.get("dirs_from") or ""
    assert src, (
        "--block ctrl copies rows out of an existing set: pass --dirs-from <that set's "
        "directory> and --rows <range>. It never re-draws, because a re-draw is a different "
        "sample under the same name.")
    src_name = os.path.basename(src.rstrip("/"))
    contract = C.set_storage(cfg, src, args["root"])
    assert contract["storage"] == "raw", (
        f"{src} is `storage: {contract['storage']}` ({contract['source']}); --block ctrl copies "
        f"the RAW contract forward, and a stored unit direction has no act.f32 to copy.")

    src_rows = [json.loads(ln) for ln in open(f"{src}/ids.jsonl")]
    d = int(cfg["bases"][args["base"]]["d"])
    act = C.read_array(f"{src}/act.f32", "float32", (len(src_rows), d))
    want = C.parse_rows(args.get("rows") or "", len(src_rows))
    assert want, "--block ctrl needs --rows: copying a whole set is a rename, not a block"

    rows = []
    for j, i in enumerate(want):
        r = dict(src_rows[i])
        r["row"] = j
        r["source"] = "ours"
        r["src_set"] = src_name
        r["src_row"] = int(i)
        rows.append(r)
    fams = sorted({r["family"] for r in rows})
    for f in fams:
        assert not C.family_centrable(cfg, f), (
            f"family {f!r} is centrable, and a copied block carries no statement about which "
            f"mean its act.f32 was measured under beyond the source set's. Copy non-centrable "
            f"families only, or re-derive.")
    # WHICH dictionary the copied `sae` rows' feature ids index. Carried forward explicitly:
    # `common.sae_rows_of` REFUSES an unkeyed row against an undeclared dictionary (H1), and
    # every one of these 131k ids is also a valid index into the 2M encoder.
    sae_key = (json.loads(open(f"{src}/storage.json").read()).get("sae_key", "")
               if os.path.exists(f"{src}/storage.json") else "")
    sae_key = sae_key or cfg["heldout"].get(src_name, {}).get("sae_key", "")
    assert sae_key or not any(r["family"] in C.SAE_FAMILIES for r in rows), (
        f"the copied rows include SAE rows but neither {src}/storage.json nor the `heldout:` "
        f"entry for {src_name!r} declares a `sae_key`; a feature id means nothing without one")
    notes.append(
        f"COPIED, not drawn: rows {args['rows']} of `{src_name}` ({src}), {len(rows)} rows, "
        f"families {fams}. Each row keeps its own fields and gains `src_set`/`src_row`, so the "
        f"provenance survives the renumbering. `act.f32` and `vecs.f16` are the source's own "
        f"bytes for those rows -- this is a copy, and the source is not modified."
    )
    return rows, act[np.asarray(want)], {
        "storage": "raw",
        "mu_stored": None,
        "family_mu": {},
        "families": {f: cfg["family_kinds"][f]["kind"] for f in fams},
        "sae_key": sae_key,
        "note": (f"RAW STORAGE, copied from {src_name} rows {args['rows']}: act.f32 holds each "
                 f"row's own vector before any mean was subtracted, and vecs.f16 is unit(act). "
                 f"Every family here is non-centrable, so the two are the same direction."),
    }, {}


# --------------------------------------------------------------------------- the product


def build(cfg, args):
    base, root = args["base"], args["root"]
    block = args.get("block") or ""
    assert block in BLOCKS, f"--block {block!r}: want one of {list(BLOCKS)}"
    set_name = args["heldout"]
    assert set_name, "heldout_v3 writes a held-out set, so it needs an explicit --set <name> (D6)"
    out_dir = C.heldout_dir(base, set_name, root)
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    notes: list[str] = []
    if block == "realact":
        rows, arr, contract, extra = _block_realact(cfg, args, notes)
        storage = "raw"
    elif block == "realact_long":
        rows, arr, contract, extra = _block_shipped(cfg, args, ["realact_long"], "unit", notes)
        storage = "unit"
    elif block == "subspace":
        rows, arr, contract, extra = _block_shipped(
            cfg, args, list(SUBSPACE_FAMILIES), "dirs_only", notes)
        storage = "dirs_only"
    else:
        rows, arr, contract, extra = _block_ctrl(cfg, args, notes)
        storage = "raw"

    assert [r["row"] for r in rows] == list(range(len(rows))), "rows are not 0..N-1"
    assert len(arr) == len(rows), f"{len(arr)} vectors for {len(rows)} rows"
    assert contract["storage"] == storage, contract
    return set_name, out_dir, block, storage, rows, np.asarray(arr, dtype=np.float32), \
        contract, notes, extra


def run(cfg, args):
    from . import bundle

    set_name, out_dir, block, storage, rows, arr, contract, notes, extra = build(cfg, args)
    n = np.linalg.norm(arr.astype(np.float64), axis=1)
    if storage == "raw":
        # vecs.f16 IS unit(act.f32) -- the same assert targets.py makes, so a reader of either
        # set does not have to trust a docstring.
        vecs = arr / np.maximum(n, 1e-12)[:, None]
    else:
        assert float(np.abs(n - 1).max()) < 2e-3, (
            f"a `storage: {storage}` block stores unit directions; got norms in "
            f"[{n.min():.6f}, {n.max():.6f}]")
        vecs = arr / np.maximum(n, 1e-12)[:, None]

    inputs = {"block": block, "bundle": bundle.SNAPSHOT,
              "from": args.get("dirs_from") or "(Celeste's v2 bundle)"}
    with C.outdir(out_dir, args, inputs=inputs) as od:
        od.write_jsonl("ids.jsonl", rows)
        if storage == "raw":
            od.write_array("act.f32", arr, "float32")
        od.write_array("vecs.f16", vecs, "float16")
        # THE STORAGE CONTRACT (H4): without a storage.json `common.set_storage` refuses the set
        # and every product that reads a direction stops at it.
        od.write_json("storage.json", contract)
        for name, payload in extra.items():
            od.write_json(name, payload)
        od.note(
            f"rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base "
            f"{args['base']} --block {block} --set {set_name} --root {args['root']}"
            + (f" --dirs-from {args['dirs_from']} --rows {args['rows']}"
               if block == "ctrl" else "")
            + f"` at repo commit {args.get('repo_commit', '?')[:12]}"
        )
        od.note(f"IMPORTED / COPIED, NOT DRAWN. Snapshot `{bundle.SNAPSHOT}`, base "
                f"`{bundle.MODEL}`, read layer {bundle.READ_LAYER}, d = {bundle.D_MODEL}. "
                f"Checkpoint these targets are held out from: `{bundle.CHECKPOINT}` "
                f"(revision `{bundle.CHECKPOINT_REVISION}`).")
        for note in notes:
            od.note(note)
        by_fam: dict[str, int] = {}
        for r in rows:
            by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
        od.note(f"n per family: {by_fam}; storage `{storage}`")
        od.note("vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 "
                "off unit")
    out = {"product": "heldout_v3", "set": set_name, "dir": out_dir, "block": block,
           "storage": storage, "rows": len(rows), "families": by_fam,
           **{k: v for k, v in extra.items()}}
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out
