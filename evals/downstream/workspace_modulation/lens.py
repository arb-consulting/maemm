"""The released Jacobian lens at every carrier cell and fitted layer, and at the synthetic mean cell at the
read layer: ranks of every single-token target form, the ten most likely word-like tokens per layer, donor
ranks for the chance lines, and a three-cell self-check. The carrier pass also serves the J-lens paper's
any-token protocol (lens rows only)."""

import json
import time
import numpy as np
from evals.downstream.common.runs import mark_stage, read_provenance, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation.runs import stage_key, write_provenance


def form_table(concepts, tok):
    ids, strs, cols = [], [], {}
    for key, c in concepts.items():
        cols[key] = []
        for f in c["lens_forms"]:
            for cand in (" " + f, f):
                enc = tok.encode(cand, add_special_tokens=False)
                if len(enc) != 1:
                    continue
                t = int(enc[0])
                if t not in ids:
                    ids.append(t)
                    strs.append(cand)
                col = ids.index(t)
                if col not in cols[key]:
                    cols[key].append(col)
                break
    return ids, strs, cols


def _min_or_none(vals):
    vals = [int(v) for v in np.ravel(vals) if v is not None]
    return min(vals) if vals else None


def cell_summary(ranks_cell, cols_own, cols_donors, fitted, read_layer, top_by_layer):
    li = fitted.index(read_layer)
    by_layer = [(_min_or_none(ranks_cell[l, cols_own]) if cols_own else None) for l in range(len(fitted))]
    mins = [(v, fitted[l]) for l, v in enumerate(by_layer) if v is not None]
    best = min(mins) if mins else (None, None)
    return {
        "rank_L42": by_layer[li],
        "min_rank": best[0],
        "min_rank_layer": best[1],
        "rank_by_layer": by_layer,
        "top10_L42": list(top_by_layer[li]),
        "donor_rank_L42": [(_min_or_none(ranks_cell[li, cd]) if cd else None) for cd in cols_donors],
        "donor_min_rank": [(_min_or_none(ranks_cell[:, cd]) if cd else None) for cd in cols_donors],
    }


def _top_wordlike(z, mask, k):
    return np.argsort(-np.where(mask, z, -np.inf), kind="stable")[:k]


def _ranks(z, ids):
    """1-based ranks of `ids` over the full vocabulary, ties broken by token id (`kind="stable"`). Recorded,
    never asserted on: see _cell_check."""
    order = np.argsort(-z, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[ids] + 1


def _cell_check(z1, zb, form_ids, own_cols, mask, top_word=C.TOP_WORD):
    """Compare one cell's read-layer logits between the single-row (z1) and the batched pass (zb). Returns
    (record, ok): the own-form logits agree within LENS_SELF_CHECK_LOGIT_TOL and the top-10 word-like sets share
    ≥ LENS_SELF_CHECK_TOP_OVERLAP (a bf16 head ties too many logits for exact ranks); the rank difference is
    recorded only."""
    ids = np.asarray([form_ids[c] for c in own_cols], dtype=np.int64)
    z1, zb = np.asarray(z1, dtype=np.float32), np.asarray(zb, dtype=np.float32)
    d = float(np.max(np.abs(z1[ids] - zb[ids]))) if len(ids) else None
    r = int(np.max(np.abs(_ranks(z1, ids).astype(np.int64) - _ranks(zb, ids).astype(np.int64)))) if len(ids) else None
    overlap = len(set(_top_wordlike(z1, mask, top_word).tolist()) & set(_top_wordlike(zb, mask, top_word).tolist()))
    rec = {
        "max_abs_logit_diff": d,
        "top10_overlap": overlap,
        "max_abs_rank_diff": r,
        "n_own_forms": int(len(ids)),
    }
    ok = (d is None or d <= C.LENS_SELF_CHECK_LOGIT_TOL) and overlap >= C.LENS_SELF_CHECK_TOP_OVERLAP
    return rec, ok


def _self_check(H_item, lens, un, mask, form_ids, own_cols, device, cells, read_layer):
    """Recompute `read_layer` one row at a time for each cell in `cells` and compare with the batched pass; raises on a failure."""
    from evals.downstream.common.lens_io import layer_logits

    def np_of(z):
        return z if isinstance(z, np.ndarray) else z.detach().float().cpu().numpy()

    Zb = layer_logits(H_item[:, read_layer], lens, un, read_layer, device)
    out = []
    for k in cells:
        z1 = np_of(layer_logits(H_item[k : k + 1, read_layer], lens, un, read_layer, device))[0]
        rec, ok = _cell_check(z1, np_of(Zb[k]), form_ids, own_cols, mask)
        out.append({"cell": int(k), **rec})
        if not ok:
            raise RuntimeError(
                f"lens self-check failed at cell {k}, single-row pass against the batched one: own-form "
                f"logit difference {rec['max_abs_logit_diff']} (tolerance {C.LENS_SELF_CHECK_LOGIT_TOL}); "
                f"top-10 word-like overlap {rec['top10_overlap']} of {C.TOP_WORD} (at least "
                f"{C.LENS_SELF_CHECK_TOP_OVERLAP} required); rank difference {rec['max_abs_rank_diff']} "
                "(recorded, not a criterion)"
            )
    return out


def mean_cell_summary(z, cols_own, cols_donors, form_ids, mask, tok, top_word=C.TOP_WORD):
    """One synthetic mean cell's lens record, at the read layer only: the any-layer fields are null, since the
    mean cell exists at layer 42 alone."""
    r = _ranks(np.asarray(z, dtype=np.float32), np.asarray(form_ids, dtype=np.int64)) if len(form_ids) else []
    return {
        "pos": C.MEAN_POS,
        "band": C.MEAN_BAND,
        "rank_L42": _min_or_none([r[c] for c in cols_own]) if cols_own else None,
        "min_rank": None,
        "min_rank_layer": None,
        "rank_by_layer": None,
        "top10_L42": [tok.decode([int(t)]) for t in _top_wordlike(np.asarray(z, dtype=np.float32), mask, top_word)],
        "donor_rank_L42": [(_min_or_none([r[c] for c in cd]) if cd else None) for cd in cols_donors],
        "donor_min_rank": [None] * len(cols_donors),
    }


def mean_cell_records(run, kept, lens, un, tok, mask, form_ids, cols, device, chunk=64):
    """{item -> the synthetic cell's lens record} over the mean store, or {} when `cells_mean` has not run."""
    from evals.downstream.common.lens_io import layer_logits

    rows = M.mean_rows(run)
    if not rows:
        return {}
    H = np.load(run.file(M.MEAN_REL))["h"]
    by_i = {int(i): k for k, (i, _p, _b) in enumerate(rows)}
    out = {}
    for s in range(0, len(H), chunk):
        Z = layer_logits(np.asarray(H[s : s + chunk], dtype=np.float32), lens, un, C.READ_LAYER, device)
        Z = Z if isinstance(Z, np.ndarray) else Z.detach().float().cpu().numpy()
        for j in range(Z.shape[0]):
            i = rows[s + j][0]
            it = next(x for x in kept if int(x["i"]) == int(i))
            out[int(i)] = mean_cell_summary(
                Z[j], cols.get(it["concept_key"], []), [cols.get(d, []) for d in it["donors"]], form_ids, mask, tok
            )
    assert set(out) == set(by_i), "the mean store and its table disagree on which items carry a synthetic cell"
    return out


# The per-layer word-like top-10 lists of every carrier cell, one line per cell.
TOP_BY_LAYER_REL = "lens/top10_by_layer.jsonl"


def band_pools(run, cells=None):
    """{(i, pos): the eight-layer pool} from the saved per-layer top-10 lists (`lens_io.pool_layers`)."""
    from evals.downstream.common.lens_io import pool_layers

    if not (run.exists(TOP_BY_LAYER_REL) and run.exists("lens/lens.json")):
        return {}
    fitted = run.read_json("lens/lens.json")["lens"]["fitted_layers"]
    want = None if cells is None else {(int(i), int(p)) for i, p in cells}
    out = {}
    for rec in run.read_jsonl(TOP_BY_LAYER_REL):
        key = (int(rec["i"]), int(rec["pos"]))
        if want is None or key in want:
            out[key] = pool_layers(rec["top10_by_layer"], fitted, C.LENS_BAND_LAYERS)
    return out


def carrier_cells_of(rec):
    """One saved item record's carrier cells: every cell but the synthetic mean one, which a previous run of
    this stage may already have appended."""
    return [c for c in (rec.get("cells") or []) if c.get("pos") != C.MEAN_POS]


def resume_lens(doc, want, kept):
    """(items, reason): the saved per-item records when lens/lens.json can stand in for this run's carrier
    pass (same lens, read layer, top-word count, form table, capture record and items); else None and the
    field that differs. Only the mean cell is then recomputed."""
    if _lens_pins(doc) != want["lens"]:
        return None, "lens"
    for f in ("read_layer", "top_word", "form_strs", "capture_record"):
        if (doc or {}).get(f) != want[f]:
            return None, f
    items = doc.get("items") or []
    if [int(x["i"]) for x in items] != [int(it["i"]) for it in kept]:
        return None, "items"
    for x, it in zip(items, kept):
        if [int(c["pos"]) for c in carrier_cells_of(x)] != [int(p) for p in it["carrier_cells"]]:
            return None, "cells"
    return items, None


def _lens_pins(doc):
    """The lens file a saved record was produced from, in the four fields `load_lens` verifies."""
    p = (doc or {}).get("lens") or {}
    return [p.get("repo"), p.get("revision"), p.get("file"), p.get("sha256")]


def stage_lens(args, run):
    chash = stage_key("lens", args, run)
    if stage_done(run, "lens", chash) and not args.force:
        print("[lens] up to date")
        return
    started = time.time()
    from evals.downstream.common.lens_io import Unembed, lens_pass, load_lens, wordlike_mask
    from evals.downstream.common.model_io import load_base

    doc = run.read_json("data/items.json")
    kept = [x for x in doc["items"] if not x["excluded"]]
    base, tok = load_base(args.device, C.MODEL, C.MODEL_REVISION)
    lens, prov = load_lens(
        args.device, C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE, C.LENS_BYTES, C.LENS_SHA256, C.READ_LAYER, C.D_MODEL
    )
    un = Unembed(base)
    form_ids, form_strs, cols = form_table(doc["concepts"], tok)
    mask = wordlike_mask(tok, un.vocab)
    fitted = prov["fitted_layers"]
    want = {
        "read_layer": C.READ_LAYER,
        "top_word": C.TOP_WORD,
        "form_strs": form_strs,
        "lens": [C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE, C.LENS_SHA256],
        "capture_record": chash.upstream.get("capture"),
    }
    saved, reason = (None, "missing")
    if run.exists("lens/lens.json") and not args.force:
        saved, reason = resume_lens(run.read_json("lens/lens.json"), want, kept)
        if saved is None:
            print(f"[lens] lens/lens.json exists but its {reason!r} differs; recomputing the carrier pass", flush=True)
    means = mean_cell_records(run, kept, lens, un, tok, mask, form_ids, cols, args.device)
    t0 = time.time()
    if saved is not None:
        recs = [
            {
                "i": x["i"],
                "single_token": x["single_token"],
                "cells": carrier_cells_of(x) + ([means[x["i"]]] if x["i"] in means else []),
            }
            for x in saved
        ]
        rows = [(x["i"], c["pos"]) for x in saved for c in carrier_cells_of(x)]
        checks = read_provenance(run).get("lens_self_check") or []
        print(f"[lens] reusing the carrier pass over {len(rows)} cells from lens/lens.json", flush=True)
        _write_lens(run, prov, form_strs, recs, checks, C.READ_LAYER, want["capture_record"])
        _finish_lens(args, run, chash, rows, means, form_ids, fitted, checks, t0, started, reused=True)
        return
    all_ranks, rows, recs, checks = [], [], [], []
    with open(run.file("lens/top10_by_layer.jsonl"), "w", encoding="utf-8") as top_out:
        for n, it in enumerate(kept):
            H = np.load(run.file(f"activations/cells/{it['i']}.npz"))["h_all"]  # [n_carrier, 64, d]
            ranks, top = lens_pass(H, lens, un, tok, mask, form_ids, args.device, top_word=C.TOP_WORD)
            if n < 3:
                checks += _self_check(
                    H, lens, un, mask, form_ids, cols.get(it["concept_key"], []), args.device, [0], C.READ_LAYER
                )
            cd = [cols.get(d, []) for d in it["donors"]]
            cells = []
            for k, p in enumerate(it["carrier_cells"]):
                cells.append(
                    {
                        "pos": p,
                        **cell_summary(ranks[k], cols.get(it["concept_key"], []), cd, fitted, C.READ_LAYER, top[k]),
                    }
                )
                top_out.write(json.dumps({"i": it["i"], "pos": p, "top10_by_layer": top[k]}, ensure_ascii=False) + "\n")
                rows.append((it["i"], p))
            if it["i"] in means:
                cells.append(means[it["i"]])
            all_ranks.append(ranks)
            recs.append({"i": it["i"], "single_token": it["single_token"], "cells": cells})
            if n % 20 == 19:
                print(f"  [lens] {n + 1}/{len(kept)}", flush=True)
    np.savez_compressed(
        run.file("lens/ranks.npz"),
        ranks=np.concatenate(all_ranks),
        fitted_layers=np.array(fitted),
        form_ids=np.array(form_ids, dtype=np.int64),
        carrier_rows=np.array(rows),
    )
    _write_lens(run, prov, form_strs, recs, checks, C.READ_LAYER, want["capture_record"])
    _finish_lens(args, run, chash, rows, means, form_ids, fitted, checks, t0, started, reused=False)


def _write_lens(run, prov, form_strs, recs, checks, read_layer, capture_record=None):
    run.write_json(
        "lens/lens.json",
        {
            "lens": prov,
            "read_layer": read_layer,
            "top_word": C.TOP_WORD,
            "form_strs": form_strs,
            "capture_record": capture_record,
            "items": recs,
        },
    )
    write_provenance(run, {"lens_file": prov, "lens_self_check": checks}, stage="lens")


def _finish_lens(args, run, chash, rows, means, form_ids, fitted, checks, t0, started, reused):
    print(
        f"[lens] {len(rows)} carrier cells x {len(fitted)} layers x {len(form_ids)} forms "
        f"({'reused' if reused else 'recomputed'}) + {len(means)} synthetic mean cells at layer "
        f"{C.READ_LAYER} only, in {time.time() - t0:.0f}s; self-check {checks}",
        flush=True,
    )
    mark_stage(
        run,
        "lens",
        chash,
        {
            "n_cells": len(rows),
            "n_mean_cells": len(means),
            "n_forms": len(form_ids),
            "carrier_pass_reused": reused,
        },
        started=started,
    )
