"""Stage `lens` (methodology §3.2): the released Jacobian lens's target ranks at every fitted layer and its
word-like top-10 lists, per item. Loading and the lens pass are evals/downstream/common/lens_io.py's."""

import time
import numpy as np
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance
from evals.downstream.common.lens_io import Unembed, lens_pass, pool_layers, wordlike_mask
from evals.downstream.common.lens_io import load_lens as _load_lens


def load_lens(device):
    """The lens this package pins, checked against its recorded size and digest before it is trusted."""
    return _load_lens(
        device,
        C.LENS_REPO,
        C.LENS_REVISION,
        C.LENS_FILE,
        C.LENS_BYTES,
        C.LENS_SHA256,
        C.READ_LAYER,
        C.D_MODEL,
    )


def pool_band(top10_by_layer, fitted, layers=C.LENS_BAND_LAYERS):
    """The eight-layer readout (`jlens_band8`): the top-10 word-like lists of layers 36, 38, ..., 50 pooled,
    case-folded and ordered by each token's best list position, from an item's saved `top10_by_layer`."""
    return pool_layers(top10_by_layer, fitted, layers)


def form_index(kept):
    """(form_ids, cols_by_item): the distinct target-form token ids over all kept items, and each item's
    columns in that list."""
    form_ids, pos, cols_by_item = [], {}, {}
    for it in kept:
        cols = set()
        for f in it["forms"]:
            for tid in it["form_ids"][f]["ids"]:
                if tid not in pos:
                    pos[tid] = len(form_ids)
                    form_ids.append(tid)
                cols.add(pos[tid])
        cols_by_item[it["i"]] = sorted(cols)
    return form_ids, cols_by_item


def item_rank_by_layer(ranks_row, cols):
    if not cols:
        return np.full(ranks_row.shape[0], np.nan)
    return ranks_row[:, cols].min(axis=1).astype(float)


def best_layer(rank_by_layer, fitted):
    if np.all(np.isnan(rank_by_layer)):
        return None, None
    k = int(np.nanargmin(rank_by_layer))
    return int(rank_by_layer[k]), int(fitted[k])


def summarise_item(i, ranks_i, cols, donor_rows, fitted, top10_by_layer, read_layer):
    fitted = list(fitted)
    l42 = fitted.index(read_layer)
    rbl = item_rank_by_layer(ranks_i, cols)
    mr, ml = best_layer(rbl, fitted)

    def _at(row):
        return None if not cols else int(item_rank_by_layer(row, cols)[l42])

    def _min(row):
        if not cols:
            return None
        m, _ = best_layer(item_rank_by_layer(row, cols), fitted)
        return m

    return {
        "i": i,
        "multi_token": not cols,
        "rank_L42": None if not cols else int(rbl[l42]),
        "min_rank": mr,
        "min_rank_layer": ml,
        "rank_by_layer": [None if np.isnan(v) else int(v) for v in rbl],
        "top10_L42": top10_by_layer[l42],
        "top10_by_layer": top10_by_layer,
        "donor_rank_L42": [_at(r) for r in donor_rows],
        "donor_min_rank": [_min(r) for r in donor_rows],
    }


def stage_lens(args, run):
    chash = stage_key("lens", args, run)
    if stage_done(run, "lens", chash) and not args.force:
        print("[lens] up to date")
        return
    started = time.time()
    from evals.downstream.workspace_understanding import model as MD

    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    H = np.load(run.file("activations/h_all.npz"))["h"]
    dev = args.device
    base, tok = MD.load_base(dev)
    lens, prov = load_lens(dev)
    un = Unembed(base)
    form_ids, cols = form_index(kept)
    mask = wordlike_mask(tok, un.vocab)
    ranks, top = lens_pass(H, lens, un, tok, mask, form_ids, dev, top_word=C.TOP_WORD)
    fitted = prov["fitted_layers"]
    recs = []
    for it in kept:
        i = it["i"]
        recs.append(
            summarise_item(i, ranks[i], cols[i], [ranks[d] for d in it["donors"]], fitted, top[i], C.READ_LAYER)
        )
    np.savez_compressed(
        run.file("lens/ranks.npz"),
        ranks=ranks,
        form_ids=np.array(form_ids, dtype=np.int64),
        fitted_layers=np.array(fitted),
    )
    run.write_json(
        "lens/lens.json", {"lens": prov, "read_layer": C.READ_LAYER, "n_forms": len(form_ids), "items": recs}
    )
    write_provenance(run, {"lens_file": prov}, stage="lens")
    print(
        f"[lens] {len(recs)} items, {len(fitted)} fitted layers, {sum(1 for r in recs if r['multi_token'])} multi-token targets",
        flush=True,
    )
    mark_stage(run, "lens", chash, {"n_items": len(recs)}, started=started)
