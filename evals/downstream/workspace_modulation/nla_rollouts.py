"""The NLA verbalizer at the two read positions (methodology "Readers"): `stage_nla`, a GPU stage sharded by
item, and `stage_nla_merge`, which concatenates the shards into rollouts/nla.json.

The verbalizer (evals/downstream/common/nla/nla_reader.py, pins `N.PINS`) reads the RAW, uncentred layer-42 row at each
cell, as it was trained; the stage holds it alone and loads no base. Each sample records `text` (what the
word rule and the judge read), `full_text`, their 64-token prefixes `text_trunc` / `full_text_trunc`,
`closed` and its seed."""

import math, time

import numpy as np

from evals.downstream.common.nla import nla_reader as N
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation.rollouts import distinct_share
from evals.downstream.workspace_modulation.runs import stage_key, write_provenance
from evals.downstream.workspace_modulation.stages import provenance_stage, shard_of, stage_record

MERGED_REL = "rollouts/nla.json"
# The fields a saved file must match to be reused.
RESUME_FIELDS = ("reader", "contract", "prompt_ids", "marker_pos", "gen", "seed_base", "upstream")
SHARD_AGREE_FIELDS = RESUME_FIELDS


def shard_rel(k, n):
    return f"rollouts/nla.shard{k}of{n}.json"


def nla_seed():
    """The verbalizer arm's seed: C.GEN_SEED plus its offset in blocks of C.SHARD_SEEDS (independent of --seed)."""
    return int(C.GEN_SEED) + int(C.ARMS["nla"]["seed_offset"]) * C.SHARD_SEEDS


def shard_seed(k):
    """Shard k's seed; a k outside the arm's block is refused."""
    if not 0 <= int(k) < C.SHARD_SEEDS:
        raise RuntimeError(
            f"shard {k} is outside the {C.SHARD_SEEDS} seeds reserved for each arm: it would draw on the "
            "stream of another arm, whose readouts would then be the same samples"
        )
    return nla_seed() + int(k)


def reader_pins():
    """What the verbalizer is, recorded beside its readouts: checkpoint, prompt rendering, budget row, sidecar, base."""
    return {
        "repo": N.PINS.repo,
        "revision": N.PINS.revision,
        "enable_thinking": N.PINS.enable_thinking,
        "trunc": N.PINS.trunc,
        "score_max_length": N.PINS.score_max_length,
        "sidecar_sha256": dict(N.PINS.sidecar_sha256),
        "base_model": [C.MODEL, C.MODEL_REVISION],
    }


def load_reader_tokenizer(pins):
    """The verbalizer's own tokenizer and chat template, without its weights."""
    from evals.downstream.common.model_io import load_tokenizer

    return load_tokenizer(pins.repo, pins.revision)


def nla_rows(items, table):
    """[(activation row, i, pos)] of every kept item's read cells; a cell with no activation row raises."""
    row_of = {(int(i), int(p)): r for r, (i, p, _band) in enumerate(table)}
    out = []
    for it in items:
        i = int(it["i"])
        for pos in M.read_positions(it):
            r = row_of.get((i, int(pos)))
            if r is None:
                raise RuntimeError(
                    f"nla_rows: cell i={i} pos={int(pos)} is not in the activation store — the item file, "
                    "the mean store and the activation store were built from different populations"
                )
            out.append((r, i, int(pos)))
    return out


def shard_rows(rows, k, n):
    """`rows` restricted to shard k of n, by whole items."""
    if n <= 1:
        return list(rows)
    order = {}
    for _r, i, _p in rows:
        order.setdefault(i, len(order))
    return [x for x in rows if order[x[1]] % n == k]


# --- guards ------------------------------------------------------------------------------


def _greedy_text(cell):
    return ((cell.get("greedy") or {}).get("text")) or ""


def per_item_distinctness(items):
    """[{i, n_cells, distinct_share}] of each item's greedy explanations: an item's cells are different
    activations, so one constant explanation means the injection or the marker is wrong."""
    return [
        {
            "i": it["i"],
            "n_cells": len(it["cells"]),
            "distinct_share": distinct_share([_greedy_text(c) for c in it["cells"]]),
        }
        for it in items
    ]


def guard_per_item_distinct(items):
    """Raise when an item repeats a greedy explanation across its cells (after the file is written)."""
    per = per_item_distinctness(items)
    below = [x for x in per if x["distinct_share"] < C.NLA_MIN_DISTINCT]
    if not below:
        return per
    w = min(below, key=lambda x: x["distinct_share"])
    raise RuntimeError(
        f"nla: {len(below)} of {len(per)} items repeat a greedy explanation across their read positions "
        f"(floor {C.NLA_MIN_DISTINCT}: with {len(C.READ_BANDS)} cells every one must differ) "
        f"— worst item i={w['i']} at {w['distinct_share']:.2f} over {w['n_cells']} cells; "
        f"items below: {[x['i'] for x in below][:10]}"
    )


def close_rate(items):
    """(share of samples whose tags closed, n samples); reported, never asserted."""
    flags = [bool(s.get("closed")) for it in items for c in it["cells"] for s in (c.get("samples") or [])]
    return (float(np.mean(flags)) if flags else float("nan")), len(flags)


def _fmt(x, nd=3):
    """A share that may be missing, formatted."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def shard_guards(items):
    """A shard's guards: the worst item's distinctness, the items below the floor, the close rate."""
    per = per_item_distinctness(items)
    worst = min(per, key=lambda x: x["distinct_share"]) if per else None
    rate, n_samples = close_rate(items)
    return {
        "distinct_per_item_min": float(worst["distinct_share"]) if worst else 1.0,
        "n_items_below": sum(1 for x in per if x["distinct_share"] < C.NLA_MIN_DISTINCT),
        "close_rate": rate,
        "n_samples": n_samples,
    }


def distinct_input_share(kept):
    """Distinct (rendered chat, position) pairs as a share of the read cells: the ceiling on pooled greedy
    distinctness (items that render the same chat hold the same activation)."""
    keys = [(it.get("chat"), int(p)) for it in kept for p in M.read_positions(it)]
    return distinct_share(keys)


def pooled_guards(items, kept):
    """The shard guards over the whole population, plus the pooled distinctness and the close-rate flag."""
    per = per_item_distinctness(items)
    worst = min(per, key=lambda x: x["distinct_share"]) if per else None
    texts = [_greedy_text(c) for it in items for c in it["cells"]]
    g = dict(shard_guards(items))
    g.update(
        {
            "n_items": len(items),
            "n_cells": len(texts),
            "distinct_pooled": distinct_share(texts),
            "distinct_input_share": distinct_input_share(kept),
            # the item the abort message names, kept in the record so the report can print it
            "distinct_per_item_worst": worst,
            "close_rate_below_min": bool(g["close_rate"] < N.PINS.min_close_rate),
            "min_close_rate": N.PINS.min_close_rate,
        }
    )
    return g


# --- documents -----------------------------------------------------------------------------------


def shard_doc(items, want, seconds):
    """One shard file."""
    return {
        "config": {
            "arm": "nla",
            **{k: want[k] for k in RESUME_FIELDS},
            "shard": want["shard"],
            "seed": want["seed_base"] + want["shard"][0],
            "guards": shard_guards(items),
            "seconds": float(seconds),
        },
        "items": items,
    }


def resume_view(cfg):
    """A saved file's generation-deciding fields."""
    return {f: cfg.get(f) for f in RESUME_FIELDS}


def cell_records(items):
    """(i, pos) -> the saved cell record."""
    return {(int(it["i"]), int(c["pos"])): c for it in items for c in it.get("cells") or []}


def resume_shard(doc, want, rows):
    """(cells, reason): {(i, pos): saved readout} when every field matches and the saved cells are a subset of
    `rows`; else None and the field that differs."""
    view = resume_view(doc.get("config") or {})
    for f in RESUME_FIELDS:
        if view.get(f) != want[f]:
            return None, f
    saved = cell_records(doc.get("items") or [])
    mine = {(int(i), int(p)) for _r, i, p in rows}
    if any(k not in mine for k in saved):
        return None, "cells"
    return saved, None


def merge_shards_from(docs, kept, n):
    """The merged rollouts/nla.json: every kept item exactly once with its read cells, and the pooled guards."""
    base = docs[0]["config"]
    for d in docs[1:]:
        for f in SHARD_AGREE_FIELDS:
            if d["config"].get(f) != base.get(f):
                raise RuntimeError(
                    f"nla_merge: shard {d['config'].get('shard')} and shard {base.get('shard')} disagree on "
                    f"{f!r}: they were generated by two different readers and must not be pooled"
                )
    seen = {}
    for d in docs:
        for it in d["items"]:
            i = int(it["i"])
            if i in seen:
                raise RuntimeError(f"nla_merge: item i={i} appears in more than one shard")
            seen[i] = it
    items = []
    for it in kept:
        i = int(it["i"])
        got = seen.pop(i, None)
        if got is None:
            raise RuntimeError(f"nla_merge: kept item i={i} is in no shard — that shard's readouts are missing")
        have = [int(c["pos"]) for c in got["cells"]]
        want = M.read_positions(it)
        if have != want:
            raise RuntimeError(f"nla_merge: item i={i} holds cells {have}, its read cells are {want}")
        items.append(got)
    if seen:
        raise RuntimeError(f"nla_merge: shards hold items that are not in the kept population: {sorted(seen)}")
    cfg = {
        k: v
        for k, v in base.items()
        if k not in ("shard", "seed", "guards", "seconds", "reused", "cells_reused", "cells_generated")
    }
    cfg.update(
        {
            "n_shards": n,
            "seeds": {str(int(d["config"]["shard"][0])): d["config"]["seed"] for d in docs},
            "guards": pooled_guards(items, kept),
            "seconds": float(sum(float(d["config"].get("seconds") or 0.0) for d in docs)),
        }
    )
    return {"config": cfg, "items": items}


def merge_shards(run, n):
    """The merged document from this run's `n` shard files; every missing shard is named."""
    missing = [shard_rel(k, n) for k in range(n) if not run.exists(shard_rel(k, n))]
    if missing:
        raise RuntimeError(f"nla_merge: {len(missing)} of {n} shard files are missing: {', '.join(missing)}")
    docs = [run.read_json(shard_rel(k, n)) for k in range(n)]
    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    return merge_shards_from(docs, kept, n)


# --- stages --------------------------------------------------------------------------------------


def stage_nla(args, run):
    k, n = shard_of(args)
    chash = stage_key("nla", args, run)
    name = stage_record("nla", k, n)
    if stage_done(run, name, chash) and not args.force:
        print(f"[nla:{k}of{n}] up to date")
        return
    started = time.time()
    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    rows = shard_rows(nla_rows(kept, M.full_table(run)), k, n)
    mine = [it for it in kept if it["i"] in {i for _r, i, _p in rows}]
    # The resume decision needs only the sidecar and the tokenizer.
    P = N.PINS
    con = N.contract(N.load_sidecar(pins=P))
    tok = load_reader_tokenizer(P)
    ids, mpos = N.prompt_ids(tok, con, pins=P)
    want = {
        "reader": reader_pins(),
        "contract": con,
        "prompt_ids": [int(t) for t in ids],
        "marker_pos": int(mpos),
        "gen": dict(C.ARMS["nla"]),
        "shard": [k, n],
        "seed_base": nla_seed(),
        "upstream": dict(chash.upstream),
    }
    rel = shard_rel(k, n) if n > 1 else MERGED_REL
    t0, cells, seconds = time.time(), {}, 0.0
    if run.exists(rel) and not args.force:
        saved = run.read_json(rel)
        cells, reason = resume_shard(saved, want, rows)
        if cells is None:
            print(f"[nla:{k}of{n}] {rel} exists but its config {reason!r} differs; regenerating", flush=True)
            cells = {}
        else:
            seconds = float(saved["config"].get("seconds") or 0.0)
    missing = [x for x in rows if (x[1], x[2]) not in cells]
    n_reused, reused = len(cells), bool(cells) and not missing
    checkpoint = None
    if missing:
        from evals.downstream.common.model_io import free_model

        H = M.load_h(run)
        H42 = np.asarray(H[[r for r, _i, _p in missing]], dtype=np.float32)
        # each `generate` call is seeded by its first cell's place in the shard's rows
        place = {(i, p): r for r, (_row, i, p) in enumerate(rows)}
        snapshot = N.download_checkpoint(pins=P)
        checkpoint = N.check_checkpoint(snapshot, P)
        mdl, _tok = N.load_verbalizer(args.device, snapshot, pins=P)
        try:
            samples, greedy = N.generate_explanations(
                mdl,
                tok,
                H42,
                ids,
                mpos,
                args.device,
                n_samples=C.ARMS["nla"]["n_samples"],
                seed=shard_seed(k),
                gen_chunk=P.gen_chunk,
                greedy=True,
                pins=P,
                row_ids=[place[(i, p)] for _row, i, p in missing],
            )
        finally:
            mdl = free_model(mdl)
        for r, (_row, i, pos) in enumerate(missing):
            cells[(i, pos)] = {"pos": pos, "greedy": greedy[r], "samples": samples[r]}
        new_seconds = time.time() - t0
        seconds += new_seconds
    else:
        new_seconds = 0.0
    by_i = {}
    for _row, i, pos in rows:
        by_i.setdefault(i, []).append(cells[(i, pos)])
    items = [{"i": it["i"], "cells": by_i[it["i"]]} for it in mine]
    doc = shard_doc(items, want, seconds)
    if n == 1:
        # --n-shards 1 is its own merge.
        doc = merge_shards_from([doc], kept, 1)
    if reused:
        doc["config"]["reused"] = True
    doc["config"]["cells_reused"] = n_reused
    doc["config"]["cells_generated"] = len(missing)
    run.write_json(rel, doc)
    guards = doc["config"]["guards"]
    write_provenance(
        run,
        {
            f"nla_shard{k}of{n}": {
                "guards": guards,
                "seed": shard_seed(k),
                "n_items": len(items),
                "n_cells": len(rows),
                "n_cells_reused": n_reused,
                "n_cells_generated": len(missing),
                "seconds": seconds,
                "new_seconds": new_seconds,
                "reused": reused,
                # None when every cell was reused and no snapshot was opened
                "checkpoint": checkpoint,
            }
        },
        stage=provenance_stage("nla", k, n),
    )
    print(
        f"[nla:{k}of{n}] {'reused ' if reused else ''}{rel}: {len(items)} items x {len(rows)} cells "
        f"x {C.ARMS['nla']['n_samples']}+greedy ({n_reused} cells reused, {len(missing)} generated in "
        f"{new_seconds:.0f}s); guards {guards}",
        flush=True,
    )
    guard_per_item_distinct(items)
    mark_stage(
        run,
        name,
        chash,
        {
            "shard": [k, n],
            "n_items": len(items),
            "n_cells": len(rows),
            "cells_reused": n_reused,
            "cells_generated": len(missing),
            "generation_seconds": seconds,
            "new_generation_seconds": new_seconds,
            "reused": reused,
            "close_rate": guards["close_rate"],
        },
        started=started,
    )


def stage_nla_merge(args, run):
    n = int(getattr(args, "n_shards", 1) or 1)
    chash = stage_key("nla_merge", args, run, n_shards=n)
    if stage_done(run, "nla_merge", chash) and not args.force:
        print("[nla_merge] up to date")
        return
    started = time.time()
    have_shards = any(run.exists(shard_rel(k, n)) for k in range(n))
    if run.exists(MERGED_REL) and not have_shards:
        # the --n-shards 1 path already wrote the merged file
        doc = run.read_json(MERGED_REL)
        print(f"[nla_merge] {MERGED_REL} was written by a single-shard run and no shard files exist: nothing to merge")
    else:
        doc = merge_shards(run, n)
        run.write_json(MERGED_REL, doc)
    guards = doc["config"]["guards"]
    write_provenance(run, {"nla_guards": guards}, stage="nla_merge")
    print(
        f"[nla_merge] {len(doc['items'])} items, {guards['n_cells']} cells over {n} shard(s); "
        f"distinct {_fmt(guards['distinct_pooled'])} against a distinct-input share of "
        f"{_fmt(guards['distinct_input_share'])}; close-tag rate {_fmt(guards['close_rate'])} over "
        f"{guards['n_samples']} samples"
        + (f" — BELOW {N.PINS.min_close_rate}, the recipe is flagged" if guards["close_rate_below_min"] else ""),
        flush=True,
    )
    mark_stage(
        run,
        "nla_merge",
        chash,
        {
            "n_items": len(doc["items"]),
            "n_cells": guards["n_cells"],
            "n_shards": n,
            "close_rate": guards["close_rate"],
        },
        started=started,
    )
