"""Product `tierb` (CPU): cos > 0.999 of OUR target blocks against Celeste's v2 TRAINING rows.

    <root>/base/<base>/tierb/
        summary.json   every count and quantile below, plus the six arrays as read
        hits.jsonl     one row per (target, bank row) pair above the threshold, capped
        maxcos.f32     [n_blocks_total, n_banks + 1] the per-target max cosine, banks then `all`
        README.md      the OutDir record (command line, inputs, wall, cost)

This is `targets._leakage` -- the same `LEAK_COS = 0.999`, the same chunked scan, now the shared
`targets.leak_scan` -- pointed at the 27B instead of the 8B archive, and run OFF the draw path.
It is a separate product rather than a step of `targets` for two reasons: the banks are 8.94M rows
/ 92 GB, so every 27B draw would pay a 92 GB read it does not need; and the blocks worth checking
(`2026-09-21_v3_ours`, the 131k half of `2026-09-21_v3_ctrl`) were drawn days before the tier-B
fetch existed and are not being re-drawn.

WHAT EACH BLOCK BUYS (infra/2026-09-18_celeste-v2-data.md, spec §8 item 2/4):

  * `ours_*` -- our own 512 realact rows. Her v3 realact 512 is already covered by her OWN leak
    check at the same threshold (0 rows dropped, max cos 0.908), so OUR realact block is the only
    new information here.
  * `ctrl_sae131k` -- the 512 rows of `2026-09-21_v3_ctrl` that are 131k-SAE encoder columns. Tier
    B holds NO 131k directions (that is the legacy chain, tier C), so this does not test whether
    the 131k dictionary leaked; it tests whether those 512 columns are collinear with the 2M
    columns she trained on, which is the out-of-dictionary sentence. For the 2M SAE itself the
    check says nothing at all: that exclusion is exact at feature-id level.

THE CENTRING CAVEAT, and why `ours` is scanned TWICE. Her rows and ours may not be centred the
same way -- the cosine of the two means is 0.9774 (`stats/mu.f32` vs the archived `whiten_mu`), and
question 3 of the bundle survey ("are v2 realact rows `unit(act)` or `unit(act - mu)`, and which
mu") is still unanswered. A duplicate that exists under one convention can sit well below 0.999
under the other, so a single reading could hide it. `ours_raw` is `unit(act)` -- uncentred both
sides, which is the reading the eval spec names -- and `ours_centred` is `unit(act - whiten_mu)`,
the base's scoring constant (`common.score_mu`). Neither is "the" answer; the pair is, and the
appendix sentence reports the higher of the two counts.

The 131k block is an encoder column: `family_kinds` says it is not `centrable`, there is no mean to
subtract, and `dirs_for` returns the same direction at either mu. It is scanned once.

    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
        uvx modal run --detach repo-maemm-m8/paper-evals/precompute/modal_app.py \\
        --product tierb --base qwen36-27b)
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C
from precompute import targets as T

# The bundle, relative to --root. Mirrored by `infra/fetch_celeste_v2.py` (tier B, 2026-09-22);
# keys are the S3 keys minus the `v2-2026-09-17/` prefix.
BUNDLE = "data/celeste-v2-2026-09-17"

# THE SIX ARRAYS, with the row count each one must have. The counts are the bundle's own
# (`infra/2026-09-18_celeste-v2-data.md` §1.1, from her `manifest.json`), and they are ASSERTED
# rather than read: a resumable fetch whose last range never landed leaves a short array that a
# scan would happily read to its end and report clean.
BANKS = (
    ("sft_mix/realact", 4_000_000),
    ("sft_mix/sae2m", 2_000_000),
    ("sft_mix/sae2m_dec", 2_000_000),
    ("rl_pool/realact_ctx64_2048", 470_566),
    ("rl_pool/sae2m", 235_283),
    ("rl_pool/sae2m_dec", 235_283),
)

# (block name, held-out set, family, centring). `mu: "score"` means the base's scoring constant
# (`common.score_mu`); `None` means uncentred. Rows are selected BY FAMILY, not by index range --
# `2026-09-21_v3_ctrl` is 512 `random` rows then 512 `sae` rows and an index range would silently
# follow the draw order if it ever changed.
BLOCKS = (
    ("ours_raw", "2026-09-21_v3_ours", "realact", None),
    ("ours_centred", "2026-09-21_v3_ours", "realact", "score"),
    ("ctrl_sae131k", "2026-09-21_v3_ctrl", "sae", None),
)

# This bundle is a Qwen3.6-27B layer-42 bank (d = 5120); there is no tier B for any other base.
BASE = "qwen36-27b"
# Bank rows per matmul. 16k x 5120 fp32 in, 16k x N_targets fp32 out -- ~400 MB of working set at
# the ~1.5k target rows below, against `targets.LEAK_CHUNK`'s 64k, which would be ~1.6 GB.
CHUNK = 16_384
# Reported beside the max, per the gate: the whole distribution in two numbers.
PCTL = 99.0


def _block_dirs(cfg, base: str, root: str, od_notes: list):
    """[(name, set, family, mu_label, rows, dirs [n, d]), ...] for every block in BLOCKS.

    `dirs_for` returns the WHOLE set at one mean; the family filter happens here, which is also
    where the `row` indices that `hits.jsonl` reports are recorded, so a hit points back at a row
    of the SET and not at an offset into a concatenation.
    """
    out = []
    cache: dict[tuple, tuple] = {}
    for name, set_name, family, centring in BLOCKS:
        set_dir = C.heldout_dir(base, set_name, root)
        assert os.path.exists(set_dir), (
            f"block {name!r} needs the held-out set {set_name!r} at {set_dir}, which is not on "
            f"this --root. Nothing here draws it."
        )
        mu = C.score_mu(cfg, base) if centring == "score" else None
        key = (set_name, mu)
        if key not in cache:
            notes: list[str] = []
            rows = C.read_jsonl(f"{set_dir}/ids.jsonl")
            cache[key] = (rows, C.dirs_for(cfg, base, set_dir, mu, root, notes))
            od_notes.extend(f"{set_name} @ mu={C.mu_label(mu, base, root)}: {ln}" for ln in notes)
        rows, dirs = cache[key]
        sel = [i for i, r in enumerate(rows) if r["family"] == family]
        assert sel, (
            f"block {name!r}: no row of {set_name} has family {family!r} "
            f"(families present: {sorted({r['family'] for r in rows})})"
        )
        out.append({
            "name": name,
            "set": set_name,
            "family": family,
            "mu": C.mu_label(mu, base, root),
            "rows": [rows[i]["row"] for i in sel],
            "ids": [rows[i]["id"] for i in sel],
            "dirs": np.asarray(dirs)[sel],
        })
        print(
            f"[tierb] block {name}: {len(sel)} {family} rows of {set_name} at "
            f"mu={C.mu_label(mu, base, root)}",
            flush=True,
        )
    return out


def _stats(best: np.ndarray, thr: float) -> dict:
    """The two numbers the gate asks for, plus the count it is really about."""
    return {
        "n": int(best.size),
        "n_above": int((best > thr).sum()),
        "max": round(float(best.max()), 6),
        f"p{PCTL:g}": round(float(np.percentile(best, PCTL)), 6),
        "median": round(float(np.median(best)), 6),
    }


def run(cfg, args):
    base, root = args["base"], args["root"]
    assert base == BASE, (
        f"product tierb is the check against Celeste's v2 training directions, which are a "
        f"{BASE} layer-42 bank; --base {base!r} has no tier B. (The 8B's equivalent runs inside "
        f"`targets` -- see targets._leakage.)"
    )
    d, thr = cfg["bases"][base]["d"], T.LEAK_COS
    bundle = f"{root.rstrip('/')}/{BUNDLE}"
    assert os.path.exists(bundle), (
        f"no {bundle}: tier B of the celeste-v2 bundle is not on this --root. "
        f"`infra/fetch_celeste_v2.py` mirrors it."
    )

    # EVERY array is opened and size-checked BEFORE the first matmul: an array that cannot be read
    # is named here, in seconds, rather than after an hour of scanning the other five (the gate).
    banks, missing = [], []
    for name, want_rows in BANKS:
        path = f"{bundle}/simple2m/{name}/dirs_f16.npy"
        try:
            n_rows, _ = T.open_bank(path, d)
            assert n_rows == want_rows, (
                f"{path} has {n_rows} rows, the bundle manifest says {want_rows} -- an INCOMPLETE "
                f"fetch, not a bank. Re-run infra/fetch_celeste_v2.py for this key."
            )
        except (AssertionError, OSError, ValueError) as e:
            missing.append({"bank": name, "path": path, "error": str(e)})
            print(f"[tierb] UNREADABLE {name}: {e}", flush=True)
            continue
        banks.append((name, path))
        print(f"[tierb] bank {name}: {n_rows} rows, {C.human(os.path.getsize(path))}", flush=True)
    assert banks, f"not one of the {len(BANKS)} tier-B arrays could be read: {missing}"

    out = f"{root.rstrip('/')}/base/{base}/tierb"
    inputs = {
        "bundle": bundle,
        "banks": {n: f"{bundle}/simple2m/{n}/dirs_f16.npy" for n, _ in BANKS},
        "blocks": {b[0]: {"set": b[1], "family": b[2], "centring": b[3] or "none"} for b in BLOCKS},
        "threshold": thr,
    }
    with C.outdir(out, args, inputs=inputs) as od:
        notes: list[str] = []
        blocks = _block_dirs(cfg, base, root, notes)
        for line in notes:
            od.note(line)

        # ONE pass over 92 GB for every block there is; the split back out is by row range.
        dirs = np.concatenate([b["dirs"] for b in blocks])
        spans, s = {}, 0
        for b in blocks:
            spans[b["name"]] = (s, s + len(b["rows"]))
            s += len(b["rows"])
        t0 = time.time()
        res = T.leak_scan(dirs, banks, d, thr=thr, chunk=CHUNK, device="numpy", label="tierb")
        scan_s = time.time() - t0

        bank_names = [n for n, _ in banks]
        summary = {
            "base": base,
            "threshold": thr,
            "scan_seconds": round(scan_s, 1),
            "bank_rows_total": sum(bk["rows"] for bk in res["banks"]),
            "banks": res["banks"],
            "unreadable": missing,
            "blocks": {},
        }
        for b in blocks:
            lo, hi = spans[b["name"]]
            per_bank = {n: _stats(res["per_bank"][n][lo:hi], thr) for n in bank_names}
            best = res["best"][lo:hi]
            top = int(np.argmax(best))
            summary["blocks"][b["name"]] = {
                "set": b["set"],
                "family": b["family"],
                "mu": b["mu"],
                "set_rows": [b["rows"][0], b["rows"][-1]],
                "all_banks": _stats(best, thr),
                "argmax": {
                    "set_row": b["rows"][top],
                    "id": b["ids"][top],
                    "bank": res["best_bank"][lo + top],
                    "bank_row": int(res["best_row"][lo + top]),
                    "cos": round(float(best[top]), 6),
                },
                "per_bank": per_bank,
            }
            st = summary["blocks"][b["name"]]["all_banks"]
            print(
                f"[tierb] {b['name']}: {st['n_above']} of {st['n']} rows above {thr}; "
                f"max {st['max']}, p{PCTL:g} {st[f'p{PCTL:g}']}",
                flush=True,
            )

        od.write_json("summary.json", summary)

        def _locate(target: int) -> tuple[str, int, object]:
            """A row of the concatenation -> (block name, its row in the SET, that row's id)."""
            for b in blocks:
                lo, hi = spans[b["name"]]
                if lo <= target < hi:
                    return b["name"], b["rows"][target - lo], b["ids"][target - lo]
            raise AssertionError(f"target row {target} is in no block; spans {spans}")

        hit_rows = []
        for h in res["hits"]:
            block, set_row, ident = _locate(h["target"])
            hit_rows.append({"block": block, "set_row": set_row, "id": ident,
                             "bank": h["bank"], "bank_row": h["bank_row"], "cos": h["cos"]})
        od.write_jsonl("hits.jsonl", hit_rows)
        od.write_array(
            "maxcos.f32",
            np.stack([res["per_bank"][n] for n in bank_names] + [res["best"]], axis=1),
            "float32",
        )
        od.note(
            f"columns of maxcos.f32, in order: {bank_names + ['all']}; rows are the blocks "
            f"concatenated in the order {[b['name'] for b in blocks]} with spans {spans}"
        )
        od.note(
            f"cos > {thr} of every block row against all {len(banks)} tier-B arrays "
            f"({summary['bank_rows_total']} rows): "
            + "; ".join(
                f"{n} {summary['blocks'][n]['all_banks']['n_above']} above, max "
                f"{summary['blocks'][n]['all_banks']['max']}"
                for n in summary["blocks"]
            )
        )
        if missing:
            od.note(f"UNREADABLE arrays, NOT scanned: {[m['bank'] for m in missing]}")
            od.status = "incomplete"
    assert not missing, (
        f"{len(missing)} of {len(BANKS)} tier-B arrays could not be read and were NOT scanned: "
        f"{[m['bank'] for m in missing]}. The results that were computed are in {out}."
    )
    return summary
