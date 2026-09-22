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
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

from .draw_sae2m import _finish

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
    assert len(ids) >= N_FEATURES, f"only {len(ids)} features, need {N_FEATURES}"

    rng = np.random.default_rng(DRAW_SEED)
    pick = np.sort(rng.choice(len(ids), size=N_FEATURES, replace=False))
    drawn, peak = ids[pick], acts[pick]

    cuts = np.quantile(np.log10(np.maximum(peak, 1e-6)), [0.25, 0.5, 0.75])
    stratum = np.searchsorted(cuts, np.log10(np.maximum(peak, 1e-6)), side="right")
    side = np.where(rng.random(N_FEATURES) < FIT_FRACTION, "train", "test")

    peak_by_id = dict(zip(drawn.tolist(), peak.tolist()))
    set_name, out_dir, rows, vecs, meta = _finish(
        cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
        peak_by_id, "log10_pool_peak_act", f"pool_heldout/sae.parquet ({len(ids):,})",
        # `peak16` (the 16M-corpus peak) is None here: this draw reads the 1B pool's `act`, and
        # `corpus_peak_16m` is a different quantity on a different corpus. The slot was added to
        # `_finish` on this branch AFTER draw_sae131k was written against the 16-argument
        # signature, which bound `cuts` to `peak16` and raised a TypeError on the meta dict.
        False, None, None, cuts, {"pool": len(ids), "eligible": int(len(ids))})
    for r in rows:
        # NOT `r["sae_key"] = "l42-1b"`, which is what this loop used to do. `_finish` already
        # stamps the FULL config key, and `common.sae_rows_of` matches on the full key -- a bare
        # name matches nothing, and because the row IS keyed (just wrongly) the unkeyed branch's
        # loud assert never fires: `scan` would simply run with n_feat = 0.
        r["heldout_note"] = HELDOUT_NOTE
    return set_name, out_dir, rows, vecs, meta


def run(cfg, args):
    set_name, out_dir, rows, vecs, meta = build(cfg, args)
    with C.outdir(out_dir, args, inputs={"sae": args.get("sae"), "pool": POOL}) as od:
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
        od.note(f"{meta['n_fit']} train / {meta['n_report']} test, seed {DRAW_SEED}; "
                f"BOTH halves are unseen -- this splits our analysis, not the training")
        od.note(f"strata: {meta['stratum_stat']} quartiles over the drawn set, recorded "
                f"not sampled, cuts {meta['cuts']}")
        od.note("HELD-OUT PROVENANCE IS WEAKER THAN THE 2M SET'S: " + HELDOUT_NOTE)
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_sae131k", "set": set_name, **meta}
