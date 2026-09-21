"""Product `stats`: pass A over the corpus. Target-INDEPENDENT, so it runs once per base and is
never invalidated by a new held-out set.

Writes two directories under `<root>/base/<base>/`:

  stats/     mu.f32 [d], mu_by_size.f32 [n_sizes, d], resid_norm_quantiles.json
  sae/<sae>/ fire_counts.i64 [F, n_sizes, 2], max_act.f16 [F], mean_when_active.f16 [F], sizes.json

`stats/mu.f32` is THE centring mean of the whole pipeline (README "Methods"): realact targets are
centred on it once, at construction, and nothing else is centred at all. A corpus ACTIVATION store
(`acts_1m/`) used to be written here and was dropped on 2026-09-15 (Tomáš): nothing downstream read
it, and at full scale it is ~8-10 GB of volume per base.

Geometry (checklist item 57, the one place it is implemented -- scan.py imports the same helper):
64-token windows starting every 16 tokens inside a document, never crossing documents;
`common.windows_of` states the exact rule for short documents and the final partial window. Every
window is forwarded as [sink] + window and position 0 is dropped, matching the scorer
(eval/eval_universal.py:139-143). The clean base is used throughout; nothing is injected.

Documents are visited in STORED (permuted) order, so the nested sizes 1/2/4/8/16M are crossed once
each and the per-size statistics are cumulative snapshots taken at the crossings.
"""

from __future__ import annotations

import os
import time

import precompute.common as C

# Column chunk for the SAE encoder matmul: [P, d] @ [d, 32768] fp32 is ~2 GB at P = 16k, which
# bounds the transient without making the matmul latency-bound.
SAE_COLS = 32768
# Every 16th window is ALSO forwarded through all layers for the residual-norm quantiles. The
# truncated forward stops at the read layer, so this is the only place the layers above it are
# computed; 1-in-16 keeps that at ~8% of the pass.
NORM_WINDOW_STRIDE = 16
# ... and of those windows' positions only a sample is kept, so the quantile arrays stay in RAM.
NORM_TARGET_SAMPLES = 1_000_000
NORM_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]


class _StopFull(Exception):
    """Raised by the last block's hook: nothing above it (final norm, lm_head) is needed."""


def _pad_batch(rows, sink, pad_id, device):
    """[sink] + window, right-padded to the longest row. Returns the model kwargs and the row lens.

    Right padding is safe for every causal architecture (including the 27B's GatedDeltaNet layers,
    where state flows strictly forward) and matches the scorer's padding_side='right'.
    """
    import numpy as np
    import torch

    n = len(rows)
    width = 1 + max(len(r) for r in rows)
    ids = np.full((n, width), pad_id, dtype=np.int64)
    am = np.zeros((n, width), dtype=np.int64)
    ids[:, 0] = sink
    am[:, 0] = 1
    for i, r in enumerate(rows):
        ids[i, 1 : 1 + len(r)] = r
        am[i, 1 : 1 + len(r)] = 1
    batch = {
        "input_ids": torch.from_numpy(ids).to(device),
        "attention_mask": torch.from_numpy(am).to(device),
    }
    return batch, [len(r) for r in rows]


def _read_batch(model, read_layer, rows, sink, pad_id, device):
    """(h [B, T, d] fp32, keep [B, T]) at the read layer; `keep` excludes the sink and the padding."""
    batch, _ = _pad_batch(rows, sink, pad_id, device)
    h, mask = C.read_resid(model, read_layer, batch, pool="all")
    keep = mask.clone()
    keep[:, 0] = False  # the sink is never a scanned position
    return h, keep


def _all_layer_norms(model, n_layers, rows, sink, pad_id, device):
    """Residual norms [n_layers, B, T] of every block OUTPUT for one batch, plus `keep`."""
    import torch

    batch, _ = _pad_batch(rows, sink, pad_id, device)
    got: dict[int, object] = {}

    def mk(i):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            got[i] = t.float().norm(dim=-1)
            if i == n_layers - 1:
                raise _StopFull  # control flow, not an error: stop before the lm_head

        return hook

    handles = [C.get_layer(model, i).register_forward_hook(mk(i)) for i in range(n_layers)]
    try:
        with torch.no_grad():
            model(**batch)
    except _StopFull:
        pass  # the only way out of a deliberately aborted forward
    finally:
        for h in handles:
            h.remove()
    assert len(got) == n_layers, f"only {len(got)} of {n_layers} block hooks fired"
    keep = batch["attention_mask"].bool().clone()
    keep[:, 0] = False
    return torch.stack([got[i] for i in range(n_layers)]), keep


class _SaeCounters:
    """Running per-feature corpus statistics, accumulated on the GPU in column chunks."""

    def __init__(self, sae, device):
        import torch

        self.sae = sae
        self.f = sae.d_sae
        self.fires0 = torch.zeros(self.f, dtype=torch.int64, device=device)
        self.fires_g = torch.zeros(self.f, dtype=torch.int64, device=device)
        self.act_sum = torch.zeros(self.f, dtype=torch.float64, device=device)
        self.max_act = torch.zeros(self.f, dtype=torch.float32, device=device)
        self.positions = 0

    def add(self, h_flat):
        import torch

        x = h_flat - self.sae.b_dec
        for c0 in range(0, self.f, SAE_COLS):
            c1 = min(c0 + SAE_COLS, self.f)
            a = torch.relu(x @ self.sae.W_enc[:, c0:c1] + self.sae.b_enc[c0:c1])
            self.fires0[c0:c1] += (a > 0).sum(0)
            self.fires_g[c0:c1] += (a > self.sae.threshold).sum(0)
            # fp32 is safe for a per-batch sum (16k rows of O(10) values); the running total is f64.
            self.act_sum[c0:c1] += a.sum(0).double()
            self.max_act[c0:c1] = torch.maximum(self.max_act[c0:c1], a.max(0).values)
        self.positions += h_flat.shape[0]

    def snapshot(self):
        import numpy as np

        return np.stack([self.fires0.cpu().numpy(), self.fires_g.cpu().numpy()], axis=1)  # [F, 2] cumulative


def _pass_a(model, cfg, args, toks, docs, sizes, sink, pad_id, sae, od_stats, od_sae):
    """Phase 2: the 64/16 window scan -- mu, per-layer norm quantiles, SAE fire statistics."""
    import numpy as np
    import torch

    base = args["base"]
    spec = cfg["bases"][base]
    read_layer, d, n_layers = spec["read_layer"], spec["d"], spec["n_layers"]
    batch_rows = int(args.get("batch") or 256)

    mu_sum = torch.zeros(d, dtype=torch.float64, device="cuda")
    mu_n = 0
    mu_by_size = np.zeros((len(sizes), d), dtype=np.float32)
    fires_by_size = np.zeros((len(sizes), sae.d_sae, 2), dtype=np.int64)
    pos_by_size = [0] * len(sizes)
    tok_by_size = [0] * len(sizes)
    counters = _SaeCounters(sae, "cuda")

    # norm sampling: 1-in-16 windows through all layers, then a fixed-probability position sample
    total_tokens = sum(r["len"] for r in docs)
    expect = total_tokens * C.SCAN_BLOCK / (C.SCAN_STRIDE * NORM_WINDOW_STRIDE)
    p_keep = min(1.0, NORM_TARGET_SAMPLES / max(expect, 1.0))
    rng = np.random.default_rng(cfg["corpus"]["seed"])
    norm_samples: list[list] = [[] for _ in range(n_layers)]
    bos_samples: list[list] = [[] for _ in range(n_layers)]

    buf, buf_full, w_global = [], [], 0
    t0, fwd_tok, done_tokens = time.time(), 0, 0

    def flush_read():
        nonlocal fwd_tok, mu_n
        if not buf:
            return
        h, keep = _read_batch(model, read_layer, buf, sink, pad_id, "cuda")
        flat = h[keep]
        mu_sum.add_(flat.sum(0, dtype=torch.float64))
        mu_n += flat.shape[0]
        counters.add(flat)
        fwd_tok += int(keep.numel())
        buf.clear()

    def flush_full():
        if not buf_full:
            return
        norms, keep = _all_layer_norms(model, n_layers, buf_full, sink, pad_id, "cuda")
        body = norms[:, keep].cpu().numpy()  # [n_layers, P]
        bos = norms[:, :, 0].cpu().numpy()  # [n_layers, B]
        sel = rng.random(body.shape[1]) < p_keep
        for ell in range(n_layers):
            norm_samples[ell].append(body[ell][sel].astype(np.float32))
            bos_samples[ell].append(bos[ell].astype(np.float32))
        buf_full.clear()

    def snapshot(si):
        mu_by_size[si] = (mu_sum / max(mu_n, 1)).cpu().numpy().astype(np.float32)
        fires_by_size[si] = counters.snapshot()
        pos_by_size[si] = counters.positions
        tok_by_size[si] = done_tokens
        print(
            f"[stats] size {sizes[si]}M: {done_tokens} corpus tokens, {counters.positions} scanned "
            f"positions, {time.time() - t0:.0f}s",
            flush=True,
        )

    si = 0
    for r in docs:
        while sizes[si] < r["size_tag"]:  # a size boundary: flush so the snapshot is exact
            flush_read()
            flush_full()
            snapshot(si)
            si += 1
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, n in C.windows_of(r["len"]):
            buf.append(ids[s : s + n])
            if w_global % NORM_WINDOW_STRIDE == 0:
                buf_full.append(ids[s : s + n])
            w_global += 1
            if len(buf) >= batch_rows:
                flush_read()
            if len(buf_full) >= batch_rows:
                flush_full()
        done_tokens += r["len"]
        if r["doc"] % 5000 == 0 and r["doc"]:
            el = time.time() - t0
            print(
                f"[stats] doc {r['doc']}/{len(docs)} {done_tokens / 1e6:.2f}M corpus tokens "
                f"| {fwd_tok / max(el, 1e-9):.0f} fwd tok/s | {done_tokens / max(el, 1e-9):.0f} corpus tok/s",
                flush=True,
            )
    flush_read()
    flush_full()
    assert si == len(sizes) - 1, f"ended at size index {si} of {len(sizes)}: a size was never crossed"
    snapshot(si)

    elapsed = time.time() - t0
    fwd_rate = fwd_tok / max(elapsed, 1e-9)
    corpus_rate = total_tokens / max(elapsed, 1e-9)

    # ---- stats/ ----
    od_stats.write_array("mu.f32", mu_by_size[-1], "float32")
    od_stats.write_array("mu_by_size.f32", mu_by_size, "float32")
    qjson = {
        "quantiles": NORM_QUANTILES,
        "block_output": True,
        "window": {"block": C.SCAN_BLOCK, "stride": C.SCAN_STRIDE},
        "subsample": {
            "window_stride": NORM_WINDOW_STRIDE,
            "position_prob": p_keep,
            "seed": cfg["corpus"]["seed"],
        },
        "layers": [],
    }
    for ell in range(n_layers):
        body = np.concatenate(norm_samples[ell]) if norm_samples[ell] else np.zeros(0, np.float32)
        bos = np.concatenate(bos_samples[ell]) if bos_samples[ell] else np.zeros(0, np.float32)
        assert body.size > 0, f"layer {ell}: no residual-norm samples were collected"
        qjson["layers"].append(
            {
                "layer": ell,
                "n": int(body.size),
                "q": [float(x) for x in np.quantile(body.astype(np.float64), NORM_QUANTILES)],
                "mean": float(body.mean()),
                "bos_n": int(bos.size),
                "bos_q": [float(x) for x in np.quantile(bos.astype(np.float64), NORM_QUANTILES)],
            }
        )
    od_stats.write_json("resid_norm_quantiles.json", qjson)
    od_stats.note(
        f"mu.f32 is the read-layer ({read_layer}) mean over ALL scanned positions of the 64/16 "
        f"windows with the sink position excluded ({mu_n} positions at the largest size). It is "
        "THE centring mean of the pipeline (README 'Methods'): `targets` subtracts it once when it "
        "builds a realact direction and nothing else centres anything. heldout/<set>/mu_512.f32 is "
        "Celeste's 512-token no-sink window mean, kept as a DIAGNOSTIC only, with its cos and norm "
        "ratio against this vector printed in the held-out set's README."
    )
    od_stats.note(f"mu_by_size.f32 rows are the cumulative means at sizes {sizes} (millions of tokens)")
    od_stats.note(
        f"resid_norm_quantiles.json: block OUTPUT norms of every layer 0..{n_layers - 1} over a "
        f"1-in-{NORM_WINDOW_STRIDE} window subsample, positions kept with probability "
        f"{p_keep:.4g}; `bos_q` is the sink position's norm, excluded from `q` and from mu"
    )
    od_stats.note(f"throughput: {fwd_rate:.0f} forwarded tok/s, {corpus_rate:.0f} corpus tok/s")

    # ---- sae/<sae>/ ----
    mean_active = (counters.act_sum / counters.fires0.clamp(min=1).double()).cpu().numpy()
    mean_active[counters.fires0.cpu().numpy() == 0] = 0.0
    od_sae.write_array("fire_counts.i64", fires_by_size.transpose(1, 0, 2), "int64")
    od_sae.write_array("max_act.f16", counters.max_act.cpu().numpy(), "float16")
    od_sae.write_array("mean_when_active.f16", mean_active, "float16")
    od_sae.write_json(
        "sizes.json",
        {
            "sizes": sizes,
            "scanned_positions": pos_by_size,
            "corpus_tokens": tok_by_size,
            "threshold": sae.threshold,
            "d_sae": sae.d_sae,
            "read_layer": read_layer,
        },
    )
    od_sae.note(
        f"fire_counts.i64 is [F={sae.d_sae}, n_sizes={len(sizes)}, 2]; the last axis is "
        f"(positions with act > 0, positions with act > gate). Counts are CUMULATIVE over the "
        f"nested sizes {sizes} (millions of tokens)."
    )
    od_sae.note(
        f"gate = the checkpoint's learned BatchTopK `threshold` buffer = {sae.threshold:.6g} "
        "(eval/eval_universal.py:77-93; NOT the older raw act > 1.0 cut)"
    )
    od_sae.note(
        "activation = relu((h - b_dec) @ W_enc + b_enc), pre-gate, fp32, over the same 64/16 "
        "windows with the sink excluded; scanned positions per size are in sizes.json"
    )
    od_sae.note("mean_when_active is the mean of the activation over act > 0 (0 for dead features)")

    n_dead = int((counters.fires0 == 0).sum())
    print(
        f"[stats] pass A done: {counters.positions} positions, {n_dead} dead features, "
        f"{fwd_rate:.0f} fwd tok/s, {corpus_rate:.0f} corpus tok/s",
        flush=True,
    )
    return {
        "positions": counters.positions,
        "tokens": total_tokens,
        "dead_features": n_dead,
        "fwd_tok_per_s": round(fwd_rate, 1),
        "corpus_tok_per_s": round(corpus_rate, 1),
        "seconds": round(elapsed, 1),
        "mu_positions": mu_n,
    }


def run(cfg, args):
    import torch

    base, root = args["base"], args["root"]
    assert base, "product stats needs --base"
    spec = cfg["bases"][base]
    # ONE --sae syntax in the whole CLI: common.sae_key_for. The inline copy that lived here (and
    # in scan.py) accepted a bare `sae2m` while every other product's --sae required the full
    # `<base>/<name>` key -- two syntaxes for one flag, which is a thing a reader gets right once.
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")

    # --corpus-name selects corpora/<name>/ instead of corpus/: a different size
    # ladder or window geometry is a different corpus, never an edit of one.
    corpus_name = args.get("corpus_name") or ""
    toks, docs = C.load_corpus(base, root, corpus_name)
    sizes = C.corpus_sizes(docs)
    print(
        f"[stats] corpus {C.corpus_dir(base, root, corpus_name)}: {len(docs)} docs, "
        f"{len(toks)} tokens, sizes {sizes}",
        flush=True,
    )

    # Fail before the model load if either output is in the way.
    outs = [C.stats_dir(base, root), C.sae_dir(sae_key, root)]
    for p in outs:
        assert args.get("force") or not os.path.exists(p), (
            f"{p} already exists; refusing to overwrite without --force"
        )

    t_load = time.time()
    model, tok = C.load_base(cfg, base)
    load_s = time.time() - t_load
    print(f"[stats] base weights loaded in {load_s:.0f}s", flush=True)
    # The counters read b_dec, W_enc, b_enc and threshold only -- never W_dec, which is
    # 43 GB in fp32 at 2^21 features and OOMs an H200 beside the 27B.
    sae = C.load_sae(C.sae_path(cfg, sae_key), spec["d"], device="cuda",
                     dtype=torch.float32, need_decoder=False)
    print(f"[stats] sae {sae_key}: F={sae.d_sae} gate={sae.threshold:.4f}", flush=True)
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink

    inputs = {
        "corpus": C.corpus_dir(base, root, corpus_name),
        "corpus_tokens": int(len(toks)),
        "sizes": sizes,
        "read_layer": spec["read_layer"],
        "base_load_seconds": round(load_s, 1),
    }
    with (
        C.outdir(C.stats_dir(base, root), args, inputs=inputs) as od_stats,
        C.outdir(C.sae_dir(sae_key, root), args, inputs={**inputs, "sae": sae_key}) as od_sae,
    ):
        passa = _pass_a(model, cfg, args, toks, docs, sizes, sink, pad_id, sae, od_stats, od_sae)
    return {"pass_a": passa, "base_load_seconds": round(load_s, 1)}


# ---------------------------------------------------------------------------------------------
# product `mu_check` (CPU): our stats/mu.f32 against Celeste's archived whiten_mu
# ---------------------------------------------------------------------------------------------

# The two archived whiten_mu paths used to live here as the pipeline's only named-mu registry.
# Since 2026-09-21 a mean is a FILE PATH wherever one is named, and the archived one is
# `bases.<base>.whiten_mu` in config.yaml -- so this is the lookup those two diagnostics share and
# NOT a second list to keep in step. Kept as a function because `mu_diag` names it too.
def archive_mu_path(cfg: dict, base: str) -> str:
    """Absolute path of base's archived `whiten_mu`, from config.yaml `bases.<base>.whiten_mu`."""
    path = cfg["bases"][base].get("whiten_mu")
    assert path, (
        f"base {base!r} declares no archived `whiten_mu:` path in config.yaml; there is nothing to "
        f"compare our own stats/mu.f32 against on this base"
    )
    return C.resolve_mu_path(path, base, root="/")
# Below this the two means are NOT the same object and every centred number has to be re-read with
# that in mind. Reported, never acted on: which mean is right is Tomáš's call, not this script's.
MU_COS_FLOOR = 0.99
README_SECTION = "## mu_check"


def run_mu_check(cfg, args):
    """Cosine and norm ratio between `stats/mu.f32` and Celeste's archived `whiten_mu.npy`.

    Appends the numbers to the stats README under a `## mu_check` heading, replacing only a
    previous `## mu_check` block: the README's own provenance header (command, date, commit, the
    pass-A notes) is never rewritten, because this product did not produce it.
    """
    import numpy as np

    base, root = args["base"], args["root"]
    assert base, "product mu_check needs --base"
    d = cfg["bases"][base]["d"]

    ours = C.stats_mu(cfg, base, root)
    apath = archive_mu_path(cfg, base)
    assert os.path.exists(apath), (
        f"missing archived whiten_mu at {apath}; look under {cfg['modal']['archive']}/data/ for the "
        f"{base} tree and report where it actually is"
    )
    theirs = np.load(apath).astype(np.float64).reshape(-1)
    assert theirs.shape == (d,), f"{apath} is {theirs.shape}, expected [{d}] for base {base!r}"

    a = ours.astype(np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(theirs))
    cos = float(a @ theirs / max(na * nb, 1e-12))
    ratio = na / max(nb, 1e-12)
    rel = float(np.linalg.norm(a - theirs) / max(nb, 1e-12))
    line = f"cos={cos:.6f} ||ours||={na:.4f} ||theirs||={nb:.4f} ratio={ratio:.6f} rel_l2_diff={rel:.6f}"
    print(f"[mu_check] {base}: {line}", flush=True)
    if cos < MU_COS_FLOOR:
        print(
            f"[mu_check] REPORTED, not acted on: cos {cos:.6f} < {MU_COS_FLOOR} -- our 64/16-window "
            f"mean and Celeste's archived whiten_mu are not the same object on {base}",
            flush=True,
        )

    readme = f"{C.stats_dir(base, root)}/README.md"
    assert os.path.exists(readme), f"no {readme}: run `--product stats --base {base}` first"
    body = open(readme).read()
    head = body.split(f"\n{README_SECTION}\n")[0].rstrip("\n")
    block = [
        "",
        README_SECTION,
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        f"- command: `{' '.join(args.get('argv') or [])}`",
        f"- our `mu.f32` (64/16 windows, sink dropped, this directory) vs Celeste's `{apath}`",
        f"- cosine: **{cos:.6f}**",
        f"- norms: ours {na:.4f}, theirs {nb:.4f}, ratio **{ratio:.6f}**",
        f"- relative L2 difference `||ours - theirs|| / ||theirs||`: {rel:.6f}",
        f"- floor for 'the same object': cos >= {MU_COS_FLOOR}; "
        + ("PASSED" if cos >= MU_COS_FLOOR else "**BELOW THE FLOOR -- reported, not acted on**"),
        "",
        "Appended by the `mu_check` product; everything above this heading is the `stats` run's own",
        "provenance and is never rewritten.",
        "",
    ]
    with open(readme, "w") as fh:
        fh.write(head + "\n" + "\n".join(block))
    print(f"[mu_check] appended to {readme}", flush=True)
    return {
        "base": base,
        "ours": f"{C.stats_dir(base, root)}/mu.f32",
        "archive": apath,
        "cos": round(cos, 6),
        "norm_ours": round(na, 4),
        "norm_archive": round(nb, 4),
        "norm_ratio": round(ratio, 6),
        "rel_l2_diff": round(rel, 6),
        "below_floor": cos < MU_COS_FLOOR,
    }
