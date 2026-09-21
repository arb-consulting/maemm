"""Product `scan`: pass B over the corpus -- the corpus-retrieval baseline and the SAE examples.

    <root>/base/<base>/scan/<set>/       topk.jsonl, quantiles.f16 [N, n_sizes, 5]
    <root>/base/<base>/sae/<sae>/examples/  <feature>.jsonl for every TESTED feature, _random256.jsonl

Same windows as pass A (`common.windows_of`, 64 tokens every 16, never crossing documents, sink
prepended and dropped), same stored document order, so the per-size snapshots are the same nested
subsets. Cosine is UNCENTRED and in fp32 over every non-sink position, with NO norm filter
(checklist item 4 and the scorer's divergence note in common.score_tokens) -- while realact targets
are `unit(act - stats/mu)`, centred ONCE at construction. That asymmetry is Celeste's and is kept.

Masking (checklist item 53): for a realact target, the windows of ITS OWN document that overlap the
shown span [p-L+1, p] are excluded from both the top-k and the quantiles. Exact or near-duplicate
documents elsewhere in the corpus are NOT masked -- that is a real property of a corpus baseline
and is reported rather than engineered away.
"""

from __future__ import annotations

import time

import numpy as np

import precompute.common as C

TOPK = 64
N_BINS = 512  # cos histogram over [-1, 1]; bin width 1/256, so a quantile is exact to 1/256
QUANTILES = [0.50, 0.90, 0.99, 0.999, 0.9999]
TGT_CHUNK = 512  # targets per histogram sub-step, to bound the int64 index transient
SAE_TOP = 128  # top windows kept per tested feature
SAE_PER_BIN = 32  # windows sampled from each of 4 activation bins of (0, max_act]
SAE_RANDOM = 256  # shared random windows
SENTINEL = -2.0  # cos of a masked / padded position: below every real cosine


class _Heap:
    """Running top-k over windows for [n_rows] parallel streams, merged on the GPU each batch.

    eval/corpus_retrieval.py:Heaps is the working example: with n_rows in the thousands and a few
    hundred candidates per batch, a full sort of [n_rows, k + batch] is microseconds next to the
    forward and needs no python-side heap.
    """

    def __init__(self, n_rows, k, device, payload_shape=()):
        import torch

        self.k = k
        self.val = torch.full((n_rows, k), -float("inf"), device=device)
        self.win = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.arg = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.payload = (
            torch.zeros((n_rows, k, *payload_shape), dtype=torch.float16, device=device)
            if payload_shape
            else None
        )

    def push(self, val, win, arg, payload=None):
        """val/arg: [n_rows, b]; win: [b]; payload: [n_rows, b, *shape]."""
        import torch

        n = val.shape[0]
        cat_v = torch.cat([self.val, val], 1)
        cat_w = torch.cat([self.win, win.unsqueeze(0).expand(n, -1)], 1)
        cat_a = torch.cat([self.arg, arg], 1)
        v, i = cat_v.topk(self.k, dim=1)
        self.val, self.win, self.arg = v, cat_w.gather(1, i), cat_a.gather(1, i)
        if self.payload is not None:
            cat_p = torch.cat([self.payload, payload], 1)
            self.payload = cat_p.gather(1, i.unsqueeze(-1).expand(-1, -1, cat_p.shape[-1]))


class _KeyReservoir:
    """Uniform sample of `k` items per row, by smallest random key (A-Res with equal weights).

    Equivalent to reservoir sampling and fully vectorised: every candidate gets one U(0,1) key from
    a seeded torch generator and the k smallest keys survive, so the result depends on the seed and
    the batch order only, never on where a flush happened to fall.
    """

    def __init__(self, n_rows, k, device, payload_shape=()):
        import torch

        self.k = k
        self.key = torch.full((n_rows, k), float("inf"), device=device)
        self.win = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.arg = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.val = torch.zeros((n_rows, k), device=device)
        self.payload = (
            torch.zeros((n_rows, k, *payload_shape), dtype=torch.float16, device=device)
            if payload_shape
            else None
        )

    def push(self, key, win, arg, val, payload=None):
        import torch

        n = key.shape[0]
        cat_k = torch.cat([self.key, key], 1)
        cat_w = torch.cat([self.win, win.unsqueeze(0).expand(n, -1)], 1)
        cat_a = torch.cat([self.arg, arg], 1)
        cat_v = torch.cat([self.val, val], 1)
        kk, i = cat_k.topk(self.k, dim=1, largest=False)
        self.key, self.win, self.arg, self.val = (
            kk,
            cat_w.gather(1, i),
            cat_a.gather(1, i),
            cat_v.gather(1, i),
        )
        if self.payload is not None:
            cat_p = torch.cat([self.payload, payload], 1)
            self.payload = cat_p.gather(1, i.unsqueeze(-1).expand(-1, -1, cat_p.shape[-1]))


def _load_targets(cfg, args, notes=None):
    """(ids rows, V [N, d] unit fp32 on the gpu, realact mask tables).

    `scan` has no `--maemm` in scope at all, so the centring convention has to be told to it
    (`--mu <file>`) or taken from the set's own stored contract -- see common.mu_for. On a
    legacy `storage: unit` set that resolves to the mean the set was built with, which reproduces
    every corpus-search number measured between 2026-09-16 and 2026-09-21 exactly.
    """
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    d = cfg["bases"][base]["d"]
    hdir = C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{hdir}/ids.jsonl")
    n = len(rows)
    mu, _ = C.mu_for(cfg, base, hdir, args, "", root, notes)
    v = C.dirs_for(cfg, base, hdir, mu, root, notes)
    assert v.shape == (n, d), f"{hdir}: dirs_for returned {v.shape} for {n} rows"
    v = torch.nn.functional.normalize(torch.from_numpy(np.asarray(v)).cuda(), dim=-1)
    doc = torch.full((n,), -1, dtype=torch.int64)
    lo = torch.zeros(n, dtype=torch.int64)
    hi = torch.zeros(n, dtype=torch.int64)
    for i, r in enumerate(rows):
        assert r["row"] == i, f"ids.jsonl row {i} says row={r['row']}"
        if r["family"] == "realact":
            doc[i], lo[i], hi[i] = r["doc"], r["p"] - r["L"] + 1, r["p"]
    return rows, v, (doc.cuda(), lo.cuda(), hi.cuda())


def _forward(model, read_layer, rows, sink, pad_id):
    """(h [B, T, d] fp32, keep [B, T]) -- the same batch shape stats.py pass A uses."""
    import torch

    width = 1 + max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    am = np.zeros((len(rows), width), dtype=np.int64)
    ids[:, 0] = sink
    am[:, 0] = 1
    for i, r in enumerate(rows):
        ids[i, 1 : 1 + len(r)] = r
        am[i, 1 : 1 + len(r)] = 1
    batch = {
        "input_ids": torch.from_numpy(ids).cuda(),
        "attention_mask": torch.from_numpy(am).cuda(),
    }
    h, mask = C.read_resid(model, read_layer, batch, pool="all")
    keep = mask.clone()
    keep[:, 0] = False
    return h, keep


def run(cfg, args):
    import os

    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product scan needs --base"
    spec = cfg["bases"][base]
    read_layer, d = spec["read_layer"], spec["d"]
    batch_rows = int(args.get("batch") or 256)
    # ONE --sae syntax in the whole CLI: common.sae_key_for, which takes a full `<base>/<name>`
    # key and refuses a bare name. This file and stats.py each carried an inline copy that DID
    # accept a bare `sae2m`, so the same flag meant two things depending on the product.
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")

    # --corpus-name selects corpora/<name>/ instead of corpus/: a different size
    # ladder or window geometry is a different corpus, never an edit of one.
    corpus_name = args.get("corpus_name") or ""
    C.assert_corpus_geometry(cfg, corpus_name)  # H7: refuse a corpus we would cut at the wrong width
    toks, docs = C.load_corpus(base, root, corpus_name)
    sizes = C.corpus_sizes(docs)
    cen_notes: list[str] = []
    rows, v, (t_doc, t_lo, t_hi) = _load_targets(cfg, args, notes=cen_notes)
    n = len(rows)
    # Filtered on the ROW's own sae_key, not on the family label: a set carrying two dictionaries
    # under `family: sae` would otherwise have the other dictionary's feature ids looked up in this
    # encoder, silently (common.sae_rows_of).
    sae_sel = C.sae_rows_of(
        rows, sae_key, declared=C.declared_sae_key(cfg, C.heldout_dir(base, set_name, root), root),
        where=C.heldout_dir(base, set_name, root),
    )
    tested = [int(r["id"]) for r in sae_sel]
    tested_row = [r["row"] for r in sae_sel]
    n_feat = len(tested)

    out_scan = C.scan_dir(base, set_name, root, corpus_name)
    # KEYED BY SET (B9, 2026-09-21). `examples/` used to be keyed by SAE alone, so a second scan of
    # the same dictionary against a different held-out set refused without --force and DESTROYED
    # the first set's examples with it -- and the eval plan runs three scans on sae2m. The scan
    # half was already set-keyed (C.scan_dir); this is the other half.
    out_ex = C.sae_examples_dir(sae_key, set_name, root, write=True, corpus_name=corpus_name)
    for p in (out_scan, out_ex):
        assert args.get("force") or not os.path.exists(p), (
            f"{p} already exists; refusing to overwrite without --force"
        )

    model, tok = C.load_base(cfg, base)
    # ENCODER ONLY (D3): everything below reads b_dec, W_enc, b_enc and threshold -- the tested
    # columns at :200-201 and the gate. W_dec is 43 GB in fp32 at 2^21 features, which is the
    # difference between fitting an H200 beside the 27B and not. stats.py:379 already had this.
    sae = C.load_sae(
        C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32, need_decoder=False
    )
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    w_enc = sae.W_enc[:, torch.as_tensor(tested, device="cuda")].contiguous() if n_feat else None
    b_enc = sae.b_enc[torch.as_tensor(tested, device="cuda")] if n_feat else None
    peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (sae.d_sae,))
    peak_t = torch.from_numpy(peak[tested].astype(np.float32)).cuda() if n_feat else None
    print(
        f"[scan] {n} targets ({n_feat} tested sae features), {len(docs)} docs, sizes {sizes}",
        flush=True,
    )

    heap = _Heap(n, TOPK, "cuda")
    hist = torch.zeros((n, N_BINS), dtype=torch.int64, device="cuda")
    snap_top, snap_hist = [], []
    f_top = _Heap(n_feat, SAE_TOP, "cuda", payload_shape=(C.SCAN_BLOCK,)) if n_feat else None
    f_bins = (
        [_KeyReservoir(n_feat, SAE_PER_BIN, "cuda", payload_shape=(C.SCAN_BLOCK,)) for _ in range(4)]
        if n_feat
        else []
    )
    r_res = _KeyReservoir(1, SAE_RANDOM, "cuda", payload_shape=(n_feat,)) if n_feat else None
    gen = torch.Generator(device="cuda").manual_seed(int(cfg["heldout"][set_name]["seed"]))

    win_doc: list = []
    win_start: list = []
    win_len: list = []
    buf, buf_meta, w_global = [], [], 0
    t0, fwd_tok, done_tokens = time.time(), 0, 0

    def flush():
        nonlocal fwd_tok, w_global
        if not buf:
            return
        h, keep = _forward(model, read_layer, buf, sink, pad_id)
        b, t = keep.shape
        wd = torch.tensor([m[0] for m in buf_meta], device="cuda")
        ws = torch.tensor([m[1] for m in buf_meta], device="cuda")
        wl = torch.tensor([len(r) for r in buf], device="cuda")
        wid = torch.arange(w_global, w_global + b, device="cuda")

        cos = torch.nn.functional.normalize(h, dim=-1) @ v.T  # [B, T, N] fp32, uncentred
        cos = cos.masked_fill(~keep.unsqueeze(-1), SENTINEL)
        # realact self-match mask: this window's document and span overlap the target's own span
        mask = (
            (wd.unsqueeze(1) == t_doc.unsqueeze(0))
            & (ws.unsqueeze(1) <= t_hi.unsqueeze(0))
            & ((ws + wl - 1).unsqueeze(1) >= t_lo.unsqueeze(0))
        )  # [B, N]
        cos = cos.masked_fill(mask.unsqueeze(1), SENTINEL)
        best, arg = cos.max(dim=1)  # [B, N]
        heap.push(best.T.contiguous(), wid, arg.T.contiguous())
        for c0 in range(0, n, TGT_CHUNK):
            c1 = min(c0 + TGT_CHUNK, n)
            sub = cos[:, :, c0:c1]
            bins = ((sub + 1.0) * (N_BINS / 2)).long().clamp_(0, N_BINS - 1)
            bins = torch.where(sub > -1.5, bins, torch.full_like(bins, N_BINS))  # sentinel bin
            off = torch.arange(c1 - c0, device="cuda") * (N_BINS + 1)
            flat = (bins + off).reshape(-1)
            cnt = torch.bincount(flat, minlength=(c1 - c0) * (N_BINS + 1))
            hist[c0:c1] += cnt.reshape(c1 - c0, N_BINS + 1)[:, :N_BINS]

        if n_feat:
            a = torch.relu((h - sae.b_dec) @ w_enc + b_enc)  # [B, T, n_feat] pre-gate
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            amax, aarg = a.max(dim=1)  # [B, n_feat]
            pay = torch.zeros((b, C.SCAN_BLOCK, n_feat), dtype=torch.float16, device="cuda")
            pay[:, : min(t - 1, C.SCAN_BLOCK)] = a[:, 1 : 1 + C.SCAN_BLOCK].to(torch.float16)
            pay = pay.permute(2, 0, 1).contiguous()  # [n_feat, B, 64]
            f_top.push(amax.T.contiguous(), wid, aarg.T.contiguous(), pay)
            q = torch.clamp(torch.ceil(amax / peak_t.clamp(min=1e-6) * 4).long() - 1, 0, 3)
            key = torch.rand((b, n_feat), generator=gen, device="cuda")
            for qi in range(4):
                sel = (q == qi) & (amax > 0)
                f_bins[qi].push(
                    torch.where(sel, key, torch.full_like(key, float("inf"))).T.contiguous(),
                    wid,
                    aarg.T.contiguous(),
                    amax.T.contiguous(),
                    pay,
                )
            rkey = torch.rand((1, b), generator=gen, device="cuda")
            r_res.push(
                rkey,
                wid,
                torch.zeros((1, b), dtype=torch.int64, device="cuda"),
                torch.zeros((1, b), device="cuda"),
                amax.unsqueeze(0).to(torch.float16),
            )

        win_doc.append(np.asarray([m[0] for m in buf_meta], dtype=np.int32))
        win_start.append(np.asarray([m[1] for m in buf_meta], dtype=np.int32))
        win_len.append(np.asarray([len(r) for r in buf], dtype=np.int32))
        fwd_tok += int(keep.numel())
        w_global += b
        buf.clear()
        buf_meta.clear()

    si = 0
    for r in docs:
        while sizes[si] < r["size_tag"]:
            flush()
            snap_top.append((heap.val.cpu().numpy(), heap.win.cpu().numpy(), heap.arg.cpu().numpy()))
            snap_hist.append(hist.cpu().numpy())
            print(
                f"[scan] size {sizes[si]}M snapshot at {done_tokens} tokens, {time.time() - t0:.0f}s",
                flush=True,
            )
            si += 1
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, ln in C.windows_of(r["len"]):
            buf.append(ids[s : s + ln])
            buf_meta.append((r["doc"], s))
            if len(buf) >= batch_rows:
                flush()
        done_tokens += r["len"]
        if r["doc"] % 5000 == 0 and r["doc"]:
            el = time.time() - t0
            print(
                f"[scan] doc {r['doc']}/{len(docs)} {done_tokens / 1e6:.2f}M tokens | "
                f"{fwd_tok / max(el, 1e-9):.0f} fwd tok/s | {done_tokens / max(el, 1e-9):.0f} corpus tok/s",
                flush=True,
            )
    flush()
    assert si == len(sizes) - 1, f"ended at size index {si} of {len(sizes)}"
    snap_top.append((heap.val.cpu().numpy(), heap.win.cpu().numpy(), heap.arg.cpu().numpy()))
    snap_hist.append(hist.cpu().numpy())
    elapsed = time.time() - t0
    wdoc = np.concatenate(win_doc)
    wstart = np.concatenate(win_start)
    wlen = np.concatenate(win_len)
    assert len(wdoc) == w_global, f"window table {len(wdoc)} != {w_global} forwarded windows"

    inputs = {
        "corpus": C.corpus_dir(base, root, corpus_name),
        "heldout": C.heldout_dir(base, set_name, root),
        "targets": n,
        "windows": int(w_global),
        "sizes": sizes,
    }
    with C.outdir(out_scan, args, inputs=inputs) as od:
        C.note_convention(od, cen_notes)
        lines, n_dropped = [], 0
        for si_, size in enumerate(sizes):
            val, win, arg = snap_top[si_]
            for i in range(n):
                top = []
                for j in range(TOPK):
                    if val[i, j] <= -1.5:  # fewer than TOPK unmasked windows at this size
                        n_dropped += 1
                        continue
                    w = int(win[i, j])
                    top.append([int(wdoc[w]), int(wstart[w]), int(arg[i, j] - 1), round(float(val[i, j]), 5)])
                lines.append({"row": i, "family": rows[i]["family"], "size": size, "top": top})
        od.write_jsonl("topk.jsonl", lines)
        q = np.stack([C.quantiles_from_hist(hs, QUANTILES) for hs in snap_hist], axis=1)  # [N, sizes, 5]
        od.write_array("quantiles.f16", q, "float16")
        od.note(
            f"topk.jsonl: one line per (target row, corpus size) with up to {TOPK} entries "
            "[doc, window start, argmax position WITHIN the window (0-based, sink already "
            "dropped), cos]; the running top-k is snapshotted at every size boundary"
        )
        od.note(
            f"quantiles.f16 is [N={n}, n_sizes={len(sizes)}, {len(QUANTILES)}] for q="
            f"{QUANTILES}, from a {N_BINS}-bin histogram of cos on [-1, 1] (bin width "
            f"{2 / N_BINS}), so each value is the bin upper edge and exact to {2 / N_BINS}"
        )
        od.note(
            "cosine is UNCENTRED, fp32, over every non-sink position of every window, with no norm "
            f"filter; window geometry {C.SCAN_BLOCK}/{C.SCAN_STRIDE} (common.windows_of), "
            f"{w_global} windows over {done_tokens} corpus tokens"
        )
        od.note(
            "realact targets mask the windows of their OWN document overlapping [p-L+1, p]; exact "
            "and near-duplicate documents elsewhere in the corpus are NOT masked"
        )
        if n_dropped:
            od.note(f"{n_dropped} top-k slots were empty (masked or too few windows) and omitted")
        od.note(
            f"throughput: {fwd_tok / elapsed:.0f} forwarded tok/s, {done_tokens / elapsed:.0f} "
            f"corpus tok/s over {elapsed:.0f}s"
        )

    ex_rows = 0
    with C.outdir(out_ex, args, inputs={**inputs, "sae": sae_key, "tested": n_feat}) as od:
        C.note_convention(od, cen_notes)
        if n_feat:
            tv, tw, ta, tp = (
                f_top.val.cpu().numpy(),
                f_top.win.cpu().numpy(),
                f_top.arg.cpu().numpy(),
                f_top.payload.cpu().numpy(),
            )
            bins = [
                (
                    b.val.cpu().numpy(),
                    b.win.cpu().numpy(),
                    b.arg.cpu().numpy(),
                    b.payload.cpu().numpy(),
                    b.key.cpu().numpy(),
                )
                for b in f_bins
            ]
            nbytes = 0
            for fi, feat in enumerate(tested):
                recs = []
                for j in range(SAE_TOP):
                    if not np.isfinite(tv[fi, j]) or tv[fi, j] <= 0:
                        continue
                    recs.append(
                        _ex(
                            tested_row[fi],
                            "top",
                            tv[fi, j],
                            tw[fi, j],
                            ta[fi, j],
                            tp[fi, j],
                            wdoc,
                            wstart,
                            wlen,
                        )
                    )
                for qi, (bv, bw, ba, bp, bk) in enumerate(bins):
                    for j in range(SAE_PER_BIN):
                        if not np.isfinite(bk[fi, j]):
                            continue
                        recs.append(
                            _ex(
                                tested_row[fi],
                                f"q{qi}",
                                bv[fi, j],
                                bw[fi, j],
                                ba[fi, j],
                                bp[fi, j],
                                wdoc,
                                wstart,
                                wlen,
                            )
                        )
                path = od.file(f"{feat}.jsonl")
                C.write_jsonl(path, recs)
                nbytes += path.stat().st_size
                ex_rows += len(recs)
            rw, rp, rk = (
                r_res.win.cpu().numpy(),
                r_res.payload.cpu().numpy(),
                r_res.key.cpu().numpy(),
            )
            rrecs = [
                {
                    "window": int(rw[0, j]),
                    "doc": int(wdoc[rw[0, j]]),
                    "start": int(wstart[rw[0, j]]),
                    "len": int(wlen[rw[0, j]]),
                    "max_act": [round(float(x), 4) for x in rp[0, j]],
                }
                for j in range(SAE_RANDOM)
                if np.isfinite(rk[0, j])
            ]
            path = od.file("_random256.jsonl")
            C.write_jsonl(path, rrecs)
            od.index["examples"] = {
                "kind": "jsonl",
                "rows": ex_rows + len(rrecs),
                "bytes": nbytes + path.stat().st_size,
            }
            od.write_json("tested.json", {"features": tested, "rows": tested_row, "sae": sae_key})
        od.note(
            f"one <feature>.jsonl per TESTED feature ({n_feat} of {sae.d_sae}), each with the top "
            f"{SAE_TOP} windows by activation plus {SAE_PER_BIN} windows sampled from each of 4 "
            "equal-width activation bins of (0, max_act] (max_act from the pass-A stats)"
        )
        od.note(
            "`acts` is the per-token pre-gate activation of that window, f16-rounded, in window "
            "order with the sink dropped and truncated to the window's `len`; `argmax` is the "
            "position of the maximum within it; the window's TEXT is recoverable from "
            "corpus/tokens.i32 at (doc, start, len)"
        )
        od.note(
            f"_random256.jsonl: {SAE_RANDOM} windows sampled uniformly over ALL {w_global} windows "
            "(one shared negative pool), each carrying the per-tested-feature MAX activation in the "
            "column order of tested.json -- per-token activations are not stored for these"
        )
        od.note("examples are taken over the FULL corpus only; they are not snapshotted per size")
        od.note(
            "sampling: smallest-random-key reservoir (equivalent to reservoir sampling), torch "
            f"generator seeded {cfg['heldout'][set_name]['seed']}"
        )

    return {
        "scan": out_scan,
        "examples": out_ex,
        "windows": int(w_global),
        "topk_rows": len(lines),
        "example_rows": ex_rows,
        "fwd_tok_per_s": round(fwd_tok / elapsed, 1),
        "corpus_tok_per_s": round(done_tokens / elapsed, 1),
        "seconds": round(elapsed, 1),
    }


def _ex(row, kind, val, win, arg, acts, wdoc, wstart, wlen):
    """One example record. `acts` is TRUNCATED to the window's real length: the payload buffer is
    64 wide, and leaving the tail in would be indistinguishable from a genuinely inactive token."""
    w = int(win)
    ln = int(wlen[w])
    return {
        "row": row,
        "kind": kind,
        "window": w,
        "doc": int(wdoc[w]),
        "start": int(wstart[w]),
        "len": ln,
        "max_act": round(float(val), 4),
        "argmax": int(arg) - 1,
        "acts": [round(float(x), 4) for x in acts[:ln]],
    }
