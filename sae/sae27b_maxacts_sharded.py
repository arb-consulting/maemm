"""Data-parallel max-activating-examples for a huge SAE (F=2^21) on Qwen3.6-27B layer 42 (torchrun, R ranks).

Each rank streams a disjoint 1/R of Ultra-FineWeb (same skip / split_dataset_by_node / BOS+ctx windows as
sae27b_gen_acts.py), runs the 27B model to layer 42 and the FULL bf16 encoder (21.5 GB) on its GPU, and keeps for EVERY
feature its top-N tokens: the window is the L=32 tokens ENDING AT the peak token (peak = last token of the window),
left-padded with pad_id when the document has < L tokens up to the peak (true length recorded). By default at most one
candidate per (feature, 512-token sequence) so the N examples are not N adjacent tokens (--no-dedupe-seq to disable).
The update is vectorised over the features touched in a batch (gather current top-N, concat candidates, topk, scatter).
Per rank output: maxacts_part_r{rank}.pt; sae27b_maxacts_merge.py takes the top-N across ranks.
dict2m port: the 27B is loaded truncated to layers 0..layer (online_gen.load_truncated_model, ~36 GB), the corpus stream is
online_gen.open_rank_stream (split_dataset_by_node THEN skip -> the SAME per-rank streams the online trainer consumed, so
the max-acts pass covers exactly the training span) and tokenisation runs in a background thread (online_gen.WindowProducer,
identical window dicts). The top-N machinery (TopNStore / topn_update / build_windows / encode_and_update) is unchanged.
"""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------------------------------------------
class TopNStore:
    """Per-feature top-N table on `device`. Values -1 == empty slot."""

    def __init__(self, F, N, L, device, pad_id=0):
        self.F, self.N, self.L, self.pad_id = F, N, L, pad_id
        self.vals = torch.full((F, N), -1.0, device=device, dtype=torch.float32)
        self.tok = torch.full((F, N, L), pad_id, device=device, dtype=torch.int32)
        self.len = torch.zeros((F, N), device=device, dtype=torch.int16)
        self.doc = torch.full((F, N), -1, device=device, dtype=torch.int64)
        self.pos = torch.full((F, N), -1, device=device, dtype=torch.int32)
        self.fire = torch.zeros(F, device=device, dtype=torch.int64)

    def slice(self, s, e):
        return (self.vals[s:e], self.tok[s:e], self.len[s:e], self.doc[s:e], self.pos[s:e])

    def to_dict(self):
        return {"max_acts": self.vals.cpu(), "max_tokens": self.tok.cpu(), "lengths": self.len.cpu(), "doc_ids": self.doc.cpu(),
                "positions": self.pos.cpu(), "fire_counts": self.fire.cpu(), "N": self.N, "L": self.L, "pad_id": self.pad_id}


@torch.no_grad()
def topn_update(top_vals, top_tok, top_len, top_doc, top_pos, feats, win_tok, win_len, win_doc, win_pos, dedupe_seq=True):
    """Vectorised top-N update for one feature chunk.
    top_*  : [Fc, N] / [Fc, N, L] slices of the store (modified in place through advanced-index assignment).
    feats  : [Fc, W, S] thresholded activations (0 = not firing) for W sequences of S tokens.
    win_*  : per-token window table for the batch, token index t = w*S + s: win_tok [W*S, L], win_len/doc/pos [W*S].
    dedupe_seq: one candidate per (feature, sequence) = the sequence's peak token; else every firing token."""
    Fc, W, S = feats.shape
    N, L = top_vals.shape[1], top_tok.shape[2]
    dev = feats.device
    if dedupe_seq:
        cvals, cpos = feats.amax(dim=2), feats.argmax(dim=2)                      # [Fc, W]
        ctok = torch.arange(W, device=dev).unsqueeze(0) * S + cpos                # [Fc, W] token index
    else:
        cvals = feats.reshape(Fc, W * S)
        ctok = torch.arange(W * S, device=dev).unsqueeze(0).expand(Fc, W * S)
    cvals = cvals.float()
    cvals = torch.where(cvals > 0, cvals, torch.full_like(cvals, -1.0))
    touched = (cvals > 0).any(dim=1).nonzero(as_tuple=True)[0]                     # [U]
    if touched.numel() == 0:
        return 0
    u = touched
    cv, ci = cvals[u], ctok[u]                                                     # [U, C]
    allv = torch.cat([top_vals[u], cv], dim=1)                                     # [U, N+C]
    newv, sel = allv.topk(N, dim=1)                                                # sorted desc
    from_old = sel < N
    old_idx = sel.clamp(max=N - 1)
    new_tidx = torch.gather(ci, 1, (sel - N).clamp(min=0))                         # [U, N] token index (valid where ~from_old)
    # tokens [U, N, L]
    old_tok = torch.gather(top_tok[u], 1, old_idx.unsqueeze(-1).expand(-1, -1, L))
    new_tok = win_tok[new_tidx]                                                    # [U, N, L]
    top_tok[u] = torch.where(from_old.unsqueeze(-1), old_tok, new_tok)
    for tab, wt in ((top_len, win_len), (top_doc, win_doc), (top_pos, win_pos)):
        old = torch.gather(tab[u], 1, old_idx)
        new = wt[new_tidx].to(tab.dtype)
        tab[u] = torch.where(from_old, old, new)
    top_vals[u] = newv
    return int(u.numel())


def build_windows(seqs, L, pad_id, device):
    """seqs: list of dicts {ids: [S] content tokens, prefix: up-to-(L-1) tokens preceding ids in the doc, doc: int, start: int}.
    Returns win_tok [W*S, L] int32 (window ENDING at each token, left-padded), win_len [W*S] int16, win_doc int64, win_pos int32."""
    W, S = len(seqs), len(seqs[0]["ids"])
    full = torch.full((W, L - 1 + S), pad_id, dtype=torch.int64)
    lens = torch.empty((W, S), dtype=torch.int16)
    docs = torch.empty((W,), dtype=torch.int64)
    starts = torch.empty((W,), dtype=torch.int64)
    for w, q in enumerate(seqs):
        pref = q["prefix"][-(L - 1):] if L > 1 else []
        if pref:
            full[w, L - 1 - len(pref):L - 1] = torch.tensor(pref, dtype=torch.int64)
        full[w, L - 1:] = torch.tensor(q["ids"], dtype=torch.int64)
        st = int(q["start"])
        lens[w] = torch.clamp(torch.arange(S) + st + 1, max=L).to(torch.int16)
        docs[w] = int(q["doc"]); starts[w] = st
    win = full.unfold(1, L, 1)                                                     # [W, S, L]
    win_tok = win.reshape(W * S, L).to(torch.int32).to(device)
    win_len = lens.reshape(-1).to(device)
    win_doc = docs.unsqueeze(1).expand(W, S).reshape(-1).to(device)
    win_pos = (starts.unsqueeze(1) + torch.arange(S).unsqueeze(0)).reshape(-1).to(torch.int32).to(device)
    return win_tok, win_len, win_doc, win_pos


@torch.no_grad()
def encode_and_update(acts_raw, seqs, store, W_enc, b_enc, b_dec, thr, feat_chunk, L, pad_id, norm_mult=10.0, dedupe_seq=True):
    """One batch: acts_raw [W*S, d] fp32 raw layer-42 residuals (BOS dropped) for the W sequences in `seqs`.
    Encodes in bf16 chunks of feat_chunk features as [Fc, T] = W_enc[c] @ (x - b_dec)^T + b_enc[c] (relu, > thr),
    zeroes tokens with ||x|| > norm_mult * median (outliers the SAE never trained on), accumulates fire counts and
    runs the vectorised top-N update. Returns the number of tokens processed."""
    T, d = acts_raw.shape
    Wb, S = len(seqs), len(seqs[0]["ids"])
    assert T == Wb * S
    F = W_enc.shape[0]
    bad = None
    if norm_mult and norm_mult > 0:
        nrm = acts_raw.norm(dim=-1)
        bad = nrm > norm_mult * nrm.median()
    xc = (acts_raw - b_dec).to(W_enc.dtype)
    win_tok, win_len, win_doc, win_pos = build_windows(seqs, L, pad_id, acts_raw.device)
    for s in range(0, F, feat_chunk):
        e = min(F, s + feat_chunk)
        pre = torch.addmm(b_enc[s:e].to(W_enc.dtype).unsqueeze(1), W_enc[s:e], xc.t())      # [Fc, T]
        pre = torch.relu_(pre)
        pre.masked_fill_(pre <= thr, 0)
        if bad is not None:
            pre.masked_fill_(bad.unsqueeze(0), 0)
        store.fire[s:e] += (pre > 0).sum(dim=1)
        topn_update(*store.slice(s, e), pre.view(e - s, Wb, S), win_tok, win_len, win_doc, win_pos, dedupe_seq=dedupe_seq)
    return T


def load_encoder(path, device, chunk=131072):
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    F, d = sd["encoder.weight"].shape
    W_enc = torch.empty(F, d, dtype=torch.bfloat16, device=device)
    for s in range(0, F, chunk):
        W_enc[s:s + chunk] = sd["encoder.weight"][s:s + chunk].to(device, dtype=torch.bfloat16)
    b_enc = sd["encoder.bias"].to(device, dtype=torch.bfloat16)
    b_dec = sd["b_dec"].float().to(device)
    thr = float(sd["threshold"].item())
    k = int(sd["k"].item()) if "k" in sd else -1
    return W_enc, b_enc, b_dec, thr, k, F, d


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True, help="merged ae.pt (bf16 encoder.weight [F,d])")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    ap.add_argument("--split", default="en")
    ap.add_argument("--dataset-skip", type=int, default=100_000, help="same as the acts_1b gen run (skip the reserved head)")
    ap.add_argument("--ctx-len", "-S", type=int, default=512, help="content tokens per model sequence (as gen_acts)")
    ap.add_argument("--window", "-L", type=int, default=32, help="tokens per stored window (ends at the peak)")
    ap.add_argument("--topn", "-N", type=int, default=5)
    ap.add_argument("--batch-seqs", "-W", type=int, default=16)
    ap.add_argument("--feat-chunk", type=int, default=131072)
    ap.add_argument("--max-tokens", type=int, default=1_000_000_000, help="GLOBAL token budget (split /R)")
    ap.add_argument("--norm-mult", type=float, default=10.0, help="zero features on tokens with ||x|| > mult*median (0=off)")
    ap.add_argument("--no-dedupe-seq", action="store_true")
    ap.add_argument("--d-model", type=int, default=5120, help="(named --d-model: torchrun eats --d as an ambiguous prefix)")
    ap.add_argument("--log-every-docs", type=int, default=2000)
    ap.add_argument("--save-every-min", type=float, default=60.0, help="periodic partial save (crash insurance)")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dev = f"cuda:{local_rank}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    import torch.distributed as dist
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(hours=3))

    def log(*a):
        print(f"[maxacts2m r{rank}] {time.strftime('%H:%M:%S')}", *a, flush=True)

    from transformers import AutoTokenizer
    from online_gen import load_truncated_model, LayerCapture, forward_layer, open_rank_stream, WindowProducer, BOS_FALLBACK
    os.makedirs(args.out_dir, exist_ok=True)
    per_rank_tokens = args.max_tokens // world
    S, L, N = args.ctx_len, args.window, args.topn

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else BOS_FALLBACK
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    model, layers, minfo = load_truncated_model(args.model, args.layer, dev, log=log)
    W_enc, b_enc, b_dec, thr, k, F, d = load_encoder(args.ae, dev, args.feat_chunk)
    assert d == args.d_model
    log(f"world={world} F={F} d={d} k={k} thr={thr:.4f} S={S} L={L} N={N} W={args.batch_seqs} budget/rank={per_rank_tokens:,} "
        f"pad_id={pad_id} bos={bos} enc_mem={W_enc.numel() * 2 / 2**30:.1f}GB")
    store = TopNStore(F, N, L, dev, pad_id=pad_id)
    cap = LayerCapture(layers, args.layer)

    ds = open_rank_stream(args.dataset, args.split, rank, world, args.dataset_skip, log=log)
    prod = WindowProducer(iter(ds), lambda t: tok(t, add_special_tokens=False, truncation=False)["input_ids"], S, args.batch_seqs,
                          rank=rank, world=world, prefix_len=L - 1, text_key="content")
    if world > 1:
        dist.barrier()

    seen = 0
    docs_used = 0
    t0 = time.time()
    t_save = time.time()
    n_chunks = (F + args.feat_chunk - 1) // args.feat_chunk
    sae_time = 0.0
    model_time = 0.0

    @torch.no_grad()
    def process(seqs):
        nonlocal seen, sae_time, model_time
        tm = time.time()
        ids = torch.tensor([[bos] + q["ids"] for q in seqs], device=dev)          # [W, S+1]
        a = forward_layer(model, cap, ids)[:, 1:, :].reshape(-1, d).float()         # [T, d] raw, BOS dropped
        if use_cuda:
            torch.cuda.synchronize()
        model_time += time.time() - tm
        ts = time.time()
        seen += encode_and_update(a, seqs, store, W_enc, b_enc, b_dec, thr, args.feat_chunk, L, pad_id,
                                  norm_mult=args.norm_mult, dedupe_seq=not args.no_dedupe_seq)
        if use_cuda:
            torch.cuda.synchronize()
        sae_time += time.time() - ts

    def save_partial(final=False):
        dct = store.to_dict()
        dct.update({"rank": rank, "world": world, "tokens_seen": seen, "docs_used": docs_used, "threshold": thr, "k": k,
                    "docs_iterated": prod.state()["docs_iterated"], "model_layers_kept": minfo.get("n_layers_kept"),
                    "F": F, "d": d, "S": S, "dataset": args.dataset, "split": args.split, "dataset_skip": args.dataset_skip,
                    "dedupe_seq": not args.no_dedupe_seq, "norm_mult": args.norm_mult, "final": final,
                    "doc_id_scheme": "doc_id = local_doc_index * world + rank = single-stream document index - dataset_skip "
                                     "(rank r streams single-stream docs dataset_skip + r + world*k)"})
        p = os.path.join(args.out_dir, f"maxacts_part_r{rank}.pt")
        torch.save(dct, p + ".tmp")
        os.replace(p + ".tmp", p)
        return p

    n_proc = 0
    last_log_docs = 0
    while seen < per_rank_tokens:
        seq_batch = prod.get()
        if seq_batch is None:
            log("corpus stream exhausted"); break
        if n_proc == 0:
            w = seq_batch[0]
            log(f"FIRST WINDOW doc_id={w['doc']} start={w['start']} text={tok.decode(w['ids'][:20])!r}")
        process(seq_batch)
        n_proc += 1
        st = prod.state()
        docs_used = st["docs_used"]
        if st["docs_iterated"] - last_log_docs >= args.log_every_docs:
            last_log_docs = st["docs_iterated"]
            el = time.time() - t0
            rate = seen / max(el, 1)
            eta = (per_rank_tokens - seen) / max(rate, 1) / 3600
            live = int((store.fire > 0).sum().item())
            log(f"docs~{st['docs_iterated']} used={docs_used} tokens={seen:,}/{per_rank_tokens:,} {rate:.0f} tok/s/rank "
                f"(model {model_time / max(el, 1) * 100:.0f}% sae {sae_time / max(el, 1) * 100:.0f}%) ETA {eta:.2f}h "
                f"live_so_far={live}/{F}")
        if (time.time() - t_save) / 60 >= args.save_every_min:
            save_partial(final=False)
            t_save = time.time()
            log(f"periodic partial saved (tokens={seen:,})")
    prod.close()
    cap.remove()
    p = save_partial(final=True)
    el = time.time() - t0
    live = int((store.fire > 0).sum().item())
    log(f"DONE tokens={seen:,} docs={docs_used} in {el / 3600:.2f}h ({seen / max(el, 1):.0f} tok/s/rank) live={live}/{F} -> {p}")
    if world > 1:
        tot = torch.tensor([float(seen)], device=dev)
        dist.all_reduce(tot)
        if rank == 0:
            log(f"ALL RANKS: {int(tot.item()):,} tokens total; {int(tot.item()) / max(el, 1):.0f} tok/s aggregate")
        dist.barrier()
        dist.destroy_process_group()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # see sae27b_train_sharded.main: skip finalisation (datasets streaming threads abort/deadlock there)


if __name__ == "__main__":
    main()
