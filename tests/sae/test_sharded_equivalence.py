"""CPU/gloo equivalence tests for the sharded 2M-SAE pipeline. Run:
    torchrun --standalone --nproc_per_node 2 scripts/tests_dict2m/test_sharded_equivalence.py
(a) sharded global BatchTopK == torch.topk on the concatenated pre-acts
(b) sharded loss + W_enc/W_dec/b_enc/b_dec grads == dense unsharded reference (incl. aux-k), to 1e-5
(c) merge_shards reproduces the (folded) unsharded parameters; merged raw-space encode == nf * normalised encode
(d) maxacts topn_update == brute-force numpy top-N (with and without per-sequence dedupe); build_windows semantics
(e) end-to-end: BatchSource gives identical batches on all ranks (both data modes), training runs, resume is exact
"""
import os, sys, json, tempfile, shutil, math
import numpy as np
import torch
import torch.distributed as dist

# the code under test, and the roots it imports from (tests live in tests/, not beside the code)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CODE = os.path.join(_REPO, "sae")
for _p in (_REPO, os.path.join(_REPO, "train"), _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
HERE = _CODE
from sae27b_train_sharded import (ShardedTrainer, select_global_batch_topk, all_reduce_, BatchSource,
                                  find_common_latest_step, world_size, rank_id)
from sae27b_merge_shards import merge
from sae27b_maxacts_sharded import TopNStore, topn_update, build_windows, encode_and_update
from sae27b_maxacts_merge import merge_parts

D, F, K, B = 16, 64, 4, 32
TOL = 1e-5


def gather_full(t, dim=0):
    W = world_size()
    parts = [torch.empty_like(t) for _ in range(W)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, dim=dim)


def bcast_obj(obj):
    lst = [obj]
    dist.broadcast_object_list(lst, src=0)
    return lst[0]


def check(cond, msg):
    ok = torch.tensor([1.0 if cond else 0.0])
    all_reduce_(ok, "min")
    if ok.item() < 1:
        raise AssertionError(f"[rank{rank_id()}] FAILED: {msg}")
    if rank_id() == 0:
        print(f"  PASS {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------
def test_a_topk():
    R, r = world_size(), rank_id()
    torch.manual_seed(1)
    pre_full = torch.relu(torch.randn(B, F))
    dist.broadcast(pre_full, src=0)
    fl = F // R
    pre_loc = pre_full[:, r * fl:(r + 1) * fl].contiguous()
    b_idx, f_idx, tau = select_global_batch_topk(pre_loc, K)
    sel_local = torch.zeros(B, F, dtype=torch.bool)
    sel_local[b_idx, f_idx + r * fl] = True
    sel = gather_full(sel_local.unsqueeze(0), dim=0).any(0)
    ref = torch.zeros(B * F, dtype=torch.bool)
    ref[torch.topk(pre_full.flatten(), K * B, sorted=False).indices] = True
    ref = ref.view(B, F)
    check(torch.equal(sel, ref) and int(sel.sum()) == K * B, f"(a) global BatchTopK set == reference topk ({int(sel.sum())} = k*B={K * B}), tau={tau:.4f}")
    # degenerate: fewer than k*B positives -> never select zeros
    pre2 = torch.zeros(B, F); pre2[0, :3] = torch.tensor([1.0, 2.0, 3.0]); dist.broadcast(pre2, src=0)
    b2, f2, tau2 = select_global_batch_topk(pre2[:, r * fl:(r + 1) * fl].contiguous(), K)
    n2 = torch.tensor([float(b2.numel())]); all_reduce_(n2, "sum")
    check(int(n2.item()) == 3, "(a) tau guard: only the 3 positive entries selected when < k*B positives exist")


def reference_loss(x, W_enc, b_enc, W_dec, b_dec, k, dead_full, shard_bounds, k_aux_local, alpha):
    xc = x - b_dec
    pre = torch.relu(xc @ W_enc.t() + b_enc)
    Bn, Fn = pre.shape
    mask = torch.zeros(Bn * Fn, dtype=torch.bool)
    mask[torch.topk(pre.detach().flatten(), k * Bn, sorted=False).indices] = True
    f = pre * mask.view(Bn, Fn)
    x_hat = f @ W_dec + b_dec
    e = x - x_hat
    mse = e.pow(2).sum(-1).mean()
    aux_recon = torch.zeros_like(x)
    any_dead = bool(dead_full.any())
    for (s, e_) in shard_bounds:
        idx = dead_full[s:e_].nonzero(as_tuple=True)[0] + s
        if idx.numel() == 0:
            continue
        pre_dead = pre[:, idx]
        ka = min(k_aux_local, idx.numel())
        tv, ti = pre_dead.topk(ka, dim=-1, sorted=False)
        aux_acts = torch.zeros_like(pre_dead).scatter(-1, ti, tv)
        aux_recon = aux_recon + aux_acts @ W_dec[idx]
    auxk = torch.zeros(())
    if any_dead:
        e_d = e.detach()
        l2 = (e_d - aux_recon).pow(2).sum(-1).mean()
        den = (e_d - e_d.mean(0, keepdim=True)).pow(2).sum(-1).mean()
        auxk = l2 / den
    return mse + alpha * auxk, mse, auxk


def test_b_loss_grads(with_dead):
    R, r = world_size(), rank_id()
    fl = F // R
    k_aux_local = 3
    tr = ShardedTrainer(D, F, K, "cpu", lr=1e-3, steps=100, warmup_steps=0, decay_start=None, auxk_alpha=1 / 32,
                        top_k_aux_local=k_aux_local, dead_tokens=1000, seed=3, use_autocast=False)
    with torch.no_grad():                         # non-trivial replicated params
        torch.manual_seed(11)
        bd = torch.randn(D) * 0.1; dist.broadcast(bd, src=0); tr.ae.b_dec.copy_(bd)
        tr.ae.b_enc.copy_(torch.randn(fl, generator=torch.Generator().manual_seed(100 + r)) * 0.1)
        if with_dead:
            tr.since_fired[torch.arange(fl) % 3 == 0] = 1000
    torch.manual_seed(5)
    x = torch.randn(B, D) * 2 + 0.5; dist.broadcast(x, src=0)
    loss, st = tr.compute_loss(x, step=10)
    loss.backward()
    all_reduce_(tr.ae.b_dec.grad, "sum")
    # reference on the gathered full parameters
    W_enc = gather_full(tr.ae.W_enc.data).clone().requires_grad_(True)
    W_dec = gather_full(tr.ae.W_dec.data).clone().requires_grad_(True)
    b_enc = gather_full(tr.ae.b_enc.data).clone().requires_grad_(True)
    b_dec = tr.ae.b_dec.data.clone().requires_grad_(True)
    dead_full = gather_full(tr.last_dead_mask)
    bounds = [(i * fl, (i + 1) * fl) for i in range(R)]
    ref_loss, ref_mse, ref_aux = reference_loss(x, W_enc, b_enc, W_dec, b_dec, K, dead_full, bounds, k_aux_local, 1 / 32)
    ref_loss.backward()
    tag = "with dead/aux" if with_dead else "no dead"
    nd = int(dead_full.sum())
    check(abs(loss.item() - ref_loss.item()) < TOL * max(1, abs(ref_loss.item())), f"(b) loss sharded {loss.item():.6f} == ref {ref_loss.item():.6f} [{tag}, n_dead={nd}]")
    check(abs(st['mse'].item() - ref_mse.item()) < TOL * max(1, ref_mse.item()) and abs(st['auxk'].item() - ref_aux.item()) < TOL * max(1, ref_aux.item()),
          f"(b) mse/auxk match ({st['mse'].item():.5f}/{st['auxk'].item():.5f}) [{tag}]")
    if with_dead:
        check(nd > 0 and ref_aux.item() > 0, f"(b) aux path exercised (n_dead={nd}, auxk={ref_aux.item():.4f})")
    sl = slice(r * fl, (r + 1) * fl)
    for name, got, ref in (("W_enc", tr.ae.W_enc.grad, W_enc.grad[sl]), ("W_dec", tr.ae.W_dec.grad, W_dec.grad[sl]),
                           ("b_enc", tr.ae.b_enc.grad, b_enc.grad[sl]), ("b_dec", tr.ae.b_dec.grad, b_dec.grad)):
        err = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        check(err <= TOL * max(1.0, scale), f"(b) grad {name} slice max|err|={err:.2e} (ref scale {scale:.2e}) [{tag}]")
    check(abs(st["l0"].item() - K) < 1e-6, f"(b) L0 == k ({st['l0'].item():.3f})")
    return tr


# ------------------------------------------------------------------------------------------------------------
def make_fake_acts(root, n_tokens=6000, n_shards=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(D, generator=g) * 3
    per = n_tokens // n_shards
    shards = []
    for c in range(n_shards):
        a = (torch.randn(per, D, generator=g) * torch.linspace(0.5, 4, D) + mu).to(torch.float16)
        p = f"shard_r0_c{c:05d}.bin"
        a.numpy().tofile(os.path.join(root, p))
        shards.append({"path": p, "n": per})
    json.dump({"rank": 0, "d": D, "shards": shards, "kept": n_tokens}, open(os.path.join(root, "manifest_r0.json"), "w"))


def test_c_merge_and_e_e2e(tmp):
    R, r = world_size(), rank_id()
    fl = F // R
    act_dir = os.path.join(tmp, "acts")
    save_dir = os.path.join(tmp, "save")
    if r == 0:
        os.makedirs(act_dir); os.makedirs(save_dir)
        make_fake_acts(act_dir)
    dist.barrier()
    # (e) identical batches across ranks in both data modes
    for mode in ("broadcast", "allgather"):
        src = BatchSource(act_dir, D, B, mode, "cpu", pool_tokens=2000, seed=0, pool_device="cpu", n_readers=2)
        xs = [src.next() for _ in range(5)]
        same = True
        for x in xs:
            x0 = x.clone(); dist.broadcast(x0, src=0)
            same &= torch.equal(x, x0)
        check(same and xs[0].shape == (B, D) and src.total == 6000, f"(e) BatchSource[{mode}] identical [B,d] batches on all ranks, total={src.total}")
        src.close()
    # train a few steps (norm_target unit), save, resume-check, merge
    src = BatchSource(act_dir, D, B, "allgather", "cpu", pool_tokens=2000, seed=1, pool_device="cpu", n_readers=2)
    nf = math.sqrt(sum(src.next().pow(2).sum(1).mean().item() for _ in range(5)) / 5)
    steps = 40
    tr = ShardedTrainer(D, F, K, "cpu", lr=3e-3, steps=steps, warmup_steps=5, decay_start=30, auxk_alpha=1 / 32,
                        top_k_aux_local=2, dead_tokens=B * 8, threshold_start_step=3, seed=7, use_autocast=False)
    tr.norm_factor = nf
    losses = []
    for s in range(steps):
        x = src.next() / nf
        loss, st = tr.train_step(x, s)
        losses.append(loss.item())
        if s == 19:
            tr.save(save_dir, 20, 20 * B, keep=2)
    tr.save(save_dir, steps, steps * B, keep=2)
    thr_saved = tr.ae.threshold.item()            # the resume check below runs one more step (moves the EMA)
    dist.barrier()
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    check(all(map(math.isfinite, losses)) and last < first, f"(e) training runs: loss {first:.4f} -> {last:.4f}, thr={tr.ae.threshold.item():.4f}, dead={int(st['n_dead'].item())}")
    b_dec_all = gather_full(tr.ae.b_dec.data.unsqueeze(0))
    check(torch.equal(b_dec_all[0].expand_as(b_dec_all), b_dec_all), "(e) replicated b_dec bit-identical across ranks after training")
    check(find_common_latest_step(save_dir, R) == steps, f"(e) checkpoint sets found on all ranks, latest step={steps}")
    # resume exactness
    tr2 = ShardedTrainer(D, F, K, "cpu", lr=3e-3, steps=steps, warmup_steps=5, decay_start=30, auxk_alpha=1 / 32,
                         top_k_aux_local=2, dead_tokens=B * 8, threshold_start_step=3, seed=99, use_autocast=False)
    st2, tok2 = tr2.load(os.path.join(save_dir, f"rank{r}", f"ae_shard_step{steps}.pt"))
    same = all(torch.equal(a.data, b.data) for a, b in zip(tr.ae.parameters(), tr2.ae.parameters()))
    same &= torch.equal(tr.ae.threshold, tr2.ae.threshold) and torch.equal(tr.since_fired, tr2.since_fired)
    same &= abs(tr2.norm_factor - nf) < 1e-12 and st2 == steps and tok2 == steps * B
    x = src.next() / nf
    l1, _ = tr.train_step(x, steps); l2, _ = tr2.train_step(x, steps)
    check(same and abs(l1.item() - l2.item()) < 1e-7, f"(e) --resume restores params/opt/threshold/norm_factor exactly (next-step loss {l1.item():.6f} == {l2.item():.6f})")
    src.close()
    # (c) merge
    W_enc_full = gather_full(tr.ae.W_enc.data); W_dec_full = gather_full(tr.ae.W_dec.data); b_enc_full = gather_full(tr.ae.b_enc.data)
    W_dec_full = W_dec_full / W_dec_full.norm(dim=1, keepdim=True)
    if r == 0:
        sd32, meta = merge(save_dir, step=steps, matrix_dtype=torch.float32, verbose=False)
        thr_ref = thr_saved * nf
        subs = {
            "shapes": sd32["encoder.weight"].shape == (F, D) and sd32["decoder.weight"].shape == (D, F),
            "encoder.weight": torch.allclose(sd32["encoder.weight"], W_enc_full, atol=1e-6),
            "decoder.weight": torch.allclose(sd32["decoder.weight"], W_dec_full.t(), atol=1e-6),
            "encoder.bias folded": torch.allclose(sd32["encoder.bias"], b_enc_full * nf, rtol=1e-6, atol=1e-6),
            "b_dec folded": torch.allclose(sd32["b_dec"], tr.ae.b_dec.data * nf, rtol=1e-6, atol=1e-6),
            "threshold folded": abs(sd32["threshold"].item() - thr_ref) <= 1e-6 * max(1.0, abs(thr_ref)),
            "k int32": sd32["k"].item() == K and sd32["k"].dtype == torch.int32,
            "keys": set(sd32) == {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold"},
            "meta norm_factor": meta["norm_factor"] == nf,
        }
        # functional: raw-space encode with the merged SAE == nf * normalised-space encode
        x_raw = torch.randn(B, D) * 3 + 1
        pre_raw = torch.relu((x_raw - sd32["b_dec"]) @ sd32["encoder.weight"].t() + sd32["encoder.bias"])
        pre_nrm = torch.relu((x_raw / nf - tr.ae.b_dec.data) @ W_enc_full.t() + b_enc_full) * nf
        subs["raw encode == nf * normalised encode"] = torch.allclose(pre_raw, pre_nrm, rtol=1e-5, atol=1e-5)
        f_raw = pre_raw * (pre_raw > sd32["threshold"])
        subs["decode shape"] = (f_raw @ sd32["decoder.weight"].t() + sd32["b_dec"]).shape == x_raw.shape
        sd16, _ = merge(save_dir, step=steps, matrix_dtype=torch.bfloat16, verbose=False)
        subs["bf16 encoder == ref.bfloat16()"] = sd16["encoder.weight"].dtype == torch.bfloat16 and torch.equal(sd16["encoder.weight"], W_enc_full.to(torch.bfloat16))
        subs["bf16 decoder == ref.bfloat16()"] = torch.equal(sd16["decoder.weight"], W_dec_full.t().to(torch.bfloat16)) and sd16["encoder.bias"].dtype == torch.float32
        subs["bf16 decoder unit cols (1e-2)"] = torch.allclose(sd16["decoder.weight"].float().norm(dim=0), torch.ones(F), atol=1e-2)
        ok = all(subs.values())
        if not ok:
            print("  (c) failing sub-checks:", [k for k, v in subs.items() if not v], flush=True)
    else:
        ok = True
    check(ok, "(c) merge_shards == gathered params, folded by norm_factor (fp32 exact; bf16 == .bfloat16() of ref); keys/dtypes/shapes match dictionary_learning; raw-space encode == nf * normalised encode")


# ------------------------------------------------------------------------------------------------------------
def brute_force_topn(batches, Fc, N, dedupe):
    """batches: list of (feats [Fc,W,S] np, seqs list) -> per feature sorted list of (val, doc, pos, tuple(tok), len)."""
    cands = [[] for _ in range(Fc)]
    for feats, seqs, L in batches:
        Fc_, W, S = feats.shape
        for w, q in enumerate(seqs):
            doc_ids = q["prefix"] + q["ids"]        # tokens in doc order: prefix then chunk
            for f in range(Fc):
                row = feats[f, w]
                if dedupe:
                    if row.max() <= 0:
                        continue
                    idxs = [int(row.argmax())]
                else:
                    idxs = [i for i in range(S) if row[i] > 0]
                for i in idxs:
                    pos = q["start"] + i
                    end = len(q["prefix"]) + i + 1
                    win = doc_ids[max(0, end - L):end]
                    ln = min(L, pos + 1)
                    win = win[-ln:] if ln < len(win) else win
                    cands[f].append((float(row[i]), q["doc"], pos, tuple(win), ln))
    out = []
    for f in range(Fc):
        c = sorted(cands[f], key=lambda t: -t[0])[:N]
        out.append(c)
    return out


def test_d_maxacts(dedupe):
    Fc, N, L, W, S, PAD = 48, 5, 8, 3, 6, 999
    rng = np.random.default_rng(0 if dedupe else 1)
    store = TopNStore(Fc, N, L, "cpu", pad_id=PAD)
    batches = []
    doc_id = 0
    for bi in range(6):
        seqs = []
        for w in range(W):
            start = int(rng.choice([0, S, 2 * S, 3 * S]))
            doc_len = start + S
            ids = rng.integers(1, 500, size=doc_len).tolist()
            seqs.append({"ids": ids[start:start + S], "prefix": ids[max(0, start - (L - 1)):start], "doc": doc_id, "start": start})
            doc_id += 1
        feats = rng.random((Fc, W, S)).astype(np.float32) * (rng.random((Fc, W, S)) < 0.3)
        win = build_windows(seqs, L, PAD, "cpu")
        store.fire += (torch.from_numpy(feats) > 0).reshape(Fc, -1).sum(1)
        topn_update(*store.slice(0, Fc), torch.from_numpy(feats), *win, dedupe_seq=dedupe)
        batches.append((feats, seqs, L))
    ref = brute_force_topn(batches, Fc, N, dedupe)
    ok = True
    n_filled = 0
    for f in range(Fc):
        got_v = store.vals[f].tolist()
        for j in range(N):
            if j < len(ref[f]):
                rv, rdoc, rpos, rwin, rlen = ref[f][j]
                ok &= abs(got_v[j] - rv) < 1e-6
                ok &= int(store.doc[f, j]) == rdoc and int(store.pos[f, j]) == rpos and int(store.len[f, j]) == rlen
                toks = store.tok[f, j].tolist()
                ok &= tuple(toks[L - rlen:]) == rwin and all(t == PAD for t in toks[:L - rlen])
                n_filled += 1
            else:
                ok &= got_v[j] == -1.0 and int(store.len[f, j]) == 0
    tag = "dedupe per sequence" if dedupe else "all firing tokens"
    check(ok, f"(d) topn_update == brute-force numpy top-N over 6 batches [{tag}] ({n_filled} filled slots checked: vals, doc, pos, len, left-padded window)")
    # merge across 'ranks' == topN over the union
    if dedupe:
        store2 = TopNStore(Fc, N, L, "cpu", pad_id=PAD)
        b2 = []
        for bi in range(4):
            seqs = [{"ids": rng.integers(1, 500, size=S).tolist(), "prefix": [], "doc": 10_000 + bi * W + w, "start": 0} for w in range(W)]
            feats = rng.random((Fc, W, S)).astype(np.float32) * (rng.random((Fc, W, S)) < 0.3)
            topn_update(*store2.slice(0, Fc), torch.from_numpy(feats), *build_windows(seqs, L, PAD, "cpu"), dedupe_seq=True)
            b2.append((feats, seqs, L))
        merged, _ = merge_parts([store.to_dict(), store2.to_dict()])
        ref2 = brute_force_topn(batches + b2, Fc, N, True)
        ok2 = True
        for f in range(Fc):
            for j in range(N):
                if j < len(ref2[f]):
                    ok2 &= abs(float(merged["max_acts"][f, j]) - ref2[f][j][0]) < 2e-3      # fp16 output
                    ok2 &= int(merged["doc_ids"][f, j]) == ref2[f][j][1] and int(merged["positions"][f, j]) == ref2[f][j][2]
                else:
                    ok2 &= float(merged["max_acts"][f, j]) == -1.0
        check(ok2, "(d) maxacts_merge over 2 partial tables == brute-force top-N over the union")
    # build_windows semantics on a hand-made doc
    doc = list(range(100, 130))
    seqs = [{"ids": doc[12:18], "prefix": doc[max(0, 12 - (L - 1)):12], "doc": 0, "start": 12},
            {"ids": doc[0:6], "prefix": [], "doc": 1, "start": 0}]
    wt, wl, wd, wp = build_windows(seqs, L, PAD, "cpu")
    ok3 = wt[0].tolist() == doc[12 - L + 1:13] and int(wl[0]) == L and int(wp[0]) == 12 and int(wd[0]) == 0
    ok3 &= wt[6 + 2].tolist() == [PAD] * (L - 3) + doc[0:3] and int(wl[8]) == 3 and int(wp[8]) == 2 and int(wd[8]) == 1
    ok3 &= wt[5].tolist() == doc[17 - L + 1:18]
    check(ok3, "(d) build_windows: window ends AT the token, spans the previous chunk via prefix, left-pads at doc start with true length")


def test_d2_encode_and_update():
    """Chunked bf16 [Fc,T] encode + fire counts + outlier mask + top-N == dense fp32 reference + brute force."""
    Fc, N, L, W, S, PAD, d = 40, 3, 6, 2, 5, 999, 8
    rng = np.random.default_rng(3)
    g = torch.Generator().manual_seed(3)
    W_enc = (torch.randn(Fc, d, generator=g) / math.sqrt(d)).to(torch.bfloat16)
    b_enc = torch.randn(Fc, generator=g) * 0.2
    b_dec = torch.randn(d, generator=g) * 0.1
    thr = 0.3125                                       # exactly representable in bf16 (the function compares in the encoder dtype)
    store = TopNStore(Fc, N, L, "cpu", pad_id=PAD)
    batches, fire_ref = [], torch.zeros(Fc, dtype=torch.long)
    doc = 0
    for bi in range(5):
        seqs = []
        for w in range(W):
            start = int(rng.choice([0, S, 2 * S]))
            ids = rng.integers(1, 500, size=start + S).tolist()
            seqs.append({"ids": ids[start:], "prefix": ids[max(0, start - (L - 1)):start], "doc": doc, "start": start}); doc += 1
        acts = torch.randn(W * S, d, generator=g) * 2
        acts[3] *= 200.0                                    # one outlier token (norm >> 10x median)
        # dense fp32 reference of what the bf16 path computes: replicate the bf16 rounding of inputs/matmul output
        xc = (acts - b_dec).to(torch.bfloat16)
        pre = torch.addmm(b_enc.to(torch.bfloat16).unsqueeze(1), W_enc, xc.t()).float()   # [Fc, T]
        pre = torch.relu(pre); pre[pre <= thr] = 0
        bad = acts.norm(dim=-1) > 10 * acts.norm(dim=-1).median()
        pre[:, bad] = 0
        fire_ref += (pre > 0).sum(1)
        feats = pre.view(Fc, W, S).numpy()
        n = encode_and_update(acts, seqs, store, W_enc, b_enc, b_dec, thr, feat_chunk=16, L=L, pad_id=PAD, norm_mult=10.0, dedupe_seq=True)
        batches.append((feats, seqs, L))
        assert n == W * S
    ref = brute_force_topn(batches, Fc, N, True)
    all_cands = brute_force_topn(batches, Fc, 10 ** 9, True)      # every candidate, for tie-robust membership
    ok = torch.equal(store.fire, fire_ref) and bool(bad.any())
    n_ties = 0
    for f in range(Fc):
        got = [(float(store.vals[f, j]), int(store.doc[f, j]), int(store.pos[f, j]), tuple(store.tok[f, j].tolist()), int(store.len[f, j])) for j in range(N)]
        ref_vals = [c[0] for c in ref[f]] + [-1.0] * (N - len(ref[f]))
        ok &= all(abs(a[0] - b) < 1e-6 for a, b in zip(got, ref_vals))          # same sorted value list (ties allowed)
        cand_set = {(round(c[0], 6), c[1], c[2], c[3], c[4]) for c in all_cands[f]}
        for v, dd, pp, tk, ln in got:
            if v > 0:                                                            # bf16 ties -> membership, not order
                ok &= (round(v, 6), dd, pp, tk[L - ln:], ln) in cand_set and all(t == PAD for t in tk[:L - ln])
        n_ties += len(ref_vals) - len(set(ref_vals))
    check(ok, f"(d2) encode_and_update (bf16 chunks of 16 over F={Fc}, outlier mask, fire counts) == dense reference + brute-force top-N (fires={int(fire_ref.sum())}, bf16 ties={n_ties})")


def main():
    dist.init_process_group("gloo")
    R, r = world_size(), rank_id()
    torch.set_num_threads(2)
    if r == 0:
        print(f"== sharded 2M-SAE tests: world={R} d={D} F={F} k={K} B={B} ==", flush=True)
    test_a_topk()
    test_b_loss_grads(with_dead=False)
    test_b_loss_grads(with_dead=True)
    tmp = bcast_obj(tempfile.mkdtemp(prefix="dict2m_test_") if r == 0 else None)
    try:
        test_c_merge_and_e_e2e(tmp)
    finally:
        dist.barrier()
        if r == 0:
            shutil.rmtree(tmp, ignore_errors=True)
    test_d_maxacts(dedupe=True)
    test_d_maxacts(dedupe=False)
    test_d2_encode_and_update()
    dist.barrier()
    if r == 0:
        print("ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
