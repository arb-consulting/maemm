"""Feature-parallel BatchTopK SAE trainer for Qwen3.6-27B layer-42 residuals (torchrun, R ranks, F = 2^21).

Rank r owns features [r*F/R, (r+1)*F/R): its own W_enc/W_dec/b_enc slices; b_dec + threshold are replicated and kept
identical. Every rank sees the SAME [B, d] batch (rank 0 broadcasts it, or every rank reads 1/R of the shards and the
batch is all-gathered: --data-mode). BatchTopK is exact and global: each rank takes its local top-(k*B) pre-acts, the
candidate values are all-gathered and the (k*B)-th largest is this batch's threshold tau; a rank keeps its entries with
pre >= tau (and > 0). Reconstruction = all_reduce(SUM) of the per-rank partial reconstructions through an autograd
function whose backward is the identity (dL/d partial_r == dL/d total).

Math follows dictionary_learning's BatchTopKTrainer/BatchTopKSAE (see MATCHED CONVENTIONS in the report):
  pre = relu((x - b_dec) @ W_enc^T + b_enc); f = BatchTopK(pre, k*B) (train) / pre * (pre > threshold) (inference)
  x_hat = f @ W_dec + b_dec; loss = ||x - x_hat||^2.sum(-1).mean() + auxk_alpha * auxk
  auxk = ||e - dec(top-k_aux dead pre-acts)||^2 / ||e - mean(e)||^2 with e = (x - x_hat).detach(), dead = not fired for
         dead_tokens tokens (10M); the top-k_aux is taken PER RANK over that rank's dead latents (k_aux_local = k_aux/R)
  threshold = EMA(beta=.999) of the min positive selected pre-act (global min), starting at threshold_start_step
  W_dec rows unit-norm (renormalised before every step; grad component parallel to the row removed)
  normalize_activations: x <- x / norm_factor (norm_factor from the first --norm-steps batches); shard checkpoints stay
  in NORMALISED space (so resume is exact); sae27b_merge_shards.py folds norm_factor into b_enc/b_dec/threshold.
Data modes: broadcast / allgather read stored fp16 shards; ONLINE (sae2m) generates the activations on the fly -- every rank
runs its own truncated Qwen3.6-27B (layers 0..42) on a disjoint Ultra-FineWeb stream (online_gen.OnlineActGenerator),
keeps the fp16 rows in a per-rank GPU shuffle pool (online_gen.OnlinePool, each row drawn at most --pool-reuse-max times)
and contributes B/R rows per step; the batch is all-gathered exactly as in allgather mode. --total-tokens sets the steps.
Speed: the dense encoder matmul runs in bf16 autocast under no_grad for the *selection* only; the loss/gradients are
recomputed in fp32 at the selected (token, feature) pairs, so the decoder forward/backward and the encoder backward are
sparse gathers/index_adds (k*B rows) instead of dense [B, F/R] x [F/R, d] matmuls. The aux path is dense but restricted
to the dead latents of the rank.
"""
import os, sys, json, time, math, glob, signal, argparse, datetime
import torch
import torch.nn as nn
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------------------------------------------
# distributed helpers
# ----------------------------------------------------------------------------------------------------------------
def dist_ok():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if dist_ok() else 1


def rank_id():
    return dist.get_rank() if dist_ok() else 0


def all_reduce_(t, op="sum"):
    if world_size() > 1:
        dist.all_reduce(t, op={"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}[op])
    return t


class _AllReduceSum(torch.autograd.Function):
    """y = sum over ranks of x. Backward = identity: every rank holds the same L(y), so dL/dx_r = dL/dy."""

    @staticmethod
    def forward(ctx, x):
        y = x.clone()
        if world_size() > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(ctx, g):
        return g


def all_reduce_sum_autograd(x):
    return _AllReduceSum.apply(x)


class _GradScale(torch.autograd.Function):
    """Identity in forward, multiplies the gradient by `s` in backward. Used on the replicated b_dec's direct
    (decoder-bias) path so that a plain all_reduce(SUM) of the per-rank b_dec grads yields the exact full gradient:
    R * (direct / R) + sum_r encoder_path_r."""

    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


def grad_scale(x, s):
    return _GradScale.apply(x, s)


def select_global_batch_topk(pre_relu, k):
    """Exact global BatchTopK across feature shards.
    pre_relu: [B, F_local] fp32, >= 0. Returns (b_idx, f_idx, tau) for the LOCAL entries that are in the global
    top-(k*B) of the concatenated [B, F] pre-acts. Ties at tau are accepted (>=) so on an exact tie the selected
    count can exceed k*B; entries equal to 0 are never selected (guards tau == 0 when < k*B positives exist)."""
    B = pre_relu.shape[0]
    kB = k * B
    flat = pre_relu.reshape(-1)
    if kB >= flat.numel():
        local_top = flat
    else:
        # CUDA topk indexes with int32: split inputs > 2^30 elements (e.g. B=8192 x F/R=262144) and re-select
        CH = 1 << 30
        if flat.numel() <= CH:
            local_top = torch.topk(flat, kB, sorted=False).values
        else:
            parts = [torch.topk(flat[i:i + CH], min(kB, flat[i:i + CH].numel()), sorted=False).values
                     for i in range(0, flat.numel(), CH)]
            local_top = torch.topk(torch.cat(parts), kB, sorted=False).values
    W = world_size()
    if W > 1:
        gathered = [torch.empty_like(local_top) for _ in range(W)]
        dist.all_gather(gathered, local_top.contiguous())
        cand = torch.cat(gathered)
    else:
        cand = local_top
    kk = min(kB, cand.numel())
    tau = torch.topk(cand, kk, sorted=False).values.min()
    mask = (pre_relu >= tau) & (pre_relu > 0)
    b_idx, f_idx = mask.nonzero(as_tuple=True)
    return b_idx, f_idx, tau


def geometric_median(points, max_iter=100, tol=1e-5):
    """Weiszfeld, as in dictionary_learning.trainers.batch_top_k.BatchTopKTrainer.geometric_median."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        w = 1.0 / torch.norm(points - guess, dim=1).clamp_min(1e-12)
        w = w / w.sum()
        guess = (w.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def lr_lambda_factory(total_steps, warmup_steps, decay_start):
    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if decay_start is not None and step >= decay_start:
            return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
        return 1.0
    return fn


# ----------------------------------------------------------------------------------------------------------------
# model shard
# ----------------------------------------------------------------------------------------------------------------
class ShardedBatchTopKSAE(nn.Module):
    """Feature shard of a BatchTopKSAE. W_enc/W_dec are stored [F_local, d] (rows = features, dictionary_learning's
    encoder.weight layout; decoder.weight = W_dec^T at merge)."""

    def __init__(self, d, f_local, k, device, seed=0):
        super().__init__()
        self.d, self.f_local = d, f_local
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(f_local, d, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.to(device))
        self.W_enc = nn.Parameter(W.clone().to(device))
        self.b_enc = nn.Parameter(torch.zeros(f_local, device=device))
        self.b_dec = nn.Parameter(torch.zeros(d, device=device))        # replicated
        self.register_buffer("threshold", torch.tensor(-1.0, device=device))
        self.register_buffer("k", torch.tensor(k, dtype=torch.int32, device=device))

    @torch.no_grad()
    def renorm_decoder_(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_grad_(self):
        g = self.W_dec.grad
        if g is None:
            return
        Wn = self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        par = (g * Wn).sum(dim=1, keepdim=True)
        g.sub_(par * Wn)


class ShardedTrainer:
    def __init__(self, d, dict_size, k, device, lr, steps, warmup_steps, decay_start, auxk_alpha=1 / 32,
                 top_k_aux_local=320, dead_tokens=10_000_000, threshold_beta=0.999, threshold_start_step=1000,
                 seed=0, use_autocast=None, adam_fused=None):
        self.R, self.r = world_size(), rank_id()
        assert dict_size % self.R == 0, f"dict_size {dict_size} not divisible by world {self.R}"
        self.d, self.F, self.k = d, dict_size, k
        self.f_local = dict_size // self.R
        self.f0 = self.r * self.f_local
        self.device = device
        self.ae = ShardedBatchTopKSAE(d, self.f_local, k, device, seed=seed * 1000 + self.r)
        self.lr = lr
        self.auxk_alpha = auxk_alpha
        self.top_k_aux_local = int(top_k_aux_local)
        self.dead_tokens = int(dead_tokens)
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.since_fired = torch.zeros(self.f_local, dtype=torch.long, device=device)
        is_cuda = torch.device(device).type == "cuda"
        self.use_autocast = is_cuda if use_autocast is None else use_autocast
        fused = is_cuda if adam_fused is None else adam_fused
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999), fused=fused if is_cuda else False)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda_factory(steps, warmup_steps, decay_start))
        self.norm_factor = 1.0
        self.last = {}

    # ------------------------------------------------------------------------------------------------------------
    def compute_loss(self, x, step, update_dead=True):
        """x: [B, d] fp32 in training (normalised) space, identical on all ranks. Returns (loss, stats)."""
        ae = self.ae
        B, d = x.shape
        xc = x - ae.b_dec                                                   # [B, d]

        # 1) dense pre-acts for SELECTION only (bf16 autocast on cuda, no grad)
        with torch.no_grad():
            with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16,
                                enabled=self.use_autocast):
                pre = torch.addmm(ae.b_enc, xc, ae.W_enc.t())               # [B, F_local]
            pre = pre.float().relu_()
            b_idx, f_idx, tau = select_global_batch_topk(pre, self.k)
            n_sel_local = torch.tensor([float(b_idx.numel())], device=x.device)

        # 2) fp32 recompute at the selected pairs -> sparse gradients for W_enc/b_enc/W_dec/xc
        xs = xc[b_idx]                                                      # [M, d]
        vals = (xs * ae.W_enc[f_idx]).sum(-1) + ae.b_enc[f_idx]             # [M]
        vals = torch.relu(vals)                                             # bf16 selection vs fp32 recompute guard
        recon_partial = torch.zeros(B, d, device=x.device, dtype=torch.float32)
        recon_partial = recon_partial.index_add(0, b_idx, vals.unsqueeze(1) * ae.W_dec[f_idx])
        recon = all_reduce_sum_autograd(recon_partial)
        x_hat = recon + grad_scale(ae.b_dec, 1.0 / self.R)
        e = x - x_hat
        mse = e.pow(2).sum(-1).mean()

        # 3) dead-feature bookkeeping (dictionary_learning order: update counters, THEN compute dead for aux)
        with torch.no_grad():
            if update_dead:
                fired = torch.zeros(self.f_local, dtype=torch.bool, device=x.device)
                fired[f_idx] = True
                self.since_fired += B
                self.since_fired[fired] = 0
            dead = self.since_fired >= self.dead_tokens
            n_dead_local = dead.sum()
            n_dead_global = all_reduce_(n_dead_local.clone(), "sum")
            self.last_dead_mask = dead

        # 4) aux-k loss on dead latents (per-rank top-k_aux over this rank's dead latents), dense but dead-only
        auxk = torch.zeros((), device=x.device)
        aux_pre_norm = torch.tensor(-1.0, device=x.device)
        if int(n_dead_global.item()) > 0 and self.auxk_alpha > 0:
            e_d = e.detach()
            dead_idx = dead.nonzero(as_tuple=True)[0]
            if dead_idx.numel() > 0:
                with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16,
                                    enabled=self.use_autocast):
                    pre_dead = torch.addmm(ae.b_enc[dead_idx], xc, ae.W_enc[dead_idx].t())   # [B, D]
                pre_dead = torch.relu(pre_dead.float())
                ka = min(self.top_k_aux_local, dead_idx.numel())
                tv, ti = pre_dead.topk(ka, dim=-1, sorted=False)
                aux_acts = torch.zeros_like(pre_dead).scatter(-1, ti, tv)
                with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16,
                                    enabled=self.use_autocast):
                    aux_partial = aux_acts @ ae.W_dec[dead_idx]                              # [B, d]
                aux_partial = aux_partial.float()
            else:
                aux_partial = torch.zeros(B, d, device=x.device, dtype=torch.float32)
            aux_recon = all_reduce_sum_autograd(aux_partial)
            l2_aux = (e_d - aux_recon).pow(2).sum(-1).mean()
            denom = (e_d - e_d.mean(dim=0, keepdim=True)).pow(2).sum(-1).mean()
            auxk = (l2_aux / denom).nan_to_num(0.0)
            aux_pre_norm = l2_aux.detach()

        loss = mse + self.auxk_alpha * auxk

        # 5) stats (identical on all ranks after the reductions)
        with torch.no_grad():
            n_sel = all_reduce_(n_sel_local.clone(), "sum")
            l0 = n_sel / B
            tot_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
            ev = 1.0 - e.pow(2).sum() / tot_var.clamp_min(1e-12)
            min_pos = vals.detach()[vals.detach() > 0]
            min_pos = min_pos.min() if min_pos.numel() else torch.tensor(float("inf"), device=x.device)
            min_pos = all_reduce_(min_pos.clone().reshape(1), "min")[0]
        stats = dict(loss=loss.detach(), mse=mse.detach(), auxk=auxk.detach(), aux_pre_norm=aux_pre_norm, ev=ev,
                     l0=l0[0], n_dead=n_dead_global.float(), tau=tau, min_pos=min_pos)
        self.last = dict(b_idx=b_idx, f_idx=f_idx, tau=tau)
        return loss, stats

    @torch.no_grad()
    def update_threshold(self, min_pos):
        if not torch.isfinite(min_pos):
            min_pos = torch.zeros((), device=self.device)
        if self.ae.threshold < 0:
            self.ae.threshold.copy_(min_pos)
        else:
            self.ae.threshold.mul_(self.threshold_beta).add_((1 - self.threshold_beta) * min_pos)

    @torch.no_grad()
    def sync_after_backward(self):
        """all-reduce the replicated b_dec grad (encoder paths summed, direct path pre-scaled by 1/R),
        project W_dec grad, global grad-norm clip at 1.0. Returns the global grad norm."""
        ae = self.ae
        if ae.b_dec.grad is None:
            ae.b_dec.grad = torch.zeros_like(ae.b_dec)
        all_reduce_(ae.b_dec.grad, "sum")
        ae.project_decoder_grad_()
        sq = torch.zeros((), device=self.device)
        for p in (ae.W_enc, ae.W_dec, ae.b_enc):
            if p.grad is not None:
                sq += p.grad.pow(2).sum()
        if self.r == 0:                                    # replicated param: count once
            sq += ae.b_dec.grad.pow(2).sum()
        all_reduce_(sq, "sum")
        gnorm = sq.sqrt()
        coef = (1.0 / (gnorm + 1e-6)).clamp(max=1.0)
        for p in ae.parameters():
            if p.grad is not None:
                p.grad.mul_(coef)
        return gnorm

    def train_step(self, x, step):
        ae = self.ae
        if step == 0:
            med = geometric_median(x)
            if world_size() > 1:
                dist.broadcast(med, src=0)
            ae.b_dec.data.copy_(med)
        ae.renorm_decoder_()
        loss, stats = self.compute_loss(x, step)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = self.sync_after_backward()
        self.opt.step()
        self.sched.step()
        if world_size() > 1:                               # belt-and-braces: keep the replica bit-identical
            dist.broadcast(ae.b_dec.data, src=0)
        if step > self.threshold_start_step:
            self.update_threshold(stats["min_pos"])
        stats["gnorm"] = gnorm
        stats["threshold"] = ae.threshold.detach().clone()
        return loss, stats

    # ------------------------------------------------------------------------------------------------------------
    def state_dict(self, step, tokens_seen, extra=None):
        ae = self.ae
        sd = {
            "format": "sae27b_sharded_v1", "rank": self.r, "world": self.R, "d": self.d, "dict_size": self.F,
            "f_local": self.f_local, "f0": self.f0, "k": self.k, "step": step, "tokens_seen": tokens_seen,
            "norm_factor": float(self.norm_factor),
            "W_enc": ae.W_enc.data, "W_dec": ae.W_dec.data, "b_enc": ae.b_enc.data, "b_dec": ae.b_dec.data,
            "threshold": ae.threshold.data.clone(), "since_fired": self.since_fired,
            "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
        }
        if extra:
            sd.update(extra)
        return sd

    def save(self, save_dir, step, tokens_seen, extra=None, keep=2):
        rd = os.path.join(save_dir, f"rank{self.r}")
        os.makedirs(rd, exist_ok=True)
        path = os.path.join(rd, f"ae_shard_step{step}.pt")
        tmp = path + ".tmp"
        torch.save(self.state_dict(step, tokens_seen, extra), tmp)
        os.replace(tmp, path)
        if keep and keep > 0:
            olds = sorted(glob.glob(os.path.join(rd, "ae_shard_step*.pt")), key=_step_of)
            for p in olds[:-keep]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return path

    def load(self, path):
        sd = torch.load(path, map_location=self.device, weights_only=False)
        assert sd["world"] == self.R and sd["rank"] == self.r, f"ckpt is rank {sd['rank']}/{sd['world']}, we are {self.r}/{self.R}"
        assert sd["dict_size"] == self.F and sd["d"] == self.d and sd["k"] == self.k
        ae = self.ae
        ae.W_enc.data.copy_(sd["W_enc"]); ae.W_dec.data.copy_(sd["W_dec"])
        ae.b_enc.data.copy_(sd["b_enc"]); ae.b_dec.data.copy_(sd["b_dec"])
        ae.threshold.data.copy_(sd["threshold"])
        self.since_fired.copy_(sd["since_fired"].to(self.device))
        self.opt.load_state_dict(sd["opt"])
        self.sched.load_state_dict(sd["sched"])
        self.norm_factor = float(sd["norm_factor"])
        return int(sd["step"]), int(sd["tokens_seen"])


def _step_of(p):
    return int(os.path.basename(p).split("step")[-1].split(".")[0])


def find_common_latest_step(save_dir, world):
    steps = None
    for r in range(world):
        s = {_step_of(p) for p in glob.glob(os.path.join(save_dir, f"rank{r}", "ae_shard_step*.pt"))}
        steps = s if steps is None else steps & s
    return max(steps) if steps else None


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
class BatchSource:
    """Yields identical [B, d] fp32 batches on every rank.
    broadcast: rank 0 owns the whole buffer (pool of pool_tokens) and broadcasts each fp16 batch.
    allgather: rank r owns shards[r::R] (pool_tokens/R each), contributes B/R rows, batch = all_gather."""

    def __init__(self, act_dir, d, B, mode, device, pool_tokens, seed, pool_device, n_readers, online=None):
        """online (mode == "online"): dict(gen_fn=callable -> fp16 [n, d] rows or None, total_tokens=int,
        capacity=int rows per rank, reuse_max=int, refill_frac=float, state_fn=callable -> dict | None, stats_fn=... | None,
        close_fn=callable | None). The generator is built by the caller (real: online_gen.OnlineActGenerator; tests: fakes)."""
        self.R, self.r = world_size(), rank_id()
        self.mode, self.B, self.d, self.device = mode, B, d, device
        is_cuda = torch.device(device).type == "cuda"
        # gloo has no fp16 all_gather/broadcast in some builds -> fp32 on cpu (tests); fp16 (lossless) on cuda
        self.comm_dtype = torch.float16 if is_cuda else torch.float32
        self.buf = None
        self.pool = None
        self.online = None
        if mode == "online":
            from online_gen import OnlinePool
            assert online is not None and B % self.R == 0, f"B={B} must be divisible by world={self.R}"
            self.online = online
            self.pool = OnlinePool(online["gen_fn"], d, online["capacity"], device, seed=seed + 7919 * self.r,
                                   reuse_max=online.get("reuse_max", 1), refill_frac=online.get("refill_frac", 0.5),
                                   dtype=self.comm_dtype)
            t = torch.tensor([float(online["total_tokens"])], device=device)
            if self.R > 1:
                dist.broadcast(t, src=0)
            self.total = int(t.item())
            return
        from sae27b_mt_prefetch_buffer import MTPrefetchDiskActBuffer
        if mode == "broadcast":
            if self.r == 0:
                self.buf = MTPrefetchDiskActBuffer(act_dir, d, B, device=device, pool_tokens=pool_tokens, seed=seed,
                                                   pool_device=pool_device, n_readers=n_readers)
                self.total = self.buf.total
            else:
                self.total = 0
            t = torch.tensor([float(self.total)], device=device)
            if self.R > 1:
                dist.broadcast(t, src=0)
            self.total = int(t.item())
        elif mode == "allgather":
            assert B % self.R == 0, f"B={B} must be divisible by world={self.R}"
            self.buf = MTPrefetchDiskActBuffer(act_dir, d, B // self.R, device=device, pool_tokens=max(1, pool_tokens // self.R),
                                               seed=seed + 7919 * self.r, pool_device=pool_device, n_readers=n_readers,
                                               shard_subset=(self.r, self.R))
            t = torch.tensor([float(self.buf.total)], device=device)
            all_reduce_(t, "sum")
            self.total = int(t.item())
        else:
            raise ValueError(mode)

    def next(self):
        """Returns [B, d] fp32 on self.device (raw activation space)."""
        if self.mode == "online":
            loc = self.pool.draw(self.B // self.R).to(self.device, dtype=self.comm_dtype).contiguous()
            if self.R == 1:
                return loc.float()
            parts = [torch.empty_like(loc) for _ in range(self.R)]
            dist.all_gather(parts, loc)
            return torch.cat(parts, dim=0).float()
        if self.mode == "broadcast":
            if self.r == 0:
                b = self.buf.next_raw().to(self.device, dtype=self.comm_dtype)
            else:
                b = torch.empty(self.B, self.d, dtype=self.comm_dtype, device=self.device)
            if self.R > 1:
                dist.broadcast(b, src=0)
            return b.float()
        else:
            loc = self.buf.next_raw().to(self.device, dtype=self.comm_dtype).contiguous()
            if self.R == 1:
                return loc.float()
            parts = [torch.empty_like(loc) for _ in range(self.R)]
            dist.all_gather(parts, loc)
            return torch.cat(parts, dim=0).float()

    def state(self):
        """Resume state of the online generator (per rank), or None for the disk modes."""
        if self.mode == "online" and self.online.get("state_fn"):
            return self.online["state_fn"]()
        return None

    def stats(self):
        """Per-rank data-side metrics (online mode: pool + generator)."""
        out = {}
        if self.mode == "online":
            out.update(self.pool.stats())
            if self.online.get("stats_fn"):
                out.update(self.online["stats_fn"]())
        return out

    def close(self):
        if self.buf is not None:
            self.buf.close()
        if self.mode == "online" and self.online.get("close_fn"):
            self.online["close_fn"]()


# ----------------------------------------------------------------------------------------------------------------
STOP = {"flag": False, "why": ""}


def _sig_handler(signum, frame):
    STOP["flag"] = True
    STOP["why"] = f"signal {signum}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", default=None, help="stored fp16 shards (broadcast/allgather modes)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--d-model", type=int, default=5120, help="(named --d-model: torchrun eats --d as an ambiguous prefix)")
    ap.add_argument("--dict-size", type=int, default=2_097_152)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--batch", "-B", type=int, default=4096)
    ap.add_argument("--data-mode", choices=["broadcast", "allgather", "online"], default="broadcast")
    # ---- online generation (--data-mode online; see online_gen.py) ----
    ap.add_argument("--total-tokens", type=int, default=1_000_000_000, help="online: training tokens -> steps = total_tokens // batch")
    ap.add_argument("--pool-rows", type=int, default=262_144, help="online: per-rank GPU shuffle-pool rows (262144 x 5120 fp16 = 2.7 GB)")
    ap.add_argument("--pool-reuse-max", type=int, default=1, help="online: max draws per pooled row (1 = every row trained on once)")
    ap.add_argument("--pool-refill-frac", type=float, default=0.5, help="online: refill when drawable rows < frac * pool-rows")
    ap.add_argument("--micro-batch", type=int, default=16, help="online: sequences per 27B forward")
    ap.add_argument("--ctx-len", type=int, default=512, help="online: content tokens per sequence ([BOS] + ctx-len forwarded)")
    ap.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    ap.add_argument("--split", default="en")
    ap.add_argument("--dataset-skip", type=int, default=100_000, help="online: docs dropped at the head of EVERY rank's stream (eval head)")
    ap.add_argument("--norm-mult", type=float, default=10.0, help="online: drop tokens with ||x|| > mult * median (0 = off)")
    ap.add_argument("--gen-qsize", type=int, default=32)
    ap.add_argument("--pool-tokens", type=int, default=2_000_000, help="global shuffle-pool size (split /R in allgather mode)")
    ap.add_argument("--pool-device", default="cuda", help="cuda|cpu: where the shuffled pool lives")
    ap.add_argument("--n-readers", type=int, default=4, help="concurrent shard readers in the prefetch thread")
    ap.add_argument("--lr", type=float, default=None, help="None -> 2e-4/sqrt(dict_size/16384)")
    ap.add_argument("--auxk-alpha", type=float, default=1 / 32)
    ap.add_argument("--top-k-aux", type=int, default=None, help="GLOBAL k_aux per token; None -> round(2560*F/131072); per rank = /R")
    ap.add_argument("--dead-tokens", type=int, default=10_000_000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay-start-frac", type=float, default=0.8)
    ap.add_argument("--threshold-beta", type=float, default=0.999)
    ap.add_argument("--threshold-start-step", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=None, help="None -> epoch_frac * corpus / B")
    ap.add_argument("--epoch-frac", type=float, default=0.98)
    ap.add_argument("--norm-target", choices=["unit", "sqrt_d", "none"], default="unit",
                    help="unit: mean ||x||^2 = 1 (dictionary_learning normalize_activations); sqrt_d: mean ||x||^2 = d; none")
    ap.add_argument("--norm-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--keep-ckpts", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="resume from the latest complete shard set in --save-dir")
    ap.add_argument("--max-hours", type=float, default=None, help="save + exit cleanly after this many hours")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="qwen36-27b-sae")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-autocast", action="store_true")
    args = ap.parse_args()

    # ---- distributed init ----
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(hours=2))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
        if args.pool_device == "cuda":
            args.pool_device = "cpu"
    torch.manual_seed(args.seed)

    def log(*a, **k):
        if rank == 0:
            print(f"[train2m] {time.strftime('%H:%M:%S')}", *a, flush=True, **k)

    os.makedirs(args.save_dir, exist_ok=True)
    pid_dir = os.path.join(args.save_dir, "pids")
    os.makedirs(pid_dir, exist_ok=True)
    with open(os.path.join(pid_dir, f"rank{rank}.pid"), "w") as f:
        f.write(str(os.getpid()))
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGUSR1, _sig_handler)

    # ---- data ----  (on resume the shard order is a fresh shuffle seeded by the step, not a replay from shard 0)
    resume_step = find_common_latest_step(args.save_dir, world) if args.resume else None
    data_seed = args.seed if not resume_step else args.seed + 1_000_003 * (1 + resume_step // 1000)
    gen = None
    online = None
    if args.data_mode == "online":
        from online_gen import OnlineActGenerator
        extra_skip, doc_start = 0, 0
        if resume_step:
            gs_path = os.path.join(args.save_dir, f"rank{rank}", f"gen_state_step{resume_step}.json")
            if os.path.exists(gs_path):
                gs = json.load(open(gs_path))
                extra_skip = doc_start = int(gs.get("docs_iterated", 0))
                print(f"[train2m r{rank}] resume: skipping {extra_skip} already-consumed docs of this rank's stream", flush=True)
        gen = OnlineActGenerator(rank=rank, world=world, device=device, model_name=args.model, layer=args.layer, dataset=args.dataset,
                                 split=args.split, dataset_skip=args.dataset_skip, extra_skip=extra_skip, doc_start=doc_start,
                                 ctx_len=args.ctx_len, micro_batch=args.micro_batch, norm_mult=args.norm_mult, d=args.d_model,
                                 qsize=args.gen_qsize, log=lambda *a: print(f"[train2m r{rank}]", *a, flush=True))
        online = dict(gen_fn=gen.next_chunk, total_tokens=args.total_tokens, capacity=args.pool_rows, reuse_max=args.pool_reuse_max,
                      refill_frac=args.pool_refill_frac, state_fn=gen.state, stats_fn=gen.stats, close_fn=gen.close)
        if use_cuda:
            torch.cuda.synchronize()
        if world > 1:
            dist.barrier()          # every rank has its 27B up before the first all_gather
        log(f"online generator up on all ranks: model={gen.model_info} bos={gen.bos} S={args.ctx_len} micro_batch={args.micro_batch} "
            f"pool_rows={args.pool_rows} reuse_max={args.pool_reuse_max} dataset={args.dataset}/{args.split} skip={args.dataset_skip}")
    else:
        assert args.act_dir, "--act-dir is required for the disk data modes"
    src = BatchSource(args.act_dir, args.d_model, args.batch, args.data_mode, device, args.pool_tokens, data_seed,
                      args.pool_device, args.n_readers, online=online)
    if args.steps is not None:
        steps = args.steps
    elif args.data_mode == "online":
        steps = args.total_tokens // args.batch
    else:
        steps = int(args.epoch_frac * src.total / args.batch)
    decay_start = int(args.decay_start_frac * steps)
    lr = args.lr if args.lr is not None else 2e-4 / math.sqrt(args.dict_size / 16384)
    top_k_aux = args.top_k_aux if args.top_k_aux is not None else int(round(2560 * args.dict_size / 131072))
    top_k_aux_local = max(1, int(round(top_k_aux / world)))
    log(f"world={world} device={device} F={args.dict_size} (F/R={args.dict_size // world}) k={args.k} B={args.batch} "
        f"corpus={src.total:,} steps={steps} (~{steps * args.batch:,} tokens) lr={lr:.3e} decay_start={decay_start} "
        f"top_k_aux={top_k_aux} (/rank {top_k_aux_local}) data_mode={args.data_mode} pool={args.pool_tokens:,}")

    tr = ShardedTrainer(args.d_model, args.dict_size, args.k, device, lr, steps, args.warmup, decay_start,
                        auxk_alpha=args.auxk_alpha, top_k_aux_local=top_k_aux_local, dead_tokens=args.dead_tokens,
                        threshold_beta=args.threshold_beta, threshold_start_step=args.threshold_start_step,
                        seed=args.seed, use_autocast=(None if not args.no_autocast else False))

    # ---- resume / norm factor ----
    start_step, tokens_seen = 0, 0
    if args.resume:
        if resume_step is None:
            log("--resume given but no complete shard set found; starting fresh")
        else:
            path = os.path.join(args.save_dir, f"rank{rank}", f"ae_shard_step{resume_step}.pt")
            start_step, tokens_seen = tr.load(path)
            log(f"resumed from step {start_step} ({tokens_seen:,} tokens) norm_factor={tr.norm_factor:.4f} data_seed={data_seed}")
    if start_step == 0:
        if args.norm_target == "none":
            tr.norm_factor = 1.0
        else:
            acc, n = torch.zeros((), device=device, dtype=torch.float64), 0
            for _ in range(args.norm_steps):
                x = src.next()
                acc += x.double().pow(2).sum(dim=1).mean()
                n += 1
            msn = (acc / n).item()                          # mean squared norm (identical on all ranks)
            tr.norm_factor = math.sqrt(msn) if args.norm_target == "unit" else math.sqrt(msn / args.d_model)
            log(f"norm: mean||x||^2={msn:.2f} -> norm_factor={tr.norm_factor:.4f} (target={args.norm_target})")

    cfg = {
        "trainer": {"trainer_class": "BatchTopKTrainer(sharded)", "dict_class": "BatchTopKSAE", "lr": lr, "steps": steps,
                    "seed": args.seed, "activation_dim": args.d_model, "dict_size": args.dict_size, "k": args.k,
                    "device": "cuda", "layer": args.layer, "lm_name": args.model, "submodule_name": f"resid_post_layer_{args.layer}",
                    "wandb_name": args.run_name or f"BatchTopK-2M-l{args.layer}", "auxk_alpha": args.auxk_alpha,
                    "top_k_aux": top_k_aux, "top_k_aux_local": top_k_aux_local, "dead_tokens": args.dead_tokens,
                    "warmup_steps": args.warmup, "decay_start": decay_start, "threshold_beta": args.threshold_beta,
                    "threshold_start_step": args.threshold_start_step, "batch": args.batch, "world": world,
                    "norm_target": args.norm_target, "norm_factor": tr.norm_factor, "data_mode": args.data_mode},
        "buffer": {"act_dir": args.act_dir, "total_tokens": src.total, "pool_tokens": args.pool_tokens, "d_submodule": args.d_model,
                   "io": "out", "out_batch_size": args.batch, "data_mode": args.data_mode,
                   **({"online": {"dataset": args.dataset, "split": args.split, "dataset_skip": args.dataset_skip, "ctx_len": args.ctx_len,
                                  "micro_batch": args.micro_batch, "norm_mult": args.norm_mult, "pool_rows": args.pool_rows,
                                  "pool_reuse_max": args.pool_reuse_max, "pool_refill_frac": args.pool_refill_frac,
                                  "bos": gen.bos, "model_layers_kept": gen.model_info.get("n_layers_kept"),
                                  "convention": "[BOS]+ctx_len content tokens forwarded; BOS position dropped; non-overlapping windows; "
                                                "docs < ctx_len dropped; tokens with ||x|| > norm_mult*median(micro-batch) dropped; "
                                                "split_dataset_by_node THEN skip(dataset_skip) per rank"}}
                      if args.data_mode == "online" else {})},
    }

    wb = None
    if args.wandb and rank == 0:
        import wandb
        idf = os.path.join(args.save_dir, "wandb_id.txt")
        wid = open(idf).read().strip() if os.path.exists(idf) else None
        wb = wandb.init(project=args.wandb_project, name=args.run_name or f"BatchTopK-2M-l{args.layer}", id=wid,
                        resume="allow", config=cfg["trainer"])
        open(idf, "w").write(wb.id)

    def write_config(step, tokens):
        if rank == 0:
            cfg["trainer"]["tokens_seen"] = tokens
            cfg["trainer"]["last_step"] = step
            cfg["trainer"]["norm_factor"] = tr.norm_factor
            with open(os.path.join(args.save_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=1)

    write_config(start_step, tokens_seen)

    # ---- loop ----
    t0 = time.time()
    t_log = time.time()
    step = start_step
    stop_flag = torch.zeros(1, device=device)
    saved_final = False
    t_data = 0.0
    m_prev = {}
    while step < steps:
        td = time.time()
        x = src.next() / tr.norm_factor
        if use_cuda:
            torch.cuda.synchronize()
        t_data += time.time() - td
        loss, st = tr.train_step(x, step)
        step += 1
        tokens_seen += args.batch

        if step % args.log_every == 0 or step == 1:
            if use_cuda:
                torch.cuda.synchronize()
            dt = time.time() - t_log
            t_log = time.time()
            n_since = args.log_every if step != 1 else 1
            its = n_since / max(dt, 1e-9)
            data_frac = t_data / max(dt, 1e-9)
            t_data = 0.0
            m = {"loss": st["loss"].item(), "mse": st["mse"].item(), "auxk": st["auxk"].item(), "ev": st["ev"].item(),
                 "l0": st["l0"].item(), "dead_frac": st["n_dead"].item() / args.dict_size, "n_dead": st["n_dead"].item(),
                 "threshold": st["threshold"].item(), "tau_batch": st["tau"].item(), "grad_norm": st["gnorm"].item(),
                 "lr": tr.sched.get_last_lr()[0], "it_s": its, "tok_s": its * args.batch, "tokens": tokens_seen,
                 "data_wait_frac": data_frac,
                 "elapsed_h": (time.time() - t0) / 3600}
            if use_cuda and rank == 0:
                m["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            ds_ = src.stats()
            if ds_:
                # gen_frac: share of the wall clock spent generating (rank 0's own generator); gen_tok_s: 27B forward tokens/s while generating
                gt = ds_.get("gen_time_s", 0.0)
                m.update({"gen_frac": (gt - m_prev.get("gen_time_s", 0.0)) / max(dt, 1e-9), "gen_tok_s": ds_.get("gen_tok_s", 0.0),
                          "pool_fresh_frac": ds_.get("pool_fresh_frac", 1.0), "outlier_frac": ds_.get("outlier_frac", 0.0),
                          "docs_iterated": ds_.get("docs_iterated", 0), "docs_used": ds_.get("docs_used", 0),
                          "tokens_generated": ds_.get("tokens_generated", 0), "pool_drawable": ds_.get("pool_drawable", 0),
                          "gen_wait_frac": (ds_.get("wait_time_s", 0.0) - m_prev.get("wait_time_s", 0.0)) / max(dt, 1e-9)})
                m_prev = {"gen_time_s": gt, "wait_time_s": ds_.get("wait_time_s", 0.0)}
            eta_h = (steps - step) / max(its, 1e-9) / 3600
            log(f"step {step}/{steps} loss={m['loss']:.4f} mse={m['mse']:.4f} auxk={m['auxk']:.4f} EV={m['ev']:.4f} "
                f"L0={m['l0']:.1f} dead={m['dead_frac']*100:.2f}% thr={m['threshold']:.4f} lr={m['lr']:.2e} "
                f"{its:.2f} it/s {m['tok_s']/1e3:.0f}k tok/s data-wait={data_frac*100:.0f}% ETA {eta_h:.1f}h"
                + (f" mem={m['gpu_mem_gb']:.0f}G" if "gpu_mem_gb" in m else "")
                + (f" | gen {m['gen_frac']*100:.0f}% {m['gen_tok_s']/1e3:.1f}k tok/s fresh={m['pool_fresh_frac']:.2f} "
                   f"outl={m['outlier_frac']*100:.2f}% docs={m['docs_used']}" if "gen_frac" in m else ""))
            if wb is not None:
                wb.log(m, step=step)

        # stop requests: signal on any rank, or the time budget (rank 0) -> everybody saves this step
        want_stop = STOP["flag"] or (args.max_hours is not None and rank == 0 and (time.time() - t0) / 3600 >= args.max_hours)
        stop_flag.fill_(1.0 if want_stop else 0.0)
        all_reduce_(stop_flag, "max")
        do_stop = bool(stop_flag.item() > 0)

        if step % args.save_every == 0 or step >= steps or do_stop:
            ts = time.time()
            p = tr.save(args.save_dir, step, tokens_seen, extra={"args": vars(args), "gen_state": src.state()}, keep=args.keep_ckpts)
            gs = src.state()
            if gs is not None:
                gs.update({"step": step, "tokens_seen": tokens_seen, "rank": rank})
                with open(os.path.join(args.save_dir, f"rank{rank}", f"gen_state_step{step}.json"), "w") as f:
                    json.dump(gs, f)
                for old in glob.glob(os.path.join(args.save_dir, f"rank{rank}", "gen_state_step*.json")):
                    if _step_of(old) not in {_step_of(q) for q in glob.glob(os.path.join(args.save_dir, f"rank{rank}", "ae_shard_step*.pt"))}:
                        try:
                            os.remove(old)
                        except OSError:
                            pass
            if world > 1:
                dist.barrier()
            write_config(step, tokens_seen)
            if rank == 0:
                with open(os.path.join(args.save_dir, "latest.json"), "w") as f:
                    json.dump({"step": step, "tokens_seen": tokens_seen, "world": world, "done": bool(step >= steps)}, f)
            log(f"saved shard set step {step} -> {os.path.dirname(p)} ({time.time() - ts:.0f}s)")
            saved_final = step >= steps
        if do_stop:
            log(f"STOP requested ({STOP['why'] or 'time budget'}) at step {step}; checkpoint written. Relaunch with --resume.")
            break

    if saved_final and rank == 0:
        open(os.path.join(args.save_dir, "TRAIN_DONE"), "w").write(f"{step} {tokens_seen}\n")
        log(f"TRAIN DONE steps={step} tokens={tokens_seen:,} in {(time.time() - t0) / 3600:.2f}h")
    if wb is not None:
        wb.finish()
    src.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    # Hard exit: skip interpreter finalisation. With the online generator the HF-datasets/pyarrow streaming threads make
    # finalisation abort (SIGABRT, seen on a network-filesystem box with gen_acts) or deadlock (Modal smoke 2026-09-10: all ranks hung >1.7 h after
    # TRAIN DONE, torchrun never returned). Everything is flushed above (checkpoints, latest.json, TRAIN_DONE, wandb).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
