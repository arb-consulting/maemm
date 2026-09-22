"""Shuffling disk-backed activation buffer feeding dictionary_learning's trainSAE.

Reads fp16 [N,d] shards written by sae27b_gen_acts.py, keeps a CPU fp16 pool of ~pool_tokens
activations, shuffles it, and yields [out_batch_size, d] fp32 GPU batches. Refills the pool
from the (globally shuffled) shard order when half-consumed; loops indefinitely so the trainer
can request `steps` batches. Mimics dictionary_learning.ActivationBuffer's __next__ contract.
"""
import os, glob, json, random
import numpy as np
import torch


class DiskActBuffer:
    def __init__(self, act_dir, d, out_batch_size, device="cuda",
                 pool_tokens=3_000_000, seed=0, pool_device="cuda"):
        self.d = d
        self.out_batch_size = out_batch_size
        self.device = device
        self.pool_device = pool_device  # keep the shuffled pool resident here (GPU avoids per-step H2D)
        self.pool_tokens = pool_tokens
        shards = []
        for mf in sorted(glob.glob(os.path.join(act_dir, "manifest_r*.json"))):
            m = json.load(open(mf))
            for e in m["shards"]:
                shards.append((os.path.join(act_dir, e["path"]), int(e["n"])))
        assert shards, f"no shards found in {act_dir}"
        self.shards = shards
        self.total = sum(n for _, n in shards)
        self.rng = random.Random(seed)
        self.order = []
        self.shard_ptr = 0
        self.pool = torch.empty(0, d, dtype=torch.float16, device=pool_device)
        self.ptr = 0  # sequential read position into the (pre-shuffled) pool
        self.config = {
            "d_submodule": d, "io": "out", "out_batch_size": out_batch_size,
            "total_tokens": self.total, "n_shards": len(shards),
            "pool_tokens": pool_tokens, "act_dir": act_dir,
        }
        print(f"[DiskActBuffer] {len(shards)} shards, {self.total:,} activations, "
              f"pool={pool_tokens:,}", flush=True)

    def _next_shard(self):
        if self.shard_ptr >= len(self.order):
            self.order = list(range(len(self.shards)))
            self.rng.shuffle(self.order)
            self.shard_ptr = 0
        path, n = self.shards[self.order[self.shard_ptr]]
        self.shard_ptr += 1
        arr = np.fromfile(path, dtype=np.float16).reshape(-1, self.d)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.pool_device)

    def refresh(self):
        # Carry over the unconsumed tail, load fresh shards, shuffle ONCE, reset pointer.
        # The pool is pre-shuffled and (by default) GPU-resident so per-step reads are cheap
        # GPU slices with no host->device copy or CPU indexing on the training hot path.
        tail = self.pool[self.ptr:].clone() if self.pool.shape[0] else self.pool
        self.pool = torch.empty(0, self.d, dtype=torch.float16, device=self.pool_device)
        parts = [tail]
        cur = tail.shape[0]
        while cur < self.pool_tokens:
            s = self._next_shard()
            parts.append(s)
            cur += s.shape[0]
        pool = torch.cat(parts, dim=0)
        del parts, tail
        perm = torch.randperm(pool.shape[0], device=self.pool_device)
        self.pool = pool[perm].contiguous()
        del pool
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pool.shape[0] - self.ptr < self.out_batch_size:
            self.refresh()
        batch = self.pool[self.ptr: self.ptr + self.out_batch_size]
        self.ptr += self.out_batch_size
        return batch.to(self.device, dtype=torch.float32)
