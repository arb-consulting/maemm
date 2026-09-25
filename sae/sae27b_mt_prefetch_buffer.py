"""DiskActBuffer with a multi-reader background prefetch thread (+ optional shard-subset for data-parallel readers).

Why: a network-filesystem share delivers ~120 MB/s per reader; a 4096x5120 fp16 batch is 42 MB, so a single
sequential reader caps training at ~3 it/s (that is what hit the 20 h wall on the F=131072 1B-token run). This buffer
  * reads the shards of the NEXT refresh concurrently with `n_readers` threads (file.readinto releases the GIL), and
  * keeps them queued in CPU RAM (queue depth 1) so refresh() only pays H2D + cat + shuffle on the training thread,
  * optionally restricts itself to shards[rank::world] (`shard_subset=(rank, world)`) so every rank of a torchrun job
    streams a disjoint 1/world of the corpus (used by sae27b_train_sharded.py --data-mode allgather).
Shard order / RNG semantics are those of DiskActBuffer (one global shuffle of the shard list per epoch).
Extra CPU RAM ~= pool_tokens x d x 2 bytes (one refresh worth of shards, e.g. 20 GB at 2M x 5120).
"""
import os, queue, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from sae27b_disk_buffer import DiskActBuffer


def read_shard_fp16(path, d):
    """Read a raw fp16 [N,d] shard into a CPU tensor; N is taken from the file size (robust to manifest drift)."""
    nbytes = os.path.getsize(path)
    n = nbytes // (2 * d)
    arr = np.empty((n, d), dtype=np.float16)
    with open(path, "rb", buffering=0) as f:
        mv = memoryview(arr).cast("B")
        got = 0
        while got < mv.nbytes:
            r = f.readinto(mv[got:])
            if not r:
                raise IOError(f"short read on {path}: {got}/{mv.nbytes} bytes")
            got += r
    return torch.from_numpy(arr)


class MTPrefetchDiskActBuffer(DiskActBuffer):
    def __init__(self, act_dir, d, out_batch_size, device="cuda", pool_tokens=3_000_000, seed=0,
                 pool_device="cuda", n_readers=4, shard_subset=None, start_prefetch=True):
        super().__init__(act_dir, d, out_batch_size, device=device, pool_tokens=pool_tokens,
                         seed=seed, pool_device=pool_device)
        if shard_subset is not None:
            r, w = shard_subset
            self.shards = self.shards[r::w]
            assert self.shards, f"shard subset {shard_subset} is empty"
            self.total = sum(n for _, n in self.shards)
            self.config["total_tokens"] = self.total
            self.config["n_shards"] = len(self.shards)
        self.n_readers = max(1, int(n_readers))
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._err = None
        self._thr = None
        if start_prefetch:
            self.start()

    def start(self):
        if self._thr is None:
            self._thr = threading.Thread(target=self._worker, daemon=True)
            self._thr.start()

    # -- shard planning (worker thread only; uses the base-class RNG/order) --
    def _plan(self):
        paths, cur = [], 0
        while cur < self.pool_tokens:
            if self.shard_ptr >= len(self.order):
                self.order = list(range(len(self.shards)))
                self.rng.shuffle(self.order)
                self.shard_ptr = 0
            path, n = self.shards[self.order[self.shard_ptr]]
            self.shard_ptr += 1
            paths.append(path)
            cur += n
        return paths

    def _load_parts(self):
        paths = self._plan()
        with ThreadPoolExecutor(max_workers=min(self.n_readers, len(paths))) as ex:
            parts = list(ex.map(lambda p: read_shard_fp16(p, self.d), paths))
        return parts

    def _worker(self):
        try:
            while not self._stop.is_set():
                parts = self._load_parts()
                while not self._stop.is_set():
                    try:
                        self._q.put(parts, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # surfaced on the training thread at the next refresh
            self._err = e

    def refresh(self):
        if self._err is not None:
            raise RuntimeError(f"prefetch worker died: {self._err!r}")
        if self._thr is None:
            self.start()
        tail = self.pool[self.ptr:].clone() if self.pool.shape[0] else self.pool
        self.pool = torch.empty(0, self.d, dtype=torch.float16, device=self.pool_device)  # free before cat
        parts = self._q.get()
        parts = [p.to(self.pool_device) for p in parts]
        pool = torch.cat([tail] + parts, dim=0)
        del parts, tail
        perm = torch.randperm(pool.shape[0], device=self.pool_device)
        self.pool = pool[perm].contiguous()
        del pool
        self.ptr = 0

    def next_raw(self):
        """[out_batch_size, d] fp16 slice on pool_device (no dtype cast) -- what the trainer broadcasts/all-gathers."""
        if self.pool.shape[0] - self.ptr < self.out_batch_size:
            self.refresh()
        batch = self.pool[self.ptr: self.ptr + self.out_batch_size]
        self.ptr += self.out_batch_size
        return batch

    def close(self):
        self._stop.set()
