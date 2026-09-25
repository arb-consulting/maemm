"""CPU/gloo tests for the ONLINE data path of the 2M-SAE trainer. Run (2 ranks):
    torchrun --standalone --nproc_per_node 2 tests/sae/test_online_pool.py
(f) online_gen pure helpers: windows_from_ids == gen_acts range semantics (+prefix), outlier_mask 10x-median rule
(g) WindowProducer: exact micro-batches, doc-id scheme doc_i*world+rank, docs_iterated/docs_used accounting, resume offset,
    exhaustion sentinel
(h) OnlinePool: draw-without-reuse (every generated row drawn at most once, fresh fraction 1.0), refill fires below half,
    chunk carry-over, reuse_max=2 halves generation (fresh fraction -> 0.5), source exhaustion recycles
(i) BatchSource[online] on 2 gloo ranks with rank-tagged fake generators: identical [B,d] batch on all ranks assembled from
    B/R rows of EVERY rank, no row repeats across steps, state()/stats()/close() plumbing; a mini training run + save writes
    the per-rank gen_state json the resume path reads
"""
import os, sys, json, math, tempfile, shutil
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
from online_gen import windows_from_ids, outlier_mask, WindowProducer, OnlinePool, shard_stream
from sae27b_train_sharded import BatchSource, ShardedTrainer, all_reduce_, world_size, rank_id

D = 16


def check(cond, msg):
    ok = torch.tensor([1.0 if cond else 0.0])
    all_reduce_(ok, "min")
    if ok.item() < 1:
        raise AssertionError(f"[rank{rank_id()}] FAILED: {msg}")
    if rank_id() == 0:
        print(f"  PASS {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------
def test_f_helpers():
    S = 8
    ids = list(range(100, 100 + 3 * S + 5))                          # 29 tokens -> 3 windows, tail of 5 dropped
    w = windows_from_ids(ids, S, prefix_len=3)
    ref = [(s, ids[s:s + S]) for s in range(0, len(ids) - S + 1, S)]  # sae27b_gen_acts loop
    ok = [(a, b) for a, b, _ in w] == ref and len(w) == 3
    ok &= w[0][2] == [] and w[1][2] == ids[S - 3:S] and w[2][2] == ids[2 * S - 3:2 * S]
    ok &= windows_from_ids(list(range(S - 1)), S) == [] and len(windows_from_ids(list(range(S)), S)) == 1
    ok &= windows_from_ids(ids, S)[1][2] == []
    check(ok, "(f) windows_from_ids == gen_acts non-overlapping range(0, n-S+1, S); short docs dropped; prefix = up to L-1 preceding tokens")
    g = torch.Generator().manual_seed(0)
    a = torch.randn(1000, D, generator=g)
    a[7] *= 500.0; a[42] *= 50.0
    m = outlier_mask(a, 10.0)
    ref = a.norm(dim=-1) <= 10.0 * a.norm(dim=-1).median()
    ok = torch.equal(m, ref) and (~m).sum() == 2 and not m[7] and not m[42]
    ok &= outlier_mask(a, 0).all() and outlier_mask(a, None).all()
    check(ok, "(f) outlier_mask drops exactly ||x|| > 10 x median (2 planted outliers); mult<=0 keeps all")


# ------------------------------------------------------------------------------------------------------------
def test_f2_shard_stream():
    """Ranks partition a head-skipped stream: rank r gets docs skip + r + world*k, disjoint, union == everything after skip;
    resume (extra_skip = docs_iterated of that rank) continues exactly; matches the WindowProducer doc-id scheme."""
    skip, world, n = 7, 8, 500
    single = list(range(n))                                    # doc "texts" = their single-stream index
    head_skipped = single[skip:]
    views = [list(shard_stream(head_skipped, r, world)) for r in range(world)]
    ok = all(v == single[skip + r::world] for r, v in enumerate(views))
    ok &= len(set().union(*map(set, views))) == n - skip and sum(map(len, views)) == n - skip     # disjoint + complete
    ok &= all(min(v) >= skip for v in views) and len({v[0] for v in views}) == world               # no head leak, distinct first docs
    # doc-id scheme: gid = k*world + rank == single-stream index - skip
    ok &= all(views[r][k] - skip == k * world + r for r in range(world) for k in range(len(views[r])))
    # resume: after this rank consumed 13 docs, extra_skip=13 continues with its 14th doc
    ok &= list(shard_stream(head_skipped, 3, world, extra_skip=13)) == views[3][13:]
    # world 1: identity
    ok &= list(shard_stream(head_skipped, 0, 1)) == head_skipped
    check(ok, f"(f2) shard_stream: {world} ranks partition the head-skipped stream disjointly (rank r = docs skip+r+{world}k), "
              f"no doc < skip leaks, gid == single-stream idx - skip, resume offset exact")


def fake_docs(n, S, seed, short_every=3):
    """doc i = list of ints (identity tokenizer): length varies; every short_every-th doc is shorter than S."""
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        L = int(rng.integers(1, S)) if i % short_every == 1 else int(rng.integers(S, 4 * S + 3))
        docs.append([i * 10_000 + t for t in range(L)])
    return docs


def test_g_producer():
    S, mb, R, r = 8, 4, world_size(), rank_id()
    docs = fake_docs(30, S, seed=1)
    prod = WindowProducer(iter(docs), lambda t: t, S, mb, rank=r, world=R, prefix_len=3)
    batches = []
    while True:
        b = prod.get(timeout=30)
        if b is None:
            break
        batches.append(b)
    wins = [w for b in batches for w in b]
    exp_wins = [(i, s, w, p) for i, d in enumerate(docs) for s, w, p in windows_from_ids(d, S, 3)]
    ok = len(wins) == len(exp_wins) and all(len(b) == mb for b in batches[:-1]) and 0 < len(batches[-1]) <= mb
    ok &= all(w["ids"] == e[2] and w["prefix"] == e[3] and w["start"] == e[1] and w["doc"] == e[0] * R + r for w, e in zip(wins, exp_wins))
    st = prod.state()
    n_long = sum(1 for d in docs if len(d) >= S)
    ok &= st["docs_iterated"] == 30 and st["docs_used"] == n_long and st["windows_made"] == len(exp_wins) and prod.exhausted
    check(ok, f"(g) WindowProducer: {len(batches)} micro-batches of {mb} windows == windows_from_ids over {len(docs)} docs; "
              f"doc ids = doc_i*{R}+{r}; docs_iterated=30 docs_used={n_long}; exhaustion sentinel")
    # resume: doc_start offsets the local index (and the id scheme) exactly like skipping docs_iterated docs of the stream
    prod2 = WindowProducer(iter(docs[12:]), lambda t: t, S, mb, rank=r, world=R, prefix_len=3, doc_start=12)
    w2 = []
    while True:
        b = prod2.get(timeout=30)
        if b is None:
            break
        w2 += b
    ref2 = [w for w in wins if w["doc"] // R >= 12]
    ok = len(w2) == len(ref2) and all(a == b for a, b in zip(w2, ref2)) and prod2.state()["docs_iterated"] == 30
    check(ok, "(g) resume: producer over docs[12:] with doc_start=12 reproduces exactly the windows/doc-ids of the tail")
    # empty doc / dict docs
    prod3 = WindowProducer(iter([{"content": ""}, {"content": list(range(S))}]), lambda t: t, S, mb, rank=r, world=R)
    b = prod3.get(timeout=30); e = prod3.get(timeout=30)
    check(b is not None and len(b) == 1 and b[0]["doc"] == 1 * R + r and e is None and prod3.state()["docs_iterated"] == 2,
          "(g) dict docs (text_key=content): empty doc skipped but counted, short/empty docs never make windows")


# ------------------------------------------------------------------------------------------------------------
def ids_of(x):
    """Row ids of a TaggedGen tensor: id = col0 * 1024 + col1 (both < 1024 so they are exact in fp16; a single fp16 column
    only represents integers exactly up to 2048)."""
    return (x[:, 0].float() * 1024 + x[:, 1].float()).round().long().tolist()


def tags_of(x):
    return x[:, 2].float().round().long().tolist()


class TaggedGen:
    """Fake activation source: chunks of `chunk` fp16 rows, row = [id_hi, id_lo, tag, noise...]; ids are unique forever."""

    def __init__(self, tag, chunk=37, limit=None, seed=0):
        self.tag, self.chunk, self.limit = tag, chunk, limit
        self.next_id = 0
        self.calls = 0
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self):
        if self.limit is not None and self.calls >= self.limit:
            return None
        self.calls += 1
        x = torch.randn(self.chunk, D, generator=self.g)
        ids = torch.arange(self.next_id, self.next_id + self.chunk)
        assert ids.max() < 1024 * 1024
        x[:, 0] = (ids // 1024).float()
        x[:, 1] = (ids % 1024).float()
        x[:, 2] = float(self.tag)
        self.next_id += self.chunk
        return x.to(torch.float16)


def test_h_pool():
    cap, n = 200, 16
    gen = TaggedGen(tag=1, chunk=37)
    pool = OnlinePool(gen, D, cap, "cpu", seed=3, reuse_max=1, refill_frac=0.5, dtype=torch.float16)
    seen = {}
    refill_points = []
    for step in range(120):
        before = pool.drawable()
        x = pool.draw(n)
        after_refills = pool.n_refills
        for i in ids_of(x):
            seen[i] = seen.get(i, 0) + 1
        if step > 0 and after_refills > refill_points[-1][1] if refill_points else False:
            refill_points.append((step, after_refills, before))
        elif not refill_points:
            refill_points.append((step, after_refills, before))
        assert x.shape == (n, D) and x.dtype == torch.float16
    ok = max(seen.values()) == 1 and len(seen) == 120 * n
    ok &= pool.stats()["pool_fresh_frac"] == 1.0 and pool.rows_drawn == 120 * n
    # every refill after the first happened when drawable < cap/2, and left the pool full
    ok &= all(b < cap // 2 for s, k, b in refill_points[1:]) and pool.drawable() >= cap // 2
    ok &= pool.rows_generated == pool.n_refills * 0 + pool.rows_generated  # tautology guard (no crash on stats)
    # generated rows == drawn + still drawable + carry (nothing lost)
    carry = 0 if pool.carry is None else pool.carry.shape[0]
    ok &= gen.next_id == pool.rows_generated + carry and pool.rows_generated == pool.rows_drawn + pool.drawable()
    check(ok, f"(h) OnlinePool reuse_max=1: {120 * n} rows drawn, all distinct (max use 1), fresh_frac=1.0; {pool.n_refills} refills "
              f"each triggered below cap/2; chunk carry-over conserves rows ({gen.next_id} generated = drawn + drawable + carry {carry})")
    # reuse_max=2: half the generation, fresh fraction -> 0.5, no row used more than twice
    gen2 = TaggedGen(tag=2, chunk=50)
    pool2 = OnlinePool(gen2, D, cap, "cpu", seed=4, reuse_max=2, refill_frac=0.5)
    seen2 = {}
    for step in range(400):
        for i in ids_of(pool2.draw(n)):
            seen2[i] = seen2.get(i, 0) + 1
    ff = pool2.stats()["pool_fresh_frac"]
    ok = max(seen2.values()) <= 2 and 0.45 <= ff <= 0.6 and pool2.rows_generated < 0.6 * pool2.rows_drawn
    check(ok, f"(h) OnlinePool reuse_max=2: max use 2, fresh_frac={ff:.3f} (~0.5), generated {pool2.rows_generated} for {pool2.rows_drawn} drawn")
    # exhaustion: the source stops -> the pool recycles what it holds instead of crashing; fresh_frac drops below 1
    gen3 = TaggedGen(tag=3, chunk=40, limit=6)   # 240 rows total
    pool3 = OnlinePool(gen3, D, cap, "cpu", seed=5, reuse_max=1)
    got = 0
    for _ in range(30):
        got += pool3.draw(n).shape[0]
    ok = got == 30 * n and pool3.source_exhausted and pool3.stats()["pool_fresh_frac"] < 1.0 and pool3.rows_generated == 240
    check(ok, "(h) OnlinePool: exhausted source -> recycles pooled rows (fresh_frac < 1) rather than failing; all 240 generated rows used")


# ------------------------------------------------------------------------------------------------------------
def test_i_batchsource(tmp):
    R, r = world_size(), rank_id()
    B = 32
    closed = {"n": 0}
    gen = TaggedGen(tag=r, chunk=45, seed=10 + r)
    online = dict(gen_fn=gen, total_tokens=12345, capacity=300, reuse_max=1, refill_frac=0.5,
                  state_fn=lambda: {"docs_iterated": 7 + r, "docs_used": 5}, stats_fn=lambda: {"gen_tok_s": 1.0, "gen_time_s": 0.5},
                  close_fn=lambda: closed.__setitem__("n", closed["n"] + 1))
    src = BatchSource(None, D, B, "online", "cpu", 0, 0, "cpu", 0, online=online)
    ok = src.total == 12345
    ids_seen = set()
    for step in range(25):
        x = src.next()
        x0 = x.clone(); dist.broadcast(x0, src=0)
        ok &= torch.equal(x, x0) and x.shape == (B, D) and x.dtype == torch.float32
        tags = tags_of(x)
        ok &= all(tags[i] == i // (B // R) for i in range(B))                    # rows [j*B/R:(j+1)*B/R] came from rank j
        keys = {(t, i) for t, i in zip(tags, ids_of(x))}
        ok &= not (keys & ids_seen) and len(keys) == B
        ids_seen |= keys
    st, stt = src.state(), src.stats()
    ok &= st == {"docs_iterated": 7 + r, "docs_used": 5} and stt["gen_tok_s"] == 1.0 and "pool_fresh_frac" in stt and stt["pool_fresh_frac"] == 1.0
    src.close()
    ok &= closed["n"] == 1
    check(ok, f"(i) BatchSource[online] {R} ranks: identical [B,d] batch everywhere = concat of B/R rows from each rank's generator, "
              f"{25 * B} rows never repeated; state()/stats()/close() plumbing")
    # mini training in online mode + save -> gen_state json per rank (what --resume reads)
    F, K = 64, 4
    gen = TaggedGen(tag=r, chunk=64, seed=20 + r)

    def gaussian_gen():
        c = gen()
        g = torch.Generator().manual_seed(ids_of(c)[0] + 1000 * r)
        mu = torch.tensor([3.0, -2.0] + [0.0] * (D - 2))
        return (torch.randn(c.shape[0], D, generator=g) * torch.linspace(0.5, 3, D) + mu).to(torch.float16)
    online = dict(gen_fn=gaussian_gen, total_tokens=40 * B, capacity=400, reuse_max=1, refill_frac=0.5,
                  state_fn=lambda: {"docs_iterated": 100 + r, "docs_used": 90})
    src = BatchSource(None, D, B, "online", "cpu", 0, 1, "cpu", 0, online=online)
    steps = src.total // B
    nf = math.sqrt(sum(src.next().pow(2).sum(1).mean().item() for _ in range(3)) / 3)
    tr = ShardedTrainer(D, F, K, "cpu", lr=3e-3, steps=steps, warmup_steps=3, decay_start=30, auxk_alpha=1 / 32, top_k_aux_local=2,
                        dead_tokens=B * 8, threshold_start_step=3, seed=7, use_autocast=False)
    tr.norm_factor = nf
    losses = []
    for s in range(steps):
        loss, st = tr.train_step(src.next() / nf, s)
        losses.append(loss.item())
    save_dir = os.path.join(tmp, "save_online")
    tr.save(save_dir, steps, steps * B, extra={"gen_state": src.state()}, keep=1)
    gs = src.state(); gs.update({"step": steps, "tokens_seen": steps * B, "rank": r})
    os.makedirs(os.path.join(save_dir, f"rank{r}"), exist_ok=True)
    json.dump(gs, open(os.path.join(save_dir, f"rank{r}", f"gen_state_step{steps}.json"), "w"))
    dist.barrier()
    back = json.load(open(os.path.join(save_dir, f"rank{r}", f"gen_state_step{steps}.json")))
    ck = torch.load(os.path.join(save_dir, f"rank{r}", f"ae_shard_step{steps}.pt"), map_location="cpu", weights_only=False)
    ok = all(map(math.isfinite, losses)) and sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    ok &= back["docs_iterated"] == 100 + r and ck["gen_state"]["docs_iterated"] == 100 + r and back["step"] == steps
    check(ok, f"(i) online-mode mini training: loss {sum(losses[:5]) / 5:.4f} -> {sum(losses[-5:]) / 5:.4f} over {steps} steps; "
              f"shard ckpt + gen_state_step{steps}.json carry docs_iterated per rank for --resume")
    src.close()


def main():
    dist.init_process_group("gloo")
    R, r = world_size(), rank_id()
    torch.set_num_threads(2)
    if r == 0:
        print(f"== online data-path tests: world={R} d={D} ==", flush=True)
    test_f_helpers()
    test_f2_shard_stream()
    test_g_producer()
    test_h_pool()
    tmp = [tempfile.mkdtemp(prefix="sae2m_online_") if r == 0 else None]
    dist.broadcast_object_list(tmp, src=0)
    try:
        test_i_batchsource(tmp[0])
    finally:
        dist.barrier()
        if r == 0:
            shutil.rmtree(tmp[0], ignore_errors=True)
    dist.barrier()
    if r == 0:
        print("ALL ONLINE TESTS PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
