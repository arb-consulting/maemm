"""ONLINE layer-42 activation generation for the 2M-feature SAE (no stored activation shards).

Each torchrun rank owns one GPU and runs BOTH its feature shard of the SAE and its own copy of Qwen3.6-27B truncated to
decoder layers 0..LAYER (43 of 64 layers; the lm_head is dropped) -- ~36 GB bf16 instead of 54 GB. Text comes from a
DISJOINT per-rank stream of Ultra-FineWeb, tokenised into non-overlapping [BOS] + S content-token sequences exactly as
scripts/sae27b_gen_acts.py did for the stored 1B-token activation set (the 131k SAE's training data):

    ids = tok(text, add_special_tokens=False)            # no chat template, no EOS
    for s in range(0, len(ids) - S + 1, S): window = ids[s:s+S]   # docs shorter than S tokens are dropped, tails dropped
    forward([BOS] + window) -> layer-LAYER resid_post [S+1, d]; keep positions 1..S (drop ONLY the BOS/sink position)
    drop tokens whose ||x|| > norm_mult * median(||x|| over the micro-batch)   (attention-sink / outlier guard, 10x)

Corpus split (FIXED 2026-09-11): ONE single stream `load_dataset(streaming).skip(dataset_skip)` (drops the reserved eval head,
docs 0..dataset_skip-1) and rank r takes every world-th document of it: itertools.islice(stream, rank, None, world), i.e. rank r
sees single-stream docs dataset_skip + r + world*k (k = local index). Disjoint by construction, no `datasets` sharding semantics
involved. The doc id scheme gid = k*world + rank therefore equals `single-stream index - dataset_skip`.
  BUG HISTORY: the first version did split_dataset_by_node(ds, rank, world).skip(dataset_skip). In datasets 4.5 `.skip()` on a
  distributed IterableDataset builds a SkipExamplesIterable with split_when_sharding=False, whose shard_data_sources() returns
  `self` -- the per-rank shard selection is silently DROPPED and every rank iterates the full stream. All 8 ranks of the 1B-token
  training run (2026-09-11) and of the smoke max-acts saw IDENTICAL documents (79% duplicate
  windows in the smoke max-acts). Cost of the fix: every rank decodes all rows of the shared stream (~8x the parquet decoding,
  ~2M rows/rank for 1B tokens = seconds) but tokenises only its own.

Components (all CPU-testable with fakes except the model):
    load_truncated_model      -- Qwen3.6-27B layers [0, layer] only, lm_head replaced by a raiser, bf16, one GPU
    WindowProducer            -- background thread: stream docs -> tokenise -> micro-batches of S-token windows
                                 (dicts {ids, prefix, doc, start} = the format sae27b_maxacts_sharded.build_windows wants)
    OnlineActGenerator        -- producer + model: forward_batch() raw fp32 acts, next_chunk() filtered fp16 rows
    OnlinePool                -- per-rank GPU shuffle pool; every row is drawn at most `reuse_max` times (1 = never reused);
                                 refills from the generator when the drawable rows fall below refill_frac * capacity
"""
import os, sys, time, queue, threading
import torch

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_DATASET = "openbmb/Ultra-FineWeb"
DEFAULT_SPLIT = "en"
BOS_FALLBACK = 248044          # Qwen3.6-27B text_config.bos_token_id (the tokenizer itself has no bos token)


class EarlyStop(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------------
def find_layers(model, n=None):
    """The decoder-layer nn.ModuleList (the one with n entries when n is given, else the longest)."""
    import torch.nn as nn
    best, best_name = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and (n is None or len(mod) == n):
            if best is None or len(mod) > len(best):
                best, best_name = mod, name
    if best is None:
        raise RuntimeError("no decoder layer ModuleList found")
    return best, best_name


class _NoHead(torch.nn.Module):
    def forward(self, *a, **k):
        raise RuntimeError("truncated model: lm_head was dropped (only layer-42 residuals are read)")


def load_truncated_model(model_name=DEFAULT_MODEL, layer=42, device="cuda:0", attn="sdpa", log=print):
    """Load Qwen3.6-27B with only decoder layers [0, layer] (layer+1 layers) in bf16 on `device`; drop lm_head.

    Primary path: truncate the config (num_hidden_layers / layer_types) BEFORE from_pretrained so the weights of the
    dropped layers are never read from disk (saves ~17 GB of I/O per rank and the transient GPU memory). Fallback: load
    the full model and slice the ModuleList (rl/rl_disagg._truncate_scorer: the HF loop is `for l in layers[:n]`, and
    layer_types is indexed by the kept i, so the layer-`layer` states are bit-identical). Returns (model, layers, info)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    n_keep = layer + 1
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    tc = cfg.text_config if hasattr(cfg, "text_config") and cfg.text_config is not None else cfg
    n_layers = int(tc.num_hidden_layers)
    assert 0 < n_keep <= n_layers, (n_keep, n_layers)
    info = {"n_layers_full": n_layers, "n_layers_kept": n_keep, "path": None}
    model = None
    if n_keep < n_layers:
        try:
            tc.num_hidden_layers = n_keep
            if getattr(tc, "layer_types", None) is not None:
                tc.layer_types = list(tc.layer_types)[:n_keep]
            model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, dtype=torch.bfloat16, attn_implementation=attn,
                                                         local_files_only=True, device_map={"": device})
            layers, lname = find_layers(model)
            assert len(layers) == n_keep, f"config truncation gave {len(layers)} layers"
            info["path"] = "config"
        except Exception as e:  # noqa
            log(f"[online_gen] config-truncated load failed ({type(e).__name__}: {str(e)[:200]}); falling back to load+slice")
            model = None
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation=attn,
                                                     local_files_only=True, device_map={"": device})
        layers, lname = find_layers(model, n=n_layers)
        if n_keep < n_layers:
            parent = model.get_submodule(lname.rsplit(".", 1)[0]) if "." in lname else model
            setattr(parent, lname.rsplit(".", 1)[-1], layers[:n_keep])
            layers, lname = find_layers(model)
            assert len(layers) == n_keep
        info["path"] = "slice"
    if hasattr(model, "lm_head") and not isinstance(model.lm_head, _NoHead):
        model.lm_head = _NoHead()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        info["gpu_alloc_gb"] = torch.cuda.memory_allocated(device) / 2**30
    info["load_s"] = time.time() - t0
    info["layers_module"] = lname
    log(f"[online_gen] model {model_name}: kept {n_keep}/{n_layers} layers via {info['path']} ({lname}), lm_head dropped, "
        f"{info.get('gpu_alloc_gb', 0):.1f} GB on {device} in {info['load_s']:.0f}s")
    return model, layers, info


class LayerCapture:
    """Forward hook on `layers[layer]` that stores the block output and aborts the forward (nothing after `layer` runs)."""

    def __init__(self, layers, layer):
        self.out = None
        self.h = layers[layer].register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.out = o[0] if isinstance(o, tuple) else o
        raise EarlyStop()

    def remove(self):
        self.h.remove()


@torch.no_grad()
def forward_layer(model, capture, ids):
    """ids [W, S+1] long on the model device -> layer output [W, S+1, d] (bf16 as produced by the block)."""
    capture.out = None
    try:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    except EarlyStop:
        pass
    assert capture.out is not None, "layer hook did not fire"
    return capture.out


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested on CPU)
# ----------------------------------------------------------------------------------------------------------------
def windows_from_ids(ids, S, prefix_len=0):
    """Non-overlapping S-token windows of a tokenised doc (== sae27b_gen_acts: range(0, len-S+1, S); short docs -> none).
    Returns list of (start, window_ids, prefix_ids) with prefix = up to prefix_len tokens preceding the window."""
    out = []
    for s in range(0, len(ids) - S + 1, S):
        out.append((s, ids[s:s + S], ids[max(0, s - prefix_len):s] if prefix_len > 0 else []))
    return out


def outlier_mask(acts, norm_mult):
    """acts [T, d] float -> bool keep-mask: ||x|| <= norm_mult * median(||x||) (norm_mult <= 0 keeps everything)."""
    if not norm_mult or norm_mult <= 0:
        return torch.ones(acts.shape[0], dtype=torch.bool, device=acts.device)
    n = acts.float().norm(dim=-1)
    return n <= norm_mult * n.median()


def shard_stream(stream, rank, world, extra_skip=0):
    """Rank r's view of an already head-skipped single stream: every world-th doc starting at index rank + world*extra_skip
    (extra_skip = docs of THIS rank already consumed, i.e. WindowProducer.docs_iterated -- resume)."""
    import itertools
    start = int(rank) + int(world) * int(extra_skip)
    return itertools.islice(iter(stream), start, None, int(world))


def open_rank_stream(dataset, split, rank, world, skip, extra_skip=0, log=print, retries=12, stagger_s=4.0):
    """Streaming Ultra-FineWeb for ONE rank (see module doc): single stream .skip(skip) -> islice(rank, None, world).
    Returns an ITERATOR of doc dicts. extra_skip: this rank's docs to drop additionally (resume). Ranks are staggered + retried."""
    from datasets import load_dataset
    time.sleep(rank * stagger_s)
    ds = None
    for attempt in range(retries):
        try:
            ds = load_dataset(dataset, split=split, streaming=True)
            break
        except Exception as e:  # noqa
            wait = min(120, 5 * 2 ** attempt)
            log(f"[online_gen r{rank}] load_dataset attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {str(e)[:120]}; retry {wait}s")
            time.sleep(wait)
    if ds is None:
        raise RuntimeError("load_dataset failed")
    if int(skip) > 0:
        ds = ds.skip(int(skip))          # single stream: the reserved eval head is gone for EVERY rank
    return shard_stream(ds, rank, world, extra_skip=extra_skip)


# ----------------------------------------------------------------------------------------------------------------
# producer thread
# ----------------------------------------------------------------------------------------------------------------
class WindowProducer:
    """Background thread: docs -> tokens -> micro-batches (lists of `micro_batch` window dicts) in a bounded queue.
    doc ids follow sae27b_maxacts_sharded: gid = doc_i * world + rank with doc_i the local index AFTER the skip
    (doc_start lets a resumed run continue the numbering). `docs_iterated` counts docs pulled from the stream
    (incl. short ones) and is what a resume must skip additionally."""

    def __init__(self, doc_iter, tokenize, S, micro_batch, rank=0, world=1, prefix_len=0, doc_start=0, text_key="content",
                 qsize=32, max_doc_tokens=None):
        self.doc_iter, self.tokenize, self.S, self.mb = doc_iter, tokenize, S, micro_batch
        self.rank, self.world, self.prefix_len, self.text_key = rank, world, prefix_len, text_key
        self.max_doc_tokens = max_doc_tokens
        self.docs_iterated = doc_start
        self.docs_used = 0
        self.windows_made = 0
        self.q = queue.Queue(maxsize=qsize)
        self.exhausted = False
        self.err = None
        self._stop = threading.Event()
        self.thr = threading.Thread(target=self._run, daemon=True)
        self.thr.start()

    def _run(self):
        batch = []
        try:
            for doc in self.doc_iter:
                if self._stop.is_set():
                    return
                doc_i = self.docs_iterated
                self.docs_iterated += 1
                text = doc[self.text_key] if isinstance(doc, dict) else doc
                if not text:
                    continue
                ids = self.tokenize(text)
                if self.max_doc_tokens and len(ids) > self.max_doc_tokens:
                    ids = ids[: self.max_doc_tokens]
                wins = windows_from_ids(ids, self.S, self.prefix_len)
                if not wins:
                    continue
                self.docs_used += 1
                gid = doc_i * self.world + self.rank
                for s, w, pref in wins:
                    batch.append({"ids": w, "prefix": pref, "doc": gid, "start": s})
                    self.windows_made += 1
                    if len(batch) >= self.mb:
                        self._put(batch)
                        batch = []
                        if self._stop.is_set():
                            return
            if batch:
                self._put(batch)
        except Exception as e:  # noqa
            self.err = e
        finally:
            self.exhausted = True
            self._put(None)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def get(self, timeout=None):
        """Next micro-batch (list of window dicts) or None when the stream is exhausted."""
        item = self.q.get(timeout=timeout)
        if item is None and self.err is not None:
            raise RuntimeError(f"window producer died: {self.err!r}")
        return item

    def state(self):
        return {"docs_iterated": int(self.docs_iterated), "docs_used": int(self.docs_used), "windows_made": int(self.windows_made)}

    def close(self):
        self._stop.set()


# ----------------------------------------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------------------------------------
class OnlineActGenerator:
    """Model + producer for one rank. next_chunk() -> fp16 [n, d] filtered layer-`layer` residuals (n <= micro_batch*S)."""

    def __init__(self, rank=0, world=1, device="cuda:0", model_name=DEFAULT_MODEL, layer=42, dataset=DEFAULT_DATASET,
                 split=DEFAULT_SPLIT, dataset_skip=100_000, extra_skip=0, doc_start=0, ctx_len=512, micro_batch=16,
                 norm_mult=10.0, d=5120, prefix_len=0, qsize=32, log=print, attn="sdpa"):
        from transformers import AutoTokenizer
        self.rank, self.world, self.device, self.S, self.mb, self.d = rank, world, device, ctx_len, micro_batch, d
        self.norm_mult, self.layer, self.log = norm_mult, layer, log
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else BOS_FALLBACK
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (
            self.tok.eos_token_id if self.tok.eos_token_id is not None else 0)
        self.model, self.layers, self.model_info = load_truncated_model(model_name, layer, device, attn=attn, log=log)
        self.capture = LayerCapture(self.layers, layer)
        stream = open_rank_stream(dataset, split, rank, world, dataset_skip, extra_skip=extra_skip, log=log)
        self.producer = WindowProducer(iter(stream), lambda t: self.tok(t, add_special_tokens=False, truncation=False)["input_ids"],
                                       ctx_len, micro_batch, rank=rank, world=world, prefix_len=prefix_len, doc_start=doc_start,
                                       text_key="content", qsize=qsize)
        self.tokens_generated = 0      # kept rows
        self.tokens_dropped = 0        # outlier rows
        self.tokens_forwarded = 0      # W*S per micro-batch
        self.gen_time = 0.0            # wall time inside forward_batch (model)
        self.wait_time = 0.0           # wall time waiting on the producer queue
        self.n_batches = 0
        self.exhausted = False

    def next_batch(self):
        """Next micro-batch of window dicts from the producer (None = corpus exhausted)."""
        t = time.time()
        b = self.producer.get()
        self.wait_time += time.time() - t
        if b is None:
            self.exhausted = True
        elif self.n_batches == 0 and not getattr(self, "_logged_first", False):
            self._logged_first = True
            w = b[0]
            self.log(f"[online_gen r{self.rank}] FIRST WINDOW doc_id={w['doc']} (single-stream idx {w['doc'] + 0}+skip) start={w['start']} "
                     f"text={self.tok.decode(w['ids'][:20])!r}")
        return b

    @torch.no_grad()
    def forward_batch(self, batch):
        """batch: list of window dicts -> raw fp32 [W*S, d] layer output on device with the BOS position dropped."""
        t = time.time()
        ids = torch.tensor([[self.bos] + w["ids"] for w in batch], device=self.device, dtype=torch.long)
        h = forward_layer(self.model, self.capture, ids)                       # [W, S+1, d]
        a = h[:, 1:, :].reshape(-1, self.d).float()
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self.gen_time += time.time() - t
        self.tokens_forwarded += a.shape[0]
        self.n_batches += 1
        return a

    @torch.no_grad()
    def next_chunk(self):
        """fp16 [n, d] rows with the outliers removed, or None when the corpus is exhausted."""
        b = self.next_batch()
        if b is None:
            return None
        a = self.forward_batch(b)
        keep = outlier_mask(a, self.norm_mult)
        n_keep = int(keep.sum())
        self.tokens_dropped += a.shape[0] - n_keep
        self.tokens_generated += n_keep
        return a[keep].to(torch.float16)

    def stats(self):
        s = self.producer.state()
        s.update({"tokens_generated": self.tokens_generated, "tokens_dropped": self.tokens_dropped, "tokens_forwarded": self.tokens_forwarded,
                  "gen_time_s": self.gen_time, "wait_time_s": self.wait_time, "n_batches": self.n_batches,
                  "gen_tok_s": self.tokens_forwarded / max(self.gen_time, 1e-9),
                  "outlier_frac": self.tokens_dropped / max(self.tokens_forwarded, 1)})
        return s

    def state(self):
        return self.producer.state()

    def close(self):
        self.producer.close()
        self.capture.remove()


# ----------------------------------------------------------------------------------------------------------------
# pool
# ----------------------------------------------------------------------------------------------------------------
class OnlinePool:
    """Per-rank shuffle pool of fp16 activation rows on `device`.

    capacity rows; `uses[i]` = remaining draws of slot i (reuse_max at fill, 0 = exhausted/replaceable). draw(n) picks n
    distinct drawable slots uniformly at random and decrements their counters; it first refills when the number of
    drawable rows is below max(n, refill_frac*capacity): every exhausted slot is overwritten with fresh generator rows
    (generator chunks that do not fit are carried over to the next refill). With reuse_max=1 every row is trained on
    exactly once (fresh fraction 1.0); reuse_max=m needs 1/m the generation and reports fresh fraction 1/m.
    gen_fn() -> fp16 [n, d] (any device) or None when the source is exhausted (then draws recycle what is left)."""

    def __init__(self, gen_fn, d, capacity, device, seed=0, reuse_max=1, refill_frac=0.5, dtype=torch.float16):
        assert reuse_max >= 1 and capacity > 0
        self.gen_fn, self.d, self.capacity, self.device = gen_fn, d, int(capacity), device
        self.reuse_max, self.refill_frac = int(reuse_max), float(refill_frac)
        self.pool = torch.empty(self.capacity, d, dtype=dtype, device=device)
        self.uses = torch.zeros(self.capacity, dtype=torch.int32, device=device)     # all exhausted -> first draw fills
        self.filled = torch.zeros(self.capacity, dtype=torch.bool, device=device)    # slots that have ever held data
        self.first_use = torch.zeros(self.capacity, dtype=torch.bool, device=device) # slot holds a row never drawn yet
        self.g = torch.Generator(device=device).manual_seed(int(seed))
        self.carry = None
        self.source_exhausted = False
        self.rows_generated = 0
        self.rows_drawn = 0
        self.rows_fresh = 0            # drawn rows on their first use
        self.n_refills = 0
        self.recycles = 0
        self.refill_time = 0.0

    def drawable(self):
        return int((self.uses > 0).sum().item())

    def _next_rows(self):
        if self.carry is not None and self.carry.shape[0] > 0:
            c, self.carry = self.carry, None
            return c
        if self.source_exhausted:
            return None
        c = self.gen_fn()
        if c is None:
            self.source_exhausted = True
            return None
        return c.to(self.device, self.pool.dtype)

    @torch.no_grad()
    def refill(self):
        t = time.time()
        slots = (self.uses == 0).nonzero(as_tuple=True)[0]
        filled = 0
        while filled < slots.numel():
            rows = self._next_rows()
            if rows is None:
                break
            take = min(rows.shape[0], slots.numel() - filled)
            idx = slots[filled:filled + take]
            self.pool[idx] = rows[:take]
            self.uses[idx] = self.reuse_max
            self.filled[idx] = True
            self.first_use[idx] = True
            filled += take
            self.rows_generated += take
            if take < rows.shape[0]:
                self.carry = rows[take:]
        self.n_refills += 1
        self.refill_time += time.time() - t
        return filled

    @torch.no_grad()
    def draw(self, n):
        """[n, d] rows on device (fp16). Refills when needed; raises if the source is exhausted and nothing is left."""
        need = max(n, int(self.refill_frac * self.capacity))
        if self.drawable() < need:
            self.refill()
        cand = (self.uses > 0).nonzero(as_tuple=True)[0]
        if cand.numel() < n:
            # source exhausted mid-run (or a generator that cannot keep up): recycle every slot that holds data
            self.uses[self.filled & (self.uses == 0)] = 1
            self.recycles += 1
            cand = (self.uses > 0).nonzero(as_tuple=True)[0]
            if cand.numel() < n:
                raise StopIteration(f"activation source exhausted: only {cand.numel()} rows in the pool for a draw of {n}")
        sel = cand[torch.randperm(cand.numel(), generator=self.g, device=self.device)[:n]]
        self.rows_fresh += int(self.first_use[sel].sum().item())
        self.first_use[sel] = False
        self.uses[sel] -= 1
        self.rows_drawn += n
        return self.pool[sel]

    def stats(self):
        return {"pool_capacity": self.capacity, "pool_drawable": self.drawable(), "pool_rows_generated": self.rows_generated,
                "pool_rows_drawn": self.rows_drawn, "pool_fresh_frac": self.rows_fresh / max(self.rows_drawn, 1),
                "pool_n_refills": self.n_refills, "pool_recycles": self.recycles, "pool_refill_time_s": self.refill_time,
                "pool_source_exhausted": self.source_exhausted}
