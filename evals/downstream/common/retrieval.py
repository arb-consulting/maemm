"""The corpus-search baseline every package runs: for each unit query direction, the held-out web windows
whose read-layer residuals point most along it (the corpus and its content check: `assets/README.md`).

The search corpus is the held-out order's first 10M tokens in 64-token windows at stride 16, ranked at
nested `SIZES` out of one forward; packages draw their evaluation text from the documents after it. A window
scores `max_t cos(h_t, q)` over `[sink] + window` on the clean base, and a query keeps its `top_k` best
non-overlapping windows. `score_corpus_part` (score matrices) and `score_corpus_topk` (a running `TopK`)
shard by contiguous blocks and merge order-independently. Torch is imported lazily.
"""

import dataclasses
import functools
import hashlib
import json
import os
import time

import numpy as np

from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance
from maem.config import READ_LAYER

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
HELDOUT_INDEX = os.path.join(ASSETS, "heldout_docs.csv")
HELDOUT_INDEX_META = os.path.join(ASSETS, "heldout_docs.meta.json")
HELDOUT_OVERLAP = os.path.join(ASSETS, "heldout_overlap.csv")
HELDOUT_OVERLAP_META = os.path.join(ASSETS, "heldout_overlap.meta.json")
#: word 7-gram share in the training text at which a document is left out of evaluation draws
OVERLAP_THRESHOLD = 0.05


@dataclasses.dataclass(frozen=True)
class Stream:
    """The public files the corpus is rebuilt from, pinned to one revision. Reads like a mapping
    (`stream["revision"]`, `dict(stream)`) and survives `dataclasses.asdict`."""

    dataset: str = "openbmb/Ultra-FineWeb"
    config: str = "default"
    split: str = "en"
    revision: str = "02c85641e3d19a854be2e09139c25adaa9518063"
    #: part 0 and part 1 of the index's `part` column, in that order
    files: tuple = ("data/ultrafineweb_en/ultrafineweb-en-part-0009-of-2048.parquet",
                    "data/ultrafineweb_en/ultrafineweb-en-part-0010-of-2048.parquet")

    def keys(self):
        return ("dataset", "config", "split", "revision", "files")

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        value = getattr(self, key)
        return list(value) if key == "files" else value


DATASET = Stream()
TEXT_COLUMNS = ("content", "text")
SCORING = "max-over-positions uncentred cosine of a window's read-layer residuals against the unit query"

HELDOUT_DOCS = 18_813
#: the search corpus: the held-out prefix of this many tokens (a document boundary) and its document count
CORPUS_TOKENS = 9_999_741
CORPUS_DOCS = 11_700
SIZES = (("1M", 996_497), ("2M", 1_999_739), ("4M", 3_999_724), ("8M", 7_999_674), ("10M", CORPUS_TOKENS))
#: the size quoted beside the paper's reconstruction evaluation: the largest prefix both corpora share
SHARED_SIZE = "8M"

WINDOW = 64                        # one window, in the dump's own tokens: also a reader's output budget
STRIDE = 16                        # window i starts here x i, so an interior token is in four windows
WINDOW_BATCH = 128                 # windows per forward pass of a corpus pass
#: most candidates `TopK` holds per query; bounds `candidate_depth(top_k)`, so top_k <= 9 on the 64/16 grid
MAX_TOP_K = 64
SUPPRESSION = ("greedy in rank order: a window is kept unless it shares a token with an already-kept window "
               "of its own document")


def window_reach(window=WINDOW, stride=STRIDE):
    """How many other windows of its document one window can share a token with: six on the 64/16 grid."""
    return 2 * (-(-int(window) // int(stride)) - 1)


def candidate_depth(k, window=WINDOW, stride=STRIDE):
    """How many best-by-score windows must be held so the `k` best non-overlapping ones are among them:
    `(1 + window_reach) * k`, since each kept window rules out at most `window_reach` others."""
    return (1 + window_reach(window, stride)) * int(k)


def check_top_k(k, window=WINDOW, stride=STRIDE):
    """Raise unless `k` non-overlapping windows fit the candidate buffer a running ranking holds."""
    most = MAX_TOP_K // (1 + window_reach(window, stride))
    if not 1 <= int(k) <= most:
        raise ValueError(f"top_k={k} is outside 1..{most}: a query's k non-overlapping windows are taken out "
                         f"of its {1 + window_reach(window, stride)} x k best by score "
                         f"(`candidate_depth`), and a running ranking holds at most MAX_TOP_K = {MAX_TOP_K} "
                         f"candidates per query")
    return int(k)


# ---------------------------------------------------------------- the held-out index

@dataclasses.dataclass(frozen=True)
class Doc:
    """One held-out document: `doc` its position in the held-out order, `(part, row)` its row in
    `Stream.files`, `offset` its first token's index in the order."""

    doc: int
    part: int
    row: int
    n_tokens: int
    offset: int


@functools.lru_cache(maxsize=None)
def load_index(path=HELDOUT_INDEX, meta_path=HELDOUT_INDEX_META):
    """The held-out order as a tuple of `Doc`. Refuses a file whose sha256 or counts differ from
    `heldout_docs.meta.json`. Cached on the path."""
    with open(meta_path, encoding="utf-8") as h:
        meta = json.load(h)
    with open(path, "rb") as h:
        raw = h.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: sha256 {digest} is not the recorded {meta['sha256']}")
    lines = raw.decode("utf-8").splitlines()
    if lines[0] != "doc,part,row,len":
        raise ValueError(f"{path}: header {lines[0]!r} is not 'doc,part,row,len'")
    docs, offset = [], 0
    for k, line in enumerate(lines[1:]):
        doc, part, row, n = (int(v) for v in line.split(","))
        if doc != k:
            raise ValueError(f"{path}: record {k} carries doc={doc}; the file is the held-out ORDER")
        docs.append(Doc(doc=doc, part=part, row=row, n_tokens=n, offset=offset))
        offset += n
    if len(docs) != meta["n_docs"] or offset != meta["n_tokens"]:
        raise ValueError(f"{path}: {len(docs)} documents of {offset} tokens, and the recorded corpus is "
                         f"{meta['n_docs']} of {meta['n_tokens']}")
    return tuple(docs)


def n_windows(n_tokens, window=WINDOW, stride=STRIDE):
    """Windows in a document: `max(1, ceil((n - window) / stride) + 1)`, the last one possibly short."""
    n = int(n_tokens)
    if n <= 0:
        raise ValueError(f"a document of {n} tokens has no window")
    return max(1, -(-(n - int(window)) // int(stride)) + 1)


def total_windows(docs, window=WINDOW, stride=STRIDE):
    """The window count of a document list, from the index alone."""
    return int(sum(n_windows(d.n_tokens, window, stride) for d in docs))


def search_docs(index, spec=None):
    """The search corpus: the held-out prefix of `corpus_tokens` tokens (a document boundary)."""
    tokens = int((spec or CorpusSpec()).corpus_tokens)
    docs = tuple(d for d in index if d.offset + d.n_tokens <= tokens)
    got = docs[-1].offset + docs[-1].n_tokens if docs else 0
    if got != tokens:
        raise ValueError(f"a corpus of {tokens} tokens does not end on a document boundary: the prefix of "
                         f"{len(docs)} documents holds {got}")
    return docs


def evaluation_docs(index, spec=None):
    """The held-out documents after the search corpus, where every package draws its own web text."""
    return tuple(index[len(search_docs(index, spec)):])


def select_documents(docs, n, seed, min_tokens=0, exclude=()):
    """The first `n` documents of a `seed` permutation of those with at least `min_tokens` tokens, skipping
    held-out indices in `exclude` after permuting. A smaller draw is a prefix of a larger one."""
    pool = [d for d in docs if d.n_tokens >= int(min_tokens)]
    if len(pool) < int(n):
        raise ValueError(f"{len(pool)} of {len(docs)} held-out documents hold {min_tokens} tokens; "
                         f"{n} wanted")
    order = np.random.default_rng(int(seed)).permutation(len(pool))
    barred = {int(i) for i in exclude}
    kept = [pool[int(t)] for t in order if int(pool[int(t)].doc) not in barred]
    if len(kept) < int(n):
        raise ValueError(f"{len(kept)} of {len(docs)} held-out documents hold {min_tokens} tokens and are "
                         f"not excluded ({len(pool) - len(kept)} of the pool are); {n} wanted")
    return kept[:int(n)]


@functools.lru_cache(maxsize=None)
def load_overlap(path=HELDOUT_OVERLAP, meta_path=HELDOUT_OVERLAP_META):
    """`((coverage_n7, coverage_n13), ...)` per held-out document, in order (`assets/README.md`). Refused
    unless the sha256 matches and there is one row per document. Cached on the path."""
    with open(meta_path, encoding="utf-8") as h:
        meta = json.load(h)
    with open(path, "rb") as h:
        raw = h.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: sha256 {digest} is not the recorded {meta['sha256']}")
    lines = raw.decode("utf-8").splitlines()
    if lines[0] != "doc,coverage_n7,coverage_n13":
        raise ValueError(f"{path}: header {lines[0]!r} is not 'doc,coverage_n7,coverage_n13'")
    rows = []
    for k, line in enumerate(lines[1:]):
        doc, n7, n13 = line.split(",")
        if int(doc) != k:
            raise ValueError(f"{path}: record {k} carries doc={doc}; the file is in the held-out ORDER")
        if not (0.0 <= float(n7) <= 1.0 and 0.0 <= float(n13) <= 1.0):
            raise ValueError(f"{path}: document {k} has coverage ({n7}, {n13}), which is not a share")
        rows.append((float(n7), float(n13)))
    if len(rows) != meta["n_docs"] or len(rows) != HELDOUT_DOCS:
        raise ValueError(f"{path}: {len(rows)} documents, and the held-out corpus is {HELDOUT_DOCS} "
                         f"(recorded: {meta['n_docs']})")
    return tuple(rows)


def overlap_excluded(threshold=OVERLAP_THRESHOLD):
    """Held-out document indices whose word 7-gram coverage is at least `threshold`."""
    return frozenset(k for k, (n7, _n13) in enumerate(load_overlap()) if n7 >= float(threshold))


def overlap_record(docs=None, min_tokens=0, threshold=OVERLAP_THRESHOLD):
    """The overlap rule, asset digest and excluded count of a draw's pool, for resume keys and records."""
    excluded = overlap_excluded(threshold)
    with open(HELDOUT_OVERLAP_META, encoding="utf-8") as h:
        sha = json.load(h)["sha256"]
    pool = [d for d in (load_index() if docs is None else docs) if d.n_tokens >= int(min_tokens)]
    return {"measure": "word 7-gram coverage by the training text", "threshold": float(threshold),
            "asset_sha256": sha, "n_pool": len(pool),
            "n_excluded": sum(1 for d in pool if int(d.doc) in excluded)}


def check_disjoint(corpus_docs, excluded):
    """Raise unless the search corpus shares no document with any set in `excluded` (name -> indices)."""
    corpus = {int(d.doc) if isinstance(d, Doc) else int(d) for d in corpus_docs}
    for name, ids in excluded.items():
        both = sorted(corpus & {int(i) for i in ids})
        if both:
            shown = ", ".join(str(i) for i in both[:5])
            more = f" (+{len(both) - 5} more)" if len(both) > 5 else ""
            raise ValueError(f"the search corpus overlaps {name}: {len(both)} shared document(s) "
                             f"[{shown}{more}]; a package's own documents come from `evaluation_docs`, "
                             f"after the corpus prefix")
    return True


@dataclasses.dataclass(frozen=True)
class CorpusSpec:
    """Source files, window grid, read layer and ranking rule; the defaults are the shared baseline."""

    dataset: Stream = DATASET
    #: tokens of the held-out order in the search corpus (595,343 windows at 10M)
    corpus_tokens: int = CORPUS_TOKENS
    window: int = WINDOW
    stride: int = STRIDE
    read_layer: int = READ_LAYER
    scoring: str = SCORING
    top_k: int = 8
    suppression: str = SUPPRESSION
    #: a top-1 cosine above this is counted as a possible near duplicate (`near_duplicates`)
    near_duplicate_cos: float = 0.95

    def __post_init__(self):
        if int(self.read_layer) != READ_LAYER:
            raise ValueError(f"the corpus is read at layer {READ_LAYER}; read_layer={self.read_layer} is "
                             f"not supported")
        if str(self.suppression) != SUPPRESSION:
            raise ValueError(f"the search keeps a query's windows under one rule ({SUPPRESSION!r}); "
                             f"suppression={self.suppression!r} is not supported")
        check_top_k(self.top_k, self.window, self.stride)

    def record(self):
        """The spec as a JSON-plain dict, for a corpus's metadata and a stage's resume key."""
        return {"dataset": dict(self.dataset), "corpus_tokens": int(self.corpus_tokens),
                "window": int(self.window), "stride": int(self.stride),
                "read_layer": int(self.read_layer), "scoring": str(self.scoring),
                "top_k": int(self.top_k), "suppression": str(self.suppression),
                "near_duplicate_cos": float(self.near_duplicate_cos)}


def sizes_windows(docs, sizes=SIZES, spec=None):
    """`((label, tokens, windows), ...)` of the nested sizes, each ending on a document boundary."""
    spec = spec or CorpusSpec()
    out, k, total = [], 0, 0
    for label, tokens in sizes:
        while k < len(docs) and docs[k].offset + docs[k].n_tokens <= int(tokens):
            total += n_windows(docs[k].n_tokens, spec.window, spec.stride)
            k += 1
        got = docs[k - 1].offset + docs[k - 1].n_tokens if k else 0
        if got != int(tokens):
            raise ValueError(f"corpus size {label} ({tokens} tokens) does not end on a document boundary: "
                             f"the prefix of {k} documents holds {got}")
        out.append((str(label), int(tokens), int(total)))
    return tuple(out)


def shared_windows(docs, spec=None, sizes=SIZES, label=SHARED_SIZE):
    """How many of `docs`' windows lie in the `SHARED_SIZE` prefix (`TopK`'s `prefix_windows`)."""
    spec = spec or CorpusSpec()
    tokens = dict(sizes)[label]
    inside = [d for d in docs if d.offset + d.n_tokens <= int(tokens)]
    if len(inside) < len(docs) and (not inside or inside[-1].offset + inside[-1].n_tokens != int(tokens)):
        raise ValueError(f"corpus size {label} ({tokens} tokens) does not end on a document boundary of "
                         f"this corpus")
    return total_windows(inside, spec.window, spec.stride)


# ---------------------------------------------------------------- rebuilding the corpus

def open_parts(dataset=DATASET, index=None):
    """The two pinned parquet files as `[part 0's text rows, part 1's]`, read only up to the last indexed
    row. The one download of the build."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    index = load_index() if index is None else index
    parts = []
    for number, name in enumerate(dataset["files"]):
        path = hf_hub_download(dataset["dataset"], name, repo_type="dataset",
                               revision=dataset["revision"])
        file = pq.ParquetFile(path)
        names = file.schema_arrow.names
        column = next((c for c in TEXT_COLUMNS if c in names), None)
        if column is None:
            raise ValueError(f"{name} has columns {names}; none of {TEXT_COLUMNS} is the text")
        need = 1 + max((int(d.row) for d in index if int(d.part) == number), default=-1)
        rows = []
        for group in range(file.num_row_groups):
            if len(rows) >= need:
                break
            rows.extend(file.read_row_group(group, columns=[column]).column(column).to_pylist())
        parts.append(rows[:need])
    return parts


def document_ids(parts, doc, tok):
    """One held-out document's token ids (no special tokens, no truncation). Raises unless the count is
    the index's: the corpus's identity check."""
    part = parts[int(doc.part)]
    if not 0 <= int(doc.row) < len(part):
        raise ValueError(f"document {doc.doc} is row {doc.row} of part {doc.part}, which holds {len(part)}")
    ids = tok(part[int(doc.row)], add_special_tokens=False).input_ids
    if len(ids) != int(doc.n_tokens):
        raise ValueError(f"document {doc.doc} (part {doc.part}, row {doc.row}) tokenises to {len(ids)} "
                         f"tokens and the index records {doc.n_tokens}: this is not the corpus the index "
                         f"describes")
    return [int(t) for t in ids]


class Corpus:
    """A built corpus: token ids as one flat int32 array plus the window grid. `doc_of[k]`, `start_of[k]`
    locate window `k`, whose index is its `(document, start)` rank and every ranking's tiebreak."""

    def __init__(self, ids, docs, window=WINDOW, stride=STRIDE):
        self.ids = np.asarray(ids, dtype=np.int32)
        self.docs = tuple(docs)
        self.window, self.stride = int(window), int(stride)
        counts = [n_windows(d.n_tokens, self.window, self.stride) for d in self.docs]
        self.doc_of = np.repeat(np.arange(len(self.docs), dtype=np.int64), counts)
        self.start_of = np.concatenate([np.arange(c, dtype=np.int64) * self.stride for c in counts]) \
            if counts else np.zeros(0, dtype=np.int64)
        # each document's start in `self.ids` (not its held-out offset), so any document selection works
        self.base = np.concatenate([[0], np.cumsum([d.n_tokens for d in self.docs])]).astype(np.int64)
        total = sum(d.n_tokens for d in self.docs)
        if self.ids.shape != (total,):
            raise ValueError(f"the corpus holds {self.ids.shape[0]} token ids and its {len(self.docs)} "
                             f"documents are {total} tokens")

    def __len__(self):
        return int(self.doc_of.shape[0])

    def doc(self, k):
        """The `Doc` window `k` was cut from."""
        return self.docs[int(self.doc_of[int(k)])]

    def window_ids(self, k):
        """Window `k`'s token ids: up to `window` of them, fewer at the end of a document."""
        j = int(self.doc_of[int(k)])
        start = int(self.start_of[int(k)])
        lo = int(self.base[j]) + start
        hi = min(lo + self.window, int(self.base[j]) + int(self.docs[j].n_tokens))
        return [int(t) for t in self.ids[lo:hi]]

    def span(self, k):
        """`(document, start, stop)` for window `k`, with `stop` the window's real end."""
        j = int(self.doc_of[int(k)])
        start = int(self.start_of[int(k)])
        return j, start, min(start + self.window, int(self.docs[j].n_tokens))

    def window_id(self, k):
        """The window's name, `web:<held-out document>:<start>`."""
        return f"web:{int(self.doc(k).doc)}:{int(self.start_of[int(k)])}"

    def block(self, start, stop):
        """`[{k, window_id, doc, start, ids}, ...]` for windows `[start, stop)`: what a GPU worker is sent."""
        return [{"k": int(k), "window_id": self.window_id(k), "doc": int(self.doc(k).doc),
                 "start": int(self.start_of[int(k)]), "ids": self.window_ids(k)}
                for k in range(int(start), int(stop))]


def build_corpus(run, spec, docs, parts, tok, rel_npz, rel_json, sizes=SIZES):
    """Rebuild `docs` from `parts` with `tok`; write the token npz and metadata (with the `sizes` table
    `corpus_sizes` reads back); return the metadata."""
    started = time.time()
    docs = tuple(docs)
    table = sizes_windows(docs, sizes, spec) if sizes else ()
    ids = np.zeros(sum(d.n_tokens for d in docs), dtype=np.int32)
    at = 0
    for d in docs:
        row = document_ids(parts, d, tok)
        ids[at:at + len(row)] = row
        at += len(row)
    corpus = Corpus(ids, docs, spec.window, spec.stride)
    save_npz(run.file(rel_npz), ids=ids,
             doc=np.array([d.doc for d in docs], dtype=np.int64),
             part=np.array([d.part for d in docs], dtype=np.int64),
             row=np.array([d.row for d in docs], dtype=np.int64),
             n_tokens=np.array([d.n_tokens for d in docs], dtype=np.int64),
             offset=np.array([d.offset for d in docs], dtype=np.int64))
    meta = {
        **spec.record(),
        "n_docs": len(docs),
        "n_tokens": int(at),
        "n_windows": len(corpus),
        "doc_min": int(docs[0].doc),
        "doc_max": int(docs[-1].doc),
        "sizes": [[label, int(tokens), int(windows)] for label, tokens, windows in table],
        "index_sha256": _index_sha256(),
        "build_seconds": time.time() - started,
    }
    run.write_json(rel_json, meta)
    return meta


def _index_sha256():
    with open(HELDOUT_INDEX_META, encoding="utf-8") as h:
        return json.load(h)["sha256"]


def load_corpus(run, rel_npz, spec=None):
    """A `Corpus` from what `build_corpus` wrote: the token ids and the document table it was cut from."""
    spec = spec or CorpusSpec()
    with np.load(run.file(rel_npz)) as z:
        docs = [Doc(doc=int(d), part=int(p), row=int(r), n_tokens=int(n), offset=int(o))
                for d, p, r, n, o in zip(z["doc"], z["part"], z["row"], z["n_tokens"], z["offset"])]
        return Corpus(z["ids"], docs, spec.window, spec.stride)


def corpus_sizes(run, rel_json, n_windows_total=None):
    """`(metadata, [(label, n_windows), ...])` from the built corpus, checked against `n_windows_total`."""
    meta = run.read_json(rel_json)
    if not meta.get("sizes"):
        raise ValueError(f"{rel_json} carries no `sizes`: a search curve's points are the corpus's own "
                         f"nested prefixes, so they are read off the corpus that was built, never off the "
                         f"config of the run doing the ranking")
    total = int(meta.get("n_windows") or 0) if n_windows_total is None else int(n_windows_total)
    if total <= 0:
        raise ValueError(f"the corpus holds {total} windows; build it first")
    sizes = [(str(label), int(windows)) for label, _tokens, windows in meta["sizes"]]
    over = [(label, n) for label, n in sizes if n > total]
    if over:
        raise ValueError(f"corpus holds {total} windows; size(s) {over} ask for more")
    return meta, sizes


# ---------------------------------------------------------------- the scoring rule

def reencode_windows(windows, mdl, tok, device, sbatch=WINDOW_BATCH):
    """Yield `(s, h [b, T, d] float32, keep [b, T] bool)` for `[sink] + window` at `READ_LAYER`: the
    token-id twin of `evals.downstream.common.scorer.reencode` (same sink and padding, no norm filter)."""
    import torch

    from maem.inject import read_resid

    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else sink
    for s in range(0, len(windows), int(sbatch)):
        batch = [list(w) for w in windows[s:s + int(sbatch)]]
        width = 1 + max(len(w) for w in batch)
        ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        ids[:, 0], mask[:, 0] = int(sink), 1
        for i, w in enumerate(batch):
            ids[i, 1:1 + len(w)] = torch.tensor(w, dtype=torch.long, device=device)
            mask[i, 1:1 + len(w)] = 1
        h, attended = read_resid(mdl, READ_LAYER, {"input_ids": ids, "attention_mask": mask}, pool="all")
        keep = attended.clone()
        keep[:, 0] = False                                  # the sink is not a position of the window
        yield s, h, keep


def window_cos(windows, Q, mdl, tok, device, batch=WINDOW_BATCH, reencode=None, mu=None):
    """`(raw, centred)`: float32 [len(windows), len(Q)] max-over-positions cosines of token-id windows
    against the unit queries `Q`; `centred` uses `h - mu` from the same forward (None without `mu`)."""
    import torch
    from torch.nn import functional as F

    reencode = reencode or reencode_windows
    Q = np.asarray(Q, dtype=np.float32)
    bank = torch.from_numpy(Q).to(device)
    centre = None if mu is None else torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(device)
    raw = np.zeros((len(windows), Q.shape[0]), dtype=np.float32)
    centred = None if mu is None else np.zeros_like(raw)
    with torch.no_grad():
        for s, h, keep in reencode(windows, mdl, tok, device, sbatch=batch):
            hf = h.float()                                              # [b, T, d]
            blocked = ~keep.bool().unsqueeze(-1)                        # sink and padding, nothing else
            # NaN scores -1 like padding, so it can never rank first; only [b, M] maxima leave the device
            cos = (F.normalize(hf, dim=-1) @ bank.T).nan_to_num(nan=-1.0).masked_fill(blocked, -1.0)
            raw[s:s + h.shape[0]] = cos.max(1).values.float().cpu().numpy()
            if centre is not None:
                cen = (F.normalize(hf - centre, dim=-1) @ bank.T).nan_to_num(nan=-1.0).masked_fill(blocked, -1.0)
                centred[s:s + h.shape[0]] = cen.max(1).values.float().cpu().numpy()
    return raw, centred


def bank_cos(texts, D, mdl, tok, device, batch=WINDOW_BATCH, reencode=None, mu=None, stats=None):
    """`(raw, centred)`: float32 [len(texts), len(D)] cosines of each text against each row of `D`, via
    the scorer's re-read without the norm filter; `centred` as in `window_cos`."""
    import functools

    import torch
    from torch.nn import functional as F

    from evals.downstream.common.scorer import reencode as _reencode

    if reencode is None:
        reencode = functools.partial(_reencode, norm_filter=False, stats=stats)
    elif stats is not None:
        raise ValueError("a caller-supplied re-encoder keeps its own tally: bind `stats` to it, not to bank_cos")

    texts = [t if t.strip() else " " for t in texts]
    D = np.asarray(D, dtype=np.float32)
    bank = torch.from_numpy(D).to(device)
    centre = None if mu is None else torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(device)
    raw = np.zeros((len(texts), D.shape[0]), dtype=np.float32)
    centred = None if mu is None else np.zeros_like(raw)
    with torch.no_grad():
        for s, h, keep, *_mask in reencode(texts, mdl, tok, device, sbatch=batch):
            hf = h.float()                                              # [b, T, d]
            blocked = ~keep.unsqueeze(-1)                               # the sink and the pads never win
            cos = (F.normalize(hf, dim=-1) @ bank.T).nan_to_num(nan=-1.0).masked_fill(blocked, -1.0)
            raw[s:s + h.shape[0]] = cos.max(1).values.float().cpu().numpy()
            if centre is not None:
                cen = (F.normalize(hf - centre, dim=-1) @ bank.T).nan_to_num(nan=-1.0).masked_fill(blocked, -1.0)
                centred[s:s + h.shape[0]] = cen.max(1).values.float().cpu().numpy()
    return raw, centred


# ---------------------------------------------------------------- ranking

def order_by(scores, k, window_ids=None):
    """Positions of the `k` best of `scores` by `(-score, window index)`, the index being `window_ids[t]`
    or the position. The one ordering both search paths use; its tiebreak makes sharding irrelevant."""
    n = len(scores)
    k = min(int(k), n)
    if k < 1:
        return []
    kth = float(np.partition(scores, n - k)[n - k])          # the k-th largest score
    cand = np.flatnonzero(scores >= kth)                     # every candidate that reaches it, ties included
    key = (lambda t: (-float(scores[t]), int(window_ids[t]))) if window_ids is not None \
        else (lambda t: (-float(scores[t]), int(t)))
    return [int(t) for t in sorted(cand, key=key)[:k]]


def suppress_overlapping(ranked, corpus, k):
    """`(kept positions in ranked, number passed over)` under `SUPPRESSION`, over `ranked` (window indices,
    best first, at least `candidate_depth(k)` deep) until `k` are kept."""
    kept, spans, passed = [], {}, 0
    for t, w in enumerate(ranked):
        if len(kept) >= int(k):
            break
        doc, lo, hi = corpus.span(w)
        if any(lo < b and a < hi for a, b in spans.get(doc, ())):
            passed += 1
            continue
        spans.setdefault(doc, []).append((lo, hi))
        kept.append(int(t))
    return kept, passed


class Ranking(dict):
    """`{n_windows: [[window index, ...] per query]}`, plus `suppressed` (passed over, per query)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suppressed = {}


def rank_retrieval(scores, size_windows, top_k, corpus):
    """A `Ranking`: for each nested size in `size_windows`, each query's `top_k` best non-overlapping
    windows within `scores[:n_windows]`."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [n_windows, n_queries], got shape {scores.shape}")
    n_total, n_act = scores.shape
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if n_total != len(corpus):
        raise ValueError(f"the scores cover {n_total} windows and the corpus they are ranked over holds "
                         f"{len(corpus)}")
    depth = candidate_depth(top_k, corpus.window, corpus.stride)
    out = Ranking()
    for n in sorted({int(x) for x in size_windows}):
        if not 0 < n <= n_total:
            raise ValueError(f"corpus size {n} windows is outside the scored corpus of {n_total}")
        out[n], out.suppressed[n] = [], []
        for j in range(n_act):
            ranked = order_by(scores[:n, j], min(depth, n))
            keep, passed = suppress_overlapping(ranked, corpus, top_k)
            out[n].append([ranked[t] for t in keep])
            out.suppressed[n].append(int(passed))
    return out


def near_duplicates(top1_cos, threshold):
    """How many queries' best window scores above `threshold` (possible near copies of the query text)."""
    v = np.asarray(top1_cos, dtype=np.float64)
    return int(np.sum(v > float(threshold)))


class TopK:
    """A running per-query ranking over windows arriving in blocks: the `candidate_depth(k)` best by score
    alone, so folding is order-independent; `resolve(corpus)` applies the overlap rule. `prefix_windows`
    adds `self.prefix`, the same over a nested size."""

    def __init__(self, n_queries, k, prefix_windows=None, depth=None):
        self.n_queries, self.k = int(n_queries), int(k)
        self.depth = candidate_depth(self.k) if depth is None else int(depth)
        if self.k < 1 or not self.k <= self.depth <= MAX_TOP_K:
            raise ValueError(f"k={k} with {self.depth} candidates per query is outside what a running "
                             f"ranking holds: 1 <= k <= depth <= MAX_TOP_K = {MAX_TOP_K}, where the depth "
                             f"is `candidate_depth(k)`, {1 + window_reach()} x k on the shared grid")
        self.cand_idx = [[] for _ in range(self.n_queries)]
        self.cand_score = [[] for _ in range(self.n_queries)]
        self._kept = None
        self.prefix_windows = None if prefix_windows is None else int(prefix_windows)
        self.prefix = None if prefix_windows is None else TopK(n_queries, k, depth=self.depth)

    def _fold(self, j, ids, vals):
        """Query `j`'s candidates and `ids`/`vals` ranked together, the `depth` best kept."""
        ids = self.cand_idx[j] + [int(w) for w in ids]
        vals = np.concatenate([np.asarray(self.cand_score[j], dtype=np.float32),
                               np.asarray(vals, dtype=np.float32)])
        keep = order_by(vals, self.depth, window_ids=ids)
        self.cand_idx[j] = [ids[t] for t in keep]
        self.cand_score[j] = [float(vals[t]) for t in keep]
        self._kept = None

    def add(self, window_idx, scores):
        """Fold in one block: corpus-wide `window_idx` and its [len(block), n_queries] `scores`."""
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(window_idx), self.n_queries):
            raise ValueError(f"a block of {len(window_idx)} windows scored {scores.shape} against "
                             f"{self.n_queries} queries")
        for j in range(self.n_queries):
            self._fold(j, window_idx, scores[:, j])
        if self.prefix is not None:
            inside = [t for t, w in enumerate(window_idx) if int(w) < self.prefix_windows]
            if inside:
                self.prefix.add([int(window_idx[t]) for t in inside], scores[inside])
        return self

    def resolve(self, corpus):
        """Apply the overlap rule over the candidates (and the prefix's) on `corpus`'s grid; return self."""
        need = candidate_depth(self.k, corpus.window, corpus.stride)
        if self.depth < need:
            raise ValueError(f"the ranking holds {self.depth} candidates per query and {self.k} "
                             f"non-overlapping windows of a {corpus.window}/{corpus.stride} grid are taken "
                             f"out of {need}")
        idx, score, suppressed = [], [], []
        for j in range(self.n_queries):
            keep, passed = suppress_overlapping(self.cand_idx[j], corpus, self.k)
            idx.append([self.cand_idx[j][t] for t in keep])
            score.append([self.cand_score[j][t] for t in keep])
            suppressed.append(int(passed))
        self._kept = (idx, score, suppressed)
        if self.prefix is not None:
            self.prefix.resolve(corpus)
        return self

    def _resolved(self, what):
        if self._kept is None:
            raise RuntimeError(f"`TopK.{what}` is the windows the overlap rule kept, and the rule runs in "
                               f"`resolve(corpus)`: until then the ranking holds candidates by score alone "
                               f"(`cand_idx`, `cand_score`)")
        return self._kept

    @property
    def idx(self):
        """[[window index, ...] per query]: the kept windows, best first."""
        return self._resolved("idx")[0]

    @property
    def score(self):
        """[[search cosine, ...] per query], aligned with `idx`."""
        return self._resolved("score")[1]

    @property
    def suppressed(self):
        """[candidates passed over, per query]: better-scoring windows that overlapped a kept one."""
        return self._resolved("suppressed")[2]

    def arrays(self):
        """The unresolved candidates as npz-ready arrays (`idx` padded with -1): what a shard saves."""
        width = max([len(r) for r in self.cand_idx], default=0)
        idx = np.full((self.n_queries, width), -1, dtype=np.int64)
        score = np.zeros((self.n_queries, width), dtype=np.float32)
        for j in range(self.n_queries):
            for t, k in enumerate(self.cand_idx[j]):
                idx[j, t], score[j, t] = k, self.cand_score[j][t]
        out = {"idx": idx, "score": score, "depth": np.int64(self.depth), "k": np.int64(self.k)}
        if self.prefix is not None:
            inner = self.prefix.arrays()
            out.update(prefix_idx=inner["idx"], prefix_score=inner["score"],
                       prefix_windows=np.int64(self.prefix_windows))
        return out

    @classmethod
    def merge(cls, parts, k):
        """One unresolved `TopK` folded from the `arrays()` of shards over disjoint blocks."""
        parts = list(parts)
        if not parts:
            raise ValueError("no parts to merge")
        depths = {int(part["depth"]) if "depth" in part else None for part in parts}
        ks = {int(part["k"]) if "k" in part else None for part in parts}
        if None in depths or len(depths) > 1 or ks != {int(k)}:
            raise ValueError(f"the parts record candidate depths {sorted(depths, key=str)} for k "
                             f"{sorted(ks, key=str)} and the merge ranks k={k}: every part of one split "
                             f"holds `candidate_depth(k)` candidates per query, kept by score alone, and a "
                             f"part that records none cannot be merged under the overlap rule")
        # a prefix ranking merges only when every part carries one, under one window count
        bounds = {int(part["prefix_windows"]) if "prefix_windows" in part else None for part in parts}
        if len(bounds) > 1:
            raise ValueError(f"the parts carry prefix rankings over {sorted(bounds, key=str)} windows; one "
                             f"split ranks one prefix")
        bound = bounds.pop()
        top = cls(int(np.asarray(parts[0]["idx"]).shape[0]), k, prefix_windows=bound, depth=depths.pop())
        for ranking, names in ((top, ("idx", "score")), (top.prefix, ("prefix_idx", "prefix_score"))):
            if ranking is None:
                continue
            for part in parts:
                idx, score = (np.asarray(part[key]) for key in names)
                if idx.shape[0] != top.n_queries:
                    raise ValueError(f"a part holds {idx.shape[0]} queries and the merge runs over "
                                     f"{top.n_queries}")
                for j in range(top.n_queries):
                    real = np.flatnonzero(idx[j] >= 0)
                    ranking._fold(j, idx[j, real].tolist(), score[j, real])
        return top


def top_windows(corpus, top, tok, trunc=None):
    """`[[{rank, k, window_id, doc, start, text, n_tokens, score, search_cos}, ...] per query]` for a
    resolved `TopK`; `trunc` adds `text_trunc`, the first `trunc` tokens."""
    out = []
    for j in range(top.n_queries):
        row = []
        for rank, k in enumerate(top.idx[j]):
            ids = corpus.window_ids(k)
            rec = {"rank": rank, "k": int(k), "window_id": corpus.window_id(k),
                   "doc": int(corpus.doc(k).doc), "start": int(corpus.start_of[int(k)]),
                   "text": tok.decode(ids, skip_special_tokens=False), "n_tokens": len(ids),
                   "score": float(top.score[j][rank]), "search_cos": float(top.score[j][rank])}
            if trunc is not None:
                rec["text_trunc"] = tok.decode(ids[:int(trunc)], skip_special_tokens=False)
            row.append(rec)
        out.append(row)
    return out


def size_label(n_tokens, sizes=SIZES):
    """A corpus size's report label: its `sizes` label, or the token count."""
    return next((str(label) for label, tokens in sizes if int(tokens) == int(n_tokens)),
                f"{int(n_tokens):,} tokens")


def search_rows(top, n_windows, n_tokens, queries=None):
    """Report rows for a resolved `TopK`, the `SHARED_SIZE` prefix first when present: means over `queries`
    (None: all) of the best cosine, the kept windows' mean cosine, and the count passed over."""
    rows = []
    shared = getattr(top, "prefix", None)
    for ranking, size, count, is_shared in (
            (shared, SHARED_SIZE, min(int(n_windows), int(top.prefix_windows or 0)) if shared else 0, True),
            (top, size_label(n_tokens), int(n_windows), False)):
        if ranking is None:
            continue
        js = [j for j in (range(ranking.n_queries) if queries is None else [int(j) for j in queries])
              if ranking.score[j]]
        kept = [ranking.score[j] for j in js]
        best = np.asarray([row[0] for row in kept], dtype=np.float64)
        rows.append({"size": size, "shared": is_shared, "full": not is_shared, "n_windows": count,
                     "n_queries": len(kept), "top_k": int(ranking.k),
                     "top1_cos": float(best.mean()) if len(best) else None,
                     "top1_se": float(best.std(ddof=1) / np.sqrt(len(best))) if len(best) > 1 else None,
                     "topk_cos": float(np.mean([np.mean(row) for row in kept])) if kept else None,
                     "suppressed": float(np.mean([ranking.suppressed[j] for j in js])) if js else None})
    return rows


def search_table(rows, lead=None):
    """`search_rows` as markdown table lines. `lead` is an optional `(header, key)` leading column."""
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    one = lambda v: "n/a" if v is None else f"{v:.1f}"
    head, rule = ("| " + lead[0] + " ", "|---") if lead else ("", "")
    out = [f"{head}| corpus size | windows | directions | top-1 search cos (mean ± SE) | "
           f"top-k search cos (mean) | overlapping candidates passed over (mean) |",
           f"{rule}|---|---|---|---|---|---|"]
    for r in rows:
        name = f"{r['size']}" + (" (the shared cross-evaluation size)" if r["shared"] else
                                 " (this package's full search corpus)" if r.get("full") else "")
        first = f"| {r.get(lead[1], '')} " if lead else ""
        out.append(f"{first}| {name} | {r['n_windows']:,} | {r['n_queries']} | {fmt(r['top1_cos'])} ± "
                   f"{fmt(r['top1_se'])} | {fmt(r['topk_cos'])} (k = {r['top_k']}) | "
                   f"{one(r.get('suppressed'))} |")
    return out


def search_bullets(rows):
    """`search_rows` as markdown list items."""
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    one = lambda v: "n/a" if v is None else f"{v:.1f}"
    what = lambda r: ("the shared cross-evaluation size, " if r["shared"] else
                      "the full search corpus, " if r.get("full") else "")
    return [f"- the search's own cosine at {r['size']} ({what(r)}{r['n_windows']:,} windows, {r['n_queries']} directions): top-1 {fmt(r['top1_cos'])} ± "
            f"{fmt(r['top1_se'])} SE, mean of the top {r['top_k']} non-overlapping windows "
            f"{fmt(r['topk_cos'])}, {one(r.get('suppressed'))} overlapping candidates passed over per "
            f"direction" for r in rows]


SEARCH_NOTE = (
    f"The search cosine is the corpus's own read of a window (token ids, no norm filter, maximum over the "
    f"window's positions). The {SHARED_SIZE} row is the same search over the nested {SHARED_SIZE} prefix of "
    f"the corpus, ranked out of the same forward pass: it is the largest size this suite and the paper's "
    f"reconstruction evaluation hold document for document, under one window grid and one scoring rule, so "
    f"it is the row to set beside that evaluation's corpus scan. Top-1 is the single best window, which is "
    f"what a reader's best-of-n is compared with there. The top k are the k best NON-OVERLAPPING windows: "
    f"walking a direction's windows best first, one is kept unless it shares a token with an already-kept "
    f"window of its document, so rank 1 is the best window either way and the other ranks are different "
    f"passages rather than the same one shifted {STRIDE} tokens; the last column is how many better-scoring "
    f"windows that rule passed over per direction, within the prefix the row is about.")


# ---------------------------------------------------------------- the sharded forward

def shard_block(n_windows_total, k, n):
    """The contiguous block of windows shard `k` of `n` owns: equal blocks, the last short. Clamped to the
    corpus, so a surplus shard owns an empty block at the end and the parts still tile."""
    k, n = int(k), int(n)
    if not (0 <= k < n):
        raise ValueError(f"bad shard {k}/{n}")
    per = -(-int(n_windows_total) // n)
    start = min(k * per, int(n_windows_total))
    return range(start, min(max((k + 1) * per, start), int(n_windows_total)))


def save_npz(path, **arrays):
    """`np.savez` to a temp name (ending `.npz`, which savez would append) and `os.replace` it."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_part(run, rel, sl, arrays):
    """Store one shard's `arrays` with the `[start, stop)` it covers, which the merge checks for tiling.
    Keep scores float32: half precision would create ranking ties."""
    save_npz(run.file(rel), **{k: np.asarray(v) for k, v in arrays.items()},
             start=np.int64(sl.start), stop=np.int64(sl.stop))


def part_config_hash(base, k, n, sl):
    """A part's resume key: the pass's key `base` chained with this shard's block."""
    return config_hash({"part": {"shard": f"{k}/{n}", "start": int(sl.start), "stop": int(sl.stop)},
                        "pass": base})


def part_states(run, n, n_windows_total, part_rel, part_stage, base):
    """`(finished shard indices, unfinished ones)`, by the npz and its stage record's key."""
    done, waiting = [], []
    for j in range(n):
        sl = shard_block(n_windows_total, j, n)
        ready = (run.exists(part_rel.format(k=j, n=n))
                 and stage_done(run, part_stage.format(k=j, n=n), part_config_hash(base, j, n, sl)))
        (done if ready else waiting).append(j)
    return done, waiting


def refuse_or_wait(merge, stage, command, n, done, waiting):
    """Missing parts: raise in the merge call, else print and let a scoring shard return."""
    if merge:
        raise RuntimeError(
            f"{stage}: the merge call found part(s) {waiting} of the {n}-way retrieval split missing "
            f"({len(done)} of {n} in place). The merge is made only once every shard has committed, and "
            f"nothing else writes this method's records: re-run the missing shard(s) with "
            f"`{command} --shard <k>/{n}` and then make this call again.")
    print(f"[{stage}] parts {done} of {n} are in place, waiting for {waiting}; run "
          f"`{command} --shard 0/{n} --merge` once every shard has committed to merge them", flush=True)


def score_corpus_part(run, corpus, device, mdl, tok, bank, k, n, sl, chash, part_rel, part_stage, owner,
                      mu=None, batch=WINDOW_BATCH):
    """Score windows `sl` of `corpus` against the unit queries `bank`; save `part_rel`'s npz (`scores`,
    plus `centred` when `mu` is given) and the part's stage record."""
    started = time.time()
    if len(sl) and sl.stop > len(corpus):
        raise ValueError(f"shard {k}/{n} wants windows [{sl.start}, {sl.stop}) and the corpus holds "
                         f"{len(corpus)}")
    n_dirs = np.asarray(bank).shape[0]
    windows = [corpus.window_ids(w) for w in sl]
    if windows:
        scores, centred = window_cos(windows, bank, mdl, tok, device, batch=batch, mu=mu)
    else:
        scores = np.zeros((0, n_dirs), dtype=np.float32)
        centred = None if mu is None else np.zeros((0, n_dirs), dtype=np.float32)
    arrays = {"scores": scores} if centred is None else {"scores": scores, "centred": centred}
    save_part(run, part_rel.format(k=k, n=n), sl,
              {key: np.asarray(v, dtype=np.float32) for key, v in arrays.items()})
    rec = {"shard": f"{k}/{n}", "start": int(sl.start), "stop": int(sl.stop), "n": len(windows),
           "n_windows": len(corpus), "n_act": n_dirs, "window_batch": batch,
           "bank_seconds": time.time() - started}
    stage = part_stage.format(k=k, n=n)
    mark_stage(run, stage, chash, rec, started=started)
    # provenance under the part's own name: concurrent shards must not share one read-modify-write file
    write_provenance(run, {stage: rec}, stage=stage)
    print(f"[{owner}] retrieval shard {k}/{n}: windows [{sl.start}, {sl.stop}) x {n_dirs} directions "
          f"in {rec['bank_seconds']:.0f}s", flush=True)
    return rec


def merge_parts(run, n, n_windows_total, n_dirs, part_rel, keys=("scores",)):
    """`{key: [n_windows, n_dirs] float32}` from the matrix parts of an `n`-way split, read in shard order.
    Raises unless the parts tile the whole corpus exactly (no gap, overlap or short cover)."""
    out = {key: np.zeros((n_windows_total, n_dirs), dtype=np.float32) for key in keys}
    expect = 0
    for j in range(n):
        rel = part_rel.format(k=j, n=n)
        if not run.exists(rel):
            raise FileNotFoundError(f"{rel}: the merge needs every part of the {n}-way split")
        with np.load(run.file(rel)) as z:
            start, stop = int(z["start"]), int(z["stop"])
            if start != expect:
                raise ValueError(f"{rel} covers windows [{start}, {stop}) and the parts before it end at "
                                 f"{expect}: the parts of one split must tile the corpus without a gap or "
                                 f"an overlap (a part file from a different --shard split?)")
            for key in keys:
                block = z[key]
                if block.shape != (stop - start, n_dirs):
                    raise ValueError(f"{rel} holds a {block.shape} score matrix for windows "
                                     f"[{start}, {stop}) and {n_dirs} directions")
                out[key][start:stop] = block
        expect = stop
    if expect != n_windows_total:
        raise ValueError(f"the {n} parts cover {expect} windows and the corpus holds {n_windows_total}")
    return out


TOPK_CHUNK = 2048


def score_corpus_topk(run, corpus, device, mdl, tok, queries, sl, top_k, mu=None, metric="raw",
                      exclude_docs=(), chunk=TOPK_CHUNK, batch=WINDOW_BATCH, score=None,
                      prefix_windows=None):
    """Search windows `sl` against `queries` in chunks, folding into a `TopK` ranked on `metric` ("raw" or
    "centred", which needs `mu`); windows of `exclude_docs` are skipped and `prefix_windows` adds
    `TopK.prefix`. Returns `(resolved TopK, n scored, n excluded)`."""
    if metric not in ("raw", "centred"):
        raise ValueError(f"unknown metric {metric!r}; 'raw' or 'centred'")
    if metric == "centred" and mu is None:
        raise ValueError("the centred geometry needs the centring mean; pass mu=")
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError(f"queries must be [n_queries, d_model], got shape {queries.shape}")
    if len(sl) and sl.stop > len(corpus):
        raise ValueError(f"windows [{sl.start}, {sl.stop}) asked for and the corpus holds {len(corpus)}")
    drop = {int(d) for d in exclude_docs}
    top = TopK(queries.shape[0], top_k, prefix_windows=prefix_windows,
               depth=candidate_depth(top_k, corpus.window, corpus.stride))
    n_scored, n_excluded = 0, 0
    for start in range(sl.start, sl.stop, int(chunk)):
        stop = min(start + int(chunk), sl.stop)
        kept = [w for w in range(start, stop) if int(corpus.doc(w).doc) not in drop]
        n_excluded += (stop - start) - len(kept)
        if not kept:
            continue
        raw, centred = (score or window_cos)([corpus.window_ids(w) for w in kept], queries, mdl, tok,
                                             device, batch=batch, mu=mu)
        top.add(kept, raw if metric == "raw" else centred)
        n_scored += len(kept)
    return top.resolve(corpus), n_scored, n_excluded


TOPK_KEYS = ("idx", "score", "depth", "k", "prefix_idx", "prefix_score", "prefix_windows")


def save_topk_part(run, rel, sl, top):
    """Store one shard's `TopK.arrays()` with its `[start, stop)`, as `save_part` does."""
    save_part(run, rel, sl, top.arrays())


def merge_topk_parts(run, n, corpus, part_rel, top_k):
    """One resolved `TopK` from the top-k parts of an `n`-way split, with `merge_parts`' tiling check."""
    n_windows_total = len(corpus)
    parts, expect = [], 0
    for j in range(n):
        rel = part_rel.format(k=j, n=n)
        if not run.exists(rel):
            raise FileNotFoundError(f"{rel}: the merge needs every part of the {n}-way split")
        with np.load(run.file(rel), allow_pickle=False) as z:
            start, stop = int(z["start"]), int(z["stop"])
            if start != expect:
                raise ValueError(f"{rel} covers windows [{start}, {stop}) and the parts before it end at "
                                 f"{expect}: the parts of one split must tile the corpus without a gap or "
                                 f"an overlap (a part file from a different --shard split?)")
            parts.append({key: z[key] for key in TOPK_KEYS if key in z.files})
        expect = stop
    if expect != n_windows_total:
        raise ValueError(f"the {n} parts cover {expect} windows and the corpus holds {n_windows_total}")
    return TopK.merge(parts, top_k).resolve(corpus)
