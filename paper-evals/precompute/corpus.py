"""Product `corpus`: the held-out token slice every other base product indexes into.

    <root>/base/<base>/corpus/{tokens.i32, docs.jsonl, index.json, README.md}

CPU only, and the ONLY function in paper-evals that talks to the network: the Ultra-FineWeb parquet
parts are not in the volume's HF cache and are streamed from the hub. Everything else runs
HF_HUB_OFFLINE=1.

Recipe (infra/precompute.md §2, infra/precompute-layout.md §5 item 2):
  * parts 0009 and 0010 of `data/ultrafineweb_en/`, read in ROW ORDER, half the token budget taken
    from the head of each part. Documents are therefore (part, row)-addressable, and the slice is
    disjoint from the old 8B activation corpus, which consumed only the head of part 0001
    (modal/maemm_modal.py:2195-2205 has the margin argument).
  * tokenized per base with add_special_tokens=False and NO truncation (the vocabularies differ:
    8B 151,669 vs 27B 248,077, checklist item 78, so each base gets its own corpus directory).
  * document order is then permuted with np.random.default_rng(corpus.seed) and nested size tags
    1/2/4/8/16 (millions) are assigned by cumulative token count in that permuted order, so every
    nested subset is a PREFIX of tokens.i32 and of docs.jsonl.
"""

from __future__ import annotations

import gc
import json
import os
import time

import precompute.common as C

TEXT_COLUMNS = ("content", "text")  # eval/corpus_retrieval.py:203
TOK_BATCH = 256  # documents per tokenizer call; the fast tokenizer is the CPU-bound step


def _go_online():
    """Turn HF_HUB_OFFLINE off for THIS product only.

    The image sets HF_HUB_OFFLINE=1 so no GPU product can silently re-download a model at an
    unpinned revision. huggingface_hub reads that env var ONCE, at import, into
    `huggingface_hub.constants.HF_HUB_OFFLINE`, and its request hook raises OfflineModeIsEnabled
    from that constant -- so setting os.environ here is not enough and the constant is patched too
    (MEASURED: the first smoke failed exactly this way). The parquet parts are the only thing this
    repo fetches at run time; everything else lives in the volume's HF cache.
    """
    import huggingface_hub.constants as hfc

    os.environ["HF_HUB_OFFLINE"] = "0"
    hfc.HF_HUB_OFFLINE = False
    try:
        import datasets.config as dsc

        dsc.HF_HUB_OFFLINE = False
    except AttributeError:  # older/newer datasets may not expose it; the hub constant is the gate
        pass
    assert not hfc.HF_HUB_OFFLINE, "failed to turn off huggingface_hub offline mode"


def _doc_stream(dataset: str, path: str, revision: str):
    """Yield (row_index_in_part, text) for every non-empty document of one parquet part, in file
    order. `row_index_in_part` counts EVERY row, including the empty ones that are skipped, so it
    addresses the source file rather than our stream."""
    from datasets import load_dataset

    # The direct-parquet route of eval/corpus_retrieval.py:196-199. Going through the repo's
    # declared configs would pick the whole `en` split; we want these two files, in this order.
    # `hf://datasets/<repo>@<revision>/<path>` is the pinned form (D9). Without it the stream takes
    # whatever `main` points at today, and every doc id, scan window and top-k list on the volume
    # indexes into the exact bytes THIS build read -- a silent reorder upstream invalidates all of
    # them with nothing raising. `arb/exp-ood`'s corpus.py:60 pins the same way.
    assert revision, (
        f"corpus.revision is empty in config.yaml: streaming {dataset} at whatever `main` points "
        f"at today is the one unpinned input this branch has (D9). Put the dataset's commit sha "
        f"there; `main` is accepted only as an explicit, recorded choice."
    )
    ds = load_dataset(
        "parquet",
        data_files={"train": f"hf://datasets/{dataset}@{revision}/{path}"},
        split="train",
        streaming=True,
    )
    for row_i, doc in enumerate(ds):
        text = ""
        for col in TEXT_COLUMNS:
            if doc.get(col):
                text = doc[col]
                break
        if text:
            yield row_i, text


def _collect(cfg, budget, tok):
    """Tokenize documents from the head of each part until its share of `budget` is reached.

    Returns (docs, parts, max_doc_len) where docs is [(part_index, row, ids)] in STREAM order.
    """
    import numpy as np

    files = cfg["corpus"]["files"]
    dataset = cfg["corpus"]["dataset"]
    revision = cfg["corpus"].get("revision") or ""
    share = budget // len(files)
    docs, parts, max_len = [], [], 0
    for part_i, path in enumerate(files):
        # The last part carries the rounding remainder so the parts sum to exactly `budget`.
        want = budget - share * (len(files) - 1) if part_i == len(files) - 1 else share
        got, first_row, last_row, n_docs, n_flush = 0, None, None, 0, 0
        t0 = time.time()
        stream = _doc_stream(dataset, path, revision)
        while got < want:
            buf, rows = [], []
            for row_i, text in stream:
                if first_row is None:
                    first_row = row_i
                buf.append(text)
                rows.append(row_i)
                if len(buf) >= TOK_BATCH:
                    break
            if not buf:
                break  # the part ran out before its share was met; the assert below reports it
            for r, ids in zip(rows, tok(buf, add_special_tokens=False)["input_ids"], strict=True):
                if got >= want:
                    break
                if not ids:  # whitespace-only document
                    continue
                arr = np.asarray(ids, dtype=np.int32)
                docs.append((part_i, r, arr))
                got += len(arr)
                n_docs += 1
                last_row = r
                max_len = max(max_len, len(arr))
            n_flush += 1
            if n_flush % 25 == 0:
                print(f"[corpus] part {path[-26:]}: {got / 1e6:.2f}M / {want / 1e6:.2f}M", flush=True)
        assert got >= want, (
            f"part {path} yielded only {got} tokens of the {want} wanted: the part ran out, which "
            f"means the slice is no longer the head of one part"
        )
        parts.append(
            {"part": part_i, "file": path, "rows": [first_row, last_row], "docs": n_docs, "tokens": got}
        )
        print(
            f"[corpus] part {path[-26:]}: {got} tokens in {n_docs} docs, rows "
            f"{first_row}..{last_row}, {time.time() - t0:.0f}s",
            flush=True,
        )
    return docs, parts, max_len


def run(cfg, args):
    """Build (or rebuild) the corpus for one base under args['root'].

    `--arm <id>` builds ONE OOD arm's in-domain corpus instead (`run_arm` at the end of this file);
    the English corpus path below is untouched by it.
    """
    import numpy as np
    from transformers import AutoTokenizer

    if args.get("arm"):
        return run_arm(cfg, args)

    _go_online()

    base = args["base"]
    root = args["root"]
    assert base, "product corpus needs --base (the tokenizers differ, so the corpus does too)"
    budget = int(args.get("tokens") or cfg["corpus"]["tokens"])
    sizes = [s for s in cfg["corpus"]["sizes"] if s * 1_000_000 <= budget]
    assert sizes, f"--tokens {budget} is below the smallest nested size {cfg['corpus']['sizes'][0]}M"
    assert budget == sizes[-1] * 1_000_000, (
        f"--tokens {budget} is not one of the nested sizes {cfg['corpus']['sizes']} (in millions); "
        f"a budget between two sizes would leave the largest subset incomplete"
    )

    snap = C.snapshot(cfg, cfg["bases"][base]["hf"])
    tok = AutoTokenizer.from_pretrained(snap)
    print(f"[corpus] base {base} vocab {len(tok)} budget {budget / 1e6:.0f}M sizes {sizes}", flush=True)

    docs, parts, max_len = _collect(cfg, budget, tok)
    total = sum(len(a) for _, _, a in docs)
    # checklist item 59: an HF streaming reader's worker threads can abort the interpreter at
    # finalisation. Drop the iterators and collect now, while a crash would still be an honest
    # failure, rather than leaving them alive across the write.
    gc.collect()

    # Permute the DOCUMENT order once, with the corpus seed, and tag by cumulative tokens.
    rng = np.random.default_rng(cfg["corpus"]["seed"])
    perm = rng.permutation(len(docs))
    rows, cum, n_clamped = [], 0, 0
    per_size = {s: 0 for s in sizes}
    for new_i, old_i in enumerate(perm):
        part_i, row, arr = docs[int(old_i)]
        tag = C.size_tag_of(cum, len(arr), sizes)
        if cum + len(arr) > tag * 1_000_000:
            n_clamped += 1
        rows.append(
            {
                "doc": new_i,
                "offset": cum,
                "len": int(len(arr)),
                "size_tag": tag,
                "part": part_i,
                "row": row,
            }
        )
        per_size[tag] += int(len(arr))
        cum += len(arr)
    assert cum == total, f"offset bookkeeping: {cum} != {total}"

    out = C.corpus_dir(base, root)
    with C.outdir(
        out,
        args,
        inputs={
            "base": base,
            "budget_tokens": budget,
            "dataset": f"{cfg['corpus']['dataset']}@{cfg['corpus'].get('revision') or 'UNPINNED'}",
            "files": cfg["corpus"]["files"],
        },
    ) as od:
        od.write_array("tokens.i32", np.concatenate([docs[int(i)][2] for i in perm]), "int32")
        od.write_jsonl("docs.jsonl", rows)
        od.note(f"tokenizer: {snap} (vocab {len(tok)}), add_special_tokens=False, no truncation")
        od.note(f"{len(rows)} documents, {total} tokens, longest document {max_len} tokens")
        od.note(
            "document order is PERMUTED by np.random.default_rng(corpus.seed="
            f"{cfg['corpus']['seed']}); tokens.i32 is in that order, so subset k is the prefix of "
            "documents with size_tag <= k"
        )
        for s in sizes:
            cumulative = sum(v for k, v in per_size.items() if k <= s)
            od.note(f"size {s}M: {per_size[s]} tokens tagged, {cumulative} cumulative")
        for p in parts:
            od.note(
                f"part {p['part']} {p['file']}: rows {p['rows'][0]}..{p['rows'][1]} "
                f"({p['docs']} non-empty docs, {p['tokens']} tokens)"
            )
        if n_clamped:
            od.note(
                f"{n_clamped} document(s) crossed their size budget and were clamped to the next "
                "tag up (only the document that crosses the total budget can do this)"
            )
    return {
        "docs": len(rows),
        "tokens": total,
        "max_doc_len": max_len,
        "per_size": per_size,
        "parts": parts,
        "out": out,
    }


# =============================================================================================
# `corpus --arm <id>`: one OOD arm's own in-domain corpus (design §2, §4)
#
#     <root>/base/<base>/corpora/<arm>/
#         tokens.i32, docs.jsonl, README.md, index.json    -- exactly the products above
#         pool_windows.i32 [pool_n, 512], pool.jsonl       -- the TARGET pool, drawn from the rows
#                                                             AFTER the corpus in the same stream
#         stream.json                                      -- where the permutation stood
#
# The arm's rows come from ONE seeded permutation (`common.arm_perm`), consumed in order: corpus
# first, then the target pool. `targets` never touches the network -- everything it needs about the
# rows after the corpus is in pool_windows.i32 / pool.jsonl, written HERE, by the one CPU product
# that is allowed online. That is also what makes the disjointness assertion a property of the file
# rather than of two independent draws agreeing.
# =============================================================================================

POOL_N = 320  # `_realact`'s n + 256 pool, taken at 64 targets per arm
POOL_WINDOW = 512  # the realact window; a pool document must have at least this many tokens
PROBE_ROWS = 256  # rows tokenized to estimate tokens/row before the first block is chosen
BLOCK_SLACK = 1.35  # how much more than the estimate a block asks for, so one pass usually suffices


def _hf_revision(dataset: str) -> str:
    """The dataset repo's current commit sha. FineWeb-2 and smol-xl are MUTABLE `main` refs, so
    this goes into the corpus README and into every target's `ids.jsonl` row (design §2)."""
    from huggingface_hub import HfApi

    return HfApi().repo_info(dataset, repo_type="dataset").sha


class _ParquetSource:
    """Rows of one or more parquet files, addressed by a GLOBAL index over the files in order.

    Only the text column is read: `cleaned_formulas` carries a 2.8 GB inline `image` column beside
    its 552k formulas, and a column projection is the difference between 30 MB and all of it.
    """

    def __init__(self, dataset: str, files: list[str], text: str):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        self.dataset, self.files, self.text = dataset, files, text
        self.fs = HfFileSystem()
        self.counts = []
        for f in files:
            with self.fs.open(self._path(f), "rb") as fh:
                md = pq.ParquetFile(fh).metadata
                self.counts.append(md.num_rows)
        self.n_rows = sum(self.counts)

    def _path(self, f: str) -> str:
        return f"datasets/{self.dataset}/{f}"

    def fetch(self, wanted: list[int]) -> dict[int, str]:
        """{global row -> text} for `wanted` (any order). One streaming pass per file touched."""
        import pyarrow.parquet as pq

        want = set(int(w) for w in wanted)
        out: dict[int, str] = {}
        base = 0
        for f, n in zip(self.files, self.counts, strict=True):
            hi = base + n
            if any(base <= w < hi for w in want):
                with self.fs.open(self._path(f), "rb") as fh:
                    pf = pq.ParquetFile(fh)
                    i = base
                    for batch in pf.iter_batches(batch_size=2048, columns=[self.text]):
                        col = batch.column(0)
                        if any(i <= w < i + len(col) for w in want):
                            vals = col.to_pylist()
                            for k, v in enumerate(vals):
                                if (i + k) in want and v:
                                    out[i + k] = v
                        i += len(col)
                    assert i == hi, f"{f}: streamed {i - base} rows, footer said {n}"
            base = hi
        return out


class _JsonlSource:
    """Rows of one or more json-LINES files (optionally zstd-compressed), downloaded to the HF cache.

    `the-stack-smol-xl`'s `data/<lang>/data.json` is json lines despite the extension (MEASURED
    2026-09-18) and Proof-Pile-2 ships `.jsonl.zst`; both are small enough (<= 182 MB) to fetch
    once into HF_HOME and read twice, which is cheaper than two network passes.
    """

    def __init__(self, dataset: str, files: list[str], text: str, zst: bool):
        from huggingface_hub import hf_hub_download

        self.dataset, self.files, self.text, self.zst = dataset, files, text, zst
        self.licences: dict[int, list] = {}  # smol-xl's per-file licence, for R8's examples
        self.paths = [hf_hub_download(dataset, f, repo_type="dataset") for f in files]
        self.counts = [sum(1 for _ in self._lines(p)) for p in self.paths]
        self.n_rows = sum(self.counts)

    def _lines(self, path: str):
        import io

        if self.zst:
            import zstandard

            with open(path, "rb") as fh:
                yield from io.TextIOWrapper(
                    zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"
                )
        else:
            with open(path, encoding="utf-8") as fh:
                yield from fh

    def fetch(self, wanted: list[int]) -> dict[int, str]:
        want = {int(w) for w in wanted}
        out: dict[int, str] = {}
        base = 0
        for path, n in zip(self.paths, self.counts, strict=True):
            hi = base + n
            if any(base <= w < hi for w in want):
                for k, line in enumerate(self._lines(path)):
                    if (base + k) in want:
                        row = json.loads(line)
                        if row.get(self.text):
                            out[base + k] = row[self.text]
                            if row.get("max_stars_repo_licenses"):
                                self.licences[base + k] = row["max_stars_repo_licenses"]
            base = hi
        return out


def _source(spec: dict):
    reader = spec["reader"]
    if reader in ("parquet", "formulas"):
        return _ParquetSource(spec["dataset"], spec["files"], spec["text"])
    if reader in ("jsonl", "jsonl_zst"):
        return _JsonlSource(spec["dataset"], spec["files"], spec["text"], zst=reader == "jsonl_zst")
    raise AssertionError(f"unknown reader {reader!r}")


def _formula_docs(pairs, tok, min_tok: int = POOL_WINDOW):
    """`formulas`: consecutive permuted rows joined by a blank line into >= `min_tok`-token documents.

    Yields (rows, ids). The joined text is tokenized as ONE document -- concatenating the ids of
    separately tokenized formulas would not be the same token sequence at the joins, and the arm
    exists precisely to measure what the tokenizer does to symbol-dense text. Rows left in the
    trailing buffer when the block ends do not make a document and are dropped with it; the stream
    position advances by the whole block either way, so the corpus / pool boundary stays exact.
    """
    import numpy as np

    buf_rows, buf_txt = [], []
    for row, text in pairs:
        buf_rows.append(row)
        buf_txt.append(text)
        if sum(len(t) for t in buf_txt) < 3 * min_tok:  # ~3 chars/token, a cheap lower bound
            continue
        ids = tok("\n\n".join(buf_txt), add_special_tokens=False)["input_ids"]
        if len(ids) >= min_tok:
            yield list(buf_rows), np.asarray(ids, dtype=np.int32)
            buf_rows, buf_txt = [], []


def _arm_stream(arm, spec, tok, seed, want_tokens):
    """Yield (perm_pos, rows, ids) of the arm's documents in PERMUTED row order.

    `perm_pos` is the index into the permutation AFTER this document -- what a consumer records to
    say where it stopped, and where the next consumer picks up. Rows are read in blocks, one
    streaming pass over the source per block, sized from the measured tokens-per-row rate, so a
    16M-token arm normally costs one pass over its slice.
    """
    import numpy as np

    src = _source(spec)
    perm = C.arm_perm(arm, src.n_rows, seed)
    is_formulas = spec["reader"] == "formulas"
    pos, rate = 0, None
    while pos < len(perm):
        take = (
            min(PROBE_ROWS, len(perm) - pos)
            if rate is None
            else int(max(1024, min(len(perm) - pos, want_tokens / max(rate, 1e-6) * BLOCK_SLACK)))
        )
        block = [int(r) for r in perm[pos : pos + take]]
        idx_of = {r: i for i, r in enumerate(block)}
        texts = src.fetch(sorted(block))
        pairs = [(r, texts[r]) for r in block if r in texts]
        got = 0
        if is_formulas:
            for rows, ids in _formula_docs(pairs, tok):
                got += len(ids)
                yield pos + idx_of[rows[-1]] + 1, rows, ids
        else:
            for s in range(0, len(pairs), TOK_BATCH):
                chunk = pairs[s : s + TOK_BATCH]
                enc = tok([t for _, t in chunk], add_special_tokens=False)["input_ids"]
                for (r, _), ids in zip(chunk, enc, strict=True):
                    if ids:
                        got += len(ids)
                        yield pos + idx_of[r] + 1, [r], np.asarray(ids, dtype=np.int32)
        rate = got / max(take, 1)
        pos += take
        print(f"[corpus] {arm}: block of {take} rows -> {got} tokens ({rate:.0f} tok/row)", flush=True)


def run_arm(cfg, args):
    """Build one OOD arm's corpus AND its target pool. CPU, network (the only online product)."""
    import numpy as np
    from transformers import AutoTokenizer

    _go_online()

    base, root, arm = args["base"], args["root"], args["arm"]
    assert base, "corpus --arm needs --base (the tokenizer, hence the corpus, is per base)"
    spec = C.ood_arm(cfg, arm)
    set_name = args.get("heldout") or ""
    seed = int(args.get("arm_seed") or cfg["heldout"].get(set_name, {}).get("seed") or 20260918)
    sizes = [int(s) for s in spec["sizes"]]
    budget = int(args.get("tokens") or sizes[-1] * 1_000_000)
    sizes = [s for s in sizes if s * 1_000_000 <= budget]
    assert sizes and budget == sizes[-1] * 1_000_000, (
        f"--tokens {budget} is not one of arm {arm}'s nested sizes {spec['sizes']} (in millions)"
    )
    pool_n = int(args.get("n") or POOL_N)

    snap = C.snapshot(cfg, cfg["bases"][base]["hf"])
    tok = AutoTokenizer.from_pretrained(snap)
    revision = _hf_revision(spec["dataset"])
    print(
        f"[corpus] arm {arm} ({spec['family']}) {spec['dataset']}@{revision[:12]} "
        f"budget {budget / 1e6:.0f}M sizes {sizes} seed {seed}",
        flush=True,
    )

    stream = _arm_stream(arm, spec, tok, seed, budget)
    docs, cum, t0 = [], 0, time.time()
    perm_pos = 0
    for pos, rows, ids in stream:
        if cum >= budget:
            break
        docs.append((rows, ids))
        cum += len(ids)
        perm_pos = pos
        if len(docs) % 2000 == 0:
            print(f"[corpus] {arm}: {cum / 1e6:.2f}M / {budget / 1e6:.2f}M in {len(docs)} docs", flush=True)
    assert cum >= budget, (
        f"arm {arm}: the source ran out at {cum} tokens of the {budget} wanted -- the slice in "
        f"config.yaml is too small for sizes {spec['sizes']}"
    )
    corpus_docs, corpus_end = docs, perm_pos
    print(
        f"[corpus] {arm}: corpus {cum} tokens in {len(corpus_docs)} docs, permutation at "
        f"{corpus_end} of the arm's rows, {time.time() - t0:.0f}s",
        flush=True,
    )

    # the TARGET POOL: the next `pool_n` documents with >= POOL_WINDOW tokens, same stream
    pool, pool_end = [], corpus_end
    for pos, rows, ids in stream:
        pool_end = pos
        if len(ids) >= POOL_WINDOW:
            pool.append((rows, ids[:POOL_WINDOW], len(ids)))
        if len(pool) >= pool_n:
            break
    assert len(pool) == pool_n, (
        f"arm {arm}: only {len(pool)} of {pool_n} pool documents have >= {POOL_WINDOW} tokens "
        f"before the arm's rows ran out"
    )
    print(f"[corpus] {arm}: pool {len(pool)} docs, permutation at {pool_end}", flush=True)
    gc.collect()  # drop the reader's iterators before the write (checklist item 59)

    rows_out, per_size, n_clamped, cum2 = [], {s: 0 for s in sizes}, 0, 0
    for i, (src_rows, ids) in enumerate(corpus_docs):
        tag = C.size_tag_of(cum2, len(ids), sizes)
        if cum2 + len(ids) > tag * 1_000_000:
            n_clamped += 1
        rows_out.append(
            {"doc": i, "offset": cum2, "len": int(len(ids)), "size_tag": tag, "rows": src_rows}
        )
        per_size[tag] += int(len(ids))
        cum2 += len(ids)
    assert cum2 == cum, f"offset bookkeeping: {cum2} != {cum}"

    out = C.corpus_dir(base, root, arm)
    inputs = {
        "base": base,
        "arm": arm,
        "family": spec["family"],
        "dataset": f"{spec['dataset']}@{revision}",
        "files": spec["files"],
        "budget_tokens": budget,
        "seed": seed,
    }
    with C.outdir(out, args, inputs=inputs) as od:
        od.write_array("tokens.i32", np.concatenate([a for _, a in corpus_docs]), "int32")
        od.write_jsonl("docs.jsonl", rows_out)
        od.write_array("pool_windows.i32", np.stack([w for _, w, _ in pool]), "int32")
        od.write_jsonl(
            "pool.jsonl",
            [
                {"pool_i": i, "rows": r, "n_tok": int(n)}
                for i, (r, _, n) in enumerate(pool)
            ],
        )
        od.write_json(
            "stream.json",
            {
                "arm": arm,
                "family": spec["family"],
                "dataset": spec["dataset"],
                "revision": revision,
                "files": spec["files"],
                "split": spec.get("split"),
                "licence": spec.get("licence"),
                "seed": seed,
                "sizes": sizes,
                "budget_tokens": budget,
                "corpus_docs": len(corpus_docs),
                "corpus_tokens": cum,
                "perm_pos_corpus_end": int(corpus_end),
                "perm_pos_pool_end": int(pool_end),
                "pool_n": len(pool),
                "pool_window": POOL_WINDOW,
                "script": spec["script"],
                "lid": spec.get("lid"),
                "unspaced": bool(spec["unspaced"]),
            },
        )
        od.section(
            "Draw",
            [
                f"Arm `{arm}` (family `{spec['family']}`) of the OOD set, design "
                "`infra/2026-09-18_ood-eval-design.md` §2/§4.",
                "",
                f"- source `{spec['dataset']}` at revision `{revision}`, files "
                f"{spec['files']}, text field `{spec['text']}`, reader `{spec['reader']}`;",
                f"- ONE permutation of the slice's rows, `np.random.default_rng({seed} ^ "
                f"crc32({arm!r}))` (`common.arm_perm`), consumed IN ORDER;",
                f"- the corpus took the first {len(corpus_docs)} documents ({cum} tokens, budget "
                f"{budget}) and stopped at permutation position {corpus_end};",
                f"- the target pool is the next {len(pool)} documents with >= {POOL_WINDOW} "
                f"tokens, ending at permutation position {pool_end}. `pool_windows.i32` holds "
                f"their first {POOL_WINDOW} tokens, which is everything `targets` needs -- so the "
                "target draw runs OFFLINE and corpus and target documents are disjoint by "
                "construction, not by a later check;",
                f"- tokenized with `{snap}` (vocab {len(tok)}), `add_special_tokens=False`, no "
                "truncation; documents are stored in permutation order, so nested subset k is the "
                "prefix of `tokens.i32` with `size_tag <= k`.",
            ],
        )
        for s in sizes:
            cumulative = sum(v for k, v in per_size.items() if k <= s)
            od.note(f"size {s}M: {per_size[s]} tokens tagged, {cumulative} cumulative")
        od.note(f"{len(rows_out)} documents, {cum} tokens; `rows` is the source row index of each")
        od.note(
            "verbatim-span rate: NOT computed here -- `targets` reports, per arm, the fraction of "
            "its 64 shown spans that occur verbatim in this tokens.i32 (design §4)"
        )
        if n_clamped:
            od.note(f"{n_clamped} document(s) crossed their size budget and were clamped up a tag")
    return {
        "arm": arm,
        "docs": len(rows_out),
        "tokens": cum,
        "per_size": per_size,
        "pool": len(pool),
        "perm_pos_corpus_end": int(corpus_end),
        "perm_pos_pool_end": int(pool_end),
        "revision": revision,
        "out": out,
    }
