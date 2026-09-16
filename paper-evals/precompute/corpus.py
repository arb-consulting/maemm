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


def _doc_stream(dataset: str, path: str):
    """Yield (row_index_in_part, text) for every non-empty document of one parquet part, in file
    order. `row_index_in_part` counts EVERY row, including the empty ones that are skipped, so it
    addresses the source file rather than our stream."""
    from datasets import load_dataset

    # The direct-parquet route of eval/corpus_retrieval.py:196-199. Going through the repo's
    # declared configs would pick the whole `en` split; we want these two files, in this order.
    ds = load_dataset(
        "parquet", data_files={"train": f"hf://datasets/{dataset}/{path}"}, split="train", streaming=True
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
    share = budget // len(files)
    docs, parts, max_len = [], [], 0
    for part_i, path in enumerate(files):
        # The last part carries the rounding remainder so the parts sum to exactly `budget`.
        want = budget - share * (len(files) - 1) if part_i == len(files) - 1 else share
        got, first_row, last_row, n_docs, n_flush = 0, None, None, 0, 0
        t0 = time.time()
        stream = _doc_stream(dataset, path)
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
    """Build (or rebuild) the corpus for one base under args['root']."""
    import numpy as np
    from transformers import AutoTokenizer

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
        out, args, inputs={"base": base, "budget_tokens": budget, "files": cfg["corpus"]["files"]}
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
