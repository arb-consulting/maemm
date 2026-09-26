"""The Jacobian lens as a reader: the ten word-like tokens a direction promotes (`eval/common/lens_io.py`),
summarised by the reference judge from the tokens alone (`eval/common/lens_summary.py`). The summary is the
arm's one text; a direction without one has no family.
"""
import time

import numpy as np

from eval.common.lens_summary import summary_request

from . import config as C

OPERATION = "lens_words"
ARM = "jlens"

WORDS = "lens/words.json"
SUMMARIES = "lens/summaries.json"
INSTRUMENT = "lens_summary"
KIND = "summary_req"


def tasks(bank, code):
    """One task for the whole bank: the lens file is gigabytes, so it is loaded once."""
    ids = [entry["vector_id"] for entry in bank]
    directions = np.asarray([entry["direction"] for entry in bank], dtype=np.float32)
    return [("lens|words", {"operation": OPERATION, "vector_ids": ids, "directions": directions,
                            "top_word": C.LENS_TOP_WORDS, "lens": dict(C.LENS), "code": code})]


def execute(worker, payload):
    """The `top_word` word-like tokens each direction promotes at the read layer, best first; the lens is
    loaded once per worker."""
    import torch

    from eval.common.lens_io import Unembed, load_lens, top_words
    started = time.time()
    if getattr(worker, "lens", None) is None:
        worker.lens, worker.lens_provenance = load_lens(
            worker.device, **payload["lens"], read_layer=worker.config.read_layer,
            d_model=worker.config.hidden_size)
        worker.lens_unembed = Unembed(worker.base)
        worker.lens_mask = wordlike(worker.tokenizer, worker.lens_unembed.vocab)
    words = top_words(np.asarray(payload["directions"], dtype=np.float32), worker.lens,
                      worker.lens_unembed, worker.tokenizer, worker.lens_mask,
                      layer=worker.config.read_layer, device=worker.device,
                      top_word=int(payload["top_word"]))
    data = {"vector_ids": list(payload["vector_ids"]), "words": [list(row) for row in words],
            "lens": {k: worker.lens_provenance[k]
                     for k in ("repo", "revision", "file", "sha256", "n_prompts")},
            "layer": int(worker.config.read_layer)}
    return {"data": data, "gpu_seconds": time.time() - started, "runtime_versions": worker.versions,
            "gpu_name": torch.cuda.get_device_name(worker.device) if worker.device.type == "cuda" else "CPU"}


def wordlike(tokenizer, vocab):
    """The word-like vocabulary mask."""
    from eval.common.lens_io import wordlike_mask

    return wordlike_mask(tokenizer, vocab)


def requests(words):
    """One summariser request per direction over the shared prompt; the vector id travels in `meta` only."""
    return [summary_request(row["words"], {"vector_id": row["vector_id"]}) for row in words]


def summaries(words, records):
    """`{vector_id: {"summary", "status"}}` from the summariser's records, in request order."""
    out = {}
    for row, record in zip(words, records):
        record = record or {}
        status = record.get("status") or "unavailable"
        text = (record.get("text") or "").strip()
        if status == "ok" and not text:
            status = "parse_fail"
        out[row["vector_id"]] = {"summary": text if status == "ok" else None, "status": status}
    return out


def families(summarised):
    """One family per direction whose summary came back: a single bundle holding that single text."""
    out = {}
    for vector_id, record in sorted(summarised.items()):
        if record.get("status") != "ok" or not record.get("summary"):
            continue
        family_id = f"{vector_id}|{ARM}"
        out[family_id] = {"family_id": family_id, "vector_id": vector_id, "arm": ARM,
                          "samples": [{"sample_id": 0, "text": record["summary"]}]}
    return out
