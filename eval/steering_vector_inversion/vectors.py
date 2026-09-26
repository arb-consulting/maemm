"""The direction bank: one difference-of-means direction per concept, its candidates and the target set.

A direction is the unit difference between the mean layer-42 activations of a concept's 48 positive and
48 negative construction texts, read on the clean base. It is not centred: a shared background cancels in
a difference of means.
"""
import time

import numpy as np

from .analysis import unit
from .data import candidate_sets, target_ids
from .execution import chunks, gpu_tasks

READ_BATCH = 64


def read_records(run, executor, records, namespace):
    """The clean base's mean activation of each record, in order (`model.ModelWorker.read`)."""
    batches = list(chunks(records, READ_BATCH))
    tasks = [(i, {"operation": "read", "records": batch}) for i, batch in enumerate(batches)]
    output = [None] * len(records)
    for index, result in gpu_tasks(run, executor, tasks, namespace):
        if len(result["data"]) != len(batches[index]):
            raise ValueError("GPU readout returned the wrong number of rows")
        output[index * READ_BATCH:index * READ_BATCH + len(batches[index])] = result["data"]
    return output


def build(run, executor, prepared):
    started = time.time()
    key = run.key("vectors", prepared, modules=("data", "model", "vectors", "analysis"))
    cached = run.cached_arrays("vectors/bank.json", key)
    if cached is not None:
        return cached
    cfg = run.config
    needed = sorted({r["text_id"] for c in prepared["concepts"] for r in c["positive"] + c["negative"]})
    records = [prepared["texts"][text_id] for text_id in needed]
    readouts = dict(zip(needed, read_records(run, executor, records, "vectors/construction_reads")))
    concepts, directions, excluded = [], [], []
    for concept in prepared["concepts"]:
        positive = [readouts[r["text_id"]] for r in concept["positive"]]
        negative = [readouts[r["text_id"]] for r in concept["negative"]]
        reason = None
        if any(r["status"] != "ok" for r in positive + negative):
            reason = "failed_construction_readout"
        else:
            p = np.mean([r["mean"] for r in positive], axis=0, dtype=np.float64).astype(np.float32)
            n = np.mean([r["mean"] for r in negative], axis=0, dtype=np.float64).astype(np.float32)
            v = unit(p - n)
            if v is None:
                reason = "degenerate_contrast"
        if reason:
            excluded.append({"concept_id": concept["concept_id"], "genre": concept["genre"], "reason": reason})
            continue
        concepts.append(concept)
        directions.append(v)
    if not concepts:
        raise ValueError("No technically usable contrastive directions")
    data = {"concepts": concepts, "directions": np.stack(directions),
            "candidates": candidate_sets(concepts, cfg.data_seed),
            "target_ids": target_ids(concepts, cfg.concepts_per_genre, cfg.data_seed), "exclusions": excluded}
    run.save_arrays("vectors/bank.json", key, data)
    run.stage_done("vectors", started, usable=len(concepts), excluded=len(excluded))
    return data
