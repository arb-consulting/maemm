"""The NLA verbalizer as a reader of the same unit directions, on the AxBench package's `NlaWorker`; the
close rate of its tags is reported and gates nothing (methodology.md, "Readers").
"""
import time

from evals.downstream.steering_vector_inversion.artifacts import seed_for
from evals.downstream.steering_vector_inversion.nla import reader_pins

OPERATION = "bipo_nla"
ARM = "nla_native"
KEEP = ("text", "full_text", "n_tokens", "closed", "eos", "capped", "seed")


def tasks(config, bank, code):
    """One task per vector: 64 sampled explanations and the greedy one under the vector's seed, with the
    reader's pins in the payload."""
    record, _ = reader_pins()
    return [(f"{entry['vector_id']}|nla", {"operation": OPERATION, "vector_id": entry["vector_id"],
                                           "direction": entry["direction"], "n_samples": config.samples,
                                           "seed": seed_for(config.seed, entry["vector_id"], "nla"), "code": code,
                                           "pins": record})
            for entry in bank]


def families(results):
    """One family per vector: the greedy explanation (sample -1) and the sampled ones."""
    out = {}
    for _, result in results:
        data = result["data"]
        rows = [(-1, data["greedy"])] + list(enumerate(data["samples"]))
        family_id = f"{data['vector_id']}|{ARM}"
        out[family_id] = {"family_id": family_id, "vector_id": data["vector_id"], "arm": ARM,
                          "close_rate": data["close_rate"],
                          "samples": [{"sample_id": i, "text": row["text"], "full_text": row["full_text"],
                                       "n_tokens": row["n_tokens"], "closed": row["closed"],
                                       "capped": row["capped"], "seed": row["seed"]} for i, row in rows]}
    return out


def close_rates(families_):
    """{by_vector, pooled, min_close_rate, below_min_close_rate} of the explanations whose tags closed."""
    _, floor = reader_pins()
    by_vector = {f["vector_id"]: f["close_rate"] for f in families_.values()
                 if f.get("arm") == ARM and f.get("close_rate") is not None}
    pooled = (sum(by_vector.values()) / len(by_vector)) if by_vector else None
    return {"by_vector": by_vector, "pooled": pooled, "min_close_rate": floor,
            "below_min_close_rate": pooled is None or pooled < floor}


def section(families_):
    """The report's lines on how often the verbalizer closed its tags."""
    rates = close_rates(families_)
    lines = ["## The verbalizer's close rate", ""]
    if rates["pooled"] is None:
        return lines + ["This run directory holds no NLA family.", ""]
    low = sorted(v for v, r in rates["by_vector"].items() if r < rates["min_close_rate"])
    return lines + [
        f"{rates['pooled']:.3f} of the verbalizer's sampled explanations closed their `<explanation>` tags, "
        f"averaged over {len(rates['by_vector'])} vectors, "
        f"{'below' if rates['below_min_close_rate'] else 'at or above'} the reader's own line of "
        f"{rates['min_close_rate']:.2f}. A judge reads the body between the tags when they close and the "
        "whole text, less a leading open tag, when they do not, so an unclosed explanation is read and "
        "not dropped; the rate is reported and gates nothing. "
        + (f"Vectors below the line: {', '.join(f'`{v}`' for v in low)}." if low
           else "No vector is below the line."), ""]


def execute(worker, payload):
    """One vector's explanations, on the verbalizer's own worker."""
    started = time.time()
    samples, greedy = worker.explain(payload["direction"], payload["n_samples"], payload["seed"], True)
    rows = [{k: s[k] for k in KEEP} for s in samples]
    data = {"vector_id": payload["vector_id"], "samples": rows, "greedy": {k: greedy[k] for k in KEEP},
            "close_rate": sum(r["closed"] for r in rows) / len(rows),
            "pins": worker.pins_data()}
    return worker.result(data, started)
