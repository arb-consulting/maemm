"""The steered model as a reader (`evals.downstream.common.plain_steer`): every vector at each strength, 64 samples and
no greedy decode, as families `<vector_id>|plain_steered@<s>`, and their text health (`health_table`).
"""
from evals.downstream.common import plain_steer as P

OPERATION = P.OPERATION
ARMS = P.ARMS
SAMPLES = P.samples_by_strength(P.MAIN_SAMPLES)
HEALTH_CSV = "tables/plain_steered_health.csv"
HEALTH_COLUMNS = ("vector_id", "kind", "arm", *P.HEALTH_COLUMNS[1:])
SAMPLE_KEYS = ("sample_id", "seed", "text", "n_tokens", "eos_terminated", "stop_token", "loglik_mean",
               "loglik_sum")


def tasks(bank, code, wanted=lambda family_id: True):
    """`[(task_id, payload)]` over every (vector, strength) cell whose family id `wanted` admits."""
    directions = {entry["vector_id"]: entry["direction"] for entry in bank}
    cells = [cell for cell in P.cells([entry["vector_id"] for entry in bank], SAMPLES)
             if wanted(f"{cell[0]}|{P.arm_name(cell[1])}")]
    return [(f"{P.ARM}|{number:04d}|{payload['rows'][0]['vector_id']}", {**payload, "code": code})
            for number, payload in enumerate(P.payloads(directions, cells))]


def execute(worker, payload):
    result = P.generate(worker, payload)
    data = {k: result[k] for k in ("rows", "sink_id", "sink_token", "eos_ids", "read_layer")}
    return {"data": data, "gpu_seconds": result["gpu_seconds"], "gen_seconds": result["gen_seconds"],
            "gpu_name": result["gpu_name"], "runtime_versions": result["runtime_versions"]}


def families(results):
    """The generated rows back into `<vector_id>|plain_steered@<s>` families, each checked complete."""
    out = {}
    for _task_id, result in results:
        data = result["data"]
        for row in data["rows"]:
            arm = P.arm_name(row["strength"])
            family_id = f"{row['vector_id']}|{arm}"
            family = out.setdefault(family_id, {
                "family_id": family_id, "vector_id": row["vector_id"], "arm": arm,
                "strength": float(row["strength"]), "coefficient": float(row["coefficient"]),
                "sink_token": data["sink_token"], "read_layer": data["read_layer"], "samples": []})
            family["samples"].append({k: row.get(k) for k in SAMPLE_KEYS})
    for family in out.values():
        family["samples"].sort(key=lambda sample: sample["sample_id"])
        if [sample["sample_id"] for sample in family["samples"]] != list(range(SAMPLES[family["strength"]])):
            raise ValueError(f"Incomplete or duplicate samples in {family['family_id']}")
    return out


def health_rows(all_families):
    """`plain_steer.health` over this arm's families, per vector and strength, with the vector's kind."""
    from .mc10 import vector_kind
    rows = [{"vector_id": family["vector_id"], "strength": family["strength"], **sample}
            for family in all_families.values() if family.get("arm") in ARMS
            for sample in family["samples"]]
    return [{**r, "kind": vector_kind(r["vector_id"]), "arm": P.arm_name(r["strength"]),
             "strength": f"{r['strength']:g}"} for r in P.health(rows)]


def health_table(root, all_families):
    """Write `HEALTH_CSV` from the saved families; its rows."""
    from . import mc10
    rows = health_rows(all_families)
    mc10._write_csv(root, HEALTH_CSV, HEALTH_COLUMNS, rows)
    return rows
