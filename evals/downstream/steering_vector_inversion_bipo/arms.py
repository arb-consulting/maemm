"""The vector bank, the inversion-prompt arms of each vector, and how a family is cut into bundles."""
import numpy as np

from evals.downstream.steering_vector_inversion.analysis import unit
from evals.downstream.steering_vector_inversion.artifacts import seed_for
from evals.downstream.steering_vector_inversion.execution import chunks
from evals.downstream.steering_vector_inversion.generate import inversion_jobs

from . import config as C
from . import lens_arm, retrieval_arm

LOOKUP_ARMS = (retrieval_arm.ARM, lens_arm.ARM)
SAMPLES = C.BUNDLES * C.BUNDLE_SIZE
ARM_MODEL = {"maem": "inverter", "base_l1": "base"}


def _direction(vector, what):
    direction = unit(vector)
    if direction is None:
        raise ValueError(f"Degenerate direction for {what}")
    return direction


def vector_bank(train_results, primary_epoch=C.PRIMARY_EPOCH):
    """One learned vector per behaviour, the checkpoint at `primary_epoch` (20)."""
    entries = []
    for result in train_results:
        behaviour, train_seed = result["behaviour"], int(result["train_seed"])
        checkpoints = result["checkpoints"]
        epoch = primary_epoch if str(primary_epoch) in checkpoints else max(int(e) for e in checkpoints)
        vector = np.asarray(checkpoints[str(epoch)], dtype=np.float32)
        entries.append({"vector_id": f"{behaviour}/bipo/s{train_seed}/+", "kind": "bipo",
                        "behaviour": behaviour, "train_seed": train_seed, "epoch": int(epoch),
                        "direction": _direction(vector, f"{behaviour} bipo"),
                        "norm": float(np.linalg.norm(vector)), "truth": [f"{behaviour}/+"]})
    return entries


def _tag(jobs, vector_id, arm):
    for job in jobs:
        job["vector_id"], job["arm"] = vector_id, arm
    return jobs


def _batches(jobs, size):
    """The greedy draw goes alone: a batch must share its decoding mode."""
    greedy = [job for job in jobs if job["greedy"]]
    sampled = [job for job in jobs if not job["greedy"]]
    return ([greedy] if greedy else []) + list(chunks(sampled, size))


def rollout_tasks(config, bank, code):
    """Every inversion-prompt arm of every vector, batched the way the pinned rollout stage batches."""
    tasks = []
    for entry in bank:
        for arm, model in ARM_MODEL.items():
            jobs = _tag(inversion_jobs(config, {"concept_id": entry["vector_id"]}, arm, entry["direction"],
                                       model=model), entry["vector_id"], arm)
            for number, batch in enumerate(_batches(jobs, config.generation_batch_size)):
                shapes = {(job["kind"], job["model"], job["greedy"]) for job in batch}
                assert len(shapes) == 1, f"Mixed generation batch in {entry['vector_id']}|{arm}"
                tasks.append((f"{entry['vector_id']}|{arm}|{number}",
                              {"operation": "generate", "jobs": batch, "code": code}))
    return tasks


def collect(results):
    """Batches back into families, each checked for the greedy row and every sampled draw once."""
    families = {}
    for task_id, result in results:
        family_id = task_id.rsplit("|", 1)[0]
        vector_id, arm = family_id.split("|")
        family = families.setdefault(family_id, {"family_id": family_id, "vector_id": vector_id,
                                                 "arm": arm, "samples": []})
        family["samples"].extend(dict(row) for row in result["data"])
    for family in families.values():
        family["samples"].sort(key=lambda sample: sample["sample_id"])
        if [sample["sample_id"] for sample in family["samples"]] != [-1, *range(SAMPLES)]:
            raise ValueError(f"Incomplete or duplicate samples in {family['family_id']}")
    return families


def heldout_families(sources, seed=C.DATA_SEED):
    """The matching side of each behaviour's held-out pairs, read as a family: 64 of them by seed (with
    replacement when there are fewer), each cut to 300 characters."""
    families = {}
    for behaviour, source in sources.items():
        rows = source["heldout"]
        if not rows:
            raise ValueError(f"No held-out pairs for {behaviour}")
        rng = np.random.default_rng(seed_for(seed, "heldout", behaviour))
        repeats = len(rows) < SAMPLES
        picks = rng.integers(0, len(rows), SAMPLES) if repeats else rng.permutation(len(rows))[:SAMPLES]
        family_id = f"{behaviour}/heldout|heldout_matching"
        families[family_id] = {
            "family_id": family_id, "vector_id": f"{behaviour}/heldout", "arm": "heldout_matching",
            "behaviour": behaviour, "truth": [f"{behaviour}/+"],
            "warning": f"only {len(rows)} held-out pairs; sampled with replacement" if repeats else None,
            "samples": [{"sample_id": index, "text": rows[int(pick)]["matching"][:300],
                         "source_id": rows[int(pick)]["source_id"]} for index, pick in enumerate(picks)]}
    return families


def family_truth(bank, family_id):
    """Truth for both kinds of family; a held-out reference is by construction its own behaviour."""
    vector_id, arm = family_id.split("|")
    if arm == "heldout_matching":
        return [f"{vector_id.split('/')[0]}/+"]
    for entry in bank:
        if entry["vector_id"] == vector_id:
            return list(entry["truth"])
    raise ValueError(f"No vector behind {family_id}")


def bundles(family, seed=C.DATA_SEED):
    """Disjoint reading bundles of at most `BUNDLE_SIZE` samples, greedy excluded.

    A generated reader's samples are shuffled by a seed from the family id and cut into `BUNDLES` bundles.
    A lookup reader (`LOOKUP_ARMS`) has one ordered reading, kept in its own order as one bundle."""
    rng = np.random.default_rng(seed_for(seed, "bundles", family["family_id"]))
    samples = {sample["sample_id"]: sample for sample in family["samples"] if sample["sample_id"] >= 0}
    order = sorted(samples)
    if family["arm"] not in LOOKUP_ARMS:
        order = rng.permutation(order)
    count = max(1, len(order) // C.BUNDLE_SIZE)
    groups = [order[b * C.BUNDLE_SIZE:(b + 1) * C.BUNDLE_SIZE] for b in range(count)]
    return [{"bundle": bundle, "sample_ids": [int(i) for i in ids],
             "texts": [samples[int(i)]["text"] for i in ids]} for bundle, ids in enumerate(groups)]
