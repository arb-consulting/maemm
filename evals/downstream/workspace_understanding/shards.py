"""Item sharding for `nla` and `nla_control` (`--shard k/n`, 0-based) and the loaders that merge shards.
A sharded stage is done when, for some n, every shard 0..n-1 has a record under the current key."""

import glob
import os
import re

from evals.downstream.common.runs import mark_stage, stage_done

# --- sharding ---------------------------------------------------------------------------------------------


def parse_shard(spec):
    """"k/n" (0-based) -> (k, n); None -> (0, 1)."""
    if spec is None or str(spec).strip() in ("", "0/1"):
        return 0, 1
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(spec))
    if not m:
        raise ValueError(f"--shard must be k/n, got {spec!r}")
    k, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard {spec!r}: need 0 <= k < n")
    return k, n


def shard_slice(kept, k, n):
    """Shard k of n of an item list, in order: contiguous blocks so a shard's items print consecutively."""
    return kept[k * len(kept) // n : (k + 1) * len(kept) // n]


def shard_record_rel(stage, k, n):
    return f"stages/{stage}.json" if n == 1 else f"stages/{stage}_shard{k}of{n}.json"


def nla_shard_rel(k, n):
    return f"rollouts/nla/shard_{k}_of_{n}.json"


def control_shard_rel(k, n):
    return f"rollouts/nla_control/shard_{k}_of_{n}.json"


def control_vectors_rel(k, n):
    return f"activations/controls/shard_{k}_of_{n}.npz"


def shard_done(run, stage, chash, k, n):
    return stage_done(run, shard_record_rel(stage, k, n)[len("stages/") : -len(".json")], chash)


def mark_shard(run, stage, chash, k, n, extra=None, started=None):
    mark_stage(run, shard_record_rel(stage, k, n)[len("stages/") : -len(".json")], chash, dict(extra or {}, shard=[k, n]),
               started=started)


def _sharded(run, stage, done):
    """True when `done` holds for the unsharded record, or for every shard 0..n-1 of one split n."""
    if done(stage):
        return True
    by_n = {}
    for p in glob.glob(run.file(f"stages/{stage}_shard*of*.json")):
        m = re.search(r"_shard(\d+)of(\d+)\.json$", p)
        if not m:
            continue
        k, n = int(m.group(1)), int(m.group(2))
        if done(f"{stage}_shard{k}of{n}"):
            by_n.setdefault(n, set()).add(k)
    return any(ks == set(range(n)) for n, ks in by_n.items())


def sharded_stage_done(run, stage, chash):
    """`_sharded` with records matching `chash`."""
    return _sharded(run, stage, lambda name: stage_done(run, name, chash))


def stage_ever_completed(run, stage):
    """True when `stage` (or every shard of one split) completed at least once, under any config."""
    return _sharded(run, stage,
                    lambda name: run.exists(f"stages/{name}.json")
                    and run.read_json(f"stages/{name}.json").get("completed") is True)


# --- loaders ----------------------------------------------------------------------------------------------


def _merge(run, pattern, label, kept_ids, keep_config_keys):
    """Every shard file matching `pattern` merged into one {"config", "shards", "items"} document; raises on
    an item in two shards or, given `kept_ids`, on an item in none."""
    paths = sorted(glob.glob(run.file(pattern)))
    if not paths:
        return None
    items, seen, shards, config = [], {}, [], None
    for p in paths:
        doc = run.read_json(os.path.relpath(p, run.path))
        config = config or doc.get("config")
        shards.append(
            {
                "file": os.path.basename(p),
                **{k: v for k, v in (doc.get("config") or {}).items() if k in keep_config_keys},
            }
        )
        for r in doc["items"]:
            if r["i"] in seen:
                raise RuntimeError(f"{label} item {r['i']} appears in {seen[r['i']]} and {os.path.basename(p)}")
            seen[r["i"]] = os.path.basename(p)
            items.append(r)
    if kept_ids is not None:
        missing = sorted(set(kept_ids) - set(seen))
        if missing:
            raise RuntimeError(
                f"{label} shards cover {len(seen)} items; missing {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
    return {"config": config, "shards": shards, "items": sorted(items, key=lambda r: r["i"])}


def load_nla(run, kept_ids=None):
    """Every rollouts/nla/*.json merged: {"config": <first shard's config>, "shards": [...], "items": [...]}."""
    return _merge(run, "rollouts/nla/*.json", "nla", kept_ids, ("shard", "injection_check", "seed"))


def load_controls(run, kept_ids=None):
    """Every rollouts/nla_control/*.json merged (same contract as load_nla)."""
    return _merge(run, "rollouts/nla_control/*.json", "nla_control", kept_ids, ("shard", "injection_check", "seeds"))
