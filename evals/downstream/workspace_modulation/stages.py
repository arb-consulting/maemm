"""The stage list, the dependency graph and the shard/arm naming of stage records.

Importable without the CLI or the config (which binds the judge profile on import), so the launcher can read
the stage list before `--judge-profile` is parsed."""

STAGES = [
    "prepare",
    "corpus",
    "capture",
    "cells_mean",
    "rollouts",
    "rollouts_merge",
    "patchscope",
    "lens",
    "nla",
    "nla_merge",
    "retrieval",
    "retrieval_merge",
    "summarise",
    "judge",
    "report",
]
ALL_STAGES = STAGES
DEPENDS = {
    "prepare": [],
    "corpus": [],
    "capture": ["prepare"],
    "cells_mean": ["capture"],
    "rollouts": ["capture", "cells_mean"],
    "rollouts_merge": ["rollouts"],
    "patchscope": ["capture", "cells_mean"],
    "lens": ["capture", "cells_mean"],
    "nla": ["capture", "cells_mean"],
    "nla_merge": ["nla"],
    "retrieval": ["corpus", "capture", "cells_mean"],
    "retrieval_merge": ["retrieval"],
    "summarise": ["lens"],
    "judge": ["rollouts_merge", "nla_merge", "retrieval_merge", "summarise", "patchscope"],
    "report": ["judge"],
}
# Stages dealt out over containers (`--shard k --n-shards n`) and the merge that reassembles each:
# `rollouts` and `nla` by item, `retrieval` by corpus window.
SHARDED = {"rollouts": "rollouts_merge", "nla": "nla_merge", "retrieval": "retrieval_merge"}
# `rollouts` is also dealt out by arm, so the arm list is part of an invocation's record name.
ARM_SHARDED = {"rollouts"}


def arm_suffix(stage, arms):
    """`.<arm>+<arm>` for an arm-sharded stage run on a subset of the arms, "" otherwise."""
    if stage not in ARM_SHARDED or not arms:
        return ""
    from evals.downstream.workspace_modulation import config as C

    names = [a for a in (arms.split(",") if isinstance(arms, str) else arms) if a]
    return "" if not names or names == list(C.ARM_ORDER) else "." + "+".join(names)


def is_stage_record(name):
    """True when `stages/<name>.json` belongs to one of this package's stages (suffixes follow a dot)."""
    return name.split(".", 1)[0] in STAGES


def levels():
    """Dependency levels over STAGES: each level's stages depend only on earlier levels'."""
    done, out = set(), []
    while len(done) < len(STAGES):
        lvl = [s for s in STAGES if s not in done and all(d in done for d in DEPENDS[s])]
        if not lvl:
            raise RuntimeError(f"stage graph has no runnable level; unresolved: {sorted(set(STAGES) - done)}")
        out.append(lvl)
        done |= set(lvl)
    return out


def shard_of(args):
    """(k, n) from --shard/--n-shards; shard 0 of 1 by default."""
    n = int(getattr(args, "n_shards", 1) or 1)
    k = int(getattr(args, "shard", 0) or 0)
    if n < 1 or not 0 <= k < n:
        raise RuntimeError(f"shard {k} is not a shard of {n}")
    return k, n


def stage_record(stage, k, n):
    """The stage-record name of one invocation of a sharded stage."""
    return stage if n <= 1 else f"{stage}.shard{k}of{n}"


def provenance_stage(stage, k, n, arms=None):
    """The provenance file of one invocation: named by stage, arm subset and shard, so concurrent containers
    never write the same file."""
    name = stage + arm_suffix(stage, arms)
    return name if stage not in SHARDED or n <= 1 else f"{name}_shard{k}of{n}"
