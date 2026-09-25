"""The stage list and dependency graph, kept apart from __main__ so importing them never imports the CLI."""

STAGES = [
    "prepare",
    "corpus",
    "capture",
    "lens",
    "rollouts",
    "retrieval",
    "patchscope",
    "nla",
    "nla_control",
    "untrained_base",
    "maem_control",
    "reread",
    "summarise",
    "judge",
    "report",
]
ALL_STAGES = STAGES
#: The stages that hold the inverter and the clean base at once; every other GPU stage holds one 27B.
TWO_MODEL_STAGES = {"rollouts", "maem_control"}
DEPENDS = {
    "prepare": [],
    "corpus": [],
    "capture": ["prepare"],
    "lens": ["capture"],
    "rollouts": ["capture"],
    "retrieval": ["capture", "corpus"],
    "patchscope": ["capture"],
    "nla": ["capture"],
    "summarise": ["lens"],
    # maem_control reads the control vectors nla_control saved, so both readers see identical inputs
    "nla_control": ["capture"],
    "untrained_base": ["capture"],
    "maem_control": ["nla_control", "rollouts"],
    "reread": ["nla", "patchscope", "rollouts"],
    "judge": ["rollouts", "patchscope", "summarise", "nla", "retrieval"],
    # the position controls and the untrained base are read by the word rule alone, in the report
    "report": ["judge", "reread", "nla_control", "maem_control", "untrained_base"],
}


def levels():
    """Dependency levels over STAGES: each level depends only on earlier ones; raises on a cycle."""
    done, out = set(), []
    while len(done) < len(STAGES):
        lvl = [s for s in STAGES if s not in done and all(d in done for d in DEPENDS[s])]
        if not lvl:
            raise RuntimeError(f"stage graph has no runnable level; unresolved: {sorted(set(STAGES) - done)}")
        out.append(lvl)
        done |= set(lvl)
    return out
