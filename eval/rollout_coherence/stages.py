"""The stages and their dependency graph (a leaf module, so the launcher imports it without __main__).
`all` runs every stage; `report` raises on a missing artifact rather than dropping a section."""
STAGES = ["prepare", "capture", "frontier_corpus", "frontier_context", "frontier_context_pairs",
          "frontier_context_judge", "frontier_context_fluency", "report"]
DEPENDS = {
    "prepare": [],
    "capture": ["prepare"],
    # the build refuses a corpus that overlaps the documents `prepare` drew
    "frontier_corpus": ["prepare"],
    # the targets are re-captured from the sources `capture` selected; retrieval searches the corpus
    "frontier_context": ["capture", "frontier_corpus"],
    "frontier_context_pairs": ["frontier_context"],
    "frontier_context_judge": ["frontier_context_pairs"],
    "frontier_context_fluency": ["frontier_context_pairs"],
    "report": ["frontier_context_judge", "frontier_context_fluency"],
}
# The stages that load a model: the base (or a generator) for a read or generation, the scorer for a
# likelihood.
GPU_STAGES = {"capture", "frontier_context", "frontier_context_fluency"}


def levels():
    """Dependency levels over STAGES: each level's stages depend only on earlier levels' and may run
    together. Raises on a cycle or a missing DEPENDS entry."""
    done, out = set(), []
    while len(done) < len(STAGES):
        lvl = [s for s in STAGES if s not in done and all(d in done for d in DEPENDS[s])]
        if not lvl:
            raise RuntimeError(f"stage graph has no runnable level; unresolved: {sorted(set(STAGES) - done)}")
        out.append(lvl)
        done |= set(lvl)
    return out
