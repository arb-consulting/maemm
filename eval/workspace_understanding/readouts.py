"""One loader for every reader's saved readouts, so the judge, the word rule and the report read the same
strings: a greedy rollout and eight samples per free-text reader, a prose summary per lens pool."""

from eval.workspace_understanding import config as C
from eval.workspace_understanding import shards as S
from eval.workspace_understanding.retrieval import READOUTS_REL as RETRIEVAL_REL
from eval.workspace_understanding.untrained_base import READOUTS_REL as UNTRAINED_BASE_REL

# The lens pools, by the condition that judges each (methodology §3.2): the layer-42 top 10, and the union
# over layers 36, 38, ..., 50 ordered by each token's best list position.
LENS_POOLS = {
    "jlens_L42_summary": "L42",
    "jlens_band8_summary": C.LENS_BAND,
}


def _index(doc, *path):
    """{item id: record} over a document's item list at `path` (default "items"); empty when absent."""
    node = doc
    for k in path or ("items",):
        if not isinstance(node, dict):
            return {}
        node = node.get(k)
    return {r["i"]: r for r in (node or [])}


def _view(index, text_key="text", greedy_key=None, samples_key="samples"):
    """{i: {"greedy": str|None, "samples": [str]}}. `greedy` is None for a reader with no greedy rollout
    (the deterministic corpus search), as against "" for a reader that generated nothing."""
    out = {}
    for i, r in index.items():
        g = r.get("greedy") or {}
        out[i] = {
            "greedy": g.get(greedy_key or text_key),
            "samples": [s[text_key] for s in (r.get(samples_key) or [])],
        }
    return out


def free_text(run):
    """{condition: {item id: {"greedy", "samples"}}} for every free-text reader present. The verbalizer is
    read through its judge views (`text`, and `text_trunc` for `nla64`, the first 64 generated ids)."""
    out = {}
    maemm = _index(run.read_json("rollouts/maemm.json") if run.exists("rollouts/maemm.json") else None)
    if maemm:
        out["maemm"] = _view(maemm)
    patch = run.read_json("rollouts/patchscope.json") if run.exists("rollouts/patchscope.json") else None
    for arm in C.PATCH_ARMS:
        idx = _index(patch, "arms", arm, "items")
        if idx:
            out[arm] = _view(idx)
    nla = _index(S.load_nla(run))
    if nla:
        out["nla"] = _view(nla)
        out["nla64"] = _view(nla, text_key="text_trunc")
    for cond, rel in (("retrieval", RETRIEVAL_REL), ("untrained_base", UNTRAINED_BASE_REL)):
        idx = _index(run.read_json(rel) if run.exists(rel) else None)
        if idx:
            out[cond] = _view(idx)
    ctl = S.load_controls(run)
    for kind in C.POSITION_CONTROLS:
        idx = {i: r[kind] for i, r in _index(ctl).items() if kind in r}
        if idx:
            out[f"nla_{kind}"] = _view(idx)
    mctl = run.read_json("rollouts/maemm_control.json") if run.exists("rollouts/maemm_control.json") else None
    for kind in C.POSITION_CONTROLS:
        idx = _index(mctl, "kinds", kind, "items")
        if idx:
            out[f"maemm_{kind}"] = _view(idx)
    return out


def lens_summaries(run):
    """{pool: {item id: prose}} over the summaries that came back ok; a missing one leaves its cell
    unavailable rather than judged against an empty string."""
    doc = run.read_json("judges/summaries.json") if run.exists("judges/summaries.json") else {}
    summ = doc.get("summaries") or {}
    out = {}
    for which in LENS_POOLS.values():
        out[which] = {
            int(i): rec["summary"] for i, rec in (summ.get(which) or {}).items() if rec.get("status") == "ok"
        }
    return out


def judged_samples(free, summaries):
    """{judged condition: {item id: [texts]}}: a free-text reader's eight samples, a lens pool's summary."""
    out = {}
    for reader in C.FREE_TEXT_CONDITIONS:
        v = free.get(reader) or {}
        out[f"{reader}_n8"] = {i: list(r["samples"]) for i, r in v.items()}
    for cond, which in LENS_POOLS.items():
        out[cond] = {i: [text] for i, text in (summaries.get(which) or {}).items()}
    return {c: out[c] for c in C.JUDGED_CONDITIONS if c in out}
