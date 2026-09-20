"""Per-section folders on the volume: `paper/<section>/` for each paper experiment.

    python -m features.paper_index --out <dir>        # build locally
    modal volume put maemm <dir>/paper /paper         # then upload

The pipeline files products by TYPE -- scan/, gcg/, maemms/<m>/rollouts/ -- which is
right for producing them and useless for finding them. Someone asked "where is 3.2"
has to know that its corpus baseline is in `scan/`, its control is a MAEMM entry, and
its targets are rows 0-1023 of a set whose name says nothing about the section.

So each section gets a folder that INDEXES its artifacts rather than copying them:
products stay in one canonical place, and `paper/<section>/` says what belongs to the
section, where each piece lives, and what state it is in. `results/` is the one real
subdirectory -- it holds what the section itself produces (tables, figures, stats).

Statuses are `ready` (on the volume now), `running`, `todo`, `discarded` (superseded,
kept so nobody re-derives it by accident) and `external` (lives outside this pipeline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHECKPOINT = "qwen36-27b/2026-09-18_rl-last16-lr5e-7"
OLD_PRIMARY = "qwen36-27b/2026-09-10_rl-8x2048-full"
SET = "2026-09-16_v1"
SAE = "sae2m"

SECTIONS: dict[str, dict] = {
    "3.2-inversion-fidelity": {
        "title": "Inversion fidelity on held-out directions (Table 1)",
        "targets": {
            "path": f"base/qwen36-27b/heldout/{SET}",
            "note": "realact rows 0-511, random 512-1023, sae 1024-1535 (131k, discarded)",
            "status": "ready",
        },
        "artifacts": [
            {"what": "MAEMM rollouts, realact + random x64",
             "path": f"maemms/{CHECKPOINT}/rollouts", "status": "running"},
            {"what": "scores for the above",
             "path": f"maemms/{CHECKPOINT}/scores/{SET}", "status": "todo"},
            {"what": "old primary, same targets -- the paired comparison",
             "path": f"maemms/{OLD_PRIMARY}/scores/{SET}__vllm", "status": "ready",
             "note": "generated on vLLM; new rollouts are HF, parity ~0.002 at bo64"},
            {"what": "corpus search, nested 1/2/4/8/16M",
             "path": f"base/qwen36-27b/scan/{SET}", "status": "ready"},
            {"what": "untrained-base control, same prompt/marker/injection",
             "path": "maemms/qwen36-27b/2026-09-16_base-control", "status": "ready"},
            {"what": "GCG / EPO arms",
             "path": f"base/qwen36-27b/gcg/{SET}", "status": "ready",
             "note": "realact arms valid; the sae arms were 131k and are discarded"},
            {"what": "corpus + centring mean",
             "path": "base/qwen36-27b/corpus, base/qwen36-27b/stats", "status": "ready",
             "note": "||mu|| 67.9258"},
            {"what": "SAE fidelity column on the 2M SAE",
             "path": f"base/qwen36-27b/sae/{SAE}", "status": "running",
             "note": "corpus stats pass; then the 40k draw, then rollouts"},
        ],
    },
    "3.3-sae-autointerp": {
        "title": "SAE autointerp: generated examples as explainer input",
        "targets": {"path": "(the 40k sae2m draw)", "status": "todo"},
        "artifacts": [
            {"what": "512-feature run on the 131k SAE", "path": "(volume, prior run)",
             "status": "discarded", "note": "$91.92; superseded by the 2M SAE"},
            {"what": "rerun on 2M", "path": f"base/qwen36-27b/sae/{SAE}",
             "status": "todo", "note": "~$92, protocol and vendored Delphi unchanged"},
        ],
    },
    "3.7-hard-to-verbalize": {
        "title": "Eliciting features that are difficult to verbalize",
        "targets": {"path": "(the 40k sae2m draw)", "status": "todo"},
        "artifacts": [
            {"what": "prior run on Qwen3-8B / 65k SAE", "path": "(8B tree)",
             "status": "discarded"},
            {"what": "rerun on the 2M SAE", "path": f"base/qwen36-27b/sae/{SAE}",
             "status": "todo", "note": "unblocked by the SAE weights landing 2026-09-20"},
        ],
    },
    "appendix-C-coherence": {
        "title": "Coherence of rollouts vs natural passages",
        "targets": {"path": "400 layer-42 activations", "status": "ready"},
        "artifacts": [{"what": "judged win/tie/loss, Sonnet 5 + Opus 5",
                       "path": "(volume, prior run)", "status": "ready",
                       "note": "SAE-independent, unaffected by the switch"}],
    },
    "3.4-bsf": {
        "title": "Block-sparse feature subspaces",
        "targets": {"path": "bsf family", "status": "todo"},
        "artifacts": [{"what": "qualitative only, Qwen3-8B", "path": "-",
                       "status": "external",
                       "note": "the family ships no provenance, so no held-out claim "
                               "can be stated for it (heldout_kind: unknown)"}],
    },
    "3.5-workspace": {
        "title": "Decoding workspace content",
        "targets": {"path": "Gurnee et al. public eval sets", "status": "external"},
        "artifacts": [{"what": "102 association + 92 two-hop items", "path": "-",
                       "status": "external",
                       "note": "different code, not on this volume; held out by "
                               "CATEGORY, asserted not measured"}],
    },
    "3.6-lora-backdoors": {
        "title": "Inverting back-doors in rank-one LoRAs",
        "targets": {"path": "32 rank-one adapters", "status": "external"},
        "artifacts": [{"what": "trojan/ package, Celeste's repo", "path": "-",
                       "status": "external",
                       "note": "self-contained and fully seeded; not on this "
                               "checkpoint and not in the shared pipeline"}],
    },
    "3.8-contrastive": {
        "title": "Contrastive direction inversion (AxBench)",
        "targets": {"path": "500 contrastive directions", "status": "external"},
        "artifacts": [{"what": "vs NLA, CAA, corpus retrieval", "path": "-",
                       "status": "external", "note": "different code, held out by CATEGORY"}],
    },
}

BADGE = {"ready": "READY", "running": "RUNNING", "todo": "TODO",
         "discarded": "DISCARDED", "external": "EXTERNAL"}


def readme(key: str, sec: dict) -> str:
    lines = [
        f"# {key} -- {sec['title']}",
        "",
        "Everything this section uses, and where it is on Modal volume `maemm`",
        "(workspace `maemms`). Paths are volume-relative; products are NOT copied here,",
        "they stay in one canonical place. `results/` holds what this section produces.",
        "",
        f"Checkpoint: `{CHECKPOINT}`  ·  SAE: `{SAE}`  ·  target set: `{SET}`",
        "",
        "## Targets",
        "",
        f"`{sec['targets']['path']}`  [{BADGE[sec['targets']['status']]}]",
    ]
    if sec["targets"].get("note"):
        lines += ["", sec["targets"]["note"]]
    lines += ["", "## Artifacts", ""]
    for a in sec["artifacts"]:
        lines.append(f"**{BADGE[a['status']]}**  {a['what']}")
        lines.append(f"    {a['path']}")
        if a.get("note"):
            lines.append(f"    {a['note']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out) / "paper"
    for key, sec in SECTIONS.items():
        d = root / key
        (d / "results").mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(readme(key, sec), encoding="utf-8")
        (d / "index.json").write_text(json.dumps(
            {"section": key, "checkpoint": CHECKPOINT, "sae": SAE, "target_set": SET, **sec},
            indent=1), encoding="utf-8")
        (d / "results" / "README.md").write_text(
            f"# {key} / results\n\nTables, figures and statistics this section "
            f"produces. Inputs are indexed in ../README.md and are not copied here.\n",
            encoding="utf-8")
    counts: dict[str, int] = {}
    for sec in SECTIONS.values():
        for a in sec["artifacts"]:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
    (root / "README.md").write_text(
        "# paper/ -- one folder per experiment\n\n"
        "The pipeline files products by TYPE (scan/, gcg/, maemms/<m>/rollouts/), which "
        "is right for producing them and useless for finding them. Each folder here "
        "indexes one section's artifacts and holds what that section produces.\n\n"
        + "\n".join(f"- `{k}/` -- {v['title']}" for k, v in SECTIONS.items())
        + f"\n\nArtifact status across all sections: {counts}\n", encoding="utf-8")
    print(f"wrote {root}")
    for k in SECTIONS:
        print(f"  paper/{k}/  README.md  index.json  results/")


if __name__ == "__main__":
    main()
