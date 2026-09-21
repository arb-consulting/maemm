"""Walk the volume and write INVENTORY.md — what data exists, right now.

    python -m features.inventory --out INVENTORY.md [--put]

`README.md` on the volume describes the LAYOUT and the CONVENTIONS: what the directories
mean, which numbers move under which convention, what will bite you. Those change
slowly and are worth writing by hand.

The INVENTORY is the opposite: which sets, scans, rollouts, scores and SAE products
actually exist. That changes hourly — between 2026-09-20 and 2026-09-21 the target sets
went from 3 to 15 — so a hand-written list is wrong before anyone reads it. Every
hand-maintained index in this project has gone stale the same way (`paper/<section>/`
statuses still said RUNNING for products that had landed).

So this generates it. Run it after a batch of jobs lands and `--put` it back to the
volume root beside README.md.

Read-only: it lists and reads small json, and writes nothing except the file it emits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict

VOL = "maemm"
BASE = "qwen36-27b"
TROJAN_VOL = "maemm-trojan-cache"


def _ls(vol: str, path: str) -> list[str]:
    """One level of a volume directory; [] when it does not exist."""
    from shutil import which

    cmd = (["modal"] if which("modal") else ["uvx", "modal"]) + ["volume", "ls", vol, path]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                            "PYTHONUTF8": "1", "MSYS_NO_PATHCONV": "1"})
    out = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith(path.rstrip("/") + "/"):
            out.append(line.split("/")[-1])
    return sorted(out)


def _leaf(vol: str, path: str) -> list[str]:
    """Filenames directly under `path` (same call; kept separate for readability)."""
    return _ls(vol, path)


def _readme_notes(vol: str, path: str, limit: int = 6) -> list[str]:
    """The `## Notes` bullets of a product README, which is where the script that
    filled a directory recorded HOW it drew what it drew and WHY.

    Pulled rather than restated: a set's own README is written by its producer at the
    moment of production, so it cannot drift from the artifact the way a hand-kept
    index does.
    """
    import os
    import tempfile
    from shutil import which

    cmd = (["modal"] if which("modal") else ["uvx", "modal"])
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "R.md")
        subprocess.run(cmd + ["volume", "get", vol, f"{path}/README.md", dst, "--force"],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8",
                            "PYTHONUTF8": "1", "MSYS_NO_PATHCONV": "1"})
        if not os.path.exists(dst):
            return []
        lines = open(dst, encoding="utf-8", errors="replace").read().splitlines()
    out, seen = [], False
    for ln in lines:
        if ln.strip().startswith("## Notes"):
            seen = True
            continue
        if seen and ln.startswith("## "):
            break
        if seen and ln.strip().startswith("- "):
            out.append(ln.strip()[2:])
    return out[:limit]


def build() -> str:
    import time

    L: list[str] = []
    w = L.append
    w("# INVENTORY — what is on the volume, generated")
    w("")
    w(f"Volume `{VOL}`, workspace `maemms`. Generated "
      f"{time.strftime('%Y-%m-%d %H:%M')} local by `python -m features.inventory`.")
    w("")
    w("This file is GENERATED and goes stale the moment a job lands — regenerate it "
      "rather than editing it. `README.md` beside it is the hand-written part: layout, "
      "conventions and the things that bite. Read that first; this is the stock list.")
    w("")

    # --- models -------------------------------------------------------------------
    w("## Models in the HF cache")
    w("")
    w("```")
    for m in _ls(VOL, "hf/hub"):
        if m.startswith("models--"):
            snaps = _ls(VOL, f"hf/hub/{m}/snapshots")
            w(f"{m.replace('models--', '').replace('--', '/'):55s} {' '.join(s[:8] for s in snaps)}")
    w("```")
    w("")
    w("`common.snapshot()` asserts exactly one snapshot per repo; two makes "
      "\"which weights did that run use\" unanswerable.")
    w("")

    # --- target sets --------------------------------------------------------------
    sets = _ls(VOL, f"base/{BASE}/heldout")
    scans = set(_ls(VOL, f"base/{BASE}/scan"))
    w(f"## Target sets ({len(sets)})")
    w("")
    w("Each entry carries the selection rule from the set's OWN README, written by the "
      "product that drew it. How a set was selected decides what a mean over it means.")
    w("")
    for s in sets:
        files = _leaf(VOL, f"base/{BASE}/heldout/{s}")
        w(f"### `{s}`")
        w("")
        w(f"- files: {', '.join(f'`{f}`' for f in files) or '—'}")
        w(f"- corpus scan: **{'yes' if s in scans else 'NO — no corpus-search baseline'}**")
        for note in _readme_notes(VOL, f"base/{BASE}/heldout/{s}"):
            w(f"- {note}")
        w("")
    w("A set with no scan has **no corpus-search baseline**. `scan` additionally needs "
      "`stats` to have run for that SAE (it reads `sae/<sae>/max_act.f16`) and fails "
      "late without it.")
    w("")

    # --- corpora ------------------------------------------------------------------
    w("## Corpora")
    w("")
    w("```")
    w(f"corpus/                     {', '.join(_leaf(VOL, f'base/{BASE}/corpus')) or '—'}")
    for c in _ls(VOL, f"base/{BASE}/corpora"):
        w(f"corpora/{c:<20s}{', '.join(_leaf(VOL, f'base/{BASE}/corpora/{c}')) or '—'}")
    w("```")
    w("")
    w("Numbers from different corpora are NOT comparable: different documents, "
      "different nested ladders, different window geometry.")
    w("")

    # --- SAE products --------------------------------------------------------------
    w("## SAE products")
    w("")
    w("```")
    for s in _ls(VOL, f"base/{BASE}/sae"):
        w(f"sae/{s:<12s}{', '.join(_leaf(VOL, f'base/{BASE}/sae/{s}')) or '—'}")
    w("```")
    w("")

    # --- per-checkpoint outputs -----------------------------------------------------
    w("## Checkpoints, rollouts and scores")
    w("")
    for m in _ls(VOL, f"maemms/{BASE}"):
        roll = [f for f in _leaf(VOL, f"maemms/{BASE}/{m}/rollouts") if f.endswith(".jsonl")]
        sc = _ls(VOL, f"maemms/{BASE}/{m}/scores")
        w(f"### `{m}`")
        w("")
        w(f"- rollouts ({len(roll)}): " + (", ".join(f"`{r[:-6]}`" for r in roll) or "—"))
        w(f"- scores ({len(sc)}): " + (", ".join(f"`{s}`" for s in sc) or "—"))
        w("")

    # --- baselines ------------------------------------------------------------------
    w("## Baselines")
    w("")
    for name, path in [("corpus search", f"base/{BASE}/scan"),
                       ("GCG / EPO", f"base/{BASE}/gcg"),
                       ("Patchscopes", f"base/{BASE}/patchscopes")]:
        entries = _ls(VOL, path)
        w(f"- **{name}** (`{path}/`): " + (", ".join(f"`{e}`" for e in entries) or "none"))
    w("")
    w("NOTE on \"do we beat the corpus\": the scans above are COSINE baselines. The "
      "non-cosine comparisons — native activation (`top1_act`), autointerp balanced "
      "accuracy, trojan verbatim@n — are separate products and are listed only where "
      "they appear above.")
    w("")

    # --- shared ---------------------------------------------------------------------
    w("## `shared/` — the agreed artifacts")
    w("")
    w("```")
    for d in _ls(VOL, "shared"):
        w(f"shared/{d:<18s}{', '.join(_leaf(VOL, f'shared/{d}')) or '—'}")
    w("```")
    w("")

    # --- other volumes --------------------------------------------------------------
    w("## Other volumes")
    w("")
    tro = _ls(TROJAN_VOL, "trojan")
    w(f"**`{TROJAN_VOL}`** holds the §3.6 rank-1 trojan work — it is NOT on this "
      f"volume. Adapters and results:")
    w("")
    w("```")
    for t in tro:
        w(f"trojan/{t}")
    w("```")
    w("")
    w("`maemm-portable-eval-hf-cache` is a read-only HF cache the trojan app borrows "
      "the base model from.")
    w("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="INVENTORY.md")
    ap.add_argument("--put", action="store_true", help="upload to the volume root")
    a = ap.parse_args()
    text = build()
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nwrote {a.out} ({len(text)} bytes)")
    if a.put:
        from shutil import which

        cmd = (["modal"] if which("modal") else ["uvx", "modal"])
        subprocess.run(cmd + ["volume", "put", VOL, a.out, "/INVENTORY.md", "--force"],
                       check=False)
        print("uploaded to the volume root")


if __name__ == "__main__":
    main()
