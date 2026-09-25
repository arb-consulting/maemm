"""Fetch an HF repo into the volume's HF cache at a PINNED revision.

    modal run features/fetch_hf.py --repo ANONYMOUS/maemm-27b-rl-last16-lr5e-7 \
        --revision <revision>

The fetch step `common.snapshot()` expects: exactly one pinned snapshot per repo.

`common.snapshot()` asserts EXACTLY ONE snapshot directory per repo in /vol/hf/hub --
two revisions in the cache means the weights a run used are ambiguous after the fact.
So this refuses to add a second one unless --force is passed, and prints what is
already there instead.
"""
from __future__ import annotations

import modal

VOL = "/vol"
APP = "maemm-fetch-hf"

vol = modal.Volume.from_name("maemm", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App(APP)


@app.function(image=image, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600, cpu=8)
def fetch(repo: str, revision: str = "", force: bool = False, dry_run: bool = False) -> dict:
    import os
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download

    hub = Path(f"{VOL}/hf/hub")
    cache_dir = hub / ("models--" + repo.replace("/", "--"))
    existing = sorted(p.name for p in (cache_dir / "snapshots").glob("*")) \
        if (cache_dir / "snapshots").exists() else []
    if existing and not force:
        return {"repo": repo, "status": "already present", "snapshots": existing,
                "note": "pass --force only to ADD a revision; common.snapshot() then "
                        "fails until the stale one is deleted"}
    if dry_run:
        return {"repo": repo, "status": "dry run", "would_fetch": revision or "main"}

    t0 = time.time()
    path = snapshot_download(repo_id=repo, revision=revision or None,
                             cache_dir=str(hub), max_workers=8)
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    vol.commit()
    snaps = sorted(p.name for p in (cache_dir / "snapshots").glob("*"))
    return {
        "repo": repo,
        "status": "fetched",
        "revision_requested": revision or "main",
        "path": path,
        "resolved_sha": os.path.basename(os.path.realpath(path)),
        "bytes": total,
        "gb": round(total / 1e9, 2),
        "wall_s": round(time.time() - t0, 1),
        "snapshots_now": snaps,
        "single_snapshot": len(snaps) == 1,
    }


@app.local_entrypoint()
def main(repo: str, revision: str = "", force: bool = False, dry_run: bool = False):
    import json

    out = fetch.remote(repo=repo, revision=revision, force=force, dry_run=dry_run)
    print(json.dumps(out, indent=1))
    if out.get("status") == "fetched" and not out.get("single_snapshot"):
        print("\nWARNING: more than one snapshot is now cached for this repo. "
              "common.snapshot() will refuse it until the stale revision is deleted.")
