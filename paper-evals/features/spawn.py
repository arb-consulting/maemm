"""Spawn a precompute product DETACHED, so a long run outlives this client.

    modal deploy precompute/modal_app.py
    python -m features.spawn --product stats --base qwen36-27b --sae qwen36-27b/sae2m --force
    python -m features.spawn --poll fc-01XXXX

`modal run` keeps a blocking `.remote()` open for the whole job. On a multi-hour
product the client's JWT expires mid-call, the call raises `AuthError: Jwt is
expired`, and the container dies with it -- losing the run and leaving a
`<product>.tmp-<date>` directory behind. Measured 2026-09-20 on the 2M SAE stats
pass, which reached the GPU and then died with an empty temp dir.

Spawning returns immediately with a FunctionCall id; the container runs to
completion regardless of what this process does. Same pattern as sae2m/README.md.

Argument names and defaults mirror `modal_app.main` exactly, so a spawned run is the
same run `modal run` would have produced -- if that file gains a parameter, add it
here too or it silently takes the default.
"""
from __future__ import annotations

import argparse
import json
import sys

import modal

APP = "maemm-paper-evals"
CPU_PRODUCTS = ("check", "unit", "corpus", "mu_check", "centred")

DEFAULTS = {
    "base": "", "maemm": "", "sae": "", "heldout": "", "force": False, "root": "/vol",
    "tokens": 0, "batch": 0, "allow_short": False, "n": 0, "rows": "", "max_new": 0,
    "gen_rows": 0, "dirs_from": "", "import_run1": False, "rescore_texts": "",
    "score_name": "", "no_sae": False, "no_marker_check": False, "max_num_seqs": 0,
    "gpu_mem": 0.0, "throughput": "", "stock_hook": False, "eager": False,
    "engine": "hf", "ps_layers": "", "no_ps_floor": False, "ps_tag": "",
    "rollouts_dir": "", "corpus_name": "", "subset": "", "amp": "",
    "ps_prompt": "", "ps_rule": "", "ps_alpha": 0.0,
    "mu": "", "re_derive": "",
}


def poll(call_id: str) -> None:
    fc = modal.FunctionCall.from_id(call_id)
    try:
        print(json.dumps(fc.get(timeout=0), indent=1, default=str))
    except TimeoutError:
        print(f"{call_id}: still running")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll", default="", help="a FunctionCall id to check instead of spawning")
    ap.add_argument("--product", default="")
    ap.add_argument("--gpu", default="", help="h100 | h200 | cpu; default from the base")
    for key, val in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true", dest=key)
        elif isinstance(val, int) and not isinstance(val, bool):
            ap.add_argument(flag, type=int, default=val, dest=key)
        elif isinstance(val, float):
            ap.add_argument(flag, type=float, default=val, dest=key)
        else:
            ap.add_argument(flag, default=val, dest=key)
    a = ap.parse_args()

    if a.poll:
        poll(a.poll)
        return
    assert a.product, "--product is required unless --poll is given"

    args = {k: getattr(a, k) for k in DEFAULTS}
    args["root"] = args["root"].rstrip("/") or "/vol"
    # The container has no checkout; modal_app records this in every README.
    args["repo_commit"] = _repo_commit()
    args["argv"] = sys.argv

    fn_name = a.gpu.lower() or ("cpu" if a.product in CPU_PRODUCTS else "gpu_h200")
    if fn_name in ("h100", "h200"):
        fn_name = f"gpu_{fn_name}"
    fn = modal.Function.from_name(APP, fn_name)
    call = fn.spawn(a.product, args)
    print(json.dumps({
        "spawned": call.object_id,
        "app": APP,
        "function": fn_name,
        "product": a.product,
        "poll": f"python -m features.spawn --poll {call.object_id}",
    }, indent=1))


def _repo_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
