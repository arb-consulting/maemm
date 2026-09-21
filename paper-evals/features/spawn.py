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

# Argument names and defaults MIRROR `precompute/modal_app.py:main` exactly -- a spawned run is
# the run `modal run` would have produced, and a parameter that exists on one path and not the
# other is a knob that gets set by accident (D7). `precompute/unit_smoke.check_spawn_mirrors_main`
# parses modal_app.py and asserts the two key sets are equal, so drift fails a CPU check rather
# than silently taking a default on an H200.
DEFAULTS = {
    "base": "", "maemm": "", "sae": "", "heldout": "", "set": "", "force": False, "root": "/vol",
    "tokens": 0, "batch": 0, "allow_short": False, "n": 0, "rows": "", "max_new": 0,
    "gen_rows": 0, "dirs_from": "", "import_run1": False, "rescore_texts": "",
    "score_name": "", "run_tag": "", "no_sae": False, "no_marker_check": False, "max_num_seqs": 0,
    "gpu_mem": 0.0, "throughput": "", "stock_hook": False, "eager": False,
    "engine": "hf", "ps_layers": "", "no_ps_floor": False, "ps_tag": "",
    "rollouts_dir": "", "corpus_name": "", "subset": "", "amp": "",
    "ps_prompt": "", "ps_rule": "", "ps_alpha": 0.0,
    "corpus": "",
    "mu": "", "re_derive": "",
    "feature_split": "", "maxact_windows": "", "include": "",
    "dry_run": False,
}
# Handled by spawn itself rather than passed through: --set is folded into --heldout here the way
# modal_app.main folds it, and --dry-run has no meaning for a spawn (there is no container to not
# start -- do not spawn it).
# `corpus` too: it is a KEY that must be resolved to a directory name before it is sent, and that
# resolution lives in modal_app.main, which this file bypasses by design. Forwarding the key
# unresolved left `corpus_name` empty and every spawned scan silently read the DEFAULT 16M corpus
# while the operator believed they had scanned the training-range one -- precisely the
# "memorised vs unseen text" confusion `corpus_key_name` exists to prevent, on the launch path
# recommended for long runs (H6).
_LOCAL_ONLY = ("set", "dry_run", "corpus")


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

    args = {k: getattr(a, k) for k in DEFAULTS if k not in _LOCAL_ONLY}
    args["heldout"] = a.heldout or a.set
    if a.corpus:
        # paper-evals/ is this file's parent; spawn is run as `python -m features.spawn` from it,
        # but be explicit so a direct invocation resolves the same way.
        from pathlib import Path as _P

        sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
        import precompute.common as C

        assert not a.corpus_name, (
            f"pass --corpus {a.corpus!r} OR --corpus-name {a.corpus_name!r}, not both: the key "
            f"resolves to the directory name and two sources for one value can only disagree"
        )
        args["corpus_name"] = C.corpus_key_name(C.load_config(), a.corpus)
        print(f"[spawn] corpus {a.corpus} -> dir {args['corpus_name'] or 'corpus'}")
    args["root"] = args["root"].rstrip("/") or "/vol"
    # The same D6 guard modal_app.main applies: a product that WRITES a set is never given the
    # live default. This path bypasses that entrypoint entirely, which is exactly how the hazard
    # got here in the first place.
    assert args["heldout"] or a.product not in ("targets", "draw_sae2m"), (
        f"product {a.product!r} WRITES a held-out set, so it needs an explicit --set <name> (D6)"
    )
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
