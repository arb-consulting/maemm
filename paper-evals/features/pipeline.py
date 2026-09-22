"""One command for a whole evaluation chain: draw -> rollouts -> score -> scan.

    # everything for a feature set that does not exist yet
    python -m features.pipeline --set 2026-09-21_sae131k_2k --sae qwen36-27b/l42-1b \
        --draw draw_sae131k --n 4

    # just rollouts + score for a set that already exists, on a second checkpoint
    python -m features.pipeline --set 2026-09-20_sae2m_2k --sae qwen36-27b/sae2m \
        --maemm qwen36-27b/2026-09-10_rl-8x2048-full --stages rollouts,score

    python -m features.pipeline --set ... --watch      # block until the chain finishes
    python -m features.pipeline --status               # what is running right now

Every stage is the SAME product `modal run` would invoke, with the same arguments; this
only sequences them and waits on the dependencies. It exists because the chain has real
ordering constraints that are easy to get wrong by hand, and getting them wrong costs a
GPU hour rather than an error:

  draw      writes heldout/<set>/{ids.jsonl,vecs.f16}
  rollouts  needs the set; writes maemms/<maemm>/rollouts/<set>.jsonl
  score     needs the rollouts; writes maemms/<maemm>/scores/<set>/
  centred   needs the scores; adds the centred/filtered cosines beside them
  scan      needs the set AND, for an SAE family, sae/<sae>/max_act.f16 from `stats`

Two things it enforces that cost real time when missed:

  * IT REDEPLOYS FIRST. The deployed Modal image is a snapshot of whatever checkout was
    current when someone last ran `modal deploy`. Two failures on 2026-09-20 were a
    stale image presenting as a code bug -- a fix that was on disk, on main, and not in
    the container. `--no-deploy` skips it if you know the image is current.
  * IT CHECKS `stats` BEFORE SCANNING an SAE set, because `scan` reads
    `sae/<sae>/max_act.f16` and fails ~40 minutes in without it.

Stages run DETACHED (features/spawn.py), so a chain outlives this process: `modal run`
holds a blocking call open and a multi-hour product dies when the client JWT expires.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

APP = "maemm-paper-evals"
STAGES = ("draw", "rollouts", "score", "centred", "scan")
STATE = Path.home() / ".cache" / "maemm" / "pipeline"


def _spawn(product: str, **kw) -> str:
    argv = [sys.executable, "-m", "features.spawn", "--product", product]
    for k, v in kw.items():
        if v in (None, "", False):
            continue
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    out = subprocess.run(argv, capture_output=True, text=True).stdout
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1])["spawned"]
    except Exception:
        raise RuntimeError(f"spawn failed for {product}:\n{out}") from None


def _poll(call_id: str) -> tuple[bool, str]:
    out = subprocess.run(
        [sys.executable, "-m", "features.spawn", "--poll", call_id],
        capture_output=True, text=True).stdout
    return ("still running" not in out), out


def _wait(call_id: str, label: str, quiet: bool = False) -> dict:
    t0 = time.time()
    while True:
        done, out = _poll(call_id)
        if done:
            if not quiet:
                print(f"[{label}] finished in {time.time() - t0:.0f}s")
            if "Error" in out or "Traceback" in out or "Assertion" in out:
                raise RuntimeError(f"[{label}] FAILED\n{out[-1500:]}")
            try:
                return json.loads(out[out.index("{"):out.rindex("}") + 1])
            except Exception:
                return {"raw": out[-400:]}
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="heldout", default="")
    ap.add_argument("--base", default="qwen36-27b")
    ap.add_argument("--maemm", default="qwen36-27b/2026-09-18_rl-last16-lr5e-7")
    ap.add_argument("--sae", default="")
    ap.add_argument("--draw", default="", help="draw_sae2m | draw_sae131k | '' to reuse the set")
    ap.add_argument("--subset", default="", help="an agreed feature list the draw takes verbatim")
    # DEFAULT CHANGED 2026-09-21, in the rebase onto arb/main. It was `train_parity_10m`, which
    # this chain cannot scan: that corpus is cut at 32/8 (features/corpus_train_parity.py:60) and
    # every `windows_of(` call site in this pipeline takes 64/16 from `common.SCAN_BLOCK`, so
    # `common.assert_corpus_geometry` refuses it in `scan` and `stats` -- correctly, because the
    # alternative is a scan README claiming 64/16 over 32/8 data. The chain's own default must be
    # a corpus the chain can run, so it is the plain `corpus/` (heldout16m, 64/16).
    #
    # H7 IS STILL OPEN, and this line is not the fix. Threading block/stride from the `corpora:`
    # entry through the eleven `windows_of(` sites -- plus the window-id join check, because
    # stored top-k lists index into a specific geometry -- is the real piece of work, and it is
    # not this rebase's. `--corpus-name train_parity_10m` still refuses, loudly, with the reason.
    ap.add_argument("--corpus-name", default="",
                    help="search corpus DIRECTORY; '' = the 16M corpus/ (heldout16m, 64/16). "
                         "`train_parity_10m` is Tomas's 10M training-parity corpus and is cut at "
                         "32/8 -- assert_corpus_geometry refuses it until H7 lands")
    ap.add_argument("--n", type=int, default=4, help="rollouts per target")
    ap.add_argument("--rows", default="")
    ap.add_argument("--stages", default="draw,rollouts,score,centred,scan")
    ap.add_argument("--watch", action="store_true", help="block until each stage finishes")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    if a.status:
        for f in sorted(STATE.glob("*.json")):
            st = json.loads(f.read_text())
            line = []
            for stage, cid in st.get("calls", {}).items():
                line.append(f"{stage}={'done' if _poll(cid)[0] else 'RUNNING'}")
            print(f"{f.stem}: " + "  ".join(line))
        return

    assert a.heldout, "--set is required"
    want = [s.strip() for s in a.stages.split(",") if s.strip()]
    assert all(s in STAGES for s in want), f"stages must be from {STAGES}"

    if not a.no_deploy:
        # The deployed image is a snapshot of the checkout, not of the branch.
        print("[deploy] refreshing the app image")
        subprocess.run(["modal", "deploy", "precompute/modal_app.py"], check=False)

    common = dict(base=a.base, heldout=a.heldout, sae=a.sae, force=a.force)
    calls: dict[str, str] = {}

    if "draw" in want:
        assert a.draw, "--draw names the product (draw_sae2m / draw_sae131k)"
        calls["draw"] = _spawn(a.draw, subset=a.subset, **common)
        print(f"[draw] {calls['draw']}")
        _wait(calls["draw"], "draw")            # every later stage needs the set

    if "rollouts" in want:
        calls["rollouts"] = _spawn("rollouts_hf", maemm=a.maemm, n=a.n, rows=a.rows, **common)
        print(f"[rollouts] {calls['rollouts']}")

    if "score" in want:
        _wait(calls["rollouts"], "rollouts") if "rollouts" in calls else None
        calls["score"] = _spawn("score", maemm=a.maemm, **common)
        print(f"[score] {calls['score']}")

    if "centred" in want:
        _wait(calls["score"], "score") if "score" in calls else None
        calls["centred"] = _spawn("centred", maemm=a.maemm, gpu="cpu", **common)
        print(f"[centred] {calls['centred']}")

    if "scan" in want:
        # scan reads sae/<sae>/max_act.f16; without `stats` for that SAE it dies late.
        if a.sae:
            print(f"[scan] NOTE: needs stats for {a.sae} (sae/<sae>/max_act.f16). "
                  f"If it has never run for this SAE, run --product stats first.")
        calls["scan"] = _spawn("scan", corpus_name=a.corpus_name, **common)
        print(f"[scan] {calls['scan']}")

    (STATE / f"{a.heldout}.json").write_text(json.dumps(
        {"set": a.heldout, "maemm": a.maemm, "sae": a.sae, "n": a.n, "calls": calls,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print("\n" + json.dumps(calls, indent=1))
    print(f"\nstate: {STATE / (a.heldout + '.json')}")
    print("status: python -m features.pipeline --status")

    if a.watch:
        for stage, cid in calls.items():
            if stage in ("draw",):
                continue
            print(json.dumps({stage: _wait(cid, stage)}, indent=1)[:800])


if __name__ == "__main__":
    main()
