#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""Run the WHOLE chain's control flow locally, with the API stubbed, before any launch.

    uv run paper-evals/autointerp/selfcheck.py

Why this exists. Four chain launches died, and not one of them died in the science:

    ap-KMkH5RvJ6SctMwqy6YYDjK   the batch path could resubmit a batch after a container restart
    ap-HB71h7dO6SezJ56W9bV7DA   the latency probe blocked until a batch ENDED instead of giving up
    ap-2jWiGbYlcuv97kS44jKVtR   `"text " (...)` -- a ruff autofix left Python calling a string
    ap-oQHRdUHILPhCSX3PawT5Hq   an image carrying a superseded acceptance threshold
    (and then) KeyError: 'mean_input_tokens' -- `gate()` formatted a key that `project()` omits
                                when the stage is fully cached, which is the branch a RELAUNCH
                                takes, so the cache working is what exposed it

Every one is a formatting, key, or control-flow fault that costs a launch and, worse, is only
reachable after real work has been paid for. `ruff` cannot see them and `ast.parse` cannot either:
calling a string is valid syntax, and a missing dict key is a runtime event. So this script drives
the real `build` -> `run` -> `chain` code paths against a synthetic volume and a stub client, which
takes seconds and costs nothing.

What it exercises, deliberately including the branches that bit:
  * `Claude.project` on an EMPTY job list and on each job kind, asserting both return paths carry
    identical keys and that `gate()` can format every one of them;
  * `gate()` under the threshold, over the threshold WITHOUT `--approved` (must refuse), and over
    it WITH `--approved` (must pass);
  * `run.run()` on both the `sync` and `batch` paths, twice, so the second pass takes the
    fully-cached branch that produced the KeyError;
  * `chain.run()` end to end: docmax wait, build reuse, pilot, acceptance, batch probe, full build,
    primary, rlI-150 and the stats writers.

It is NOT a test of the numbers. The stub answers are canned, so balanced accuracies here are
meaningless; what is checked is that every stage runs, writes its products, and formats its output.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
if str(PAPER_EVALS) not in sys.path:
    sys.path.insert(0, str(PAPER_EVALS))

import precompute.common as C  # noqa: E402
from autointerp import chain as CH  # noqa: E402
from autointerp import run as R  # noqa: E402

# Captured at IMPORT, because `main()` rebinds `R.Claude` to the stub: a body that looked up
# `R.Claude` at call time would find the stub and recurse into itself.
_REAL_CLAUDE = R.Claude

N_FEAT = 6
N_ARMS = ("C16", "C4", "M", "C4M", "C32", "C16M16")
STRATA = 4


class _CountTokens:
    input_tokens = 1000


class _StubMessages:
    def count_tokens(self, **_kw):
        return _CountTokens()


class _StubClient:
    messages = _StubMessages()


class StubClaude:
    """The Anthropic client's surface, answering from canned text. No network, no key.

    It counts calls so the checks can assert that a second pass hits the cache instead of the
    client, which is the property the relaunch path depends on.
    """

    def __init__(self, model, api_key, timeout_s=90.0, max_retries=8, cache_prompt=True,
                 on_commit=None):
        self.model = model
        self.cache_prompt = cache_prompt
        self.on_commit = on_commit
        self.totals = {"calls": 0, "fails": 0, "in": 0, "out": 0,
                       "cache_write": 0, "cache_read": 0, "cost": 0.0}
        self._lock = threading.Lock()  # the real `_account` and `snapshot` take it
        self.sent = 0

    # -- the real class's interface -------------------------------------------------------
    params = _REAL_CLAUDE.params
    snapshot = _REAL_CLAUDE.snapshot
    _account = _REAL_CLAUDE._account

    def _answer(self, user: str, max_tokens: int) -> str:
        if max_tokens >= 500 and "Latent explanation" not in user:
            return "[EXPLANATION]: a stub description of a stub feature."
        n = user.count("\nExample ")
        rng = random.Random(len(user))
        return "[" + ",".join(str(rng.randint(0, 1)) for _ in range(max(1, n))) + "]"

    def complete(self, system, user, max_tokens, fewshot=None):
        self.sent += 1
        usage = {"in": 1000, "out": 20, "cache_write": 0, "cache_read": 0,
                 "cost": 0.002, "batch": False, "stop_reason": "end_turn"}
        self._account(usage)
        return self._answer(user, max_tokens), usage

    def run_batch(self, jobs, label, ledger="", max_wait_s=0.0):
        self.sent += len(jobs)
        out = {}
        for j in jobs:
            usage = {"in": 1000, "out": 20, "cache_write": 0, "cache_read": 0,
                     "cost": 0.001, "batch": True, "stop_reason": "end_turn"}
            self._account(usage)
            out[j["key"]] = {"text": self._answer(j["user"], j["max_tokens"]), "usage": usage}
        return out, {"batch_ids": ["msgbatch_stub"], "chunks": 1, "wall_s": 1.0,
                     "requests": len(jobs), "errors": 0, "abandoned": False, "still_running": []}

    # `project` calls `self._client.messages.count_tokens`, so the stub supplies that too and the
    # REAL projection code runs -- which is the point: this check exists because a projection
    # formatting bug reached production.
    _client = _StubClient()

    def project(self, jobs, batch, sample=12):
        return _REAL_CLAUDE.project(self, jobs, batch, sample)



def synth_build(root: Path, base: str, set_name: str, name: str, feats: list[int]) -> Path:
    """A build directory with the real schema: meta, arm and test rows per feature."""
    d = root / "base" / base / "autointerp" / set_name / name
    d.mkdir(parents=True, exist_ok=True)
    table = []
    for i, f in enumerate(feats):
        meta = {
            "kind": "meta", "feature": f, "row": 1024 + i, "stratum": i % STRATA,
            "density": 1e-5, "corpus_peak": 10.0, "fires_gated": 100, "fire_fraction": 0.5,
            "gate": 1.5846, "n_pos": 20, "n_neg": 20, "pool_c16": 60, "pool_c4": 50,
            "pool_m": 64, "n_mark_neg": 1, "n_top_fallback": 1 if i == 0 else 0,
            "n_pos_bands": 19, "n_neg_nearmiss": 10, "pool_zero": 100, "pool_nearmiss": 100,
            "shown_docs": 30,
            "draw1": {"n_pos": 20, "n_neg": 20, "n_pos_bands": 19, "n_top_fallback": 0,
                      "n_near": 10, "n_near_short": 0, "n_zero": 10, "pool_zero": 100,
                      "pool_nearmiss": 100, "n_mark_neg": 1, "short_bands": []},
            "draw2": {"n_pos": 20, "n_neg": 20, "n_pos_bands": 19, "n_top_fallback": 0,
                      "n_near": 10, "n_near_short": 0, "n_zero": 10, "pool_zero": 100,
                      "pool_nearmiss": 100, "n_mark_neg": 1, "short_bands": []},
        }
        rows = [meta]
        for a in N_ARMS:
            rows.append({"kind": "arm", "arm": a, "n": 16,
                         "block": f"Example 1:  stub text for {a}\nActivations: (\"stub\" : 9)",
                         "examples": []})
        for tag in ("test", "test2"):
            for k in range(40):
                rows.append({
                    "kind": tag, "i": k, "label": 1 if k < 20 else 0,
                    "src": "corpus" if k < 20 else ("nearmiss-qband" if k < 30 else "random"),
                    "band": "q1" if k < 20 else "-", "window": k, "doc": 1000 * i + k,
                    "start": 0, "max_act": 5.0 if k < 20 else 0.0,
                    "text": f"stub window {k} of feature {f}",
                    "text_fuzz": f"stub <<window>> {k} of feature {f}",
                    "n_marked": 1, "n_tok": 8,
                })
        C.write_jsonl(d / f"{f}.jsonl", rows)
        table.append({k: v for k, v in meta.items() if k != "kind"})
    with open(d / "features.json", "w") as fh:
        json.dump({"features": table}, fh)
    with open(d / "build.json", "w") as fh:
        json.dump({
            "base": base, "set": set_name, "maemm": "stub/maemm", "engine": "vllm",
            "sae": "stub/sae", "gate": 1.5846, "n_features": len(feats), "feat_seed": 1234,
            "shuffle_seed": 20260916, "n_examples": 16, "corpus_prefix_m": 4, "n_pos": 20,
            "n_neg": 20, "n_neg_nearmiss": 10, "nearmiss_source": "qband",
            "gate_consistent_positives": True, "allow_top_fallback": True,
            "arms": {a: [] for a in N_ARMS}, "mean_marked_fraction": 0.2,
            "token_join_mismatches": "0/1", "flags": [],
            "n_short_draw1": 0, "n_short_draw2": 0, "n_empty_draw2": 0, "n_no_pos_draw1": 0,
            "no_pos_draw1_by_stratum": {"0": 0}, "n_short_c4": 0, "n_short_neg": 0,
            "min_pos_draw1": 20, "min_pos_draw2": 20,
            "n_top_fallback_features": 1, "n_top_fallback_positives": 1,
        }, fh)
    return d


def check_projection_keys(cfg):
    """Both `project` return paths must carry the keys `gate()` formats -- including the EMPTY one.

    This is the exact fault that killed ap-oLxIvCsa6PEdJf2NdUnsGo: a relaunch found every call
    cached, `project([])` took its early return, and `gate()` formatted a key it did not have.
    """
    cl = StubClaude(cfg["autointerp"]["scorer_model"], "stub")
    job = {"key": "k", "kind": "explain", "system": "s", "user": "u", "max_tokens": 600}
    full = cl.project([job], batch=False)
    empty = cl.project([], batch=False)
    assert set(full) == set(empty), (
        f"project() return paths disagree: full-only {sorted(set(full) - set(empty))}, "
        f"empty-only {sorted(set(empty) - set(full))}"
    )
    fmt = "{jobs} {usd:.2f} {usd_per_arm:.2f} {mean_input_tokens:.0f} {path}"
    for pr in (full, empty):
        pr = {**pr, "usd_per_arm": pr["usd"]}
        fmt.format(**pr)  # the same fields gate() interpolates
    # the kind-based estimator must actually distinguish the two kinds
    e = cl.project([{**job, "kind": "explain"}], batch=False)["assumed_output_tokens"]
    s = cl.project([{**job, "kind": "score"}], batch=False)["assumed_output_tokens"]
    assert e > s, f"explain ({e}) must project more output tokens than a scorer call ({s})"
    print(f"[selfcheck] projection keys OK; assumed output explain {e} vs score {s}")


def check_gate(cfg, tmp: Path, base: str, set_name: str):
    """`gate()` must pass under the threshold, refuse over it, and pass over it when approved."""
    feats = list(range(100, 100 + N_FEAT))
    synth_build(tmp, base, set_name, "gate_build", feats)
    outcomes = {}
    for label, stop_above, approved in [("under", 1000.0, False), ("over", 0.0001, False),
                                        ("over-approved", 0.0001, True)]:
        args = base_args(tmp, base, set_name, "gate_build", f"gate_run_{label}")
        args.update({"stop_above_usd": stop_above, "approved": approved, "path": "sync"})
        res = R.run(cfg, args)
        outcomes[label] = res["stopped_at"]
    assert outcomes["under"] is None, f"gate refused under the threshold: {outcomes['under']}"
    assert outcomes["over"] is not None, "gate did NOT refuse a stage over the threshold"
    assert outcomes["over-approved"] is None, (
        f"--approved did not release the stage: {outcomes['over-approved']}"
    )
    print(f"[selfcheck] gate OK: under={outcomes['under']}, over refused, approved released")


def base_args(tmp: Path, base: str, set_name: str, build_dir: str, run_dir: str) -> dict:
    return {
        "base": base, "heldout": set_name, "root": str(tmp), "maemm": "stub/maemm",
        "build_dir": build_dir, "run_dir": run_dir, "arms": "", "rows": "", "force": True,
        "argv": ["selfcheck"], "repo_commit": "selfcheck", "gpu": "CPU", "usd_per_s": 0.0,
        "t0": time.time(), "max_cost_usd": 1000.0, "stop_above_usd": 1000.0, "approved": True,
        "concurrency": 4, "scorers": "detection,fuzzing", "path": "sync",
    }


def check_run_both_paths(cfg, tmp: Path, base: str, set_name: str):
    """`run.run()` on sync and batch, each twice -- the second pass is the FULLY CACHED branch."""
    feats = list(range(200, 200 + N_FEAT))
    synth_build(tmp, base, set_name, "run_build", feats)
    for path in ("sync", "batch"):
        first = second = None
        for i in range(2):
            args = base_args(tmp, base, set_name, "run_build", f"run_{path}")
            args["path"] = path
            res = R.run(cfg, args)
            if i == 0:
                first = res
            else:
                second = res
        assert first["calls_this_call"] > 0, f"{path}: first pass sent nothing"
        assert second["cache_hits"] > 0, f"{path}: second pass did not read the cache"
        assert second["calls_this_call"] == 0, (
            f"{path}: second pass sent {second['calls_this_call']} calls that were already cached"
        )
        assert second["stopped_at"] is None, f"{path}: cached replay stopped: {second['stopped_at']}"
        print(f"[selfcheck] run path={path} OK: {first['calls_this_call']} sent, "
              f"then {second['cache_hits']} cache hits and 0 sent")


def check_followup_arms(cfg, tmp: Path, base: str, set_name: str):
    """`--explain2` and `--crossfam` as follow-up arms on a FINISHED run, through `--cache-dir`.

    The property the launch depends on: pointed at a finished run's cache but a NEW product
    directory, only the new arms send calls, every existing arm replays for free, and the finished
    run's own `summary/` is not rewritten.
    """
    feats = list(range(400, 400 + N_FEAT))
    synth_build(tmp, base, set_name, "fu_build", feats)
    base_run = base_args(tmp, base, set_name, "fu_build", "fu_base")
    base_run["arms"] = "C16,M"
    first = R.run(cfg, base_run)
    summary = tmp / "runs" / "fu_base" / "summary" / "scores.jsonl"
    before = summary.read_bytes()
    shared = str(tmp / "runs" / "fu_base" / "cache")

    a = base_args(tmp, base, set_name, "fu_build", "fu_explain2")
    a.update({"arms": "C16", "explain2": True, "cache_dir": shared})
    res = R.run(cfg, a)
    rows = C.read_jsonl(tmp / "runs" / "fu_explain2" / "summary" / "scores.jsonl")
    arms = {r["arm"] for r in rows}
    assert "C16-explain2" in arms, f"--explain2 produced no C16-explain2 rows: {sorted(arms)}"
    assert res["calls_this_call"] > 0, "--explain2 sent nothing: the fresh explanation was not paid"
    assert res["calls_this_call"] < first["calls_this_call"], (
        f"--explain2 re-sent existing arms: {res['calls_this_call']} >= "
        f"{first['calls_this_call']} of the base run"
    )
    assert summary.read_bytes() == before, "the base run's summary/ was rewritten"

    b = base_args(tmp, base, set_name, "fu_build", "fu_crossfam")
    b.update({"arms": "C16,M", "crossfam": "C16,M", "scorers": "detection", "cache_dir": shared})
    R.run(cfg, b)
    rows = C.read_jsonl(tmp / "runs" / "fu_crossfam" / "summary" / "scores.jsonl")
    xarms = {r["arm"] for r in rows if r["arm"].startswith("X")}
    assert xarms == {"XC16-q", "XM-q"}, f"--crossfam arms wrong: {sorted(xarms)}"
    assert all(r["scorer"] == "detection" for r in rows if r["arm"].startswith("X")), (
        "crossfam scored something other than detection"
    )
    print(f"[selfcheck] follow-up arms OK: explain2 sent {res['calls_this_call']} of "
          f"{first['calls_this_call']}, crossfam arms {sorted(xarms)}, base summary untouched")


def check_chain(cfg, tmp: Path, base: str, set_name: str):
    """The chain's whole control flow, with docmax already present and the API stubbed."""
    keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    dm = Path(C.sae_dir(keys[0], str(tmp))) / "examples_docmax" / set_name
    dm.mkdir(parents=True, exist_ok=True)
    (dm / "tested.json").write_text("{}")
    chain_dir = "selfcheck_chain"
    for n_feat, tag in ((N_FEAT, "pilot"), (N_FEAT, "full"), (N_FEAT, "rlI")):
        synth_build(tmp, base, set_name, f"{chain_dir}_{tag}", list(range(300, 300 + n_feat)))
    args = base_args(tmp, base, set_name, "", "")
    args.update({"chain_dir": chain_dir, "maemm": "stub/maemm", "maemm2": "stub/maemm2",
                 "on_commit": None, "on_reload": None})
    res = CH.run(cfg, args)
    assert res["status"] == "done", f"chain did not finish: {res}"
    status = json.loads((tmp / "runs" / chain_dir / "STATUS.json").read_text())
    assert status["state"] == "done", status
    stages = [s["stage"] for s in status["stages"]]
    for want in ("docmax_present", "build_pilot_present", "run_pilot", "acceptance",
                 "batch_probe", "run_primary", "run_rlI"):
        assert want in stages, f"chain never reached {want}: {stages}"
    for f in ("pilot.md", "results.md", "results-rlI.md", "costs.json"):
        p = tmp / "runs" / chain_dir / f
        assert p.exists() and p.stat().st_size > 0, f"chain did not write {f}"
    print(f"[selfcheck] chain OK: stages {stages}, wrote pilot.md / results.md / results-rlI.md")


def main() -> int:
    cfg = C.load_config()
    base = "qwen36-27b"
    set_name = sorted(cfg["heldout"])[-1]
    R.Claude = StubClaude
    # `run.run()` asserts a key is present before it builds any job. The stub never reads it and
    # never opens a socket; this placeholder only satisfies that guard, and is removed afterwards
    # so a real key in the environment is neither used nor shadowed for anything else.
    had_key = "ANTHROPIC_API_KEY" in os.environ
    if not had_key:
        os.environ["ANTHROPIC_API_KEY"] = "selfcheck-placeholder-not-a-key"
    tmp = Path(tempfile.mkdtemp(prefix="autointerp-selfcheck-"))
    try:
        check_projection_keys(cfg)
        check_gate(cfg, tmp, base, set_name)
        check_run_both_paths(cfg, tmp, base, set_name)
        check_followup_arms(cfg, tmp, base, set_name)
        check_chain(cfg, tmp, base, set_name)
    finally:
        R.Claude = _REAL_CLAUDE
        if not had_key:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("[selfcheck] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
