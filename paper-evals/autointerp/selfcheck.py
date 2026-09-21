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
from autointerp import build as B  # noqa: E402
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


VENDORED = Path(__file__).resolve().parent / "third_party" / "delphi-4fea06e"


def _upstream(rel: str) -> dict:
    """Exec one vendored upstream prompt module and return its globals.

    `exec` rather than `import`: the files sit under a path with a dot in it, they are data and
    not a package, and their real package does `from ...clients.client import Client` at module
    scope. The two PROMPT modules are pure string constants plus one `prompt()` function, so
    executing them has no effect beyond binding those names.
    """
    src = (VENDORED / rel).read_text()
    ns: dict = {}
    exec(compile(src, str(VENDORED / rel), "exec"), ns)  # noqa: S102 -- vendored constants
    return ns


def check_delphi_verbatim():
    """Every DELPHI_* string `run.py` sends is byte-identical to upstream at 4fea06e.

    The fidelity claim used to rest on a markdown note saying the strings had been transcribed
    correctly, which is not a check -- and three of our four statements about Delphi turned out to
    be wrong when the source was finally fetched (902d8ce). This compares against
    `third_party/delphi-4fea06e/`, which is `git show 4fea06e:<path>` and nothing else, so a drift
    on either side fails loudly here instead of silently in a prompt.

    Upstream keeps its constants in triple-quoted literals with leading/trailing newlines that the
    prompt builder does not strip; we strip them and let the API's own message formatting do the
    rest. `.strip()` on both sides is therefore the comparison, and it is the ONLY normalisation.
    """
    fz = _upstream("scorers/classifier/prompts/fuzz_prompt.py")
    dt = _upstream("scorers/classifier/prompts/detection_prompt.py")
    ex = _upstream("explainers/default/prompts.py")

    def eq(what, ours, theirs):
        assert ours.strip() == theirs.strip(), (
            f"{what} DIFFERS from upstream 4fea06e.\n"
            f"  ours   ({len(ours):4d} ch): {ours[:200]!r}\n"
            f"  theirs ({len(theirs):4d} ch): {theirs[:200]!r}"
        )

    eq("DELPHI_FUZZ_SYSTEM", R.DELPHI_FUZZ_SYSTEM, fz["DSCORER_SYSTEM_PROMPT"])
    eq("DELPHI_DETECTION_SYSTEM", R.DELPHI_DETECTION_SYSTEM, dt["DSCORER_SYSTEM_PROMPT"])
    eq("DELPHI_EXPLAINER_SYSTEM", R.DELPHI_EXPLAINER_SYSTEM,
       ex["SYSTEM"].replace("{prompt}", "").rstrip())

    # The two few-shot lists, turn by turn, in order, roles included. A list that agreed on
    # content but sent the turns in the wrong order, or as the wrong role, would be a different
    # prompt; zip(strict=True) also catches a list of the wrong length.
    for name, ours, theirs_src in (
        ("DELPHI_FUZZ_FEWSHOT", R.DELPHI_FUZZ_FEWSHOT, fz),
        ("DELPHI_DETECTION_FEWSHOT", R.DELPHI_DETECTION_FEWSHOT, dt),
    ):
        want = []
        for n in ("ONE", "TWO", "THREE"):
            want.append(("user", theirs_src[f"DSCORER_EXAMPLE_{n}"]))
            want.append(("assistant", theirs_src[f"DSCORER_RESPONSE_{n}"]))
        assert len(ours) == len(want), f"{name}: {len(ours)} turns, upstream has {len(want)}"
        for i, (turn, (role, txt)) in enumerate(zip(ours, want, strict=True)):
            assert turn["role"] == role, f"{name}[{i}] role {turn['role']!r} != {role!r}"
            eq(f"{name}[{i}]", turn["content"], txt)

    # The explainer shots are (activations block + explanation) concatenated the way
    # `prompt_builder.py::build_examples` concatenates them.
    for i, ours in enumerate((R.DELPHI_EXPLAINER_FEWSHOT, R.DELPHI_EXPLAINER_FEWSHOT_2,
                              R.DELPHI_EXPLAINER_FEWSHOT_3), start=1):
        turns = {t["role"]: t["content"] for t in ours}
        assert set(turns) == {"user", "assistant"}, f"shot {i}: roles {sorted(turns)}"
        eq(f"DELPHI_EXPLAINER_FEWSHOT_{i}", turns["user"], ex[f"EXAMPLE_{i}_ACTIVATIONS"])
        eq(f"DELPHI_EXPLAINER_FEWSHOT_{i} answer", turns["assistant"],
           ex[f"EXAMPLE_{i}_EXPLANATION"])

    assert R.DELPHI_COMMIT == "4fea06e6e8b68eeaf302474325fca13df95c5d6f", R.DELPHI_COMMIT
    print(f"[selfcheck] delphi verbatim OK: 3 systems, {len(R.DELPHI_FUZZ_FEWSHOT)} fuzz turns, "
          f"{len(R.DELPHI_DETECTION_FEWSHOT)} detection turns, 3 explainer shots, "
          f"all byte-identical to {VENDORED.name} after strip()")


class _StubTok:
    """The one method `build.token_pieces` calls. A real tokenizer would pull in transformers and
    a model download for a check that is purely about WHICH INDICES get marked."""

    def decode(self, ids):
        return f"t{int(ids[0])} "


def check_fuzz_marking():
    """`--fuzz-marks delphi` forces upstream's `len - len//4` index in; `scattered` does not.

    Made to fail first: with the `forced` term deleted from build.py's delphi branch this asserts
    at seed 0 (the sampled set misses index 24), so it exercises the branch rather than restating
    that a set contains what was put into it.
    """
    import numpy as np

    tok, ids = _StubTok(), list(range(1, 33))
    forced = len(ids) - len(ids) // 4          # 24 for a 32-token window
    pieces = B.token_pieces(tok, ids)
    hits = {"delphi": 0, "scattered": 0}
    for seed in range(200):
        for kind in hits:
            t = B.render_test(tok, ids, None, 1.0, random.Random(seed), 3, kind)
            marks = _marked_indices(t["text_fuzz"], pieces)
            assert marks, f"{kind} seed {seed}: no mark placed"
            hits[kind] += int(forced in marks)
    assert hits["delphi"] == 200, (
        f"--fuzz-marks delphi marked the forced index {forced} on only {hits['delphi']}/200 seeds"
    )
    assert hits["scattered"] < 200, (
        f"--fuzz-marks scattered also marked index {forced} on every one of 200 seeds -- the "
        f"check cannot tell the two branches apart"
    )
    # The flag touches NEGATIVES only: a positive is marked from its own activations either way.
    acts = np.zeros(32, dtype=np.float32)
    acts[7] = 5.0
    a = B.render_test(tok, ids, acts, 1.0, random.Random(0), 3, "delphi")
    b = B.render_test(tok, ids, acts, 1.0, random.Random(0), 3, "contiguous")
    assert a["text_fuzz"] == b["text_fuzz"], "fuzz_marks changed a POSITIVE's marking"
    print(f"[selfcheck] fuzz marking OK: delphi forces index {forced} on 200/200 seeds, "
          f"scattered on {hits['scattered']}/200; positives identical under both")


def _marked_indices(text_fuzz: str, pieces: list[str]) -> set[int]:
    """Which token indices ended up inside `<< >>`, recovered from the rendered string.

    Recovered rather than read off `marks`, so the test covers `marked_text`'s run-grouping too:
    a rule that marked the right indices but rendered the delimiters wrong would still fail.
    """
    out, i, pos = set(), 0, 0
    depth_open = text_fuzz
    for i, piece in enumerate(pieces):
        j = depth_open.find(piece, pos)
        if j < 0:
            continue
        before = depth_open[:j]
        if before.count("<<") > before.count(">>"):
            out.add(i)
        pos = j + len(piece)
    return out


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
        "base": base, "heldout": set_name, "root": str(tmp), "maemm": "stub/maemm", "sae": "",
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


def check_relative_marking():
    """The generated-text fallback: marks when the gate marks nothing, never otherwise.

    The defect it exists for is silent -- a block with no token above the gate reaches the
    explainer as bare `Example n:` lines, the explainer answers topically anyway, and `run.py`
    records an ordinary explanation. Nothing raised; the arm was simply scored on a description
    written from unmarked text. So every branch is pinned here, including the ones that must NOT
    fire, because a fallback that also rewrites healthy blocks is a worse bug than the one it fixes.
    """
    import autointerp.build as B

    class _Tok:
        """`token_pieces` decodes ONE id at a time; that is the whole interface needed here."""
        def decode(self, ids):
            return "".join(f"t{int(i)}" for i in ids)

    tok, gate, peak = _Tok(), 1.5, 10.0
    ids = [1, 2, 3, 4]

    # (a) something clears the gate: the fallback must not touch it, either way round.
    hot = [0.2, 2.0, 0.1, 0.3]
    a = B.render_example(tok, ids, hot, peak, gate, rel_fallback=False)
    b = B.render_example(tok, ids, hot, peak, gate, rel_fallback=True)
    assert a["n_marked"] == b["n_marked"] == 1, (a["n_marked"], b["n_marked"])
    assert a["text_marked"] == b["text_marked"], "the fallback rewrote a block the gate marked"
    assert b["marking"] == "gate", b["marking"]

    # (b) nothing clears the gate: OFF leaves it bare, ON marks at >= 0.5 x the block's own peak.
    cold = [0.10, 0.80, 0.40, 0.39]           # peak 0.8, half 0.40 -> marks 0.80 and 0.40, not 0.39
    off = B.render_example(tok, ids, cold, peak, gate, rel_fallback=False)
    on = B.render_example(tok, ids, cold, peak, gate, rel_fallback=True)
    assert off["n_marked"] == 0 and off["marking"] == "gate", (off["n_marked"], off["marking"])
    assert not off["activations"], "an unmarked block must have an empty Activations line"
    assert on["n_marked"] == 2, f"expected the peak and the half-peak token: {on['n_marked']}"
    assert on["marking"] == "relative", on["marking"]
    assert on["block_peak"] == 0.8, on["block_peak"]
    assert on["peak_frac"] == 0.08, on["peak_frac"]      # 0.8 / 10.0, the corpus peak

    # (c) the boundary is INCLUSIVE: a token exactly at half the peak marks.
    edge = B.render_example(tok, [1, 2], [1.0, 0.5], peak, gate, rel_fallback=True)
    assert edge["n_marked"] == 2, f"0.5 x peak must mark: {edge['n_marked']}"

    # (d) a peak of zero or a non-finite one is UNMARKABLE, never all-marked. Marking everything
    # would tell the explainer the feature fires on every token, which is worse than silence.
    # +inf as well as NaN: both crash `quant_act`'s int(ceil(...)) if they reach it, and both are
    # guarded in TWO places (the hoisted `finite` check and the `isfinite(pk)` one), so neither
    # alone going missing changes the answer here -- which is the point of checking both values.
    for bad, what in (([0.0, 0.0], "all zero"), ([float("nan"), 0.0], "NaN"),
                      ([float("inf"), 0.0], "+inf")):
        r = B.render_example(tok, [1, 2], bad, peak, gate, rel_fallback=True)
        assert r["n_marked"] == 0, f"{what}: marked {r['n_marked']} tokens"
        assert r["marking"] == "unmarkable", f"{what}: {r['marking']}"

    # (e) `--mark delphi` is a different rule and the fallback must not shadow it.
    d = B.render_example(tok, ids, cold, peak, gate, mark="delphi", rel_fallback=True)
    assert d["marking"] == "delphi", d["marking"]
    print("  relative marking: gate untouched, fallback fires only when bare, 0/NaN unmarkable")


def check_nla_arms(cfg, tmp: Path, base: str):
    """The three NLA decisions that have no other local test: family filter, arm guard, arm B.

    None of these can be reached through `synth_build`, which fabricates a build directory and so
    starts one stage downstream of everything here. They are checked at the functions instead --
    which is why those functions exist rather than being inline in `build.run` / `sae_self.run`.
    """
    import numpy as np

    import autointerp.sae_self as SS

    # (a) the family filter keeps Ari's `sae2m_enc` label beside `sae`, and `--sae` is honoured.
    assert set(B.FAMILIES) == set(SS.FAMILIES), "build and sae_self must agree on the families"
    assert "sae2m_enc" in B.FAMILIES and "sae" in B.FAMILIES, B.FAMILIES
    hdir = Path(C.heldout_dir(base, "selfcheck_fam", str(tmp)))
    hdir.mkdir(parents=True, exist_ok=True)
    sae_key = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base][-1]
    # Rows carry their own `sae_key`, the way targets.py and features/draw_sae2m.py stamp it since
    # 2026-09-21. Before the conventions layer this fixture left the field off and the filter still
    # answered; `common.sae_rows_of` now refuses an unkeyed SAE row that no set declares, so the
    # unkeyed fixture made this check die on the guard instead of exercising the family filter.
    C.write_jsonl(hdir / "ids.jsonl", [
        {"row": 0, "family": "realact", "id": "doc1:p2:L3"},
        {"row": 1, "family": "sae2m_enc", "id": 4242, "sae_key": sae_key},
        {"row": 2, "family": "random", "id": "g0"},
        {"row": 3, "family": "sae2m_enc", "id": 777, "sae_key": sae_key},
    ])
    rows, sae_rows, feats, key, side = SS._sae_rows(
        cfg, {"base": base, "root": str(tmp), "heldout": "selfcheck_fam", "sae": sae_key}
    )
    assert sae_rows == [1, 3] and feats == [4242, 777], (sae_rows, feats)
    assert key == sae_key, f"--sae was not honoured: {key}"
    # These fixture rows carry no `sae_side`, which reads as `enc` -- the shape every set drawn
    # before 2026-09-21 has, and the shape `--sae-side` must leave untouched.
    assert side == "enc", f"the default side moved to {side!r}"
    assert len(rows) == 4, "the full ids.jsonl must come back, not only the SAE rows"

    # And the guard itself, on the same fixture: an SAE row with NO `sae_key`, in a set that
    # declares none, must refuse rather than default to whatever `--sae` was typed. This is the
    # path that silently scored 512 wrong features on 2026-09-16_v1 before the layer landed.
    C.write_jsonl(hdir / "ids.jsonl", [
        {"row": 0, "family": "realact", "id": "doc1:p2:L3"},
        {"row": 1, "family": "sae2m_enc", "id": 4242},
    ])
    try:
        SS._sae_rows(
            cfg, {"base": base, "root": str(tmp), "heldout": "selfcheck_fam", "sae": sae_key}
        )
    except AssertionError as exc:
        assert "no `sae_key` field" in str(exc), f"refused for the wrong reason: {exc}"
    else:
        raise AssertionError("an unkeyed SAE row in an undeclared set was selected, not refused")

    # (b) arm B's description: tags stripped, whole text when the tag never closed.
    txt = "blah <explanation>\n  neurons that fire on dates \n</explanation> tail"
    hit = B.nla_description(txt)
    assert hit == {
        "tag_found": True,
        "n_chars": len(txt),
        "description": "neurons that fire on dates",
    }, hit
    miss = B.nla_description("  <explanation>never closed  ")
    assert miss["tag_found"] is False and miss["description"] == "<explanation>never closed", miss
    assert B.nla_description("")["description"] == "", "an empty rollout gives an empty description"

    # ...and the CHOICE: the highest sae_self peak wins, which is also arm A's first example.
    peaks = np.array([0.4, 3.1, 1.2, 0.0])
    order = np.argsort(-peaks, kind="stable")
    texts = {0: "<explanation>a</explanation>", 1: "<explanation>b</explanation>",
             2: "<explanation>c</explanation>", 3: "<explanation>d</explanation>"}
    k_best = int(order[0])
    assert k_best == 1, f"argsort(-peaks) must put the highest peak first, got {order.tolist()}"
    assert B.nla_description(texts[k_best])["description"] == "b", "the peak rollout must be chosen"

    # (b2) the covariate spellings: `targets.py` writes density/fires_gated on a `sae` row,
    # `draw_sae2m.py` writes gated_fires and no density on a `sae2m_enc` one. Both are
    # descriptive; a missing one must be None, not a KeyError that stops the build.
    sae_row = {"row": 1, "family": "sae", "id": 5, "stratum": 2, "density": 1e-5, "fires_gated": 91}
    enc_row = {"row": 1, "family": "sae2m_enc", "id": 5, "stratum": 2, "gated_fires": 240}
    assert B._covariate(sae_row, "density") == 1e-5
    assert B._covariate(sae_row, "fires_gated", "gated_fires") == 91, "the sae spelling wins first"
    assert B._covariate(enc_row, "density") is None, "the 2M draw carries no density"
    assert B._covariate(enc_row, "fires_gated", "gated_fires") == 240, "the sae2m_enc spelling"
    assert B._covariate({"gated_fires": None}, "fires_gated", "gated_fires") is None

    # (c) the arm/maemm guard, both directions.
    assert B.check_arm_maemm(["C4", "NLA"], "b/nla", "nla") is True
    assert B.check_arm_maemm(["C4", "C16", "M"], "b/maemm", "full") is False
    for arms, mtype, needle in (
        (["C4", "M"], "nla", "may only build"),
        (["C4", "C16M16"], "nla", "may only build"),
        (["C4", "NLA"], "full", "point --maemm at the `type: nla` entry"),
    ):
        try:
            B.check_arm_maemm(arms, "b/x", mtype)
        except AssertionError as e:
            assert needle in str(e), f"arms {arms} type {mtype}: wrong assert fired: {e}"
        else:
            raise AssertionError(f"check_arm_maemm accepted arms {arms} with a {mtype} maemm")
    print("[selfcheck] NLA arms OK: FAMILIES filter, _covariate, nla_description, check_arm_maemm")


def check_corpus_fallback():
    """A SAE with no `scan` examples/ still builds C4 + NLA; a C16 request refuses by name.

    `examples/<feature>.jsonl` is scan's 16M product and does not exist for the 2M SAE (~$9 to
    make). MEASURED 2026-09-21 on the volume: `--arms C4,NLA` died with
    `FileNotFoundError: .../sae/sae2m/examples/2323.jsonl` after the feature list was already
    printed. The two halves of that file fail differently -- the positive pool has an honest
    substitute, a C16 arm does not -- which is what these two checks pin.
    """
    # (a) the refusal: any arm whose corpus source is "c16", named, with the scan product named.
    for arms in (["C16", "NLA"], ["C4", "C32"], ["C4M", "C16M16"]):
        try:
            B.check_corpus_source(arms, False, "/v/sae/x/examples", "b/x", 4)
        except AssertionError as e:
            assert "run `--product scan`" in str(e) and "examples/tested.json" in str(e), e
            assert all(a in str(e) for a in arms if B.ARM_SPECS[a][0] == "c16"), (
                f"the refusal must NAME the offending arms: {e}"
            )
        else:
            raise AssertionError(f"check_corpus_source accepted {arms} with no scan examples/")
    # ...and the arms that need no c16 pool go through, with the source recorded.
    src = B.check_corpus_source(["C4", "NLA", "M"], False, "/v/sae/x/examples", "b/x", 4)
    assert src == "examples_4m (the 4M prefix; scan's examples/ is absent)", src
    assert B.check_corpus_source(["C16", "C4"], True, "/v/e", "b/x", 4) == "examples/ (scan, 16M)"

    # (b) the pool: with examples/ present the 4M rows are NOT candidates (they are what C4
    # shows); without it they are, band-labelled exactly as the docmax rows are.
    def row(w, act, kind):
        return {"row": 0, "kind": kind, "window": w, "doc": w, "start": 0, "len": 4,
                "max_act": act, "argmax": 0, "acts": [act, 0, 0, 0]}

    peak = 8.0
    ex = [row(1, 8.0, "top"), row(2, 7.0, "top"), row(3, 2.0, "q0"), row(4, 5.0, "q2")]
    ex4 = [row(1, 8.0, "top"), row(5, 6.0, "top")]          # window 1 is already in ex
    doc = [row(6, 3.0, "docmax"), row(4, 5.0, "docmax")]    # window 4 is already a band row

    with_scan = B.candidate_rows(ex, ex4, doc, peak, True)
    assert [c["window"] for c in with_scan] == [3, 4, 6], with_scan
    assert [c["kind"] for c in with_scan] == ["q0", "q2", "q1"], with_scan
    assert all(c["window"] != 5 for c in with_scan), "examples_4m rows are C4's, not candidates"

    without = B.candidate_rows([], ex4, doc, peak, False)
    assert [c["window"] for c in without] == [1, 5, 6, 4], without
    # bands recomputed from max_act against the corpus peak, scan.py:257's own formula
    assert [c["kind"] for c in without] == ["q3", "q2", "q1", "q2"], without
    assert all(c["acts"] for c in without), "a candidate must keep its per-token activations"
    print("[selfcheck] corpus fallback OK: c16 refused by name, examples_4m pool band-labelled")


def check_scores_subset(tmp: Path):
    """A `scores/` directory written by `score --rows` holds the SELECTED targets only.

    sae_self used to index its stored arrays by the held-out set's own row numbers, which is
    right only when the whole set was scored. MEASURED 2026-09-21 on the 2k set's 8-row smoke:
    `ValueError: cannot reshape array of size 32 into shape (2000, 4)`. The arrays are built here
    at the SUBSET shape and read back through the same helper the stage uses, so the bug is
    reproduced rather than described.
    """
    import numpy as np

    import autointerp.sae_self as SS

    n = 4
    scored = [1029, 1035, 1053, 1061]  # what `score --rows 1029,1035,1053,1061` would write
    sdir = tmp / "scores_subset"
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "rows.json", "w") as fh:
        json.dump({"rows": scored, "n": n, "families": ["sae"] * len(scored)}, fh)
    # argmax.i16 is [N_sel, n] over the SCORED rows: row i of the file is `scored[i]`.
    arg = (np.arange(len(scored) * n, dtype=np.int16)).reshape(len(scored), n)
    arg.tofile(sdir / "argmax.i16")

    score_rows, score_ix, sel_ix = SS.scored_rows_of(str(sdir), n, [1035, 1061])
    assert score_rows == scored and list(sel_ix) == [1, 3], (score_rows, sel_ix)
    back = C.read_array(sdir / "argmax.i16", "int16", (len(score_rows), n))[sel_ix]
    assert back.tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]], back.tolist()
    # ...and the CSR flattening, which is `score_ix[row] * n + k`, not `row * n + k`
    assert [score_ix[r] * n + k for r in (1035, 1061) for k in range(n)] == [4, 5, 6, 7, 12, 13, 14, 15]

    # A row the score never covered is named, not reshaped into nonsense.
    try:
        SS.scored_rows_of(str(sdir), n, [1035, 1298])
    except AssertionError as e:
        assert "does NOT hold rows [1298]" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("scored_rows_of accepted a row the scores directory does not hold")
    # A scores directory from a different n is refused too.
    try:
        SS.scored_rows_of(str(sdir), 64, [1035])
    except AssertionError as e:
        assert "not the same run" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("scored_rows_of accepted a scores dir written at another n")

    # The FULL-SET case must stay the identity, or every pilot number would move.
    full = list(range(6))
    with open(sdir / "rows.json", "w") as fh:
        json.dump({"rows": full, "n": n}, fh)
    _sr, _ix, sel_full = SS.scored_rows_of(str(sdir), n, full)
    assert list(sel_full) == full, "a fully scored set must index by its own row numbers"
    print("[selfcheck] scores subset OK: sel_ix, the CSR flattening, both refusals, full-set identity")


def check_chain(cfg, tmp: Path, base: str, set_name: str):
    """The chain's whole control flow, with docmax already present and the API stubbed.

    The SAE key is passed explicitly and the marker is written under THAT key: since the base
    carries two SAEs, `chain.wait_for_docmax` resolves through `common.sae_key_for`, which refuses
    to guess. This check would previously have written the marker under whichever key came first
    in config.yaml and watched the chain look for it under the same accident.
    """
    keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    sae_key = keys[-1]  # not keys[0]: deliberately NOT the one a silent fallback would pick
    dm = Path(C.sae_dir(sae_key, str(tmp))) / "examples_docmax" / set_name
    dm.mkdir(parents=True, exist_ok=True)
    (dm / "tested.json").write_text("{}")
    chain_dir = "selfcheck_chain"
    for n_feat, tag in ((N_FEAT, "pilot"), (N_FEAT, "full"), (N_FEAT, "rlI")):
        synth_build(tmp, base, set_name, f"{chain_dir}_{tag}", list(range(300, 300 + n_feat)))
    args = base_args(tmp, base, set_name, "", "")
    args.update({"chain_dir": chain_dir, "maemm": "stub/maemm", "maemm2": "stub/maemm2",
                 "sae": sae_key, "on_commit": None, "on_reload": None})
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
    set_name = C.default_heldout(cfg)  # the latest NON-imported set, as every entrypoint uses
    R.Claude = StubClaude
    # `run.run()` asserts a key is present before it builds any job. The stub never reads it and
    # never opens a socket; this placeholder only satisfies that guard, and is removed afterwards
    # so a real key in the environment is neither used nor shadowed for anything else.
    had_key = "ANTHROPIC_API_KEY" in os.environ
    if not had_key:
        os.environ["ANTHROPIC_API_KEY"] = "selfcheck-placeholder-not-a-key"
    tmp = Path(tempfile.mkdtemp(prefix="autointerp-selfcheck-"))
    try:
        check_delphi_verbatim()
        check_fuzz_marking()
        check_projection_keys(cfg)
        check_gate(cfg, tmp, base, set_name)
        check_run_both_paths(cfg, tmp, base, set_name)
        check_followup_arms(cfg, tmp, base, set_name)
        check_relative_marking()
        check_nla_arms(cfg, tmp, base)
        check_scores_subset(tmp)
        check_corpus_fallback()
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
