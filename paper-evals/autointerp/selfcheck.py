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
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
if str(PAPER_EVALS) not in sys.path:
    sys.path.insert(0, str(PAPER_EVALS))

import precompute.common as C  # noqa: E402
from autointerp import build as B  # noqa: E402
from autointerp import chain as CH  # noqa: E402
from autointerp import run as R  # noqa: E402
from autointerp import sae_self as SS  # noqa: E402

# Captured at IMPORT, because `main()` rebinds `R.Claude` to the stub: a body that looked up
# `R.Claude` at call time would find the stub and recurse into itself.
_REAL_CLAUDE = R.Claude

N_FEAT = 6
# The arms the paper's run builds (build.FULL_ARMS), so the fabricated build
# directories below carry the same names the real products do.
N_ARMS = ("C16", "M", "M-jac16", "M-cos16")
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


def check_nla_body_tokens():
    """Arm A must show the `<explanation>` BODY, with the activations kept ALIGNED to it.

    Juan's review point: arm B stripped the verbalizer's tags and arm A did not, so the two NLA
    arms read different text from one rollout and only B read the description. The risk in fixing
    it is an ALIGNMENT one -- a body selected by character offsets and activations selected by
    token index are two different things, and a mask that is off by one silently attributes the
    wrong token's activation to the wrong word.

    The mask is `rollouts_nla.explanation_token_mask`, the ONE slicer every judge-facing consumer
    shares (Ari, bdb0705). This branch's own `nla_body_tokens` was dropped in the 2026-09-21
    rebase: it returned a boolean rather than the three statuses, and on an UNCLOSED tag it fell
    back to the whole decode -- handing the judge the opening tag and the preamble, which is the
    defect the fix was for. The unclosed case below is what pins that difference.
    """
    import autointerp.build as B
    from precompute.rollouts_nla import explanation_token_mask

    class _Tok:
        """One char per token id, so a character offset IS a token index and the mapping is
        checkable by hand rather than by trusting the function under test."""
        def decode(self, ids, **kw):
            return "".join(chr(int(i)) for i in ids)

    tok = _Tok()
    text = "pre<explanation>BODY</explanation>post"
    ids = [ord(c) for c in text]
    acts = [float(i) for i in range(len(ids))]
    mask, status = explanation_token_mask(B.token_pieces(tok, ids))
    assert status == "closed", status
    kept = [i for i, m in enumerate(mask) if m]
    assert "".join(chr(ids[i]) for i in kept) == "BODY", "".join(chr(ids[i]) for i in kept)
    # The activations must be the SAME NUMBERS the full rollout carried at those positions --
    # selected in place, never recomputed and never re-indexed from zero.
    lo = text.index("BODY")
    assert [acts[i] for i in kept] == [float(lo + j) for j in range(4)], [acts[i] for i in kept]

    # An UNCLOSED tag keeps everything AFTER the opening tag -- not the whole decode, which is
    # what the dropped `nla_body_tokens` fallback did.
    bad = "pre<explanation>never closed"
    b_mask, b_status = explanation_token_mask(B.token_pieces(tok, [ord(c) for c in bad]))
    assert b_status == "unclosed", b_status
    kept_bad = "".join(c for c, m in zip(bad, b_mask, strict=True) if m)
    assert kept_bad == "never closed", kept_bad

    # NO tag at all keeps every token and says so, rather than dropping the feature and
    # shrinking this arm relative to the others in a paired comparison.
    none = "no tags here at all"
    n_mask, n_status = explanation_token_mask(B.token_pieces(tok, [ord(c) for c in none]))
    assert n_status == "none" and all(n_mask), (n_status, n_mask)
    print("  nla body: tags stripped, activations aligned in place, unclosed/none stated")


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

    # (b) arm B's description: tags stripped, and the three tag states NAMED.
    #
    # UPDATED in the 2026-09-21 rebase, deliberately. `nla_description` now goes through
    # `rollouts_nla.explanation_body` (Ari, bdb0705) instead of `extract_explanation`, which
    # changes what an UNCLOSED answer contributes and adds `tag_status`. The old expectations
    # here -- `tag_found: False` and the whole raw decode as the description -- are exactly the
    # behaviour that change removed, so they are rewritten rather than relaxed: an answer that ran
    # into max_new is still an answer, and handing the judge its opening tag and chat preamble was
    # the defect (Juan's review).
    txt = "blah <explanation>\n  neurons that fire on dates \n</explanation> tail"
    hit = B.nla_description(txt)
    assert hit == {
        "tag_found": True,
        "tag_status": "closed",
        "n_chars": len(txt),
        "description": "neurons that fire on dates",
    }, hit
    raw_unclosed = "  <explanation>never closed  "
    miss = B.nla_description(raw_unclosed)
    assert miss["tag_status"] == "unclosed" and miss["tag_found"] is True, miss
    assert miss["description"] == "never closed", (
        f"an unclosed answer must contribute what follows its OPENING tag, not the raw decode: "
        f"{miss['description']!r}")
    assert "<explanation>" not in miss["description"], (
        "the opening tag reached the judge -- this is the exact defect explanation_body replaced")
    none = B.nla_description("no tags here")
    assert none["tag_status"] == "none" and none["tag_found"] is False, none
    assert none["description"] == "no tags here", none
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
    assert B.check_arm_maemm(["C16", "NLA"], "b/nla", "nla") is True
    assert B.check_arm_maemm(["C16", "M", "M-cos16"], "b/maemm", "full") is False
    for arms, mtype, needle in (
        (["C16", "M"], "nla", "may only build"),
        (["C16", "M-jac16"], "nla", "may only build"),
        (["C16", "M-cos16"], "nla", "may only build"),
        (["C16", "NLA"], "full", "point --maemm at the `type: nla` entry"),
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
    for arms in (["C16-win", "NLA"], ["C16", "C32"], ["C4M", "C16M16"]):
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
    src = B.check_corpus_source(["C16", "NLA", "M"], False, "/v/sae/x/examples", "b/x", 4)
    assert src == "examples_4m (the 4M prefix; scan's examples/ is absent)", src
    assert B.check_corpus_source(["C16-win", "C4M"], True, "/v/e", "b/x", 4) == \
        "examples/ (scan, the test corpus)"
    # The two sides can be on different corpora: the LABEL follows the test side, the refusal the
    # shown side. A shown side with examples/ and a test side without must say `examples_4m`.
    assert B.check_corpus_source(["C16-win"], True, "/v/e", "b/x", 4, t_use_examples=False) == \
        "examples_4m (the 4M prefix; scan's examples/ is absent)"

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



# ---------------------------------------------------------------------------------------------
# The two-corpus build, end to end on a synthetic volume (M6, 2026-09-23)
# ---------------------------------------------------------------------------------------------

SC_SHOWN = "selfcheck_shown10m"   # deliberately NOT a registered `corpora:` key: the geometry
SC_TEST = ""                      # assert is exercised by the registered ones, not by a fixture
SC_NROLL = 64
SC_NFEAT = 3
SC_TOKW = 16
SC_DOCLEN = 64


class _FakeAutoTokenizer:
    """`build.run` does `from transformers import AutoTokenizer`; the CPU selfcheck container has
    no transformers and no model. Decoding is the only thing build asks of it."""

    @staticmethod
    def from_pretrained(_path):
        return _StubTok()


def _ex_row(row, kind, doc, act, ln=SC_DOCLEN):
    """One stored example window, in `scan`'s own schema (precompute/scan.py:486-501)."""
    acts = [0.0] * ln
    acts[3] = float(act)
    return {"row": row, "kind": kind, "window": doc, "doc": doc, "start": 0, "len": ln,
            "max_act": float(act), "argmax": 3, "acts": acts}


def _write_corpus(root, base, name, n_docs):
    import numpy as np

    d = C.corpus_dir(base, str(root), name)
    Path(d).mkdir(parents=True, exist_ok=True)
    docs = [{"doc": i, "offset": i * SC_DOCLEN, "len": SC_DOCLEN, "size_tag": 16}
            for i in range(n_docs)]
    C.write_jsonl(f"{d}/docs.jsonl", docs)
    # Token ids are the DOCUMENT's index times 1000 plus the position, so a recovered window says
    # which corpus and which document it came from and a cross-corpus mix-up is visible in the
    # rendered text rather than being a plausible-looking string.
    off = 1 if name else 0
    toks = np.arange(n_docs * SC_DOCLEN, dtype=np.int32) + off * 1_000_000
    toks.tofile(f"{d}/tokens.i32")
    return d


def _write_two_corpus_volume(cfg, tmp: Path, base: str):
    """A whole synthetic volume: two corpora, two example pools, sae_self, scores, random pool."""
    import numpy as np

    root, set_name = str(tmp / "vol2"), "selfcheck_2corp"
    sae_key = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base][0]
    maemm = [k for k in cfg["maemms"]
             if C.split_key(k, "maemm")[0] == base and cfg["maemms"][k]["type"] != "nla"][0]
    feats = [11, 22, 33][:SC_NFEAT]
    rows = list(range(SC_NFEAT))
    gate, peak = 1.0, 8.0

    # ---- corpora. The SHOWN one carries 24 documents (the C16 arm needs 16) and the TEST one 96
    # (bands, near-misses and the random pool, over two disjoint draws). Document ids OVERLAP
    # between them on purpose: they are different documents that share an integer, which is
    # exactly what the cross-corpus disjointness rule has to not be fooled by.
    _write_corpus(tmp / "vol2", base, "", 96)
    _write_corpus(tmp / "vol2", base, SC_SHOWN, 24)

    # ---- held-out set
    hdir = C.heldout_dir(base, set_name, root)
    Path(hdir).mkdir(parents=True, exist_ok=True)
    C.write_jsonl(f"{hdir}/ids.jsonl", [
        {"row": r, "family": "sae", "id": f, "sae_key": sae_key, "stratum": i % 4,
         "density": 1e-5, "fires_gated": 90 + i}
        for i, (r, f) in enumerate(zip(rows, feats, strict=True))
    ])

    sdir_sae = C.sae_dir(sae_key, root)
    mx = np.zeros(max(feats) + 1, dtype=np.float16)
    for f in feats:
        mx[f] = peak
    Path(sdir_sae).mkdir(parents=True, exist_ok=True)
    mx.tofile(f"{sdir_sae}/max_act.f16")

    # ---- SHOWN examples: examples_docmax on the shown corpus, one window per document.
    exdoc = SS.examples_docmax_dir(sae_key, set_name, root, SC_SHOWN)
    Path(exdoc).mkdir(parents=True, exist_ok=True)
    json.dump({"features": feats, "rows": rows, "sae": sae_key}, open(f"{exdoc}/tested.json", "w"))
    for r, f in zip(rows, feats, strict=True):
        C.write_jsonl(f"{exdoc}/{f}.jsonl",
                      [_ex_row(r, "docmax", doc, peak - 0.1 * doc) for doc in range(24)])

    # ---- TEST examples: scan's band rows on the default corpus, plus a `top` tier.
    exd = C.sae_examples_dir(sae_key, set_name, root, corpus_name=SC_TEST, write=True)
    Path(exd).mkdir(parents=True, exist_ok=True)
    json.dump({"features": feats, "rows": rows, "sae": sae_key}, open(f"{exd}/tested.json", "w"))
    for r, f in zip(rows, feats, strict=True):
        rws = []
        doc = 0
        for qi, band in enumerate(B.BANDS):          # 4 gate-passing windows per band
            for _ in range(4):
                rws.append(_ex_row(r, band, doc, peak * (qi + 1) / 4 - 0.01))
                doc += 1
        for _ in range(6):                           # below-gate band rows = the near-miss pool
            rws.append(_ex_row(r, "q0", doc, gate * 0.5))
            doc += 1
        for _ in range(4):
            rws.append(_ex_row(r, "top", doc, peak))
            doc += 1
        C.write_jsonl(f"{exd}/{f}.jsonl", rws)

    # ---- TEST document-diverse pool (A4's own pool), also on the test corpus, on documents no
    # band row uses. Without it `build` refuses rather than drawing a knowingly short test set.
    t_exdoc = SS.examples_docmax_dir(sae_key, set_name, root, SC_TEST)
    Path(t_exdoc).mkdir(parents=True, exist_ok=True)
    json.dump({"features": feats, "rows": rows, "sae": sae_key}, open(f"{t_exdoc}/tested.json", "w"))
    for r, f in zip(rows, feats, strict=True):
        C.write_jsonl(f"{t_exdoc}/{f}.jsonl",
                      [_ex_row(r, "docmax", doc, peak * 0.9) for doc in range(26, 40)])

    # ---- the shared negative pool, on the TEST corpus, over documents no band row uses.
    pdir = SS.random_pool_dir(sae_key, set_name, root, SC_TEST)
    Path(pdir).mkdir(parents=True, exist_ok=True)
    n_win = 40
    wins = [{"window": i, "doc": 40 + i, "start": 0, "len": SC_DOCLEN} for i in range(n_win)]
    C.write_jsonl(f"{pdir}/windows.jsonl", wins)
    json.dump({"n_windows": n_win, "features": feats, "gate": gate},
              open(f"{pdir}/pool.json", "w"))
    pm = np.zeros((len(feats), n_win), dtype=np.float16)
    pm[:, n_win // 2:] = np.float16(gate * 0.5)      # half zero-activation, half near-miss
    pm.tofile(f"{pdir}/max_act.f16")
    np.zeros(len(feats) * n_win + 1, dtype=np.int64).tofile(f"{pdir}/tok_off.i64")
    np.zeros(0, dtype=np.int16).tofile(f"{pdir}/tok_pos.i16")
    np.zeros(0, dtype=np.float16).tofile(f"{pdir}/tok_val.f16")

    # ---- sae_self: per-token activations on the MAEMM's own rollouts.
    sdir = C.scores_dir(maemm, set_name, root, "vllm", "")
    self_dir = f"{sdir}/sae_self"
    Path(self_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    acts = rng.random((len(rows), SC_NROLL, SC_TOKW)).astype(np.float16) * 4.0
    ids = np.zeros((len(rows), SC_NROLL, SC_TOKW), dtype=np.int32)
    for i in range(len(rows)):
        for k in range(SC_NROLL):
            # A rollout's vocabulary is a function of k, so the 64 texts are genuinely different
            # and a content-word Jaccard over them is not degenerate.
            ids[i, k] = np.arange(SC_TOKW) + 10 * (k % 8) + 100 * (k // 8) + 1000 * i
    acts.tofile(f"{self_dir}/sae_self.f16")
    ids.tofile(f"{self_dir}/sae_self_ids.i32")
    json.dump({
        "gate": gate, "rows": rows, "n": SC_NROLL, "width": SC_TOKW,
        "checks": {"argmax_ok": True, "csr_value_mismatches": 0, "csr_membership_mismatches": 0},
        "per_target": [{"row": r, "fire_fraction": 0.5} for r in rows],
    }, open(f"{self_dir}/sae_self.json", "w"))

    # ---- score's first-pass residuals, which M-cos16 reads and nothing else here does.
    d_model = int(cfg["bases"][base]["d"])
    best = rng.standard_normal((len(rows), SC_NROLL, d_model)).astype(np.float16)
    mu = np.full(d_model, 3.0, dtype=np.float32)     # a LARGE shared mean, the thing to subtract
    best = (best.astype(np.float32) + mu[None, None, :]).astype(np.float16)
    best.tofile(f"{sdir}/best_act.f16")
    json.dump({"rows": rows, "n": SC_NROLL, "families": ["sae"] * len(rows),
               "score_max_length": SC_TOKW, "mu": None}, open(f"{sdir}/rows.json", "w"))
    Path(f"{root}/base/{base}/stats").mkdir(parents=True, exist_ok=True)
    mu.tofile(f"{root}/base/{base}/stats/selfcheck_mu.f32")
    return root, set_name, sae_key, maemm, feats


def check_two_corpora(cfg, tmp: Path, base: str):
    """The WHOLE `build` stage, on CPU, with the shown examples and the test windows on DIFFERENT
    corpora -- the one thing the 2026-09-22 spec update bought, and the one thing no other check
    here reaches, because every other check starts from a fabricated build directory.

    What it pins, all of it MEASURED from the products the run writes:
      * `C16`'s examples come from the SHOWN corpus and the test items from the TEST corpus, by
        the token ids each one recovers (the two corpora are numbered a million apart);
      * `M`, `M-jac16` and `M-cos16` each show exactly 16 of the SAME 64 rollouts, and the three
        selections are different sets;
      * `M-cos16` reads `best_act.f16` and takes no forward pass -- the fixture provides no model;
      * `build.json` records both corpora, the disjointness regime and the per-arm count of shown
        examples above the feature's corpus peak;
      * and the MUTATION: with the shown corpus pointed at the test corpus the C16 examples stop
        coming from the shown corpus, which is what makes the first assertion a test.
    """

    cfg = json.loads(json.dumps(cfg))                # a private copy: this check edits it
    root, set_name, sae_key, maemm, feats = _write_two_corpus_volume(cfg, tmp, base)
    # A small test set, so the fixture needs tens of documents and not thousands. The ARM counts
    # are NOT touched: N = 16 per arm is what is under test.
    cfg["autointerp"].update({"n_pos": 4, "n_neg": 4, "n_neg_nearmiss": 2,
                              "random_pool_windows": 40})
    cfg["bases"][base]["whiten_mu"] = "base/{base}/stats/selfcheck_mu.f32"
    arms = "C16,M,M-jac16,M-cos16"

    def build(name, shown, test=""):
        args = {"base": base, "maemm": maemm, "sae": sae_key, "heldout": set_name, "root": root,
                "engine": "vllm", "arms": arms, "n_feat": len(feats), "build_dir": name,
                "corpus_name": shown, "test_corpus_name": test, "force": True, "argv": ["selfcheck"]}
        return B.run(cfg, args), f"{C.base_dir(base, root)}/autointerp/{set_name}/{name}"

    real_snapshot, real_tf = C.snapshot, sys.modules.get("transformers")
    C.snapshot = lambda *_a, **_k: "(selfcheck stub tokenizer)"
    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)
    try:
        _res, out = build("two_corpora", SC_SHOWN, SC_TEST)
        info = json.load(open(f"{out}/build.json"))
        assert info["two_corpora"] is True and info["shown_corpus"] == SC_SHOWN, info
        assert "by corpus separation" in info["disjointness"], info["disjointness"]

        rows = C.read_jsonl(f"{out}/{feats[0]}.jsonl")
        arm_rows = {r["arm"]: r for r in rows if r["kind"] == "arm"}
        assert set(arm_rows) == {"C16", "M", "M-jac16", "M-cos16"}, sorted(arm_rows)
        for a in ("C16", "M", "M-jac16", "M-cos16"):
            assert arm_rows[a]["n"] == 16, f"arm {a} shows {arm_rows[a]['n']} examples, not 16"
        # THE SPLIT ITSELF, read off the rendered text. The shown corpus's token ids are offset by
        # 1,000,000 and the test corpus's are not, so one substring decides which memmap each
        # block came out of. `_StubTok` decodes id i as "t<i> ".
        assert "t1000" in arm_rows["C16"]["block"], "C16 was not rendered from the shown corpus"
        tests = [r for r in rows if r["kind"] == "test"]
        assert tests, "no draw-1 test items"
        assert all("t1000" not in t["text"] for t in tests), (
            "a test item was rendered from the SHOWN corpus -- the split does not hold"
        )
        # The three M arms are the same 64 rollouts, differently chosen.
        sel = {a: [e["k"] for e in arm_rows[a]["examples"]] for a in ("M", "M-jac16", "M-cos16")}
        assert all(len(set(v)) == 16 for v in sel.values()), sel
        assert sel["M"] == sorted(sel["M"], key=lambda k: sel["M"].index(k))
        for a, b in (("M-jac16", "M"), ("M-cos16", "M"), ("M-cos16", "M-jac16")):
            assert set(sel[a]) != set(sel[b]), f"{a} selected exactly the same 16 rollouts as {b}"
        assert sel["M-jac16"][0] == sel["M"][0] == sel["M-cos16"][0], (
            "every selection arm is seeded with the top-activation rollout"
        )
        # The per-arm clamp counter reaches build.json (plan M6; SMOKES.md:3731's table gap).
        by_arm = info["n_shown_exceeding_corpus_peak_by_arm"]
        assert set(by_arm) >= set(arm_rows), by_arm
        assert by_arm["C16"] == 0, "a corpus window cannot exceed the feature's own corpus peak"

        # ---- the MUTATION: one corpus on both sides, everything else identical.
        _res2, out2 = build("one_corpus", SC_TEST, SC_TEST)
        info2 = json.load(open(f"{out2}/build.json"))
        assert info2["two_corpora"] is False and "asserted per feature" in info2["disjointness"]
        rows2 = C.read_jsonl(f"{out2}/{feats[0]}.jsonl")
        c16_2 = next(r for r in rows2 if r["kind"] == "arm" and r["arm"] == "C16")
        assert "t1000" not in c16_2["block"], (
            "the shown-corpus assertion above does not discriminate: C16 renders from the shown "
            "corpus even when the shown corpus IS the test corpus"
        )
    finally:
        C.snapshot = real_snapshot
        if real_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real_tf
    print(f"[selfcheck] two corpora OK: C16 from {SC_SHOWN}, test items from the default corpus, "
          f"16 examples on each of M / M-jac16 / M-cos16, best_act read with no GPU")


def check_products_set(cfg, tmp: Path, base: str):
    """`build --set <decoder twin> --sae-side dec --products-set <encoder set>` (M6-dec).

    The corpus-side pools (shown docmax, test bands, random-pool negatives) are per FEATURE and are
    stored under the ENCODER set's name; the M arms are the TWIN's own rollouts. What must hold:
      * the twin build reads every corpus pool from the products set and renders the IDENTICAL
        C16 block and the IDENTICAL test items as the encoder build -- which is what lets C16 and
        the three nulls replay from the call cache -- while its M block comes from the twin's
        `sae_self__dec` (its token ids carry a marker the encoder rollouts never have);
      * the twin's SAE rows are 1..3 and the products set's records are stamped 0..2, so a build
        that joined the corpus side on the twin's own row would trip `_rows`'s row assert: the
        success below is only reachable through the feature-id row map;
      * `build.json` says which set each side came from;
      * MUTATIONS: without `--products-set` the twin build refuses (no pools under its name);
        a twin whose feature ids are not the products set's, in order, refuses; the default side
        on a decoder-only set refuses (it has no encoder rows).
    """
    import re

    import numpy as np

    cfg = json.loads(json.dumps(cfg))
    root, set_name, sae_key, maemm, feats = _write_two_corpus_volume(cfg, tmp / "ps", base)
    cfg["autointerp"].update({"n_pos": 4, "n_neg": 4, "n_neg_nearmiss": 2,
                              "random_pool_windows": 40})
    cfg["bases"][base]["whiten_mu"] = "base/{base}/stats/selfcheck_mu.f32"
    arms = "C16,M,M-jac16,M-cos16"
    twin = "selfcheck_2corp_dec"

    def write_twin(name, fs):
        hdir = C.heldout_dir(base, name, root)
        Path(hdir).mkdir(parents=True, exist_ok=True)
        C.write_jsonl(f"{hdir}/ids.jsonl", [{"row": 0, "family": "random", "id": 0}] + [
            {"row": 1 + i, "family": "sae", "id": f, "sae_key": sae_key, "sae_side": "dec",
             "vector": "dec", "stratum": i % 4, "density": 1e-5, "fires_gated": 90 + i}
            for i, f in enumerate(fs)])

    write_twin(twin, feats)
    rows = [1 + i for i in range(len(feats))]
    sdir = C.scores_dir(maemm, twin, root, "vllm", "")
    self_dir = f"{sdir}/sae_self__dec"
    Path(self_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)
    acts = rng.random((len(rows), SC_NROLL, SC_TOKW)).astype(np.float16) * 4.0
    ids = np.zeros((len(rows), SC_NROLL, SC_TOKW), dtype=np.int32)
    for i in range(len(rows)):
        for k in range(SC_NROLL):
            ids[i, k] = np.arange(SC_TOKW) + 10 * (k % 8) + 100 * (k // 8) + 1000 * i + 700_000
    acts.tofile(f"{self_dir}/sae_self.f16")
    ids.tofile(f"{self_dir}/sae_self_ids.i32")
    json.dump({
        "gate": 1.0, "rows": rows, "n": SC_NROLL, "width": SC_TOKW, "sae_side": "dec",
        "checks": {"argmax_ok": True, "csr_value_mismatches": 0, "csr_membership_mismatches": 0},
        "per_target": [{"row": r, "fire_fraction": 0.5} for r in rows],
    }, open(f"{self_dir}/sae_self.json", "w"))
    d_model = int(cfg["bases"][base]["d"])
    best = (rng.standard_normal((len(rows), SC_NROLL, d_model)) + 3.0).astype(np.float16)
    best.tofile(f"{sdir}/best_act.f16")
    json.dump({"rows": rows, "n": SC_NROLL, "families": ["sae"] * len(rows),
               "score_max_length": SC_TOKW, "mu": None}, open(f"{sdir}/rows.json", "w"))

    def build(name, heldout, **extra):
        args = {"base": base, "maemm": maemm, "sae": sae_key, "heldout": heldout, "root": root,
                "engine": "vllm", "arms": arms, "n_feat": len(feats), "build_dir": name,
                "corpus_name": SC_SHOWN, "test_corpus_name": SC_TEST, "force": True,
                "argv": ["selfcheck"], **extra}
        B.run(cfg, args)
        return f"{C.base_dir(base, root)}/autointerp/{heldout}/{name}"

    def refuses(msg, name, heldout, **extra):
        try:
            build(name, heldout, **extra)
        except AssertionError as e:
            assert msg in str(e), f"wrong refusal ({msg!r} expected): {e}"
        else:
            raise AssertionError(f"build {name} did not refuse ({msg!r} expected)")

    marker = re.compile(r"t7\d{5}\b")
    real_snapshot, real_tf = C.snapshot, sys.modules.get("transformers")
    C.snapshot = lambda *_a, **_k: "(selfcheck stub tokenizer)"
    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)
    try:
        out_enc = build("enc", set_name)
        out_dec = build("dec", twin, sae_side="dec", products_set=set_name)
        info = json.load(open(f"{out_dec}/build.json"))
        sides = info["set_sides"]
        assert sides["rollout_side"]["set"] == twin and sides["rollout_side"]["sae_side"] == "dec"
        assert sides["rollout_side"]["sae_self"].endswith("/sae_self__dec"), sides
        assert sides["corpus_side"]["set"] == set_name, sides
        assert "by feature id" in sides["corpus_side"]["row_map"], sides
        assert f"/{set_name}__" in info["random_pool"] + "__", info["random_pool"]
        info_enc = json.load(open(f"{out_enc}/build.json"))
        assert info_enc["set_sides"]["corpus_side"]["set"] == set_name
        assert info_enc["set_sides"]["corpus_side"]["row_map"] == "identity (one set)"
        for i, f in enumerate(feats):
            e_rows = C.read_jsonl(f"{out_enc}/{f}.jsonl")
            d_rows = C.read_jsonl(f"{out_dec}/{f}.jsonl")
            assert d_rows[0]["row"] == 1 + i and e_rows[0]["row"] == i, (d_rows[0], e_rows[0])
            e_arm = {r["arm"]: r for r in e_rows if r["kind"] == "arm"}
            d_arm = {r["arm"]: r for r in d_rows if r["kind"] == "arm"}
            assert d_arm["C16"]["block"] == e_arm["C16"]["block"], (
                f"feature {f}: the twin's C16 block differs from the encoder build's -- the corpus "
                f"side was not read from the products set, and C16 would not replay from cache")
            e_t = [(r["kind"], r["i"], r["text"], r.get("label"), r.get("text_fuzz"))
                   for r in e_rows if r["kind"].startswith("test")]
            d_t = [(r["kind"], r["i"], r["text"], r.get("label"), r.get("text_fuzz"))
                   for r in d_rows if r["kind"].startswith("test")]
            assert e_t and d_t == e_t, f"feature {f}: the twin's test items differ from the encoder's"
            for a in ("M", "M-jac16", "M-cos16"):
                assert marker.search(d_arm[a]["block"]), (
                    f"feature {f}: the twin's {a} block is not from sae_self__dec")
                assert not marker.search(e_arm[a]["block"]), f"feature {f}: marker in the enc {a}"

        # MUTATION: the twin on its own -- no corpus pools exist under its name.
        refuses("document-diverse", "dec_nops", twin, sae_side="dec")
        # MUTATION: a twin whose features are the products set's in ANOTHER order.
        write_twin("selfcheck_2corp_decperm", list(reversed(feats)))
        refuses("NOT the same ids", "decperm", "selfcheck_2corp_decperm", sae_side="dec",
                products_set=set_name)
        # MUTATION: the default (enc) side on a decoder-only set.
        refuses("has no enc rows", "dec_enc", twin, products_set=set_name)
    finally:
        C.snapshot = real_snapshot
        if real_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real_tf
    print(f"[selfcheck] products set OK: twin C16 blocks and test items identical to the encoder "
          f"build's over {len(feats)} features, M arms from sae_self__dec, 3 mutation gates")


def check_corpus_key(cfg, base: str):
    """ONE key string for the producer and the consumer of the three corpus pools.

    M2 and M6 arrived at the same need by two spellings: M6 keyed `random_pool` / `examples_4m` /
    `examples_docmax` by the corpus DIRECTORY, M2 keyed them by `scan`'s key, which additionally
    carries the `--run-tag`. Reconciled 2026-09-23 onto the scan spelling
    (`precompute.top1_act.scan_key_of`), because `scan` writes the `examples/` half that `build`
    reads beside these three and two spellings of one path is exactly the cross-corpus join the
    change exists to stop. What is pinned here:

      * the key rule, including `train_parity_10m__paper0923` -- products under that name are on
        the volume and must still be addressed by `--corpus-name train_parity_10m --run-tag
        paper0923`, and the empty/empty case must still be today's unsuffixed path;
      * that the producer (`sae_self.corpus_key_of` off a launch's args) and the consumer
        (`build`'s `corpus_key_for(shown_corpus, run_tag)`) land on the SAME three directories;
      * the MUTATION: the pre-reconcile spelling, the bare corpus directory, disagrees with all
        three as soon as a run carries a tag -- so the assertion above is a real comparison.
    """

    from precompute.top1_act import scan_key_of

    root, set_name = "/vol", C.default_heldout(cfg)
    sae_key = C.sae_key_for(cfg, base, f"{base}/l42-1b")
    checks = mut = 0
    for corpus, tag, want in (
        ("", "", ""),                                                   # today's path
        ("train_parity_10m", "", "train_parity_10m"),
        ("train_parity_10m", "paper0923", "train_parity_10m__paper0923"),  # on the volume
        ("", "paper0923", "paper0923"),                                 # as `scan` keys it
    ):
        got = SS.corpus_key_for(corpus, tag)
        assert got == want, f"corpus_key_for({corpus!r}, {tag!r}) = {got!r}, want {want!r}"
        assert got == scan_key_of(corpus, 0, tag), (
            f"the autointerp pools and `scan` spell the key of ({corpus!r}, {tag!r}) differently"
        )
        checks += 1

    # PRODUCER: what a `--stage examples_docmax --corpus-name ... --run-tag ...` launch writes.
    args = {"corpus_name": "train_parity_10m", "run_tag": "paper0923"}
    pkey = SS.corpus_key_of(cfg, args)          # asserts the declared geometry on the way
    # CONSUMER: what `build` addresses, derived from its own two arguments, not from `pkey`.
    ckey = SS.corpus_key_for("train_parity_10m", "paper0923")
    assert pkey == ckey == "train_parity_10m__paper0923", (pkey, ckey)
    checks += 1
    dirs = [f(sae_key, set_name, root, pkey) for f in
            (SS.random_pool_dir, SS.examples_4m_dir, SS.examples_docmax_dir)]
    assert all(d.endswith(f"/{set_name}__train_parity_10m__paper0923") for d in dirs), dirs
    checks += 1

    # MUTATION: key by the corpus directory alone, as before the reconciliation.
    stale = [f(sae_key, set_name, root, "train_parity_10m") for f in
             (SS.random_pool_dir, SS.examples_4m_dir, SS.examples_docmax_dir)]
    assert all(a != b for a, b in zip(dirs, stale, strict=True)), (
        "the bare corpus directory and the tagged key give the same path, so this check cannot "
        "tell the reconciled spelling from the one it replaced"
    )
    mut += 1
    # MUTATION: a comma-joined --corpus-name is refused rather than silently taking the first.
    try:
        SS.corpus_of(cfg, {"corpus_name": "train_parity_10m,corpus"})
    except AssertionError as e:
        assert "ONE" in str(e), str(e)
        mut += 1
    else:
        raise AssertionError("a corpus-side stage accepted several corpora")
    print(f"[selfcheck] corpus key OK: {checks} checks, {mut} mutation gates; producer and "
          f"consumer both address {set_name}__train_parity_10m__paper0923")


def check_nla_rollout_stem(cfg, tmp: Path, base: str):
    """The NLA build addresses the TAGGED, HF-spelled verbalizer product — and nothing else.

    THE DEFECT, 2026-09-22 (M6-2's third refusal, `ap-uFQQ6lDL5eKZ5bRv9FwbYL`). `build` read its
    NLA rollouts with `C.rollout_stem(set_name, engine)`, dropping both axes at once:

      * no RUN TAG, so a tagged run asked for the bare `<set>`;
      * this stage's `--engine`, which defaults to `vllm`, instead of the verbalizer's `hf`.

    Together they named `<set>__vllm.jsonl` — absent, so it refused. The tempting repair,
    `--engine hf`, would have been WORSE than the refusal: it resolves to the bare `<set>.jsonl`,
    and the untagged 09-21 production rollouts of a DIFFERENT generation over DIFFERENT rows are
    on the volume under exactly that name. A plausible NLA number about the wrong text.

    The fixture is the volume's own shape: an untagged whole-set file (the trap) beside the
    tagged `__rows512-1023` chunk M4 actually wrote. What is pinned:

      * the tagged chunk is what comes back, whole, through `common.read_rollouts`;
      * MUTATION — with the tagged product removed and the untagged file still there, the read
        REFUSES and names the untagged file rather than consuming it;
      * MUTATION — the pre-fix stem and the fixed stem are different strings, and the pre-fix one
        is not a file, so the assertion above is a real comparison and not a tautology;
      * the SIBLING: `engine_of` sends an `nla` maemm's `scores_dir` to the same HF spelling
        `sae_self` wrote, which the `vllm` default missed too.
    """
    nla = [k for k in cfg["maemms"]
           if C.split_key(k, "maemm")[0] == base and cfg["maemms"][k]["type"] == "nla"][0]
    root = str(tmp / "nlastem")
    set_name, tag = "selfcheck_v3_ctrl", "paper0923"
    rdir = Path(C.rollouts_dir(nla, root))
    rdir.mkdir(parents=True, exist_ok=True)

    def _write(stem: str, rows: list[int], mark: str):
        C.write_jsonl(str(rdir / f"{stem}.jsonl"),
                      [{"row": r, "k": 0, "text": f"{mark}:{r}"} for r in rows])
        (rdir / f"{stem}.summary.json").write_text(json.dumps({"rows": rows, "n": 1}))

    # The UNTAGGED whole-set product of the other generation run — the thing that must never be
    # read when a tag is given. Rows 0-1 stand in for the volume's 0-1023.
    untagged = C.rollout_stem(set_name, "hf", "")
    _write(untagged, [0, 1], "WRONG-RUN")
    # M4's product: HF-spelled, tagged, and written as ONE `--rows` chunk.
    tagged = C.rollout_chunk_stem(C.rollout_stem(set_name, "hf", tag), "2-3")
    _write(tagged, [2, 3], "nla")

    recs, stem = B.read_nla_rollouts(nla, set_name, root, tag)
    assert stem == f"{set_name}__{tag}", f"the NLA stem is not the tagged HF stem: {stem!r}"
    assert sorted(r["row"] for r in recs) == [2, 3], (
        f"the tagged chunk is not what came back: {[r['row'] for r in recs]}"
    )
    assert all(r["text"].startswith("nla:") for r in recs), (
        f"the UNTAGGED product's text reached the build: {[r['text'] for r in recs]}"
    )

    # MUTATION 1: the pre-fix spelling. Different string, and not a file — so the pass above is a
    # comparison and the refusal it replaced is reproduced here rather than described.
    prefix_stem = C.rollout_stem(set_name, "vllm")
    assert prefix_stem != stem, "the pre-fix and fixed stems are the same string"
    assert not (rdir / f"{prefix_stem}.jsonl").exists(), (
        "the fixture accidentally contains the pre-fix path, so its absence proves nothing"
    )

    # MUTATION 2: remove the tagged product. The untagged file is STILL there and must not be
    # accepted in its place; the refusal has to name it, because reaching for it is the mistake.
    for p in (rdir / f"{tagged}.jsonl", rdir / f"{tagged}.summary.json"):
        p.unlink()
    try:
        B.read_nla_rollouts(nla, set_name, root, tag)
    except AssertionError as e:
        msg = str(e)
        assert f"--run-tag {tag!r}" in msg, f"the refusal does not name the run tag: {msg}"
        assert f"{untagged}.jsonl" in msg, (
            f"the refusal does not name the untagged product it declined to read: {msg}"
        )
    else:
        raise AssertionError(
            "the untagged rollouts of another generation run were accepted under a --run-tag"
        )
    # ...and the untagged READ itself still works, so the refusal is about the TAG and not about
    # the fixture being unreadable.
    bare, _ = B.read_nla_rollouts(nla, set_name, root, "")
    assert sorted(r["row"] for r in bare) == [0, 1], bare

    # THE SIBLING call site: scores_dir, where `sae_self.json` is read from.
    full = [k for k in cfg["maemms"]
            if C.split_key(k, "maemm")[0] == base and cfg["maemms"][k]["type"] != "nla"][0]
    assert B.engine_of(cfg, nla, "vllm", quiet=True) == B.NLA_ENGINE, "an nla maemm is HF-spelled"
    assert B.engine_of(cfg, full, "vllm", quiet=True) == "vllm", "a MAEMM keeps its --engine"
    assert B.engine_of(cfg, full, "hf", quiet=True) == "hf"
    sd = C.scores_dir(nla, set_name, root, B.engine_of(cfg, nla, "vllm", quiet=True), tag)
    assert sd.endswith(f"/scores/{set_name}__{tag}"), sd
    assert sd != C.scores_dir(nla, set_name, root, "vllm", tag), (
        "the vLLM default and the NLA engine give the same scores/ path, so the sibling fix is "
        "not exercised by this check"
    )
    print(f"[selfcheck] NLA rollout stem OK: tagged HF chunk read, untagged refused by name, "
          f"scores/ at {os.path.basename(sd)}")


def check_examples_resolution(cfg, tmp: Path, base: str):
    """`build.resolve_examples` finds the scan of a `--with-set` call, and the legacy dir refuses.

    THE DEFECT, 2026-09-23. `scan --set 2026-09-21_v3_realact --with-set ...,2026-09-21_v3_ctrl`
    writes ONE examples product, named after the scan's own `--set`:
    `examples/2026-09-21_v3_realact__paper0923`. A `--set 2026-09-21_v3_ctrl` build looked for
    `examples/2026-09-21_v3_ctrl__paper0923`, did not find it, and `common.sae_examples_dir`'s
    reader fell back to the LEGACY unkeyed `examples/` -- September's `2026-09-16_v1` scan of a
    different set -- with a stdout note as the only trace. A C16 arm over another set's features
    is a plausible number about the wrong thing.

    Pinned here, on a synthetic `sae/<sae>/examples/` tree and nothing else (this is about which
    DIRECTORY is chosen, and the whole build is exercised by `check_two_corpora`):

      * the preferred name wins outright when it is there, and the row map is its own;
      * the with-set sibling is found by FEATURE COVER, and the row map comes from ITS tested.json
        -- the scan re-indexes rows across banks, so a sibling's records carry the scan's row and
        checking them against the set's own would fire on every one;
      * a sibling that does NOT cover this set's features is not resolved into;
      * two covering siblings at one key refuse rather than picking one;
      * the LEGACY unkeyed directory refuses, and the refusal NAMES the key that was expected;
      * with neither keyed nor legacy present the keyed path comes back absent, which is what lets
        a 2M-SAE build fall back to `examples_4m` instead of dying.
    """

    root = str(tmp / "exres")
    sae_key = C.sae_key_for(cfg, base, f"{base}/l42-1b")
    parent = f"{C.sae_dir(sae_key, root)}/examples"
    key, this_set, other = "train_parity_10m__paper0923", "v3_ctrl", "v3_realact"
    feats, my_rows = [11, 22, 33], [1024, 1025, 1026]

    def write(name, features, rows, sae=sae_key):
        d = f"{parent}/{name}"
        Path(d).mkdir(parents=True, exist_ok=True)
        json.dump({"features": features, "rows": rows, "sae": sae},
                  open(f"{d}/tested.json", "w"))
        return d

    def call(set_name=this_set, corpus_key=key):
        return B.resolve_examples(sae_key, set_name, root, corpus_key, feats, "shown")

    checks = mut = 0
    # (1) nothing at all: the keyed path, absent, so `build` falls back rather than dying
    d, row_of, how = call()
    assert how == "absent" and row_of == {} and d.endswith(f"/examples/{this_set}__{key}"), (d, how)
    checks += 1

    # (2) the with-set sibling, named after the OTHER bank, covering this set's features
    sib = write(f"{other}__{key}", [7, *feats, 99], [0, *my_rows, 2047])
    d, row_of, how = call()
    assert d == sib and how.startswith("with-set sibling"), (d, how)
    assert row_of == {7: 0, 11: 1024, 22: 1025, 33: 1026, 99: 2047}, row_of
    checks += 1

    # (3) the PREFERRED name wins outright once it exists -- no search, so an ambiguity among the
    #     other banks cannot reach it
    own = write(f"{this_set}__{key}", feats, [0, 1, 2])
    d, row_of, how = call()
    assert d == own and how == "preferred" and row_of == {11: 0, 22: 1, 33: 2}, (d, how, row_of)
    checks += 1
    shutil.rmtree(own)

    # (4) A TAG-ONLY KEY CONSUMES ITS SUFFIX WHOLE. `sib`, still on disk, is
    #     `<other>__train_parity_10m__paper0923` and it ENDS WITH `__paper0923` -- but the
    #     tag-only key `paper0923` names the scan of the UNSUFFIXED corpus, a different product.
    #     `endswith` matched both, the pair refused as ambiguous (2026-09-23), and the test side
    #     of the eval-1 autointerp run was left with no examples at all.
    tag = "paper0923"
    plain = write(f"{other}__{tag}", feats, my_rows)
    d, row_of, how = call(corpus_key=tag)
    assert d == plain and how.startswith("with-set sibling"), (d, how)
    assert row_of == dict(zip(feats, my_rows, strict=True)), row_of
    checks += 1

    # MUTATION: with the tag-only scan gone the longer-key sibling does NOT answer in its place
    shutil.rmtree(plain)
    d, _row_of, how = call(corpus_key=tag)
    assert how == "absent", (
        f"a `<set>__<corpus>__{tag}` scan answered for the tag-only key {tag!r}, which names the "
        f"scan of the unsuffixed corpus ({d}, how={how!r})"
    )
    mut += 1

    # MUTATION: a sibling that does not cover this set's features is not this call's product
    shutil.rmtree(sib)
    write(f"{other}__{key}", [7, 11, 99], [0, 1, 2])
    d, _row_of, how = call()
    assert how == "absent", (
        f"a sibling missing features {sorted(set(feats) - {7, 11, 99})} was resolved into ({d})"
    )
    mut += 1

    # MUTATION: a sibling at ANOTHER key is never this key's product, however well it covers
    write(f"{other}__other_corpus__paper0923", feats, my_rows)
    assert call()[2] == "absent", "a sibling at another corpus key was resolved into"
    mut += 1

    # MUTATION: two covering siblings at one key -- refuse, do not pick
    shutil.rmtree(f"{parent}/{other}__{key}")
    write(f"{other}__{key}", feats, my_rows)
    write(f"v3_ours__{key}", feats, my_rows)
    try:
        call()
    except AssertionError as e:
        assert "choose between" in str(e), str(e)
        mut += 1
    else:
        raise AssertionError("two candidate example scans did not refuse")
    shutil.rmtree(f"{parent}/v3_ours__{key}")
    shutil.rmtree(f"{parent}/{other}__{key}")

    # MUTATION: the LEGACY unkeyed directory is refused, by name and by the key it should carry
    json.dump({"features": feats, "rows": my_rows, "sae": sae_key},
              open(f"{parent}/tested.json", "w"))
    try:
        call()
    except AssertionError as e:
        msg = str(e)
        assert "LEGACY" in msg and this_set in msg and key in msg, msg
        mut += 1
    else:
        raise AssertionError("the legacy unkeyed examples/ was read instead of refused")
    # ... and it is refused for the OTHER set too: the fallback had no set in it to check
    try:
        call(set_name=other)
    except AssertionError as e:
        assert "LEGACY" in str(e), str(e)
        mut += 1
    else:
        raise AssertionError("the legacy fallback still fires for some set")

    print(f"[selfcheck] examples resolution OK: {checks} checks, {mut} mutation gates; the "
          f"with-set sibling is found by feature cover and the legacy unkeyed dir refuses")


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
        check_nla_body_tokens()
        check_relative_marking()
        check_nla_arms(cfg, tmp, base)
        check_scores_subset(tmp)
        check_corpus_fallback()
        check_two_corpora(cfg, tmp, base)
        check_products_set(cfg, tmp, base)
        check_corpus_key(cfg, base)
        check_nla_rollout_stem(cfg, tmp, base)
        check_examples_resolution(cfg, tmp, base)
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
