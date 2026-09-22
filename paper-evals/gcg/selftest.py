#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.10.0", "transformers==5.15.0", "numpy==2.4.6", "pyyaml"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""CPU unit smoke for the M3 discrete-search changes. No GPU, no weights, no volume, no network.

    uv run paper-evals/gcg/selftest.py

WHAT IS COVERED, and why each one is here rather than in `precompute/unit_smoke.py`: these are
`gcg/`'s own invariants, and `precompute/` belongs to M0a.

  1. the chunk's file names are a function of the SELECTION, not of how --rows was spelled;
  2. a kept staging directory is found, repaired and carried -- and is never deleted;
  3. only WHOLE directions are carried (all pop members, all three streams);
  4. the collision gate refuses each of the three shapes it exists for;
  5. a committed chunk is reused rather than re-run, and a DIFFERENT call is not;
  6. `exact_cos` hands `common.score_ids` both centred arguments and reduces the second cosine the
     same way it reduces the first;
  7. the scoring mean comes from one place, and prefers M0a's function the moment it exists.

EVERY CHECK MUTATES ITS OWN INPUT and asserts the gate fires: a gate that has never failed has
never been shown to be a gate (plan §1).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import precompute.common as C  # noqa: E402
from gcg import gcg as G  # noqa: E402


def _fires(fn, *a, **kw) -> str:
    """Run `fn` and return the AssertionError it must raise. A pass here is a FAILED check."""
    try:
        fn(*a, **kw)
    except AssertionError as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did NOT refuse: the gate is not a gate")


def _final(row, member=0, pop=1, **kw):
    d = {"row": row, "family": "realact", "member": member, "lam": 0.0, "cos": 0.5 + 0.01 * row,
         "nll": 3.0, "iters": 5, "seq_len": 32, "mode": "gcg", "init": "random32"}
    d.update(kw)
    return d


def _stage(d: Path, files: dict, rows: list[int], pop: int = 1, torn: bool = False):
    """Write a staging directory holding `rows` as whole directions, optionally with a torn tail."""
    d.mkdir(parents=True, exist_ok=True)
    fin = [_final(r, m, pop) for r in rows for m in range(pop)]
    C.write_jsonl(d / files["finals"], fin)
    for k in ("trajectory", "top64"):
        C.write_jsonl(d / files[k], [{"row": r, "family": "realact"} for r in rows])
    if torn:
        with open(d / files["finals"], "a") as fh:
            fh.write('{"row": 99, "family": "rea')
    return d


# ---------------------------------------------------------------------------------------------


def check_rows_spec_is_a_function_of_the_selection():
    """`--rows 0-3` and `--rows 3,1,0,2` are ONE chunk and write ONE set of file names.

    The file name is what stops two containers overwriting each other, and it is also what lets a
    retry find its own partial. Building it from the typed string instead of the parsed selection
    would give `0-3` and `0,1,2,3` two names for one chunk: the retry would write a second copy
    beside the first and the reader would double-count every direction in it.
    """
    for spec, want in (("0-3", "0-3"), ("3,1,0,2", "0-3"), ("7", "7"), ("0-2,5", "0-2,5"),
                       ("5,0,1,2", "0-2,5")):
        got = G.rows_spec_of(C.parse_rows(spec, 64))
        assert got == want, f"--rows {spec!r} -> {got!r}, want {want!r}"
    assert G.chunk_files("0-3")["finals"] == "finals__rows0-3.jsonl"
    assert G.chunk_files("0-2,5")["summary"] == "summary__rows0-2_5.json"
    # a whole-family run keeps the spelling every product on the volume already has
    assert G.chunk_files("") == {
        "finals": "finals.jsonl", "trajectory": "trajectory.jsonl",
        "top64": "top64.jsonl", "summary": "summary.json",
    }
    # MUTATION: a spec that is not a row spec is not silently turned into a file name
    msg = _fires(G.chunk_files, "0-3; rm -rf")
    assert "row spec" in msg, msg
    print("  rows_spec: the chunk's name is the selection, not the spelling")


def check_a_partial_is_carried_and_never_deleted():
    """A kept staging directory is found, its torn tail dropped, and the directory LEFT IN PLACE."""
    files = G.chunk_files("0-3")
    with tempfile.TemporaryDirectory() as td:
        arm = Path(td) / "epo-random32-paper0923"
        thin = _stage(arm.with_name(arm.name + ".tmp-2026-09-23-111-aaa"), files, [0])
        fat = _stage(arm.with_name(arm.name + ".tmp-2026-09-23-222-bbb"), files, [0, 1, 2],
                     torn=True)
        carry = G.find_partial(str(arm), files, pop=1)
        assert carry, "a staging directory with three whole directions was not found"
        rows = {int(f["row"]) for f in C.read_jsonl(str(Path(carry) / files["finals"]))}
        assert rows == {0, 1, 2}, f"carried {sorted(rows)}, want [0, 1, 2] (the torn line dropped)"
        # NOTHING IS DELETED: that is the whole point of the change.
        assert thin.is_dir() and fat.is_dir(), "the carry deleted a staging directory"
        assert (fat / files["finals"]).is_file()

        # MUTATION 1: take the top64 row of direction 2 away -> it is no longer a WHOLE direction
        C.write_jsonl(fat / files["top64"], [{"row": r} for r in (0, 1)])
        carry2 = G.find_partial(str(arm), files, pop=1)
        rows2 = {int(f["row"]) for f in C.read_jsonl(str(Path(carry2) / files["finals"]))}
        assert rows2 == {0, 1}, f"a half-written direction was carried: {sorted(rows2)}"

        # MUTATION 2: this chunk's files are not there at all -> nothing to carry, no crash
        for k in ("finals", "trajectory", "top64"):
            (thin / files[k]).unlink()
            (fat / files[k]).unlink()
        assert G.find_partial(str(arm), files, pop=1) == "", "carried something from nothing"
    print("  carry: whole directions only, torn tail dropped, staging dirs kept")


def check_a_partial_of_a_population_needs_every_member():
    """pop 3: a direction with two of its three members written is not a whole direction."""
    files = G.chunk_files("0-1")
    with tempfile.TemporaryDirectory() as td:
        arm = Path(td) / "epo-random32-x"
        st = _stage(arm.with_name(arm.name + ".tmp-2026-09-23-1-a"), files, [0, 1], pop=3)
        carry = G.find_partial(str(arm), files, pop=3)
        assert {int(f["row"]) for f in C.read_jsonl(str(Path(carry) / files["finals"]))} == {0, 1}
        # MUTATION: drop one member of direction 1
        fin = [f for f in C.read_jsonl(str(st / files["finals"]))
               if not (f["row"] == 1 and f["member"] == 2)]
        C.write_jsonl(st / files["finals"], fin)
        carry = G.find_partial(str(arm), files, pop=3)
        got = {int(f["row"]) for f in C.read_jsonl(str(Path(carry) / files["finals"]))}
        assert got == {0}, f"a direction missing a member was carried: {sorted(got)}"
    print("  carry: a direction is whole only with every pop member")


def check_torn_in_the_middle_is_not_tolerated():
    """A file torn anywhere but at its end is a different failure and is refused, not repaired."""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "finals.jsonl"
        f.write_text('{"row": 0}\nNOT JSON\n{"row": 1}\n')
        msg = _fires(G.read_jsonl_tolerant, f, "test")
        assert "not the last line" in msg, msg
        f.write_text('{"row": 0}\n{"row": 1}\n{"row": 2')
        assert [r["row"] for r in G.read_jsonl_tolerant(f, "test")] == [0, 1]
    print("  tolerant read: a torn TAIL is dropped, a torn middle is refused")


def check_the_collision_gate_refuses_all_three_shapes():
    """The gate that replaces OutDir's refusal for an additive, chunked write."""
    files = G.chunk_files("0-3")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "arm"
        d.mkdir()
        G.assert_writable(str(d), files, True, False)  # empty directory: fine
        # 1. this exact chunk is already committed
        (d / files["finals"]).write_text("")
        msg = _fires(G.assert_writable, str(d), files, True, False)
        assert "this exact chunk" in msg, msg
        G.assert_writable(str(d), files, True, True)  # ... and --force says so out loud
        (d / files["finals"]).unlink()
        # 2. a chunk beside the arm's whole-set product
        (d / "finals.jsonl").write_text("")
        msg = _fires(G.assert_writable, str(d), files, True, False)
        assert "WHOLE-SET product" in msg, msg
        (d / "finals.jsonl").unlink()
        # 3. a whole-set run beside chunks
        (d / files["finals"]).write_text("")
        msg = _fires(G.assert_writable, str(d), G.chunk_files(""), False, False)
        assert "chunks" in msg, msg
        # a SIBLING chunk is not a collision -- that is the whole point of chunking
        G.assert_writable(str(d), G.chunk_files("4-7"), True, False)
    print("  collision gate: three refusals, and a sibling chunk is not one of them")


def check_a_committed_chunk_is_reused_not_rerun():
    """A retry after a dropped client returns the committed product; a DIFFERENT call does not."""
    files = G.chunk_files("0-1")
    a = {"mode": "gcg", "init": "random32", "iters": 5, "pop": 1, "children": 512, "seq_len": 32,
         "topk": 512}
    summary = {
        "arm": "gcg-random32", "family": "realact", "n_directions": 2, "rows": [0, 1],
        "config": dict(a), "mean_final_cos": 0.5, "mean_init_cos": 0.1, "mean_nll": 3.0,
        "alphabet_size": 100, "sae": None,
        "totals": {"cand_forwards": 10, "filter_reject_rate": 0.1,
                   "cos_check_same_batch_max": 1e-6, "cos_check_rebatch_max": 1e-6},
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "arm"
        d.mkdir()
        assert G.existing_chunk(str(d), files, [0, 1], a) is None, "reused a product that is absent"
        (d / files["summary"]).write_text(json.dumps(summary))
        got = G.existing_chunk(str(d), files, [0, 1], a)
        assert got and got["n_directions_run"] == 0 and got["reused"], got
        # MUTATION 1: the same rows at a DIFFERENT configuration is a different experiment
        assert G.existing_chunk(str(d), files, [0, 1], {**a, "iters": 300}) is None
        # MUTATION 2: the same configuration over different rows is a different chunk
        assert G.existing_chunk(str(d), files, [0, 2], a) is None
    print("  reuse: the same call is returned, a different one is not")


def check_exact_cos_asks_for_both_cosines_and_reduces_them_alike():
    """`exact_cos` hands `score_ids` both centred arguments, and `cos_centred` is a max over kept.

    `common.score_ids` owns the formula; what is checked here is gcg's half -- that the pair is
    passed at all (the objective must stay uncentred, so this is the ONLY call that may) and that
    the second cosine is reduced by the same `common.agg` as the first.
    """
    import torch

    d = torch.zeros(4)
    seen = {}

    def fake_score_ids(model, tok, id_lists, dirs, read_layer, **kw):
        seen.update(kw)
        n = len(id_lists)
        keep = torch.zeros((n, 5), dtype=torch.bool)
        keep[:, 1:4] = True
        cos = torch.full((n, 5), float("nan"))
        cos[:, 1:4] = torch.tensor([0.1, 0.7, 0.3])
        cen = torch.full((n, 5), float("nan"))
        cen[:, 1:4] = torch.tensor([0.9, 0.2, 0.4])   # a DIFFERENT argmax on purpose
        out = {"cos": cos, "keep": keep, "norm": torch.ones((n, 5))}
        if kw.get("dirs_centred") is not None:
            out["cos_centred"] = cen
        return out

    real = C.score_ids
    try:
        C.score_ids = fake_score_ids
        # without the pair: the historical call, and no centred column anywhere
        res = G.exact_cos(None, None, [[1, 2, 3]], d, 42, 8, "cpu")
        assert seen["dirs_centred"] is None and seen["mu"] is None, seen
        assert "cos_centred" not in res, "an uncentred call produced a centred column"
        assert abs(float(res["cos"][0]) - 0.7) < 1e-6 and int(res["amax"][0]) == 1
        # with it: both columns, each a max over KEPT positions, argmaxes recorded separately
        mu = np.zeros(4, np.float32)
        res = G.exact_cos(None, None, [[1, 2, 3]], d, 42, 8, "cpu", d_centred_cpu=d, mu=mu)
        assert seen["dirs_centred"] is not None, "the centred direction never reached score_ids"
        assert seen["mu"] is mu, "the scoring mean never reached score_ids"
        assert abs(float(res["cos_centred"][0]) - 0.9) < 1e-6, res["cos_centred"]
        assert int(res["amax_centred"][0]) == 0 and int(res["amax"][0]) == 1, (
            "the two cosines were given one argmax; centring moves the ranking"
        )
    finally:
        C.score_ids = real
    print("  exact_cos: both cosines from one forward, reduced alike, argmaxes apart")


def check_the_scoring_mean_has_exactly_one_source():
    """`score_mu_spec` is the base's whiten_mu -- and M0a's function the moment it exists."""
    cfg = {"bases": {"b": {"whiten_mu": "/vol/archive/x.npy", "d": 4}}}
    assert G.score_mu_spec(cfg, "b") == "/vol/archive/x.npy"
    # M0a's function wins over the fallback, by name, with no edit here
    assert not hasattr(C, G.M0A_SCORE_MU_FN), (
        f"common.{G.M0A_SCORE_MU_FN} EXISTS: M0a's scoring-mean function has landed, so this "
        f"check's premise is stale -- confirm gcg is calling it and update the placeholder note "
        f"in gcg.py:score_mu_spec and CHANGES-pipeline.md"
    )
    try:
        setattr(C, G.M0A_SCORE_MU_FN, lambda cfg, base: "FROM-M0A")
        assert G.score_mu_spec(cfg, "b") == "FROM-M0A", "M0a's function was not preferred"
    finally:
        delattr(C, G.M0A_SCORE_MU_FN)
    # MUTATION: no mean anywhere is a refusal, not a locally computed mean
    msg = _fires(G.score_mu_spec, {"bases": {"b": {"d": 4}}}, "b")
    assert "placeholder" in msg and "whiten_mu" in msg, msg
    print("  score_mu_spec: one source, M0a's function preferred, absence refused")


CHECKS = [
    check_rows_spec_is_a_function_of_the_selection,
    check_a_partial_is_carried_and_never_deleted,
    check_a_partial_of_a_population_needs_every_member,
    check_torn_in_the_middle_is_not_tolerated,
    check_the_collision_gate_refuses_all_three_shapes,
    check_a_committed_chunk_is_reused_not_rerun,
    check_exact_cos_asks_for_both_cosines_and_reduces_them_alike,
    check_the_scoring_mean_has_exactly_one_source,
]


def run_all():
    import time

    for fn in CHECKS:
        t0 = time.time()
        fn()
        print(f"[m3] ok  {fn.__name__:<52} {time.time() - t0:5.2f}s", flush=True)
    print(f"[m3] {len(CHECKS)}/{len(CHECKS)} checks passed", flush=True)


if __name__ == "__main__":
    run_all()
