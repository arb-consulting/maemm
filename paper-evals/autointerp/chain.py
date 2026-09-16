"""Stage `chain`: the whole remaining autointerp run as ONE detached Modal call.

    <root>/runs/<chain>/STATUS.json   rewritten and committed at every stage boundary
    <root>/runs/<chain>/pilot.md      the 64-feature pilot's tables
    <root>/runs/<chain>/results.md    the 512-feature primary run's tables
    <root>/runs/<chain>/costs.json    every stage's cost, and the grand total

Why this exists: every earlier stage was launched from a local client and watched from the
session, so a session that ended left the chain unattended. This function owns the whole sequence
instead, on a 12 h timeout, and the only thing anyone has to look at afterwards is `STATUS.json`.

The sequence, each step gated on the one before:

    wait for `examples_docmax` -> build the 64-feature pilot on it -> run the pilot (sync) ->
    ACCEPTANCE CHECKS -> build all 512 -> primary run (--approved) -> build rlI-150 ->
    rlI-150 on M and C4+M -> stats for both -> done

The acceptance checks are the coordinator's, and a failure ABORTS the chain with the reason in
STATUS.json rather than spending the full run's budget on a broken test set:

  * the floor arm's mean balanced accuracy is within [0.42, 0.58] -- a floor that scores above
    that is reading the test set rather than the description, and the whole comparison is void;
  * every test positive fires at the gate (A1), asserted from the build's own records;
  * the build's document-level disjointness assertions passed (they raise inside `build`, so
    reaching this point is the check) and no test item shares a document with a shown example;
  * C4 is filled at 16 on every feature (A3);
  * draw shortfalls are RECORDED -- they do not abort, because the design's answer to a short
    pool is to record it, but they are surfaced in STATUS.json.

The API path for the two full runs is decided by MEASUREMENT, not by a flag: a small Message Batch
is timed end to end and the full runs use the batch path (half price) only if it came back inside
`BATCH_OK_S`.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import precompute.common as C

DOCMAX_POLL_S = 60.0
DOCMAX_MAX_WAIT_S = 3 * 3600
# A batch that comes back inside this is worth half price; one that does not would make the two
# full runs unpredictable, so they go on the sync path instead.
BATCH_OK_S = 1800.0
BATCH_PROBE_N = 8
FLOOR_LO, FLOOR_HI = 0.42, 0.58


class Status:
    """STATUS.json, rewritten and committed at every stage boundary."""

    def __init__(self, path: str, on_commit=None, meta: dict | None = None):
        self.path = path
        self.on_commit = on_commit
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # A restart must not lose the earlier stages' record, so an existing STATUS is read back
        # and continued rather than truncated.
        prior = {}
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    prior = json.load(fh)
            except json.JSONDecodeError:
                prior = {}
        self.doc = {
            **prior,
            "state": "starting",
            "restarts": int(prior.get("restarts", 0)) + (1 if prior else 0),
            "stages": list(prior.get("stages", [])),
            "started": prior.get("started") or time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "costs_usd": dict(prior.get("costs_usd", {})),
            "checks": dict(prior.get("checks", {})),
            **(meta or {}),
        }
        self.t0 = time.time()
        self.write()

    def write(self):
        self.doc["updated"] = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
        self.doc["elapsed_s"] = round(time.time() - self.t0, 1)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self.doc, fh, indent=1)
        os.replace(tmp, self.path)
        if self.on_commit:
            self.on_commit()
        print(f"[chain] STATUS {self.doc['state']}", flush=True)

    def stage(self, name: str, **kw):
        self.doc["state"] = name
        self.doc["stages"].append({"stage": name, "at": time.strftime("%H:%M:%SZ", time.gmtime()),
                                   **kw})
        self.write()

    def fail(self, why: str, **kw):
        self.doc["state"] = "FAILED"
        self.doc["error"] = why
        self.doc.update(kw)
        self.write()

    def done(self, **kw):
        self.doc["state"] = "done"
        self.doc.update(kw)
        self.write()


def build_if_needed(cfg, args, st: Status, name: str, maemm: str, build_dir: str, n_feat: int):
    """Run `build` unless its output is already there.

    MEASURED 2026-09-16: a Modal container polling a batch was SIGTERMed at 1223 s and Modal
    RE-SCHEDULED the input. A chain that simply restarted would hit `OutDir`'s refusal to overwrite
    an existing product and die on its own earlier success, so every step of this chain has to be
    idempotent. The LLM stages already are, through the prompt cache; the builds are made so here.
    """
    out_dir = (f"{args['root']}/base/{args['base']}/autointerp/{args['heldout']}/{build_dir}")
    if os.path.exists(f"{out_dir}/build.json"):
        info = json.load(open(f"{out_dir}/build.json"))
        st.stage(f"{name}_present", build_dir=build_dir, features=info.get("n_features"))
        return {"out": out_dir, "reused": True, **{k: info.get(k) for k in
                ("n_features", "n_short_draw1", "n_short_draw2", "n_empty_draw2", "n_short_c4")}}
    st.stage(name)
    from autointerp import build as B

    return B.run(cfg, _sub(args, maemm=maemm, build_dir=build_dir, n_feat=n_feat))


def _sub(args: dict, **over) -> dict:
    """Args for one sub-stage: the chain's own container fields, plus the stage's flags.

    `t0` is reset per sub-stage so each product's README records its OWN wall, not the chain's.
    """
    out = {**args, **over, "t0": time.time()}
    out.pop("dry_run", None)
    return out


def wait_for_docmax(cfg, args, st: Status) -> str:
    """Block until `examples_docmax` has landed on the volume, reloading to see other containers."""
    base, root, set_name = args["base"], args["root"], args["heldout"]
    keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    path = f"{C.sae_dir(keys[0], root)}/examples_docmax/{set_name}"
    marker = f"{path}/tested.json"
    reload_ = args.get("on_reload")
    t0 = time.time()
    while not os.path.exists(marker):
        if time.time() - t0 > DOCMAX_MAX_WAIT_S:
            raise TimeoutError(
                f"{marker} did not appear within {DOCMAX_MAX_WAIT_S / 3600:.1f} h; the "
                f"`examples_docmax` scan is not going to land and the chain must not build a test "
                f"set without it"
            )
        print(f"[chain] waiting for {marker} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(DOCMAX_POLL_S)
        if reload_:
            reload_()
    st.stage("docmax_present", path=path, waited_s=round(time.time() - t0, 1))
    return path


def batch_probe(cfg, args) -> tuple[str, float]:
    """Time a small Message Batch end to end and pick the path for the two full runs."""
    from autointerp.run import Cache, Claude

    key = os.environ.get("ANTHROPIC_API_KEY")
    assert key, "no ANTHROPIC_API_KEY: the `anthropic` secret must be mounted on this function"
    cl = Claude(cfg["autointerp"]["scorer_model"], key)
    cache = Cache(f"{args['root']}/runs/{args['chain_dir']}/probe_cache", on_commit=args.get("on_commit"))
    jobs = [
        {"key": f"probe|batch|{i}", "system": "You are a test harness. Reply with the word OK.",
         "user": f"Reply with exactly the word OK. (probe {i})", "max_tokens": 16}
        for i in range(BATCH_PROBE_N)
    ]
    todo = [j for j in jobs
            if cache.get(Cache.key(j["key"], cl.params(j["system"], j["user"], j["max_tokens"])))
            is None]
    if not todo:
        return "batch", 0.0
    t0 = time.time()
    # The probe must ABANDON at the threshold, not discover the answer by waiting past it.
    # MEASURED 2026-09-16: an 8-request probe sat `in_progress` for 2097 s while `run_batch`
    # blocked, because the batch had not ENDED -- the chain would have waited up to Anthropic's
    # 24 h batch SLA to learn that batch latency is unacceptable.
    got, info = cl.run_batch(todo, "batch-probe", max_wait_s=BATCH_OK_S)
    for j in todo:
        rec = got.get(j["key"])
        if rec is not None:
            cache.put(Cache.key(j["key"], cl.params(j["system"], j["user"], j["max_tokens"])), rec)
    wall = time.time() - t0
    path = "batch" if wall <= BATCH_OK_S and len(got) == len(todo) else "sync"
    if info.get("abandoned"):
        path = "sync"
    print(f"[chain] batch probe: {len(got)}/{len(todo)} in {wall:.0f}s "
          f"(threshold {BATCH_OK_S:.0f}s) -> full runs use `{path}`", flush=True)
    return path, wall


def acceptance(build_info: dict, scores: list[dict], floor_arm: str, st: Status) -> tuple[bool, dict]:
    """The coordinator's gate. Returns (ok, report); a False aborts before the full run's budget."""
    import numpy as np

    rep: dict = {}
    floor = [r["bal_acc"] for r in scores
             if r["arm"] == floor_arm and r["scorer"] == "detection" and r["bal_acc"] is not None]
    rep["floor_arm"] = floor_arm
    rep["floor_n"] = len(floor)
    rep["floor_mean"] = round(float(np.mean(floor)), 4) if floor else None
    rep["floor_ok"] = bool(floor) and FLOOR_LO <= float(np.mean(floor)) <= FLOOR_HI
    rep["floor_window"] = [FLOOR_LO, FLOOR_HI]

    rep["n_short_c4"] = int(build_info.get("n_short_c4", 0))
    rep["c4_ok"] = rep["n_short_c4"] == 0
    rep["n_short_draw1"] = int(build_info.get("n_short_draw1", 0))
    # Draw 1 is the set EVERY arm is scored on, so a feature with no draw-1 positive is a feature
    # missing from every contrast. Gated, not merely recorded: MEASURED on the first pilot, 7 of 64
    # -- all in the rarest density quartile -- had none, which at 512 projects to ~55 lost q0
    # features and would quietly hollow out the per-quartile table.
    rep["n_no_pos_draw1"] = int(build_info.get("n_no_pos_draw1", 0))
    rep["no_pos_draw1_by_stratum"] = build_info.get("no_pos_draw1_by_stratum", {})
    rep["n_top_fallback_features"] = int(build_info.get("n_top_fallback_features", 0))
    rep["n_top_fallback_positives"] = int(build_info.get("n_top_fallback_positives", 0))
    rep["n_short_draw2"] = int(build_info.get("n_short_draw2", 0))
    rep["n_empty_draw2"] = int(build_info.get("n_empty_draw2", 0))
    rep["min_pos_draw1"] = int(build_info.get("min_pos_draw1", 0))
    rep["min_pos_draw2"] = int(build_info.get("min_pos_draw2", 0))
    # Gate-consistency and document-level disjointness are ASSERTED inside `build`, which raises;
    # reaching this point at all is the check passing. Recorded so STATUS says so explicitly.
    rep["gate_consistent_positives"] = bool(build_info.get("gate_consistent_positives"))
    rep["disjointness_asserted_in_build"] = True
    rep["n_features"] = int(build_info.get("n_features", 0))
    # A draw that could not be filled is RECORDED, not fatal: the design's answer to a short pool
    # is to record it. An EMPTY draw 2 on more than a third of features is fatal, because the null
    # is then not measurable and every contrast loses its threshold.
    rep["draw2_ok"] = rep["n_empty_draw2"] <= max(1, rep["n_features"] // 3)
    rep["draw1_ok"] = rep["n_no_pos_draw1"] <= max(1, rep["n_features"] // 20)
    rep["draw1_window"] = f"<= {max(1, rep['n_features'] // 20)} of {rep['n_features']}"
    ok = bool(rep["floor_ok"] and rep["c4_ok"] and rep["draw1_ok"] and rep["draw2_ok"]
              and rep["gate_consistent_positives"])
    rep["ok"] = ok
    st.doc["checks"] = rep
    st.stage("acceptance", **{k: rep[k] for k in ("ok", "floor_mean", "n_short_c4",
                                                  "n_no_pos_draw1", "n_empty_draw2",
                                                  "n_short_draw1")})
    return ok, rep


def write_tables(run_name: str, root: str, out_path: str, label: str) -> str | None:
    """Run `autointerp/stats.py` against a finished run, in-process and OFFLINE.

    Its `Vol` reads `<data_dir>/<relative path>`, and inside this container the volume IS mounted
    at `<root>`, so `--no-fetch --data-dir <root>` reads the products directly with no `modal
    volume get` and no network.
    """
    import autointerp.stats as S

    try:
        S.main(run=run_name, out=out_path, data_dir=root, modal_cmd="uvx modal",
               refetch=False, no_fetch=True, label=label)
    except Exception as e:  # noqa: BLE001 -- a missing table must not lose a finished run
        print(f"[chain] stats for {run_name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None
    return out_path


def run(cfg, args):
    from autointerp import run as R

    root = args["root"]
    chain_dir = args.get("chain_dir") or f"{time.strftime('%Y-%m-%d')}_autointerp-chain"
    base = f"{root}/runs/{chain_dir}"
    st = Status(
        f"{base}/STATUS.json",
        on_commit=args.get("on_commit"),
        meta={"chain_dir": chain_dir, "base": args["base"], "set": args["heldout"],
              "repo_commit": args.get("repo_commit", ""), "argv": args.get("argv", [])},
    )
    costs: dict[str, float] = {}
    out: dict[str, dict] = {}
    primary = args.get("maemm") or "qwen36-27b/2026-09-10_rl-8x2048-full"
    secondary = args.get("maemm2") or "qwen36-27b/2026-09-08_rlI-150"
    floor_arm = str(cfg["autointerp"]["floor_arm"])
    pilot_dir = f"{chain_dir}_pilot"
    full_dir = f"{chain_dir}_full"
    rlI_dir = f"{chain_dir}_rlI"

    try:
        wait_for_docmax(cfg, args, st)

        out["build_pilot"] = build_if_needed(cfg, args, st, "build_pilot", primary, pilot_dir, 0)
        st.doc["build_pilot"] = out["build_pilot"]
        st.write()

        st.stage("run_pilot", path="sync")
        out["run_pilot"] = R.run(cfg, _sub(
            args, maemm=primary, build_dir=pilot_dir, run_dir=pilot_dir, path="sync",
            arms="", approved=True, max_cost_usd=80.0))
        costs["pilot"] = out["run_pilot"]["cost_this_call"]
        st.doc["run_pilot"] = out["run_pilot"]
        st.doc["costs_usd"] = costs
        st.write()

        scores = C.read_jsonl(f"{root}/runs/{pilot_dir}/summary/scores.jsonl")
        binfo = json.load(open(f"{root}/base/{args['base']}/autointerp/{args['heldout']}"
                               f"/{pilot_dir}/build.json"))
        ok, rep = acceptance(binfo, scores, floor_arm, st)
        write_tables(pilot_dir, root, f"{base}/pilot.md", "pilot")
        if not ok:
            st.fail(f"acceptance checks failed: {json.dumps(rep)}")
            return {"chain": chain_dir, "status": "FAILED", "checks": rep, "costs": costs}

        path, probe_wall = batch_probe(cfg, _sub(args, chain_dir=chain_dir))
        st.stage("batch_probe", path=path, wall_s=round(probe_wall, 1), threshold_s=BATCH_OK_S)

        out["build_full"] = build_if_needed(cfg, args, st, "build_full", primary, full_dir, 512)
        st.doc["build_full"] = out["build_full"]
        st.write()

        st.stage("run_primary", path=path)
        out["run_primary"] = R.run(cfg, _sub(
            args, maemm=primary, build_dir=full_dir, run_dir=full_dir, path=path,
            arms="C16,C4,M,C4M,C32,C16M16", approved=True, max_cost_usd=300.0))
        costs["primary"] = out["run_primary"]["cost_this_call"]
        st.doc["run_primary"] = out["run_primary"]
        st.doc["costs_usd"] = costs
        st.write()
        write_tables(full_dir, root, f"{base}/results.md", "results")

        out["build_rlI"] = build_if_needed(cfg, args, st, "build_rlI", secondary, rlI_dir, 512)
        st.doc["build_rlI"] = out["build_rlI"]
        st.write()

        st.stage("run_rlI", path=path)
        out["run_rlI"] = R.run(cfg, _sub(
            args, maemm=secondary, build_dir=rlI_dir, run_dir=rlI_dir, path=path,
            arms="M,C4M", approved=True, max_cost_usd=120.0))
        costs["rlI"] = out["run_rlI"]["cost_this_call"]
        st.doc["run_rlI"] = out["run_rlI"]
        st.doc["costs_usd"] = costs
        st.write()
        write_tables(rlI_dir, root, f"{base}/results-rlI.md", "results-rlI")

        with open(f"{base}/costs.json", "w") as fh:
            json.dump({"per_run": costs, "total_usd": round(sum(costs.values()), 4),
                       "path": path, "batch_probe_s": round(probe_wall, 1),
                       "runs": {k: v.get("out") for k, v in out.items() if isinstance(v, dict)}},
                      fh, indent=1)
        st.done(costs_usd=costs, total_usd=round(sum(costs.values()), 4), path=path,
                tables=[f"{base}/pilot.md", f"{base}/results.md", f"{base}/results-rlI.md"])
        return {"chain": chain_dir, "status": "done", "costs": costs,
                "total_usd": round(sum(costs.values()), 4), "path": path, "checks": rep}
    except Exception as e:  # noqa: BLE001 -- the chain's whole point is to record why it stopped
        traceback.print_exc()
        st.fail(f"{type(e).__name__}: {str(e)[:500]}", traceback=traceback.format_exc()[-4000:])
        raise
