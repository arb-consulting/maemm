"""CLI: python -m evals.downstream.workspace_modulation <stage|all> (--run-id R | --output-dir D) [options]; see README."""

import argparse, importlib, os, sys, time
from evals.downstream.common import judges as _judges

# The config binds its judges on import, so the profile is read off argv first.
if __name__ == "__main__":
    _judges.activate_from_argv(sys.argv[1:], default="sonnet")

from evals.downstream.common.runs import RunDir, config_constants, invocation_args, refuse_profile_switch, write_config_once
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation.runs import write_provenance
from evals.downstream.workspace_modulation.stages import ALL_STAGES, DEPENDS, levels, provenance_stage, shard_of

# stage -> (module, function), imported only when the stage runs, so --help needs no GPU packages.
STAGE_MODULES = {
    "prepare": ("items", "stage_prepare"),
    "capture": ("capture", "stage_capture"),
    "cells_mean": ("mean_cell", "stage_cells_mean"),
    "rollouts": ("rollouts", "stage_rollouts"),
    "rollouts_merge": ("rollouts", "stage_rollouts_merge"),
    "patchscope": ("patchscope", "stage_patchscope"),
    "lens": ("lens", "stage_lens"),
    "nla": ("nla_rollouts", "stage_nla"),
    "nla_merge": ("nla_rollouts", "stage_nla_merge"),
    "corpus": ("retrieval", "stage_corpus"),
    "retrieval": ("retrieval", "stage_retrieval"),
    "retrieval_merge": ("retrieval", "stage_retrieval_merge"),
    "summarise": ("summarise", "stage_summarise"),
    "judge": ("judge", "stage_judge"),
    "report": ("report", "stage_report"),
}

DEFAULT_OUTPUT_BASE = "evals/downstream/out/workspace_modulation"


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m evals.downstream.workspace_modulation")
    ap.add_argument("stage", choices=ALL_STAGES + ["all"])
    ap.add_argument("--output-dir", default=None, help="exact run directory; mutually exclusive with --run-id")
    ap.add_argument("--run-id", default=None, help=f"run name under {DEFAULT_OUTPUT_BASE}")
    ap.add_argument("--seed", type=int, default=C.GEN_SEED)
    # None means "the package's own cap" (runs.resolve_judge_budget)
    ap.add_argument("--judge-budget-usd", type=float, default=None)
    ap.add_argument("--gate", type=int, default=0, help="judge: run only the gate, on N requests, and stop")
    ap.add_argument("--retry-truncated", type=int, default=0,
                    help="judge: re-ask only the naming replies cut at their cap, at this max_tokens")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", default=",".join(C.ARM_ORDER), help="comma-separated generation arms to run")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    _judges.add_profile_argument(ap, default=_judges.active())
    args = ap.parse_args(argv)
    if args.output_dir is None and args.run_id is None:
        ap.error("one of --run-id or --output-dir is required")
    return args


def resolved_run_dir(args):
    """--output-dir, or DEFAULT_OUTPUT_BASE/--run-id; exactly one of the two."""
    if args.output_dir is not None and args.run_id is not None:
        raise ValueError("supply an exact --output-dir or a --run-id, not both")
    if args.output_dir is not None:
        return args.output_dir
    if args.run_id is None:
        raise ValueError("supply an exact --output-dir or a --run-id")
    return os.path.join(DEFAULT_OUTPUT_BASE, args.run_id)


def resolve_stage(name):
    """The stage function of STAGE_MODULES[name], imported now."""
    module_name, attr = STAGE_MODULES[name]
    module = importlib.import_module(f"evals.downstream.workspace_modulation.{module_name}")
    return getattr(module, attr)


def stage_table():
    """name -> a wrapper that imports and runs the stage when called."""
    return {name: (lambda args, run, _name=name: resolve_stage(_name)(args, run)) for name in ALL_STAGES}


def run_stage(name, args, run):
    table = stage_table()
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    write_provenance(
        run,
        {"invocation_args": invocation_args(args)},
        stage=provenance_stage(name, *shard_of(args), arms=getattr(args, "arms", "")),
    )
    try:
        table[name](args, run)
    except Exception as e:
        raise RuntimeError(f"stage {name} failed: {e}") from e
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


def _completed_at_least_once(run, stage):
    """True when `stage` has a completed record under its own name or one of its invocations' names
    (`rollouts.shard0of2`, `rollouts.reg`, ...)."""
    d = os.path.join(run.path, "stages")
    if not os.path.isdir(d):
        return False
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        rec = name[:-5]
        if (rec == stage or rec.startswith(stage + ".")) and run.read_json(f"stages/{rec}.json").get("completed"):
            return True
    return False


def _refuse_if_never_run(run, stage, args):
    """A single named stage (not `all`, not --force) refuses before config.json is written when a stage it
    depends on never completed here. Each stage's own key check is the finer gate."""
    if stage == "all" or getattr(args, "force", False):
        return
    missing = [up for up in DEPENDS.get(stage, []) if not _completed_at_least_once(run, up)]
    if missing:
        raise RuntimeError(f"{stage}: stage(s) {missing} have not completed for this run directory; "
                           f"run them first (or pass --force to bypass this check)")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _judges.activate(args.judge_profile)
    run = RunDir(resolved_run_dir(args))
    refuse_profile_switch(run, C.JUDGE_PROFILE)
    _refuse_if_never_run(run, args.stage, args)
    write_config_once(run, config_constants(C), args)
    write_provenance(run)
    for name in ([s for l in levels() for s in l] if args.stage == "all" else [args.stage]):
        run_stage(name, args, run)


if __name__ == "__main__":
    main()
