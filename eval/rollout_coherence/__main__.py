"""CLI: python -m eval.rollout_coherence <stage|all> (--run-id R | --output-dir D) [--judge-profile P]
[--judge-budget-usd B] [--smoke] [--force] [--device] [--shard k/n] [--merge] [--context-methods]"""

import argparse, os, sys, time

from eval.common import judges as _judges

DEFAULT_JUDGE_PROFILE = "sonnet"

# The profile is read before the config is imported, since the config binds its judge at import.
if __name__ == "__main__":
    _judges.activate_from_argv(sys.argv[1:], default=DEFAULT_JUDGE_PROFILE)

from eval.rollout_coherence import config as C
from eval.common.runs import (
    config_constants,
    invocation_args,
    refuse_profile_switch,
    write_config_once,
    write_provenance,
)
from eval.rollout_coherence.runs import RunDir, stage_names
from eval.rollout_coherence.stages import DEPENDS, STAGES, levels

# Where --run-id resolves. There is no default run id, so no invocation spends against a run nobody named.
DEFAULT_OUTPUT_BASE = "eval/out/rollout_coherence"


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m eval.rollout_coherence")
    ap.add_argument("stage", choices=STAGES + ["all"], help="one stage, or `all` in dependency order")
    ap.add_argument("--output-dir", default=None, help="exact run directory; mutually exclusive with --run-id")
    ap.add_argument("--run-id", default=None, help=f"run name under {DEFAULT_OUTPUT_BASE}; mutually exclusive with --output-dir")
    ap.add_argument("--judge-budget-usd", type=float, default=C.JUDGE_BUDGET_USD,
                    help="the cap every judged stage spends against, in one ledger")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    # Selectors of one part of the fanned-out `frontier_context`. The last two are stored under a leading
    # underscore, which keeps them out of config.json: they say what one invocation does, not what the run is.
    ap.add_argument("--shard", default="0/1", help="the retrieval forward: this part, k/n")
    ap.add_argument("--merge", dest="_merge", action="store_true",
                    help="the retrieval forward's merge call: refuse rather than wait on a missing part")
    ap.add_argument("--context-methods", dest="_context_methods", default=",".join(C.CONTEXT_METHODS),
                    help="frontier_context: which of " + ",".join(C.CONTEXT_METHODS))
    _judges.add_profile_argument(ap, default=DEFAULT_JUDGE_PROFILE)
    args = ap.parse_args(argv)
    if args.output_dir is None and args.run_id is None:
        ap.error("one of --run-id or --output-dir is required: every invocation names the run it writes to")
    return args


def resolved_run_dir(args):
    """--output-dir is the exact run directory; --run-id names a directory under DEFAULT_OUTPUT_BASE;
    supplying both, or neither, is an error."""
    if args.output_dir is not None and args.run_id is not None:
        raise ValueError("supply an exact --output-dir or a --run-id, not both")
    if args.output_dir is not None:
        return args.output_dir
    if args.run_id is None:
        raise ValueError("supply an exact --output-dir or a --run-id")
    return os.path.join(DEFAULT_OUTPUT_BASE, args.run_id)


def run_stage(name, args, run):
    # Imported here: only the process that runs a stage pays for torch or matplotlib.
    from eval.rollout_coherence import (documents, fluency, frontier_context, frontier_corpus,
                                        frontier_pairs, judge, model, report)

    table = {
        "prepare": documents.stage_prepare,
        "capture": model.stage_capture,
        "frontier_corpus": frontier_corpus.stage_frontier_corpus,
        "frontier_context": frontier_context.stage_frontier_context,
        "frontier_context_pairs": frontier_pairs.stage_frontier_context_pairs,
        "frontier_context_judge": judge.stage_frontier_context_judge,
        "frontier_context_fluency": fluency.stage_frontier_context_fluency,
        "report": report.stage_report,
    }
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    write_provenance(run, {"invocation_args": invocation_args(args)}, stage=name)
    try:
        table[name](args, run)
    except Exception as e:
        raise RuntimeError(f"stage {name} failed: {e}") from e
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


def _ever_completed(run, stage):
    """Has `stage` completed at least once here, as one record or as a set of per-job records
    (`stages/frontier_context_<method>.json`)?"""
    if _completed(run, stage):
        return True
    names = stage_names(run, [f"{stage}_"])
    return bool(names) and all(_completed(run, n) for n in names)


def _completed(run, name):
    rel = f"stages/{name}.json"
    return run.exists(rel) and run.read_json(rel).get("completed") is True


def _refuse_if_never_run(run, stage, args):
    """For a single named stage without --force: refuse, before config.json is written, when a stage it
    depends on has never completed. Each stage's own gate checks the dependency's hash as well."""
    if stage == "all" or getattr(args, "force", False):
        return
    missing = [up for up in DEPENDS.get(stage, []) if not _ever_completed(run, up)]
    if missing:
        raise RuntimeError(f"{stage}: stage(s) {missing} have not completed for this run directory; "
                           f"run them first (or pass --force to bypass this check)")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _judges.activate(args.judge_profile)   # raises if this process bound its config under another
    run = RunDir(resolved_run_dir(args))
    refuse_profile_switch(run, C.JUDGE_PROFILE)
    _refuse_if_never_run(run, args.stage, args)
    write_config_once(run, config_constants(C), args)
    write_provenance(run, {"judge_profile": C.JUDGE_PROFILE})
    names = [s for lvl in levels() for s in lvl] if args.stage == "all" else [args.stage]
    for name in names:
        run_stage(name, args, run)


if __name__ == "__main__":
    main()
    # A hard exit: a tokenizer or model thread can abort the interpreter at teardown (rc -6), which the
    # launcher would read as a failed stage. Every artifact is already closed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
