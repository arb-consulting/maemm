"""CLI: python -m evals.downstream.workspace_understanding <stage> [--output-dir D | --run-id R] [--seed S]
[--judge-budget-usd B] [--judge-profile sonnet|sol] [--smoke] [--force] [--device cuda:0] [--shard k/n] [--merge]"""

import argparse, os, sys, time
from evals.downstream.common import judges as _judges

DEFAULT_PROFILE = "sonnet"  # Claude Sonnet 5, the paper's judge for this package

# Before the config is imported: it binds its judges at import (evals/downstream/common/judges.py).
if __name__ == "__main__":
    _judges.activate_from_argv(sys.argv[1:], default=DEFAULT_PROFILE)

from evals.downstream.workspace_understanding import config as C
from evals.downstream.common.runs import RunDir, config_constants, invocation_args, refuse_profile_switch, write_config_once
from evals.downstream.workspace_understanding.runs import write_provenance
from evals.downstream.workspace_understanding import shards as S
from evals.downstream.workspace_understanding.stages import ALL_STAGES, DEPENDS, levels

# --run-id resolves under this; there is no default run id.
DEFAULT_OUTPUT_BASE = "evals/downstream/out/workspace_understanding"


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m evals.downstream.workspace_understanding")
    ap.add_argument("stage", choices=ALL_STAGES + ["all"])
    ap.add_argument("--output-dir", default=None, help="exact run directory; mutually exclusive with --run-id")
    ap.add_argument(
        "--run-id",
        default=None,
        help=f"run name under {DEFAULT_OUTPUT_BASE}; one of --run-id / --output-dir is required",
    )
    ap.add_argument("--seed", type=int, default=C.GEN_SEED)
    ap.add_argument("--judge-budget-usd", type=float, default=C.JUDGE_BUDGET_USD)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None,
                    help="nla / nla_control: items of shard k of n, as k/n (0-based); "
                         "retrieval: corpus windows of shard k of n, as k/n (0-based)")
    # a leading underscore keeps it out of config.json's `args` block
    ap.add_argument("--merge", dest="_merge", action="store_true",
                    help="retrieval: this call is the designated merge of a sharded corpus forward, and "
                         "fails rather than returns if a part is missing")
    _judges.add_profile_argument(ap, default=DEFAULT_PROFILE)
    return ap.parse_args(argv)


def resolved_run_dir(args):
    """--output-dir is the exact run directory; --run-id names a directory under DEFAULT_OUTPUT_BASE;
    supplying both, or neither, is an error."""
    if args.output_dir is not None and args.run_id is not None:
        raise ValueError("supply an exact --output-dir or a --run-id, not both")
    if args.output_dir is not None:
        return args.output_dir
    if args.run_id is None:
        raise ValueError("supply --run-id <name> or --output-dir <directory>; there is no default run")
    return os.path.join(DEFAULT_OUTPUT_BASE, args.run_id)


def run_stage(name, args, run):
    from evals.downstream.workspace_understanding import corpus, items, judge, lens, maem_control, model
    from evals.downstream.workspace_understanding import nla, nla_control, patchscope, report, reread, retrieval
    from evals.downstream.workspace_understanding import untrained_base

    table = {
        "prepare": items.stage_prepare,
        "corpus": corpus.stage_corpus,
        "capture": model.stage_capture,
        "lens": lens.stage_lens,
        "rollouts": model.stage_rollouts,
        "retrieval": retrieval.stage_retrieval,
        "patchscope": patchscope.stage_patchscope,
        "nla": nla.stage_nla,
        "nla_control": nla_control.stage_nla_control,
        "untrained_base": untrained_base.stage_untrained_base,
        "maem_control": maem_control.stage_maem_control,
        "reread": reread.stage_reread,
        "summarise": judge.stage_summarise,
        "judge": judge.stage_judge,
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


def _refuse_if_never_run(run, stage, args):
    """Refuse a single named stage (not `all`, not --force) whose dependencies never completed, before
    config.json records this invocation. Each stage's own key check is the finer gate."""
    if stage == "all" or getattr(args, "force", False):
        return
    # sharded stages write one record per shard, so completion is read through shards.stage_ever_completed
    missing = [up for up in DEPENDS.get(stage, []) if not S.stage_ever_completed(run, up)]
    if missing:
        raise RuntimeError(f"{stage}: stage(s) {missing} have not completed for this run directory; "
                           f"run them first (or pass --force to bypass this check)")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # raises when this process already bound its config under another profile
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
