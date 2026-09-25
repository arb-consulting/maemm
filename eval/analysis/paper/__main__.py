"""`python -m eval.analysis.paper {steering,workspace,coherence} ... --out DIR`: one section per call."""
import argparse
import os

from . import coherence, steering, steering_figure, workspace


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.analysis.paper")
    sub = parser.add_subparsers(dest="section", required=True)
    s = sub.add_parser("steering", help="tab:steer, fig:steer and the steering numbers")
    s.add_argument("--axbench-run", required=True)
    s.add_argument("--bipo-run", required=True)
    s.add_argument("--judge", default="sol", help="judge whose verdicts the tables read (the paper: sol)")
    s.add_argument("--method-label", default="MAEM", help="the method's name in the figure legend")
    s.add_argument("--skip-lengths", action="store_true",
                   help="skip the mean-token count (~1 GB read without text_length.csv)")
    w = sub.add_parser("workspace", help="tab:workspace-word, tab:workspace-judge and the workspace numbers")
    w.add_argument("--wu-run", required=True)
    w.add_argument("--wm-run", required=True)
    c = sub.add_parser("coherence", help="fig:coherence and the coherence numbers")
    c.add_argument("--run", required=True)
    for p in (s, w, c):
        p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = os.path.join(args.out, args.section)
    if args.section == "steering":
        steering.build(args.axbench_run, args.bipo_run, out, args.judge, lengths=not args.skip_lengths)
        steering_figure.build(args.axbench_run, out, args.judge, args.method_label)
    elif args.section == "workspace":
        workspace.build(args.wu_run, args.wm_run, out)
    else:
        coherence.build(args.run, out)
    print(out)


if __name__ == "__main__":
    main()
