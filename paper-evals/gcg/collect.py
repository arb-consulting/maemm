#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15"]
# ///
"""Read ONE gcg arm back -- whole, or as the union of its `--rows` chunks.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-m3/paper-evals/gcg/collect.py --base qwen36-27b \\
        --set 2026-09-21_v3_realact --arm epo-random32-paper0923 --fetch)

Local, CPU, no GPU, no model, no volume write. Since 2026-09-23 a `--rows` call of `gcg` writes
`finals__rows<spec>.jsonl` and its three siblings into the arm's ONE directory (the additive
product write), so a 64-direction arm run as 16 parallel chunks is 16 file-quadruples in one
place. This is the reader that makes them one product again, and the place the two invariants a
chunked run can violate are checked:

  * the chunks cover DISJOINT directions -- an overlap means two containers searched one direction
    and the union would double-count it;
  * the chunks are ONE experiment -- every field of `_INVARIANT` agrees, so an arm whose second
    half was run at different `--iters` is a refusal and not a mean.

A whole-set `finals.jsonl` beside chunks of the same arm is refused on sight, the same shape
`common.read_rollouts` refuses for rollouts: that is a whole-set run and a chunked run claiming one
product, and silently preferring either is how a partial gets read as if it were complete.

THE NUMBER THIS PRINTS. Per direction, the reported final is the member with the best UNCENTRED
`cos` -- the objective the search actually optimised -- and the printed column is that final's
`cos_centred`, the rescoring through `common.score_ids` under the headline centring (spec §1.4).
Both are carried, with a standard error over directions, because the paper's row is the centred one
and the appendix's run detail is the uncentred one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import precompute.common as C  # noqa: E402

# What every chunk of one arm must agree on: these describe the EXPERIMENT, and two chunks that
# disagree on one of them are two experiments wearing one arm name.
_INVARIANT = ("base", "set", "family", "arm", "read_layer", "score_mu", "mu", "config")
# ... except the parts of `config` that are per-chunk by construction.
_CONFIG_PER_CHUNK = ()


def _summaries(d: Path) -> tuple[list[Path], Path | None]:
    """(the chunk summaries, the whole-set summary or None) in one arm directory."""
    chunks = sorted(d.glob("summary__rows*.json"))
    whole = d / "summary.json"
    return chunks, (whole if whole.is_file() else None)


def load_arm(d: Path) -> dict:
    """Every final of one arm directory, chunked or not, with the two invariants checked."""
    assert d.is_dir(), f"no arm directory at {d}"
    chunks, whole = _summaries(d)
    assert chunks or whole, (
        f"{d} holds neither summary.json nor summary__rows*.json: it is not a gcg arm directory"
    )
    assert not (chunks and whole), (
        f"{d} holds the whole-set product AND {len(chunks)} `--rows` chunk(s) "
        f"({[p.name for p in chunks]}): that is two runs claiming one arm. Keep one and move the "
        f"other aside."
    )
    parts = [whole] if whole else chunks
    finals: list[dict] = []
    seen: dict[int, str] = {}
    head: dict | None = None
    for sp in parts:
        with open(sp) as fh:
            s = json.load(fh)
        if head is None:
            head = s
        else:
            bad = {k: (head.get(k), s.get(k)) for k in _INVARIANT if head.get(k) != s.get(k)}
            assert not bad, (
                f"{sp.name} disagrees with {parts[0].name} on {sorted(bad)}: the chunks of one arm "
                f"must be one experiment. Values: {bad}"
            )
        for r in s["rows"]:
            assert int(r) not in seen, (
                f"direction {r} is in both {seen[int(r)]} and {sp.name}: the chunks of one arm "
                f"must cover DISJOINT directions, or the union double-counts one search"
            )
            seen[int(r)] = sp.name
        fpath = d / s.get("files", {}).get("finals", "finals.jsonl")
        assert fpath.is_file(), f"{sp.name} names {fpath.name}, which is not in {d}"
        rows = C.read_jsonl(str(fpath))
        got = {int(f["row"]) for f in rows}
        assert got == set(int(r) for r in s["rows"]), (
            f"{fpath.name} holds directions {sorted(got)[:6]}... but {sp.name} claims "
            f"{sorted(s['rows'])[:6]}...: the chunk is incomplete"
        )
        finals += rows
    return {"summary": head, "finals": finals, "rows": sorted(seen), "parts": [p.name for p in parts]}


def reduce_arm(arm: dict) -> dict:
    """Per direction: the reported final and its two cosines. Then the arm's means and SEs.

    The reported final is the member with the best UNCENTRED cos, because that is what the search
    selected on; its `cos_centred` is the paper's number. Taking the best centred value over the
    Pareto front instead would be a centred search the run did not do.
    """
    by_row: dict[int, list[dict]] = {}
    for f in arm["finals"]:
        by_row.setdefault(int(f["row"]), []).append(f)
    per = []
    for row in sorted(by_row):
        fs = by_row[row]
        best = max(fs, key=lambda f: f["cos"])
        per.append({
            "row": row,
            "family_row": best.get("family_row"),
            "member": best["member"],
            "lam": best["lam"],
            "cos": float(best["cos"]),
            "cos_centred": (
                None if best.get("cos_centred") is None else float(best["cos_centred"])
            ),
            "nll": float(best["nll"]),
            "members": len(fs),
        })

    def stat(vals):
        v = np.asarray([x for x in vals if x is not None], float)
        if not v.size:
            return {"n": 0, "mean": None, "se": None}
        se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
        return {"n": int(v.size), "mean": float(v.mean()), "se": se}

    return {
        "per_direction": per,
        "cos": stat([p["cos"] for p in per]),
        "cos_centred": stat([p["cos_centred"] for p in per]),
        "nll": stat([p["nll"] for p in per]),
    }


def fetch(vol_dir: str, dest: Path, modal_cmd: str) -> None:
    """Mirror one arm directory off the volume. Read-only; existing local files are replaced."""
    dest.mkdir(parents=True, exist_ok=True)
    rel = vol_dir[len("/vol/"):] if vol_dir.startswith("/vol/") else vol_dir.lstrip("/")
    cmd = [*modal_cmd.split(), "volume", "get", "--force", "maemm", rel, str(dest.parent)]
    print(f"[collect] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    base: Annotated[str, typer.Option(help="config.yaml `bases:` key")] = "qwen36-27b",
    set_: Annotated[str, typer.Option("--set", help="the held-out set")] = "2026-09-21_v3_realact",
    family: Annotated[str, typer.Option(help="the target family")] = "realact",
    arm: Annotated[str, typer.Option(help="`<mode>-<init>` plus any --arm-suffix")] = "",
    dir_: Annotated[Path | None, typer.Option(
        "--dir", help="a local arm directory; skips --base/--set")] = None,
    root: Annotated[str, typer.Option(help="the run root the arm was written under")] = "/vol",
    do_fetch: Annotated[bool, typer.Option(
        "--fetch/--no-fetch", help="mirror the arm off the volume first")] = False,
    mirror: Annotated[Path, typer.Option(help="where --fetch puts the mirror")] = HERE / "data",
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    out: Annotated[Path | None, typer.Option(help="write the per-direction table as json")] = None,
) -> None:
    if dir_ is None:
        assert arm, "--arm <mode>-<init>[-<suffix>] is required unless --dir names the directory"
        vol_dir = C.gcg_dir(base, set_, family, arm, root)
        dir_ = mirror / Path(vol_dir).name
        if do_fetch:
            fetch(vol_dir, dir_, modal_cmd)
    armd = load_arm(Path(dir_))
    red = reduce_arm(armd)
    s = armd["summary"]
    print(f"\n# {s['arm']} on {s['base']}/{s['set']} ({s['family']})\n")
    print(f"- parts: {len(armd['parts'])} ({', '.join(armd['parts'])})")
    print(f"- directions: {len(red['per_direction'])} -> {armd['rows'][:4]}...{armd['rows'][-2:]}")
    print(f"- objective mu: {s.get('mu')} | scoring mu: {s.get('score_mu')}")
    print(f"- config: {s['config']['mode']} pop {s['config']['pop']} x {s['config']['children']} "
          f"x {s['config']['iters']} iters, T={s['config']['seq_len']}\n")
    print("| column | n | mean | se |")
    print("|---|---|---|---|")
    for k, label in (("cos_centred", "cos_centred (PRINTED)"), ("cos", "cos (objective)"),
                     ("nll", "nll")):
        st = red[k]
        m = "--" if st["mean"] is None else f"{st['mean']:.4f}"
        se = "--" if st["se"] is None else f"{st['se']:.4f}"
        print(f"| {label} | {st['n']} | {m} | {se} |")
    if red["cos_centred"]["n"] != len(red["per_direction"]):
        print(
            f"\nWARNING: {len(red['per_direction']) - red['cos_centred']['n']} direction(s) carry "
            f"no cos_centred. A product written before 2026-09-23, or a family with no mean."
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"summary": s, "parts": armd["parts"], **red}, fh, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
