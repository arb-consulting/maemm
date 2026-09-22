#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""`paper/inversion-eval/data/corpus_top1_activation.csv` -- does the corpus search's TOP-1 window
for an SAE feature actually make that feature fire?

One row per (base, sae held-out feature): the window the corpus scan picked by COSINE against
`unit(W_enc[:, f])` at the 16M corpus size, and that window's PRE-GATE activation of the SAME
feature, against the checkpoint's learned BatchTopK gate.

Local, CPU, no model: every number is read back off the volume, where
`precompute/top1_act.py` produced it (`base/<base>/sae/<sae>/top1_act/<set>/top1_act.jsonl`). That
product carries TWO independent activations per row -- `join_*` from the scan's own
`examples/<feature>.jsonl`, and `fwd` from one forward of the same window -- so `source` says which
one this CSV took (`examples-join` where the cosine top-1 is also an activation example of its own
feature, `forward` otherwise) and the join-vs-forward agreement is reported as a check.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-precompute/paper-evals/reconstruction/corpus_top1_activation.py)

Cross-checks, all fatal:
  * the product's top-1 (doc, start, argmax, cos) is re-derived from `scan/<set>/topk.jsonl` rank 0
    at the largest corpus size and must match entry for entry;
  * `feature`, `stratum` and `density` must match `heldout/<set>/ids.jsonl`;
  * where both sources exist they must agree to `--tol` RELATIVE (the two forwards differ only by
    bf16 batch shape -- paper-evals/README.md, "The scorer's batch-shape noise floor").
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
import yaml
from rich.console import Console
from rich.table import Table as RichTable

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
sys.path.insert(0, str(HERE))
from stats import ROOTS, Vol, mirror_dir  # noqa: E402 -- fetcher, root map and mirror rule

# 27B primary first, as every table in reconstruction/ orders them.
BASES = ("qwen36-27b", "qwen3-8b")
SET = "2026-09-16_v1"
# The paper tree is outside this repo; this script is the only writer of that one file.
OUT_DEFAULT = Path(
    "/home/gavento/dev/mimir/2026-09-maemms/paper/inversion-eval/data/corpus_top1_activation.csv"
)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def spearman(x, y) -> float:
    """Spearman rho = Pearson on average ranks (ties averaged), so it is exact under ties."""
    rx, ry = _avg_rank(np.asarray(x, float)), _avg_rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _avg_rank(v):
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(len(v), dtype=float)
    # average the ranks inside each tie group
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def _sae_key(cfg: dict, base: str, want: str = "") -> str:
    """WHICH SAE of `base`: `--sae` when given, else the single one -- refusing to guess with two.

    This is `common.sae_key_for`'s rule, restated rather than imported: this file is a standalone
    `uv run` script with its own dependency block and no paper-evals on sys.path, so it reads
    config.yaml with plain yaml and cannot call into precompute/. The message names the options.
    """
    keys = [k for k in cfg["saes"] if k.split("/", 1)[0] == base]
    assert keys, f"base {base} has no SAE in config.yaml"
    want = (want or "").strip()
    if want:
        assert want in keys, f"--sae {want!r} is not one of base {base}'s SAEs {sorted(keys)}"
        return want
    assert len(keys) == 1, (
        f"base {base} has {len(keys)} SAEs in config.yaml ({sorted(keys)}), so nothing can pick "
        f"one for you: pass --sae <base>/<name>"
    )
    return keys[0]


def _rows_for_base(vol: Vol, cfg: dict, base: str, tol: float, sae_want: str = "") -> tuple[list[dict], dict]:
    """The CSV rows for one base, plus that base's own summary dict. Raises on any disagreement."""
    sae = _sae_key(cfg, base, sae_want if sae_want.split("/", 1)[0] == base else "")
    sae_dir = f"base/{base}/sae/{sae.split('/', 1)[1]}"
    recs = vol.jsonl(f"{sae_dir}/top1_act/{SET}/top1_act.jsonl")
    assert recs, (
        f"{sae_dir}/top1_act/{SET}/top1_act.jsonl is not on the volume: run "
        f"`--product top1_act --base {base} --set {SET}` first"
    )
    summary = vol.json(f"{sae_dir}/top1_act/{SET}/summary.json")
    assert summary is not None, f"{sae_dir}/top1_act/{SET}/summary.json is missing"

    ids = {r["row"]: r for r in (vol.jsonl(f"base/{base}/heldout/{SET}/ids.jsonl") or [])}
    assert ids, f"base/{base}/heldout/{SET}/ids.jsonl is missing"
    size = int(summary["corpus_size_m"])
    # On the ROW's own sae_key where the set has one, not the family label alone (H8). This is a
    # standalone `uv run` script with no paper-evals on sys.path, so it restates the rule rather
    # than importing common.sae_rows_of: a row that names a different dictionary is skipped, and a
    # set whose rows name none is accepted only because `sae` picked it (single-SAE base, or the
    # --sae the caller typed) -- which is the same contract, stated where it is used.
    id_key = {r["row"]: r.get("sae_key") for r in ids.values()}
    topk = {
        r["row"]: r["top"][0]
        for r in (vol.jsonl(f"base/{base}/scan/{SET}/topk.jsonl") or [])
        if r["size"] == size
        and r["family"] in ("sae", "sae2m_enc")
        and id_key.get(r["row"], sae) == sae
    }
    assert topk, f"base/{base}/scan/{SET}/topk.jsonl has no sae rows at size {size}M"

    out, worst_rel = [], 0.0
    for e in recs:
        row = e["row"]
        ref = ids[row]
        assert (int(ref["id"]), ref["stratum"], ref["density"]) == (
            e["feature"],
            e["stratum"],
            e["density"],
        ), f"{base} row {row}: top1_act and ids.jsonl disagree on feature / stratum / density"
        doc, start, argmax, cos = topk[row]
        assert [doc, start, argmax, cos] == [e["doc"], e["start"], e["argmax"], e["top1_cos"]], (
            f"{base} row {row}: top1_act's top-1 {[e['doc'], e['start'], e['argmax'], e['top1_cos']]} "
            f"is not topk.jsonl's rank 0 at {size}M {[doc, start, argmax, cos]}"
        )
        joined = bool(e["joined"])
        act_max = float(e["join_act_max"] if joined else e["act_max"])
        act_at = float(e["join_act_at_argmax"] if joined else e["act_at_argmax"])
        if joined:
            rel = abs(act_max - e["act_max"]) / max(abs(e["act_max"]), 1e-9)
            worst_rel = max(worst_rel, rel)
        gate = float(e["gate"])
        out.append(
            {
                "base": base,
                "feature": e["feature"],
                "sae_row": row,
                "stratum": e["stratum"],
                "density": e["density"],
                "top1_cos": cos,
                "top1_doc": doc,
                "top1_start": start,
                "top1_argmax": argmax,
                "act_max": round(act_max, 5),
                "act_at_argmax": round(act_at, 5),
                "gate": gate,
                "passes_gate": act_max > gate,
                "source": "examples-join" if joined else "forward",
            }
        )
    assert worst_rel <= tol, (
        f"{base}: the examples join and this product's own forward differ by {worst_rel:.2e} "
        f"relative on some row, above --tol {tol:.0e}"
    )
    assert len(out) == 512, f"{base}: {len(out)} sae features, expected 512"
    summary["join_vs_forward_max_rel_diff"] = worst_rel
    return out, summary


def _slice_stats(rows: list[dict]) -> dict:
    a = np.array([r["act_max"] for r in rows])
    g = np.array([r["gate"] for r in rows])
    c = np.array([r["top1_cos"] for r in rows])
    return {
        "n": len(rows),
        "frac_pass_gate": float(np.mean([r["passes_gate"] for r in rows])),
        "median_act_over_gate": float(np.median(a / g)),
        "spearman_cos_act": spearman(c, a),
    }


@app.command()
def main(
    root_tag: Annotated[str, typer.Option(help=f"which volume root: {sorted(ROOTS)}")] = "full",
    out: Annotated[Path, typer.Option(help="the CSV to write")] = OUT_DEFAULT,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what data/ already has")] = False,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    data_dir: Annotated[Path | None, typer.Option(
        help="override the default mirror ($MAEMM_MIRROR, else "
             "$XDG_CACHE_HOME/maemm-paper-evals/mirror/<root>)")] = None,
    tol: Annotated[float, typer.Option(help="max relative examples-join vs forward disagreement")] = 5e-2,
    sae: Annotated[str, typer.Option(help="which SAE, as `<base>/<name>`; needed when a base has two")] = "",
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
):
    cfg = yaml.safe_load((PAPER_EVALS / "config.yaml").read_text())
    vol = Vol(root_tag, data_dir or mirror_dir(ROOTS[root_tag]), modal_cmd, refetch, quiet,
              offline=not fetch)

    rows, summaries = [], {}
    for base in BASES:
        r, s = _rows_for_base(vol, cfg, base, tol, sae)
        rows += r
        summaries[base] = s

    t = RichTable(title=f"corpus top-1 window: does the feature fire there? ({SET}, 16M corpus)")
    for col in ("base", "slice", "n", "src join/fwd", "frac > gate", "median act/gate", "rho(cos, act)"):
        t.add_column(col, justify="right" if col != "base" else "left")
    for base in BASES:
        br = [r for r in rows if r["base"] == base]
        for label, sl in [("all", br)] + [(f"q{q}", [r for r in br if r["stratum"] == q]) for q in range(4)]:
            st = _slice_stats(sl)
            nj = sum(r["source"] == "examples-join" for r in sl)
            t.add_row(
                base if label == "all" else "",
                label,
                str(st["n"]),
                f"{nj}/{st['n'] - nj}",
                f"{st['frac_pass_gate']:.4f}",
                f"{st['median_act_over_gate']:.2f}",
                f"{st['spearman_cos_act']:.4f}",
            )
    console.print(t)
    for base in BASES:
        s = summaries[base]
        console.print(
            f"[dim]{base}: gate {s['gate']:.4f}, examples-join {s['n_joined']}/{s['n_features']} "
            f"({s['n_joined_top']} in the activation top-128), forward-only {s['n_forward_only']}; "
            f"join-vs-forward max rel diff {s['join_vs_forward_max_rel_diff']:.2e}[/dim]"
        )

    df = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"For each SAE held-out feature of {SET}, the corpus scan's COSINE top-1 window at the",
        "16M corpus size and that window's PRE-GATE activation of the SAME feature,",
        "relu((h - b_dec) @ W_enc[:, f] + b_enc[f]) at the read layer, over the window's non-sink",
        "positions (64/16 scan geometry, sink prepended and dropped, clean base).",
        "act_max is the max over the window, act_at_argmax the value at the COSINE argmax token;",
        "gate is the checkpoint's own learned BatchTopK threshold and passes_gate is act_max > gate.",
        "source: `examples-join` = the scan's own examples/<feature>.jsonl entry for that window,",
        "`forward` = precompute/top1_act.py's forward, for windows that are not activation examples.",
        f"The two agree to < {tol:.0e} relative wherever both exist (bf16 batch-shape noise only).",
        f"Written by reconstruction/corpus_top1_activation.py from base/<base>/sae/<sae>/top1_act/{SET}/.",
    ]
    with open(out, "w") as fh:
        fh.write("".join(f"# {ln}\n" for ln in header))
        df.write_csv(fh)
    console.print(f"wrote {out} ({len(df)} rows, {out.stat().st_size / 1e3:.1f} kB)")
    if vol.missing:
        console.print(f"[yellow]{len(vol.missing)} files not on the volume: {vol.missing[:5]}[/yellow]")
    console.print(f"fetched {vol.fetched} files ({vol.bytes / 1e6:.1f} MB) into {vol.local}")


if __name__ == "__main__":
    app()
