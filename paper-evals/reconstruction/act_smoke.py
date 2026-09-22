#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "rich>=13"]
# ///
"""How hard does a target feature actually fire, on each kind of text that claims to describe it?

Local, CPU, no GPU, no model, no network. Reads a LOCAL MIRROR of the volume (`--data <dir>`,
whose subpaths are the volume's own), so whoever fetched the files decides what is there.

    uv run paper-evals/reconstruction/act_smoke.py --data ~/mirror --out out/act_smoke.md
    uv run paper-evals/reconstruction/act_smoke.py --selftest

THE QUESTION. On the 2M SAE the MAEMM's rollouts reach ~26% of a feature's corpus peak. Is that a
property of the MAEMM, or of the dictionary? The only way to tell from here is to measure the SAME
ratio on the 131k dictionary with the same estimator, which is what this script does: for each
target feature it reports the RAW pre-gate peak activation of that feature on

    maemm     the MAEMM's own rollouts             (sae_self over that MAEMM's scores)
    nla       the NLA verbalizer's texts           (sae_self over the NLA entry's scores)
    corpus    the corpus-search top windows        (examples_4m / examples_docmax)

against `corpus_peak`, the feature's maximum over the 16M corpus scan (`sae/<sae>/max_act.f16`).
Three numbers per (feature, source): the peak, peak / corpus_peak, and the fraction of that
source's items with ANY token above the SAE's learned gate.

WHAT IS AND IS NOT COMPARABLE. The two SAEs' rows are not matched on anything: different
dictionaries, different held-out draws, different corpus-scan products (the 2M side has only the
4M-prefix `examples_4m`, the 131k side has the 16M `examples_docmax`), and different rollout
counts (n = 4 against n = 64). The rollout count is the one that biases a peak, since a max over
64 samples beats a max over 4, so the 131k MAEMM is ALSO reported as `maemm_bo4` -- its first four
rollouts only -- and that is the row to read against the 2M side. The corpus difference is not
corrected for and is stated in the output: a 4M-prefix scan searches a quarter of the text a 16M
one does, which can only push the 2M side's corpus peak DOWN.

Nothing here recomputes an activation. Every number is read from what `sae_self` and `scan`
already measured on the clean base, so this script cannot disagree with them about what fired.

A missing file is reported as `absent`, never as a zero -- a source nobody ran and a source that
never fired are opposite findings and must not print the same.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

# How many of a feature's stored corpus windows count as "the top windows an arm would show".
# 16 is `autointerp.n_examples`: the question is what the text an explainer SEES reaches, not what
# the whole stored pool does, and the pool includes the deliberately-weak band rows.
CORPUS_TOP_N = 16

# The two comparisons, laid out rather than inferred. Each `source` is (label, kind, path
# template, extra), where the template is formatted with the spec's own fields and joined onto
# `--data`. `n_first` restricts a sae_self source to its first k rollouts (the matched-n row).
SPECS = [
    {
        "sae": "qwen36-27b/sae2m",
        "base": "qwen36-27b",
        "set": "2026-09-20_sae2m_2k",
        # Two roots, because the smoke was run in two batches over disjoint rows. A feature is
        # looked for in each; the first root that has it wins, and the root is recorded per row.
        "roots": ["tmp/nla-smoke", "tmp/nla-smoke-x"],
        "sae_dir": "base/qwen36-27b/sae/sae2m",
        "corpus": ("examples_4m", "the 4M-prefix scan -- a QUARTER of the 16M the peak is from"),
        "sources": [
            ("nla", "sae_self", "maemms/qwen36-27b/2026-07-14_nla-av/scores/{set}/sae_self", None),
            (
                "maemm",
                "sae_self",
                "maemms/qwen36-27b/2026-09-18_rl-last16-lr5e-7/scores/{set}/sae_self",
                None,
            ),
        ],
    },
    {
        "sae": "qwen36-27b/l42-1b",
        "base": "qwen36-27b",
        "set": "2026-09-16_v1",
        "roots": ["tmp/nla-smoke", ""],
        "sae_dir": "base/qwen36-27b/sae/l42-1b",
        "corpus": ("examples_docmax", "the 16M document-diverse scan, one window per document"),
        "sources": [
            ("nla", "sae_self", "maemms/qwen36-27b/2026-07-14_nla-av/scores/{set}/sae_self", None),
            # The OLD PRIMARY, through vLLM, at n = 64 over all 512 sae rows.
            (
                "maemm",
                "sae_self",
                "maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/{set}__vllm/sae_self",
                None,
            ),
            # The same product restricted to its first 4 rollouts: the only row comparable with
            # the 2M side, whose MAEMM ran at n = 4.
            (
                "maemm_bo4",
                "sae_self",
                "maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/{set}__vllm/sae_self",
                4,
            ),
        ],
    },
]
SOURCE_ORDER = ("maemm", "maemm_bo4", "nla", "corpus")

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------------------------
# readers -- every one of them tolerates an absent source and says so
# ---------------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def read_array(path: str | Path, dtype: str, shape):
    """`common.read_array` without importing the package: this script runs standalone."""
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_sae_self(d: str | Path):
    """(meta, act [N, n, W]) of a `sae_self/` product, or None when it is not in the mirror.

    `sae_self.f16` is the PRE-GATE per-token activation of each row's own target feature on its
    own rollouts, NaN outside the kept tokens; `width` is the run's scoring window + 1 (the NLA
    arm scores at 256, every other arm at 95), absent on a product written before that was
    per-run. Both come from `autointerp/sae_self.py`'s own writer.
    """
    d = Path(d)
    meta_path = d / "sae_self.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path) as fh:
        meta = json.load(fh)
    width = int(meta.get("width", 96))
    shape = (len(meta["rows"]), int(meta["n"]), width)
    act = read_array(d / "sae_self.f16", "float16", shape).astype(np.float32)
    return meta, act


def peaks_of(act_row: np.ndarray) -> np.ndarray:
    """[n] per-rollout peak activation from one row's [n, W] block, NaN outside `keep` -> 0.

    A rollout with nothing kept (an empty generation) peaks at 0.0 and counts as not firing,
    which is what it is -- not a missing measurement.
    """
    finite = np.where(np.isfinite(act_row), act_row, -np.inf)
    pk = finite.max(axis=1)
    return np.where(np.isfinite(pk), pk, 0.0)


def corpus_windows(path: str | Path, top_n: int = CORPUS_TOP_N):
    """[k] peak activations of a feature's top corpus windows, or None when the file is absent.

    `kind == "top"` rows are the ranked top windows where a file has them (`examples_4m`); a
    document-ranked file (`examples_docmax`) has no such label, so all of its rows are candidates.
    Either way they are re-sorted here by the stored `max_act` and cut at `top_n`, so the number
    is "what the strongest windows an arm would show reach", not "what the whole stored pool
    reaches" -- the pool deliberately includes weak band rows.
    """
    if not os.path.exists(path):
        return None
    rows = read_jsonl(path)
    tops = [r for r in rows if r.get("kind") == "top"] or rows
    vals = sorted((float(r["max_act"]) for r in tops), reverse=True)
    return np.array(vals[:top_n], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------------------------


def summarise(vals: np.ndarray | None, gate: float, corpus_peak: float) -> dict:
    """One source's three numbers for one feature. `None` in -> `{"absent": True}` out."""
    if vals is None or not len(vals):
        return {"absent": True}
    peak = float(np.max(vals))
    return {
        "absent": False,
        "n_items": int(len(vals)),
        "peak": round(peak, 4),
        "ratio": (round(peak / corpus_peak, 4) if corpus_peak > 0 else None),
        "fired_frac": round(float(np.mean(vals > gate)), 4),
        "fired_any": bool(peak > gate),
    }


def collect(spec: dict, data: Path, want_rows: set[int] | None) -> dict:
    """Every feature of one SAE's comparison, as {feature: {...}} plus the spec's own metadata.

    Each source is looked for under every root in the spec, in order; the first root that carries
    the feature wins and is recorded, which is how a smoke split across two roots is read back as
    one table without concatenating anything by hand.
    """
    out: dict = {
        "sae": spec["sae"],
        "set": spec["set"],
        "corpus_product": spec["corpus"][0],
        "corpus_note": spec["corpus"][1],
        "roots": [],
        "gate": None,
        "features": [],
        "missing": [],
    }
    # ---- the sae_self products, per root -----------------------------------------------------
    loaded: dict[str, list[tuple[str, dict, np.ndarray, int | None]]] = {}
    gates: set[float] = set()
    for root in spec["roots"]:
        rdir = data / root if root else data
        for label, kind, tmpl, n_first in spec["sources"]:
            assert kind == "sae_self", f"unknown source kind {kind!r}"
            meta, act = load_sae_self(rdir / tmpl.format(set=spec["set"]))
            if meta is None:
                out["missing"].append(f"{label} @ {root or '<root>'}: no sae_self.json")
                continue
            loaded.setdefault(label, []).append((root or "<root>", meta, act, n_first))
            gates.add(round(float(meta["gate"]), 6))
            if root not in out["roots"]:
                out["roots"].append(root or "<root>")
    assert len(gates) <= 1, (
        f"{spec['sae']}: the sae_self products disagree on the SAE gate {sorted(gates)} -- they "
        f"are not all the same dictionary, and their activations are not comparable"
    )
    gate = float(next(iter(gates))) if gates else float("nan")
    out["gate"] = None if math.isnan(gate) else round(gate, 6)

    # ---- the held-out rows: feature id and stratum ---------------------------------------------
    ids: dict[int, dict] = {}
    for root in spec["roots"]:
        rdir = data / root if root else data
        p = rdir / f"base/{spec['base']}/heldout/{spec['set']}/ids.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                ids.setdefault(int(r["row"]), r)
    if not ids:
        out["missing"].append(f"no ids.jsonl for {spec['set']} under any root")

    # ---- corpus peaks ---------------------------------------------------------------------------
    peak_tab = None
    for root in spec["roots"]:
        p = data / (root or ".") / spec["sae_dir"] / "max_act.f16"
        if p.exists():
            peak_tab = read_array(p, "float16", (-1,)).astype(np.float32)
            break
    if peak_tab is None:
        out["missing"].append(f"no {spec['sae_dir']}/max_act.f16 under any root")

    # ---- the rows to report ----------------------------------------------------------------------
    rows_seen: dict[int, str] = {}
    for _label, entries in loaded.items():
        for root, meta, _act, _nf in entries:
            for r in meta["rows"]:
                rows_seen.setdefault(int(r), root)
    rows = sorted(r for r in rows_seen if want_rows is None or r in want_rows)

    for row in rows:
        rec = ids.get(row, {})
        feat = int(rec.get("id", -1))
        cpeak = float(peak_tab[feat]) if (peak_tab is not None and 0 <= feat < len(peak_tab)) else 0.0
        entry = {
            "row": row,
            "feature": feat,
            "stratum": rec.get("stratum"),
            "sae_key": rec.get("sae_key"),  # present only on sets drawn after 2026-09-21
            "family": rec.get("family"),
            "root": rows_seen[row],
            "corpus_peak": round(cpeak, 4),
            "sources": {},
        }
        for label, entries in loaded.items():
            vals = None
            for _root, meta, act, n_first in entries:
                idx = {int(r): i for i, r in enumerate(meta["rows"])}
                if row not in idx:
                    continue
                block = act[idx[row]]
                if n_first:
                    block = block[:n_first]
                vals = peaks_of(block)
                break
            entry["sources"][label] = summarise(vals, gate, cpeak)
        # corpus is per-feature files rather than one array, so it is resolved here
        cvals = None
        for root in spec["roots"]:
            rdir = data / root if root else data
            p = rdir / spec["sae_dir"] / spec["corpus"][0] / spec["set"] / f"{feat}.jsonl"
            cvals = corpus_windows(p)
            if cvals is not None:
                break
        entry["sources"]["corpus"] = summarise(cvals, gate, cpeak)
        out["features"].append(entry)
    return out


def aggregate(block: dict) -> dict:
    """Median / mean ratio and firing rate per source, over the features that HAVE that source."""
    agg: dict[str, dict] = {}
    for label in SOURCE_ORDER:
        have = [f["sources"][label] for f in block["features"] if label in f["sources"]]
        have = [s for s in have if not s["absent"]]
        if not have:
            continue
        ratios = np.array([s["ratio"] for s in have if s["ratio"] is not None], dtype=float)
        agg[label] = {
            "n_features": len(have),
            "median_ratio": (round(float(np.median(ratios)), 4) if len(ratios) else None),
            "mean_ratio": (round(float(np.mean(ratios)), 4) if len(ratios) else None),
            "median_peak": round(float(np.median([s["peak"] for s in have])), 4),
            "frac_features_firing": round(float(np.mean([s["fired_any"] for s in have])), 4),
            "mean_item_fired_frac": round(float(np.mean([s["fired_frac"] for s in have])), 4),
        }
    return agg


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------


def _cell(s: dict, key: str) -> str:
    if s.get("absent"):
        return "absent"
    v = s.get(key)
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render(blocks: list[dict]) -> str:
    lines = [
        "# act_smoke — raw peak activation of each target feature, by source",
        "",
        "Every number is READ from what `sae_self` and `scan` already measured on the clean base; "
        "nothing here recomputes an activation. `ratio` is peak / `corpus_peak`, the feature's "
        "maximum over the 16M corpus scan (`sae/<sae>/max_act.f16`). `fired` is the fraction of "
        "that source's items with ANY token above the SAE's learned gate. An absent source prints "
        "`absent`, never 0.",
        "",
    ]
    for b in blocks:
        present = [
            s
            for s in SOURCE_ORDER
            if any(s in f["sources"] and not f["sources"][s]["absent"] for f in b["features"])
        ]
        lines += [
            f"## {b['sae']} — set `{b['set']}`",
            "",
            f"- gate: {b['gate']}",
            f"- corpus source: `{b['corpus_product']}` — {b['corpus_note']}, "
            f"top {CORPUS_TOP_N} windows by stored `max_act`",
            f"- roots read: {', '.join(b['roots']) or '(none)'}",
            f"- features: {len(b['features'])}",
        ]
        if b["missing"]:
            lines += [f"- MISSING (reported, not counted as zero): {'; '.join(b['missing'])}"]
        lines += [""]
        head = ["feature", "row", "stratum", "corpus_peak"]
        for s in present:
            head += [f"{s} peak", f"{s} ratio", f"{s} fired"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for f in b["features"]:
            cells = [str(f["feature"]), str(f["row"]), str(f["stratum"]), f"{f['corpus_peak']:.3f}"]
            for s in present:
                src = f["sources"].get(s, {"absent": True})
                cells += [_cell(src, "peak"), _cell(src, "ratio"), _cell(src, "fired_frac")]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "### Summary",
            "",
            "| source | features | median ratio | mean ratio | median peak | "
            "features firing | mean item fired-frac |",
            "|---|---|---|---|---|---|---|",
        ]
        for s, a in b["aggregate"].items():
            lines.append(
                f"| {s} | {a['n_features']} | {a['median_ratio']} | {a['mean_ratio']} | "
                f"{a['median_peak']} | {a['frac_features_firing']} | {a['mean_item_fired_frac']} |"
            )
        lines += [""]
    lines += [
        "## Reading these two tables against each other",
        "",
        "They are NOT matched. Different dictionaries, different held-out draws, different corpus "
        "scans (4M prefix against 16M document-diverse) and different rollout counts. The rollout "
        "count is the one that biases a peak — a max over 64 samples beats a max over 4 — so "
        "`maemm_bo4` is the 131k row to read against the 2M `maemm`. The corpus difference is not "
        "corrected for and pushes the 2M side's corpus peak DOWN, i.e. its ratios UP.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------


def analyse(data: Path, sae: str = "", rows: str = "") -> list[dict]:
    want = None
    if rows:
        want = set()
        for part in rows.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                want.update(range(int(a), int(b) + 1))
            else:
                want.add(int(part))
    blocks = []
    for spec in SPECS:
        if sae and spec["sae"] != sae:
            continue
        b = collect(spec, data, want)
        b["aggregate"] = aggregate(b)
        blocks.append(b)
    assert blocks, f"--sae {sae!r} matched none of {[s['sae'] for s in SPECS]}"
    return blocks


@app.command()
def main(
    data: Annotated[Path | None, typer.Option(help="local mirror; its subpaths are the volume's own")] = None,
    out: Annotated[Path | None, typer.Option(help="markdown out; <out>.json lands beside it")] = None,
    sae: Annotated[str, typer.Option(help="restrict to one SAE config key")] = "",
    rows: Annotated[str, typer.Option(help="restrict to these held-out rows, e.g. 0,1,4-6")] = "",
    selftest: Annotated[bool, typer.Option(help="run the synthetic-mirror checks and exit")] = False,
):
    if selftest:
        run_selftest()
        return
    assert data is not None, "--data <dir> is required (the local mirror of the volume)"
    blocks = analyse(data, sae, rows)
    md = render(blocks)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        with open(out.with_suffix(out.suffix + ".json"), "w") as fh:
            json.dump(blocks, fh, indent=1)
        print(f"[act_smoke] wrote {out} and {out.with_suffix(out.suffix + '.json')}")
    for b in blocks:
        print(f"\n== {b['sae']} ({b['set']}), {len(b['features'])} features, gate {b['gate']}")
        for s, a in b["aggregate"].items():
            print(
                f"   {s:<11} median ratio {a['median_ratio']}  median peak {a['median_peak']}  "
                f"features firing {a['frac_features_firing']}  (n={a['n_features']})"
            )
        for m in b["missing"]:
            print(f"   MISSING {m}")


# ---------------------------------------------------------------------------------------------
# --selftest: the whole pipeline on a synthetic mirror, numbers checked by hand
# ---------------------------------------------------------------------------------------------


def _write_sae_self(d: Path, rows_: list[int], n: int, width: int, gate: float, act: np.ndarray):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "sae_self.json", "w") as fh:
        json.dump(
            {"rows": rows_, "n": n, "gate": gate, "width": width, "per_target": [{"row": r} for r in rows_]},
            fh,
        )
    act.astype(np.float16).tofile(d / "sae_self.f16")


def _write_examples(d: Path, feat: int, vals: list[float], kind: str | None = "top"):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{feat}.jsonl", "w") as fh:
        for i, v in enumerate(vals):
            row = {
                "row": 0,
                "window": i,
                "doc": i,
                "start": 0,
                "len": 4,
                "max_act": v,
                "argmax": 0,
                "acts": [v, 0.0, 0.0, 0.0],
            }
            if kind is not None:
                row["kind"] = kind
            fh.write(json.dumps(row) + "\n")


def run_selftest() -> None:
    gate = 2.0
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        spec = {
            "sae": "t/sae",
            "base": "b",
            "set": "S",
            "roots": ["r1", "r2"],
            "sae_dir": "base/b/sae/sae",
            "corpus": ("examples_4m", "synthetic"),
            "sources": [
                ("nla", "sae_self", "maemms/nla/scores/{set}/sae_self", None),
                ("maemm", "sae_self", "maemms/m/scores/{set}/sae_self", None),
                ("maemm_bo4", "sae_self", "maemms/m/scores/{set}/sae_self", 2),
            ],
        }
        # row 0 lives in r1, row 7 in r2 -- the split-root case the real smoke has.
        for root, rows_, blocks in (
            (
                "r1",
                [0],
                np.array([[[1.0, 5.0, np.nan], [0.5, 0.5, np.nan], [9.0, 0.0, np.nan], [0.1, 0.1, np.nan]]]),
            ),
            (
                "r2",
                [7],
                np.array([[[3.0, 0.0, np.nan], [0.0, 0.0, np.nan], [0.0, 0.0, np.nan], [0.0, 0.0, np.nan]]]),
            ),
        ):
            _write_sae_self(data / root / "maemms/m/scores/S/sae_self", rows_, 4, 3, gate, blocks)
            # the NLA product exists in r1 only: r2's must come back `absent`, not 0
            if root == "r1":
                _write_sae_self(data / root / "maemms/nla/scores/S/sae_self", rows_, 4, 3, gate, blocks * 0.5)
            hd = data / root / "base/b/heldout/S"
            hd.mkdir(parents=True, exist_ok=True)
            with open(hd / "ids.jsonl", "w") as fh:
                for r in rows_:
                    fh.write(
                        json.dumps(
                            {"row": r, "family": "sae", "sae_key": "t/sae", "id": 100 + r, "stratum": r % 4}
                        )
                        + "\n"
                    )
        # corpus_peak table: feature 100 peaks at 10.0, feature 107 at 4.0
        tab = np.zeros(200, dtype=np.float16)
        tab[100], tab[107] = 10.0, 4.0
        (data / "r1" / "base/b/sae/sae").mkdir(parents=True, exist_ok=True)
        tab.tofile(data / "r1" / "base/b/sae/sae/max_act.f16")
        # 18 windows for feature 100 so the CORPUS_TOP_N cut is exercised; none for 107
        _write_examples(data / "r1" / "base/b/sae/sae/examples_4m/S", 100, [8.0, 7.0] + [0.1] * 16)

        block = collect(spec, data, None)
        block["aggregate"] = aggregate(block)
        by_row = {f["row"]: f for f in block["features"]}
        assert sorted(by_row) == [0, 7], sorted(by_row)
        assert block["gate"] == gate, block["gate"]

        f0 = by_row[0]
        assert f0["feature"] == 100 and f0["corpus_peak"] == 10.0, f0
        assert f0["root"] == "r1" and f0["sae_key"] == "t/sae", f0
        # maemm: per-rollout peaks are 5, 0.5, 9, 0.1 -> peak 9, ratio 0.9, 2 of 4 over the gate
        m = f0["sources"]["maemm"]
        assert (m["peak"], m["ratio"], m["fired_frac"], m["n_items"]) == (9.0, 0.9, 0.5, 4), m
        # maemm_bo4: FIRST TWO rollouts only -> peaks 5, 0.5 -> peak 5, one of two fires
        b4 = f0["sources"]["maemm_bo4"]
        assert (b4["peak"], b4["ratio"], b4["fired_frac"], b4["n_items"]) == (5.0, 0.5, 0.5, 2), b4
        # nla: the same block halved -> peak 4.5, and 2.5 / 4.5 are over a gate of 2
        nl = f0["sources"]["nla"]
        assert (nl["peak"], nl["ratio"], nl["fired_frac"]) == (4.5, 0.45, 0.5), nl
        # corpus: 18 stored windows, cut to CORPUS_TOP_N by max_act -> peak 8, 2 of 16 fire
        cp = f0["sources"]["corpus"]
        assert (cp["peak"], cp["ratio"], cp["n_items"]) == (8.0, 0.8, CORPUS_TOP_N), cp
        assert cp["fired_frac"] == round(2 / CORPUS_TOP_N, 4), cp

        f7 = by_row[7]
        assert f7["feature"] == 107 and f7["root"] == "r2", f7
        assert f7["sources"]["nla"]["absent"] is True, "r2 has no NLA product: absent, not 0"
        assert f7["sources"]["corpus"]["absent"] is True, "feature 107 has no examples file"
        # a source that exists and never fires is NOT absent: peak 3 > gate on 1 of 4
        assert f7["sources"]["maemm"] == {
            "absent": False,
            "n_items": 4,
            "peak": 3.0,
            "ratio": 0.75,
            "fired_frac": 0.25,
            "fired_any": True,
        }, f7["sources"]["maemm"]

        agg = block["aggregate"]
        assert agg["maemm"]["n_features"] == 2 and agg["nla"]["n_features"] == 1, agg
        assert agg["maemm"]["median_ratio"] == round((0.9 + 0.75) / 2, 4), agg["maemm"]
        assert agg["corpus"]["n_features"] == 1 and agg["corpus"]["median_ratio"] == 0.8, agg

        md = render([block])
        for needle in ("absent", "maemm_bo4 ratio", "t/sae", "gate: 2.0", CORPUS_TOP_N and "top 16"):
            assert needle in md, f"the rendered table is missing {needle!r}"
        assert "| 100 | 0 | 0 | 10.000 |" in md, md[:1200]

        # a gate disagreement between two products of the "same" SAE must be refused
        _write_sae_self(data / "r2" / "maemms/nla/scores/S/sae_self", [7], 4, 3, 99.0, np.zeros((1, 4, 3)))
        try:
            collect(spec, data, None)
        except AssertionError as e:
            assert "disagree on the SAE gate" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("collect accepted two different gates for one SAE")
    print(
        "[act_smoke] selftest OK: split roots, bo4 restriction, absent vs zero, "
        "corpus top-N cut, aggregates, the rendered table and the gate guard"
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_selftest()
    else:
        app()
