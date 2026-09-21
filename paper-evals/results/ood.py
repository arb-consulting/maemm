#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9",
#                 "polars>=1", "rich>=13"]   # the last two: reconstruction/stats_ood.py
# ///
"""Eval 3's arms table: the generalisation eval, from the products already on the volume.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run --with fasttext --with huggingface-hub \\
       repo-maemm/paper-evals/results/ood.py --set 2026-09-21_ood_q1)

Local, CPU, no GPU, no model. Every number is READ from what `scan`, `score` and `nll` wrote;
this file recomputes nothing except the per-arm aggregation and its confidence interval.

THE TABLE (design `infra/2026-09-18_ood-eval-design.md` §0, §6; review R3, R9). One row per arm:

  bo64            the MAEMM's unbiased best-of-64, in BOTH cosines -- `bo_c_64` (centred, the
                  headline where the run centred on something) and `bo_64` (raw). `score` writes
                  the centred half only when the run centred, so a `--mu none` arm shows an em
                  dash there, never a one-sided number silently relabelled.
  corpus          the in-domain corpus search's top-1 on the SAME target, at the scanned size.
  delta, CI       Δ_i = bo64_i(MAEMM) − top1_i(in-domain), PAIRED per target, with the design's
                  10,000-resample percentile bootstrap over the arm's targets (the estimator is
                  `reconstruction/stats_ood.boot_ci`, imported rather than rewritten). The
                  clustered SE is carried in the CSV beside it: within one arm every target comes
                  from a distinct pool document (`targets._ood_arm_draw` walks distinct pool
                  indices), so the clustering correction is a no-op here and the CSV says so
                  rather than leaving the reader to assume it.
  outcome         the three-state verdict of R9 -- exceeds / inconclusive / reversed -- from the
                  CI alone. "inconclusive" is a failure to reject, never "does not generalise".
  lid             R3's language id: fastText lid218e on the top-1 and top-4 rollouts, and the
                  rate at which the arm's own language comes back. A cosine above the corpus does
                  NOT certify the output language (the hand-picked test found Ukrainian → Russian),
                  which is why this column sits beside Δ and not in a footnote. On code and maths
                  arms fastText is not meaningful and the column is the `code_like` rate from the
                  same regex classifier R7 uses -- one classifier, two uses.

NOTHING IS KEYED ON A CHECKPOINT NAME. Sources come from `results.common.discover_sources`
iterating `config.yaml`'s `maemms:` against the volume, so both generations and the untrained-base
control are rows because they are on the volume, not because this file names them. An `role:
control` source becomes the control column; every other source gets its own arms table.

A source, a scan or an arm that is ABSENT is skipped and listed, never zero-filled.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.common as R  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

BASE = "qwen36-27b"
# The design's level-1 conjunction is over the 21 arms that are not `diag` and not the English
# pipeline check (§6). `diag` (`formulas`) is a tokenizer-boundary diagnostic, reported in its own
# row and excluded; `ufw_en` is the convention check of §8 (a) and is reported as an arm.
CONJUNCTION_EXCLUDES = ("diag",)


# ---------------------------------------------------------------------------------------------
# readers -- the scan layout, the set, and the estimator imported from stats_ood
# ---------------------------------------------------------------------------------------------


def _stats_ood():
    """`reconstruction/stats_ood.py`'s estimators and lid, imported rather than rewritten.

    It is a uv script with its own dependency block, so `fasttext` and `huggingface_hub` are only
    importable when this file is run with them (`uv run --with fasttext --with huggingface-hub`).
    Without them `--lid` degrades to "not run" and says so; the cosine half never depends on it.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "reconstruction" / "stats_ood.py"
    spec = importlib.util.spec_from_file_location("stats_ood", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stats_ood"] = mod
    spec.loader.exec_module(mod)
    return mod


SCAN_RE = re.compile(r"^(?P<set>.+?)__(?P<corpus>[^_].*?)(?:__(?P<mb>[0-9.]+)m)?$")


def C_resolve_mu(mu, base: str) -> str:
    """A `mu:` value as one comparable string. `{base}` expands; None/none is the literal "none"."""
    if mu is None or str(mu).strip().lower() in ("", "none", "null"):
        return "none"
    t = str(mu).strip().replace("{base}", base)
    # config spells a mu RELATIVE to --root ("base/<base>/stats/mu.f32"); `score` records the
    # ABSOLUTE path it actually read ("/vol/base/..."). Same file, two spellings, and comparing
    # them raw would call every source incomparable.
    for pre in ("/vol/", "vol/", "/"):
        if t.startswith(pre):
            t = t[len(pre):]
            break
    return t


def C_corpus_dir(cfg: dict, arm: str) -> str:
    """The corpus DIRECTORY of an OOD arm, through `corpora:` -- never the arm id by assumption.

    `precompute/common.load_config` synthesises one `corpora:` entry per `ood_arms:` key, named
    `ood_<arm>` with `dir: <arm>`. Reading `dir` rather than assuming the two are equal keeps this
    correct if an arm is ever given a directory that is not its own name.
    """
    spec = (cfg.get("corpora") or {}).get(f"ood_{arm}")
    return (spec or {}).get("dir") or arm


MU_RE = re.compile(r"^- CENTRING: mu=(\S+) from ", re.M)


def scan_mu_of(vol: R.Vol, base: str, scan_dir: str) -> str | None:
    """The mean a scan centred its targets on, from the scan's OWN README.

    `common.note_convention` writes `- CENTRING: mu=<path> from <source>` as the first note of
    every product that resolves one, so the mean is recorded by the producer rather than inferred
    from a directory name -- which is the difference between a number that is comparable and a
    number that is merely next to another one. The FIRST such line is the `--set` set's; a scan
    with `--with-set` adds one per extra bank.
    """
    p = vol.get(f"base/{base}/scan/{scan_dir}/README.md")
    if p is None:
        return None
    m = MU_RE.search(p.read_text())
    return m.group(1) if m else None


def scan_dirs_of(vol: R.Vol, base: str, set_name: str) -> dict[str, tuple[str, float | None]]:
    """{scan directory -> (corpus directory name, the M-token bound or None)} for this set.

    The pipeline keys a scan by (set, corpus): `scan/<set>` is the base's own English corpus
    unbounded, `scan/<set>__<corpus>` another corpus, and `__<M>m` on top when the scan stopped at
    a nested prefix (`precompute/scan.py`, `common.scan_dir`). The OOD branch's pre-rebase layout
    was `scan/<set>/<corpus>-<M>m/`; sets drawn before the rebase still carry it and are read by
    `reconstruction/stats_ood.py`, not here -- this file reads only what this pipeline writes.
    """
    out: dict[str, tuple[str, float | None]] = {}
    for d in vol.ls(f"base/{base}/scan"):
        if d == set_name:
            out[d] = ("", None)
            continue
        m = SCAN_RE.match(d)
        if m and m.group("set") == set_name:
            mb = m.group("mb")
            out[d] = (m.group("corpus"), float(mb) if mb else None)
    return out


def scan_top1(vol: R.Vol, base: str, scan_dir: str, set_name: str) -> dict[tuple[int, float], float]:
    """{(set row, corpus size in M) -> top-1 cosine} from one scan's `topk.jsonl`.

    A target with an EMPTY top list at a size had no unmasked window there; it is left out rather
    than scored 0, so the pairing below drops it on both sides instead of inventing a loss.
    """
    rows = vol.jsonl(f"base/{base}/scan/{scan_dir}/topk.jsonl")
    if rows is None:
        return {}
    out: dict[tuple[int, float], float] = {}
    for r in rows:
        if r.get("set", set_name) != set_name or not r.get("top"):
            continue
        out[(int(r.get("set_row", r["row"])), float(r["size"]))] = float(r["top"][0][3])
    return out


def load_ood_ids(vol: R.Vol, base: str, set_name: str) -> list[dict]:
    rows = vol.jsonl(f"base/{base}/heldout/{set_name}/ids.jsonl")
    assert rows, f"no ids.jsonl for set {set_name!r} under base/{base}/heldout/"
    assert all("arm" in r for r in rows), (
        f"{set_name} has rows without an `arm` field: it is not an OOD set, and every table below "
        f"is stratified by arm"
    )
    return rows


# `score`'s three cosines, by the key its per_target.jsonl uses:
#   asym     cos(h,      unit(act - mu))   THE SCAN'S CONVENTION -- the only one Δ can use
#   centred  cos(h - mu, unit(act - mu))   the pipeline's symmetric headline
#   raw      cos(h,      unit(act))        both sides uncentred
BO64 = {"asym": "bo_a_64", "centred": "bo_c_64", "raw": "bo_64"}


def bo64_of(rec: dict, which: str) -> float | None:
    """The unbiased best-of-64 of one target in one of the three cosines, or None when absent.

    `score` writes `bo_c_*` and `bo_a_*` only when the run centred AND every rollout of the row
    was finite, so absence means "this run has no such number for this row" -- a different fact
    from a low one, and never filled with a zero.
    """
    v = rec.get(BO64[which])
    return None if v is None else float(v)


# ---------------------------------------------------------------------------------------------
# the per-arm table
# ---------------------------------------------------------------------------------------------


def arm_rows(
    ids: list[dict],
    src: R.Source,
    top1_by_arm: dict[str, dict[tuple[int, float], float]],
    size_m: float,
    which: str,
    boot_ci,
    outcome,
    control: dict[int, dict] | None,
    comparable: bool = True,
) -> tuple[list[dict], list[str]]:
    """One record per arm: both cosines, the corpus cell, Δ with its CI, and the verdict."""
    by_arm: dict[str, list[dict]] = {}
    for r in ids:
        by_arm.setdefault(r["arm"], []).append(r)
    recs, skipped = [], []
    for arm, rows in by_arm.items():
        # IN-DOMAIN means this arm's OWN corpus. Every scan carries every target (design §4: the
        # scan's cost is per corpus token, so one pass answers for all of them), so row r has a
        # top-1 in all 23 scans and only ONE of them is in its domain -- the rest are the
        # cross-domain cells. Flattening them into one {row: cos} map silently mixed the two.
        top1 = top1_by_arm.get(arm) or {}
        if comparable and not top1:
            skipped.append(f"arm `{arm}`: no scan of its own corpus, so no in-domain cell")
            continue
        pairs, raw, cen, asy, ctrl, corp, docs = [], [], [], [], [], [], []
        for r in rows:
            pt = src.per_target.get(int(r["row"]))
            if pt is None:
                continue
            # THE COSINES DO NOT NEED THE CORPUS. A source with no scan at its own mean still has
            # a bo64 per target, and reporting it is the whole point of the `comparable` split --
            # dropping the row entirely would turn "we cannot difference this" into "we measured
            # nothing", which are opposite findings.
            raw.append(bo64_of(pt, "raw"))
            cen.append(bo64_of(pt, "centred"))
            asy.append(bo64_of(pt, "asym"))
            if control is not None and int(r["row"]) in control:
                cb = bo64_of(control[int(r["row"])], which)
                if cb is not None:
                    ctrl.append(cb)
            c = top1.get((int(r["row"]), size_m))
            m = bo64_of(pt, which)
            if c is None or m is None:
                continue
            corp.append(c)
            pairs.append(m - c)
            # the cluster label of the KEPT pair, collected here rather than sliced off the arm's
            # rows afterwards: a target the scan has no cell for drops out, and a positional slice
            # would then label the survivors with their neighbours' documents.
            docs.append(r.get("doc", ("row", int(r["row"]))))
        if comparable and len(pairs) < 2:
            skipped.append(
                f"arm `{arm}`: {len(pairs)} paired targets (needs > 1) -- its own scan cell at "
                f"{size_m}M or the score rows are missing"
            )
            continue
        if not cen and not raw:
            skipped.append(f"arm `{arm}`: no scored rows for this source")
            continue
        d = np.asarray(pairs, dtype=float) if pairs else np.zeros(0)
        # Δ IS ONLY MEANINGFUL WHEN BOTH SIDES SCORE AGAINST THE SAME TARGET VECTOR. The scan
        # centres on the mean it was given; a checkpoint's rollouts are scored on the mean IT
        # declares. When those differ the two cosines are angles to two different directions and
        # their difference is not a margin -- so the cosines are still reported and Δ is not.
        # MEASURED on this set: stats/mu.f32 and whiten_mu agree at cos 0.977, which puts
        # unit(act - mu) a median cos 0.969 apart over the 368 targets -- the same order as the
        # effects being measured, not a rounding difference.
        mean, lo, hi = boot_ci(d) if comparable else (None, None, None)
        _, se_cl, _, n_clust = (
            R.cluster_bootstrap(d, docs) if comparable else (None, None, None, len(set(docs)))
        )
        recs.append(
            {
                "arm": arm,
                "family": rows[0]["family"],
                "n": int(d.size) if comparable else len([x for x in asy if x is not None]),
                "bo64_asym": _mean(asy),
                "bo64_centred": _mean(cen),
                "bo64_raw": _mean(raw),
                "corpus_top1": float(np.mean(corp)) if corp else None,
                "control_bo64": _mean(ctrl) if ctrl else None,
                "delta": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "comparable": comparable,
                "se_clustered": se_cl,
                "n_clusters": n_clust,
                "win_frac": float((d > 0).mean()) if comparable else None,
                "outcome": outcome(lo, hi) if comparable else "not comparable",
            }
        )
    recs.sort(key=lambda r: (r["family"], r["arm"]))
    return recs, skipped


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def lid_rates(mod, vol: R.Vol, cfg: dict, ids: list[dict], src: R.Source, set_name: str,
              lid_model: Path | None) -> tuple[dict[str, dict], str]:
    """{arm -> {lid_top1_rate, lid_top4_rate | code_like_top1_rate}} (review R3), and a note.

    The classifier is chosen by the ARM, not by the source: `ood_arms.<arm>.lid` is a list of the
    NLLB labels that count as this arm's language, and `null` there says fastText is not
    meaningful for it -- the code and maths arms, which report the `code_like` regex rate instead.
    """
    stem = f"{set_name}__{src.engine}" if src.engine != "hf" else set_name
    if src.run_tag:
        stem += f"__{src.run_tag}"
    texts = mod.rollout_texts(vol, src.base, src.maemm, set_name, stem)
    if not texts:
        return {}, f"no rollout texts at maemms/{src.maemm}/rollouts/{stem}.jsonl: lid not run"
    model = mod.load_lid(lid_model)
    arms = cfg["ood_arms"]
    out: dict[str, dict] = {}
    per_arm: dict[str, list[tuple[list[str], list[str] | None]]] = {}
    for r in ids:
        t = texts.get(int(r["row"]))
        if t:
            per_arm.setdefault(r["arm"], []).append((t, arms[r["arm"]].get("lid")))
    for arm, items in per_arm.items():
        want = items[0][1]
        if want is None:
            rates = [1.0 if mod.code_like(t[0]) else 0.0 for t, _ in items]
            out[arm] = {"lid_top1_rate": None, "lid_top4_rate": None,
                        "code_like_top1_rate": float(np.mean(rates))}
            continue
        if model is None:
            out[arm] = {"lid_top1_rate": None, "lid_top4_rate": None, "code_like_top1_rate": None}
            continue
        t1, t4 = [], []
        for t, _ in items:
            labs = [mod.lid_label(model, x)[0] for x in t[:4]]
            t1.append(1.0 if labs and labs[0] in want else 0.0)
            t4.append(1.0 if any(x in want for x in labs) else 0.0)
        out[arm] = {"lid_top1_rate": float(np.mean(t1)), "lid_top4_rate": float(np.mean(t4)),
                    "code_like_top1_rate": None}
    note = "" if model is not None else (
        "fastText lid218e was not loadable, so the language columns are absent on the lang/ctrl "
        "arms (run with `--with fasttext --with huggingface-hub`); the code_like column is "
        "unaffected, it is a regex in common.py"
    )
    return out, note


# ---------------------------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    set_name: Annotated[str, typer.Option("--set", help="the OOD held-out set")] = "",
    base: Annotated[str, typer.Option()] = BASE,
    size: Annotated[
        float, typer.Option(help="the corpus size in M tokens the claim is read at (0 = the "
                                 "largest every arm's scan actually carries)")
    ] = 0.0,
    root: Annotated[str, typer.Option(help="a volume-relative root prefix")] = "",
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="the local mirror")] = None,
    lid: Annotated[bool, typer.Option(help="run fastText lid218e on the rollouts (review R3)")] = True,
    lid_model: Annotated[Path | None, typer.Option(help="a local lid218e model.bin")] = None,
    refetch: Annotated[bool, typer.Option()] = False,
    offline: Annotated[bool, typer.Option(help="read the mirror only; never call modal")] = False,
    quiet: Annotated[bool, typer.Option()] = False,
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
) -> None:
    """The per-arm generalisation table for one OOD set."""
    assert set_name, "--set is required: this file has no default set, by D6's rule for writers"
    cfg = R.load_config()
    assert set_name in cfg["heldout"], f"{set_name!r} is not a set in config.yaml"
    here = Path(__file__).resolve().parent
    out_dir = Path(out) if out else here / "ood"
    vol = R.Vol(root, Path(data_dir) if data_dir else out_dir / "_mirror",
                modal_cmd=modal_cmd, refetch=refetch, quiet=quiet, offline=offline)
    mod = _stats_ood()

    ids = load_ood_ids(vol, base, set_name)
    sources, absent = R.discover_sources(vol, cfg, base, set_name)
    usable, notes = [], [f"config'd checkpoint with no products on this set: `{a}`" for a in absent]
    for s in sources:
        why = R.load_source(vol, s)
        (notes.append(f"source `{s.label}` unusable: {why}") if why else usable.append(s))
    assert usable, f"no scored source for {set_name}: nothing to tabulate ({notes})"

    # the scans of this set, and the size cell the claim is read at
    scans = scan_dirs_of(vol, base, set_name)
    # {(corpus dir, resolved mu) -> {(row, size) -> top-1}}. A scan is (set x corpus x MEAN): the
    # same corpus scanned under two centrings gives two different sets of numbers, and only the
    # one matching a checkpoint's own mean can be differenced against that checkpoint.
    top1_by_corpus: dict[tuple[str, str], dict] = {}
    english: dict[str, dict] = {}
    scan_mus: dict[str, str] = {}
    for d, (c, _) in scans.items():
        mu_here = C_resolve_mu(scan_mu_of(vol, base, d), base)
        scan_mus[d] = mu_here
        if c:
            top1_by_corpus[(c, mu_here)] = scan_top1(vol, base, d, set_name)
        else:
            english[d] = scan_top1(vol, base, d, set_name)
    assert top1_by_corpus, (
        f"no in-domain scan for {set_name} under base/{base}/scan/: every Δ below is "
        f"MAEMM minus corpus search, so there is no table without one"
    )
    sizes = sorted({sz for t in top1_by_corpus.values() for _, sz in t})
    if size:
        assert size in sizes, f"--size {size} is not among the scanned sizes {sizes}"
        size_m = size
    else:
        # the largest size EVERY in-domain scan carries: a per-arm mix of sizes is not one table.
        common = set.intersection(*({sz for _, sz in t} for t in top1_by_corpus.values() if t))
        assert common, f"the in-domain scans share no corpus size: {sizes}"
        size_m = max(common)

    # {arm -> its OWN corpus's {(row, size) -> top-1}}. The corpus directory of arm `a` is
    # `corpora[f"ood_{a}"].dir`, which is `a` itself for every arm declared in `ood_arms:`.
    dir_of_arm = {a: C_corpus_dir(cfg, a) for a in {r["arm"] for r in ids}}
    notes.append(
        "scans read: " + ", ".join(f"`{d}` at mu={scan_mus[d]}" for d in sorted(scan_mus))
    )

    ctrl_src = next((s for s in usable if s.role == "control"), None)
    control = ctrl_src.per_target if ctrl_src else None

    o = R.Out(
        out_dir,
        f"OOD generalisation: `{set_name}`",
        [
            f"Base `{base}`, {len({r['arm'] for r in ids})} arms x "
            f"{len(ids) // max(1, len({r['arm'] for r in ids}))} targets, "
            f"in-domain corpus search at **{size_m:g}M tokens**.",
            "",
            "Δ = the MAEMM's unbiased best-of-64 minus the in-domain corpus search's top-1, paired "
            "per target; CI is the design's 10,000-resample percentile bootstrap over the arm's "
            "targets. The outcome is three-state (review R9): **exceeds** (CI above zero), "
            "**inconclusive** (CI covers zero -- a failure to reject, not a finding of no "
            "generalisation), **reversed** (CI below zero).",
            "",
            f"Run under `MODAL_PROFILE={R.env_hint()}`.",
        ],
    )

    verdicts: dict[str, dict[str, str]] = {}
    for src in usable:
        if src.role == "control":
            continue
        got_mu = C_resolve_mu(src.mu, base)
        # THE SCAN AT THIS SOURCE'S OWN MEAN, arm by arm. Not "the scan", and not the one named on
        # the command line: a checkpoint trained on whiten_mu is differenced against the whiten_mu
        # scan and the old primary against the stats_mu one, from the same table.
        top1_by_arm = {a: top1_by_corpus.get((d, got_mu), {}) for a, d in dir_of_arm.items()}
        comparable = any(top1_by_arm.values())
        missing = sorted(a for a, t in top1_by_arm.items() if not t)
        if not comparable:
            have = sorted({m for _, m in top1_by_corpus})
            notes.append(
                f"source `{src.label}` centres on {got_mu} and NO in-domain scan was run at that "
                f"mean (scans present at: {have}). Its bo64 columns are reported and Δ is not: "
                f"differencing it against a scan at another mean would subtract two angles to two "
                f"DIFFERENT target vectors. On this set stats/mu.f32 and whiten_mu agree at cos "
                f"0.977 and put unit(act - mu) a median cos 0.969 apart -- the size of the effect "
                f"being measured, not a rounding difference."
            )
        elif missing:
            notes.append(
                f"source `{src.label}` (mu={got_mu}): arms with no scan of their own corpus at "
                f"that mean: " + ", ".join(f"`{a}`" for a in missing)
            )
        recs, skipped = arm_rows(
            ids, src, top1_by_arm, size_m, "asym", mod.boot_ci, mod.outcome, control,
            comparable=comparable,
        )
        notes += skipped
        lids: dict[str, dict] = {}
        if lid:
            lids, lnote = lid_rates(mod, vol, cfg, ids, src, set_name, lid_model)
            if lnote:
                notes.append(lnote)
        lang_col = []
        for r in recs:
            li = lids.get(r["arm"], {})
            r.update({k: li.get(k) for k in ("lid_top1_rate", "lid_top4_rate", "code_like_top1_rate")})
            lang_col.append(
                R.num(r["lid_top1_rate"], 3) if r["lid_top1_rate"] is not None
                else (f"{R.num(r['code_like_top1_rate'], 3)} (code)"
                      if r["code_like_top1_rate"] is not None else "—")
            )
        verdicts[src.label] = {r["arm"]: r["outcome"] for r in recs}
        header = ["arm", "family", "n", "bo64 (asym)", f"corpus {size_m:g}M",
                  "control bo64", "Δ", "95% CI", "win", "outcome", "lang / code"]
        rows_md = [
            [r["arm"], r["family"], r["n"], R.num(r["bo64_asym"]),
             R.num(r["corpus_top1"]), R.num(r["control_bo64"]),
             R.num(r["delta"]), f"[{R.num(r['ci_lo'], 3)}, {R.num(r['ci_hi'], 3)}]",
             R.num(r["win_frac"], 2), r["outcome"], lc]
            for r, lc in zip(recs, lang_col, strict=True)
        ]
        csv_header = ["arm", "family", "n", "bo64_asym", "bo64_centred", "bo64_raw", "corpus_top1",
                      "corpus_size_m", "control_bo64", "delta", "ci_lo", "ci_hi",
                      "se_clustered", "n_clusters", "win_frac", "outcome",
                      "lid_top1_rate", "lid_top4_rate", "code_like_top1_rate"]
        csv_rows = [
            [r["arm"], r["family"], r["n"], r["bo64_asym"], r["bo64_centred"], r["bo64_raw"],
             r["corpus_top1"],
             size_m, r["control_bo64"], r["delta"], r["ci_lo"], r["ci_hi"], r["se_clustered"],
             r["n_clusters"], r["win_frac"], r["outcome"], r["lid_top1_rate"],
             r["lid_top4_rate"], r["code_like_top1_rate"]]
            for r in recs
        ]
        o.table(
            f"arms_{src.label.replace('/', '_').replace(':', '__').replace('@', '_at_')}",
            f"Arms — {src.label}"
            + ("" if src.centred else "  (this run centred on NOTHING: `--mu none`)"),
            f"bo64 against the in-domain {size_m:g}M corpus search, paired per target. BOTH "
            f"SIDES ARE THE ASYMMETRIC CONVENTION -- `cos(h, unit(act - mu))`, uncentred scorer "
            f"against the centred target -- which is what `scan` computes for a corpus window and "
            f"what the paper's bo64 0.569 and corpus 0.351 are stated in. The symmetric cosines "
            f"(`cos_centred`, `cos`) are in the CSV. The control column is "
            f"{'`' + ctrl_src.label + '`' if ctrl_src else 'absent'}. `lang / code` is the rate at "
            f"which the top-1 rollout comes back in the arm's own language (fastText lid218e), or "
            f"the code-like rate where fastText is not meaningful. **READ THE CONVENTION NOTE "
            f"BELOW BEFORE COMPARING Δ WITH THE DESIGN'S +0.218.**",
            header, rows_md, csv_header=csv_header, csv_rows=csv_rows,
        )

        conj = [r for r in recs if r["family"] not in CONJUNCTION_EXCLUDES]
        n_ex = sum(1 for r in conj if r["outcome"] == "exceeds")
        named = [f"`{r['arm']}` ({r['outcome']})" for r in conj if r["outcome"] != "exceeds"]
        o.section(
            "\n".join(
                [
                    f"### Pre-registered claim — {src.label}",
                    "",
                    f"Level 1, design §0: *on every arm, the best of 64 MAEMM rollouts aligns with "
                    f"the target more closely than the best window of an in-domain corpus search "
                    f"in the target's own domain.* Read at **{size_m:g}M** corpus tokens, over the "
                    f"{len(conj)} arms of the conjunction (`{'`, `'.join(CONJUNCTION_EXCLUDES)}` "
                    f"excluded, design §2):",
                    "",
                    f"**{n_ex} of {len(conj)} arms exceed.**"
                    + ("" if not named else "  Not exceeding: " + ", ".join(named) + "."),
                    "",
                ]
            )
        )

    # the English in-distribution reference, recomputed from the frozen 2026-09-16 scan by the one
    # script that owns it (R1: own-document corpus hits excluded, so the margin is like-for-like)
    # The English reference. `stats_ood.english_reference` returns, per corpus size:
    #   {n, top1_all, top1_noown, n_noown, own_is_top1, no_noown_candidate}
    # -- it does NOT carry a bo64 or a margin, so the margin is formed here against the paper's
    # frozen English bo64 and that constant is named in the caption rather than hidden in a sum.
    EN_BO64 = 0.569   # design §0, the paper's 512-target English best-of-64
    ref = mod.english_reference(vol, base=base)
    if ref:
        o.table(
            "english_reference", "English in-distribution reference (review R1)",
            f"Recomputed from `scan/2026-09-16_v1/topk.jsonl`. `top1 (no own doc)` EXCLUDES corpus "
            f"windows from the target's own document; the OOD corpora contain no target documents "
            f"(design §2), so that is the like-for-like reference, and the paper's frozen 0.371 at "
            f"4M counts the own document and stays in its own table. The margin is "
            f"{EN_BO64} (the paper's English bo64, design §0) minus that column.",
            ["size (M)", "n", "top1 (all)", "top1 (no own doc)", "margin vs bo64 "
             f"{EN_BO64}", "own doc IS top-1", "no non-own candidate"],
            [[k, v["n"], R.num(v["top1_all"]), R.num(v["top1_noown"]),
              R.num(EN_BO64 - v["top1_noown"]) if v["n_noown"] else None,
              v["own_is_top1"], v["no_noown_candidate"]]
             for k, v in sorted(ref.items(), key=lambda kv: float(kv[0]))],
        )
    else:
        notes.append("the English reference scan is not on this root: no `en_ref` row")
    if english:
        notes.append(
            "cross-domain scans of the base's own English corpus present: "
            + ", ".join(sorted(english))
        )

    o.section(
        "\n".join([
            "### The cosine convention, and what Δ here is and is not",
            "",
            "`scan` scores a corpus window as `normalize(h) @ v` with `v = unit(act - mu)`: the "
            "corpus activation UNCENTRED against a CENTRED target (precompute/scan.py; design §4, "
            "'uncentred cosine in the scan'). That is the paper's convention and it is unchanged "
            "-- the English reference below reproduces the design's R1 numbers to the digit "
            "(0.3137 / 0.3511 / 0.3851 at 1/4/16M, own document top-1 on 114 of 512 targets at "
            "4M).",
            "",
            "`score` on this branch emits two SYMMETRIC cosines instead: `cos` = "
            "cos(h, unit(act)) and `cos_centred` = cos(h - mu, unit(act - mu)). Neither is "
            "`cos(h, unit(act - mu))`, the asymmetric number the paper's bo64 0.569 is, and on a "
            "`storage: raw` set there is no flag that produces it -- the legacy path got it for "
            "free because a `storage: unit` set's stored rows ARE `unit(act - mu)`, so `dirs` was "
            "already the centred target.",
            "",
            "Consequences, stated rather than smoothed:",
            "",
            "- the CORPUS side of every Δ is exactly the paper's; the MAEMM side is not, so the "
            "MAGNITUDE of Δ here is not comparable with the design's English margin of +0.218 "
            "(4M) or +0.256 (1M), and neither is bo64 comparable with 0.569;",
            "- the pre-registered claim is a per-arm SIGN ('the best of 64 rollouts aligns more "
            "closely than the best corpus window'), and a sign is testable under any one "
            "convention applied to both sides of the comparison -- which is why the verdicts are "
            "reported and the margins are labelled;",
            "- both cosines are in the CSV, so the arms can be re-read under either without "
            "re-running anything on the GPU.",
            "",
            "Closing this properly means one of: `score` gaining the asymmetric cosine on a raw "
            "set, or the scan centring its corpus activations to match `cos_centred`. The second "
            "invalidates the frozen English reference; the first does not. Not decided here.",
            "",
        ])
    )
    for n in notes:
        o.note(n)
    path = o.finish([])
    print(f"[ood] {path}")
    for label, v in verdicts.items():
        print(f"[ood] {label}: " + ", ".join(f"{a}={s}" for a, s in sorted(v.items())))
    if vol.missing:
        print(f"[ood] {len(vol.missing)} files were not on the volume (listed in tables.md)")


if __name__ == "__main__":
    app()
