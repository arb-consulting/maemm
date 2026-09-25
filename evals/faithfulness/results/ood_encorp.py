#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9",
#                 "polars>=1", "rich>=13"]   # the last two: reconstruction/stats_ood.py
# ///
"""M9: the Exemplifier's best-of-8 against the 10M ENGLISH TRAINING-CORPUS search, per OOD arm.

    cd <repo>
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=<your-profile>; \\
     uv run evals/faithfulness/results/ood_encorp.py \\
       --set 2026-09-23_ood_full --data-dir runs/2026-09-23_m5-ood/_mirror \\
       --out runs/2026-09-24_m9-ood-encorp)

WHAT THIS IS, AND WHAT `results/ood.py` ALREADY IS. `ood.py` differences each arm's best-of-8
against **that arm's own corpus** -- the pre-registered level-1 contrast, and the one Fig.
`fig:evals` panel (b) draws. This file differences the SAME best-of-8 against the **English
10M training corpus** (`corpora.train10m`, directory `train_parity_10m`), the single
comparator the paper's main fidelity result uses, scanned once against all 11,264 OOD targets
(`scan/<set>__train_parity_10m__paper0923`). The two are different claims and neither replaces
the other: an own-domain search reads text of the target's own language, a training-corpus search
reads the English text this MAEM was trained to invert against.

CONVENTION, and the one thing that would silently break it. Both sides are CENTRED about the
base's scoring constant: the scan ran `--centre --run-tag paper0923` and the MAEM's `cos_centred`
is taken about the same constant, so Δ is a paired difference and not the subtraction of two
angles to two different vectors. `scan_mu_of` is asserted here rather than assumed -- an uncentred
scan of the same corpus would land in a directory of the same name and be differenced without a
word (`results/ood.py`'s own warning, kept true by asserting on the README's recorded mode).

ESTIMATORS, imported and not rewritten: `reconstruction/stats_ood.boot_ci` (the design's
10,000-resample percentile bootstrap over the arm's targets) and `results.common.cluster_bootstrap`
(the document-clustered SE carried beside it), which is exactly the pair `results/ood.py` uses. The
best-of-8 map is `results.ood.centred_bo_k`, the same recompute from `cos_centred.f16`, so the bo8
column here is the same number as the bo8 column there, by construction rather than by agreement.

Local, CPU, no GPU, no model.
"""

from __future__ import annotations

import csv as _csv
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.common as R  # noqa: E402
import results.ood as OOD  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

BASE = "qwen36-27b"
EN_CORPUS_DIR = "train_parity_10m"      # `corpora.train10m.dir`
EN_SIZE = 10.0                          # the ladder's top, the size the paper's fidelity row uses
MAEM = "qwen36-27b/2026-09-18_rl-final"

# The cells rows this file owns. `encorp10` is the 10M ENGLISH TRAINING corpus, beside the existing
# `corp10` (the arm's OWN corpus): two different comparators, two key families, never one key
# meaning different corpora on different arms.
PER_ARM = ("encorp10.cos", "endiff.cos.bo8", "endiff.verdict")
CONJ = ("ood.conj.endiff.nexceed", "ood.conj.endiff.ninconcl", "ood.conj.endiff.nreversed",
        "ood.conj.endiff.min")


def owned_keys(cfg: dict) -> set[str]:
    keys = {f"ood.{OOD.cell_id_of(a)}.{f}" for a in cfg.get("ood_arms", {}) for f in PER_ARM}
    return keys | set(CONJ)


def pick_source(vol: R.Vol, cfg: dict, base: str, set_name: str, maem: str):
    """The scored run this table differences, and the notes about the ones it did not pick."""
    sources, absent = R.discover_sources(vol, cfg, base, set_name)
    notes = [f"config'd checkpoint with no products on this set: `{a}`" for a in absent]
    usable = []
    for s in sources:
        why = R.load_source(vol, s)
        (notes.append(f"source `{s.label}` unusable: {why}") if why else usable.append(s))
    want = [s for s in usable if s.role != "control" and s.maem == maem
            and OOD.has_col(s, "centred")]
    assert want, (
        f"no scored source for {maem} on {set_name} carrying `bo_c_64`/`cos_centred`: this table "
        f"is the centred pair and has nothing to read ({notes})")
    for s in usable:
        if s not in want:
            notes.append(f"source `{s.label}` not used (role={s.role}, maem={s.maem})")
    return want[0], notes


def own_domain_cells(path: Path | None) -> dict[str, dict]:
    """{arm -> its row of `results/ood/arms_*.csv`} -- the OWN-domain column, read, not recomputed.

    The own-domain corpus top-1 and its Δ are `results/ood.py`'s numbers and stay its numbers: this
    file prints them beside its own column so the two comparators can be read against each other,
    and re-deriving them here would be a second implementation of one cell.
    """
    if path is None:
        return {}
    with open(path, newline="") as fh:
        return {r["arm"]: r for r in _csv.DictReader(fh)}


@app.command()
def main(
    set_name: Annotated[str, typer.Option("--set", help="the OOD held-out set")] = "",
    base: Annotated[str, typer.Option()] = BASE,
    maem: Annotated[str, typer.Option(help="the scored checkpoint to difference")] = MAEM,
    scan_dir: Annotated[
        str, typer.Option(help="the English-corpus scan directory under base/<base>/scan/ "
                               "(default: <set>__train_parity_10m__paper0923)")
    ] = "",
    size: Annotated[float, typer.Option(help="the corpus size in M tokens to read")] = EN_SIZE,
    arms_csv: Annotated[
        Path | None, typer.Option(help="results/ood/arms_*.csv, for the own-domain column")
    ] = None,
    root: Annotated[str, typer.Option(help="a volume-relative root prefix")] = "",
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="the local mirror")] = None,
    cells: Annotated[
        str, typer.Option(help="paper/numbers/cells.csv: merge this run's rows into it in place, "
                               "by key. `default` resolves the paper project's copy; empty (the "
                               "default) writes only the fragment CSV beside the table")
    ] = "",
    refetch: Annotated[bool, typer.Option()] = False,
    offline: Annotated[bool, typer.Option(help="read the mirror only; never call modal")] = False,
    quiet: Annotated[bool, typer.Option()] = False,
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
) -> None:
    """The per-arm table of best-of-8 against the English 10M training-corpus search."""
    assert set_name, "--set is required"
    cfg = R.load_config()
    assert set_name in cfg["heldout"], f"{set_name!r} is not a set in config.yaml"
    scan = scan_dir or f"{set_name}__{EN_CORPUS_DIR}__paper0923"
    out_dir = Path(out) if out else R.out_dir("ood_encorp")
    cells_target = None
    if (cells or "").strip():
        import results.faithfulness as _FA

        cells_target = _FA.resolve_cells_path(cells)

    vol = R.Vol(root, Path(data_dir) if data_dir else R.mirror_dir(root),
                modal_cmd=modal_cmd, refetch=refetch, quiet=quiet, offline=offline)
    SD = OOD._stats_ood()

    ids = OOD.load_ood_ids(vol, base, set_name)
    src, notes = pick_source(vol, cfg, base, set_name, maem)

    # THE SCAN'S OWN RECORD OF ITS MODE AND ITS MEAN, asserted before a single difference is taken.
    mu_raw, centred = OOD.scan_mu_of(vol, base, scan)
    mu_scan, mu_src = OOD.C_resolve_mu(mu_raw, base), OOD.C_resolve_mu(src.mu, base)
    assert centred, (
        f"scan `{scan}` does not record `--centre`: its windows are not centred about the scoring "
        f"constant, so a `cos_centred` Δ against them subtracts two angles to two different "
        f"vectors. Nothing downstream can catch this, which is why it stops here")
    assert mu_scan == mu_src, (
        f"scan `{scan}` is at mu={mu_scan} and source `{src.label}` at mu={mu_src}: the two sides "
        f"score against different target vectors and their difference is not a margin")
    notes.append(f"English-corpus scan `{scan}`: `--centre` recorded, mu={mu_scan}, "
                 f"the same mean source `{src.label}` recorded")

    en = OOD.scan_top1(vol, base, scan, set_name)
    assert en, f"no `topk.jsonl` rows for {set_name} in scan `{scan}`"
    have_sizes = sorted({sz for _, sz in en})
    assert size in have_sizes, f"--size {size:g} is not among the scan's sizes {have_sizes}"
    told = OOD.scan_sizes_of(vol, base, scan)
    if told:
        assert max(told) == max(have_sizes), (
            f"scan `{scan}` says `- sizes: {told}` in its README but its topk.jsonl carries "
            f"{have_sizes}: the product is truncated or two runs are in one directory")
    notes.append(f"English corpus read at {size:g}M of the scan's ladder {have_sizes}; "
                 f"{len(en)} (row, size) cells, {sum(1 for (_, s) in en if s == size)} at {size:g}M")

    bo8, bo8_note = OOD.centred_bo_k(vol, src)
    notes.append(f"source `{src.label}`: {bo8_note}")

    own = own_domain_cells(arms_csv)
    if arms_csv and not own:
        notes.append(f"`{arms_csv}` carried no arm rows: the own-domain column is empty")

    by_arm: dict[str, list[dict]] = {}
    for r in ids:
        by_arm.setdefault(r["arm"], []).append(r)

    recs, skipped = [], []
    for arm, rows in sorted(by_arm.items()):
        pairs, docs = [], []
        maem, corp = [], []
        for r in rows:
            row = int(r["row"])
            c = en.get((row, size))
            m = bo8.get(row)
            if c is None or m is None or not np.isfinite(float(m)):
                continue
            pairs.append(float(m) - c)
            maem.append(float(m))
            corp.append(c)
            # The cluster label of the KEPT pair, collected here and not sliced off the arm's rows
            # afterwards: a target the scan has no cell for drops out, and a positional slice would
            # label the survivors with their neighbours' documents.
            docs.append(r.get("doc", ("row", row)))
        if len(pairs) < 2:
            skipped.append(f"arm `{arm}`: {len(pairs)} paired targets (needs > 1)")
            continue
        d = np.asarray(pairs, dtype=float)
        mean, lo, hi = SD.boot_ci(d)
        _, se_cl, _, n_clust = R.cluster_bootstrap(d, docs)
        o_row = own.get(arm, {})
        recs.append({
            "arm": arm,
            "family": rows[0]["family"],
            "n": int(d.size),
            "bo8_centred": float(np.mean(maem)),
            "en_top1": float(np.mean(corp)),
            "en_size_m": size,
            "own_top1": o_row.get("corpus_top1") or "",
            "own_size_m": o_row.get("corpus_size_m") or "",
            "own_delta8": o_row.get("delta8") or "",
            "delta8": mean,
            "ci8_lo": lo,
            "ci8_hi": hi,
            "se8_clustered": se_cl,
            "n8_clusters": n_clust,
            "win8_frac": float((d > 0).mean()),
            "outcome8": SD.outcome(lo, hi),
        })
    recs.sort(key=lambda r: (r["family"], r["arm"]))
    assert recs, f"no arm had a paired cell against `{scan}` at {size:g}M"

    conj = [r for r in recs if r["family"] not in OOD.CONJUNCTION_EXCLUDES
            and r["arm"] not in OOD.CONJUNCTION_EXCLUDE_ARMS]
    counts = {v: sum(1 for r in conj if r["outcome8"] == v) for v in OOD.VERDICTS}
    worst = min(conj, key=lambda r: r["delta8"]) if conj else None

    # ------------------------------------------------------------------ the document
    o = R.Out(
        out_dir,
        f"OOD arms against the 10M English training corpus: `{set_name}`",
        [
            f"Base `{base}`, {len(recs)} arms x {recs[0]['n']} targets (kept pairs), corpus search "
            f"over the **English 10M training corpus** (`corpora.train10m`, dir "
            f"`{EN_CORPUS_DIR}`) at **{size:g}M tokens**, scan `{scan}`.",
            "",
            "Δ = the MAEM's unbiased best-of-8 minus the English corpus search's top-1 on the "
            "SAME target, paired; CI is the design's 10,000-resample percentile bootstrap over "
            "the arm's targets and the SE beside it is the document-clustered one "
            "(`results.common.cluster_bootstrap`). The outcome is the same three-state verdict "
            "`results/ood.py` reports: **exceeds** (CI above zero), **inconclusive** (CI covers "
            "zero), **reversed** (CI below zero).",
            "",
            "THE OWN-DOMAIN COLUMNS ARE `results/ood.py`'s NUMBERS, printed for comparison and "
            "not recomputed here. The two Δ columns are different contrasts: own-domain search "
            "reads the target's own language, the English search reads the training distribution.",
            "",
            f"Run under `MODAL_PROFILE={R.env_hint()}`.",
        ],
    )
    o.table(
        "arms_encorp", f"Arms — {src.label}",
        f"`bo8 (centred)` is `results.ood.centred_bo_k`'s recompute from `cos_centred.f16`, the "
        f"same number `results/ood.py`'s bo8 column carries. `en corpus top-1` is the top-1 window "
        f"of the {size:g}M English training corpus on the same target. `own corpus top-1` and "
        f"`own Δ` are read from `{arms_csv}`.",
        ["arm", "family", "n", "bo8 (centred)", f"en corpus top-1 ({size:g}M)", "Δ vs en corpus",
         "95% CI", "win", "outcome", "own corpus top-1", "own Mtok", "own Δ"],
        [[r["arm"], r["family"], r["n"], R.num(r["bo8_centred"]), R.num(r["en_top1"]),
          R.num(r["delta8"]), f"[{r['ci8_lo']:.3f}, {r['ci8_hi']:.3f}]", R.num(r["win8_frac"], 2),
          r["outcome8"],
          R.num(float(r["own_top1"]), 4) if r["own_top1"] else None,
          r["own_size_m"], R.num(float(r["own_delta8"]), 4) if r["own_delta8"] else None]
         for r in recs],
    )
    o.section("\n".join([
        f"### The level-1 conjunction, read against the English corpus",
        "",
        f"Over the {len(conj)} arms of the conjunction (`ufw_en`, the English anchor, and the "
        f"`diag` family are excluded exactly as in `results/ood.py`): "
        f"**{counts['exceeds']} exceed, {counts['inconclusive']} inconclusive, "
        f"{counts['reversed']} reversed**."
        + (f" The smallest margin is `{worst['arm']}` at {worst['delta8']:.4f} "
           f"[{worst['ci8_lo']:.4f}, {worst['ci8_hi']:.4f}]." if worst else ""),
    ]))

    # ------------------------------------------------------------------ the cells rows
    today = time.strftime("%Y-%m-%d")
    prov = f"results/ood_encorp.py on {set_name} @ {src.label}"
    crows: list[dict] = []

    def put(key, value, *, se=None, lo=None, hi=None, n=None, note=""):
        if value is None:
            return
        if lo is None or hi is None:
            lo = hi = None
        assert "\n" not in note, f"{key}: a newline in `note` would break the CSV record"
        crows.append({"key": key, "value": str(value), "se": "" if se is None else str(se),
                      "lo": "" if lo is None else str(lo), "hi": "" if hi is None else str(hi),
                      "n": "" if n is None else str(int(n)), "status": "provisional", "run": "R4",
                      "source": prov, "date": today, "note": note})

    for r in recs:
        cid = OOD.cell_id_of(r["arm"])
        put(f"ood.{cid}.encorp10.cos", R.num(r["en_top1"], 4), n=r["n"],
            note=f"top-1 of the {size:g}M English TRAINING corpus (train_parity_10m) on this "
                 f"arm's targets; not the arm's own corpus")
        put(f"ood.{cid}.endiff.cos.bo8", R.num(r["delta8"], 4), lo=R.num(r["ci8_lo"], 4),
            hi=R.num(r["ci8_hi"], 4), n=r["n"], se=R.num(r["se8_clustered"], 4),
            note=f"bo8 centred minus the {size:g}M English training-corpus top-1, paired per "
                 f"target; 10,000-resample percentile CI")
        put(f"ood.{cid}.endiff.verdict", r["outcome8"], n=r["n"],
            note="three-state verdict on the bo8 pair against the English training corpus")
    for key, want in (("nexceed", "exceeds"), ("ninconcl", "inconclusive"),
                      ("nreversed", "reversed")):
        put(f"ood.conj.endiff.{key}", counts[want], n=len(conj),
            note=f"level-1 conjunction against the English training corpus, {len(conj)} arms")
    if worst:
        put("ood.conj.endiff.min", R.num(worst["delta8"], 4), lo=R.num(worst["ci8_lo"], 4),
            hi=R.num(worst["ci8_hi"], 4), n=worst["n"], se=R.num(worst["se8_clustered"], 4),
            note=f"the SMALLEST per-arm margin over the {len(conj)} conjunction arms "
                 f"(`{worst['arm']}`)")

    frag = out_dir / f"cells_{src.label.replace('/', '_').replace(':', '__').replace('@', '_at_')}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(frag, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(OOD.CELLS_COLUMNS))
        w.writeheader()
        w.writerows(crows)
    R.write_csv(out_dir / "arms_encorp.csv",
                ["arm", "family", "n", "bo8_centred", "en_top1", "en_size_m", "delta8",
                 "ci8_lo", "ci8_hi", "se8_clustered", "n8_clusters", "win8_frac", "outcome8",
                 "own_top1", "own_size_m", "own_delta8"],
                [[r[k] for k in ("arm", "family", "n", "bo8_centred", "en_top1", "en_size_m",
                                 "delta8", "ci8_lo", "ci8_hi", "se8_clustered", "n8_clusters",
                                 "win8_frac", "outcome8", "own_top1", "own_size_m", "own_delta8")]
                 for r in recs])
    if cells_target is not None:
        import results.faithfulness as FA

        res = FA.write_cells(cells_target, crows, owned_keys(cfg))
        notes.append(f"cells merged into `{res['path']}`: {len(res['rewritten'])} rewritten, "
                     f"{len(res['appended'])} appended, {res['untouched']} untouched")
    else:
        notes.append(f"cells NOT merged (no --cells); the fragment is `{frag.name}`")

    for n in skipped:
        notes.append(n)
    for n in notes:
        o.note(n)
    path = o.finish([])
    print(f"[ood_encorp] {path}")
    print(f"[ood_encorp] conjunction: " + ", ".join(f"{v}={counts[v]}" for v in OOD.VERDICTS))
    if worst:
        print(f"[ood_encorp] smallest margin: {worst['arm']} {worst['delta8']:.4f} "
              f"[{worst['ci8_lo']:.4f}, {worst['ci8_hi']:.4f}]")
    if vol.missing:
        print(f"[ood_encorp] {len(vol.missing)} files were not on the volume")


if __name__ == "__main__":
    app()
