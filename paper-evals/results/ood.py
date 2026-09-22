#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9",
#                 "transformers>=4.44",   # the tokenizer, for the classifier ceilings
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
import warnings
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import precompute.common as PC  # noqa: E402
import results.common as R  # noqa: E402

# R7's regex classifier, the one definition shared by both layers (precompute/common.py:2883).
code_like = PC.code_like

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

BASE = "qwen36-27b"
# The design's level-1 conjunction is over 21 arms: "lang 8, code 8, math 4, `ufw_zh`" (§6).
# `formulas` is the `diag` family -- a tokenizer-boundary diagnostic, reported in its own row and
# outside the conjunction (§2) -- and `ufw_en` is the §8(a) pipeline check, reported as an arm but
# not counted: it is English, and the claim is about the arms that are not.
CONJUNCTION_EXCLUDES = ("diag",)
CONJUNCTION_EXCLUDE_ARMS = ("ufw_en",)


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


SIZE_RE = re.compile(r"^([0-9.]+)m$")


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


def declared_mu(cfg: dict, key: str, base: str) -> str:
    """The mean `config.yaml` DECLARES this checkpoint was trained to receive, as one string.

    `precompute.common.input_mu` refuses an entry with no `mu:` key at all rather than guessing,
    and `C_resolve_mu` puts the answer in the same spelling as a product-recorded mean. A bare
    checkpoint name is prefixed with the base, the way `reconstruction/stats_ood.tables` accepts
    `--maemm`, so both readers resolve the same key the same way.
    """
    full = key if "/" in key else f"{base}/{key}"
    return C_resolve_mu(PC.input_mu(cfg, full), base)


def assert_control_paired(cfg: dict, base: str, maemm_key: str, control_key: str) -> None:
    """SPEC §7 R4: a control may only be differenced against a MAEMM that DECLARES the same mean.

    LOUD, because the failure it guards is silent: the control column sits beside Δ and reads as
    "what the untrained base scores on the same targets", which it only is when both arms were
    injected at the same centring. Two cosines taken about two different means are two angles to
    two different vectors and their difference is not a margin.

    As of 2026-09-23 the base control `qwen36-27b/2026-09-16_base-control` declares the 27B
    `whiten_mu` path in `config.yaml` (it was `mu: null` before, which was an explicit statement
    that it took a RAW activation), so the pairing HOLDS for the intended pair -- rl-last16
    against that control. It does not hold for the old primary, which declares `stats/mu.f32`.
    """
    a, b = declared_mu(cfg, maemm_key, base), declared_mu(cfg, control_key, base)
    assert a == b and a != "unknown", (
        f"CONTROL NOT PAIRED (spec §7 R4): maemm {maemm_key!r} declares mu={a!r} and control "
        f"{control_key!r} declares mu={b!r} in config.yaml. The control column is the same targets "
        f"under the same injection with base weights; at two means it is a control for a different "
        f"experiment and the difference is not a difference. As of 2026-09-23 "
        f"`qwen36-27b/2026-09-16_base-control` declares the 27B whiten_mu path (it was `null` "
        f"before), so the intended pair -- rl-last16 against that control -- does pair; the old "
        f"primary `2026-09-10_rl-8x2048-full` declares `base/{{base}}/stats/mu.f32` and does not. "
        f"Fix the config or tabulate the two generations separately; nothing here will guess."
    )


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


def parse_scan_dir(name: str, set_name: str) -> tuple[str, float | None, str] | None:
    """(corpus dir, the M-token bound or None, the run tag) for a scan of `set_name`, or None.

    `common.scan_dir` writes `scan/<set>` for the one unbounded scan of the base's own corpus and
    `scan/<set>__<key>` otherwise, where `<key>` is `<corpus label>[__<M>m][__<tag>]`
    (precompute/scan.py). SPLIT, not matched: a regex with two optional trailing groups reads
    `…__ces_Latn__1m__mu-whiten` as one corpus called `ces_Latn__1m__mu-whiten`, and the whole
    whiten pass then goes missing -- the reader reports "no scan at that mean" and withholds every
    Δ, which looks exactly like the scans never having run. No corpus directory contains `__`, so
    splitting on it is unambiguous.

    The OOD branch's pre-rebase layout was `scan/<set>/<corpus>-<M>m/`; sets drawn before the
    rebase still carry it and are read by `reconstruction/stats_ood.py`, not here.
    """
    if name == set_name:
        return ("", None, "")
    if not name.startswith(set_name + "__"):
        return None
    parts = [p for p in name[len(set_name) + 2 :].split("__") if p]
    if not parts:
        return None
    corpus, rest = parts[0], parts[1:]
    mb = None
    if rest:
        m = SIZE_RE.match(rest[0])
        if m:
            mb, rest = float(m.group(1)), rest[1:]
    return (corpus, mb, "__".join(rest))


def scan_dirs_of(vol: R.Vol, base: str, set_name: str) -> dict[str, tuple[str, float | None]]:
    """{scan directory -> (corpus directory name, the M-token bound or None)} for this set."""
    out: dict[str, tuple[str, float | None]] = {}
    for d in vol.ls(f"base/{base}/scan"):
        got = parse_scan_dir(d, set_name)
        if got is not None:
            out[d] = (got[0], got[1])
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
#   asym     cos(h,      unit(act - mu))   the PRE-M0a scan convention
#   centred  cos(h - mu, unit(act - mu))   THE ONE THIS FILE READS since 2026-09-23
#   raw      cos(h,      unit(act))        both sides uncentred
#
# `centred` IS THE READ (M0a change 3; spec §2, "centred against centred on both sides of every
# arm"). Both maps below stay exactly as they are -- the columns are on the products and the CSV
# carries all three -- but nothing selects `asym` any more: **`cos_asym.f16` and `bo_a_64` stay on
# disk and are read by NO driver**, not here and not in `reconstruction/stats_ood.py`. They are
# kept because deleting a stored column makes every product written before today unreadable, not
# because anything prints them.
#
# A `centred` READ IS ONLY VALID AGAINST A SCAN RUN WITH `--centre`. Before M0a the scan's window
# side was uncentred (`normalize(h) @ v`) while its target side was centred, so differencing a
# doubly-centred MAEMM cosine against a singly-centred corpus top-1 was meaningless -- the defect
# `results/ood/tables.md:103-105` describes and the reason `cos_asym` was added at all
# (`SMOKES.md:4483-4490`). `precompute/scan.py --centre` is what buys this read.
# AND NOTHING HERE CAN CATCH THE MIX: `scan_dirs_of` filters scans by SET only, and
# `top1_by_corpus` is keyed on (corpus, mean) alone, so an OLD UNCENTRED scan of the same set at
# the same mean lands in the same cell as a `--centre` one and is differenced silently. The guard
# is a set directory that only centred scans ever wrote, not code.
BO64 = {"asym": "bo_a_64", "centred": "bo_c_64", "raw": "bo_64"}


# `rollouts_rel_of` MOVED to `reconstruction/stats_ood.rollouts_rel_from_readme` on 2026-09-22.
# `stats_ood.rollout_texts` still rebuilt the rollouts path from its `--stem` and so still had the
# defect this half was fixed for in 11b7cd0; one rule, in the layer both readers already import.


SCORE_ARRAY = {"asym": "cos_asym.f16", "centred": "cos_centred.f16", "raw": "cos.f16"}


def per_rollout_scores(vol: R.Vol, src: R.Source, which: str):
    """[N, n] per-rollout score of a scores directory, or None.

    `cos_*.f16` is [N, n, T] with NaN outside the kept tokens, so a rollout's score is the nanmax
    over its token axis -- the same reduction `score` itself does to build `max_cos_*`. Needed
    because the rollouts jsonl carries NO score, so "the top-1 rollout" cannot be read off it:
    appending in file order gives the FIRST SAMPLED draw at T = 1.0, an arbitrary one of 64.
    Review R3 asks for the top-1 BY SCORE -- the rollout the headline best-of-64 is about.
    """
    import numpy as _np

    name = SCORE_ARRAY[which]
    idx = vol.json(f"{src.scores_rel}/index.json") or {}
    meta = idx.get(name)
    if not meta or "shape" not in meta:
        return None
    shape = tuple(int(x) for x in meta["shape"])
    arr = vol.array(f"{src.scores_rel}/{name}", "float16", shape)
    if arr is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # an all-NaN rollout is legal; it scores NaN
        return _np.nanmax(arr.astype(_np.float32), axis=-1)


def ranked_texts(vol: R.Vol, src: R.Source, which: str, rows: list[dict]):
    """({row -> [text] best-scoring FIRST}, a note). Falls back to file order, saying so."""
    import numpy as _np

    by_row: dict[int, list[str]] = {}
    for r in rows:
        by_row.setdefault(int(r["row"]), []).append(r.get("text", ""))
    sc = per_rollout_scores(vol, src, which)
    if sc is None:
        return by_row, (
            f"`{src.scores_rel}/{SCORE_ARRAY[which]}` is not on the volume, so the language-id "
            f"column is taken on rollout k=0 -- the FIRST SAMPLED draw, not the top-1 by score. "
            f"Read it as `lid_k0_rate`."
        )
    # CROSS-CHECK against the number `score` wrote for the same reduction. If these disagree the
    # ranking is against a different array than the table's cosines and the column is not R3's.
    worst, n_chk = 0.0, 0
    for row, rec in src.per_target.items():
        want = rec.get({"asym": "max_cos_asym", "centred": "max_cos_centred", "raw": "max_cos"}[which])
        if want is None or row >= sc.shape[0]:
            continue
        got = float(_np.nanmax(sc[row]))
        worst = max(worst, abs(got - float(want)))
        n_chk += 1
    assert n_chk and worst < 0.01, (
        f"per-rollout {which} scores disagree with {src.scores_rel}/per_target.jsonl by {worst:.5f} "
        f"over {n_chk} rows: the ranking would not be by the cosine this table reports"
    )
    out = {}
    for row, texts in by_row.items():
        if row < sc.shape[0] and len(texts) <= sc.shape[1]:
            order = _np.argsort(-_np.nan_to_num(sc[row][: len(texts)], nan=-2.0))
            out[row] = [texts[i] for i in order]
        else:
            out[row] = texts
    return out, f"language-id taken on the top-1 BY SCORE ({which}); max |check| {worst:.5f}"


def has_asym(src: R.Source) -> bool:
    """Does this scores directory carry the asymmetric cosine at all?

    A directory scored before `cos_asym` existed has only the two symmetric columns. It is not
    wrong, it is SUPERSEDED by the `--score-tag asym` re-score of the same rollouts, and reporting
    it as a table of skipped arms would read as a failure rather than as an older product.
    """
    return any("bo_a_64" in r for r in src.per_target.values())


def bo64_of(rec: dict, which: str) -> float | None:
    """The unbiased best-of-64 of one target in one of the three cosines, or None when absent.

    `score` writes `bo_c_*` and `bo_a_*` only when the run centred AND every rollout of the row
    was finite, so absence means "this run has no such number for this row" -- a different fact
    from a low one, and never filled with a zero.
    """
    v = rec.get(BO64[which])
    return None if v is None else float(v)


def has_col(src: R.Source, which: str) -> bool:
    """Does this scores directory carry the bo64 column this table actually READS?

    Sibling of `has_asym`, and the predicate the CONTROL column has to be chosen by: since M0a the
    read is `centred`, so picking the control by `bo_a_64` would happily pick a directory whose
    `bo_c_64` is empty -- exactly the silently-empty-column-beside-a-stated-Δ failure that choice
    exists to prevent.
    """
    return any(BO64[which] in r for r in src.per_target.values())


PANEL_BO_K = 8


def centred_bo_k(vol: R.Vol, src: R.Source, k: int = PANEL_BO_K) -> tuple[dict[int, float], str]:
    """({row: unbiased best-of-k of `cos_centred`}, a note) -- computed from the ARRAY.

    SPEC §2 NAMES `bo_c_8` AND NO SUCH COLUMN EXISTS anywhere in `paper-evals/`: `score` stores the
    centred ladder at k = 64 only (`BO64` above), so panel c's headline -- the Exemplifier's bo8
    against the own-domain corpus -- has to be recomputed from `cos_centred.f16` or it is silently
    absent from the one table the panel is drawn from.

    THE SAME PATH AS `results/faithfulness.centred_bok`, deliberately and not by coincidence: per
    (row, rollout) the best over the kept tokens -- `per_rollout_scores`, the nanmax `score` itself
    does, which is `results.common.best_per_rollout(..., empty=nan)` over the whole array at once --
    and then `results.common.bo_unbiased`, THE best-of-k estimator of this pipeline (M0a), over the
    rollouts. A NaN draw is a rollout with no kept centred token: it is DROPPED, never filled, and
    the estimator is applied to the finite ones at the same k. A row with fewer than k finite draws
    has NO bo-k and is left out rather than clamped to a smaller k, so the cell prints an em dash
    rather than a number of a different quantity.

    Reads the whole `cos_centred.f16`, which is the same array `ranked_texts` ranks the language-id
    column on; `Vol` caches by existence, so the two share one fetch.
    """
    sc = per_rollout_scores(vol, src, "centred")
    if sc is None:
        return {}, (
            f"`{src.scores_rel}/{SCORE_ARRAY['centred']}` is not on the volume, so this source has "
            f"no bo{k} column (a run that centred on nothing writes no centred cosine at all)"
        )
    order = [int(r) for r in src.rows_meta.get("rows", [])] or list(range(sc.shape[0]))
    out: dict[int, float] = {}
    n_nan_rows = n_short = 0
    for i, row in enumerate(order):
        if i >= sc.shape[0]:
            break
        best = sc[i]
        live = best[np.isfinite(best)]
        if live.size < best.size:
            n_nan_rows += 1
        if live.size < k:
            n_short += 1
            continue
        out[row] = float(R.bo_unbiased(live, k))
    note = f"bo{k} (centred) recomputed from `{SCORE_ARRAY['centred']}` on {len(out)} rows"
    if n_nan_rows:
        note += f"; {n_nan_rows} rows had a rollout with no kept centred token (dropped, not filled)"
    if n_short:
        note += f"; {n_short} rows had fewer than {k} finite draws and carry NO bo{k}"
    return out, note


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
    bo8: dict[int, float] | None = None,
    control_bo8: dict[int, float] | None = None,
) -> tuple[list[dict], list[str]]:
    """One record per arm: the cosines, the corpus cell, Δ with its CI, and the verdict.

    `bo8` / `control_bo8` are `centred_bo_k`'s {row: bo8} maps -- recomputed from the array, since
    no `bo_c_8` column exists -- and an arm whose rows are all absent from them carries None there,
    which prints as an em dash.
    """
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
        cen8, ctrl8 = [], []
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
            # SPEC §2's headline cell, and the only one in this table that is RECOMPUTED rather
            # than read: there is no `bo_c_8` column to read (`centred_bo_k`).
            if bo8 is not None:
                cen8.append(bo8.get(int(r["row"])))
            if control_bo8 is not None:
                ctrl8.append(control_bo8.get(int(r["row"])))
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
                "bo8_centred": _mean(cen8),
                "bo64_raw": _mean(raw),
                "corpus_top1": float(np.mean(corp)) if corp else None,
                "control_bo64": _mean(ctrl) if ctrl else None,
                "control_bo8": _mean(ctrl8),
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


def corpus_window_texts(vol: R.Vol, base: str, cdir: str, top1_rows, tok):
    """The decoded corpus top-1 WINDOW for each (row -> (doc, start)), or {} if unavailable.

    This text is in the arm's language BY CONSTRUCTION -- it is a window of that arm's own corpus
    -- so a classifier's rate on it is the classifier's CEILING, not a property of the MAEMM.
    Without the ceiling beside it a low rate is unreadable: it can mean the model did not produce
    the language, or that the classifier cannot name it (lid218e labels Chinese `yue_Hant`), or
    that the predicate needs a longer window than a rollout has (`code_like`).
    """
    docs = vol.jsonl(f"base/{base}/corpora/{cdir}/docs.jsonl") if cdir else None
    if docs is None or tok is None:
        return {}
    idx = vol.json(f"base/{base}/corpora/{cdir}/index.json") or {}
    meta = idx.get("tokens.i32")
    if not meta or "shape" not in meta:
        return {}
    toks = vol.array(f"base/{base}/corpora/{cdir}/tokens.i32", "int32",
                     tuple(int(x) for x in meta["shape"]))
    if toks is None:
        return {}
    span = {int(d["doc"]): (int(d["offset"]), int(d["len"])) for d in docs}
    out = {}
    for row, (doc, start) in top1_rows.items():
        if doc not in span:
            continue
        off, ln = span[doc]
        a = off + start
        b = min(a + PC.SCAN_BLOCK, off + ln)
        if b > a:
            out[row] = tok.decode([int(t) for t in toks[a:b]])
    return out


def load_tokenizer(cfg: dict, base: str):
    """The base's tokenizer, or None with the reason printed. Only the ceilings need it."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        print("[ood] transformers not installed: no classifier ceilings "
              "(`uv run --with transformers ...`)", flush=True)
        return None
    try:
        return AutoTokenizer.from_pretrained(cfg["bases"][base]["hf"])
    except Exception as e:  # noqa: BLE001 -- an absent cache is a skip, not a failure
        print(f"[ood] tokenizer unavailable ({type(e).__name__}): no classifier ceilings", flush=True)
        return None


def lid_rates(mod, vol: R.Vol, cfg: dict, ids: list[dict], src: R.Source, set_name: str,
              lid_model: Path | None, which: str = "centred",
              ceilings: dict[str, dict] | None = None) -> tuple[dict[str, dict], list[str]]:
    """{arm -> {lid_top1_rate, lid_top4_rate | code_like_top1_rate}} (review R3), and a note.

    The classifier is chosen by the ARM, not by the source: `ood_arms.<arm>.lid` is a list of the
    NLLB labels that count as this arm's language, and `null` there says fastText is not
    meaningful for it -- the code and maths arms, which report the `code_like` regex rate instead.
    """
    rel = mod.rollouts_rel_from_readme(vol, src.scores_rel)
    if rel is None:
        return {}, [f"`{src.scores_rel}/README.md` does not name its rollouts file: lid not run"]
    # Read the path the README gave, directly. `stats_ood.rollout_texts` composes
    # `maemms/{base}/{maemm}/...` from its own arguments, and `src.maemm` is the CONFIG KEY --
    # `<base>/<name>` -- so handing it that doubles the base and the fetch silently returns
    # nothing. One path, from the producer, is the whole point of reading the README.
    rows = vol.jsonl(rel)
    if not rows:
        return {}, [f"no rollout texts at {rel}: lid not run"]
    texts, rank_note = ranked_texts(vol, src, which, rows)
    model = mod.load_lid(lid_model)
    arms = cfg["ood_arms"]
    ceilings = ceilings or {}
    out: dict[str, dict] = {}
    per_arm: dict[str, list[tuple[list[str], list[str] | None]]] = {}
    for r in ids:
        t = texts.get(int(r["row"]))
        if t:
            per_arm.setdefault(r["arm"], []).append((t, arms[r["arm"]].get("lid")))
    for arm, items in per_arm.items():
        want = items[0][1]
        ceil_txt = ceilings.get(arm) or {}
        if want is None:
            # `code_like` needs R7's 512-token window; a rollout is <= 64 tokens and for SQL none
            # of its nine markers can occur at all. Reported WITH the ceiling measured on the
            # arm's own corpus windows, and marked not measurable when that ceiling is at or
            # below the rate -- the column then carries no information either way.
            rates = [1.0 if code_like(t[0]) else 0.0 for t, _ in items]
            ceil = (float(np.mean([1.0 if code_like(x) else 0.0 for x in ceil_txt.values()]))
                    if ceil_txt else None)
            out[arm] = {"lid_top1_rate": None, "lid_top4_rate": None,
                        "code_like_top1_rate": float(np.mean(rates)),
                        "ceiling": ceil, "ceiling_kind": "code_like"}
            continue
        if model is None:
            out[arm] = {"lid_top1_rate": None, "lid_top4_rate": None, "code_like_top1_rate": None,
                        "ceiling": None, "ceiling_kind": "lid"}
            continue
        t1, t4 = [], []
        for t, _ in items:
            labs = [mod.lid_label(model, x)[0] for x in t[:4]]
            t1.append(1.0 if labs and labs[0] in want else 0.0)
            t4.append(1.0 if any(x in want for x in labs) else 0.0)
        ceil = (float(np.mean([1.0 if mod.lid_label(model, x)[0] in want else 0.0
                               for x in ceil_txt.values()])) if ceil_txt else None)
        out[arm] = {"lid_top1_rate": float(np.mean(t1)), "lid_top4_rate": float(np.mean(t4)),
                    "code_like_top1_rate": None, "ceiling": ceil, "ceiling_kind": "lid"}
    notes = [rank_note]
    if model is None:
        notes.append(
            "fastText lid218e was not loadable, so the language columns are absent on the "
            "lang/ctrl arms (run with `--with fasttext --with huggingface-hub`)"
        )
    return out, notes


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
    ceiling: Annotated[
        bool, typer.Option(help="measure each classifier's ceiling on the arm's OWN corpus "
                                "top-1 windows; needs transformers for the tokenizer")
    ] = True,
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

    # The control column must come from a directory that HAS the cosine this table READS, or the
    # column is silently empty beside a Δ that is stated in it. Since M0a that is `centred`, not
    # `asym` -- see `has_col`.
    ctrls = [s for s in usable if s.role == "control"]
    ctrl_src = (next((s for s in ctrls if has_col(s, "centred")), None)
                or (ctrls[0] if ctrls else None))
    control = ctrl_src.per_target if ctrl_src else None
    for s_ in ctrls:
        if s_ is not ctrl_src:
            notes.append(f"control `{s_.label}` not used: superseded by `{ctrl_src.label}`")
    # SPEC §7 R4, first half: THE CONTROL IS RESOLVED FROM CONFIG WITH ITS CENTRING DECLARED.
    # `input_mu` refuses a `maemms:` entry that has no `mu:` key at all, so a control whose
    # convention was never established stops the table here instead of becoming an unlabelled
    # column; the pairing itself is asserted per source below, where the MAEMM is known.
    if ctrl_src is not None:
        notes.append(
            f"control `{ctrl_src.label}` is `{ctrl_src.maemm}`, which config.yaml declares at "
            f"mu={declared_mu(cfg, ctrl_src.maemm, base)}; the run itself recorded "
            f"mu={C_resolve_mu(ctrl_src.mu, base)}"
        )
    # Panel c's tick (spec §4: "untrained base bo8"), recomputed once from the control's own
    # `cos_centred.f16` and shared by every source's table.
    ctrl_bo8: dict[int, float] = {}
    if ctrl_src is not None:
        ctrl_bo8, ctrl_bo8_note = centred_bo_k(vol, ctrl_src)
        notes.append(f"control `{ctrl_src.label}`: {ctrl_bo8_note}")

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

    # The classifiers' ceilings, measured on each arm's OWN corpus top-1 windows at the size the
    # claim is read at. Computed once and shared by every source: it is a property of the corpus
    # and the classifier, not of a checkpoint.
    ceilings: dict[str, dict] = {}
    tok = load_tokenizer(cfg, base) if ceiling else None
    if tok is not None:
        for arm, cdir in sorted(dir_of_arm.items()):
            d = next((d for d, (c, mb) in scans.items() if c == cdir and (mb or 0) == size_m), None)
            if d is None:
                continue
            rows_ = vol.jsonl(f"base/{base}/scan/{d}/topk.jsonl") or []
            want = {}
            for r in rows_:
                if r.get("set", set_name) == set_name and r.get("top") and float(r["size"]) == size_m:
                    want[int(r.get("set_row", r["row"]))] = (int(r["top"][0][0]), int(r["top"][0][1]))
            got = corpus_window_texts(vol, base, cdir, want, tok)
            if got:
                ceilings[arm] = got
        print(f"[ood] classifier ceilings from {len(ceilings)} arms' own corpus windows", flush=True)

    verdicts: dict[str, dict[str, str]] = {}
    superseded = [
        s_ for s_ in usable
        if s_.role != "control" and not has_asym(s_)
        and any(o.maemm == s_.maemm and has_asym(o) for o in usable)
    ]
    for s_ in superseded:
        notes.append(
            f"source `{s_.label}` was scored before `cos_asym` existed and is superseded by the "
            f"`--score-tag asym` re-score of the SAME rollouts; not tabulated"
        )
    for src in usable:
        if src.role == "control" or src in superseded:
            continue
        got_mu = C_resolve_mu(src.mu, base)
        # SPEC §7 R4, second half: the control column that is about to be printed beside this
        # source's Δ must be a control FOR THIS SOURCE. Asserted on what config DECLARES, which is
        # the thing R4 asks for and the thing a reader of the table can check.
        if ctrl_src is not None:
            assert_control_paired(cfg, base, src.maemm, ctrl_src.maemm)
            # The declared pairing is config-level and cannot separate two RUNS of the same control
            # key at two means, which is exactly what this set carries (`…@vllm` and
            # `…@vllm:mu-stats__asym`). A note, not an assertion: the column is still the right
            # checkpoint, and which run is chosen is `has_col`'s order, not a declared fact.
            ctrl_run_mu = C_resolve_mu(ctrl_src.mu, base)
            if ctrl_run_mu != got_mu:
                notes.append(
                    f"CONTROL RUN MEAN DIFFERS: source `{src.label}` recorded mu={got_mu} and the "
                    f"control run `{ctrl_src.label}` recorded mu={ctrl_run_mu}. Both checkpoints "
                    f"DECLARE the same mean, so the pairing assertion passes, but the two products "
                    f"were scored about different vectors and the `control bo64`/`control bo8` "
                    f"cells are not this source's control. Read them as a separate measurement."
                )
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
        src_bo8, bo8_note = centred_bo_k(vol, src)
        notes.append(f"`{src.label}`: {bo8_note}")
        recs, skipped = arm_rows(
            ids, src, top1_by_arm, size_m, "centred", mod.boot_ci, mod.outcome, control,
            comparable=comparable, bo8=src_bo8, control_bo8=ctrl_bo8,
        )
        notes += skipped
        lids: dict[str, dict] = {}
        if lid:
            lids, lnotes = lid_rates(
                mod, vol, cfg, ids, src, set_name, lid_model, "centred", ceilings
            )
            notes += [f"`{src.label}`: {x}" for x in lnotes if x]
        lang_col = []
        for r in recs:
            li = lids.get(r["arm"], {})
            r.update({k: li.get(k) for k in ("lid_top1_rate", "lid_top4_rate",
                                             "code_like_top1_rate", "ceiling", "ceiling_kind")})
            ceil, kind = r.get("ceiling"), r.get("ceiling_kind")
            rate = r["lid_top1_rate"] if r["lid_top1_rate"] is not None else r["code_like_top1_rate"]
            if rate is None:
                lang_col.append("—")
            elif ceil is None:
                lang_col.append(R.num(rate, 3) + (" (code)" if kind == "code_like" else ""))
            elif kind == "code_like":
                # NOT MEASURABLE, whatever the two numbers are. `code_like` needs 3 of 9 markers
                # and is calibrated for R7's 512-token windows; a rollout is <= 64 tokens, and on
                # SQL none of the nine markers can occur at all. The ceiling below -- the rate on
                # the arm's OWN corpus windows, which are genuine code -- is 0.01-0.26, so the
                # predicate barely fires on the thing it is supposed to detect and the rollout
                # rate is uninterpretable in either direction.
                lang_col.append(f"n/m ({R.num(rate, 2)} vs ceiling {R.num(ceil, 2)})")
            else:
                lang_col.append(f"{R.num(rate, 3)} / {R.num(ceil, 2)}")
        verdicts[src.label] = {r["arm"]: r["outcome"] for r in recs}
        header = ["arm", "family", "n", "bo8 (centred)", "bo64 (centred)", f"corpus {size_m:g}M",
                  "control bo8", "control bo64", "Δ", "95% CI", "win", "outcome",
                  "lang / ceiling"]
        rows_md = [
            [r["arm"], r["family"], r["n"], R.num(r["bo8_centred"]), R.num(r["bo64_centred"]),
             R.num(r["corpus_top1"]), R.num(r["control_bo8"]), R.num(r["control_bo64"]),
             R.num(r["delta"]), f"[{R.num(r['ci_lo'], 3)}, {R.num(r['ci_hi'], 3)}]",
             R.num(r["win_frac"], 2), r["outcome"], lc]
            for r, lc in zip(recs, lang_col, strict=True)
        ]
        csv_header = ["arm", "family", "n", "bo8_centred", "bo64_centred", "bo64_asym", "bo64_raw",
                      "corpus_top1",
                      "corpus_size_m", "control_bo8", "control_bo64", "delta", "ci_lo", "ci_hi",
                      "se_clustered", "n_clusters", "win_frac", "outcome",
                      "lid_top1_rate", "lid_top4_rate", "code_like_top1_rate",
                      "classifier_ceiling", "ceiling_kind"]
        csv_rows = [
            [r["arm"], r["family"], r["n"], r["bo8_centred"], r["bo64_centred"], r["bo64_asym"],
             r["bo64_raw"], r["corpus_top1"],
             size_m, r["control_bo8"], r["control_bo64"], r["delta"], r["ci_lo"], r["ci_hi"],
             r["se_clustered"],
             r["n_clusters"], r["win_frac"], r["outcome"], r["lid_top1_rate"],
             r["lid_top4_rate"], r["code_like_top1_rate"], r.get("ceiling"), r.get("ceiling_kind")]
            for r in recs
        ]
        o.table(
            f"arms_{src.label.replace('/', '_').replace(':', '__').replace('@', '_at_')}",
            f"Arms — {src.label}"
            + ("" if src.centred else "  (this run centred on NOTHING: `--mu none`)"),
            f"bo64 against the in-domain {size_m:g}M corpus search, paired per target. BOTH "
            f"SIDES ARE THE CENTRED CONVENTION -- `cos(h - mu, unit(act - mu))`, spec §2's "
            f"'centred against centred' -- which holds ONLY IF the scan behind the corpus column "
            f"ran with `--centre`; nothing in this file can verify that, and an uncentred scan of "
            f"the same set at the same mean lands in the same cell. `bo8 (centred)` is "
            f"RECOMPUTED from `cos_centred.f16` -- no `bo_c_8` column is stored anywhere -- and "
            f"an em dash there is a row with fewer than 8 finite centred draws, never a zero. The "
            f"asymmetric and raw cosines are in the CSV and are read by nothing. Δ is still bo64 "
            f"minus the corpus top-1; re-pointing the verdict at bo8 is M5's. The control column "
            f"is "
            f"{'`' + ctrl_src.label + '`' if ctrl_src else 'absent'}. `lang / code` is the rate at "
            f"which the top-1 rollout comes back in the arm's own language (fastText lid218e), or "
            f"the code-like rate where fastText is not meaningful. **READ THE CONVENTION NOTE "
            f"BELOW BEFORE COMPARING Δ WITH THE DESIGN'S +0.218.**",
            header, rows_md, csv_header=csv_header, csv_rows=csv_rows,
        )

        conj = [
            r for r in recs
            if r["family"] not in CONJUNCTION_EXCLUDES and r["arm"] not in CONJUNCTION_EXCLUDE_ARMS
        ]
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
                    f"{len(conj)} arms of the conjunction (design §6: lang 8, code 8, math 4, "
                    f"`ufw_zh`; the `diag` arm `formulas` and the §8(a) English pipeline check "
                    f"`ufw_en` are reported as rows but not counted):",
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
            "SINCE M0a (2026-09-23) BOTH SIDES ARE CENTRED, and about the SAME constant. `score` "
            "centres on the scoring constant -- `bases.<base>.whiten_mu`, read by "
            "`precompute.common.score_mu`, the one mean both arguments of every centred cosine "
            "are taken about and deliberately decoupled from each MAEMM's injection `mu:` -- and "
            "`precompute/scan.py --centre` subtracts the same constant from every corpus window "
            "before the dot product. `cos_centred` on the MAEMM side against a `--centre` scan's "
            "top-1 is therefore one convention applied to both sides, which is what makes Δ a "
            "paired difference rather than the subtraction of two angles to two different "
            "vectors. Second sentence of the same fact: this file no longer reads `cos_asym` at "
            "all, and `bo_a_64` / `cos_asym.f16` remain on disk read by no driver.",
            "",
            "WHAT THIS FILE CANNOT CHECK, stated because the failure is silent. `scan_dirs_of` "
            "selects scans by SET, and `top1_by_corpus` is keyed on (corpus, mean) alone: a scan "
            "run WITHOUT `--centre`, at the same mean and on the same set, lands in exactly the "
            "same cell and would be differenced without a word. Nothing in a scan's `topk.jsonl` "
            "distinguishes the two modes; the scan's own README states which mode it ran in, and "
            "the operational guard is a set directory that only centred scans ever wrote. Before "
            "M0a the window side was uncentred while the target side was centred, so a `centred` "
            "read was meaningless and `cos_asym` existed precisely to work around it "
            "(`results/ood/tables.md:103-105`, `SMOKES.md:4483-4490`).",
            "",
            "Consequences, stated rather than smoothed:",
            "",
            "- the paper's frozen English numbers are ASYMMETRIC-convention numbers (bo64 0.569, "
            "corpus 0.351 at 4M, margin +0.218 / +0.256 at 4M / 1M), so neither the bo8/bo64 "
            "columns above nor Δ is comparable with them by MAGNITUDE. The English reference "
            "table above is recomputed from the frozen 2026-09-16 UNCENTRED scan and stays in its "
            "own convention on purpose;",
            "- the per-arm claim is a SIGN ('the best of k rollouts aligns more closely than the "
            "best corpus window'), and a sign is testable under any one convention applied to "
            "both sides -- which is what the outcome column reports;",
            "- `bo8 (centred)` is RECOMPUTED from `cos_centred.f16` through the pipeline's one "
            "best-of-k estimator (`results.common.bo_unbiased`, the unbiased order statistic over "
            "all finite draws), because `score` stores the centred ladder at k = 64 only and no "
            "`bo_c_8` column exists anywhere. A row with fewer than 8 finite centred draws carries "
            "NO bo8 and prints an em dash rather than a clamped k, which would be a different "
            "quantity under the same name;",
            "- all three cosines are in the CSV, so the arms can be re-read under any of them "
            "without re-running anything on the GPU.",
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
