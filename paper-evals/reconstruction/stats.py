#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""The analysis layer of paper-evals: every table the paper reads, from the small files on the volume.

Local, CPU, no GPU and no model. It fetches ONLY small files (per_target.jsonl, ids.jsonl,
rows.json, index.json, per_feature.jsonl, finals.jsonl, quantiles.f16, topk.jsonl, cos.f16,
argmax.i16 and the two `centred` arrays) into `reconstruction/data/<root-tag>/`, mirroring the
volume's own paths, and writes markdown + CSV into `reconstruction/out/<root-tag>/`. Both are
gitignored. `best_act.f16` is NEVER fetched (335 MB per MAEMM at full scale); the centred cosine
that needs it is computed ON the volume by `precompute/centred.py` and read back as [N, n].

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-precompute/paper-evals/reconstruction/stats.py --root-tag smoke)

Tables (reconstruction/README.md has what each one means and the estimator formula):

  (a) per family x MAEMM: mean cos and best-of-n for n in {1,2,4,8,16,32,64}, UNBIASED
  (b) bo-sensitivity: the MAEMM ranking per family at bo 1 / 4 / 16 / 64 (checklist item 27)
  (c) paired MAEMM comparison on the 27B, per family and per SAE density stratum (items 25, 63)
  (d) the corpus-scan baseline per corpus size, plus the scan's cos quantiles (item 67)
  (e) the SAE repo's own max-activating windows (the `sae-repo-top32` column)
  (f) GCG / EPO finals against the MAEMM on the same rows (the reachability ceiling)
  (g) argmax-position distribution per family x MAEMM (item 10)
  (h) the centred and norm-filtered secondaries against the primary (items 4, 60)
  (i) distribution summaries for the sae family: quantiles of per-target best-of-64 (item 61)
  (j) the Patchscopes floor and injected cells against the MAEMM on the same rows (item 12)
  (k) per-FEATURE sae: the MAEMM against corpus search and the SAE repo's own windows
  (l) deciles of the per-feature difference (MAEMM - corpus top-1)

Every table carries n, bo, seed, the MAEMM weight sha, the set name and the corpus size (item 29),
in its caption or in its own columns. A product that is missing on the chosen root makes its table
print a note and be skipped -- never an exception: the smoke root legitimately has 8-row scores,
one scan size and only some GCG arms.
"""

from __future__ import annotations

import json
import math
import subprocess
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
CONFIG = PAPER_EVALS / "config.yaml"
VOLUME = "maemm"
# --root-tag -> the volume-relative prefix the pipeline's --root wrote under. "" is the volume root
# (--root /vol), which is where the full run lands.
ROOTS = {"smoke": "runs/2026-09-15_paper-evals-smoke", "full": ""}
BO_KS = (1, 2, 4, 8, 16, 32, 64)
BO_RANK = (1, 4, 16, 64)
SCAN_QUANTILES = (0.50, 0.90, 0.99, 0.999, 0.9999)
ARGMAX_BINS = 8
SAE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------------------------
# fetching (modal volume get, one small file at a time)
# ---------------------------------------------------------------------------------------------


class Vol:
    """The volume, seen through `modal volume ls/get`, with a local mirror and a miss log.

    Every fetch is cached by existence, so re-running the tables costs no network. `--refetch`
    forces. A file that is not there is recorded in `self.missing` and returns None -- the caller
    decides whether its table can still be built.
    """

    def __init__(
        self, root_tag: str, data_dir: Path, modal_cmd: str, refetch: bool, quiet: bool, offline: bool
    ):
        assert root_tag in ROOTS, f"--root-tag must be one of {sorted(ROOTS)}, got {root_tag!r}"
        self.prefix = ROOTS[root_tag]
        self.local = data_dir
        self.cmd = modal_cmd.split()
        self.refetch = refetch
        self.quiet = quiet
        self.offline = offline  # --no-fetch: answer everything from the local mirror
        self.missing: list[str] = []
        self.fetched = 0
        self.bytes = 0

    def _remote(self, rel: str) -> str:
        return f"{self.prefix}/{rel}".lstrip("/") if self.prefix else rel

    def ls(self, rel: str) -> list[str]:
        """Basenames under a volume directory, or [] when it does not exist."""
        if self.offline:
            d = self.local / rel
            return sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        r = subprocess.run(
            [*self.cmd, "volume", "ls", VOLUME, self._remote(rel)], capture_output=True, text=True
        )
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith(("Directory", "─", "┃", "│", "┏", "┡", "└")):
                out.append(line.rstrip("/").split("/")[-1])
        return sorted(out)

    def get(self, rel: str) -> Path | None:
        """Fetch one file; returns its local path, or None (and logs) when it is not on the volume."""
        dst = self.local / rel
        if dst.exists() and not self.refetch:
            return dst
        if self.offline:
            self.missing.append(rel)
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [*self.cmd, "volume", "get", "--force", VOLUME, self._remote(rel), str(dst)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not dst.exists():
            self.missing.append(rel)
            return None
        self.fetched += 1
        self.bytes += dst.stat().st_size
        if not self.quiet:
            console.print(f"[dim]fetched {rel} ({dst.stat().st_size / 1e6:.2f} MB)[/dim]")
        return dst

    def jsonl(self, rel: str) -> list[dict] | None:
        p = self.get(rel)
        if p is None:
            return None
        with open(p) as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def json(self, rel: str) -> dict | None:
        p = self.get(rel)
        if p is None:
            return None
        with open(p) as fh:
            return json.load(fh)

    def array(self, rel: str, dtype: str, shape) -> np.ndarray | None:
        p = self.get(rel)
        if p is None:
            return None
        return np.fromfile(p, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------------


def bo_weights(N: int, k: int) -> np.ndarray:
    """Order-statistic weights of the UNBIASED best-of-k estimator from N per-rollout scores.

        E[max of k draws] = sum_{i=1..N} x_(i) * C(i-1, k-1) / C(N, k)          (x sorted ASCENDING)

    The i-th smallest of the N observed scores is the maximum of a k-subset exactly when the other
    k-1 members come from the i-1 scores below it, and every k-subset of the N is equally likely.
    This is unbiased for any k <= N and uses ALL N rollouts, unlike the disjoint-group estimator
    `common.best_of_k_means` the products store (which averages floor(N/k) group maxima and is the
    same number only at k = N). Checklist item 25.
    """
    assert 1 <= k <= N, f"best-of-{k} asked of {N} rollouts"
    denom = math.comb(N, k)
    return np.array([math.comb(i - 1, k - 1) / denom for i in range(1, N + 1)], dtype=np.float64)


def bo_unbiased(best: np.ndarray, k: int) -> np.ndarray:
    """[N_targets] per-target unbiased best-of-k, from [N_targets, N] per-rollout scores."""
    N = best.shape[1]
    return np.sort(best, axis=1) @ bo_weights(N, k)


def bo_naive(best: np.ndarray, k: int) -> np.ndarray:
    """`common.best_of_k_means` per target: disjoint consecutive groups of k, mean of the maxima."""
    N = best.shape[1]
    g = N // k
    return best[:, : g * k].reshape(best.shape[0], g, k).max(2).mean(1)


def se(x: np.ndarray) -> float:
    """Standard error of a mean ACROSS TARGETS. Rollouts of one target are correlated, so this is
    the right scale only when x is one value per target -- which is how every table here uses it."""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def sign_test(d: np.ndarray) -> tuple[float, float, int]:
    """(two-sided exact sign-test p, win fraction, number of non-ties) for paired differences."""
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    m = pos + neg
    if m == 0:
        return float("nan"), float("nan"), 0
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) / 2.0**m)
    return p, pos / m, m


# Row order in every table: the PRIMARY MAEMM the paper's claims are about, then the
# untrained-base CONTROL it is read against, then the secondaries kept because their computation
# was already paid for. `role` comes from config.yaml (`primary: true` / `role: control`).
ROLE_ORDER = {"primary": 0, "control": 1, "secondary": 2}


def role_rank(s) -> int:
    return ROLE_ORDER.get(s.role, len(ROLE_ORDER))


def pm(mean: float, err: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {err:.{nd}f}" if np.isfinite(err) else f"{mean:.{nd}f}"


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


class Scores:
    """One (base, maemm, set) scores directory, with the per-rollout statistics recomputed.

    `per_target.jsonl`'s own aggregates are read too and ASSERTED against the recomputation from
    cos.f16: a disagreement means the fetched arrays and the fetched aggregates are not from the
    same run, which is exactly the failure a cached data/ directory could hide.
    """

    def __init__(self, vol: Vol, base: str, label: str, set_name: str, ids: dict, d: str):
        """`d` is the scores directory, volume-relative. `set_name` is the LOGICAL held-out set the
        rows index into (a `__vllm` stem and a `patchscopes` cell both resolve to the bare set), so
        every table joins on the same ids.jsonl."""
        self.base, self.maemm, self.set = base, label, set_name
        self.short = label
        self.dir = d
        # `primary: true` in config.yaml marks THE MAEMM the paper's claims are about (Tomas,
        # 2026-09-16: qwen36-27b/2026-09-10_rl-8x2048-full); every other row is secondary and is
        # shown because its computation was already paid for. Set by main(), not read here, so a
        # patchscopes cell (which has no config entry) simply stays False.
        self.primary = False
        self.role = ""
        idx = vol.json(f"{d}/index.json")
        rows_json = vol.json(f"{d}/rows.json")
        per_t = vol.jsonl(f"{d}/per_target.jsonl")
        if idx is None or rows_json is None or per_t is None or "cos.f16" not in idx:
            self.ok = False
            return
        n_t, n, width = idx["cos.f16"]["shape"]
        cos = vol.array(f"{d}/cos.f16", "float16", (n_t, n, width))
        argmax = vol.array(f"{d}/argmax.i16", "int16", (n_t, n))
        if cos is None or argmax is None:
            self.ok = False
            return
        cos = cos.astype(np.float32)
        self.rows = list(rows_json["rows"])
        self.n = int(rows_json["n"])
        assert self.n == n and len(self.rows) == n_t, f"{d}: rows.json disagrees with cos.f16's shape"
        self.kept = (~np.isnan(cos)).sum(2)
        self.best = np.where(np.isnan(cos).all(2), -1.0, np.nanmax(cos, axis=2))
        self.argmax = argmax.astype(np.int64)
        self.per_t = {int(r["row"]): r for r in per_t}
        self.fam = np.array([ids[r]["family"] for r in self.rows])
        self.seed = per_t[0].get("seed")
        self.sha = str(per_t[0].get("checkpoint_sha", ""))[:12]
        # the recomputation check -- per_target's mean_cos is the mean over the same n rollouts
        mine = self.best.mean(1)
        theirs = np.array([self.per_t[r]["mean_cos"] for r in self.rows])
        worst = float(np.abs(mine - theirs).max())
        assert worst < 2e-3, (
            f"{d}: per_target.jsonl's mean_cos and the mean recomputed from cos.f16 differ by up to "
            f"{worst:.5f} -- the fetched aggregates and arrays are not from the same run "
            f"(fp16 storage alone is worth <1e-3)"
        )
        self.recompute_max_diff = worst
        # the two `centred` secondaries, present only where precompute/centred.py has run
        self.centred = vol.array(f"{d}/cos_centred_best.f16", "float16", (n_t, n))
        self.filtered = vol.array(f"{d}/cos_filtered_best.f16", "float16", (n_t, n))
        if self.centred is not None:
            self.centred = self.centred.astype(np.float32)
        if self.filtered is not None:
            self.filtered = self.filtered.astype(np.float32)
        self.centred_json = vol.json(f"{d}/centred.json")
        self.ok = True

    def mask(self, family: str) -> np.ndarray:
        return self.fam == family

    @property
    def families(self) -> list[str]:
        return sorted(set(self.fam.tolist()))


def load_ids(vol: Vol, base: str, set_name: str) -> dict[int, dict] | None:
    rows = vol.jsonl(f"base/{base}/heldout/{set_name}/ids.jsonl")
    if rows is None:
        return None
    return {int(r["row"]): r for r in rows}


def discover(vol: Vol, cfg: dict) -> dict:
    """What actually exists on this root: sets per base, scores per (maemm, set), gcg arms, cells.

    Nothing is assumed from config.yaml except the names of the bases, MAEMMs and SAEs; the smoke
    root has a different set of products from the full one and both must work.
    """
    found: dict = {"bases": {}}
    for base in cfg["bases"]:
        sets = vol.ls(f"base/{base}/heldout")
        maemms = {}
        for key in cfg["maemms"]:
            if key.split("/")[0] != base:
                continue
            # score writes scores/<set> for the HF engine, scores/<set>__vllm for the vLLM one and
            # scores/<set>__rescore-* for the rescore variant. The first two are both primary --
            # the 27B's rollouts go through vLLM -- and are labelled apart; __rescore-* is not a
            # rollout grid and never enters these tables.
            got = []
            for stem in vol.ls(f"maemms/{key}/scores"):
                if stem in sets:
                    got.append((stem, stem, ""))
                elif stem.endswith("__vllm") and stem[: -len("__vllm")] in sets:
                    got.append((stem, stem[: -len("__vllm")], "vllm"))
            if got:
                maemms[key] = got
        gcg = {}
        for s in sets:
            entries = vol.ls(f"base/{base}/gcg/{s}")
            arms = []
            for e in entries:
                if e.endswith(tuple(f".tmp-{x}" for x in ("",))) or ".tmp-" in e:
                    continue  # a run still in flight
                sub = vol.ls(f"base/{base}/gcg/{s}/{e}")
                if "finals.jsonl" in sub:
                    arms.append(e)  # old layout: <set>/<arm>
                else:
                    arms += [f"{e}/{a}" for a in sub if ".tmp-" not in a]  # <set>/<family>/<arm>
            if arms:
                gcg[s] = arms
        # patchscopes cells: <root>/base/<base>/patchscopes/<set>/<cell>/scores/
        ps = {}
        for s_ in sets:
            cells = [
                c
                for c in vol.ls(f"base/{base}/patchscopes/{s_}")
                if ".tmp-" not in c and "scores" in vol.ls(f"base/{base}/patchscopes/{s_}/{c}")
            ]
            if cells:
                ps[s_] = cells
        saes = [k for k in cfg["saes"] if k.split("/")[0] == base]
        found["bases"][base] = {
            "sets": sets,
            "maemms": maemms,
            "gcg": gcg,
            "saes": saes,
            "patchscopes": ps,
        }
    return found


# ---------------------------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------------------------


class Out:
    """Collects the tables, prints them and writes `<out>/<letter>_<name>.{md,csv}` + index.md."""

    def __init__(self, out_dir: Path, root_tag: str, root_path: str):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root_tag, self.root_path = root_tag, root_path
        self.entries: list[tuple[str, str, str]] = []  # (letter, title, caption)
        self.notes: list[str] = []

    def table(self, letter: str, name: str, title: str, caption: str, df: pl.DataFrame) -> None:
        rt = RichTable(title=f"({letter}) {title}", caption=caption, header_style="bold")
        for c in df.columns:
            rt.add_column(c, justify="right" if df[c].dtype.is_numeric() else "left")
        for row in df.iter_rows():
            rt.add_row(*["" if v is None else str(v) for v in row])
        console.print(rt)
        stem = self.dir / f"{letter}_{name}"
        df.write_csv(stem.with_suffix(".csv"))
        md = [f"### ({letter}) {title}", "", f"*{caption}*", ""]
        md.append("| " + " | ".join(df.columns) + " |")
        md.append("|" + "|".join("---" for _ in df.columns) + "|")
        for row in df.iter_rows():
            md.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        md.append("")
        stem.with_suffix(".md").write_text("\n".join(md))
        self.entries.append((letter, title, caption))

    def skip(self, letter: str, why: str) -> None:
        console.print(f"[yellow]({letter}) skipped: {why}[/yellow]")
        self.notes.append(f"({letter}) skipped: {why}")

    def finish(self) -> None:
        lines = [
            f"# reconstruction/stats.py on the `{self.root_tag}` root",
            "",
            f"- volume root: `/vol/{self.root_path}`" if self.root_path else "- volume root: `/vol`",
            f"- command: `{' '.join(sys.argv)}`",
            "",
        ]
        for letter, title, caption in self.entries:
            lines += [f"- [({letter}) {title}]({letter}_*.md) — {caption}"]
        if self.notes:
            lines += ["", "## Skipped", ""] + [f"- {n}" for n in self.notes]
        (self.dir / "index.md").write_text("\n".join(lines) + "\n")
        console.print(f"[green]wrote {len(self.entries)} tables to {self.dir}[/green]")


def caption(sides, corpus: dict, extra: str = "") -> str:
    """The provenance line every table carries (checklist item 29): n, bo, seed, set, corpus.

    Built from the UNION over the tables' inputs, so a root that mixes sets or rollout budgets says
    so instead of quoting the first one. The per-MAEMM weight sha is a COLUMN, not a caption field.
    """
    uniq = lambda xs: ", ".join(str(x) for x in sorted({x for x in xs if x is not None}))  # noqa: E731
    return (
        f"set {uniq(s.set for s in sides)} | n {uniq(s.n for s in sides)} | "
        f"bo {uniq(s.n for s in sides)} | seed {uniq(s.seed for s in sides)} | "
        f"corpus {uniq(corpus.get(s.base, '?') for s in sides)}" + (f" | {extra}" if extra else "")
    )


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------


def table_a(out: Out, sides: list[Scores], corpus: dict) -> None:
    """(a) per family x MAEMM: mean cos and the unbiased best-of-k curve."""
    if not sides:
        return out.skip("a", "no scores directories on this root")
    rows = []
    for s in sides:
        for fam in s.families:
            m = s.mask(fam)
            b = s.best[m]
            rec = {
                "base": s.base,
                "maemm": s.short,
                "role": s.role,
                "family": fam,
                "targets": int(m.sum()),
                "sha": s.sha,
                "mean cos (bo1)": pm(float(b.mean()), se(b.mean(1))),
            }
            for k in BO_KS:
                if k > s.n:
                    continue
                v = bo_unbiased(b, k)
                rec[f"bo{k}"] = pm(float(v.mean()), se(v))
            rec[f"bo{s.n} naive"] = f"{float(bo_naive(b, s.n).mean()):.4f}"
            rows.append(rec)
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "a",
        "family_x_maemm",
        "mean cosine and unbiased best-of-n, per family x MAEMM",
        caption(
            sides,
            corpus,
            "rows are ordered PRIMARY, then the untrained-base CONTROL, then the secondaries "
            "(the `role` column; config.yaml `primary: true` / `role: control`); "
            "bo-k UNBIASED order statistic, ± SE across targets; the last column is the "
            "disjoint-group estimator the product stores, which equals the unbiased one at k = n",
        ),
        df,
    )


def table_b(out: Out, sides: list[Scores], corpus: dict) -> None:
    """(b) bo-sensitivity: does the MAEMM ranking survive the compute axis? (checklist item 27)"""
    by_base: dict[str, list[Scores]] = {}
    for s in sides:
        by_base.setdefault(s.base, []).append(s)
    # every MAEMM on this root gets a column, in one fixed order, so the table reads across bases
    shorts = sorted({s.short for s in sides})
    rows = []
    for base, group in sorted(by_base.items()):
        fams = sorted({f for s in group for f in s.families})
        for fam in fams:
            for k in BO_RANK:
                vals = {}
                for s in group:
                    m = s.mask(fam)
                    if not m.any() or k > s.n:
                        continue
                    vals[s.short] = float(bo_unbiased(s.best[m], k).mean())
                order = sorted(vals, key=lambda x: -vals[x])
                rows.append(
                    {
                        "base": base,
                        "family": fam,
                        "bo": k,
                        **{name: (round(vals[name], 4) if name in vals else None) for name in shorts},
                        "ranking": " > ".join(order) if len(order) > 1 else (order[0] if order else ""),
                    }
                )
    if not rows:
        return out.skip("b", "no scores directories on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "b",
        "bo_sensitivity",
        "MAEMM ranking per family at bo 1 / 4 / 16 / 64 (unbiased)",
        caption(
            sides,
            corpus,
            "a ranking that flips with bo is the point of the table (checklist item 27); "
            "a blank cell is a MAEMM with no rows of that family on this root",
        ),
        df,
    )


def table_c(out: Out, sides: list[Scores], ids_of: dict, corpus: dict) -> None:
    """(c) paired MAEMM comparison per family and per SAE density stratum (items 25, 63).

    A is the PRIMARY where one is declared, and it is paired against EVERY other side on the same
    (base, set) -- the untrained-base control first, then the secondaries -- so "primary minus
    control" is a row of this table rather than a subtraction the reader does by eye. The MAEMM
    names live in the `A` / `B` COLUMNS rather than in the column headers, because with a variable
    number of B sides a per-side header is not a fixed schema. `c_*.csv` is not consumed by the
    paper scripts (paper/inversion-eval/data/README-data.md: only a,d,e,g,i,j,k,l are), so the
    schema change stops here.
    """
    by_key: dict[tuple[str, str], list[Scores]] = {}
    for s in sides:
        by_key.setdefault((s.base, s.set), []).append(s)
    rows = []
    for (base, _set), group in sorted(by_key.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: (role_rank(x), x.short))
        a = group[0]
        ia = {r: i for i, r in enumerate(a.rows)}
        ids = ids_of[(base, _set)]
        for b in group[1:]:
            common = sorted(set(a.rows) & set(b.rows))
            if not common:
                continue
            ib = {r: i for i, r in enumerate(b.rows)}
            strata = {}
            for r in common:
                fam = ids[r]["family"]
                strata.setdefault((fam, "all"), []).append(r)
                if fam == "sae" and ids[r].get("stratum") is not None:
                    strata.setdefault((fam, f"density q{ids[r]['stratum']}"), []).append(r)
            for (fam, slice_), rs in sorted(strata.items()):
                ja = np.array([ia[r] for r in rs])
                jb = np.array([ib[r] for r in rs])
                k = min(a.n, b.n)
                for stat, fn in (
                    (f"best-of-{k} (unbiased)", lambda x, k=k: bo_unbiased(x, k)),
                    ("mean cos", lambda x: x.mean(1)),
                ):
                    va, vb = fn(a.best[ja]), fn(b.best[jb])
                    d = va - vb
                    p, win, m = sign_test(d)
                    rows.append(
                        {
                            "base": base,
                            "A": a.short,
                            "B": b.short,
                            "B role": b.role,
                            "family": fam,
                            "slice": slice_,
                            "statistic": stat,
                            "targets": len(rs),
                            "A value": round(float(va.mean()), 4),
                            "B value": round(float(vb.mean()), 4),
                            "diff (A-B)": pm(float(d.mean()), se(d)),
                            "A wins": f"{win:.3f}" if m else "",
                            "sign-test p": f"{p:.3g}" if m else "",
                        }
                    )
    if not rows:
        return out.skip("c", "fewer than two MAEMMs scored on one base, or no shared target rows")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "c",
        "paired_maemms",
        "paired per-target MAEMM comparison (same directions, same rollout budget)",
        caption(
            sides,
            corpus,
            "A is the PRIMARY MAEMM where one is declared, paired against EVERY other side on "
            "the same base and set (the untrained-base CONTROL first, then the secondaries -- "
            "`B role` says which). A - B is per target, then "
            "averaged; ± SE across targets; the sign test is two-sided "
            "exact over non-ties. SAE strata are the ids.jsonl corpus-density quartiles "
            "(q0 = rarest), checklist item 63",
        ),
        df,
    )


def table_d(out: Out, vol: Vol, found: dict, ids_by_base: dict, corpus: dict) -> None:
    """(d) the corpus-retrieval baseline and the scan's cosine quantiles, per corpus size."""
    rows = []
    for base, info in sorted(found["bases"].items()):
        for set_name in info["sets"]:
            topk = vol.jsonl(f"base/{base}/scan/{set_name}/topk.jsonl")
            idx = vol.json(f"base/{base}/scan/{set_name}/index.json")
            ids = ids_by_base.get((base, set_name))
            if topk is None or ids is None:
                continue
            q = None
            if idx and "quantiles.f16" in idx:
                shape = tuple(idx["quantiles.f16"]["shape"])
                arr = vol.array(f"base/{base}/scan/{set_name}/quantiles.f16", "float16", shape)
                q = None if arr is None else arr.astype(np.float32)
            sizes = sorted({int(r["size"]) for r in topk})
            for size in sizes:
                by_fam: dict[str, list[dict]] = {}
                for r in topk:
                    if int(r["size"]) == size:
                        by_fam.setdefault(r["family"], []).append(r)
                for fam, recs in sorted(by_fam.items()):
                    top1 = np.array([r["top"][0][3] for r in recs if r["top"]])
                    mean64 = np.array([float(np.mean([t[3] for t in r["top"]])) for r in recs if r["top"]])
                    rec = {
                        "base": base,
                        "set": set_name,
                        "family": fam,
                        "corpus": f"{size}M",
                        "targets": len(top1),
                        "top-1 cos": pm(float(top1.mean()), se(top1)),
                        "mean of top-64": pm(float(mean64.mean()), se(mean64)),
                    }
                    if q is not None:
                        si = sizes.index(size)
                        sel = [r["row"] for r in recs]
                        for qi, qv in enumerate(SCAN_QUANTILES):
                            rec[f"p{qv * 100:g}"] = round(float(q[sel, si, qi].mean()), 4)
                    rows.append(rec)
    if not rows:
        return out.skip("d", "no scan product on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    first = df.row(0, named=True)
    out.table(
        "d",
        "corpus_scan",
        "corpus-retrieval baseline and cosine quantiles, per nested corpus size",
        f"scan geometry 64/16 (common.windows_of) | sizes are nested prefixes | "
        f"corpus max {corpus.get(first['base'], '?')} | top-1 and mean-of-top-64 are per target, "
        f"then averaged ± SE; pXX are the scan's own cos quantiles over ALL windows, averaged over "
        f"targets -- the calibration of what a given cosine means (checklist item 67)",
        df,
    )


def table_e(out: Out, vol: Vol, found: dict) -> None:
    """(e) the SAE repo's own shipped max-activating windows: the `sae-repo-top32` column."""
    rows = []
    for base, info in sorted(found["bases"].items()):
        for sae in info["saes"]:
            for set_name in info["sets"]:
                d = f"base/{base}/sae/{sae.split('/')[-1]}/repo_examples/{set_name}"
                per_f = vol.jsonl(f"{d}/per_feature.jsonl")
                if per_f is None:
                    continue
                summ = vol.json(f"{d}/summary.json") or {}
                mx = np.array([r["max_cos"] for r in per_f])
                mn = np.array([r["mean_cos"] for r in per_f])
                ff = np.array([r["frac_fired"] for r in per_f])
                rr = np.array([r["peak_pearson_r"] for r in per_f if r["peak_pearson_r"] is not None])
                ag = np.array([r["argmax_agree"] for r in per_f if r.get("argmax_agree") is not None])
                rows.append(
                    {
                        "base": base,
                        "sae": sae,
                        "set": set_name,
                        "features": len(per_f),
                        "windows/feature": summ.get("windows_per_feature", ""),
                        "max_cos (the column)": pm(float(mx.mean()), se(mx)),
                        "mean_cos": pm(float(mn.mean()), se(mn)),
                        "frac fired": round(float(ff.mean()), 4),
                        "features ever firing": round(float((ff > 0).mean()), 4),
                        "peak r (mean)": round(float(rr.mean()), 4) if len(rr) else "",
                        "peak r (median)": round(float(np.median(rr)), 4) if len(rr) else "",
                        "argmax agree": round(float(ag.mean()), 4) if len(ag) else "",
                    }
                )
    if not rows:
        return out.skip("e", "no repo_examples product on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "e",
        "sae_repo_examples",
        "the SAE repo's own max-activating windows through our scorer (sae family only)",
        "`max_cos` IS the sae-repo-top32 baseline column (best of the feature's shipped windows, "
        "the analogue of a best-of-32 rollout draw); `frac fired` is the share of a feature's own "
        "windows whose peak clears the checkpoint's learned gate; `peak r` / `argmax agree` are "
        "the repo-vs-us activation agreement, ± SE across features",
        df,
    )


# Decimal places for an ARM's OWN mean columns, per base. The 27B arms carry SEs around 0.03, so a
# fourth decimal there is false precision; the 8B's are around 0.01. The PAIRED columns keep 4 dp on
# purpose -- they are per-direction differences with much tighter SEs, and rounding them to the
# arm's precision would erase real signal (-0.0426 and -0.0450 would both read -0.04).
ARM_MEAN_DP = {"qwen36-27b": 2, "qwen3-8b": 3}


def _gcg_row(base, fam, arm, view, slice_, dirs, per_dir, sides, set_name):
    """One emitted row of table (f): the arm's numbers over `dirs`, plus each MAEMM on those rows."""
    nd = ARM_MEAN_DP.get(base, 4)
    cos = np.array([per_dir[r]["cos"] for r in dirs])
    init = np.array([per_dir[r]["init_cos"] for r in dirs])
    nll = np.array([per_dir[r]["nll"] for r in dirs if per_dir[r].get("nll") is not None])
    rec = {
        "base": base,
        "family": fam,
        "view": view,
        "arm": arm,
        "slice": slice_,
        "dirs": len(dirs),
        "rows": f"{min(dirs)}-{max(dirs)}",
        "GCG cos": pm(float(cos.mean()), se(cos), nd),
        "init cos": pm(float(init.mean()), se(init), nd),
        "nll": round(float(nll.mean()), nd) if len(nll) else "",
    }
    want = set(dirs)
    for s_ in sides:
        if s_.base != base or s_.set != set_name:
            continue
        pos = np.array([i for i, r in enumerate(s_.rows) if r in want])
        if not len(pos):
            continue
        paired = np.array([per_dir[s_.rows[i]]["cos"] for i in pos])
        bo = bo_unbiased(s_.best[pos], s_.n)
        d = paired - bo
        rec[f"{s_.short} bo{s_.n}"] = round(float(bo.mean()), 4)
        rec[f"{s_.short} mean{s_.n}"] = round(float(s_.best[pos].mean(1).mean()), 4)
        rec[f"GCG - {s_.short} bo{s_.n}"] = pm(float(d.mean()), se(d))
        rec[f"GCG wins vs {s_.short}"] = round(float((paired >= bo).mean()), 3)
    return rec


def table_f(out: Out, vol: Vol, found: dict, sides: list[Scores], ids_of: dict, corpus: dict) -> None:
    """(f) GCG / EPO -- the reachability ceiling -- PAIRED against each MAEMM on the same directions.

    Layout is `<root>/base/<base>/gcg/<set>/<family>/<arm>/finals.jsonl`; a realact direction and an
    sae direction are different objects and never share an arm directory. With the final draw the
    GCG rows and the MAEMM rows index the SAME held-out set, so every comparison here is per
    direction -- which is what lets the table carry a paired difference and a win fraction rather
    than two means side by side.

    TWO SAE VIEWS, because the sae arms were run twice on different draws and they answer different
    questions (2026-09-16):

      * `*-strat` arms are the REPORTED sae rows: 8 targets from each of the four density quartiles
        (32 in all), so the arm's "all" row is a stratified estimate of the whole sae family and the
        per-quartile rows are where the interesting variation is.
      * the earlier plain sae arms took the first 32 sae rows, which are ALL q0 -- the rarest
        quartile. They are kept as a "rare-stratum (q0) view", not as a competing estimate of the
        family, because averaging them would silently report the rarest quartile as the whole.

    A direction missing from `finals.jsonl` (8B realact gcg-random32 excludes row 17) is absent from
    the join by construction, and `dirs` says how many survived. The per-direction best member is
    taken from the finals; where `summary.json` also carries `per_dir_best_cos` the two agree
    exactly (MEASURED 2026-09-16 on the 27B stratified arms: max |difference| = 0.0).
    """
    rows = []
    for base, info in sorted(found["bases"].items()):
        for set_name, arms in sorted(info["gcg"].items()):
            ids = ids_of.get((base, set_name), {})
            for arm in arms:
                fin = vol.jsonl(f"base/{base}/gcg/{set_name}/{arm}/finals.jsonl")
                if not fin:
                    continue
                name = arm.split("/")[-1]
                strat = name.endswith("-strat")
                by_fam: dict[str, list[dict]] = {}
                for r in fin:
                    by_fam.setdefault(r["family"], []).append(r)
                for fam, recs in sorted(by_fam.items()):
                    per_dir: dict[int, dict] = {}
                    for r in recs:
                        cur = per_dir.get(int(r["row"]))
                        if cur is None or r["cos"] > cur["cos"]:
                            per_dir[int(r["row"])] = r
                    dirs = sorted(per_dir)
                    if name.endswith("-smoke"):
                        # a partial/abandoned run kept on the volume beside the arm that replaced
                        # it (27B realact epo-corpus-smoke: 25 of 32 directions). It is LABELLED
                        # rather than dropped -- the rows are real and the `dirs` column is honest
                        # -- but the label must make it impossible to average in as a result.
                        view = "SUPERSEDED partial run -- not a reported arm"
                    elif fam != "sae":
                        view = ""
                    else:
                        view = "reported (stratified)" if strat else "rare-stratum q0 view"
                    rows.append(_gcg_row(base, fam, name, view, "all", dirs, per_dir, sides, set_name))
                    if fam == "sae" and strat:
                        qs = sorted({ids.get(r, {}).get("stratum") for r in dirs} - {None})
                        for q in qs:
                            sub = [r for r in dirs if ids.get(r, {}).get("stratum") == q]
                            if sub:
                                rows.append(
                                    _gcg_row(
                                        base, fam, name, view, f"density q{q}", sub, per_dir, sides, set_name
                                    )
                                )
                    # EPO holds several members per direction, one lambda each, selected by its own
                    # L_lambda -- so the arm traces a Pareto front in one run. These rows are
                    # PER-MEMBER means, not per-target bests, and carry no MAEMM columns: comparing
                    # one member against the inverter would be a different claim from the arm's
                    # reachability figure, which is the `all` row above.
                    lams = sorted({r["lam"] for r in recs if r.get("lam") is not None})
                    if len(lams) > 1:
                        nd = ARM_MEAN_DP.get(base, 4)
                        for lam in lams:
                            mem = [r for r in recs if r.get("lam") == lam]
                            if not mem:
                                continue
                            c = np.array([r["cos"] for r in mem])
                            nl = np.array([r["nll"] for r in mem if r.get("nll") is not None])
                            rows.append(
                                {
                                    "base": base,
                                    "family": fam,
                                    "view": view,
                                    "arm": name,
                                    "slice": f"lam {lam:g} (member)",
                                    "dirs": len({r["row"] for r in mem}),
                                    "rows": f"{min(r['row'] for r in mem)}-{max(r['row'] for r in mem)}",
                                    "GCG cos": pm(float(c.mean()), se(c), nd),
                                    "init cos": "",
                                    "nll": round(float(nl.mean()), nd) if len(nl) else "",
                                }
                            )
    if not rows:
        return out.skip("f", "no gcg arms with finals.jsonl on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "f",
        "gcg_ceiling",
        "GCG / EPO reachability ceiling, paired against each MAEMM on the same directions",
        caption(
            sides,
            corpus,
            "per (base, family, arm, slice): the per-direction best member's final cosine, its init "
            "cosine and its NLL, ± SE across TARGETS; MAEMM columns are the unbiased best-of-n and "
            "the mean-of-n on EXACTLY those directions, with the paired difference ± SE and the "
            "fraction of directions GCG wins, ordered primary first. An arm's own mean columns "
            "are shown to 2 dp on the 27B and 3 on the 8B, the precision their SEs support; the "
            "PAIRED columns keep 4 dp because they are per-direction differences with much tighter "
            "SEs. `lam ... (member)` rows are an EPO arm's PER-MEMBER means (one lambda each, the "
            "cosine-for-fluency trade), not per-target bests, and carry no MAEMM columns. "
            "The sae family has TWO views: "
            "`*-strat` arms (8 targets per density quartile, 32 total) are the REPORTED rows and "
            "carry the per-quartile breakdown, while the earlier plain sae arms are the first 32 "
            "sae rows -- all q0 -- and are kept as a rare-stratum view, never averaged in as the "
            "family. CAVEAT, stated once: the two sides are not compute-matched and not the same "
            "object -- GCG optimises ONE fixed T=32 token string with ~77k candidate forwards "
            "against the scorer itself, while the MAEMM draws n sampled rollouts from a prompt and "
            "never sees the metric. GCG bounds what the metric is reachable to; it is not a "
            "baseline the inverter competes with",
        ),
        df,
    )


def table_g(out: Out, sides: list[Scores], corpus: dict) -> None:
    """(g) where in a rollout the best-scoring token sits (checklist item 10)."""
    if not sides:
        return out.skip("g", "no scores directories on this root")
    edges = np.linspace(0, 1, ARGMAX_BINS + 1)
    rows = []
    for s in sides:
        for fam in s.families:
            m = s.mask(fam)
            arg, kept = s.argmax[m], s.kept[m]
            ok = (arg >= 0) & (kept >= 1)
            rel = arg[ok] / np.maximum(kept[ok], 1)
            hist, _ = np.histogram(rel, bins=edges)
            rec = {
                "base": s.base,
                "maemm": s.short,
                "family": fam,
                "rollouts": int(ok.sum()),
                "mean rel pos": round(float(rel.mean()), 4),
            }
            for i in range(ARGMAX_BINS):
                rec[f"[{edges[i]:.2f},{edges[i + 1]:.2f})"] = round(float(hist[i] / max(ok.sum(), 1)), 4)
            rec["at last token"] = round(float((arg[ok] == kept[ok] - 1).mean()), 4)
            rec["mean kept tok"] = round(float(kept[ok].mean()), 2)
            rows.append(rec)
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "g",
        "argmax_position",
        "argmax-token position as a fraction of the scored tokens",
        caption(
            sides,
            corpus,
            "bins are fractions of `argmax / n_kept_tokens`; 'at last token' is argmax == n-1. "
            "The recipe moves this a lot (77-82% vs ~4% last-token across recipes, checklist "
            "item 10), so it is reported per family x MAEMM, never pooled",
        ),
        df,
    )


def table_h(out: Out, sides: list[Scores], corpus: dict) -> None:
    """(h) the centred and norm-filtered secondaries against the primary (items 4, 60)."""
    rows = []
    for s in sides:
        if s.centred is None and s.filtered is None:
            continue
        for fam in s.families:
            m = s.mask(fam)
            p1, p64 = float(s.best[m].mean()), float(s.best[m].max(1).mean())
            rec = {
                "base": s.base,
                "maemm": s.short,
                "family": fam,
                "targets": int(m.sum()),
                "primary bo1": round(p1, 4),
                "primary bo64": round(p64, 4),
            }
            for label, arr in (("centred", s.centred), ("filtered", s.filtered)):
                if arr is None:
                    continue
                a = arr[m]
                with np.errstate(invalid="ignore"):
                    v1 = float(np.nanmean(a))
                    v64 = float(np.nanmean(np.nanmax(a, axis=1)))
                rec[f"{label} bo1"] = round(v1, 4)
                rec[f"Δ bo1 ({label})"] = f"{v1 - p1:+.4f}"
                rec[f"{label} bo64"] = round(v64, 4)
                rec[f"Δ bo64 ({label})"] = f"{v64 - p64:+.4f}"
            if s.centred_json:
                rec["frac tok dropped"] = s.centred_json.get("frac_tokens_dropped")
            rows.append(rec)
    if not rows:
        return out.skip("h", "no cos_centred_best.f16 / cos_filtered_best.f16 (run `--product centred`)")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "h",
        "centred_filtered",
        "secondary readings: both-sides-centred cosine, and the 10x-median norm filter",
        caption(
            sides,
            corpus,
            "centred = cos(best_act - mu, v) at the primary's argmax (precompute/centred.py). "
            "For `realact` that is the both-sides-centred convention (v is already unit(X[p]-mu)); "
            "for `sae` / `random` the target was never centred, so the column is a DIAGNOSTIC. "
            "filtered = the primary with tokens above 10x the row's median residual norm dropped",
        ),
        df,
    )


def table_i(out: Out, sides: list[Scores], corpus: dict) -> None:
    """(i) distributions, not means, for the sae family (checklist item 61)."""
    rows = []
    for s in sides:
        for fam in s.families:
            if fam != "sae":
                continue
            m = s.mask(fam)
            v = bo_unbiased(s.best[m], s.n)
            rec = {
                "base": s.base,
                "maemm": s.short,
                "family": fam,
                "targets": int(m.sum()),
                "mean": round(float(v.mean()), 4),
                "SE": round(se(v), 4),
            }
            for q in SAE_QUANTILES:
                rec[f"p{q * 100:g}"] = round(float(np.quantile(v, q)), 4)
            rec["min"] = round(float(v.min()), 4)
            rec["max"] = round(float(v.max()), 4)
            rows.append(rec)
    if not rows:
        return out.skip("i", "no sae-family rows in any scores directory on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "i",
        "sae_distribution",
        "per-target best-of-64 distribution, sae family",
        caption(
            sides,
            corpus,
            "quantiles of the PER-TARGET unbiased best-of-n, over targets. Means mislead on "
            "bimodal SAE metrics (checklist item 61), which is what this table is for",
        ),
        df,
    )


def _tag(cell: str) -> str:
    """The `--ps-tag` suffix of a patchscopes cell name, or "" (see precompute/patchscopes.py)."""
    return cell.split("__", 1)[1] if "__" in cell else ""


def table_j(out: Out, ps_sides: list[Scores], sides: list[Scores], corpus: dict) -> None:
    """(j) the Patchscopes baseline: the no-injection floor, the injected cells, and the MAEMM.

    Every cell went through THE scoring path (`score --rollouts-dir`), so the three columns are one
    number computed one way (checklist item 12). The MAEMM column is restricted to the cell's own
    rows AND re-estimated at the cell's own bo, because the baseline is mostly a sampling-luck
    effect (the 8B trial: 3.6x from bo1 to bo64, against run1's 1.2x) and comparing a bo-8 cell
    with a bo-64 MAEMM would credit the difference to the inverter.
    """
    if not ps_sides:
        return out.skip("j", "no patchscopes cells with a scores/ directory on this root")
    rows = []
    for ps in sorted(
        ps_sides, key=lambda x: (x.base, x.set, _tag(x.short), x.short.split("__")[0] != "floor", x.short)
    ):
        for fam in ps.families:
            m = ps.mask(fam)
            b = ps.best[m]
            rec = {
                "base": ps.base,
                "set": ps.set,
                "cell": ps.short,
                "family": fam,
                "dirs": int(m.sum()),
                "bo": ps.n,
                "mean cos (bo1)": pm(float(b.mean()), se(b.mean(1))),
                "cell best-of-bo": pm(float(bo_unbiased(b, ps.n).mean()), se(bo_unbiased(b, ps.n))),
            }
            # The matched floor: same (base, set) AND SAME --ps-tag, so a cell tagged `__bo32`
            # is read against the bo-32 floor and never against the sweep's bo-8 one. The whole
            # point of the floor is that it is matched -- an unmatched bo is what flattered the 8B
            # screen's floor (trial README §5a) -- so this pairs on the tag, not on the prefix.
            floor = next(
                (
                    f
                    for f in ps_sides
                    if f.base == ps.base
                    and f.set == ps.set
                    and f.short.split("__")[0] == "floor"
                    and _tag(f.short) == _tag(ps.short)
                ),
                None,
            )
            if floor is not None and floor is not ps:
                shared = sorted(set(ps.rows) & set(floor.rows))
                ip = np.array([ps.rows.index(r) for r in shared if ps.fam[ps.rows.index(r)] == fam])
                if len(ip):
                    jf = np.array(
                        [floor.rows.index(r) for r in shared if floor.fam[floor.rows.index(r)] == fam]
                    )
                    k = min(ps.n, floor.n)
                    lift = bo_unbiased(ps.best[ip], k) - bo_unbiased(floor.best[jf], k)
                    p, win, _ = sign_test(lift)
                    rec["lift over floor"] = pm(float(lift.mean()), se(lift))
                    rec["beats its floor"] = f"{win:.3f}" if np.isfinite(win) else ""
                    rec["sign-test p"] = f"{p:.3g}" if np.isfinite(p) else ""
            for s_ in sides:
                if s_.base != ps.base or s_.set != ps.set:
                    continue
                pos = np.array([i for i, r in enumerate(s_.rows) if r in set(ps.rows) and s_.fam[i] == fam])
                if not len(pos):
                    continue
                k = min(ps.n, s_.n)
                rec[f"{s_.short} @cell bo"] = round(float(bo_unbiased(s_.best[pos], k).mean()), 4)
                rec[f"{s_.short} @bo{s_.n}"] = round(float(bo_unbiased(s_.best[pos], s_.n).mean()), 4)
            rows.append(rec)
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "j",
        "patchscopes",
        "the Patchscopes baseline: no-injection floor, injected cells, and the MAEMM on the same rows",
        caption(
            ps_sides,
            corpus,
            "`bo` is the CELL's rollout budget and differs from the MAEMM's 64; the MAEMM is "
            "therefore shown twice -- re-estimated at the cell's own bo on the cell's own rows, "
            "and at its own bo. The floor is the same prompt with NO injection anywhere in the "
            "forward pass, generated once and scored against every direction at the SAME bo",
        ),
        df,
    )


def _sae_own_act(vol: Vol, side: Scores, feat_of: dict[int, int]) -> dict[int, float] | None:
    """{target row -> max over its rollouts of ITS OWN feature's gated activation}, or None.

    `score` stores, per (target, rollout), the SAE features whose PRE-GATE activation at the argmax
    token clears the checkpoint's learned gate, as a flat CSR (sae_idx / sae_val / sae_off). The
    question here is narrower than the CSR: did the rollout make the feature the direction IS fire,
    and how hard. A feature absent from a row's slice did not clear the gate there, and contributes
    0.0 rather than a missing value -- "did not fire" is the measurement, not a gap.

    The CSR is ~48 MB per MAEMM, which is why this is fetched for the PRIMARY only (the local
    fetch budget is 100 MB and cos.f16 + topk already spend half of it). Returns None when the
    arrays are not present, and the caller drops the column with a note rather than failing.
    """
    idx = vol.json(f"{side.dir}/index.json") or {}
    if not all(k in idx for k in ("sae_idx.i32", "sae_val.f16", "sae_off.i64")):
        return None
    n_ent = idx["sae_idx.i32"]["shape"][0]
    sidx = vol.array(f"{side.dir}/sae_idx.i32", "int32", (n_ent,))
    sval = vol.array(f"{side.dir}/sae_val.f16", "float16", (n_ent,))
    soff = vol.array(f"{side.dir}/sae_off.i64", "int64", (idx["sae_off.i64"]["shape"][0],))
    if sidx is None or sval is None or soff is None:
        return None
    out: dict[int, float] = {}
    n = side.n
    for i, row in enumerate(side.rows):
        f = feat_of.get(row)
        if f is None:
            continue
        best = 0.0
        for k in range(n):
            j = i * n + k
            lo, hi = int(soff[j]), int(soff[j + 1])
            if hi <= lo:
                continue
            hit = np.nonzero(sidx[lo:hi] == f)[0]
            if len(hit):
                best = max(best, float(sval[lo + hit[0]]))
        out[row] = best
    return out


def _sae_inputs(vol: Vol, found: dict, side: Scores):
    """(corpus top-1 at the largest size, repo per-feature row) for the side's base/set, or None."""
    base, set_name = side.base, side.set
    topk = vol.jsonl(f"base/{base}/scan/{set_name}/topk.jsonl")
    if topk is None:
        return None, None
    sizes = sorted({int(r["size"]) for r in topk})
    top1 = {int(r["row"]): float(r["top"][0][3]) for r in topk if int(r["size"]) == sizes[-1] and r["top"]}
    repo = None
    for sae in found["bases"][base]["saes"]:
        rows = vol.jsonl(f"base/{base}/sae/{sae.split('/')[-1]}/repo_examples/{set_name}/per_feature.jsonl")
        if rows:
            repo = {int(r["row"]): r for r in rows}
            break
    return top1, repo


def table_k(out: Out, vol: Vol, found: dict, sides: list[Scores], ids_of: dict, corpus: dict) -> None:
    """(k) per-FEATURE sae comparison: the MAEMM against the two searches, on the same features.

    Tests a specific claim (Celeste's note): "the 27B MAEMMs invert badly on ~30% of SAE features
    (~40% worse than corpus search)". The table is built to confirm or correct that number rather
    than to illustrate it, so it reports BOTH thresholds the claim mixes -- an absolute shortfall of
    > 0.05 cosine and a relative shortfall of > 40% -- against BOTH searches, and splits by the
    ids.jsonl density quartile, because paired differences are stratum-dependent (item 63).

    Baselines, per feature: the corpus scan's top-1 window at the LARGEST corpus size, and the SAE
    repo's own best shipped window (`per_feature.max_cos`, the `sae-repo-top32` column). "Win" is
    MAEMM >= baseline. The MAEMM statistic is the unbiased best-of-n, i.e. the inverter is given its
    full rollout budget while each search is given its single best text.
    """
    rows = []
    for side in sides:
        top1, repo = _sae_inputs(vol, found, side)
        if top1 is None:
            continue
        ids = ids_of.get((side.base, side.set), {})
        m = side.mask("sae")
        if not m.any():
            continue
        sel = [r for i, r in enumerate(side.rows) if m[i]]
        bo = bo_unbiased(side.best[m], side.n)
        feat_of = {r: ids[r].get("id") for r in sel if r in ids}
        acts = _sae_own_act(vol, side, feat_of) if side.primary else None
        recs = []
        for pos, row in enumerate(sel):
            if row not in top1:
                continue
            r_repo = (repo or {}).get(row)
            recs.append(
                {
                    "row": row,
                    "stratum": ids.get(row, {}).get("stratum"),
                    "maemm": float(bo[pos]),
                    "corpus": float(top1[row]),
                    "repo": None if r_repo is None else float(r_repo["max_cos"]),
                    "act": None if acts is None else acts.get(row),
                }
            )
        if not recs:
            continue
        for slice_ in [
            "all",
            *[f"density q{q}" for q in sorted({r["stratum"] for r in recs if r["stratum"] is not None})],
        ]:
            sub = recs if slice_ == "all" else [r for r in recs if f"density q{r['stratum']}" == slice_]
            if not sub:
                continue
            mm = np.array([r["maemm"] for r in sub])
            cc = np.array([r["corpus"] for r in sub])
            rec = {
                "base": side.base,
                "maemm": side.short,
                "role": side.role,
                "slice": slice_,
                "features": len(sub),
                f"MAEMM bo{side.n}": round(float(mm.mean()), 4),
                "corpus top-1": round(float(cc.mean()), 4),
                "wins vs corpus": round(float((mm >= cc).mean()), 4),
                "worse >0.05 abs": round(float(((cc - mm) > 0.05).mean()), 4),
                "worse >40% rel": round(float((((cc - mm) / np.maximum(cc, 1e-9)) > 0.40).mean()), 4),
            }
            rp = [r for r in sub if r["repo"] is not None]
            if rp:
                mr = np.array([r["maemm"] for r in rp])
                vr = np.array([r["repo"] for r in rp])
                rec["repo top"] = round(float(vr.mean()), 4)
                rec["wins vs repo"] = round(float((mr >= vr).mean()), 4)
                rec["worse >0.05 abs (repo)"] = round(float(((vr - mr) > 0.05).mean()), 4)
                rec["worse >40% rel (repo)"] = round(
                    float((((vr - mr) / np.maximum(vr, 1e-9)) > 0.40).mean()), 4
                )
            ac = [r["act"] for r in sub if r["act"] is not None]
            if ac:
                rec["own-feature act (max over rollouts)"] = round(float(np.mean(ac)), 3)
                rec["frac firing"] = round(float(np.mean([a > 0 for a in ac])), 4)
            rows.append(rec)
    if not rows:
        return out.skip("k", "no sae rows with both a corpus scan and a scores directory on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "k",
        "sae_per_feature",
        "per-feature sae: the MAEMM against corpus search and the SAE repo's own windows",
        caption(
            sides,
            corpus,
            "one row per (MAEMM, slice) over the SAME features; the MAEMM gets its full best-of-n "
            "while each search gets its single best text. 'worse >40% rel' is "
            "(baseline - MAEMM)/baseline > 0.4. The own-feature activation column is the max over a "
            "feature's rollouts of ITS OWN feature's gated activation at the argmax token, fetched "
            "for the PRIMARY only (the CSR is ~48 MB per MAEMM)",
        ),
        df,
    )


def table_l(out: Out, vol: Vol, found: dict, sides: list[Scores], ids_of: dict, corpus: dict) -> None:
    """(l) the shape of (k): deciles of the per-feature difference MAEMM - corpus top-1."""
    rows = []
    for side in sides:
        top1, _ = _sae_inputs(vol, found, side)
        if top1 is None:
            continue
        m = side.mask("sae")
        if not m.any():
            continue
        sel = [r for i, r in enumerate(side.rows) if m[i]]
        bo = bo_unbiased(side.best[m], side.n)
        d = np.array([bo[i] - top1[r] for i, r in enumerate(sel) if r in top1])
        if not len(d):
            continue
        rec = {
            "base": side.base,
            "maemm": side.short,
            "role": side.role,
            "features": len(d),
            "mean diff": round(float(d.mean()), 4),
            "SE": round(se(d), 4),
            "frac > 0": round(float((d > 0).mean()), 4),
        }
        for q in range(1, 10):
            rec[f"d{q}"] = round(float(np.quantile(d, q / 10)), 4)
        rows.append(rec)
    if not rows:
        return out.skip("l", "no sae rows with both a corpus scan and a scores directory on this root")
    df = pl.DataFrame(rows, infer_schema_length=None)
    out.table(
        "l",
        "sae_diff_deciles",
        "deciles of the per-feature difference (MAEMM best-of-n - corpus top-1), sae family",
        caption(
            sides,
            corpus,
            "d1..d9 are the deciles of the PER-FEATURE difference; d1 is the 10% of features where "
            "the MAEMM falls furthest behind the corpus search. A mean near zero with wide deciles "
            "is the bimodality checklist item 61 warns a mean would hide",
        ),
        df,
    )


# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    root_tag: Annotated[str, typer.Option(help="smoke | full -- which --root the pipeline wrote")] = "smoke",
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what data/ already has")] = False,
    tables: Annotated[str, typer.Option(help="letters to build, e.g. 'adg' (default: a-j)")] = "",
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    data_dir: Annotated[Path | None, typer.Option(help="override reconstruction/data/<root-tag>")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="override reconstruction/out/<root-tag>")] = None,
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
    width: Annotated[int, typer.Option(help="console width; 0 = the terminal's, or 200 when piped")] = 0,
) -> None:
    assert root_tag in ROOTS, f"--root-tag must be one of {sorted(ROOTS)}, got {root_tag!r}"
    # a piped run defaults to 200 columns: rich would otherwise fall back to 80 and elide every
    # cell of the wide tables into "0.…", which is how the numbers get lost in a log
    console.width = width or (console.width if sys.stdout.isatty() else 200)
    want = set(tables) if tables else set("abcdefghijkl")
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    vol = Vol(root_tag, data_dir or (HERE / "data" / root_tag), modal_cmd, refetch, quiet, offline=not fetch)
    out = Out(out_dir or (HERE / "out" / root_tag), root_tag, ROOTS[root_tag])

    console.print(f"[bold]root[/bold] /vol/{ROOTS[root_tag] or ''} -> data {vol.local}")
    found = discover(vol, cfg)
    ids_by_base: dict = {}
    for base, info in found["bases"].items():
        for s in info["sets"]:
            got = load_ids(vol, base, s)
            if got is not None:
                ids_by_base[(base, s)] = got

    sides: list[Scores] = []
    ps_sides: list[Scores] = []
    corpus: dict[str, str] = {}
    for base, info in found["bases"].items():
        for maemm, stems in info["maemms"].items():
            for stem, set_name, engine in stems:
                ids = ids_by_base.get((base, set_name))
                if ids is None:
                    continue
                label = maemm.split("/")[-1] + (f"@{engine}" if engine else "")
                sc = Scores(vol, base, label, set_name, ids, f"maemms/{maemm}/scores/{stem}")
                if sc.ok:
                    entry = cfg["maemms"].get(maemm, {})
                    sc.primary = bool(entry.get("primary", False))
                    # `role: control` (or `type: base`) marks the UNTRAINED-BASE CONTROL: the clean
                    # base through the identical prompt / injection / sampling path. It is neither
                    # the primary nor a secondary MAEMM and sorts between them.
                    control = entry.get("role") == "control" or entry.get("type") == "base"
                    sc.role = "primary" if sc.primary else ("control" if control else "secondary")
                    assert not (sc.primary and control), (
                        f"{maemm}: config.yaml marks it BOTH `primary: true` and the control"
                    )
                    sides.append(sc)
                else:
                    console.print(f"[yellow]incomplete scores: {maemm} / {stem}[/yellow]")
        for set_name, cells in info.get("patchscopes", {}).items():
            ids = ids_by_base.get((base, set_name))
            if ids is None:
                continue
            for cell in cells:
                sc = Scores(
                    vol,
                    base,
                    cell,
                    set_name,
                    ids,
                    f"base/{base}/patchscopes/{set_name}/{cell}/scores",
                )
                if sc.ok:
                    ps_sides.append(sc)
                else:
                    console.print(f"[yellow]incomplete patchscopes scores: {cell}[/yellow]")
        # the corpus size a base's numbers were calibrated against, from the scan's own sizes
        for s in info["sets"]:
            topk = vol.get(f"base/{base}/scan/{s}/topk.jsonl")
            if topk is not None:
                with open(topk) as fh:
                    sizes = {json.loads(line)["size"] for line in fh if line.strip()}
                corpus[base] = f"{max(sizes)}M"
                break
    # Order every table PRIMARY FIRST. `primary: true` in config.yaml names the MAEMM the paper's
    # claims are about; the rest are secondary rows kept because their computation is already done.
    # With no primary declared this is a no-op and the caption says the order is config order.
    n_primary = sum(s.primary for s in sides)
    sides.sort(key=lambda x: (role_rank(x), x.base, x.short))
    if n_primary:
        console.print(
            "[bold]primary MAEMM[/bold]: "
            + ", ".join(s.short for s in sides if s.primary)
            + " (config.yaml `primary: true`); "
            + f"{sum(s.role == 'control' for s in sides)} control row(s), "
            + f"{sum(s.role == 'secondary' for s in sides)} secondary row(s)"
        )
    else:
        console.print("[yellow]no `primary: true` in config.yaml -- tables keep config order[/yellow]")

    # ids for the per-base paired table, keyed by base at the set the scores actually used
    ids_flat = {(s.base, s.set): ids_by_base[(s.base, s.set)] for s in sides}
    console.print(
        f"[bold]{len(sides)} scores directories[/bold]: "
        + ", ".join(f"{s.base}/{s.short}:{s.set}[{len(s.rows)}x{s.n}]" for s in sides)
    )
    if ps_sides:
        console.print(
            f"[bold]{len(ps_sides)} patchscopes cells[/bold]: "
            + ", ".join(f"{s.base}/{s.short}[{len(s.rows)}x{s.n}]" for s in ps_sides)
        )

    if "a" in want:
        table_a(out, sides, corpus)
    if "b" in want:
        table_b(out, sides, corpus)
    if "c" in want:
        table_c(out, sides, ids_flat, corpus)
    if "d" in want:
        table_d(out, vol, found, ids_by_base, corpus)
    if "e" in want:
        table_e(out, vol, found)
    if "f" in want:
        table_f(out, vol, found, sides, ids_flat, corpus)
    if "g" in want:
        table_g(out, sides, corpus)
    if "h" in want:
        table_h(out, sides, corpus)
    if "i" in want:
        table_i(out, sides, corpus)
    if "j" in want:
        table_j(out, ps_sides, sides, corpus)
    if "k" in want:
        table_k(out, vol, found, sides, ids_flat, corpus)
    if "l" in want:
        table_l(out, vol, found, sides, ids_flat, corpus)
    out.finish()
    if vol.missing:
        console.print(f"[yellow]{len(vol.missing)} files not on the volume:[/yellow]")
        for m in vol.missing[:20]:
            console.print(f"  [dim]{m}[/dim]")
    console.print(f"fetched {vol.fetched} files ({vol.bytes / 1e6:.1f} MB) into {vol.local}")


if __name__ == "__main__":
    app()
