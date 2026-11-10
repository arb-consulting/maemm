"""Shared statistics (Wilson intervals, the paired item bootstrap, the paired
sign-flip permutation test with Holm's correction) and the long-format row and CSV table writer."""

import csv, math, os, random
import numpy as np

# `ci_method` values; "" marks a plain count with no interval
CI_WILSON = "wilson95"
CI_BOOT = "bootstrap_item_10000_seed0"


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def boot_indices(n, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n)) if n else np.zeros((n_boot, 0), dtype=int)


def _pct(values):
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def boot_mean(a, idx):
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return (float("nan"),) * 3
    est = float(np.nanmean(a))
    # per-resample mean of finite values; an all-missing resample is nan, without np.nanmean's warning
    vals = a[idx]
    finite = np.isfinite(vals)
    counts = finite.sum(axis=1)
    sums = np.where(finite, vals, 0.0).sum(axis=1)
    samples = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    lo, hi = _pct(samples)
    return est, lo, hi


def paired_diff(a, b, idx):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.where(ok, a - b, np.nan)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    est, lo, hi = boot_mean(d, idx)
    return est, lo, hi, n


def row(metric, condition, group, est, lo, hi, n_total, n_valid, ci_method="", **extra):
    budget_type = extra.pop("budget_type", "")
    budget = extra.pop("budget", "")
    r = {
        "metric": metric,
        "condition": condition,
        "group": group,
        "budget_type": budget_type,
        "budget": budget,
        "estimate": est,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_total": n_total,
        "n_valid": n_valid,
        "ci_method": ci_method,
    }
    r.update(extra)
    return r


#: sign-flip test settings: flips per comparison, and the seed of the one RNG a family shares
N_FLIPS = 20000
FLIP_SEED = 0


def sign_flip_p(diffs, rng, n_flips=N_FLIPS):
    """Two-sided paired sign-flip p-value for the mean of `diffs`: zero differences are dropped, and
    p = (hits + 1) / (n_flips + 1). With no non-zero difference p is 1 and `rng` is untouched."""
    nonzero = [d for d in diffs if d]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - 1e-12
               for _ in range(n_flips))
    return (hits + 1) / (n_flips + 1)


def sign_flip_tests(diff_lists, n_flips=N_FLIPS, seed=FLIP_SEED):
    """`sign_flip_p` for each list, all drawn in order from one `random.Random(seed)`."""
    rng = random.Random(seed)
    return [sign_flip_p(d, rng, n_flips) for d in diff_lists]


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out, running = [None] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        out[i] = running
    return out


def finite_n(vec):
    """The number of non-missing (not None, not NaN) entries: `n_valid` for a NaN-padded vector."""
    return sum(1 for v in vec if v is not None and not (isinstance(v, float) and math.isnan(v)))


# CSV column order: these first, then each table's extra columns in first-seen order.
BASE_COLUMNS = (
    "metric",
    "condition",
    "group",
    "budget_type",
    "budget",
    "estimate",
    "ci_lower",
    "ci_upper",
    "n_total",
    "n_valid",
    "ci_method",
)


def write_tables(T, out_dir):
    for name, rows in T.items():
        if not rows:
            continue
        present = set()
        for r in rows:
            present.update(r.keys())
        cols = [k for k in BASE_COLUMNS if k in present]
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(os.path.join(out_dir, name + ".csv"), "w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()})
