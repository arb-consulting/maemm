"""Concept-level summaries: stratified bootstrap intervals and paired per-concept differences."""
import numpy as np


def unit(vector):
    """The vector at unit length, or None for a zero or nonfinite one."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        return None
    return vector / norm


def finite_mean(values):
    x = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None


def bootstrap(values, genres, n_boot=10000, seed=0):
    """Macro average; resample complete concepts within their genre."""
    values = np.asarray(values, dtype=np.float64)
    genres = np.asarray(genres)
    keep = np.isfinite(values)
    values, genres = values[keep], genres[keep]
    if not len(values):
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "n_valid": 0}
    rng = np.random.default_rng(seed)
    boot_sum = np.zeros(n_boot)
    for genre in sorted(set(genres)):
        group = values[genres == genre]
        indices = rng.integers(len(group), size=(n_boot, len(group)))
        boot_sum += group[indices].sum(axis=1)
    lower, upper = np.quantile(boot_sum / len(values), [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_lower": float(lower),
            "ci_upper": float(upper), "n_valid": int(len(values))}


#: What identifies one summarised cell; `judge` keeps two judges' verdicts from pooling.
SUMMARY_KEY = ("metric", "condition", "judge", "budget_type", "budget")


def summarize_rows(rows, concepts, n_boot=10000):
    """One summary row per cell and genre group ("all" and each genre); rows hold one estimate per concept."""
    groups = {}
    metadata = {str(c["concept_id"]): c for c in concepts}
    for row in rows:
        if str(row["concept_id"]) not in metadata:
            raise ValueError("Score references a concept outside this population")
        key = tuple(row.get(field, "") if field == "judge" else row.get(field) for field in SUMMARY_KEY)
        groups.setdefault(key, []).append(row)
    output = []
    for key, records in sorted(groups.items(), key=lambda pair: str(pair[0])):
        if len({str(r["concept_id"]) for r in records}) != len(records):
            raise ValueError(f"Duplicate concept summary for {key}")
        for group in ("all", "text", "code", "math"):
            selected = [r for r in records if group == "all" or metadata[str(r["concept_id"])]["genre"] == group]
            if not selected:
                continue
            genres = [metadata[str(r["concept_id"])]["genre"] for r in selected]
            values = [np.nan if r["estimate"] is None else r["estimate"] for r in selected]
            summary = bootstrap(values, genres, n_boot, seed=0)
            output.append(dict(zip(SUMMARY_KEY, key), group=group, n_total=len(selected),
                               ci_method="stratified_concept_bootstrap", **summary))
    return output


def paired_rows(rows, left, right):
    """Per-concept `left - right` estimates, paired within one judge, metric and budget."""
    index = {}
    for row in rows:
        key = (row["concept_id"], row["metric"], row.get("judge", ""), row.get("budget_type"), row.get("budget"))
        index.setdefault(key, {})[row["condition"]] = row["estimate"]
    output = []
    for (concept, metric, judge, kind, budget), values in index.items():
        if left not in values or right not in values:
            continue
        a, b = values[left], values[right]
        value = None if a is None or b is None else float(a - b)
        output.append({"concept_id": concept, "metric": metric, "condition": f"{left}_minus_{right}",
                       "judge": judge, "budget_type": kind, "budget": budget, "estimate": value})
    return output
