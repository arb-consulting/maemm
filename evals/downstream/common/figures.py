"""Package-agnostic matplotlib helpers (imported lazily): the PDF + 300-DPI PNG pair, row selection from a
long-format table, asymmetric error bars, the unavailable-cell test and the smoke banner."""

import math, os

SMOKE_BANNER = "SMOKE RUN (n = {n}) — not results"


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42})
    return plt


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=300)
    import matplotlib.pyplot as plt

    plt.close(fig)


def sel(rows, **kw):
    """Rows matching every keyword, compared as strings so CSV and in-memory tables select alike."""
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]


def one(rows, **kw):
    s = sel(rows, **kw)
    return s[0] if s else None


def err(r):
    """The asymmetric [below, above] half-widths, clamped at zero (a Wilson bound can be an ulp past the
    estimate at k == 0 or k == n)."""
    return [max(0.0, r["estimate"] - r["ci_lower"]), max(0.0, r["ci_upper"] - r["estimate"])]


def is_unavailable(est):
    """True when the estimate is "", None or NaN: the cell is drawn empty, never as zero."""
    return est in ("", None) or (isinstance(est, float) and math.isnan(est))


def suptitle(base, smoke_n):
    """`base`, headed by the smoke banner as a line of its own when this is a --smoke run."""
    lines = [SMOKE_BANNER.format(n=smoke_n)] if smoke_n is not None else []
    return "\n".join(lines + [base])
