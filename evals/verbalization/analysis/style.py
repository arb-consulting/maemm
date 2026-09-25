"""Shared figure palette for the verbalization figures.

Each plot script used to carry its own copy of these constants. They are identical everywhere, so
they live here once and a palette change lands on all eight figures at the same time.

`apply_rcparams()` is deliberately opt-in: only recovery_vs_rarity.py ever set it, and calling it
from the other scripts would silently restyle figures that are already in the report.

Imported as `from style import SERIES, ...`, which resolves because CPython puts a script's own
directory on sys.path[0] -- these are run as `python evals/verbalization/analysis/<script>.py`.
"""

# six-colour categorical series; scripts index into it positionally, so order is load-bearing
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def apply_rcparams():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
