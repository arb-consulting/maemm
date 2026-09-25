"""THE failure criterion of the 27B unverbalizability analysis, in one place.

    fails  = none of the feature's b sampled texts clears the SAE's learned gate (tau = 1.5846),
             i.e. `fire_fraction == 0` in a from_precompute dump
    dead   = the feature never clears the gate anywhere in the 16M corpus scan; excluded, since no
             text is known to fire it (evals/verbalization/report/data/dead_27b_l42-1b.json, 10 of 131,072)

Replaces `norm_act < 0.10`, the earlier convention. The two select nearly the same features on the
27B -- failures are almost all complete misses -- but the gate needs no corpus-peak denominator and
is the SAE's own definition of firing.
"""
import json
from pathlib import Path

import numpy as np

import dumplib

GATE = 1.5845966339111328
DEAD_PATH = Path(__file__).resolve().parent.parent / "report" / "data" / "dead_27b_l42-1b.json"


def dead_ids():
    return set(json.load(open(DEAD_PATH))["dead"])


def load_perdir(path):
    """from_precompute dump -> dict of arrays over NON-dead features, plus `fail` (bool) and
    `n_dead_dropped`. Every per-feature list in the dump is filtered the same way.

    The columns and the dead-feature filter come from dumplib.PerDir; what this adds is the gate
    criterion itself -- dropping dead features and deriving `fail` -- which is the thing this
    module exists to keep in one place.
    """
    p = dumplib.PerDir.load(path)
    dead = dead_ids()
    keep = np.array([int(f) not in dead for f in p.feature])
    out = dict(p.select(keep).col)
    out["feature"] = out["feature"].astype(int)
    out["fail"] = np.asarray(out["fire_fraction"], float) == 0
    out["n_dead_dropped"] = int((~keep).sum())
    return out


def none_of_k(fired_per_draw, k=8):
    """Unbiased P(none of k draws fires) from n >= k per-draw 0/1 outcomes (1 - best-of-k fired)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "evals/faithfulness"))
    from results.common import bo_unbiased
    return 1.0 - bo_unbiased(np.asarray(fired_per_draw, float), k)
