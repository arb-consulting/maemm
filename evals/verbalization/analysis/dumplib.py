"""Shared loaders for the eval_dirs dumps, and the figure scaffolding built on them.

Every script here starts from the same two artifacts -- a per-feature dump from
`modal_8b_verbalization.py::eval_dirs` and the corpus scan from `scan_fire` -- and every script
used to parse them itself. The parsing had drifted into two incompatible shapes:

  COLUMN  {tag: {column: np.ndarray}}          the plot scripts; vectorised over features
  ROW     {tag: {feature_id: {column: value}}} the table scripts; one feature at a time

`PerDir` is both views over one parse, so the shape a caller wants is a property access rather
than a reason to re-read the file. `.col`/`__getitem__` is the column view; `.rows` is the row
view, and it yields native Python scalars straight off the JSON -- not numpy ones -- because the
table scripts feed them to `round()` and `csv`, where `np.float64` would print differently.

Also here: the rarity axis (`Scan`), which was four copies of the same
`log10(sae_nfire[feat] / n_tok)`, and the figure boilerplate (`savefig`, `write_data`, `deciles`,
`auc_rarer_worse`) that every plot script carried inline.

Imported as `import dumplib as D`, which resolves because CPython puts a script's own directory on
sys.path[0] -- these are run as `python evals/verbalization/analysis/<script>.py`. Same arrangement as
`style.py` and `featlib.py`.
"""
import json
import os

import numpy as np

# data/mlp42_neurons_worker.py and scan_fire both scan 4000 windows x 256 tokens. Used only when a
# caller passes no n_tok AND the npz carries none -- the 27B-era sae_match.npz predates the key.
N_TOK_DEFAULT = 4000 * 256


def split_spec(spec):
    """'[TAG=]PATH' -> (tag, path). A bare path is tagged with its basename.

    The scripts disagreed on whether TAG= was optional -- four required it (and raised ValueError
    from split() when it was missing), three defaulted to the basename. Optional everywhere is the
    permissive union: every existing invocation passes TAG=PATH and is unaffected.
    """
    spec = os.fspath(spec)          # callers that already hold a pathlib.Path pass it straight in
    return tuple(spec.split("=", 1)) if "=" in spec else (os.path.basename(spec), spec)


class PerDir:
    """One arm's per-feature dump from eval_dirs: `perdir_8b_<tag>.json`.

    The `sae` block is a dict of equal-length columns -- feature, best_act, corpus_peak, norm_act,
    and (on dumps that carry the null) null_mean/null_p95/null_max. Which columns exist varies by
    dump: the 27B dumps have `fire_fraction` and the `cos` families, the 8B dumps have the nulls
    and `ex_top16`/`ex_last`. Callers are expected to probe with `in` rather than assume.
    """

    def __init__(self, tag, path, meta, raw):
        self.tag, self.path, self.meta = tag, path, meta
        self._raw = raw                                   # JSON lists, for the row view
        self.col = {k: np.asarray(v) for k, v in raw.items()}

    @classmethod
    def load(cls, spec):
        """'[TAG=]PATH' (or a bare path) -> PerDir."""
        tag, path = split_spec(spec)
        meta = json.load(open(path))
        # non-list entries would break the column/row alignment; no dump has one today, but the
        # sae block is written by the evaluator and is not frozen
        raw = {k: v for k, v in meta["perdir"]["sae"].items() if isinstance(v, list)}
        return cls(tag, path, meta, raw)

    # ---- column view -------------------------------------------------------------------------
    def __getitem__(self, k):
        return self.col[k]            # KeyError is load-bearing: callers probe for optional columns

    def __contains__(self, k):
        return k in self.col

    def __len__(self):
        return len(self.col["feature"])

    def get(self, k, default=None):
        return self.col.get(k, default)

    @property
    def feature(self):
        return self.col["feature"]

    # ---- row view ----------------------------------------------------------------------------
    @property
    def rows(self):
        """{feature_id: {column: value}} with native Python scalars, as the table scripts want."""
        feats = [int(f) for f in self._raw["feature"]]
        return {f: {k: v[i] for k, v in self._raw.items()} for i, f in enumerate(feats)}

    # ---- derived -----------------------------------------------------------------------------
    def select(self, idx):
        """New PerDir keeping only rows `idx` (a boolean mask or an index array). Meta is shared.

        A column whose length is not the feature count is passed through untouched rather than
        indexed -- the `sae` block is written by the evaluator and has carried non-per-feature
        entries before.
        """
        ix = np.asarray(idx)
        keep = np.flatnonzero(ix) if ix.dtype == bool else ix
        n = len(self._raw["feature"])
        return PerDir(self.tag, self.path, self.meta,
                      {k: ([v[int(i)] for i in keep] if len(v) == n else v)
                       for k, v in self._raw.items()})

    def cos(self, family):
        """The `cos` family arrays (random / realact / ...), or None on dumps without them.

        The 8B dumps carry no `cos` block at all, so this cannot be a plain
        `meta["perdir"]["cos"][family]` -- that raised KeyError one level up from the `.get()`
        that was meant to tolerate it.
        """
        v = (self.meta.get("perdir", {}).get("cos") or {}).get(family)
        return None if v is None else np.asarray(v, float)


def load_perdir(specs):
    """['[TAG=]PATH', ...] -> {tag: PerDir}, in the order given."""
    return {p.tag: p for p in (PerDir.load(s) for s in specs)}


def shared_features(arms):
    """Assert every arm covers the same features in the same order, and return them.

    The paired figures subtract arms row-by-row, so a silent mismatch would compare feature i of
    one arm against a different feature of another.
    """
    arms = list(arms.values()) if isinstance(arms, dict) else list(arms)
    feat = arms[0].feature
    for p in arms[1:]:
        assert np.array_equal(p.feature, feat), \
            f"all arms must share a feature list ({arms[0].tag} vs {p.tag})"
    return feat


def load_texts(specs):
    """['[TAG=]PATH', ...] -> {tag: {feature: [generation, ...]}} from `texts_8b_<tag>.json`."""
    out = {}
    for spec in specs:
        tag, path = split_spec(spec)
        d = json.load(open(path))
        feats, rows, texts = d["feats"], d["rows"], d["texts"]
        by = {}
        for i, r in enumerate(rows):
            by.setdefault(int(feats[int(r)]), []).append(texts[i])
        out[tag] = by
    return out


class Scan:
    """The corpus scan from `scan_fire`: `sae_match_8b.npz`.

    `sae_nfire[F]` counts tokens on which feature F had pre-topk activation > 0 over an n_tok
    scan. That count, normalised, is the rarity axis every figure in this folder is plotted
    against.
    """

    def __init__(self, npz_path, n_tok=None):
        self.path = npz_path
        self.z = np.load(npz_path)
        # an explicit n_tok wins, so 27B callers that pass --n-tok keep their own number; the
        # npz's own key is preferred over the constant, which is only a last resort
        self.n_tok = int(n_tok if n_tok is not None else
                         self.z["n_tok"] if "n_tok" in self.z else N_TOK_DEFAULT)
        self.nfire = self.z["sae_nfire"]

    @property
    def fire_pct(self):
        """Per-feature firing rate in PERCENT of tokens, indexed by feature id."""
        return self.nfire / float(self.n_tok) * 100.0

    def log10_freq(self, feats):
        """log10 firing frequency for `feats`. Zero counts are floored at one hit, not dropped:
        a feature that never fired is the rarest point on the axis, not a missing one."""
        return np.log10(np.maximum(self.nfire.astype(float)[feats], 1) / self.n_tok)

    def label(self, gate="act > 0"):
        return f"log10 firing frequency ({gate}, {self.n_tok / 1e6:.2f}M tokens)"


# ---- figure scaffolding -----------------------------------------------------------------------

def savefig(fig, out, stem, dpi=170, pdf_dpi=True, close=True):
    """Write `<out>/<stem>.png` and `.pdf` at the report's fixed dpi/bbox.

    `pdf_dpi=False` omits dpi from the PDF save. It is invisible on a pure-vector figure, but a
    figure with RASTERIZED content embeds that content as an image whose resolution follows this
    dpi -- fig2's R^2 matrix is an imshow, and recovery_vs_rarity.py has always written its PDF
    without dpi, so the committed fig2_r2_matrix.pdf sits at the matplotlib default and passing
    dpi silently rewrites it 20% larger.
    """
    for ext in ("png", "pdf"):
        kw = {"dpi": dpi} if ext == "png" or pdf_dpi else {}
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), bbox_inches="tight", **kw)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)


def write_data(out, stem, obj):
    """Write the figure's numbers next to it as `<out>/data/<stem>.json`, and return the path.

    Every figure in the report has one of these; it is what the writeup quotes from, so it is
    written even for figures whose numbers are also printed to stdout.
    """
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    p = os.path.join(out, "data", f"{stem}.json")
    json.dump(obj, open(p, "w"), indent=1)
    return p


def quantile_bins(x, nbins):
    """Equal-count bins over x -> (bin index per point, the edges). Equal-count rather than
    equal-width because the rarity axis is heavily skewed and fixed-width bins leave the rare tail
    with single-digit counts."""
    q = np.quantile(x, np.linspace(0, 1, nbins + 1))
    return np.clip(np.digitize(x, q[1:-1]), 0, nbins - 1), q


def deciles(x, y, nbins=10):
    """Bin x into `nbins` equal-count bins -> (x mean, y mean, y standard error, count) per bin."""
    b, _ = quantile_bins(x, nbins)
    m = [b == i for i in range(nbins)]
    return (np.array([x[s].mean() for s in m]),
            np.array([y[s].mean() for s in m]),
            np.array([y[s].std(ddof=1) / np.sqrt(max(s.sum(), 1)) for s in m]),
            np.array([s.sum() for s in m]))


def auc_rarer_worse(x, unv):
    """P(a rarer feature is the unverbalized one). x = rarity axis, unv = 1/0.

    Rank-based, so it is invariant to which log the rarity axis is in -- the point of reporting it
    next to a decile curve is that it does not depend on the binning.
    """
    from scipy import stats
    r = stats.rankdata(-x)
    n1, n0 = unv.sum(), (1 - unv).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[unv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
