"""Product `mu_diag` (GPU): WHY `stats/mu.f32` and Celeste's `whiten_mu.npy` disagree.

`mu_check` (stats.py) reports THAT they disagree -- cos 0.989 on the 8B, 0.978 on the 27B. This
product decomposes that gap into the four candidate causes, on the same smoke corpus, with ONE
forward pass per geometry:

  (a) GEOMETRY. `mu_512` = the mean under HER geometry (a document's first 512 tokens,
      `add_special_tokens=False`, NO sink, every position, documents shorter than 512 dropped --
      `targets._forward_512`, which is `data/collect_acts.py:52-60`) over every corpus document
      with >= 512 tokens. Against ours and against hers.
  (b) POSITION MIX. Our 64-token windows put 16/64 = 25% of their positions at window-relative
      index < 16; her 512-token windows put 16/512 = 3% there. Both means are recomputed over
      positions >= 16 only (counted AFTER the sink in our geometry) to take that out.
  (c) SINK AND MASSIVE-ACTIVATION TOKENS. Per geometry: the fraction of positions whose residual
      norm exceeds 10x that geometry's own median (`targets.NORM_FILTER_MULT`'s cut, applied here
      as a diagnostic, never as a filter) and the cosine between the mean WITH and WITHOUT them --
      i.e. how much of the mean's direction those few tokens carry. The sink's own norm (ours, the
      position that is excluded by construction) and position 0's norm (hers, an ordinary corpus
      token that is NOT excluded) are reported beside it.
  (d) SAMPLING NOISE. Each geometry's mean from two disjoint halves of the corpus (documents split
      by index parity) and the cosine between the halves: the floor below which no comparison here
      is meaningful.

What this CANNOT separate: the corpus. Ours is Ultra-FineWeb parts 0009-0010 (`corpus`); hers is
the head of part 0001 on the 8B and her own corpus on the 27B. Every row below is computed on OUR
corpus, so the residual left after (a)-(d) is "her corpus + everything else" and is not attributed.

Writes `<root>/base/<base>/stats/mu_diag/`: README.md, `mus.f32` [k, d] (the vectors, named in
`table.json`) and `table.json` (every comparison and every scalar).

Cost control: our geometry forwards ~4.1 token-copies per corpus token, so the corpus walk is
capped at `--tokens` (default 500k) corpus tokens; HER geometry uses every document >= 512 tokens
(cheap: one 512-token forward per document). The cap makes (d) a CONSERVATIVE bound -- sampling
noise at the cap is an upper bound on the noise of the full corpus mean.
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C
from precompute import stats as S
from precompute import targets as T

# The "massive activation" cut, as a multiple of the geometry's OWN median residual norm. 10x is
# the multiple targets.py already uses (NORM_FILTER_MULT) for its selection-time raw-norm filter.
MASSIVE_MULT = 10.0
# Window-relative position floor for (b). Celeste's realact draw never uses p < 16
# (targets.REALACT_P_MIN), which is what makes 16 the interesting cut rather than an arbitrary one.
POS_MIN = 16
# Positions buffered before the massive cut's threshold is fixed; see _MuAcc.add.
MEDIAN_WARM = 32768
# Corpus tokens walked under OUR geometry when --tokens is not given.
DEFAULT_TOKENS = 500_000
# Windows per read forward (stats.py pass A uses the same default).
BATCH_ROWS = 256
# Below this many >= 512-token documents, mu_512 is a mean over so few documents that (a) is
# reported with a warning instead of read as a geometry effect.
DOCS512_FLOOR = 200
# Warm-median vs full-corpus-median disagreement above which the threshold is called out.
MEDIAN_DRIFT = 0.25


class _MuAcc:
    """Running means of ONE geometry, split five ways, from one pass over its positions.

    `add` takes the kept positions of a batch as [P, d] fp32 plus, per position, its
    window-relative index and its document's parity. The massive-activation cut needs a median
    that is not known at the first batch, so the first MEDIAN_WARM positions are BUFFERED, the
    threshold is fixed from them, and the buffer is then replayed into the same accumulators: one
    forward pass, nothing read twice.
    """

    def __init__(self, name: str, d: int, device: str):
        import torch

        self.name = name
        self.d = d
        self.device = device
        self.sum_all = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_pos = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_light = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_half = [torch.zeros(d, dtype=torch.float64, device=device) for _ in range(2)]
        self.n_all = 0
        self.n_pos = 0
        self.n_light = 0
        self.n_half = [0, 0]
        self.norms: list[np.ndarray] = []
        self.ref_norms: list[np.ndarray] = []  # the sink (ours) / position 0 (hers)
        self.thr: float | None = None
        self.med_warm = 0.0
        self._warm: list[tuple] = []
        self._warm_n = 0

    def add(self, x, pos, half) -> None:
        import torch

        assert x.ndim == 2 and x.shape[1] == self.d, (
            f"{self.name}: expected [P, {self.d}] activations, got {tuple(x.shape)}"
        )
        assert pos.shape == (x.shape[0],) and half.shape == (x.shape[0],), (
            f"{self.name}: pos {tuple(pos.shape)} and half {tuple(half.shape)} must both be "
            f"[{x.shape[0]}], one entry per position"
        )
        x = x.to(self.device, torch.float32)
        norms = x.norm(dim=-1)
        self.norms.append(norms.cpu().numpy().astype(np.float32))
        item = (x, norms, pos.to(self.device), half.to(self.device))
        if self.thr is None:
            self._warm.append(item)
            self._warm_n += int(x.shape[0])
            if self._warm_n >= MEDIAN_WARM:
                self._fix_threshold()
            return
        self._apply(*item)

    def add_ref(self, norms: np.ndarray) -> None:
        """The reference position's norms for this batch: the sink (ours) or position 0 (hers)."""
        self.ref_norms.append(np.asarray(norms, dtype=np.float32))

    def _fix_threshold(self) -> None:
        import torch

        assert self._warm, f"{self.name}: no positions buffered, cannot fix the massive cut"
        med = float(torch.cat([n for _, n, _, _ in self._warm]).median())
        assert med > 0.0, (
            f"{self.name}: median residual norm over the first {self._warm_n} positions is {med}, "
            "which cannot be right for a block output"
        )
        self.med_warm = med
        self.thr = MASSIVE_MULT * med
        print(
            f"[mu_diag] {self.name}: massive cut = {MASSIVE_MULT:g}x median {med:.2f} = "
            f"{self.thr:.2f}, fixed on the first {self._warm_n} positions",
            flush=True,
        )
        for item in self._warm:
            self._apply(*item)
        self._warm.clear()

    def _apply(self, x, norms, pos, half) -> None:
        import torch

        self.sum_all += x.sum(0, dtype=torch.float64)
        self.n_all += int(x.shape[0])
        m = pos >= POS_MIN
        n_pos = int(m.sum())
        if n_pos:
            self.sum_pos += x[m].sum(0, dtype=torch.float64)
            self.n_pos += n_pos
        light = norms <= self.thr
        n_light = int(light.sum())
        if n_light:
            self.sum_light += x[light].sum(0, dtype=torch.float64)
            self.n_light += n_light
        for h in (0, 1):
            hm = half == h
            n_h = int(hm.sum())
            if n_h:
                self.sum_half[h] += x[hm].sum(0, dtype=torch.float64)
                self.n_half[h] += n_h

    def _mean(self, total, n: int) -> np.ndarray:
        assert n > 0, f"{self.name}: a mean was requested over 0 positions"
        return (total / n).cpu().numpy().astype(np.float32)

    def finish(self) -> dict:
        """The five means plus this geometry's scalars. Call once, after the last `add`."""
        if self.thr is None:  # fewer than MEDIAN_WARM positions in the whole pass
            self._fix_threshold()
        assert self.n_all > 0, f"{self.name}: no positions were accumulated"
        assert self.n_half[0] > 0 and self.n_half[1] > 0, (
            f"{self.name}: split-half needs both parities, got {self.n_half} positions"
        )
        norms = np.concatenate(self.norms)
        med = float(np.median(norms))
        drift = abs(med - self.med_warm) / max(self.med_warm, 1e-9)
        if drift > MEDIAN_DRIFT:
            print(
                f"[mu_diag] REPORTED, not acted on: {self.name}: the warm-buffer median "
                f"{self.med_warm:.2f} that fixed the massive cut is {drift:.1%} away from the "
                f"full-pass median {med:.2f}; the cut is 10x the FORMER",
                flush=True,
            )
        ref = np.concatenate(self.ref_norms) if self.ref_norms else np.zeros(0, np.float32)
        return {
            "mu": self._mean(self.sum_all, self.n_all),
            "mu_pos16": self._mean(self.sum_pos, self.n_pos),
            "mu_light": self._mean(self.sum_light, self.n_light),
            "mu_half0": self._mean(self.sum_half[0], self.n_half[0]),
            "mu_half1": self._mean(self.sum_half[1], self.n_half[1]),
            "scalars": {
                "positions": self.n_all,
                "positions_pos16": self.n_pos,
                "positions_half": list(self.n_half),
                "median_norm": round(med, 3),
                "median_norm_warm": round(self.med_warm, 3),
                "massive_threshold": round(float(self.thr), 3),
                "massive_positions": self.n_all - self.n_light,
                "massive_fraction": round(1.0 - self.n_light / self.n_all, 6),
                "mean_norm": round(float(norms.mean()), 3),
                "ref_norm_n": int(ref.size),
                "ref_norm_mean": round(float(ref.mean()), 3) if ref.size else None,
                "ref_norm_median": round(float(np.median(ref)), 3) if ref.size else None,
            },
        }


def _pass_ours(model, args, toks, docs, read_layer, sink, pad_id, acc) -> dict:
    """Our geometry: 64/16 windows, `[sink] + window` forwarded, the sink dropped (stats.py pass A).

    The batch is built by `stats._pad_batch`, the very function that produced `stats/mu.f32`, so
    this arm cannot drift from the vector it is being compared against.
    """
    import torch

    max_tokens = int(args.get("tokens") or DEFAULT_TOKENS)
    buf: list = []
    buf_doc: list[int] = []
    walked_tokens, walked_docs, n_windows = 0, 0, 0
    t0 = time.time()

    def flush():
        nonlocal n_windows
        if not buf:
            return
        batch, _ = S._pad_batch(buf, sink, pad_id, "cuda")
        h, mask = C.read_resid(model, read_layer, batch, pool="all")
        keep = mask.clone()
        keep[:, 0] = False  # the sink is never a scanned position -- stats.py:_read_batch
        cols = torch.arange(h.shape[1], device=h.device).expand(h.shape[0], -1)
        rows = torch.tensor(buf_doc, device=h.device, dtype=torch.int64).unsqueeze(1)
        rows = rows.expand(-1, h.shape[1])
        # window-relative position: column 0 is the sink, so the window starts at column 1
        acc.add(h[keep], (cols[keep] - 1), rows[keep] % 2)
        acc.add_ref(h[:, 0].norm(dim=-1).cpu().numpy())
        n_windows += len(buf)
        buf.clear()
        buf_doc.clear()

    for r in docs:
        if walked_tokens >= max_tokens:
            break
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, n in C.windows_of(r["len"]):
            buf.append(ids[s : s + n])
            buf_doc.append(int(r["doc"]))
            if len(buf) >= BATCH_ROWS:
                flush()
        walked_tokens += int(r["len"])
        walked_docs += 1
        if walked_docs % 200 == 0:
            print(
                f"[mu_diag] ours: {walked_docs} docs, {walked_tokens / 1e6:.2f}M corpus tokens, "
                f"{n_windows} windows, {time.time() - t0:.0f}s",
                flush=True,
            )
    flush()
    el = time.time() - t0
    print(
        f"[mu_diag] ours: {walked_docs} docs / {walked_tokens} corpus tokens / {n_windows} windows "
        f"/ {acc.n_all} positions in {el:.0f}s",
        flush=True,
    )
    return {
        "docs": walked_docs,
        "corpus_tokens": walked_tokens,
        "windows": n_windows,
        "seconds": round(el, 1),
        "token_cap": max_tokens,
        "capped": walked_tokens >= max_tokens,
    }


def _pass_hers(model, toks, docs512, read_layer, acc) -> dict:
    """Her geometry: the first 512 tokens of each >= 512-token document, no sink, all positions.

    `targets._forward_512` IS Celeste's recipe as this repo reads it (module docstring there); it
    is called here rather than re-implemented so the two cannot diverge.
    """
    import torch

    t0 = time.time()
    for s in range(0, len(docs512), T.REALACT_DOCS_PER_FWD):
        chunk = docs512[s : s + T.REALACT_DOCS_PER_FWD]
        h = T._forward_512(model, read_layer, chunk, toks, "cuda")  # [n, 512, d] fp32, cpu
        n, width = int(h.shape[0]), int(h.shape[1])
        assert width == T.REALACT_WINDOW, f"expected {T.REALACT_WINDOW}-token windows, got {width}"
        pos = torch.arange(width).unsqueeze(0).expand(n, -1)
        half = torch.tensor([int(r["doc"]) % 2 for r in chunk], dtype=torch.int64).unsqueeze(1)
        acc.add(h.reshape(-1, h.shape[2]), pos.reshape(-1), half.expand(-1, width).reshape(-1))
        acc.add_ref(h[:, 0].norm(dim=-1).numpy())
        if (s // T.REALACT_DOCS_PER_FWD) % 10 == 0:
            print(
                f"[mu_diag] hers: {min(s + T.REALACT_DOCS_PER_FWD, len(docs512))}/{len(docs512)} "
                f"documents, {time.time() - t0:.0f}s",
                flush=True,
            )
    el = time.time() - t0
    print(f"[mu_diag] hers: {len(docs512)} documents / {acc.n_all} positions in {el:.0f}s", flush=True)
    return {"docs": len(docs512), "window": T.REALACT_WINDOW, "seconds": round(el, 1)}


def _cmp(name: str, a: np.ndarray, b: np.ndarray, note: str) -> dict:
    """cos and norm ratio ||a|| / ||b|| between two means, in float64."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape, f"{name}: shapes {a.shape} and {b.shape} differ"
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    assert na > 0 and nb > 0, f"{name}: a zero mean ({na=}, {nb=}) has no direction"
    return {
        "row": name,
        "cos": round(float(a @ b / (na * nb)), 6),
        "norm_ratio": round(na / nb, 6),
        "norm_a": round(na, 4),
        "norm_b": round(nb, 4),
        "note": note,
    }


def run(cfg, args):
    base, root = args["base"], args["root"]
    assert base, "product mu_diag needs --base"
    assert base in S.ARCHIVE_MU, (
        f"no archived whiten_mu path known for base {base!r} ({sorted(S.ARCHIVE_MU)}); mu_diag has "
        "nothing to compare against on this base"
    )
    spec = cfg["bases"][base]
    read_layer, d = spec["read_layer"], spec["d"]

    # Everything that can fail without a GPU fails before the weights load.
    out = f"{C.stats_dir(base, root)}/mu_diag"
    assert args.get("force") or not os.path.exists(out), (
        f"{out} already exists; refusing to overwrite without --force"
    )
    toks, docs = C.load_corpus(base, root)
    ours = C.stats_mu(cfg, base, root).astype(np.float64)  # fails loudly if `stats` never ran
    apath = os.path.join(cfg["modal"]["archive"], S.ARCHIVE_MU[base])
    assert os.path.exists(apath), (
        f"missing archived whiten_mu at {apath}; look under {cfg['modal']['archive']}/data/ for the "
        f"{base} tree and report where it actually is"
    )
    hers = np.load(apath).astype(np.float64).reshape(-1)
    assert hers.shape == (d,), f"{apath} is {hers.shape}, expected [{d}] for base {base!r}"

    docs512 = [r for r in docs if r["len"] >= T.REALACT_WINDOW]
    assert docs512, (
        f"no corpus document in {C.corpus_dir(base, root)} has >= {T.REALACT_WINDOW} tokens, so "
        "her geometry cannot be reproduced here at all"
    )
    if len(docs512) < DOCS512_FLOOR:
        print(
            f"[mu_diag] REPORTED, not acted on: only {len(docs512)} corpus documents have "
            f">= {T.REALACT_WINDOW} tokens (floor {DOCS512_FLOOR}); mu_512 is a thin mean here and "
            "row (a) must be read with the split-half noise of row (d) beside it",
            flush=True,
        )

    t_load = time.time()
    model, tok = C.load_base(cfg, base)
    load_s = time.time() - t_load
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink

    acc_ours = _MuAcc("ours (64/16, sink dropped)", d, "cuda")
    acc_hers = _MuAcc("hers (512, no sink)", d, "cuda")
    pass_ours = _pass_ours(model, args, toks, docs, read_layer, sink, pad_id, acc_ours)
    pass_hers = _pass_hers(model, toks, docs512, read_layer, acc_hers)
    g_ours = acc_ours.finish()
    g_hers = acc_hers.finish()

    names = [
        "stats_mu",  # 0: the stored 64/16 mean of the FULL smoke corpus -- the anchor "ours"
        "whiten_mu",  # 1: Celeste's archived mean -- the anchor "hers"
        "mu_64",  # 2: our geometry, recomputed here on the walked prefix
        "mu_64_pos16",
        "mu_64_light",
        "mu_64_half0",
        "mu_64_half1",
        "mu_512",  # 7: her geometry on OUR corpus
        "mu_512_pos16",
        "mu_512_light",
        "mu_512_half0",
        "mu_512_half1",
    ]
    vecs = {
        "stats_mu": ours.astype(np.float32),
        "whiten_mu": hers.astype(np.float32),
        "mu_64": g_ours["mu"],
        "mu_64_pos16": g_ours["mu_pos16"],
        "mu_64_light": g_ours["mu_light"],
        "mu_64_half0": g_ours["mu_half0"],
        "mu_64_half1": g_ours["mu_half1"],
        "mu_512": g_hers["mu"],
        "mu_512_pos16": g_hers["mu_pos16"],
        "mu_512_light": g_hers["mu_light"],
        "mu_512_half0": g_hers["mu_half0"],
        "mu_512_half1": g_hers["mu_half1"],
    }
    mus = np.stack([vecs[n] for n in names]).astype(np.float32)

    v = vecs
    rows = [
        _cmp("anchor: ours vs hers", v["stats_mu"], v["whiten_mu"], "what mu_check reports"),
        _cmp("anchor: mu_512 vs hers", v["mu_512"], v["whiten_mu"], "her geometry on OUR corpus"),
        _cmp(
            "control: mu_64 vs ours",
            v["mu_64"],
            v["stats_mu"],
            f"the {pass_ours['corpus_tokens']}-token walk against the stored full-corpus mean",
        ),
        _cmp("(a) mu_512 vs ours", v["mu_512"], v["stats_mu"], "geometry alone, same corpus"),
        _cmp("(b) mu_64_pos16 vs ours", v["mu_64_pos16"], v["stats_mu"], "our own early positions"),
        _cmp("(b) mu_64_pos16 vs hers", v["mu_64_pos16"], v["whiten_mu"], ""),
        _cmp("(b) mu_64_pos16 vs mu_512", v["mu_64_pos16"], v["mu_512"], "both geometries, pos >= 16"),
        _cmp("(b) mu_512_pos16 vs ours", v["mu_512_pos16"], v["stats_mu"], ""),
        _cmp("(b) mu_512_pos16 vs hers", v["mu_512_pos16"], v["whiten_mu"], ""),
        _cmp("(b) mu_512_pos16 vs mu_512", v["mu_512_pos16"], v["mu_512"], "her own early positions"),
        _cmp(
            "(c) mu_64_light vs mu_64",
            v["mu_64_light"],
            v["mu_64"],
            f"{g_ours['scalars']['massive_fraction']:.4%} of our positions dropped",
        ),
        _cmp(
            "(c) mu_512_light vs mu_512",
            v["mu_512_light"],
            v["mu_512"],
            f"{g_hers['scalars']['massive_fraction']:.4%} of her positions dropped",
        ),
        _cmp("(d) mu_64 halves", v["mu_64_half0"], v["mu_64_half1"], "our geometry, doc parity"),
        _cmp("(d) mu_512 halves", v["mu_512_half0"], v["mu_512_half1"], "her geometry, doc parity"),
    ]
    for r in rows:
        print(f"[mu_diag] {r['row']:<34} cos={r['cos']:.6f} ratio={r['norm_ratio']:.6f}", flush=True)

    inputs = {
        "corpus": C.corpus_dir(base, root),
        "corpus_docs": len(docs),
        "corpus_tokens": int(len(toks)),
        "docs_ge_512": len(docs512),
        "read_layer": read_layer,
        "archive_whiten_mu": apath,
        "stats_mu": f"{C.stats_dir(base, root)}/mu.f32",
        "base_load_seconds": round(load_s, 1),
    }
    with C.outdir(out, args, inputs=inputs) as od:
        od.write_array("mus.f32", mus, "float32")
        od.write_json(
            "table.json",
            {
                "base": base,
                "read_layer": read_layer,
                "d": d,
                "massive_mult": MASSIVE_MULT,
                "pos_min": POS_MIN,
                "docs_ge_512": len(docs512),
                "docs_ge_512_floor": DOCS512_FLOOR,
                "pass_ours": pass_ours,
                "pass_hers": pass_hers,
                "geometry": {"ours": g_ours["scalars"], "hers": g_hers["scalars"]},
                "mus": {"file": "mus.f32", "names": names, "shape": list(mus.shape)},
                "comparisons": rows,
            },
        )
        od.section(
            "What each row separates",
            [
                "`ours` = `stats/mu.f32` (64-token windows every 16 tokens, `[sink] + window`",
                "forwarded, the sink dropped) over the FULL smoke corpus. `hers` =",
                f"`{apath}` (512-token windows, `add_special_tokens=False`, no sink, every",
                "position, documents shorter than 512 dropped) over HER corpus.",
                "",
                "- **(a) geometry**: `mu_512` is HER geometry on OUR corpus. Its cosine against",
                "  `ours` is the geometry effect with the corpus held fixed; its cosine against",
                "  `hers` is what is left once geometry is removed -- corpus and everything else.",
                "- **(b) position mix**: our windows put 25% of their positions at window-relative",
                f"  index < {POS_MIN}, hers put {POS_MIN / T.REALACT_WINDOW:.1%} there. Both means",
                f"  are recomputed over positions >= {POS_MIN} (counted after the sink for ours).",
                f"- **(c) massive activations**: positions with residual norm > {MASSIVE_MULT:g}x",
                "  that geometry's own median. `*_light` is the mean with them removed, so the",
                "  cosine says how much of the mean's DIRECTION those few tokens carry. The sink",
                "  (ours) is excluded by construction and reported only as `ref_norm_*`; position",
                "  0 (hers) is an ordinary token and IS in her mean.",
                "- **(d) sampling noise**: the same geometry over documents of even vs odd index.",
                "  The token cap makes this an UPPER bound on the full corpus's noise.",
                "",
                "NOT separated here: the corpus itself. Every vector except `whiten_mu` is",
                "computed on our corpus, so whatever (a)-(d) leave unexplained is her corpus plus",
                "any remaining convention difference, and this product cannot split those.",
            ],
        )
        for r in rows:
            note = f" ({r['note']})" if r["note"] else ""
            od.note(f"{r['row']}: cos {r['cos']:.6f}, norm ratio {r['norm_ratio']:.6f}{note}")
        od.note(
            f"our geometry walked {pass_ours['docs']} documents / {pass_ours['corpus_tokens']} "
            f"corpus tokens ({'CAPPED' if pass_ours['capped'] else 'whole corpus'}, cap "
            f"{pass_ours['token_cap']}) = {g_ours['scalars']['positions']} positions in "
            f"{pass_ours['seconds']}s"
        )
        od.note(
            f"her geometry used all {pass_hers['docs']} documents with >= {T.REALACT_WINDOW} "
            f"tokens = {g_hers['scalars']['positions']} positions in {pass_hers['seconds']}s"
            + ("" if len(docs512) >= DOCS512_FLOOR else f" -- BELOW the {DOCS512_FLOOR}-document floor")
        )
        od.note(
            f"median residual norm at layer {read_layer}: ours {g_ours['scalars']['median_norm']}, "
            f"hers {g_hers['scalars']['median_norm']}; massive cut at "
            f"{g_ours['scalars']['massive_threshold']} / {g_hers['scalars']['massive_threshold']}"
        )
        od.note(
            f"reference position norm: our SINK (dropped from mu) mean "
            f"{g_ours['scalars']['ref_norm_mean']}, her POSITION 0 (kept in her mean) mean "
            f"{g_hers['scalars']['ref_norm_mean']}"
        )
        od.note("mus.f32 is [k, d] float32; row order is table.json's `mus.names`")

    return {
        "base": base,
        "out": out,
        "docs_ge_512": len(docs512),
        "pass_ours": pass_ours,
        "pass_hers": pass_hers,
        "scalars": {"ours": g_ours["scalars"], "hers": g_hers["scalars"]},
        "comparisons": {r["row"]: {"cos": r["cos"], "norm_ratio": r["norm_ratio"]} for r in rows},
        "base_load_seconds": round(load_s, 1),
    }
