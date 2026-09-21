"""Stage `build` (P2 of the autointerp design): the rendered example sets and the shared test set.

    <root>/base/<base>/autointerp/<set>/<date>_build/
        <feature>.jsonl   one file per tested feature: a `meta` row, one `arm` row per
                          example-set variant (the rendered explainer user message plus the
                          provenance of every example shown), and 40 `test` rows
        build.json        the draw, the seeds, the arm definitions, every shortfall and the
                          marking statistics
        features.json     the feature table (row, id, stratum, density, corpus peak, fire fraction)

CPU only: everything it needs is already on the volume. The corpus `examples/<feature>.jsonl` rows
carry `(doc, start, len, acts)` and NO text (precompute/scan.py:486-501), so the window's tokens are
recovered from `corpus/tokens.i32` at that offset and the recovered length is ASSERTED equal to
`len(acts)` -- that assert is the whole join.

Arms (design §2; N = `autointerp.n_examples` = 16 unless the variant says otherwise):

  C16      the top-16 corpus windows by peak activation over the full 16M corpus
  C4       the top-16 among those whose document lies in the nested 4M prefix
  M        the top-16 of the MAEMM's 64 rollouts by peak target-feature activation (needs `sae_self`)
  C4M      8 corpus (C4 ranks 1-8) + 8 rollouts (M ranks 1-8), shuffled
  C16M16   ADDITIVE ablation, MATCHED-N: all 16 of C16 plus all 16 of M, N = 32, read against
           C16-N32 (32 corpus windows) so the only difference is where the second 16 come from
  C32      the matched-N corpus control C16M16 is read against: 32 corpus windows at 16M
  C16-N8 / M-N8 / M-N32   descriptive pilot points only (A9: N = 16 is fixed a priori)
  E        NOT RUN. `--epo-strings <jsonl>` is a documented hook, see `_epo_arm` below

Rendering is Delphi's (facts §3, transcription at repo-maemm/eval/autointerp_detection.py:229-450):
activating tokens wrapped `<<like this>>`, then an `Activations:` line of `("tok" : n)` pairs with
n = ceil(10 * act / peak_f) clamped to [0, 10] and peak_f = the feature's corpus max at 16M
(`sae/<sae>/max_act.f16`). Only the first 10 activating tokens of an example are listed, as Delphi
does. All activations everywhere are the stored PRE-GATE ones.

Test set (design §3 as amended 2026-09-16), identical across arms and never shown to any explainer:
  * 20 positives, 5 from each of the four stored activation bands `q0..q3`, drawn ONLY from rows
    whose peak pre-gate activation exceeds the SAE gate. Those bands are EQUAL-WIDTH bins of
    (0, max_act], not Delphi's quantiles (precompute/scan.py:257) -- stated, not relabelled. A
    short band carries its deficit to the next band DOWN; what is still short is filled from the
    top-ranked windows beyond those any arm shows, counted as `n_top_fallback`.
  * 20 negatives from the 2048-window `random_pool` (`sae_self.run_random_pool`), by default all
    with a per-feature maximum of exactly 0 (Delphi's published `non_activating_source "random"`);
    `autointerp.n_neg_nearmiss` moves part of the quota to near-miss windows (0 < max <= gate).
  * DOCUMENT-level disjointness from every shown example, and between test items, ASSERTED.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np

import precompute.common as C
from precompute.rollouts_nla import (explanation_body, explanation_token_mask,
                                     extract_explanation)

# The held-out families whose targets ARE SAE features, so a "the feature's own activation on
# this text" arm is meaningful for them. `sae` is the 131k `l42-1b` draw (config.yaml's
# 2026-09-16_v1); `sae2m_enc` is the label Ari's `features/draw_sae2m.py` writes for the 2M-SAE
# encoder columns -- a different dictionary and a different draw, but the same KIND of target
# (row["id"] is the feature index either way), which is all anything here needs. Kept as a tuple
# rather than collapsed to one name because a set can carry both and the labels are provenance.
FAMILIES = C.SAE_FAMILIES
# Delphi lists at most this many activating tokens per example (facts §3, explainer.py).
MAX_SHOWN_ACTS = 10
BANDS = ("q0", "q1", "q2", "q3")
# Delphi's `example_ctx_len`, used by the PREPARED-NOT-RUN `--centre32` corpus arm below.
CENTRE32_LEN = 32
# Delphi's explainer highlight threshold, FETCHED 2026-09-17 from `explainers/explainer.py`
# @4fea06e: `threshold: float = 0.3`, applied as `max(activations) * self.threshold`.
REL_MARK_FRAC = 0.5   # the relative fallback's fraction of a block's own peak
DELPHI_MARK_FRAC = 0.3

# The NLA arm's example count = config.yaml's `nla.n`. Kept as a constant here because ARM_SPECS
# is a literal table and a 4 in it would look like a typo beside the 16s.
NLA_N = 4
# The arm whose examples ARE the MAEMM's / verbalizer's rollouts, by rollout source. An `nla`
# entry may only be built into "NLA": the others are named in the paper as MAEMM arms and a
# verbalizer's text under the label `M` would be a mislabelled number, not a variant.
ROLLOUT_ARMS_NOT_NLA = ("M", "M-div", "C4M", "C16M16", "M-N8", "M-N32")
NLA_ARM = "NLA"


# ---------------------------------------------------------------------------------------------
# Delphi rendering
# ---------------------------------------------------------------------------------------------


def quant_act(act: float, peak: float) -> int:
    """Delphi's activation display: `(act * 10 / max_activation).ceil().clamp(0, 10)`.

    `latents/samplers.py`, reimplemented at repo-maemm/eval/autointerp_detection.py:345-354.
    `peak` is the feature's CORPUS max at 16M, i.e. Delphi's per-latent global maximum. A MAEMM
    rollout can exceed that (act/peak > 1); the clamp at 10 is Delphi's own and hides it, which is
    why every arm row also stores the raw activation.
    """
    if peak <= 0:
        return 0
    return int(min(10, max(0, math.ceil(10.0 * act / peak))))


def token_pieces(tok, ids) -> list[str]:
    """One decoded string per token id, so a marker lands on exactly the token it belongs to.

    Byte-level BPE can split a codepoint across two tokens, in which case the per-token decode
    emits a replacement character and the join is not the same string as `tok.decode(ids)`. The
    caller reports the rate rather than asserting, exactly as `score.py` does for its own round
    trip (precompute/score.py:145-150).
    """
    return [tok.decode([int(i)]) for i in ids]


def marked_text(pieces: list[str], marks) -> str:
    """`pieces` joined with `<<`/`>>` around each MAXIMAL RUN of marked tokens.

    Delphi: "If a sequence of consecutive tokens all are important, the entire sequence of tokens
    will be contained between delimiters <<just like this>>."
    """
    out: list[str] = []
    inside = False
    for piece, m in zip(pieces, marks, strict=True):
        if m and not inside:
            out.append("<<")
            inside = True
        elif not m and inside:
            out.append(">>")
            inside = False
        out.append(piece)
    if inside:
        out.append(">>")
    return "".join(out)


def exemplar_block(examples: list[dict]) -> str:
    """Delphi's explainer rendering of a whole example set.

    `Example {1-based}:  {marked text}` then `Activations: ("tok" : n), ...` -- the two-space gap
    and the 1-based numbering are Delphi's explainer few-shot's own (the DETECTION prompt is
    0-based and single-spaced; both are reproduced where they belong).
    """
    out = []
    for j, e in enumerate(examples):
        line = f"Example {j + 1}:  {e['text_marked']}"
        pairs = ", ".join(f'("{t}" : {n})' for t, n in e["activations"])
        out.append(f"{line}\nActivations: {pairs}" if pairs else line)
    return "\n".join(out)


def centre_on_peak(ids, acts, width: int = CENTRE32_LEN):
    """Re-cut a window to `width` tokens centred on its peak activation. Returns (ids, acts).

    Delphi's `example_ctx_len 32` + `center_examples True`. Our corpus windows are 64 tokens with
    the peak anywhere in them; a MAEMM rollout, by contrast, tends to END at its peak because that
    is where the RL reward is, so centring is not symmetric between the arms and this is offered
    for the CORPUS side only.
    """
    a = np.asarray(acts, dtype=np.float32)
    if len(a) <= width:
        return ids, list(a)
    c = int(a.argmax())
    lo = max(0, min(c - width // 2, len(a) - width))
    return ids[lo : lo + width], [float(x) for x in a[lo : lo + width]]


def render_example(tok, ids, acts, peak: float, gate: float, mark: str = "gate",
                   rel_fallback: bool = False) -> dict:
    """One rendered explainer example from an id list and its per-token pre-gate activations.

    TWO MARKING RULES, and the published run used the first. FETCHED 2026-09-17 from Delphi
    @4fea06e (`explainers/explainer.py`), which settles what "Delphi's rule" is:

      `mark="gate"`   (DEFAULT, what the 2026-09-16 512-feature run used): a token is marked iff
                      its pre-gate activation exceeds the SAE's learned gate, and the
                      `Activations:` line lists the marked tokens by DESCENDING activation, first
                      10. This is the paper's own fire rule applied to rendering. It is NOT
                      Delphi's, and calling it "Delphi's post-TopK activating" was wrong.
      `mark="delphi"` Delphi's actual rule: `threshold = max(activations) * 0.3` -- 0.3 x THIS
                      EXAMPLE'S OWN max, not the gate and not the corpus peak -- and the
                      `Activations:` line lists them in TEXT ORDER, first 10 (`_highlight` and
                      `_join_activations` iterate the sequence; `if activation_count > 10: break`).

    The difference is not cosmetic across arms: a rollout carries higher activation relative to the
    corpus peak than a corpus window does, so the gate rule marks proportionally more of a rollout,
    and 299 of 512 features have shown rollouts exceeding the corpus peak.
    """
    pieces = token_pieces(tok, ids)
    a = [float(x) for x in acts]
    # A NON-FINITE ACTIVATION MAKES THE WHOLE BLOCK UNMARKABLE, and does it HERE rather than in
    # the marking branch below, because `quant_act` raises on NaN and +inf (int(ceil(inf))) and
    # would take the process down before any guard in that branch could run. `sae_self` writes NaN
    # outside `keep` and every caller slices by `keep`, so this should be unreachable -- which is
    # the reason to make it a stated outcome instead of a crash: if it ever does arrive, a block
    # that says "nothing marked here" is recoverable and a traceback in the middle of a paid build
    # is not.
    finite = all(math.isfinite(x) for x in a)
    quant = [quant_act(x, peak) if math.isfinite(x) else 0 for x in a]
    if not finite:
        return {
            "text": "".join(pieces),
            "text_marked": "".join(pieces),
            "activations": [],
            "n_marked": 0,
            "peak_act": 0.0,
            "marking": "unmarkable",
            "block_peak": 0.0,
            "peak_frac": None,
            "n_tok": len(pieces),
            "join_ok": "".join(pieces) == tok.decode([int(i) for i in ids]),
        }
    if mark == "delphi":
        marking = "delphi"
        thr = max(a) * DELPHI_MARK_FRAC if a else 0.0
        marks = [x > thr for x in a]
        shown = [(pieces[i], quant[i]) for i in range(len(pieces)) if marks[i]][:MAX_SHOWN_ACTS]
    else:
        marks = [x > gate for x in a]
        marking = "gate"
        # THE RELATIVE FALLBACK (Tomas, 2026-09-21), for the GENERATED-TEXT arms only.
        # `mark="gate"` marks a token iff it clears the SAE's learned gate, which is the paper's
        # own fire rule -- and on the 2M dictionary a MAEMM rollout frequently clears it nowhere,
        # so the block reaches the explainer as bare `Example n:` lines with no `<<>>` and no
        # `Activations:` line at all. MEASURED 2026-09-21 on the 32-feature pilot: 15/32 of
        # rl-last16's blocks, 23/32 of the old primary's and 26/32 of NLA's, against 0/32 for
        # every corpus arm. One explainer answer opens "there's no explicit token
        # highlighting/activation data provided" and is recorded as an ordinary explanation, so
        # the arm was being scored on a description written from unmarked text.
        #
        # When nothing clears the gate, mark relative to THIS BLOCK's own peak instead. That is
        # Delphi's rule in shape (`mark="delphi"` uses 0.3 x the example's own max) at a stricter
        # fraction. It is a FALLBACK, not a replacement: a block with anything above the gate is
        # marked exactly as before, so this cannot move a block that was already fine.
        #
        # The corpus arms never take this path -- they are passed `rel_fallback=False` -- because
        # their peak IS the corpus peak by construction and a corpus window that fires nowhere is
        # a real fact about the feature, not a rendering failure.
        if rel_fallback and not any(marks):
            pk = max(a) if a else 0.0
            if math.isfinite(pk) and pk > 0:
                # `>=`, not `>`: at the fraction's own boundary the peak token itself must mark,
                # and with rel_frac = 0.5 a two-token block at (pk, pk/2) marks both.
                marks = [x >= REL_MARK_FRAC * pk for x in a]
                marking = "relative"
            else:
                # A block whose peak is <= 0 or non-finite stays UNMARKED and is counted. There is
                # no fraction of zero that marks anything, and marking everything would be worse
                # than marking nothing: it would tell the explainer the feature fires everywhere.
                marking = "unmarkable"
        shown = sorted(
            ((pieces[i], quant[i], a[i]) for i in range(len(pieces)) if marks[i]),
            key=lambda t: -t[2],
        )[:MAX_SHOWN_ACTS]
        shown = [(t, n) for t, n, _ in shown]
    block_peak = float(max(acts)) if len(acts) else 0.0
    return {
        "text": "".join(pieces),
        "text_marked": marked_text(pieces, marks),
        "activations": shown,
        "n_marked": int(sum(marks)),
        "peak_act": round(block_peak, 4),
        # Provenance of the marking, per block, so a reader can see WHICH rule produced a block and
        # how weak it was: `marking` is gate | relative | unmarkable | delphi, `peak_frac` is this
        # block's peak as a fraction of the feature's 16M corpus peak (the quantisation
        # denominator), which is the number that says how far below the corpus this text sits.
        "marking": marking,
        "block_peak": round(block_peak, 4),
        "peak_frac": round(block_peak / peak, 4) if peak > 0 else None,
        "n_tok": len(pieces),
        "join_ok": "".join(pieces) == tok.decode([int(i) for i in ids]),
    }


def render_test(tok, ids, acts, gate: float, rng: random.Random, n_mark_neg: int,
                fuzz_marks: str = "contiguous") -> dict:
    """One test item: the plain text the DETECTION scorer sees and the marked text FUZZING sees.

    Marking rule: design amendment A2's single rule, `act > gate` -- the same rule
    `render_example` uses, so a fuzzing mark and an explainer mark mean the same event. That is a
    DEVIATION from Delphi, whose fuzzing scorer marks `act > 0.3 * max_activation`
    (related-work/2026-09-15_delphi-updates-and-negatives.md:49); the gate is the fire rule the
    rest of this paper uses. Every test positive is gate-consistent (A1), so its peak is marked by
    construction.

    A negative window has no activation to mark, so marks are placed at random. TWO RULES, and the
    published run used the first:

      `fuzz_marks="contiguous"` (DEFAULT, the 512-feature run): one contiguous run of `n_mark_neg`
                      tokens at a seeded random start. Reconstructed from Delphi's INTRUDER paper
                      when its fuzzing source was not to hand. It is a detectable artefact --
                      negatives carry one block, positives carry 1-2 scattered marks.
      `fuzz_marks="scattered"` below-threshold tokens chosen by `random.sample`, i.e. scattered
                      but with no forced index -- an intermediate rule, kept because the
                      sensitivity run used it.
      `fuzz_marks="delphi"` UPSTREAM's rule at 4fea06e, vendored at third_party/delphi-4fea06e/:
                      scattered AND with the index `len-len//4` forced in. See the branch below
                      for the two places we still differ.

    `n_mark_neg` is the caller's `n_incorrect`. Upstream computes it as `ceil(mean number of
    NONZERO activations over the test positives)` (`fuzz.py:61-69`); we use the mean number of
    MARKED tokens over the rendered positives, floored, min 1. Those agree when the marking rule
    and the nonzero rule pick the same tokens and differ otherwise -- ours is tied to what the
    positives actually show, which is the quantity the negative is meant to imitate. Listed as a
    deviation in the README.
    """
    pieces = token_pieces(tok, ids)
    if acts is not None:
        a = np.asarray(acts, dtype=np.float32)
        marks = list(a > gate)
        if not any(marks) and len(a):
            # f16 TOLERANCE, not a second rule. A positive is selected on the stored scalar
            # `max_act` (an f32 max, rounded to 4 dp) while `acts` is the f16-rounded per-token
            # payload, so a peak a hair above the gate can quantise a hair below it and leave a
            # gate-consistent positive (A1) with no mark at all. The peak is marked in that case
            # and only that case; it is the same token either way.
            marks[int(a.argmax())] = True
    elif fuzz_marks == "delphi":
        # UPSTREAM's rule, `scorers/classifier/sample.py:118-166` @4fea06e, vendored verbatim at
        # third_party/delphi-4fea06e/. Two differences from "scattered", both deliberate on
        # upstream's part and one of them carrying upstream's own `# TODO: This is wrong`:
        #
        #   1. the index `len(str_toks) - len(str_toks)//4` is FORCED IN whenever it is below
        #      threshold, and only `n_incorrect - 1` further indices are sampled. Upstream's
        #      windows are centred so that position IS the activating token, and marking it on a
        #      negative is what makes the negative a plausible false positive rather than a
        #      random smear. Our windows are centred the same way (`centre_on_peak`), so the
        #      position means the same thing here.
        #   2. `random.seed(22)` -- upstream reseeds the GLOBAL RNG inside the marking function,
        #      so every example in a run gets the same sample sequence. We keep our own seeded
        #      `rng` instead: reseeding a shared global from inside a renderer would make this
        #      function's output depend on call order, and the run is parallel. NOT byte-identical
        #      to upstream for that reason, and the README says so.
        k = max(1, min(n_mark_neg, len(pieces)))
        forced = len(pieces) - len(pieces) // 4
        rest = [i for i in range(len(pieces)) if i != forced]
        idx = ({forced, *rng.sample(rest, min(k - 1, len(rest)))} if 0 <= forced < len(pieces)
               else set(rng.sample(range(len(pieces)), k)))
        marks = [i in idx for i in range(len(pieces))]
    elif fuzz_marks == "scattered":
        k = max(1, min(n_mark_neg, len(pieces)))
        idx = set(rng.sample(range(len(pieces)), k))
        marks = [i in idx for i in range(len(pieces))]
    else:
        k = max(1, min(n_mark_neg, len(pieces)))
        start = rng.randrange(0, max(1, len(pieces) - k + 1))
        marks = [start <= i < start + k for i in range(len(pieces))]
    return {
        "text": "".join(pieces).strip(),
        "text_fuzz": marked_text(pieces, marks).strip(),
        "n_marked": int(sum(marks)),
        "n_tok": len(pieces),
    }


# ---------------------------------------------------------------------------------------------
# corpus window recovery
# ---------------------------------------------------------------------------------------------


class _Corpus:
    """`tokens.i32` + `docs.jsonl`, with the window geometry asserted on every recovery."""

    def __init__(self, base: str, root: str):
        self.toks, docs = C.load_corpus(base, root)
        self.docs = {int(r["doc"]): r for r in docs}

    def ids(self, doc: int, start: int, ln: int):
        r = self.docs[int(doc)]
        wins = C.windows_of(int(r["len"]))
        assert (int(start), int(ln)) in wins, (
            f"doc {doc}: window (start={start}, len={ln}) is not one of common.windows_of's "
            f"{len(wins)} windows for a {r['len']}-token document -- the examples and this build "
            f"disagree on the scan geometry ({C.SCAN_BLOCK}/{C.SCAN_STRIDE})"
        )
        off = int(r["offset"]) + int(start)
        return np.asarray(self.toks[off : off + int(ln)])

    def size_tag(self, doc: int) -> int:
        return int(self.docs[int(doc)]["size_tag"])


class _RandomPool:
    """The shared negative pool written by `autointerp/sae_self.py:run_random_pool`.

    2,048 corpus windows encoded for every tested feature with PER-TOKEN pre-gate activations.
    It replaces `scan`'s `_random256`, which carries 256 windows and a per-feature maximum only and
    cannot supply 20 zero-activation negatives for the densest tested features.
    """

    def __init__(self, path: str):
        self.path = path
        assert os.path.exists(f"{path}/pool.json"), (
            f"no random pool at {path}: run `--stage random_pool` on this (base, set) first"
        )
        with open(f"{path}/pool.json") as fh:
            self.info = json.load(fh)
        self.windows = C.read_jsonl(f"{path}/windows.jsonl")
        self.n_win = int(self.info["n_windows"])
        self.col = {int(f): i for i, f in enumerate(self.info["features"])}
        n_feat = len(self.info["features"])
        self.max_act = C.read_array(f"{path}/max_act.f16", "float16", (n_feat, self.n_win))
        self.off = C.read_array(f"{path}/tok_off.i64", "int64", (n_feat * self.n_win + 1,))
        nnz = int(self.off[-1])
        self.pos = C.read_array(f"{path}/tok_pos.i16", "int16", (nnz,))
        self.val = C.read_array(f"{path}/tok_val.f16", "float16", (nnz,))
        self.gate = float(self.info["gate"])

    def maxima(self, feature: int):
        return self.max_act[self.col[feature]].astype(np.float32)

    def acts(self, feature: int, w: int):
        """The window's per-token pre-gate activations of `feature`, densified to its length.

        NOT used by the current negative rule: amendment A5 marks every negative, near-miss
        included, at RANDOM for fuzzing, so only `maxima` is needed to select them. It is the
        reader for the per-token half of the product, which `run_random_pool` writes because a
        near-miss negative rendered with its OWN activations is the obvious next variant and
        re-running the GPU pass to get it would cost another $0.14 and another hour of wall.
        """
        row = self.col[feature] * self.n_win + w
        lo, hi = int(self.off[row]), int(self.off[row + 1])
        a = np.zeros(int(self.windows[w]["len"]), dtype=np.float32)
        if hi > lo:
            a[self.pos[lo:hi].astype(np.int64)] = self.val[lo:hi].astype(np.float32)
        return a


def _overlaps(a: dict, b: dict) -> bool:
    """Two corpus windows of the same document whose token ranges intersect.

    At stride 16 in 64-token windows the top-k of a feature is routinely four overlapping cuts of
    the same passage; showing them as four independent examples would be the dedup bug Celeste's
    pipeline handles with an 8-gram filter (repo-maemm-master/eval/autointerp_detection.py:102-105).
    This is the same job done exactly rather than approximately, because we have the offsets.
    """
    if a["doc"] != b["doc"]:
        return False
    return a["start"] < b["start"] + b["len"] and b["start"] < a["start"] + a["len"]


def _trigrams(text: str) -> set[str]:
    w = text.split()
    return {" ".join(w[i : i + 3]) for i in range(max(0, len(w) - 2))} or {text}


def diversify(pool: list[dict], n: int, jaccard_max: float = 0.5) -> list[dict]:
    """`n` rollouts by near-duplicate removal then QUANTILE sampling of the peak activation.

    Two steps, in this order:
      1. greedily drop any rollout whose word-trigram Jaccard against an already-kept one exceeds
         `jaccard_max` -- exact-text dedup does not catch "16 variants of one template";
      2. from what survives, take `n` at evenly spaced quantiles of `max_act` (the pool arrives
         sorted by peak, descending), so the set spans the activation range instead of piling at
         the top.
    Falls back to the surviving order when fewer than `n` remain, and the shortfall is recorded by
    the caller like every other.
    """
    kept: list[dict] = []
    grams: list[set[str]] = []
    for e in pool:
        g = _trigrams(e["text"])
        if any(len(g & h) / max(1, len(g | h)) > jaccard_max for h in grams):
            continue
        kept.append(e)
        grams.append(g)
    if len(kept) <= n:
        return kept
    idx = [round(i * (len(kept) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    seen: set[int] = set()
    out = []
    for i in idx:
        while i in seen and i + 1 < len(kept):
            i += 1
        seen.add(i)
        out.append(kept[i])
    return out


def dedup(rows: list[dict]) -> list[dict]:
    """`rows` in order, dropping any window that overlaps one already kept."""
    kept: list[dict] = []
    for r in rows:
        if not any(_overlaps(r, k) for k in kept):
            kept.append(r)
    return kept


# ---------------------------------------------------------------------------------------------
# the feature draw
# ---------------------------------------------------------------------------------------------


def band_of(max_act: float, peak: float) -> str:
    """The stored activation band of a window, by `precompute/scan.py:257`'s own formula.

    `q = ceil(amax / peak * 4) - 1`, clamped to 0..3: EQUAL-WIDTH bins of (0, max_act], not
    quantiles. It is reimplemented here so that a row from `examples_docmax/` -- which scan.py
    never binned -- carries a band label meaning exactly what a `q0..q3` row's does, and the
    band-stratified draw can run over both pools at once.
    """
    if peak <= 0:
        return BANDS[0]
    q = int(min(3, max(0, math.ceil(max_act / peak * 4) - 1)))
    return BANDS[q]


def draw_features(sae_rows: list[dict], n_feat: int, seed: int) -> list[dict]:
    """`n_feat` rows, n_feat/4 from each density quartile, by a fixed seed. 0 means every row.

    The quartile (`stratum`) is the pre-registered stratification axis of the held-out set
    (precompute/targets.py:229-252), so the pilot is stratified the same way the full set is and a
    per-quartile table is available at both scales.
    """
    if not n_feat or n_feat >= len(sae_rows):
        return list(sae_rows)
    strata = sorted({int(r["stratum"]) for r in sae_rows})
    per = n_feat // len(strata)
    assert per * len(strata) == n_feat, (
        f"--n-feat {n_feat} is not divisible by the {len(strata)} density quartiles"
    )
    rng = random.Random(seed)
    out: list[dict] = []
    for s in strata:
        pool = sorted((r for r in sae_rows if int(r["stratum"]) == s), key=lambda r: r["row"])
        assert len(pool) >= per, f"stratum {s} has {len(pool)} features, need {per}"
        out += rng.sample(pool, per)
    return sorted(out, key=lambda r: r["row"])


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------

# PREPARED, NOT RUN (Tomas decides; 2026-09-16). `--centre32` re-cuts every CORPUS example to a
# 32-token window CENTRED on its peak token before rendering, which is Delphi's own
# `example_ctx_len 32` and `center_examples True` and is also what our earlier fork did
# (repo-maemm/eval/autointerp_detection.py:715, `--win-ctx 32`). Nothing new has to be computed:
# the per-token activations are already stored for the full 64-token window, so this is a
# rendering change plus one fresh explainer call per (feature, arm). PROJECTED at n = 512 over the
# C-arms: ~3,072 explainer calls at the measured $0.0085 = ~$26, plus scoring if it is scored as
# its own arm. It is a separate ARM, never a silent change to C16: the two must be comparable.
# name -> (corpus source, n corpus, rollout source, n rollouts). "c16" is the full-corpus top-k,
# "c4" the 4M-prefix top-k. The order of the tuple is the order examples are concatenated in
# before the shuffle.
ARM_SPECS = {
    "C16": ("c16", 16, None, 0),
    "C4": ("c4", 16, None, 0),
    "M": (None, 0, "m", 16),
    "C4M": ("c4", 8, "m", 8),
    # AMENDMENT 2026-09-16: the additive ablation is now MATCHED-N at 16M -- C16M16 (N=32) against
    # C16-N32 (N=32, corpus only). The old C4M16 mixed an additive change with a corpus-size change
    # and with C4's shortfall, so nothing it showed could be attributed.
    "C16M16": ("c16", 16, "m", 16),
    "C32": ("c16", 32, None, 0),
    # The document-diverse corpus arm (Tomas, 2026-09-21, eval 2). `examples_docmax` keeps ONE
    # window per document and ranks documents, so its top-16 cannot be sixteen windows of one
    # document the way `c16`'s window ranking can. Two things follow and both are recorded on
    # every build rather than argued here: it is the only corpus arm available on the 2M SAE,
    # which has no `scan` examples/ (a ~$9 pass nobody has paid for), and it SHOWS windows drawn
    # from the same pool the test positives come from -- so each shown document leaves the
    # candidate pool for EVERY arm (A4, `shown_docs`), not just for this one.
    "DOCMAX": ("docmax", 16, None, 0),
    # The method review's test: the M arm shows 16 variants of ONE template (median pairwise
    # word-trigram Jaccard within the shown set 0.046 against the corpus set's 0.001, 40x), so its
    # loss may be an artefact of an undiversified top-16 rather than of the source. `M-div` keeps
    # the same 64 rollouts and the same N, and changes only the CHOICE: near-duplicates dropped by
    # trigram Jaccard, then quantile sampling across the peak-activation range (Delphi's
    # `train_type: "quantiles"` analogue) instead of the top 16.
    "M-div": (None, 0, "mdiv", 16),
    # The NLA arm (Tomas, 2026-09-21). Same shape as M -- rollouts rendered with their per-token
    # activation marks from `sae_self` -- but the rollout source is the NLA VERBALIZER, so it is a
    # different arm and never a relabelled M. n = 4 because that is `nla.n`: the verbalizer answers
    # at 200 tokens and four samples per target is what the budget buys, so this arm is NOT
    # matched-N against C4/C16 (16) and the build README says so on every run.
    NLA_ARM: (None, 0, "m", NLA_N),
    # Descriptive pilot points only (amendment A9: N = 16 is fixed a priori, not selected).
    "C16-N8": ("c16", 8, None, 0),
    "M-N8": (None, 0, "m", 8),
    "M-N32": (None, 0, "m", 32),
}
# The arms the FULL run scores. The rest are pilot-only descriptive points. `F` and `C16-draw2`
# are scorer-only pseudo-arms that `run.py` adds: F reuses another feature's description, and
# C16-draw2 reuses C16's own description on the second, disjoint test draw (amendments A6, A7).
FULL_ARMS = ("C16", "C4", "M", "C4M", "C32", "C16M16")


def _covariate(row: dict, *names: str):
    """The first of `names` present on a held-out row, else None -- for DESCRIPTIVE fields only.

    The two SAE families spell their covariates differently (`targets.py`'s `density` /
    `fires_gated` against `draw_sae2m.py`'s `gated_fires`, and the 2M draw carries no density at
    all). stats.py reports by these and nothing selects on them, so an absent one is a missing
    covariate, not a reason to refuse to build. Anything load-bearing -- `row`, `id`, `family`,
    `stratum` -- is read directly and still raises when it is not there.
    """
    for n in names:
        if row.get(n) is not None:
            return row[n]
    return None


def check_corpus_source(arm_names, use_examples: bool, ex_dir: str, sae_key: str, prefix_m: int) -> str:
    """The label for where test POSITIVES come from. Asserts no arm needs a pool that is absent.

    `scan`'s 16M `examples/<feature>.jsonl` carries both the top-k the C16 arms show AND the
    q-band rows the positive draw uses. It does not exist for every SAE -- the 2M one would cost
    a ~$9 scan -- and the two halves fail differently when it is missing. The positive pool has an
    honest substitute (`examples_4m`, band-labelled here, which is what this returns). A C16 arm
    does NOT: filling it from the 4M prefix would put a quarter of the corpus behind the C16
    label, so it is refused by name instead.
    """
    need_c16 = sorted({a for a in arm_names if ARM_SPECS[a][0] == "c16"})
    assert use_examples or not need_c16, (
        f"arms {need_c16} show corpus windows from `scan`'s 16M examples, and there is no "
        f"{ex_dir}/tested.json: run `--product scan` for sae {sae_key!r} first, or drop those "
        f"arms (C4 reads `examples_4m`, which is present). Nothing is guessed from the "
        f"{prefix_m}M prefix in their place -- a C16 arm filled from a quarter of the corpus "
        f"would carry the C16 label and not be C16."
    )
    if use_examples:
        return "examples/ (scan, 16M)"
    return f"examples_4m (the {prefix_m}M prefix; scan's examples/ is absent)"


def candidate_rows(ex_rows, ex4_rows, doc_rows, peak: float, use_examples: bool) -> list[dict]:
    """The test set's positive / near-miss candidates, band-labelled and deduplicated by window.

    With `scan`'s examples/ present this is its stored `q0..q3` rows plus the document-diverse
    pool; its `top` rows are NOT candidates, because they are what the arms SHOW. Without it there
    are no stored bands at all, so `examples_4m`'s rows become the pool -- band-labelled by
    `band_of`, exactly as the document-diverse rows already are, so a `q2` row means the same
    thing whichever product it came from. `examples_4m` writes only `kind: "top"` rows
    (`sae_self.run_examples_4m`: the top 128 by peak), so all of them are eligible; what keeps an
    arm's own example out of its own test set is the document-level disjointness rule (A4), not
    the row's kind.

    First writer of a window wins, which is why the order is bands, then 4M, then docmax: a row
    that scan already binned keeps scan's own label.
    """
    out = [dict(e) for e in ex_rows if e["kind"] in BANDS]
    seen = {int(e["window"]) for e in out}
    extra = ([] if use_examples else list(ex4_rows)) + list(doc_rows)
    for e in extra:
        if int(e["window"]) in seen:
            continue
        seen.add(int(e["window"]))
        out.append({**e, "kind": band_of(float(e["max_act"]), peak)})
    return out


def check_arm_maemm(arm_names, maemm: str, maemm_type: str) -> bool:
    """True iff `--maemm` is the NLA verbalizer. Asserts that the arms asked for match what it is.

    An `nla` entry's rollouts may only build the "NLA" arm. `M`, `C4M`, `C16M16` and the
    descriptive M points are named in the paper as the MAEMM's arms, and filling them from a
    verbalizer would produce a correctly-shaped, WRONGLY-LABELLED number that nothing downstream
    could detect -- the rows look identical, only their provenance differs. The converse is the
    same mistake mirrored, so asking for "NLA" with a MAEMM is refused too.
    """
    is_nla = maemm_type == "nla"
    if is_nla:
        bad = [a for a in arm_names if a in ROLLOUT_ARMS_NOT_NLA]
        assert not bad, (
            f"--maemm {maemm!r} is a `type: nla` entry (the activation verbalizer), so its "
            f"rollouts may only build the {NLA_ARM!r} arm; {bad} are the MAEMM rollout arms and "
            f"would be mislabelled. Drop them, or point --maemm at a MAEMM."
        )
    else:
        assert NLA_ARM not in arm_names, (
            f"arm {NLA_ARM!r} asks for the activation verbalizer's rollouts but --maemm {maemm!r} "
            f"is type {maemm_type!r}; point --maemm at the `type: nla` entry"
        )
    return is_nla


def nla_description(raw: str) -> dict:
    """Arm B's description from ONE NLA rollout's raw text: the <explanation> body, or all of it.

    The tags are the verbalizer's output FORMAT, not content, so they are stripped with the same
    regex `rollouts_nla` records its per-row `explanation` with -- one extractor, not two. A
    rollout that never closed its tag contributes its whole text and says so (`tag_found: false`)
    rather than being dropped: a silently missing description would shrink arm B's feature set
    relative to every other arm, and the comparison is paired.
    """
    # explanation_body, not extract_explanation: the old fallback handed the judge the WHOLE raw
    # decode -- opening tag and chat preamble included -- whenever the answer ran into max_new and
    # never closed its tag (Juan, 2026-09-21). An unclosed answer now contributes everything after
    # its opening tag; only a decode with no tag at all is passed whole, and says so.
    body, status = explanation_body(raw)
    return {
        "tag_found": status != "none",
        "tag_status": status,
        "n_chars": len(raw),
        "description": body,
    }


def _epo_arm(path: str, feature: int, tok, peak: float, gate: float, mark: str = "gate",
             rel_fallback: bool = False):
    """The E arm's documented hook: per-feature EPO / GCG strings as a fourth example source.

    NOT RUN in the pilot (design §8: at the measured ~$1.09 per 27B epo target, 512 features is
    ~$560 and needs a decision). The hook is real rather than a stub: point `--epo-strings` at a
    jsonl whose rows are

        {"feature": <int>, "strings": [{"ids": [...], "acts": [...]}, ...]}

    where `acts` is the per-token PRE-GATE activation of `feature` on `ids`, measured on the clean
    base at the read layer exactly as `sae_self` measures a rollout. `paper-evals/gcg/gcg.py`
    stores `sae_peak_act` / `sae_peak_pos` on its pop finals but NOT the per-token vector
    (gcg.py:1004-1010), so producing this file needs one extra scoring pass over the finals; it is
    that pass, not this renderer, that the E arm is waiting on.
    """
    rows = {int(r["feature"]): r for r in C.read_jsonl(path)}
    assert feature in rows, f"{path} has no row for feature {feature}"
    out = []
    for i, s in enumerate(rows[feature]["strings"]):
        assert "ids" in s and "acts" in s, (
            f"{path}, feature {feature}, string {i}: the E arm needs per-token `acts` alongside "
            f"`ids` (see autointerp/build.py:_epo_arm); got keys {sorted(s)}"
        )
        assert len(s["ids"]) == len(s["acts"]), (
            f"{path}, feature {feature}, string {i}: {len(s['ids'])} ids but {len(s['acts'])} acts"
        )
        e = render_example(tok, s["ids"], s["acts"], peak, gate, mark,
                           rel_fallback=rel_fallback)
        out.append({**e, "src": "epo", "k": i})
    return out


# ---------------------------------------------------------------------------------------------


def draw_test(
    *, feat, ex_rows, tops, pool, corpus, tok, gate, rng_test, rng_mark, cfgv,
    shown_windows_ids, shown_docs, used_docs, flags, tag, fuzz_marks="contiguous",
):
    """One test draw for one feature: n_pos positives + n_neg negatives, rendered both ways.

    Called TWICE per feature. The first draw is the evaluation set every arm is scored on; the
    second is disjoint from it and from every shown example, and `run.py` scores C16 on it as well.
    The per-feature difference between the two C16 numbers is the NULL of every reported
    difference (design amendment A7, which replaced the temperature-0 repeat: a repeat measures
    judge jitter, while a second draw measures the test-set sampling that every contrast is
    actually exposed to).

    The rules, all amendments of 2026-09-16:
      A1  a positive is a window whose peak pre-gate activation EXCEEDS THE GATE, and the band
          draw runs over gate-passing rows only. MEASURED on feature 845 before the amendment: 17
          of 20 band-drawn positives sat below the gate on text unrelated to the feature ("Termites
          can cause costly damage to your floors"), the judge called 3 of them positive against a
          TNR of 20/20, and every arm landed near 0.6 balanced accuracy whatever its description
          said -- including a C16 description that was exactly right.
      A4  DOCUMENT-level disjointness: no test item shares a `doc` with any shown example or with
          any other test item, in either draw. Window overlap alone is not enough at stride 16 --
          two windows of one document 200 tokens apart do not overlap and are the same passage.
      A5  negatives are 10 zero-activation windows from the 2048-window `random_pool` plus 10
          near-miss windows (0 < peak <= gate) from the below-gate band rows, falling back to
          zero-activation randoms when the band rows run out. Delphi's random rule stays the
          reference; the near-miss half is what keeps the negative side off its ceiling.

    `used_docs` is MUTATED, which is how the second draw stays disjoint from the first.
    Returns (items, info); `info` carries every shortfall, all of which are flagged, none filled.
    """
    n_pos, n_neg, n_near, nearmiss_source, gate_positives, allow_top_fallback = cfgv
    per_band = n_pos // len(BANDS)
    gate_ok = (lambda e: float(e["max_act"]) > gate) if gate_positives else (lambda e: True)

    def free(e):
        return int(e["doc"]) not in used_docs

    # ---- positives: bands high to low, a short band carrying its deficit to the next band DOWN
    pos_rows: list[dict] = []
    short_bands: list[tuple[str, int]] = []
    deficit = 0
    for band in reversed(BANDS):
        cand = [e for e in ex_rows if e["kind"] == band and gate_ok(e) and free(e)]
        rng_test.shuffle(cand)
        take: list[dict] = []
        for e in cand:
            if len(take) >= per_band + deficit:
                break
            if not free(e):
                continue
            take.append(e)
            used_docs.add(int(e["doc"]))
        deficit = per_band + deficit - len(take)
        if len(take) < per_band:
            short_bands.append((band, len(take)))
        for e in take:
            e = dict(e)
            e["band"] = band
            pos_rows.append(e)
    # Fallback tier: top-ranked windows BEYOND the ones any arm shows. Gate-passing by
    # construction, and the only place left once the bands are exhausted. Counted and flagged.
    n_top_fallback = 0
    if len(pos_rows) < n_pos and allow_top_fallback:
        extra = [e for e in tops
                 if int(e["window"]) not in shown_windows_ids and gate_ok(e) and free(e)]
        rng_test.shuffle(extra)
        for e in extra:
            if len(pos_rows) >= n_pos:
                break
            if not free(e):
                continue
            used_docs.add(int(e["doc"]))
            e = dict(e)
            e["band"] = "top"
            pos_rows.append(e)
            n_top_fallback += 1
    if short_bands or len(pos_rows) < n_pos:
        flags.append(
            f"feature {feat} [{tag}]: positives {len(pos_rows)}/{n_pos} -- short bands "
            f"{short_bands}, {n_top_fallback} filled from top-beyond-shown"
        )

    # ---- negatives
    mx = pool.maxima(feat)
    pool_free = np.asarray([int(w["doc"]) not in used_docs for w in pool.windows])
    zero_ix = [int(i) for i in np.nonzero((mx == 0.0) & pool_free)[0]]
    near_ix = [int(i) for i in np.nonzero((mx > 0.0) & (mx <= gate) & pool_free)[0]]
    want_near = min(n_near, n_neg)
    near_pick: list[tuple[str, object]] = []
    if nearmiss_source == "qband":
        qnear = [e for e in ex_rows
                 if e["kind"] in BANDS and float(e["max_act"]) <= gate and free(e)]
        rng_test.shuffle(qnear)
        for e in qnear:
            if len(near_pick) >= want_near:
                break
            if not free(e):
                continue
            used_docs.add(int(e["doc"]))
            near_pick.append(("qband", e))
    else:
        rng_test.shuffle(near_ix)
        for i in near_ix:
            if len(near_pick) >= want_near:
                break
            doc = int(pool.windows[i]["doc"])
            if doc in used_docs:
                continue
            used_docs.add(doc)
            near_pick.append(("random", i))
    n_near_short = want_near - len(near_pick)
    if n_near_short:
        flags.append(
            f"feature {feat} [{tag}]: {len(near_pick)}/{want_near} near-miss negatives from "
            f"{nearmiss_source}; the shortfall falls back to zero-activation randoms (A5)"
        )
    want_zero = n_neg - len(near_pick)
    rng_test.shuffle(zero_ix)
    zero_pick: list[int] = []
    for i in zero_ix:
        if len(zero_pick) >= want_zero:
            break
        doc = int(pool.windows[i]["doc"])
        if doc in used_docs:
            continue
        used_docs.add(doc)
        zero_pick.append(i)
    if len(zero_pick) < want_zero:
        flags.append(
            f"feature {feat} [{tag}]: {len(zero_pick)}/{want_zero} zero-activation negatives "
            f"(pool has {len(zero_ix)} doc-free zero windows of {pool.n_win})"
        )

    # ---- render
    items: list[dict] = []
    for e in pos_rows:
        ids = corpus.ids(e["doc"], e["start"], e["len"])
        assert len(ids) == len(e["acts"]), (
            f"feature {feat}, test positive (doc {e['doc']}, start {e['start']}): recovered "
            f"{len(ids)} tokens vs {len(e['acts'])} stored acts"
        )
        t = render_test(tok, ids, e["acts"], gate, rng_mark, 0, fuzz_marks)
        items.append({**t, "label": 1, "src": "corpus", "band": e["band"],
                      "window": e["window"], "doc": e["doc"], "start": e["start"],
                      "max_act": e["max_act"]})
    # The negatives' random mark length is the positives' MEAN mark count, rounded down, so it is
    # computed from the rendered positives rather than guessed.
    n_mark_neg = max(1, int(np.mean([it["n_marked"] for it in items])) if items else 1)
    # EVERY negative, near-miss included, is marked at RANDOM for fuzzing: fuzzing's ground truth
    # is "this marking is wrong", and a near-miss marked at its own sub-gate peak would be a
    # marking that is arguably right. Delphi's intruder construction is the one copied here.
    for src, w in near_pick:
        if src == "qband":
            ids = corpus.ids(w["doc"], w["start"], w["len"])
            meta_w = {"window": w["window"], "doc": w["doc"], "start": w["start"],
                      "max_act": w["max_act"]}
        else:
            pw = pool.windows[w]
            ids = corpus.ids(pw["doc"], pw["start"], pw["len"])
            meta_w = {"window": pw["window"], "doc": pw["doc"], "start": pw["start"],
                      "max_act": round(float(mx[w]), 4)}
        t = render_test(tok, ids, None, gate, rng_mark, n_mark_neg, fuzz_marks)
        items.append({**t, "label": 0, "src": f"nearmiss-{src}", "band": "-", **meta_w})
    for w in zero_pick:
        pw = pool.windows[w]
        ids = corpus.ids(pw["doc"], pw["start"], pw["len"])
        t = render_test(tok, ids, None, gate, rng_mark, n_mark_neg, fuzz_marks)
        items.append({**t, "label": 0, "src": "random", "band": "-",
                      "window": pw["window"], "doc": pw["doc"], "start": pw["start"],
                      "max_act": 0.0})

    item_docs = {int(it["doc"]) for it in items}
    assert not (item_docs & shown_docs), (
        f"feature {feat} [{tag}]: test items share documents "
        f"{sorted(item_docs & shown_docs)[:5]} with windows shown to an explainer -- "
        f"document-level disjointness (A4) is broken"
    )
    assert len(item_docs) == len(items), (
        f"feature {feat} [{tag}]: {len(items)} test items over only {len(item_docs)} documents"
    )
    # ONE shuffle, seeded by the feature, so every arm scores the same items in the same order and
    # in the same batches (repo-maemm/eval/autointerp_detection.py:2958-2969).
    rng_test.shuffle(items)
    info = {
        "n_pos": sum(1 for it in items if it["label"] == 1),
        "n_neg": sum(1 for it in items if it["label"] == 0),
        "n_pos_bands": sum(1 for it in items if it["label"] == 1 and it["band"] != "top"),
        "n_top_fallback": n_top_fallback,
        "n_near": len(near_pick),
        "n_near_short": n_near_short,
        "n_zero": len(zero_pick),
        "pool_zero": len(zero_ix),
        "pool_nearmiss": len(near_ix),
        "n_mark_neg": n_mark_neg,
        "short_bands": short_bands,
    }
    return items, info


def run(cfg, args):
    from transformers import AutoTokenizer

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base and maemm, "stage build needs --base and --maemm (the M arms' rollouts)"
    ac = cfg["autointerp"]
    n_ex = int(args.get("n_examples") or ac["n_examples"])
    n_feat = int(args.get("n_feat") or ac["pilot_features"])
    feat_seed = int(args.get("feat_seed") or ac["feat_seed"])
    shuffle_seed = int(ac["shuffle_seed"])
    prefix_m = int(ac["corpus_prefix_m"])
    n_pos, n_neg = int(ac["n_pos"]), int(ac["n_neg"])
    n_neg_nearmiss = int(ac["n_neg_nearmiss"])
    nearmiss_source = str(ac["nearmiss_source"])
    assert nearmiss_source in ("qband", "random"), (
        f"autointerp.nearmiss_source must be 'qband' or 'random', got {nearmiss_source!r}"
    )
    gate_positives = bool(ac["gate_consistent_positives"])
    centre32 = bool(args.get("centre32"))
    mark = str(args.get("mark") or "gate")
    # WHICH ARMS GET THE RELATIVE FALLBACK. `gate` (default) reproduces every run made before
    # 2026-09-21 byte for byte; `relative` turns it on for the GENERATED-TEXT arms -- the MAEMM
    # rollouts, the NLA rollouts and the EPO strings -- and never for the corpus arms, whose peak
    # IS the corpus peak and whose unmarked blocks are a fact about the feature rather than a
    # rendering failure. A flag and not a new default, because the published 512-feature run and
    # the 32-feature gate-marked pilot must both stay reproducible from their command lines.
    rollout_mark = str(args.get("rollout_mark") or "gate")
    assert rollout_mark in ("gate", "relative"), (
        f"--rollout-mark must be gate or relative, got {rollout_mark!r}"
    )
    rel_fallback = rollout_mark == "relative"
    assert mark in ("gate", "delphi"), f"--mark must be 'gate' or 'delphi', got {mark!r}"
    fuzz_marks = str(args.get("fuzz_marks") or "contiguous")
    assert fuzz_marks in ("contiguous", "scattered", "delphi"), (
        f"--fuzz-marks must be contiguous, scattered or delphi, got {fuzz_marks!r}"
    )
    allow_top_fallback = bool(ac["allow_top_fallback"])
    engine = args.get("engine") or "vllm"
    arm_names = [a for a in (args.get("arms") or "").split(",") if a] or list(ARM_SPECS)
    for a in arm_names:
        assert a in ARM_SPECS, f"unknown arm {a!r}, want some of {list(ARM_SPECS)}"
    assert n_ex == 16, (
        f"autointerp.n_examples is {n_ex}: ARM_SPECS pins the per-arm counts explicitly, so "
        f"changing N means editing them, not this number"
    )

    # `--sae` when the base carries more than one (qwen36-27b does). Same rule as
    # score._sae_for and sae_self._sae_rows: build reads THEIR outputs, so it must resolve the
    # same key they did or it would render examples for one dictionary from another's scan.
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    ex_dir = C.sae_examples_dir(sae_key, set_name, root, corpus_name=args.get("corpus_name") or "")
    # `scan`'s 16M product: `<feature>.jsonl` with the top-k AND the q-band rows, plus the
    # per-token activations of each. It does not exist for every SAE -- the 2M one would cost a
    # ~$9 scan to make -- and when it is absent BOTH things it feeds have to come from somewhere
    # else: the C16 arms (which then simply cannot be built) and the test set's positive pool
    # (which falls back to the 4M-prefix `examples_4m` below, band-labelled the same way the
    # document-diverse pool already is).
    use_examples = os.path.exists(f"{ex_dir}/tested.json")
    # Amendment A3: C4 reads the 4M prefix's OWN top-128, not the 4M-prefix members of the 16M
    # ranking (median 14 candidates after dedup, fewer than 16 on 38 of 64 pilot features).
    ex4_dir = f"{C.sae_dir(sae_key, root)}/examples_4m/{set_name}"
    assert os.path.exists(f"{ex4_dir}/tested.json"), (
        f"no {prefix_m}M example scan at {ex4_dir}: run `--stage examples_4m` on this (base, set) "
        f"first -- the C4 arm is its top-128, not a filter of the 16M one (amendment A3)"
    )
    # The test set's positive pool. MEASURED 2026-09-16: with only `examples/`'s q-bands and its
    # top-128 to draw from, gate-consistent positives (A1) under document-level disjointness (A4)
    # left draw 1 short on 29 of 64 pilot features and draw 2 EMPTY on 21. `examples_docmax/`
    # ranks DOCUMENTS instead of windows -- one window from each of the top 256 documents -- which
    # is the pool A4 actually needs.
    exdoc_dir = f"{C.sae_dir(sae_key, root)}/examples_docmax/{set_name}"
    use_docmax = os.path.exists(f"{exdoc_dir}/tested.json")
    assert use_docmax or args.get("allow_short"), (
        f"no document-diverse example scan at {exdoc_dir}: run `--stage examples_docmax` on this "
        f"(base, set) first, or pass --allow-short to build a knowingly short test set"
    )
    hdir = C.heldout_dir(base, set_name, root)
    sdir = C.scores_dir(maemm, set_name, root, engine)
    self_dir = f"{sdir}/sae_self{args.get('out_suffix') or ''}"

    rows_meta = C.read_jsonl(f"{hdir}/ids.jsonl")
    # On the ROW's own sae_key, not on the family label -- see common.sae_rows_of. With two
    # dictionaries under one `family: sae` label, the family-only filter renders the 131k arm from
    # the 2M scan and nothing raises.
    sae_rows = C.sae_rows_of(
        rows_meta, sae_key, FAMILIES, side="enc",
        declared=C.declared_sae_key(cfg, hdir, root), where=hdir,
    )
    assert sae_rows, (
        f"{hdir}/ids.jsonl has no encoder rows of dictionary {sae_key!r} in the SAE families "
        f"{FAMILIES}; it carries families {sorted({r['family'] for r in rows_meta})} and "
        f"dictionaries "
        f"{sorted({r.get('sae_key', '(unkeyed)') for r in rows_meta if r['family'] in FAMILIES})}"
    )
    picked = draw_features(sae_rows, n_feat, feat_seed)
    if args.get("rows"):
        # --rows OVERRIDES the stratified draw rather than intersecting it: a shakeout asks for
        # specific rows and must get exactly those, not "whichever of them the draw happened to
        # pick" (MEASURED 2026-09-16: --rows 1024-1025 quietly built one feature).
        want = set(C.parse_rows(args["rows"], len(rows_meta)))
        picked = [r for r in sae_rows if r["row"] in want]
        assert picked, (
            f"--rows {args['rows']!r} selected none of the {len(sae_rows)} {'/'.join(FAMILIES)} rows"
        )
    positive_source = check_corpus_source(arm_names, use_examples, ex_dir, sae_key, prefix_m)
    if not use_examples:
        print(
            f"[build] no {ex_dir}/tested.json: the positive pool and the band labels come from "
            f"`examples_4m` ({prefix_m}M prefix) instead, and no C16 arm can be built",
            flush=True,
        )
    is_nla = check_arm_maemm(arm_names, maemm, cfg["maemms"][maemm]["type"])
    # Arm B ("the NLA text IS the description") needs the rollout's OWN text, not the rendered,
    # activation-marked example the explainer sees, so it comes from the rollouts file rather than
    # from sae_self's re-encoded ids. NOTE this is the DEFAULT-amp rollouts directory: an `--amp`
    # variant lands in maemms/<base>/<nla>/variants/ and neither sae_self nor this stage reads
    # from there, so an amp sweep needs its own (rollouts -> score -> sae_self -> build) chain.
    nla_text: dict[tuple[int, int], str] = {}
    if is_nla:
        rpath = C.rollouts_path(maemm, set_name, root, engine)
        assert os.path.exists(rpath), (
            f"no rollouts at {rpath}: the NLA arms need the verbalizer's own texts, and this is "
            f"the same file `sae_self` measured its activations on"
        )
        nla_text = {(int(x["row"]), int(x["k"])): x["text"] for x in C.read_jsonl(rpath)}
    nla_desc_rows: list[dict] = []
    print(f"[build] {len(picked)} features, arms {arm_names}", flush=True)

    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][base]["hf"]))
    corpus = _Corpus(base, root)
    d_sae_peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (-1,))
    pool = _RandomPool(f"{C.sae_dir(sae_key, root)}/random_pool/{set_name}")

    self_meta = json.load(open(f"{self_dir}/sae_self.json"))
    gate = float(self_meta["gate"])
    # `build` used to read the product and ignore its `checks`, so a sae_self that failed its own
    # argmax / CSR checks would still have been consumed. The checks are now a precondition.
    _ck = self_meta.get("checks") or {}
    assert _ck.get("argmax_ok", True) and not _ck.get("csr_value_mismatches", 0) \
        and not _ck.get("csr_membership_mismatches", 0), (
        f"{self_dir}/sae_self.json reports failed checks {_ck}: its per-token activations cannot "
        f"be used to build the M arms"
    )
    self_rows = list(self_meta["rows"])
    n_roll = int(self_meta["n"])
    # `width` is written by sae_self since the scoring window became per-run (the NLA arm scores
    # at 256, not the protocol's 95); a sae_self.json from before that carries none and is the
    # protocol width.
    shape = (len(self_rows), n_roll, int(self_meta.get("width", C.SCORE_WIDTH)))
    self_act = C.read_array(f"{self_dir}/sae_self.f16", "float16", shape).astype(np.float32)
    self_ids = C.read_array(f"{self_dir}/sae_self_ids.i32", "int32", shape)
    self_ix = {r: i for i, r in enumerate(self_rows)}
    fire_of = {int(p["row"]): float(p["fire_fraction"]) for p in self_meta["per_target"]}

    missing = [r["row"] for r in picked if r["row"] not in self_ix]
    assert not missing, (
        f"{self_dir} has no rows {missing[:8]}: run `--stage sae_self` over the features this "
        f"build draws (it has rows {self_rows[0]}..{self_rows[-1]})"
    )

    flags: list[str] = []
    feat_table: list[dict] = []
    mark_frac: list[float] = []
    join_bad = 0
    join_total = 0
    n_exceed_peak = 0
    # PER ARM, how its blocks were marked. `unmarked` is the count this whole fallback exists to
    # drive to zero: a block the explainer saw with no `<<>>` and no `Activations:` line at all.
    # Counted for EVERY build, gate-marked or relative, so the two are comparable side by side.
    mark_counts: dict[str, dict[str, int]] = {}
    build_name = args.get("build_dir") or time.strftime("%Y-%m-%d") + "_build"
    out_dir = f"{C.base_dir(base, root)}/autointerp/{set_name}/{build_name}"

    with C.outdir(
        out_dir,
        args,
        inputs={
            "examples": ex_dir,
            "heldout": hdir,
            "sae_self": self_dir,
            "maemm": maemm,
            "engine": engine,
            "features": len(picked),
            "arms": ",".join(arm_names),
            "gate": gate,
        },
    ) as od:
        for r in picked:
            feat = int(r["id"])
            peak = float(d_sae_peak[feat])
            # One RNG per PURPOSE, each seeded from (shuffle_seed, feature, purpose), so a run
            # with a different --arms subset draws the SAME test set and the same item order:
            # a single shared stream would make the test set depend on how many arms consumed
            # it first.
            rng_arm = random.Random(shuffle_seed + feat)
            rng_test = random.Random(shuffle_seed + feat + 1_000_003)
            rng_mark = random.Random(shuffle_seed + feat + 2_000_003)
            ex_rows = []
            if use_examples:
                ex_rows = C.read_jsonl(f"{ex_dir}/{feat}.jsonl")
                for e in ex_rows:
                    assert e["row"] == r["row"], f"{ex_dir}/{feat}.jsonl row {e['row']} != {r['row']}"
            ex4_rows = C.read_jsonl(f"{ex4_dir}/{feat}.jsonl")
            for e in ex4_rows:
                assert e["row"] == r["row"], f"{ex4_dir}/{feat}.jsonl row {e['row']} != {r['row']}"

            # ---- corpus pools -----------------------------------------------------------
            # C16 is scan's 16M top-k. With no examples/ there is none, and check_corpus_source
            # has already refused any arm that would have shown it, so an empty pool here can
            # only mean "no C16 arm asked for one".
            tops = sorted(
                (e for e in ex_rows if e["kind"] == "top"), key=lambda e: -float(e["max_act"])
            )
            c16_pool = dedup(tops)
            doc_rows = C.read_jsonl(f"{exdoc_dir}/{feat}.jsonl") if use_docmax else []
            cand_rows = candidate_rows(ex_rows, ex4_rows, doc_rows, peak, use_examples)
            c4_pool = dedup(
                sorted(ex4_rows, key=lambda e: -float(e["max_act"]))
            )

            def corpus_ex(e, feat=feat, peak=peak):
                ids = corpus.ids(e["doc"], e["start"], e["len"])
                assert len(ids) == len(e["acts"]), (
                    f"feature {feat}, window (doc {e['doc']}, start {e['start']}, len {e['len']}): "
                    f"recovered {len(ids)} tokens but the stored acts are {len(e['acts'])} long"
                )
                w_ids, w_acts = (
                    centre_on_peak(ids, e["acts"]) if centre32 else (ids, e["acts"])
                )
                out = render_example(tok, w_ids, w_acts, peak, gate, mark)
                return {
                    **out,
                    "src": "corpus",
                    "window": e["window"],
                    "doc": e["doc"],
                    "start": e["start"],
                    "len": e["len"],
                    "size_tag": corpus.size_tag(e["doc"]),
                    "max_act": e["max_act"],
                }

            # ---- rollout pool -----------------------------------------------------------
            i = self_ix[r["row"]]
            acts = self_act[i]  # [n, T]
            rids = self_ids[i]
            peaks = np.nan_to_num(np.nanmax(np.where(np.isfinite(acts), acts, -np.inf), 1), nan=0.0)
            peaks = np.where(np.isfinite(peaks), peaks, 0.0)
            order = np.argsort(-peaks, kind="stable")
            # The corpus pools are deduplicated and the rollout pool was not, so a repeated
            # rollout cost the M arm an example slot at matched N -- in the direction of the effect
            # under test. Exact-text dedup, and the count is recorded per feature.
            roll_pool = []
            seen_text: set[str] = set()
            n_dup_roll = 0
            nla_status: dict[str, int] = {}
            if is_nla:
                # Rank NLA rollouts by their peak INSIDE the <explanation> body. Ranking on the
                # whole decode can pick a rollout for an activation on a tag or preamble token,
                # which the explainer is then not shown.
                body_peaks = np.zeros_like(peaks)
                for k in range(len(rids)):
                    ok = rids[k] >= 0
                    m, _ = explanation_token_mask(token_pieces(tok, rids[k][ok]))
                    a_ok = np.asarray(acts[k][ok], dtype=np.float64)[np.asarray(m, dtype=bool)]
                    a_ok = a_ok[np.isfinite(a_ok)]
                    body_peaks[k] = float(a_ok.max()) if a_ok.size else 0.0
                order = np.argsort(-body_peaks, kind="stable")
            for k in order.tolist():
                keep = rids[k] >= 0
                if not keep.any():
                    continue
                if is_nla:
                    # Show the explainer the verbalizer's ANSWER, not its scaffolding: the
                    # tokens of the <explanation> body only, with their activation marks kept
                    # aligned (Juan, 2026-09-21 -- judges were getting the full NLA output).
                    ids_k, acts_k = rids[k][keep], acts[k][keep]
                    m, status = explanation_token_mask(token_pieces(tok, ids_k))
                    nla_status[status] = nla_status.get(status, 0) + 1
                    m = np.asarray(m, dtype=bool)
                    if not m.any():
                        continue
                    e = render_example(tok, ids_k[m], acts_k[m], peak, gate, mark,
                                       rel_fallback=rel_fallback)
                    e["tag_status"] = status
                else:
                    e = render_example(tok, rids[k][keep], acts[k][keep], peak, gate, mark,
                                       rel_fallback=rel_fallback)
                if e["text"] in seen_text:
                    n_dup_roll += 1
                    continue
                seen_text.add(e["text"])
                roll_pool.append({**e, "src": "rollout", "k": int(k),
                                  "max_act": round(float(peaks[k]), 4)})

            # `doc_rows` is already one window per document; rank the documents by that window's
            # peak and deduplicate on the window key the way the other corpus pools do. `dedup`
            # is by window, not by document, so it is a no-op here unless docmax ever emits two
            # windows for one document -- in which case the pool must not silently show both.
            docmax_pool = dedup(sorted(doc_rows, key=lambda e: -float(e["max_act"])))
            pools = {"c16": c16_pool, "c4": c4_pool, "m": roll_pool, "docmax": docmax_pool,
                     "mdiv": diversify(roll_pool, 16, float(ac.get("mdiv_jaccard", 0.5)))}

            # ---- arm B's description: the NLA text itself, no explainer call ----------------
            # The verbalizer wrote FOUR answers for this feature; the one to use is the one that
            # actually drove the feature, i.e. the rollout with the highest sae_self peak -- the
            # same ordering the NLA arm's examples are ranked by, so arm A's first example and
            # arm B's description come from the same rollout. The `<explanation>` tags are the
            # verbalizer's output format, not content, so they are stripped with the SAME regex
            # rollouts_nla records `explanation` with; a rollout that never closed its tag
            # contributes its whole text and says so (`tag_found: false`) rather than being
            # dropped, because a missing description would silently shrink arm B's feature set.
            if is_nla:
                # `order` is the rollouts sorted by DESCENDING sae_self peak, so order[0] is the
                # answer that actually drove the feature -- and it is the same rollout arm A shows
                # first, so the two arms' inputs come from one text rather than two.
                k_best = int(order[0]) if len(order) else -1
                raw = nla_text.get((r["row"], k_best), "")
                nla_desc_rows.append(
                    {
                        "feature": feat,
                        "row": r["row"],
                        "k": k_best,
                        "peak": round(float(peaks[k_best]) if k_best >= 0 else 0.0, 4),
                        **nla_description(raw),
                    }
                )

            # ---- arms -------------------------------------------------------------------
            arm_rows = []
            shown_windows: list[dict] = []
            for name in arm_names:
                csrc, cn, msrc, mn = ARM_SPECS[name]
                picks = []
                if cn:
                    # NOT `pool`: that name is the shared _RandomPool of negatives, and rebinding
                    # it here is how the first build of the amended code died.
                    src_pool = pools[csrc]
                    if len(src_pool) < cn:
                        flags.append(
                            f"feature {feat}: arm {name} wanted {cn} {csrc} windows, "
                            f"has {len(src_pool)}"
                        )
                    picks += [corpus_ex(e) for e in src_pool[:cn]]
                if mn:
                    if len(pools[msrc]) < mn:
                        flags.append(
                            f"feature {feat}: arm {name} wanted {mn} rollouts, has {len(pools[msrc])}"
                        )
                    picks += pools[msrc][:mn]
                if cn and mn:
                    rng_arm.shuffle(picks)
                shown_windows += [p for p in picks if p["src"] == "corpus"]
                mc = mark_counts.setdefault(
                    name, {"blocks": 0, "gate": 0, "relative": 0, "unmarkable": 0,
                           "delphi": 0, "unmarked": 0}
                )
                for p in picks:
                    mc["blocks"] += 1
                    mc[p.get("marking", "gate")] = mc.get(p.get("marking", "gate"), 0) + 1
                    if not p["n_marked"]:
                        mc["unmarked"] += 1
                    mark_frac.append(p["n_marked"] / max(1, p["n_tok"]))
                    join_total += 1
                    join_bad += 0 if p["join_ok"] else 1
                    if float(p["peak_act"]) > peak:
                        n_exceed_peak += 1
                arm_rows.append(
                    {
                        "kind": "arm",
                        "arm": name,
                        "n": len(picks),
                        "block": exemplar_block(picks),
                        "examples": [
                            {
                                k: v
                                for k, v in p.items()
                                if k in ("src", "k", "window", "doc", "start", "len", "size_tag",
                                         "max_act", "n_marked", "n_tok",
                                         "marking", "block_peak", "peak_frac")
                            }
                            for p in picks
                        ],
                    }
                )
            if args.get("epo_strings"):
                picks = _epo_arm(args["epo_strings"], feat, tok, peak, gate, mark,
                                 rel_fallback=rel_fallback)
                arm_rows.append(
                    {"kind": "arm", "arm": "E", "n": len(picks), "block": exemplar_block(picks),
                     "examples": [{"src": "epo", "k": p["k"], "n_marked": p["n_marked"],
                                   "n_tok": p["n_tok"]} for p in picks]}
                )

            # ---- test set ---------------------------------------------------------------
            shown_docs = {int(p["doc"]) for p in shown_windows}
            shown_windows_ids = {int(p["window"]) for p in shown_windows}
            used_docs = set(shown_docs)
            # DRAW 1 IS ALLOCATED FIRST. It was briefly the other way round, to protect the null
            # from an empty draw 2, and that was the wrong trade: draw 1 is the set EVERY arm is
            # scored on, so starving it drops the feature from every contrast, while draw 2 is used
            # by one arm (`C16-draw2`, the null) and a null measured on fewer features is still a
            # null. MEASURED on the first pilot: 7 of 64 features -- all in the rarest density
            # quartile -- had no draw-1 positive at all and fell out of every comparison, which at
            # 512 features projects to ~55 lost q0 features. Draw 2 now takes the remainder, and
            # its own shortfall is recorded.
            draws = {}
            for tag in ("test", "test2"):
                items, info = draw_test(
                    feat=feat,
                    ex_rows=cand_rows,
                    tops=tops,
                    pool=pool,
                    corpus=corpus,
                    tok=tok,
                    gate=gate,
                    rng_test=rng_test,
                    rng_mark=rng_mark,
                    cfgv=(n_pos, n_neg, n_neg_nearmiss, nearmiss_source, gate_positives,
                          allow_top_fallback),
                    shown_windows_ids=shown_windows_ids,
                    shown_docs=shown_docs,
                    used_docs=used_docs,
                    flags=flags,
                    tag=tag,
                    fuzz_marks=fuzz_marks,
                )
                draws[tag] = (items, info)
            items, info1 = draws["test"]
            info2 = draws["test2"][1]
            test_rows = [{"kind": tag, "i": i, **it}
                         for tag in ("test", "test2")
                         for i, it in enumerate(draws[tag][0])]
            n_mark_neg = info1["n_mark_neg"]
            n_top_fallback = info1["n_top_fallback"]


            meta = {
                "kind": "meta",
                "feature": feat,
                "row": r["row"],
                "stratum": int(r["stratum"]),
                # FIELD NAMES DIFFER BY FAMILY, which is why these are not r["..."]. `targets.py`
                # writes `density` (gated fires / scanned positions) and `fires_gated` on a `sae`
                # row; `features/draw_sae2m.py` writes `gated_fires` and NO density on a
                # `sae2m_enc` one. Both are descriptive covariates -- stats.py reports by them and
                # nothing selects on them -- so a missing one is None rather than a KeyError that
                # would stop a build over a column nobody gates on. `stratum` IS required and is
                # on both.
                "density": _covariate(r, "density"),
                "corpus_peak": round(peak, 4),
                "fires_gated": _covariate(r, "fires_gated", "gated_fires"),
                "fire_fraction": fire_of[r["row"]],
                "gate": gate,
                "n_pos": info1["n_pos"],
                "n_neg": info1["n_neg"],
                "pool_c16": len(c16_pool),
                "pool_c4": len(c4_pool),
                "pool_m": len(roll_pool),
                "n_dup_rollouts": n_dup_roll,
                # NLA arm only: how each shown rollout's <explanation> body was found
                # (closed / unclosed = ran into max_new / none = no tag, shown whole).
                "nla_tag_status": nla_status,
                "pool_mdiv": len(pools["mdiv"]) if "mdiv" in pools else 0,
                "pool_cand": len(cand_rows),
                "pool_cand_gated": sum(1 for e in cand_rows if float(e["max_act"]) > gate),
                "n_mark_neg": n_mark_neg,
                "n_top_fallback": n_top_fallback,
                "n_pos_bands": info1["n_pos_bands"],
                "n_neg_nearmiss": info1["n_near"],
                "pool_zero": info1["pool_zero"],
                "pool_nearmiss": info1["pool_nearmiss"],
                "shown_docs": len(shown_docs),
                "draw1": info1,
                "draw2": info2,
            }
            C.write_jsonl(od.file(f"{feat}.jsonl"), [meta, *arm_rows, *test_rows])
            feat_table.append({k: v for k, v in meta.items() if k != "kind"})

        if is_nla:
            od.write_jsonl("nla_desc.jsonl", nla_desc_rows)
            n_tagged = sum(1 for x in nla_desc_rows if x["tag_found"])
            n_empty_desc = sum(1 for x in nla_desc_rows if not x["description"])
            od.note(
                f"`nla_desc.jsonl` is arm B's input: for each feature, the NLA rollout with the "
                f"HIGHEST sae_self peak activation, with its <explanation> tags stripped -- the "
                f"description that arm scores WITHOUT any explainer call. {n_tagged} of "
                f"{len(nla_desc_rows)} carried a closed tag (the rest contribute their whole "
                f"text, flagged `tag_found: false`); {n_empty_desc} are empty and `run` scores no "
                f"arm for those features. `run --arms ...,NLA-desc` picks the arm up from this "
                f"file; `build` itself has no such arm."
            )
            od.note(
                f"ARM PROVENANCE: rollout source = the NLA VERBALIZER `{maemm}` "
                f"({cfg['maemms'][maemm].get('hf', '?')}), n = {NLA_N} texts per feature at "
                f"nla.max_new {cfg['maemms'][maemm]['nla']['max_new']}. The NLA arm is therefore "
                f"NOT matched-N against C4/C16 (16 examples each) and its texts are ~3x longer; "
                f"both differences are properties of the baseline at its own operating point and "
                f"neither is corrected for here. The MAEMM rollout arms "
                f"({', '.join(ROLLOUT_ARMS_NOT_NLA)}) are REFUSED with an `nla` --maemm."
            )
        od.write_json("features.json", {"features": feat_table})
        od.write_json(
            "build.json",
            {
                "base": base,
                "set": set_name,
                "maemm": maemm,
                "engine": engine,
                "sae": sae_key,
                "gate": gate,
                "n_features": len(picked),
                "feat_seed": feat_seed,
                "shuffle_seed": shuffle_seed,
                "n_examples": n_ex,
                "corpus_prefix_m": prefix_m,
                "centre32": centre32,
                "mark": mark,
                "fuzz_marks": fuzz_marks,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_neg_nearmiss": n_neg_nearmiss,
                "nearmiss_source": nearmiss_source,
                "gate_consistent_positives": gate_positives,
                "allow_top_fallback": allow_top_fallback,
                "examples": ex_dir if use_examples else "(absent -- no scan examples/ for this SAE)",
                "examples_4m": ex4_dir,
                # WHERE THE TEST POSITIVES CAME FROM. Not a detail: a 4M-prefix pool searches a
                # quarter of the text a 16M one does, so a positive drawn from it is drawn from a
                # weaker pool, and the same feature's numbers are not comparable across the two.
                "positive_source": positive_source,
                "examples_docmax": exdoc_dir if use_docmax else "(absent -- test set is short)",
                "random_pool": pool.path,
                "random_pool_windows": pool.n_win,
                "arms": {a: ARM_SPECS[a] for a in arm_names},
                "rollout_source": ("nla-verbalizer" if is_nla else "maemm"),
                "nla_desc": ("nla_desc.jsonl" if is_nla else "(not an nla maemm)"),
                "epo_strings": args.get("epo_strings") or "(E arm not run: hook only)",
                "mean_marked_fraction": round(float(np.mean(mark_frac)) if mark_frac else 0.0, 4),
                "token_join_mismatches": f"{join_bad}/{join_total}",
                # The coordinator's acceptance numbers, as single figures rather than a flag count.
                "n_short_draw1": sum(1 for f in feat_table if f["draw1"]["n_pos"] < n_pos),
                "n_short_draw2": sum(1 for f in feat_table if f["draw2"]["n_pos"] < n_pos),
                "n_empty_draw2": sum(1 for f in feat_table if f["draw2"]["n_pos"] == 0),
                "n_short_c4": sum(1 for f in feat_table if f["pool_c4"] < 16),
                "n_short_neg": sum(1 for f in feat_table if f["draw1"]["n_neg"] < n_neg),
                "n_no_pos_draw1": sum(1 for f in feat_table if f["draw1"]["n_pos"] == 0),
                # The features that will fall out of every contrast, and which quartile loses them.
                "no_pos_draw1_by_stratum": {
                    str(q): sum(1 for f in feat_table
                                if f["draw1"]["n_pos"] == 0 and f["stratum"] == q)
                    for q in sorted({f["stratum"] for f in feat_table})
                },
                "n_top_fallback_features": sum(1 for f in feat_table if f["n_top_fallback"] > 0),
                "n_top_fallback_positives": sum(f["n_top_fallback"] for f in feat_table),
                # How often Delphi's quantisation clamp is actually active. `ceil(10*act/peak_f)`
                # clamps at 10 when a shown example's activation EXCEEDS the feature's corpus peak,
                # which corpus examples cannot do by construction but rollouts can. Recorded here
                # so the paper's cell reads from a summary file instead of needing the ~90 MB of
                # per-feature jsonl.
                "n_features_with_dup_rollouts": sum(
                    1 for f in feat_table if f.get("n_dup_rollouts", 0) > 0
                ),
                "n_dup_rollouts_total": sum(f.get("n_dup_rollouts", 0) for f in feat_table),
                "n_shown_exceeding_corpus_peak": n_exceed_peak,
                "rollout_mark": rollout_mark,
                # Per arm: how many blocks each marking rule produced, and how many reached the
                # explainer with NOTHING marked. The last number is the one to read against a
                # gate-marked build of the same features.
                "marking_counts": mark_counts,
                "n_shown_examples": join_total,
                "min_pos_draw1": min((f["draw1"]["n_pos"] for f in feat_table), default=0),
                "min_pos_draw2": min((f["draw2"]["n_pos"] for f in feat_table), default=0),
                "flags": flags,
            },
        )
        od.index["features"] = {"kind": "jsonl", "rows": len(picked),
                                "bytes": sum(os.path.getsize(od.file(f"{r['id']}.jsonl")) for r in picked)}
        od.note(
            f"one `<feature>.jsonl` per tested feature: a `meta` row, one `arm` row per variant "
            f"({', '.join(arm_names)}) carrying the rendered explainer user message in `block` and "
            f"the provenance of every example in `examples`, then {n_pos + n_neg} `test` rows."
        )
        od.note(
            f"rendering is Delphi's (commit 4fea06e6e8b6, transcription at "
            f"repo-maemm/eval/autointerp_detection.py:229-450): activating tokens wrapped "
            f"<<like this>>, an `Activations:` line of (\"tok\" : n) pairs with "
            f"n = ceil(10*act/peak_f) clamped to [0,10], peak_f = the feature's 16M corpus max "
            f"from sae/{C.split_key(sae_key, 'sae')[1]}/max_act.f16, first {MAX_SHOWN_ACTS} "
            f"activating tokens only. Mean marked fraction of a shown example: "
            f"{float(np.mean(mark_frac)) if mark_frac else 0:.4f}."
        )
        centre_note = (
            f"CORPUS examples are re-cut to {CENTRE32_LEN} tokens CENTRED on the peak "
            f"(--centre32, Delphi's example_ctx_len/center_examples). "
            if centre32
            else ""
        )
        od.note(
            centre_note
            + "corpus windows are recovered from corpus/tokens.i32 at (doc, start, len) and the "
            "recovered length is ASSERTED equal to len(acts); the window is also asserted to be "
            "one of common.windows_of's cuts. Per-token decode joins back to tok.decode(ids) on "
            f"{join_total - join_bad}/{join_total} examples (byte-level BPE can split a codepoint)."
        )
        od.note(
            f"dedup: within a source, a window overlapping one already kept IS DROPPED (stride "
            f"{C.SCAN_STRIDE} in {C.SCAN_BLOCK}-token windows makes the top-k of a feature "
            f"routinely four cuts of one passage). C4 comes from `{ex4_dir}` -- the "
            f"{prefix_m}M prefix's OWN top-128 (amendment A3), not the {prefix_m}M members of the "
            f"16M ranking, which left a median of 14 candidates after dedup."
        )
        od.note(
            f"test positives are GATE-CONSISTENT (amendment 2026-09-16): only a window whose "
            f"stored peak pre-gate activation exceeds the gate {gate:.4f} counts, and the band "
            f"draw runs over gate-passing rows only. Without it, MEASURED on feature 845, 17 of "
            f"20 band-drawn positives sat below the gate on text unrelated to the feature and "
            f"pinned every arm near 0.6 balanced accuracy whatever its description said. Bands are "
            f"EQUAL-WIDTH bins of (0, max_act] (precompute/scan.py:257), not Delphi's quantiles; "
            f"a band that cannot fill its quota of {n_pos // len(BANDS)} carries the deficit to "
            f"the next band DOWN, and what is still short is filled from the top-ranked windows "
            f"BEYOND those any arm shows (labelled band `top`, counted per feature as "
            f"`n_top_fallback`). The candidate pool is the stored q-bands PLUS "
            f"`examples_docmax/` -- one window from each of a feature's top 256 DOCUMENTS -- "
            f"deduplicated by window id, each row banded by scan.py's own "
            f"`ceil(max_act/peak*4)-1`. Without the document-diverse half, A1 and A4 together left "
            f"draw 1 short on 29 of 64 pilot features and draw 2 empty on 21."
        )
        od.note(
            "TWO test draws per feature, `test` and `test2` (amendment A7): disjoint from each "
            "other and from every shown example at the document level, under identical rules. "
            "`run.py` scores C16 on both; the per-feature difference between the two numbers is "
            "the NULL distribution reported beside every win fraction and every \"worse on X%\" "
            "figure. It replaces the temperature-0 repeat, which measured judge jitter rather "
            "than the test-set sampling every contrast is actually exposed to."
        )
        od.note(
            f"negatives (amendment A5): {n_neg} per feature = {n_neg - n_neg_nearmiss} "
            f"zero-activation windows from the 2048-window `random_pool` (Delphi's published "
            f"`non_activating_source = random`) + {n_neg_nearmiss} near-miss windows "
            f"(0 < peak <= gate) from `{nearmiss_source}`, falling back to zero-activation "
            f"randoms when those run out. Delphi's random rule stays the reference; the near-miss "
            f"half is what keeps the negative side off its ceiling, and `src` on every test row "
            f"says which half an item is in so detection can be reported on both separately. "
            f"scan's `_random256` is NOT used: 256 windows carrying only a per-feature maximum "
            f"cannot supply the zero-activation half for the densest tested features (its "
            f"per-feature minimum was measured at 14 < 20)."
        )
        od.note(
            "DOCUMENT-level disjointness, asserted: no test item shares a `doc` with any window "
            "shown to any arm's explainer, and no two test items share a document. Window overlap "
            "alone is not enough at stride 16 -- two windows of one document 200 tokens apart do "
            "not overlap and are still the same passage. Excluding across ALL arms is what keeps "
            "the test set identical per arm, which is what makes the comparison paired."
        )
        od.note(
            f"fuzzing marks (DEVIATION, stated): positives mark the peak token and every token "
            f"above the SAE gate {gate:.4f}; Delphi marks act > 0.3 * max_activation. Negatives "
            f"have nothing to mark, so a contiguous run of the feature's mean positive mark count "
            f"is marked at a seeded random start -- [RECONSTRUCTED], Delphi's fuzzing _prepare is "
            f"not in our transcription."
        )
        od.note(f"{len(flags)} flags (build.json `flags`), none silently filled")
        if flags:
            od.status = "flagged"

    return {
        "out": out_dir,
        "features": len(picked),
        "arms": arm_names,
        "n_short_draw1": sum(1 for f in feat_table if f["draw1"]["n_pos"] < n_pos),
        "n_short_draw2": sum(1 for f in feat_table if f["draw2"]["n_pos"] < n_pos),
        "n_empty_draw2": sum(1 for f in feat_table if f["draw2"]["n_pos"] == 0),
        "n_no_pos_draw1": sum(1 for f in feat_table if f["draw1"]["n_pos"] == 0),
        "n_short_c4": sum(1 for f in feat_table if f["pool_c4"] < 16),
        "n_top_fallback_features": sum(1 for f in feat_table if f["n_top_fallback"] > 0),
        "n_top_fallback_positives": sum(f["n_top_fallback"] for f in feat_table),
        "flags": len(flags),
        "mean_marked_fraction": round(float(np.mean(mark_frac)) if mark_frac else 0.0, 4),
        "token_join_mismatches": join_bad,
        "first_flags": flags[:5],
    }
