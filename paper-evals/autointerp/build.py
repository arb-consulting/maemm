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

FAMILY = "sae"
# Delphi lists at most this many activating tokens per example (facts §3, explainer.py).
MAX_SHOWN_ACTS = 10
BANDS = ("q0", "q1", "q2", "q3")


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


def render_example(tok, ids, acts, peak: float, gate: float) -> dict:
    """One rendered explainer example from an id list and its per-token pre-gate activations.

    MARKING RULE (design amendment A2, ONE rule for explainer examples, fuzzing marks and
    rollouts alike): a token is marked iff its pre-gate activation EXCEEDS THE GATE -- Delphi's
    post-TopK "activating" -- and the `Activations:` line lists the marked tokens by DESCENDING
    activation, top 10, so the peak token is always present. The build had marked `act > 0`, which
    is a post-ReLU non-zero and put markers on 70-86% of the tokens of a dense feature's example.
    """
    pieces = token_pieces(tok, ids)
    a = [float(x) for x in acts]
    quant = [quant_act(x, peak) for x in a]
    marks = [x > gate for x in a]
    shown = sorted(
        ((pieces[i], quant[i], a[i]) for i in range(len(pieces)) if marks[i]),
        key=lambda t: -t[2],
    )[:MAX_SHOWN_ACTS]
    shown = [(t, n) for t, n, _ in shown]
    return {
        "text": "".join(pieces),
        "text_marked": marked_text(pieces, marks),
        "activations": shown,
        "n_marked": int(sum(marks)),
        "peak_act": round(float(max(acts)) if len(acts) else 0.0, 4),
        "n_tok": len(pieces),
        "join_ok": "".join(pieces) == tok.decode([int(i) for i in ids]),
    }


def render_test(tok, ids, acts, gate: float, rng: random.Random, n_mark_neg: int) -> dict:
    """One test item: the plain text the DETECTION scorer sees and the marked text FUZZING sees.

    Marking rule: design amendment A2's single rule, `act > gate` -- the same rule
    `render_example` uses, so a fuzzing mark and an explainer mark mean the same event. That is a
    DEVIATION from Delphi, whose fuzzing scorer marks `act > 0.3 * max_activation`
    (related-work/2026-09-15_delphi-updates-and-negatives.md:49); the gate is the fire rule the
    rest of this paper uses. Every test positive is gate-consistent (A1), so its peak is marked by
    construction.

    A negative window has no activation to mark, so a contiguous run of `n_mark_neg` tokens is
    marked at a seeded random start -- the construction Delphi's intruder scorer uses for its
    intruder example ("a random selection of tokens is highlighted, the count matching the average
    in the activating examples, rounded down"). Delphi's own fuzzing `_prepare` is NOT in our
    transcription, so this is [RECONSTRUCTED] and build.json says so.
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
    # Descriptive pilot points only (amendment A9: N = 16 is fixed a priori, not selected).
    "C16-N8": ("c16", 8, None, 0),
    "M-N8": (None, 0, "m", 8),
    "M-N32": (None, 0, "m", 32),
}
# The arms the FULL run scores. The rest are pilot-only descriptive points. `F` and `C16-draw2`
# are scorer-only pseudo-arms that `run.py` adds: F reuses another feature's description, and
# C16-draw2 reuses C16's own description on the second, disjoint test draw (amendments A6, A7).
FULL_ARMS = ("C16", "C4", "M", "C4M", "C32", "C16M16")


def _epo_arm(path: str, feature: int, tok, peak: float, gate: float):
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
        e = render_example(tok, s["ids"], s["acts"], peak, gate)
        out.append({**e, "src": "epo", "k": i})
    return out


# ---------------------------------------------------------------------------------------------


def draw_test(
    *, feat, ex_rows, tops, pool, corpus, tok, gate, rng_test, rng_mark, cfgv,
    shown_windows_ids, shown_docs, used_docs, flags, tag,
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
        t = render_test(tok, ids, e["acts"], gate, rng_mark, 0)
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
        t = render_test(tok, ids, None, gate, rng_mark, n_mark_neg)
        items.append({**t, "label": 0, "src": f"nearmiss-{src}", "band": "-", **meta_w})
    for w in zero_pick:
        pw = pool.windows[w]
        ids = corpus.ids(pw["doc"], pw["start"], pw["len"])
        t = render_test(tok, ids, None, gate, rng_mark, n_mark_neg)
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
    allow_top_fallback = bool(ac["allow_top_fallback"])
    engine = args.get("engine") or "vllm"
    arm_names = [a for a in (args.get("arms") or "").split(",") if a] or list(ARM_SPECS)
    for a in arm_names:
        assert a in ARM_SPECS, f"unknown arm {a!r}, want some of {list(ARM_SPECS)}"
    assert n_ex == 16, (
        f"autointerp.n_examples is {n_ex}: ARM_SPECS pins the per-arm counts explicitly, so "
        f"changing N means editing them, not this number"
    )

    sae_keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    assert len(sae_keys) == 1, f"base {base} has {len(sae_keys)} SAEs in config, expected exactly 1"
    sae_key = sae_keys[0]
    ex_dir = f"{C.sae_dir(sae_key, root)}/examples"
    # Amendment A3: C4 reads the 4M prefix's OWN top-128, not the 4M-prefix members of the 16M
    # ranking (median 14 candidates after dedup, fewer than 16 on 38 of 64 pilot features).
    ex4_dir = f"{C.sae_dir(sae_key, root)}/examples_4m/{set_name}"
    assert os.path.exists(f"{ex4_dir}/tested.json"), (
        f"no {prefix_m}M example scan at {ex4_dir}: run `--stage examples_4m` on this (base, set) "
        f"first -- the C4 arm is its top-128, not a filter of the 16M one (amendment A3)"
    )
    hdir = C.heldout_dir(base, set_name, root)
    sdir = C.scores_dir(maemm, set_name, root, engine)
    self_dir = f"{sdir}/sae_self{args.get('out_suffix') or ''}"

    rows_meta = C.read_jsonl(f"{hdir}/ids.jsonl")
    sae_rows = [r for r in rows_meta if r["family"] == FAMILY]
    picked = draw_features(sae_rows, n_feat, feat_seed)
    if args.get("rows"):
        # --rows OVERRIDES the stratified draw rather than intersecting it: a shakeout asks for
        # specific rows and must get exactly those, not "whichever of them the draw happened to
        # pick" (MEASURED 2026-09-16: --rows 1024-1025 quietly built one feature).
        want = set(C.parse_rows(args["rows"], len(rows_meta)))
        picked = [r for r in sae_rows if r["row"] in want]
        assert picked, f"--rows {args['rows']!r} selected none of the {FAMILY} rows"
    print(f"[build] {len(picked)} features, arms {arm_names}", flush=True)

    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][base]["hf"]))
    corpus = _Corpus(base, root)
    d_sae_peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (-1,))
    pool = _RandomPool(f"{C.sae_dir(sae_key, root)}/random_pool/{set_name}")

    self_meta = json.load(open(f"{self_dir}/sae_self.json"))
    gate = float(self_meta["gate"])
    self_rows = list(self_meta["rows"])
    n_roll = int(self_meta["n"])
    shape = (len(self_rows), n_roll, C.SCORE_WIDTH)
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
            ex_rows = C.read_jsonl(f"{ex_dir}/{feat}.jsonl")
            for e in ex_rows:
                assert e["row"] == r["row"], f"{ex_dir}/{feat}.jsonl row {e['row']} != {r['row']}"

            # ---- corpus pools -----------------------------------------------------------
            tops = sorted(
                (e for e in ex_rows if e["kind"] == "top"), key=lambda e: -float(e["max_act"])
            )
            c16_pool = dedup(tops)
            ex4_rows = C.read_jsonl(f"{ex4_dir}/{feat}.jsonl")
            c4_pool = dedup(
                sorted(ex4_rows, key=lambda e: -float(e["max_act"]))
            )

            def corpus_ex(e, feat=feat, peak=peak):
                ids = corpus.ids(e["doc"], e["start"], e["len"])
                assert len(ids) == len(e["acts"]), (
                    f"feature {feat}, window (doc {e['doc']}, start {e['start']}, len {e['len']}): "
                    f"recovered {len(ids)} tokens but the stored acts are {len(e['acts'])} long"
                )
                out = render_example(tok, ids, e["acts"], peak, gate)
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
            roll_pool = []
            for k in order.tolist():
                keep = rids[k] >= 0
                if not keep.any():
                    continue
                e = render_example(tok, rids[k][keep], acts[k][keep], peak, gate)
                roll_pool.append({**e, "src": "rollout", "k": int(k), "max_act": round(float(peaks[k]), 4)})

            pools = {"c16": c16_pool, "c4": c4_pool, "m": roll_pool}

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
                for p in picks:
                    mark_frac.append(p["n_marked"] / max(1, p["n_tok"]))
                    join_total += 1
                    join_bad += 0 if p["join_ok"] else 1
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
                                         "max_act", "n_marked", "n_tok")
                            }
                            for p in picks
                        ],
                    }
                )
            if args.get("epo_strings"):
                picks = _epo_arm(args["epo_strings"], feat, tok, peak, gate)
                arm_rows.append(
                    {"kind": "arm", "arm": "E", "n": len(picks), "block": exemplar_block(picks),
                     "examples": [{"src": "epo", "k": p["k"], "n_marked": p["n_marked"],
                                   "n_tok": p["n_tok"]} for p in picks]}
                )

            # ---- test set ---------------------------------------------------------------
            shown_docs = {int(p["doc"]) for p in shown_windows}
            shown_windows_ids = {int(p["window"]) for p in shown_windows}
            used_docs = set(shown_docs)
            draws = []
            for tag in ("test", "test2"):
                items, info = draw_test(
                    feat=feat,
                    ex_rows=ex_rows,
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
                )
                draws.append((tag, items, info))
            items, info1 = draws[0][1], draws[0][2]
            info2 = draws[1][2]
            test_rows = [{"kind": tag, "i": i, **it}
                         for tag, its, _ in draws for i, it in enumerate(its)]
            n_mark_neg = info1["n_mark_neg"]
            n_top_fallback = info1["n_top_fallback"]


            meta = {
                "kind": "meta",
                "feature": feat,
                "row": r["row"],
                "stratum": int(r["stratum"]),
                "density": r["density"],
                "corpus_peak": round(peak, 4),
                "fires_gated": r["fires_gated"],
                "fire_fraction": fire_of[r["row"]],
                "gate": gate,
                "n_pos": info1["n_pos"],
                "n_neg": info1["n_neg"],
                "pool_c16": len(c16_pool),
                "pool_c4": len(c4_pool),
                "pool_m": len(roll_pool),
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
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_neg_nearmiss": n_neg_nearmiss,
                "nearmiss_source": nearmiss_source,
                "gate_consistent_positives": gate_positives,
                "allow_top_fallback": allow_top_fallback,
                "examples_4m": ex4_dir,
                "random_pool": pool.path,
                "random_pool_windows": pool.n_win,
                "arms": {a: ARM_SPECS[a] for a in arm_names},
                "epo_strings": args.get("epo_strings") or "(E arm not run: hook only)",
                "mean_marked_fraction": round(float(np.mean(mark_frac)) if mark_frac else 0.0, 4),
                "token_join_mismatches": f"{join_bad}/{join_total}",
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
        od.note(
            f"corpus windows are recovered from corpus/tokens.i32 at (doc, start, len) and the "
            f"recovered length is ASSERTED equal to len(acts); the window is also asserted to be "
            f"one of common.windows_of's cuts. Per-token decode joins back to tok.decode(ids) on "
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
            f"`n_top_fallback`)."
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
        "flags": len(flags),
        "mean_marked_fraction": round(float(np.mean(mark_frac)) if mark_frac else 0.0, 4),
        "token_join_mismatches": join_bad,
        "first_flags": flags[:5],
    }
