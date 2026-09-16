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
  C4M16    ADDITIVE ablation: all 16 of C4 plus all 16 of M, N = 32
  C16-N8 / C16-N32 / M-N8 / M-N32   the same two sources at N in {8, 32}
  C16-rep  a byte-identical copy of C16 under a second name, so `run` explains and scores it a
           second time: the run-to-run noise floor every difference is quoted against
  E        NOT RUN. `--epo-strings <jsonl>` is a documented hook, see `_epo_arm` below

Rendering is Delphi's (facts §3, transcription at repo-maemm/eval/autointerp_detection.py:229-450):
activating tokens wrapped `<<like this>>`, then an `Activations:` line of `("tok" : n)` pairs with
n = ceil(10 * act / peak_f) clamped to [0, 10] and peak_f = the feature's corpus max at 16M
(`sae/<sae>/max_act.f16`). Only the first 10 activating tokens of an example are listed, as Delphi
does. All activations everywhere are the stored PRE-GATE ones.

Test set (design §3), identical across arms and never shown to any explainer:
  * 20 positives, 5 from each of the four stored activation bands `q0..q3`. Those bands are
    EQUAL-WIDTH bins of (0, max_act], not Delphi's quantiles (precompute/scan.py:257) -- stated
    here and in build.json rather than relabelled.
  * 20 negatives from the shared `_random256` pool, restricted to windows whose stored per-feature
    maximum is exactly 0 (Delphi's published `non_activating_source "random"` rule).
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


def render_example(tok, ids, acts, peak: float) -> dict:
    """One rendered explainer example from an id list and its per-token pre-gate activations."""
    pieces = token_pieces(tok, ids)
    quant = [quant_act(float(a), peak) for a in acts]
    marks = [q >= 1 for q in quant]  # Delphi marks every ACTIVATING token; q >= 1 iff act > 0
    shown = [(pieces[i], quant[i]) for i in range(len(pieces)) if marks[i]][:MAX_SHOWN_ACTS]
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

    Marking rule (design §3): the peak token and every token above the SAE's learned gate. That is
    a DEVIATION from Delphi, whose fuzzing scorer marks `act > 0.3 * max_activation`
    (related-work/2026-09-15_delphi-updates-and-negatives.md:49); the gate is the fire rule the
    rest of this paper uses, so it is the one used here and the difference is stated rather than
    silently absorbed.

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
        if len(a):
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
    "C4M16": ("c4", 16, "m", 16),
    "C16-N8": ("c16", 8, None, 0),
    "C16-N32": ("c16", 32, None, 0),
    "M-N8": (None, 0, "m", 8),
    "M-N32": (None, 0, "m", 32),
    "C16-rep": ("c16", 16, None, 0),
}


def _epo_arm(path: str, feature: int, tok, peak: float):
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
        e = render_example(tok, s["ids"], s["acts"], peak)
        out.append({**e, "src": "epo", "k": i})
    return out


# ---------------------------------------------------------------------------------------------


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
    tested = json.load(open(f"{ex_dir}/tested.json"))
    col_of = {int(f): i for i, f in enumerate(tested["features"])}
    d_sae_peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (-1,))
    random_pool = C.read_jsonl(f"{ex_dir}/_random256.jsonl")

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
            c4_pool = dedup([e for e in tops if corpus.size_tag(e["doc"]) <= prefix_m])

            def corpus_ex(e, feat=feat, peak=peak):
                ids = corpus.ids(e["doc"], e["start"], e["len"])
                assert len(ids) == len(e["acts"]), (
                    f"feature {feat}, window (doc {e['doc']}, start {e['start']}, len {e['len']}): "
                    f"recovered {len(ids)} tokens but the stored acts are {len(e['acts'])} long"
                )
                out = render_example(tok, ids, e["acts"], peak)
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
                e = render_example(tok, rids[k][keep], acts[k][keep], peak)
                roll_pool.append({**e, "src": "rollout", "k": int(k), "max_act": round(float(peaks[k]), 4)})

            pools = {"c16": c16_pool, "c4": c4_pool, "m": roll_pool}

            # ---- arms -------------------------------------------------------------------
            arm_rows = []
            shown_windows: list[dict] = []
            for name in arm_names:
                csrc, cn, msrc, mn = ARM_SPECS[name]
                picks = []
                if cn:
                    pool = pools[csrc]
                    if len(pool) < cn:
                        flags.append(
                            f"feature {feat}: arm {name} wanted {cn} {csrc} windows, "
                            f"has {len(pool)}"
                        )
                    picks += [corpus_ex(e) for e in pool[:cn]]
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
                picks = _epo_arm(args["epo_strings"], feat, tok, peak)
                arm_rows.append(
                    {"kind": "arm", "arm": "E", "n": len(picks), "block": exemplar_block(picks),
                     "examples": [{"src": "epo", "k": p["k"], "n_marked": p["n_marked"],
                                   "n_tok": p["n_tok"]} for p in picks]}
                )

            # ---- test set ---------------------------------------------------------------
            # The stored bands are independent reservoirs, so a band window CAN also be a `top`
            # window (facts §4.4: the top band overlaps kind:"top"). The design assumed they were
            # disjoint; they are not, so every candidate that overlaps ANY window shown to ANY
            # explainer is excluded here and the count is recorded. Excluding across all arms keeps
            # the test set identical for every arm, which is what makes the comparison paired.
            # Bands are walked from the HIGHEST activation band down; a band that cannot fill its
            # quota carries the deficit to the next band DOWN, which is the design's rule stated as
            # an algorithm. A deficit still standing after q0 is flagged, never silently filled from
            # above.
            per_band = n_pos // len(BANDS)
            used = list(shown_windows)
            pos_rows: list[dict] = []
            short_bands: list[tuple[str, int]] = []
            deficit = 0
            for band in reversed(BANDS):
                cand = [
                    e
                    for e in ex_rows
                    if e["kind"] == band and not any(_overlaps(e, u) for u in used)
                ]
                rng_test.shuffle(cand)
                want = per_band + deficit
                take = cand[:want]
                deficit = want - len(take)
                if len(take) < per_band:
                    short_bands.append((band, len(take)))
                for e in take:
                    e = dict(e)
                    e["band"] = band
                    e["band_index"] = BANDS.index(band)
                    pos_rows.append(e)
                    used.append(e)
            n_excluded = sum(
                1
                for e in ex_rows
                if e["kind"] in BANDS and any(_overlaps(e, u) for u in shown_windows)
            )
            if short_bands or len(pos_rows) < n_pos:
                flags.append(
                    f"feature {feat}: positives short -- bands {short_bands}, "
                    f"{len(pos_rows)}/{n_pos} after carrying the deficit down "
                    f"({n_excluded} band windows excluded for overlapping a shown example)"
                )

            col = col_of[feat]
            zero = [w for w in random_pool if float(w["max_act"][col]) == 0.0]
            if len(zero) < n_neg:
                flags.append(f"feature {feat}: only {len(zero)} zero-activation negatives of {n_neg}")
            neg_rows = rng_test.sample(zero, min(n_neg, len(zero)))

            items = []
            for e in pos_rows:
                ids = corpus.ids(e["doc"], e["start"], e["len"])
                assert len(ids) == len(e["acts"]), (
                    f"feature {feat}, test positive (doc {e['doc']}, start {e['start']}): "
                    f"recovered {len(ids)} tokens vs {len(e['acts'])} stored acts"
                )
                t = render_test(tok, ids, e["acts"], gate, rng_mark, 0)
                items.append({**t, "label": 1, "src": "corpus", "band": e["band"],
                              "window": e["window"], "doc": e["doc"], "start": e["start"],
                              "max_act": e["max_act"]})
            # The negatives' random mark length is the positives' MEAN mark count, rounded down --
            # so it is computed from the rendered positives above, not guessed.
            n_mark_neg = max(1, int(np.mean([it["n_marked"] for it in items])) if items else 1)
            for w in neg_rows:
                ids = corpus.ids(w["doc"], w["start"], w["len"])
                t = render_test(tok, ids, None, gate, rng_mark, n_mark_neg)
                items.append({**t, "label": 0, "src": "random", "band": "-",
                              "window": w["window"], "doc": w["doc"], "start": w["start"],
                              "max_act": 0.0})
            # ONE shuffle, seeded by the feature, so every arm scores the same items in the same
            # order and in the same batches (repo-maemm/eval/autointerp_detection.py:2958-2969).
            rng_test.shuffle(items)
            test_rows = [{"kind": "test", "i": i, **it} for i, it in enumerate(items)]

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
                "n_pos": sum(1 for it in items if it["label"] == 1),
                "n_neg": sum(1 for it in items if it["label"] == 0),
                "pool_c16": len(c16_pool),
                "pool_c4": len(c4_pool),
                "pool_m": len(roll_pool),
                "n_mark_neg": n_mark_neg,
                "band_windows_excluded": n_excluded,
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
            f"routinely four cuts of one passage). C4 is the same ranking restricted to documents "
            f"tagged <= {prefix_m}M, i.e. the nested prefix, NOT a re-scan."
        )
        od.note(
            f"test set: {n_pos} positives, {n_pos // len(BANDS)} from each stored band "
            f"{list(BANDS)}. Those bands are EQUAL-WIDTH bins of (0, max_act] "
            f"(precompute/scan.py:257), not Delphi's quantiles. {n_neg} negatives from the shared "
            f"_random256 pool restricted to a stored per-feature maximum of exactly 0 (Delphi's "
            f"`non_activating_source = random`). Any candidate positive OVERLAPPING a window shown "
            f"to any explainer is excluded -- the stored bands and the stored top-128 are "
            f"independent reservoirs and DO collide, which the design had assumed away."
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
