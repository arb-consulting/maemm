"""Product `gcg`: discrete-token search on the scorer's own objective -- the reachability ceiling.

    <root>/base/<base>/gcg/<set>/<arm>/  finals.jsonl, trajectory.jsonl, top64.jsonl, summary.json

Trimmed from `eval/gcg_search.py` of an earlier version of this codebase (1288 lines). What
survived is the search itself; what went is everything that belonged to that version's own direction
cache and rescore chain -- the `lens` and `maem` init arms, `--seq-len-mode rollout`, the
concordance probe, the centred/derangement columns, the `samples.jsonl` dump, and every import from
`rescore_metrics` / `sentence_start`. The two things worth re-implementing from those modules are
re-implemented here: the 95-token bound (`MAX_REENC_TOK`, taken from `common.SCORE_MAX_LENGTH` so
there is one constant) and `enc_trunc`. Centring is NOT re-implemented: the objective is the
UNCENTRED cosine, the same number `common.score_ids` produces everywhere else in this pipeline.

THE OBJECTIVE, per candidate string x of T ids and unit direction d:

    cos(x)      = max over KEPT positions t of cos(unit(h_t), d)     [read layer, sink excluded]
    L_lambda(x) = cos(x) - lambda * nll(x)

`cos` goes through `common.score_ids` -- the SAME function `score.py` scores rollouts with and
`repo_examples` scores the SAE repo's windows with, reached without a tokenizer because the search
optimises IDS. So the loop's number, the finals' number and the rollouts' number are one number by
construction. `nll` is the mean per-token NLL of the string's own ids under the clean base (teacher
forcing, no prompt, no sink, predict ids[1:] from ids[:-1]) through a hand fp32 lm_head, self-
checked against `model(...).logits` on the first batch of the process.

ARMS. `<mode>-<init>`, one output directory each:

  gcg-random32   pop 1, lambda 0            -- the pure reachability ceiling from random ASCII
  gcg-corpus     pop 1, lambda 0            -- ... started from the best corpus window we retrieved
  epo-random32   pop 3, lambda 0.1/0.19/0.37 -- the fluency-penalised Pareto front, random start
  epo-corpus     pop 3, same grid           -- ... started from the corpus window

GCG IS EPO AT pop=1, lambda=0, so there is one loop. Each EPO member holds its own lambda and is
selected by its own `L_lambda`, so the per-member finals trace a Pareto front in one run at no extra
forward. All pop members of one run start from the SAME init (as in the fork) except `random32`,
which draws one string per member.

CHUNKS (2026-09-23). A 64-direction 27B `epo` arm is ~15.5 GPU-h and does not fit one container, so
it is cut into `--rows` chunks that run in parallel. `gcg_dir` is `gcg/<set>/<family>/<arm>` with NO
`--rows` component, so every chunk of one arm writes into ONE directory -- which is safe exactly
because that write is M0a's ADDITIVE one (`common.OutDir`, keep_existing, 2026-09-23): each call
stages in a temp directory unique to its process and moves in only its own files, so nothing a
sibling chunk wrote is copied, renamed over or removed. What makes the files disjoint is their
NAMES: a call given `--rows` writes `finals__rows<spec>.jsonl`, `trajectory__rows<spec>.jsonl`,
`top64__rows<spec>.jsonl` and `summary__rows<spec>.json` through `common.rollout_chunk_stem`, the
same spelling the rollouts chunks use, and `gcg/collect.py` reads the union back as one product. A
run with no `--rows` keeps the bare `finals.jsonl` spelling every product on the volume already has,
and a whole-set file beside chunks of the same arm is a refusal on both sides.

A RE-RUN NEVER DESTROYS A PARTIAL (2026-09-23). An additive call's staging directory is
`<arm>.tmp-<date>-<pid>-<hex>`, so a retry can no longer delete the temp its predecessor streamed
into (the one-shot path's `<arm>.tmp-<date>` collision, `common.py:2712-2714`, is what the old
plan's `<cmd> && break` retry loop walked into). `find_partial` then makes the retry a RESUME: it
looks for kept staging directories holding this chunk's own streams, repairs a torn last line into
`<arm>.carry-<stamp>`, and carries the whole directions from it. Nothing is ever deleted.
`--resume-from` still names a directory by hand; `--no-auto-resume` turns the automatic half off. A
call whose chunk is already complete returns that product instead of paying for a container.

TWO COSINES ON THE FINALS (2026-09-23). The objective is still the uncentred cosine -- that is what
the loop selects on and what every column but one reports. The PRINTED number of the paper's
discrete-search rows is the CENTRED rescoring of the same final string,
`max_t cos(unit(h_t - mu), unit(act - mu))` with the same mean on both sides, which
`common.score_ids` produces from the same forward when handed `dirs_centred` and `mu`
(`common.py:2141-2154`). So `cos_centred` on a final is a rescoring through the headline scorer,
not a second search, and the two rows are a lower bound on what a centred search would reach. The
mean is `score_mu_spec` below -- one name, M0a's constant.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import time
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

import precompute.common as C

# ---------------------------------------------------------------------------------------------
# constants that are part of the protocol
# ---------------------------------------------------------------------------------------------

# evals/heldout/eval_universal.py:138's truncation, named once in common.py. A string longer than this would
# be scored shorter than it was optimised; `common.score_ids` refuses such a row outright.
MAX_REENC_TOK = C.SCORE_MAX_LENGTH

# The name of M0a's scoring-mean function in precompute/common.py, tried first by `score_mu_spec`.
# M0a HAS LANDED (merged into `evals/pipeline-v3`, 2026-09-23) and `common.score_mu` is what runs;
# the name-based lookup below started working the moment the name existed, with no edit here. The
# fallback is kept as the refusal path for a base with no `whiten_mu` -- see `gcg/selftest.py`'s
# `check_the_scoring_mean_has_exactly_one_source`, which pins the delegation in both directions.
M0A_SCORE_MU_FN = "score_mu"


def score_mu_spec(cfg: dict, base: str):
    """The mean BOTH sides of the centred cosine are centred on, as config spells a mu (a path).

    THE SCORING CONSTANT IS NOT THIS MODULE'S TO CHOOSE. M0a owns it and exposes it as one function
    in `precompute/common.py`; this is the single place gcg spells the name, so adopting M0a's
    function is one line. The fallback is the same constant read straight from config --
    `bases.<base>.whiten_mu`, which plan §1's ownership table names as the scoring constant -- and
    it is deliberately NOT a locally computed mean: a product that cannot find the file stops.
    """
    fn = getattr(C, M0A_SCORE_MU_FN, None)
    if fn is not None:
        return fn(cfg, base)
    spec = cfg["bases"][base].get("whiten_mu")
    assert spec, (
        f"the centred rescoring has no mean: precompute/common.py exposes no {M0A_SCORE_MU_FN}() "
        f"and bases.{base}.whiten_mu is unset. See gcg/gcg.py:score_mu_spec -- this is M3's "
        f"placeholder for M0a's scoring-mean function, not a licence to compute one here."
    )
    return spec

# The candidate alphabet size, PER BASE: the fork's single 94,325 is Qwen3-8B's (vocab 151,669) and
# says nothing about the 27B, whose vocab is 248,077. `None` means "not measured yet": the run
# prints and records the realised count and skips the assert, loudly. Fill the number in afterwards
# so the next run is checked -- every cost number is priced against a pool of this size.
# MEASURED 2026-09-16 on the 8B: 90,909 (151,669 vocab -> 90,939 printable-ASCII -> 90,935
# round-tripping -> 90,909 after 26 special/added ids). The 94,325 below is the FORK's plan figure
# and is 3.6% high; it is kept as the declared expectation, because the assert exists to catch the
# tokenizer moving (a +-10% band does that) and silently re-calibrating it to our own measurement
# would erase the fact that the plan's number was wrong.
# MEASURED 2026-09-16 on the 27B: 126,220 (248,077 vocab -> 126,278 printable-ASCII -> 126,253
# round-tripping -> 126,220 after 33 special/added ids). There was no prior figure for this base --
# the fork only ever ran on the 8B -- so this one IS the measurement, recorded here so the next run
# is checked against it rather than against nothing.
ALPHABET_EXPECT = {"qwen3-8b": 94_325, "qwen36-27b": 126_220}
ALPHABET_REL_TOL = 0.10

# Tolerances for the end-of-direction CHECK block, which re-scores the finals through
# `common.score_ids` twice: once at the loop's OWN --sbatch (same input, same batch geometry, so
# anything above float noise is a bookkeeping bug) and once at `common.SCORE_CHUNK`, the chunk every
# other product scores at. The second is deliberately looser: a bf16 forward's reduction order is
# batch-shape dependent, which is why SCORE_CHUNK is fixed at 32 pipeline-wide in the first place.
COS_TOL_SAME_BATCH = 1e-4  # ADVISORY: printed, not asserted (pop rows re-run where pop*children ran)
COS_TOL_SAME_BATCH_HARD = 1e-2  # the assert: beyond this it is bookkeeping, not arithmetic
COS_TOL_REBATCH = 1e-2

# The fp32 logit tile the NLL pass may materialise at once ([rows, T-1, V] fp32, x3 for log_softmax
# and its exp). 2 GiB leaves room for the model, the fp32 W_E and the fp32 lm_head copy.
NLL_LOGIT_BYTES = 2 << 30

# Retokenisation filtering rejects a large fraction of single-substitution candidates, so the draw
# is oversampled to clear `children` in one round. A run that hits this cap produces fewer
# candidates per iteration than it is charged for, which is why the cap is reported, not absorbed.
MAX_TOPUP_ROUNDS = 6

# Per-member running set of the best DISTINCT candidate strings ever seen, written to top64.jsonl.
TOP_KEEP = 64
# The set is pruned back to TOP_PRUNE_TO whenever it exceeds TOP_PRUNE_AT, so a 150 x 512 run does
# not hold 76,800 id tuples. Pruning never touches the top TOP_PRUNE_TO, so the top 64 is exact.
TOP_PRUNE_AT = 4096
TOP_PRUNE_TO = 1024

# Kept out of summary.json's per-run block: they are already in finals.jsonl in full and would
# multiply the summary's size by the sequence length.
_BULKY_FINAL_KEYS = ("ids", "init_ids", "per_token_cos")

# The three per-direction streams, in the order a direction writes them; `--resume-from` copies
# exactly these out of a kept temp dir.
# The three streams as LOGICAL names. `chunk_files` turns them into the file names one call writes.
STREAMS = ("finals", "trajectory", "top64")


def rows_spec_of(rows: list[int]) -> str:
    """A canonical `--rows` spelling of a selection: "0-3", "4-6,8", "7".

    The chunk's file names are built from THIS, not from the string the caller typed, so two calls
    that select the same rows write the same files whatever spelling they used -- which is what
    makes a re-run find its own partial instead of writing a second copy beside it.
    """
    assert rows, "a row selection is never empty"
    parts, lo, hi = [], rows[0], rows[0]
    for r in rows[1:]:
        if r == hi + 1:
            hi = r
            continue
        parts.append(f"{lo}-{hi}" if lo != hi else f"{lo}")
        lo = hi = r
    parts.append(f"{lo}-{hi}" if lo != hi else f"{lo}")
    return ",".join(parts)


def chunk_files(rows_spec: str) -> dict[str, str]:
    """logical stream -> the file this call writes. `finals.jsonl`, or `finals__rows0-3.jsonl`.

    `common.rollout_chunk_stem` is M0a's spelling of "one `--rows` chunk of one product", used here
    unchanged so a reader of either product learns the convention once. An empty spec is the
    whole-set run and keeps the historical names.
    """
    out = {k: f"{C.rollout_chunk_stem(k, rows_spec)}.jsonl" for k in STREAMS}
    out["summary"] = f"{C.rollout_chunk_stem('summary', rows_spec)}.json"
    return out

MODES = ("gcg", "epo")
INITS = ("random32", "corpus")

# The measured configurations (RUNBOOK section 2g). `--mode gcg` and `--mode epo` are compute-
# matched in TOTAL candidate forwards (150 x 512 = 76,800 against 300 x 255 = 76,500), not per
# iteration. The fork's runs used `--init maem --seq-len-mode rollout`; ours are fixed T=32.
MODE_DEFAULTS = {
    "gcg": {"iters": 150, "pop": 1, "children": 512, "lam_grid": "0"},
    "epo": {"iters": 300, "pop": 3, "children": 85, "lam_grid": "0.1,0.19,0.37"},
}
# Oversample per init arm. 1.5 for random32 is the fork's value (checklist item 65). The corpus
# init started at the fork's conservative 2.2 (sized for its MEASURED 0.547 rejection rate on
# arbitrary strings); MEASURED HERE 2026-09-16 on 8B realact row 0, it rejects **0.005** of
# candidates against random32's 0.035 -- natural text that has been roundtrip-repaired barely
# breaks under a single substitution -- so 2.2 was drawing and decoding twice as many candidates as
# it kept, for nothing. 1.5 with the realised rate reported by every run (`filter_reject_rate`,
# and `topup_capped_iters` if a draw ever falls short) is the measured setting.
INIT_OVERSAMPLE = {"random32": 1.5, "corpus": 1.5}

_WU32: dict[int, object] = {}  # id(model) -> the fp32 [d, V] lm_head, built once
_NLL_CHECKED = [False]  # the NLL self-check fires on the FIRST call of the process, once


def arm_name(mode: str, init: str, suffix: str = "") -> str:
    """`<mode>-<init>`, plus `-<suffix>` when one is given.

    The suffix is what lets a SECOND run over a different row selection live beside the first
    instead of overwriting it -- `sae/gcg-corpus` is rows 0-31 (all density quartile q0, because
    targets.py lays the sae family out stratum-major) and `sae/gcg-corpus-strat` is 8 rows of each
    quartile. It is NOT part of `--arm`, which must stay parseable as `<mode>-<init>`.
    """
    return f"{mode}-{init}" + (f"-{suffix}" if suffix else "")


ARM_SUFFIX_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# ---------------------------------------------------------------------------------------------
# the candidate alphabet
# ---------------------------------------------------------------------------------------------


def ascii_alphabet(tok, base: str):
    """Ids that decode to non-empty printable ASCII and round-trip to themselves as a single id.

    Three filters: (1) `decode([i])` is non-empty and every character is printable ASCII (32..126 --
    `str.isprintable()` accepts non-ASCII printables, so the ASCII test is explicit); (2)
    `encode(decode([i]))` is exactly `[i]`; (3) not a special id and not an added-vocabulary id.
    Filter (3) is what keeps the SINK out of the optimised string: a string containing the sink id
    would be scored against a residual stream `common.score_ids` never builds.
    """
    t0 = time.time()
    n_vocab = len(tok)
    ids = list(range(n_vocab))
    dec = tok.batch_decode([[i] for i in ids])
    cand = [i for i, s in zip(ids, dec, strict=True) if s and all(32 <= ord(c) <= 126 for c in s)]
    # one batched re-encode rather than V single calls
    enc = tok([dec[i] for i in cand], add_special_tokens=False).input_ids
    rt = [i for i, e in zip(cand, enc, strict=True) if len(e) == 1 and e[0] == i]
    bad = set(tok.all_special_ids or []) | set(tok.get_added_vocab().values())
    keep = [i for i in rt if i not in bad]
    n = len(keep)
    print(
        f"[gcg] alphabet {base}: {n_vocab} vocab -> {len(cand)} printable-ASCII -> {len(rt)} "
        f"round-tripping -> {n} after dropping {len(bad)} special/added ids ({time.time() - t0:.1f}s)",
        flush=True,
    )
    want = ALPHABET_EXPECT.get(base)
    if want is None:
        print(
            f"[gcg] NOTE base {base!r} has no measured alphabet expectation: MEASURED {n}. Put it "
            f"in gcg/gcg.py ALPHABET_EXPECT so the next run is checked.",
            flush=True,
        )
    else:
        assert abs(n - want) <= ALPHABET_REL_TOL * want, (
            f"base {base}: alphabet is {n} ids, expected {want} +-{ALPHABET_REL_TOL:.0%} -- the "
            f"tokenizer changed and every cost number moves with it"
        )
    return np.asarray(keep, np.int64), n_vocab


def is_ascii_id(tok, i) -> bool:
    s = tok.decode([int(i)])
    return bool(s) and all(32 <= ord(c) <= 126 for c in s)


def space_prefixed(tok, alpha):
    """The sub-alphabet whose ids decode to ' ' + printable ASCII -- see `init_random`."""
    dec = tok.batch_decode([[int(i)] for i in alpha])
    sp = np.asarray(
        [int(i) for i, s in zip(alpha, dec, strict=True) if s.startswith(" ")], np.int64
    )
    print(f"[gcg] space-prefixed sub-alphabet: {len(sp)} of {len(alpha)} ids", flush=True)
    return sp


def enc_trunc(tok, text):
    """`sentence_start.enc_trunc`, re-implemented: ids of `text` alone, capped at MAX_REENC_TOK.

    add_special_tokens=False and truncation at the same 95 the scorer truncates at, so an init
    string's ids are bounded by the same rule that bounds a rollout's.
    """
    ids = list(tok(text, add_special_tokens=False).input_ids)
    return ids[:MAX_REENC_TOK], len(ids) > MAX_REENC_TOK


# ---------------------------------------------------------------------------------------------
# forward paths
# ---------------------------------------------------------------------------------------------


def input_embeddings(model):
    """The input-embedding module, unwrapping DDP + PEFT exactly as `common.get_layer` does."""
    m = model.module if hasattr(model, "module") else model
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.get_input_embeddings()


def lm_head_fp32(model):
    """`lm_head.weight.T` in fp32, built once per process.

    The NLL pass runs once or twice per search ITERATION, so casting the [V, d] head each time would
    put pure bandwidth into `nll_s` -- a timing this product exists to measure.
    """
    k = id(model)
    if k not in _WU32:
        _WU32[k] = model.get_output_embeddings().weight.detach().float().T.contiguous()
    return _WU32[k]


def read_resid_grad(model, layer, kwargs):
    """`common.read_resid` WITHOUT its `@torch.no_grad()` and taking `inputs_embeds=`.

    Same forward hook on `common.get_layer(model, layer)`'s OUTPUT raising `StopForward` inside the
    hook, so the layers above and the lm_head never run. Autograd is fine across the raise: the
    graph up to `layer` is fully built before the hook fires and the captured tensor holds it.
    Returns h [B, L, d] fp32 WITH grad_fn; the caller owns the backward.
    """
    captured: dict = {}
    handle = C.get_layer(model, layer).register_forward_hook(C.read_layer_hook(captured))
    try:
        model(**kwargs, use_cache=False)
    except C.StopForward:
        pass
    finally:
        handle.remove()
    assert "h" in captured, f"the grad read hook at layer {layer} never fired (wrong layer object?)"
    return captured["h"]


def keep_mask_full(mask):
    """The GRADIENT path's keep rule, and it must be the SCORER's keep rule.

    `common.score_ids` keeps every non-pad, non-sink position and applies NO norm filter (the
    pipeline-wide divergence from eval_universal.py:71,145-147). The fork also dropped tokens above
    10x the row median here; keeping that would let the gradient optimise against a mask the
    selection step does not use. Boolean, no gradient.
    """
    keep = mask.clone()
    keep[:, 0] = False  # the sink is in the forward and is never a candidate token
    return keep


def sae_peak_pos(acts_row, peak):
    """Text-token index of a feature's peak pre-gate activation, or -1 when it never fires.

    `acts_row` is one scored row (column 0 is the sink, column j+1 is text token j; non-kept
    positions are NaN). relu makes a dead feature's whole row 0, and then argmax would name
    position 0 arbitrarily -- a later "peak == cos-argmax" comparison would read that arbitrary 0
    as a genuine disagreement. -1 says the question does not apply.
    """
    if not peak > 0.0:
        return -1
    return int(np.nanargmax(np.asarray(acts_row))) - 1


def shape_sweep(model, tok, ids, d_cpu, read_layer, device, shapes=(1, 8, 32, 128, 256, 512)):
    """Score ONE id list at several batch shapes and return {n_rows: (cos, argmax)}.

    `common.score_ids` right-pads and chunks, so a row is reduced in a tile whose M dimension is the
    batch's, and a bf16 forward's accumulation order goes with it. Checklist item 11 recorded that
    the chunk size shifts a per-row cosine measurably; this measures the shift for one string by
    replicating it n times and reading row 0 back, which reproduces the loop's chunk shape exactly.
    Only ever called when the end-of-direction CHECK has already failed, so it costs nothing in the
    normal path.
    """
    out = {}
    for n in shapes:
        r = exact_cos(model, tok, [list(ids)] * n, d_cpu, read_layer, n, device)
        out[n] = (float(r["cos"][0]), int(r["amax"][0]))
    return out


def exact_cos(
    model, tok, id_lists, d_cpu, read_layer, sbatch, device, want_tokens=False,
    sae=None, feature_id=None, d_centred_cpu=None, mu=None,
):
    """THE scoring path: `max_t cos(unit(h_t), d)` over `common.score_ids`' kept tokens.

    One direction, many id lists -- the direction is broadcast to one row per candidate. This is the
    product's only cosine; there is no faster in-loop variant to drift from, which is what the fork
    needed its end-of-direction check to prove and what is now true by construction.

    `d_centred_cpu` + `mu` additionally read the CENTRED cosine
    `max_t cos(unit(h_t - mu), unit(act - mu))` off the SAME forward -- `common.score_ids` takes the
    pair and adds one einsum (`common.py:2141-2154`). It is RECORDED, never optimised: the loop
    selects on the uncentred number throughout, and the centred one is computed once, on the
    finals, because it is the column the paper prints (module docstring). Passing one of the two
    without the other is refused by `score_ids`, not by a check here, so there is one rule.

    `sae` + `feature_id` additionally read that feature's PRE-GATE activation
    (`common.sae_encode`, relu((h - b_dec) @ W_enc[:, f] + b_enc[f])) off the SAME forward, per
    kept token. It is RECORDED, never optimised: the objective stays the cosine, and the activation
    is there so an `sae` final can be read as "did the string actually make the feature fire, or
    only align with its encoder column?". Only ever called on the pop finals, so it costs nothing.
    """
    import torch

    n = len(id_lists)
    dirs = d_cpu.detach().cpu().float().unsqueeze(0).expand(n, -1)
    dirs_c = (
        None if d_centred_cpu is None
        else d_centred_cpu.detach().cpu().float().unsqueeze(0).expand(n, -1)
    )
    acts = torch.full((n, C.SCORE_WIDTH), float("nan")) if sae is not None else None

    def on_chunk(s, h, _cos, keep, _ids):
        a = C.sae_encode(sae, h, [int(feature_id)])[..., 0]  # [b, t]
        acts[s : s + h.shape[0], : h.shape[1]] = torch.where(
            keep, a, torch.full_like(a, float("nan"))
        ).cpu()

    out = C.score_ids(
        model, tok, id_lists, dirs, read_layer, sbatch=sbatch, device=device,
        on_chunk=None if sae is None else on_chunk,
        dirs_centred=dirs_c, mu=mu,
    )
    best, arg = C.agg(out["cos"], out["keep"])
    res = {
        # column j+1 of the scored row is text token j
        "cos": best.numpy().astype(np.float32),
        "amax": (arg - 1).numpy().astype(np.int32),
    }
    if dirs_c is not None:
        # The centred cosine is reduced the SAME way -- max over kept positions, through common.agg
        # -- and its argmax is recorded separately because the two maxima need not land on the same
        # token: centring moves the ranking, which is the whole reason the printed column is a
        # rescoring of the string rather than a relabelling of the loop's own number.
        best_c, arg_c = C.agg(out["cos_centred"], out["keep"])
        res["cos_centred"] = best_c.numpy().astype(np.float32)
        res["amax_centred"] = (arg_c - 1).numpy().astype(np.int32)
    if want_tokens:
        cos = out["cos"]
        keep = out["keep"]
        res["per_token"] = [
            [round(float(v), 6) for v in cos[i][keep[i]].tolist()] for i in range(n)
        ]
        # the read-layer residual norm AT the argmax, the cosine's denominator: a small norm is
        # what turns a bf16 reduction-order difference into a large cosine difference
        res["norm_at_argmax"] = np.asarray(
            [float(out["norm"][i, int(arg[i])]) for i in range(n)], np.float32
        )
    if sae is not None:
        keep = out["keep"]
        res["sae_peak"] = np.asarray(
            [float(acts[i][keep[i]].max()) for i in range(n)], np.float32
        )
        # the activation AT the cosine's argmax, which is the token the objective actually selected
        res["sae_at_argmax"] = np.asarray(
            [float(acts[i, int(arg[i])]) for i in range(n)], np.float32
        )
        res["sae_peak_pos"] = np.asarray(
            [sae_peak_pos(acts[i], res["sae_peak"][i]) for i in range(n)], np.int32
        )
    return res


def mean_nll(model, ids_t, want_entropy=False):
    """Mean per-token NLL of each row's own ids under the clean base; optionally the mean entropy.

    Teacher forcing with NO prompt and NO sink: predict ids[:, 1:] from ids[:, :-1]. The transformer
    body is called directly and the head applied by hand in fp32 -- the body's output already
    carries the final norm, so `h @ lm_head.weight.T` IS the model's logit computation, and doing it
    in fp32 avoids a bf16 head's noise. Rows are all the same length (the search works at a fixed
    T), so there is no padding and the attention mask is all ones.

    Unchanged from the fork apart from the print prefix.
    """
    import torch

    n, t_len = ids_t.shape
    assert t_len >= 2, f"NLL needs at least 2 tokens, got {t_len}"
    w_f = lm_head_fp32(model)  # [d, V] fp32
    v = w_f.shape[1]
    per = max(1, int(NLL_LOGIT_BYTES // (3 * 4 * (t_len - 1) * v)))
    nll = np.zeros(n, np.float32)
    ent = np.zeros(n, np.float32)
    checked = _NLL_CHECKED[0]
    with torch.no_grad():
        for s in range(0, n, per):
            ids = ids_t[s : s + per]
            am = torch.ones_like(ids[:, :-1])
            h = model.model(
                input_ids=ids[:, :-1], attention_mask=am, use_cache=False
            ).last_hidden_state
            lg = h.float() @ w_f
            lp = torch.log_softmax(lg, -1)
            tgt = ids[:, 1:]
            tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [b, T-1]
            nll[s : s + len(ids)] = (-tl).mean(1).float().cpu().numpy()
            if want_entropy:
                ent[s : s + len(ids)] = (-(lp.exp() * lp).sum(-1)).mean(1).float().cpu().numpy()
            if not checked:
                # The reference `model(...).logits` leaves a BF16 lm_head whose ulp at |logit| ~ 20
                # is 0.125, while the path above deliberately projects in fp32. The two therefore
                # differ by bf16 rounding and nothing else, so what is asserted is the quantity this
                # function RETURNS, at a tolerance sized to that ulp; the vocabulary-wide worst case
                # is printed, not asserted, because it is dominated by ties the reference cannot
                # resolve. This is also the check that catches `model.model` not being the body.
                ref = model(input_ids=ids[:, :-1], attention_mask=am, use_cache=False).logits.float()
                rlp = torch.log_softmax(ref, -1)
                rt = rlp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                dmean = (rt.mean(1) - tl.mean(1)).abs()
                dall = (rlp - lp).abs().max().item()
                assert dmean.mean().item() < 0.05 and dmean.max().item() < 0.3, (
                    f"the hand fp32 lm_head NLL differs from model.logits by mean "
                    f"{dmean.mean().item():.4f} / max {dmean.max().item():.4f} nats -- more than a "
                    f"bf16 lm_head can explain; `model.model` may not be the transformer body"
                )
                print(
                    f"[gcg] nll path checked against model.logits on {len(ids)} rows: mean-NLL "
                    f"|d| mean {dmean.mean().item():.2e} max {dmean.max().item():.2e} nats | "
                    f"vocab-wide max {dall:.2e} (bf16 ulp)",
                    flush=True,
                )
                checked = _NLL_CHECKED[0] = True
                del ref, rlp, rt
            del lg, lp
    return nll, (ent if want_entropy else None)


def grad_pass(model, dev, w_e32, ids_t, d, lam_t, tau, sink_id, mdtype, want_nll, read_layer):
    """One-hot gradient of the SURROGATE objective at the current members' strings.

    Surrogate, per member m:  tau * logsumexp_t(cos_t / tau)  -  lambda_m * nll_soft_m
    over the same kept t the exact scorer uses. The exact objective is a HARD max, which routes the
    whole gradient through one token of T; the soft max at tau=0.02 spreads it over the near-maximal
    tokens without moving the optimum much. Selection stays hard and exact.

    `nll_soft` is the relaxation of the exact NLL: the targets are the one-hot rows themselves, so
    `-(onehot[:, 1:] * logp).sum(-1)` equals the exact mean NLL at a hard one-hot AND carries
    gradient through the target side, which a gather-based NLL would drop. It uses a bf16 head: it
    feeds a top-k ranking, not a reported number, and every reported NLL comes from `mean_nll`.

    SIGN CONVENTION: the LOSS (`-objective`, GCG's convention) is backwarded, so the returned
    `grad = -onehot.grad` is the predicted INCREASE in the objective from turning position t into
    token v. Higher is better everywhere downstream.
    """
    import torch
    import torch.nn.functional as F

    pop, t_len = ids_t.shape
    v = w_e32.shape[0]
    onehot = torch.zeros(pop, t_len, v, device=dev, dtype=torch.float32)
    onehot.scatter_(2, ids_t.unsqueeze(-1), 1.0)
    onehot.requires_grad_(True)
    emb = onehot @ w_e32  # [pop, T, d] fp32
    emb_b = emb.to(mdtype)
    sink_e = w_e32[sink_id].to(mdtype).view(1, 1, -1).expand(pop, 1, -1)
    inp = torch.cat([sink_e, emb_b], dim=1)  # the sink is IN the forward and NOT optimised
    am = torch.ones(pop, t_len + 1, dtype=torch.long, device=dev)  # (it has no one-hot row)
    h = read_resid_grad(model, read_layer, {"inputs_embeds": inp, "attention_mask": am})
    keep = keep_mask_full(am.bool())
    assert bool(keep.any(1).all()), "a member has no kept token at all -- the mask is wrong"
    c = torch.einsum("btd,d->bt", F.normalize(h, dim=-1), d)
    c = c.masked_fill(~keep, -float("inf"))
    soft = tau * torch.logsumexp(c / tau, dim=1)  # [pop]
    obj = soft
    nll_soft = None
    if want_nll:
        hh = model.model(
            inputs_embeds=emb_b[:, :-1],
            attention_mask=torch.ones(pop, t_len - 1, dtype=torch.long, device=dev),
            use_cache=False,
        ).last_hidden_state
        w_u = model.get_output_embeddings().weight
        lp = torch.log_softmax((hh @ w_u.T).float(), -1)
        nll_soft = -(onehot[:, 1:, :] * lp).sum(-1).mean(1)  # [pop]
        obj = obj - lam_t * nll_soft
    (-obj.sum()).backward()
    grad = -onehot.grad.detach()
    out = (
        grad,
        soft.detach().float().cpu().numpy(),
        None if nll_soft is None else nll_soft.detach().float().cpu().numpy(),
    )
    del onehot, emb, emb_b, inp, h, c, soft, obj, nll_soft
    return out


# ---------------------------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------------------------


def retok_ok(tok, cand):
    """`filter_ids`: a candidate survives only if `decode(ids)` re-encodes to the SAME id list.

    Checked on the FULL assembled sequence, never a substring -- and here the string IS the whole
    input. This is also what makes the finals reproducible: anything downstream that re-tokenizes
    the stored `string` must land on the stored `ids`, or it would score a different segmentation
    than the one the loop optimised.
    """
    texts = tok.batch_decode(cand)
    enc = tok(texts, add_special_tokens=False).input_ids
    ok = [i for i, (e, c) in enumerate(zip(enc, cand, strict=True)) if list(e) == list(c)]
    return ok, texts


def sample_candidates(rng, tok, cur, pool, children, oversample, counters):
    """`children` retokenisation-surviving single-substitution candidates per member.

    `pool` is [pop, T, K] of admissible replacement ids per (member, position) -- the gradient's
    top-k restricted to the ASCII alphabet. Each draw picks ONE random position and one uniform id
    from that position's pool. Oversample, filter, keep the first `children`; top up if short.
    """
    pop, t_len = cur.shape
    out: list[list[list[int]]] = [[] for _ in range(pop)]
    n_draw = max(children, int(math.ceil(children * oversample)))
    rounds = 0
    while rounds < MAX_TOPUP_ROUNDS and any(len(o) < children for o in out):
        need = [max(0, children - len(o)) for o in out]
        k = max(need)
        k = max(1, int(math.ceil(k * oversample))) if rounds else n_draw
        flat, owner = [], []
        for m in range(pop):
            if len(out[m]) >= children:
                continue
            pos = rng.integers(0, t_len, size=k)
            sel = rng.integers(0, pool.shape[2], size=k)
            for j in range(k):
                row = cur[m].copy()
                row[pos[j]] = pool[m, pos[j], sel[j]]
                flat.append(row.tolist())
                owner.append(m)
        if not flat:
            break
        ok, _ = retok_ok(tok, flat)
        counters["drawn"] += len(flat)
        counters["kept"] += len(ok)
        for i in ok:
            m = owner[i]
            if len(out[m]) < children:
                out[m].append(flat[i])
        rounds += 1
    short = [m for m, o in enumerate(out) if len(o) < children]
    if short:
        counters["capped_iters"] += 1
        counters["capped_members"] += len(short)
    return out


# ---------------------------------------------------------------------------------------------
# initialisation arms
# ---------------------------------------------------------------------------------------------


def init_random(rng, alpha, pop, t_len, tok, alpha_sp, tries=200, counters=None):
    """`t_len` random ASCII ids per member, drawn so that the STRING ROUND-TRIPS.

    MEASURED by the fork, and it is why this is not one line: a string of T ids drawn uniformly from
    the ASCII alphabet almost never survives `decode -> encode -> the same ids`, the in-loop
    retokenisation filter then rejects ~0.9-1.0 of candidates, and the loop reports its init as a
    result. The fix is (1) draw from `alpha_sp`, the SPACE-PREFIXED sub-alphabet -- BPE merges
    almost never cross a space boundary in ASCII, so concatenating such tokens is stable by
    construction -- and (2) verify anyway, redrawing the offending member. Neither part changes what
    the search may REACH: substitutions are still drawn from the full alphabet and the filter still
    adjudicates every candidate.
    """
    src = alpha_sp if alpha_sp is not None and len(alpha_sp) >= 256 else alpha
    out, redraws = [], 0
    for _ in range(pop):
        for _k in range(tries):
            ids = src[rng.integers(0, len(src), size=t_len)].astype(np.int64).tolist()
            ok, _ = retok_ok(tok, [ids])
            if ok:
                break
            redraws += 1
        else:
            raise AssertionError(
                f"--init random32: no round-tripping draw of {t_len} ids in {tries} tries -- the "
                f"space-prefixed sub-alphabet ({len(src)} ids) is not doing its job on this "
                f"tokenizer"
            )
        out.append(ids)
    if counters is not None:
        counters["init_redraws"] += redraws
    return np.asarray(out, np.int64)


def roundtrip_repair(tok, ids, alpha_sp, rng, counters=None, tries=64):
    """Make an init string round-trip, changing as few tokens as possible.

    MEASURED FAILURE THIS EXISTS FOR (the fork, 2026-09-10): an init built by CUTTING text at an
    arbitrary token boundary does not always survive `decode -> encode -> the same ids`, `retok_ok`
    then rejects 100% of candidates for the whole run, 0 candidate forwards happen, and the init's
    own score is written out as if it were a search result. The corpus init cuts a window, so it is
    exactly that case.

    The repair walks to the FIRST position where re-encoding diverges and replaces that token with a
    space-prefixed id: that breaks the merge which caused the divergence while leaving every other
    token -- in particular the tail, where the metric's argmax lives -- alone. It iterates because
    one repair can expose another.
    """
    ids = [int(x) for x in ids]
    n_fix = 0
    for _ in range(tries):
        ok, _t = retok_ok(tok, [ids])
        if ok:
            if counters is not None:
                counters["init_repairs"] += n_fix
            return np.asarray(ids, np.int64), n_fix
        enc = list(tok(tok.decode(ids), add_special_tokens=False).input_ids)
        m = min(len(enc), len(ids))
        j = next((i for i in range(m) if enc[i] != ids[i]), max(0, m - 1))
        j = max(0, min(j, len(ids) - 1))
        ids[j] = int(alpha_sp[rng.integers(0, len(alpha_sp))])
        n_fix += 1
    raise AssertionError(
        f"could not make the init round-trip in {tries} single-token repairs -- every candidate "
        f"would be rejected by the retokenisation filter and the search would silently do nothing"
    )


def init_corpus(tok, rng, alpha_sp, scan_top, toks, docs, row, t_len, counters):
    """The best CORPUS WINDOW we retrieved for this direction, cut to `t_len` around its argmax.

    The scan (`precompute/scan.py`) already searched the whole corpus for each held-out direction
    and stored, per corpus size, the top-64 windows as `[doc, window start, argmax WITHIN the
    window, cos]`. This init takes the top-1 at the LARGEST size -- the strongest natural text this
    project has for the direction -- reads its ids straight out of `corpus/tokens.i32`, and keeps
    the `t_len` tokens ENDING at `max(argmax, t_len - 1)`, i.e. the tail of the window that still
    contains the token the direction actually fires on.

    RESOLVED AGAINST THE BRIEF: the scan's windows are 64/16 (`common.windows_of`), but a document
    of <= 64 tokens is ONE window of its whole length, so a top-1 window can be SHORTER than
    `t_len`. Front-padding it would reintroduce exactly the confound the fork's `--seq-len-mode
    rollout` was invented to remove ("the search improves the inversion" vs "the search prepends
    helpful context"). Instead the top-k list is walked down to the highest-ranked window with at
    least `t_len` tokens, and that rank is recorded on every output row.

    Returns (ids [t_len], info) where `info` carries the provenance and the window's OWN scan
    cosine -- the PRE-REPAIR number, over the whole (up to 64-token) window, which is not the same
    quantity as the post-repair `init_cos` the loop measures on the cut. Both are in `finals.jsonl`.
    """
    entries = scan_top[row]
    size = max(entries)
    top = entries[size]
    assert top, f"target row {row}: the scan's top-k at corpus size {size}M is empty"
    chosen = None
    for rank, (doc, start, argmax, wcos) in enumerate(top):
        rec = docs[doc]
        assert int(rec["doc"]) == int(doc), (
            f"corpus docs.jsonl is not self-indexed: row {doc} carries doc={rec['doc']}"
        )
        lens = [ln for (s, ln) in C.windows_of(int(rec["len"])) if s == int(start)]
        assert len(lens) == 1, (
            f"target row {row}: the scan names a window at start {start} of doc {doc} "
            f"({rec['len']} tokens) that common.windows_of does not produce ({len(lens)} matches) "
            f"-- the scan and this init disagree on the 64/16 geometry"
        )
        wlen = lens[0]
        assert 0 <= int(argmax) < wlen, (
            f"target row {row}: scan argmax {argmax} outside its own window of {wlen} tokens"
        )
        if wlen >= t_len:
            chosen = (rank, int(doc), int(start), int(argmax), float(wcos), wlen)
            break
        counters["init_short_windows"] += 1
    assert chosen is not None, (
        f"target row {row}: not one of the {len(top)} top-k windows at size {size}M has the "
        f"{t_len} tokens the search needs (the shortest corpus documents are single windows)"
    )
    rank, doc, start, argmax, wcos, wlen = chosen
    off = int(docs[doc]["offset"])
    window = [int(x) for x in toks[off + start : off + start + wlen]]
    assert len(window) == wlen, f"corpus read gave {len(window)} ids for a {wlen}-token window"
    end = max(argmax, t_len - 1)
    cut = window[end - t_len + 1 : end + 1]
    assert len(cut) == t_len, f"the cut is {len(cut)} ids, expected {t_len}"
    ids, n_fix = roundtrip_repair(tok, cut, alpha_sp, rng, counters)
    text = tok.decode(ids.tolist())
    re_ids, over = enc_trunc(tok, text)
    assert not over and re_ids == ids.tolist(), (
        f"target row {row}: the repaired corpus init does not re-encode to its own ids "
        f"({len(re_ids)} ids back, truncated={over}) -- roundtrip_repair did not do its job"
    )
    info = {
        "init_topk_rank": rank,
        "init_corpus_size_m": size,
        "init_doc": doc,
        "init_window_start": start,
        "init_window_len": wlen,
        "init_window_argmax": argmax,
        # the scan's cosine of the WHOLE window (up to 64 tokens), before the cut and the repair
        "init_window_cos": round(wcos, 6),
        "init_cut_end": end,
        "init_repairs": n_fix,
    }
    return ids, info


# ---------------------------------------------------------------------------------------------
# per-string reporting metrics
# ---------------------------------------------------------------------------------------------


def distinct_n(ids, n):
    if len(ids) < n:
        return None
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / float(len(grams))


def _sync(dev):
    import torch

    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


def parse_lam_grid(spec, pop):
    """"lo:hi" -> `pop` log-spaced lambdas; "a,b,c" -> that list (length pop, or 1 broadcast)."""
    spec = spec.strip()
    if ":" in spec:
        lo, hi = (float(x) for x in spec.split(":", 1))
        assert lo > 0 and hi > 0, f"a log-uniform grid needs positive endpoints, got {spec}"
        if pop == 1:
            return [lo]
        # float(), not the numpy scalars linspace hands back: they reach summary.json and
        # json.dump refuses numpy.float64 -- which would discard a finished run at its last line.
        return [float(x) for x in np.exp(np.linspace(math.log(lo), math.log(hi), pop))]
    vals = [float(x) for x in spec.split(",") if x.strip()]
    assert vals, f"--lam-grid {spec!r} parses to nothing"
    if len(vals) == 1:
        return vals * pop
    assert len(vals) == pop, f"--lam-grid has {len(vals)} values but --pop is {pop}"
    return vals


# ---------------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------------


class _TopSet:
    """Per-member running set of the best DISTINCT candidate strings, keyed by their ids.

    Pruned back to TOP_PRUNE_TO whenever it passes TOP_PRUNE_AT, so a 150 x 512 run holds ~1k id
    tuples instead of 76,800. Pruning never removes anything inside the top TOP_PRUNE_TO, so the
    top TOP_KEEP this reports is exact.
    """

    def __init__(self):
        self.best: dict[tuple, float] = {}

    def push(self, ids, cos):
        key = tuple(int(x) for x in ids)
        prev = self.best.get(key)
        if prev is None or cos > prev:
            self.best[key] = float(cos)
        if len(self.best) > TOP_PRUNE_AT:
            self.prune(TOP_PRUNE_TO)

    def prune(self, k):
        top = sorted(self.best.items(), key=lambda kv: -kv[1])[:k]
        self.best = dict(top)

    def top(self, k):
        return sorted(self.best.items(), key=lambda kv: -kv[1])[:k]


def run_direction(
    a, model, tok, dev, alpha, alpha_t, w_e32, d_cpu, row, fam, lams, rng, arm,
    sink_id, mdtype, read_layer, alpha_sp, init_ctx, span_text, sae_ctx=None, family_row=None,
    d_centred_cpu=None, mu_vec=None,
):
    """One direction, one arm. Returns (per-member finals, top64 rows, trajectory rows, timings).

    `d_centred_cpu` + `mu_vec` are the centred rescoring's two halves; when both are given every
    final also carries `cos_centred`. They reach ONE call -- the end-of-direction `fresh` one --
    and no part of the loop sees them.
    """
    import torch

    pop = a["pop"]
    t_len = a["seq_len"]
    d = d_cpu.to(dev)
    want_nll = bool(np.any(np.asarray(lams) > 0))
    lam_t = torch.as_tensor(lams, dtype=torch.float32, device=dev)
    tm = {"grad_s": 0.0, "cand_s": 0.0, "fwd_s": 0.0, "nll_s": 0.0, "misc_s": 0.0}
    counters = {
        "drawn": 0, "kept": 0, "capped_iters": 0, "capped_members": 0,
        "init_redraws": 0, "init_repairs": 0, "init_short_windows": 0,
    }
    t_dir = time.time()

    # ---- init ---------------------------------------------------------------------------------
    init_info: dict = {}
    if a["init"] == "random32":
        cur = init_random(rng, alpha, pop, t_len, tok, alpha_sp, counters=counters)
    elif a["init"] == "corpus":
        one, init_info = init_corpus(
            tok, rng, alpha_sp, init_ctx["scan_top"], init_ctx["toks"], init_ctx["docs"],
            row, t_len, counters,
        )
        # All pop members start from the SAME string (the fork's rule for a shared init): with
        # distinct lambdas they select differently from iteration 1 and diverge on their own.
        cur = np.stack([one.copy() for _ in range(pop)])
    else:
        raise ValueError(f"unknown --init {a['init']!r}, want one of {list(INITS)}")
    assert cur.ndim == 2 and cur.shape == (pop, t_len), (
        f"the init returned {cur.shape}, expected ({pop}, {t_len})"
    )
    assert 2 <= t_len <= MAX_REENC_TOK, (
        f"T={t_len} is outside [2, {MAX_REENC_TOK}] -- the NLL needs a context token and the "
        f"scorer truncates at {MAX_REENC_TOK}"
    )
    init_ids_l = [cur[m].tolist() for m in range(pop)]
    init_texts = [tok.decode(x) for x in init_ids_l]

    _sync(dev)
    t0 = time.time()
    ex = exact_cos(model, tok, cur.tolist(), d_cpu, read_layer, a["sbatch"], dev)
    cur_cos = ex["cos"].astype(np.float64)
    init_cos = cur_cos.copy()
    # The init's own NLL is measured whether or not the objective uses one: it costs a single
    # pop-row forward and it is the only way to say what a fluency-penalised arm's fluency DID.
    init_nll = mean_nll(model, torch.as_tensor(cur, device=dev))[0].astype(np.float64)
    cur_nll = init_nll.copy() if want_nll else np.zeros(pop)
    _sync(dev)
    tm["misc_s"] += time.time() - t0
    cur_l = cur_cos - np.asarray(lams) * cur_nll
    l_at_restart = cur_l.copy()

    tops = [_TopSet() for _ in range(pop)]
    for m in range(pop):
        tops[m].push(init_ids_l[m], cur_cos[m])

    traj: list[dict] = []
    n_cand = 0
    last_soft = np.full(pop, float("nan"))
    t_loop = time.time()
    for it in range(a["iters"]):
        # (a) gradient
        t0 = time.time()
        grad, last_soft, _ns = grad_pass(
            model, dev, w_e32, torch.as_tensor(cur, device=dev), d, lam_t, a["tau"], sink_id,
            mdtype, want_nll, read_layer,
        )
        _sync(dev)
        tm["grad_s"] += time.time() - t0

        # (b) candidates
        t0 = time.time()
        ga = grad.index_select(2, alpha_t)  # [pop, T, |A|]
        top = ga.topk(min(a["topk"], ga.shape[2]), dim=-1).indices
        pool = alpha_t[top].cpu().numpy()  # [pop, T, topk] real ids
        del ga, top, grad
        cands = sample_candidates(
            rng, tok, cur, pool, a["children"], a["filter_oversample"], counters
        )
        _sync(dev)
        tm["cand_s"] += time.time() - t0
        # A no-op must be an ERROR, not a result: if not one candidate survived the retokenisation
        # filter the search cannot move and every later iteration inherits the same string.
        assert any(cl for cl in cands), (
            f"{fam}:{row} iteration {it}: NOT ONE of the {pop * a['children']} candidates survived "
            f"the retokenisation filter ({counters['drawn']} drawn, {counters['kept']} kept so "
            f"far). The current string does not round-trip, so no single substitution of it can; "
            f"the search would sit still for the rest of the run and report its init as a result. "
            f"Init arm {a['init']!r}."
        )

        # (c) selection -- ONE exact pass over every member's candidates
        flat, owner = [], []
        for m, cl in enumerate(cands):
            flat.extend(cl)
            owner.extend([m] * len(cl))
        t0 = time.time()
        ce = exact_cos(model, tok, flat, d_cpu, read_layer, a["sbatch"], dev)
        _sync(dev)
        tm["fwd_s"] += time.time() - t0
        n_cand += len(flat)
        cc = ce["cos"].astype(np.float64)
        nn = np.zeros(len(flat))
        if want_nll:
            t0 = time.time()
            nn = mean_nll(
                model, torch.as_tensor(np.asarray(flat, np.int64), device=dev)
            )[0].astype(np.float64)
            _sync(dev)
            tm["nll_s"] += time.time() - t0
        own = np.asarray(owner)
        for m in range(pop):
            sel = np.flatnonzero(own == m)
            if not len(sel):
                continue
            for j in sel:
                tops[m].push(flat[j], cc[j])
            l_m = cc[sel] - lams[m] * nn[sel]
            j = int(sel[int(np.argmax(l_m))])
            if cc[j] - lams[m] * nn[j] > cur_l[m]:
                cur[m] = np.asarray(flat[j], np.int64)
                cur_cos[m], cur_nll[m] = cc[j], nn[j]
                cur_l[m] = cc[j] - lams[m] * nn[j]

        # (d) restart the STALLED member -- the one making the least PROGRESS in its OWN objective
        # since the last restart (a lambda-invariant reading; "the worst L" would restart the
        # highest-lambda member forever), ties broken by the lowest cosine. Off by default.
        if a["restart_every"] and it and it % a["restart_every"] == 0 and pop > 1:
            prog = cur_l - l_at_restart
            m = int(np.lexsort((cur_cos, prog))[0])
            cur[m] = init_random(rng, alpha, 1, t_len, tok, alpha_sp, counters=counters)[0]
            t0 = time.time()
            cur_cos[m] = exact_cos(
                model, tok, [cur[m].tolist()], d_cpu, read_layer, a["sbatch"], dev
            )["cos"][0]
            if want_nll:
                cur_nll[m] = mean_nll(model, torch.as_tensor(cur[m : m + 1], device=dev))[0][0]
            _sync(dev)
            tm["misc_s"] += time.time() - t0
            cur_l[m] = cur_cos[m] - lams[m] * cur_nll[m]
            l_at_restart = cur_l.copy()

        reject = 1.0 - counters["kept"] / max(1, counters["drawn"])
        if it % a["log_every"] == 0 or it == a["iters"] - 1:
            el = time.time() - t_loop
            for m in range(pop):
                traj.append({
                    "row": int(row), "family": fam, "arm": arm, "member": m, "lam": lams[m],
                    "iter": it,
                    "cos": round(float(cur_cos[m]), 6),
                    # tau*logsumexp at the START of this iteration, against the hard max in `cos`:
                    # their gap is the only calibration signal --tau has and it costs nothing
                    "cos_soft": (
                        None if not np.isfinite(last_soft[m]) else round(float(last_soft[m]), 6)
                    ),
                    "nll": (round(float(cur_nll[m]), 6) if want_nll else None),
                    "L": round(float(cur_l[m]), 6),
                    "filter_reject_rate": round(reject, 6),
                    "elapsed_s": round(el, 2),
                    "cand_forwards": n_cand,
                })
            rate = n_cand / max(1e-9, el)
            eta = (a["iters"] - it - 1) * (el / (it + 1)) / 60.0
            print(
                f"[gcg] {fam}:{row} it {it:4d}/{a['iters']} best cos {float(cur_cos.max()):.4f} "
                f"| {n_cand} cands, {rate:.0f} cand/s, reject {reject:.3f}, eta {eta:.1f} min/dir "
                f"(grad {tm['grad_s']:.0f}s cand {tm['cand_s']:.0f}s fwd {tm['fwd_s']:.0f}s "
                f"nll {tm['nll_s']:.0f}s)",
                flush=True,
            )

    loop_s = time.time() - t_loop

    # ---- finals + the CHECK block ---------------------------------------------------------------
    ids_l = [cur[m].tolist() for m in range(pop)]
    t0 = time.time()
    sae = sae_ctx["sae"] if sae_ctx else None
    feat = sae_ctx["feature_id"] if sae_ctx else None
    fresh = exact_cos(
        model, tok, ids_l, d_cpu, read_layer, a["sbatch"], dev, want_tokens=True,
        sae=sae, feature_id=feat, d_centred_cpu=d_centred_cpu, mu=mu_vec,
    )
    rebatch = exact_cos(model, tok, ids_l, d_cpu, read_layer, C.SCORE_CHUNK, dev)
    nl, ent = mean_nll(model, torch.as_tensor(cur, device=dev), want_entropy=True)
    _sync(dev)
    tm["misc_s"] += time.time() - t0

    assert (fresh["amax"] >= 0).all(), "a final string has no kept token"
    d_same = float(np.abs(fresh["cos"] - cur_cos).max())
    d_re = float(np.abs(rebatch["cos"] - cur_cos).max())
    # Evidence for classifying a failure. `cos` is a MAX OVER POSITIONS, which is not Lipschitz in
    # the per-token values: when the top two positions are within the forward's own bf16 noise, the
    # argmax flips between them and the reported cos moves by the whole gap even though no per-token
    # value moved by more than ~1e-5. A delta at or below the top1-top2 gap is that flip; a delta
    # well ABOVE every gap is bookkeeping. Cheap: `fresh` already carries the per-token cosines.
    worst = int(np.argmax(np.abs(fresh["cos"] - cur_cos)))
    pt = sorted(fresh["per_token"][worst], reverse=True)
    gap = float(pt[0] - pt[1]) if len(pt) > 1 else float("inf")
    sweep = ""
    if d_same >= COS_TOL_SAME_BATCH_HARD:
        sw = shape_sweep(model, tok, ids_l[worst], d_cpu, read_layer, dev)
        lo = min(c for c, _ in sw.values())
        hi = max(c for c, _ in sw.values())
        sweep = (
            " | batch-shape sweep of this exact string: "
            + ", ".join(f"n={n}: {c:.6f}@{p}" for n, (c, p) in sw.items())
            + f" (spread {hi - lo:.2e})"
        )
    assert d_same < COS_TOL_SAME_BATCH_HARD, (
        f"{fam}:{row}: the loop's own cos does not reproduce a fresh common.score_ids call "
        f"(max |d| {d_same:.2e} > {COS_TOL_SAME_BATCH_HARD:.0e}) on member {worst}: loop "
        f"{float(cur_cos[worst]):.6f} vs fresh {float(fresh['cos'][worst]):.6f} at position "
        f"{int(fresh['amax'][worst])}, top1-top2 per-token gap {gap:.2e}, residual norm there "
        f"{float(fresh['norm_at_argmax'][worst]):.3f}. NOTE the loop scored this "
        f"string inside a {a['pop'] * a['children']}-candidate batch and the fresh call scores "
        f"{pop} row(s), so the batch SHAPE differs and a bf16 forward's reduction order with it; "
        f"a delta at or below the gap is an argmax flip between two near-tied positions, a delta "
        f"above every gap is a bookkeeping bug in the loop" + sweep
    )
    if d_same >= COS_TOL_SAME_BATCH:
        print(
            f"[gcg] {fam}:{row} NOTE loop-vs-fresh cos delta {d_same:.2e} exceeds the advisory "
            f"{COS_TOL_SAME_BATCH:.0e} (batch-geometry noise; hard bound "
            f"{COS_TOL_SAME_BATCH_HARD:.0e})",
            flush=True,
        )
    assert d_re < COS_TOL_REBATCH, (
        f"{fam}:{row}: the loop's cos vs common.score_ids at SCORE_CHUNK={C.SCORE_CHUNK} differs "
        f"by {d_re:.2e} > {COS_TOL_REBATCH:.0e} -- more than a bf16 forward's batch-shape-dependent "
        f"reduction order can explain, so the pipeline's own chunk would report a different number"
    )
    print(
        f"[gcg] {fam}:{row} CHECK loop cos vs score_ids @sbatch {a['sbatch']} max |d| "
        f"{d_same:.2e} | @SCORE_CHUNK {C.SCORE_CHUNK} max |d| {d_re:.2e}",
        flush=True,
    )

    # ---- top-64 distinct, NLL'd in one batch ----------------------------------------------------
    top_rows: list[dict] = []
    t0 = time.time()
    for m in range(pop):
        entries = tops[m].top(TOP_KEEP)
        assert entries, f"{fam}:{row} member {m}: the top set is empty"
        ids_batch = np.asarray([list(k) for k, _v in entries], np.int64)
        nll_batch = mean_nll(model, torch.as_tensor(ids_batch, device=dev))[0]
        for rank, ((key, cos), nllv) in enumerate(zip(entries, nll_batch, strict=True)):
            top_rows.append({
                "row": int(row), "family": fam, "arm": arm, "member": m, "lam": lams[m],
                "rank": rank, "string": tok.decode(list(key)), "ids": list(key),
                "cos": round(float(cos), 6), "nll": round(float(nllv), 6),
            })
        tops[m].prune(TOP_KEEP)
    _sync(dev)
    tm["misc_s"] += time.time() - t0

    wall_s = time.time() - t_dir
    finals = []
    for m in range(pop):
        ids = ids_l[m]
        finals.append({
            "row": int(row), "family": fam, "arm": arm, "mode": a["mode"], "member": m,
            "lam": lams[m], "init": a["init"], "seq_len": t_len, "iters": a["iters"],
            "string": tok.decode(ids), "ids": ids,
            "cos": round(float(fresh["cos"][m]), 6),
            # THE PRINTED COLUMN of the paper's discrete-search rows: the same final string
            # rescored under the headline centring, from the same forward as `cos`. None when the
            # family is not centrable -- see `_load_targets`.
            "cos_centred": (
                None if "cos_centred" not in fresh
                else round(float(fresh["cos_centred"][m]), 6)
            ),
            "argmax_centred": (
                None if "amax_centred" not in fresh else int(fresh["amax_centred"][m])
            ),
            "cos_loop": round(float(cur_cos[m]), 6),
            "cos_rebatch": round(float(rebatch["cos"][m]), 6),
            "nll": round(float(nl[m]), 6),
            "ppl": round(float(math.exp(min(50.0, nl[m]))), 3),
            "per_token_cos": fresh["per_token"][m],
            "argmax": int(fresh["amax"][m]),
            "argmax_tok": tok.decode([ids[int(fresh["amax"][m])]]),
            "init_cos": round(float(init_cos[m]), 6),
            "init_nll": round(float(init_nll[m]), 6),
            "init_string": init_texts[m],
            "init_ids": init_ids_l[m],
            "token_entropy": round(float(ent[m]), 6),
            "nonascii_frac": round(float(np.mean([not is_ascii_id(tok, i) for i in ids])), 4),
            "distinct2": distinct_n(ids, 2),
            "distinct3": distinct_n(ids, 3),
            "cand_forwards": n_cand,
            "wall_s": round(wall_s, 2),
            "target_span_text": span_text,
            "family_row": family_row,
            **(
                {}
                if sae_ctx is None
                else {
                    # RECORDED, NOT OPTIMISED: the objective is the cosine to the encoder column.
                    # These say whether the string the search found also makes the feature FIRE.
                    "sae_feature": int(feat),
                    "sae_threshold": round(float(sae.threshold), 6),
                    "sae_act_at_argmax": round(float(fresh["sae_at_argmax"][m]), 6),
                    "sae_peak_act": round(float(fresh["sae_peak"][m]), 6),
                    "sae_peak_pos": int(fresh["sae_peak_pos"][m]),
                    "sae_fired": bool(fresh["sae_peak"][m] > sae.threshold),
                }
            ),
            **init_info,
        })

    tm["gpu_seconds"] = sum(tm[k] for k in ("grad_s", "cand_s", "fwd_s", "nll_s", "misc_s"))
    tm["loop_s"] = loop_s
    tm["wall_s"] = wall_s
    tm["iters"] = a["iters"]
    tm["cand_forwards"] = n_cand
    tm["cand_per_s_fwd"] = n_cand / max(1e-9, tm["fwd_s"])
    tm["cand_per_s_total"] = n_cand / max(1e-9, tm["gpu_seconds"])
    tm["filter_drawn"] = counters["drawn"]
    tm["filter_kept"] = counters["kept"]
    tm["filter_reject_rate"] = 1.0 - counters["kept"] / max(1, counters["drawn"])
    tm["topup_capped_iters"] = counters["capped_iters"]
    tm["topup_capped_members"] = counters["capped_members"]
    tm["init_redraws"] = counters["init_redraws"]
    tm["init_repairs"] = counters["init_repairs"]
    tm["init_short_windows"] = counters["init_short_windows"]
    tm["cos_check_same_batch"] = d_same
    tm["cos_check_rebatch"] = d_re
    tm["distinct_candidates_kept"] = int(sum(len(t.best) for t in tops))
    # The per-iteration assert above already raises on the FIRST iteration that produces no
    # candidate, so this cannot fire behind it. It is here because `filter_reject_rate == 1.0` with
    # `cand_forwards == 0` is the exact signature of the defect that silently produced a bogus
    # result in the fork, and a run must not be able to WRITE that signature to disk.
    assert tm["cand_forwards"] > 0 and tm["filter_reject_rate"] < 1.0, (
        f"{fam}:{row}: {tm['cand_forwards']} candidate forwards at rejection rate "
        f"{tm['filter_reject_rate']:.3f} -- the search did not move and its init's own score would "
        f"be reported as a result (init arm {a['init']!r}, T={t_len})"
    )
    return finals, top_rows, traj, tm


# ---------------------------------------------------------------------------------------------
# the product
# ---------------------------------------------------------------------------------------------


def read_jsonl_tolerant(path: Path, what: str) -> list[dict]:
    """Every COMPLETE json line of a STREAMED file. A torn last line is dropped, loudly.

    `C.read_jsonl` is right for a committed product and wrong here: these files are read out of a
    staging directory a container died in, and the one thing a kill mid-`write` leaves is a
    truncated final line. Dropping it costs the direction it belonged to, which is re-run; refusing
    to parse it would cost every direction before it, which is the loss this path exists to avoid.
    A torn line anywhere but at the end is a different failure and is NOT tolerated.
    """
    rows, text = [], path.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            assert i == len(lines) - 1, (
                f"{path}: line {i + 1} of {len(lines)} is not json, and it is not the last line. "
                f"A streamed file is torn only at its end; this one is damaged in the middle and "
                f"is not carried."
            )
            print(f"[gcg] {what}: dropping a torn last line of {path.name}", flush=True)
    return rows


def find_partial(out_dir: str, files: dict[str, str], pop: int) -> str:
    """A kept staging directory holding whole directions of THIS chunk -> a repaired carry dir.

    WHY THIS EXISTS. The launcher retries a failed chunk by re-running the same command. Under the
    additive write each attempt stages in `<arm>.tmp-<date>-<pid>-<hex>`, so the predecessor's
    partial is no longer deleted -- but nothing reads it either, and a 4-direction EPO chunk that
    died on direction 4 would pay for the first three again. This finds it.

    Returns the path of a NEW `<arm>.carry-<stamp>` directory holding only whole directions (all
    `pop` members in finals, and the same rows in all three streams), under the file names this
    chunk resumes from, or "" when there is nothing to carry. The staging directory it read is
    LEFT WHERE IT IS: this path never deletes a partial, it only copies out of one.
    """
    out = Path(out_dir)
    cands = sorted(out.parent.glob(f"{out.name}.tmp-*")) if out.parent.exists() else []
    best: tuple[int, dict[str, list[dict]], Path] | None = None
    for d in cands:
        if not d.is_dir() or not (d / files["finals"]).is_file():
            continue
        streams = {}
        try:
            for k in STREAMS:
                streams[k] = read_jsonl_tolerant(d / files[k], f"carry from {d.name}") if (
                    d / files[k]).is_file() else []
        except (AssertionError, OSError) as e:
            print(f"[gcg] carry: skipping {d.name}: {e}", flush=True)
            continue
        whole = {
            r for r in {int(f["row"]) for f in streams["finals"]}
            if sum(1 for f in streams["finals"] if int(f["row"]) == r) == pop
            and all(any(int(x["row"]) == r for x in streams[k]) for k in STREAMS)
        }
        if whole and (best is None or len(whole) > best[0]):
            best = (len(whole), {k: [r for r in v if int(r["row"]) in whole] for k, v in
                                streams.items()}, d)
    if best is None:
        return ""
    n, kept, src = best
    # the same unique-per-attempt spelling the additive staging directory uses, so two carries
    # never write into one directory and neither can shadow the other's files
    carry = out.with_name(
        f"{out.name}.carry-{time.strftime('%Y-%m-%d')}-{os.getpid()}-{random.randrange(16**4):04x}"
    )
    carry.mkdir(parents=True)
    for k in STREAMS:
        C.write_jsonl(carry / files[k], kept[k])
    print(
        f"[gcg] carry: {n} whole direction(s) recovered from {src} into {carry} -- the staging "
        f"directory is left in place, nothing was deleted",
        flush=True,
    )
    return str(carry)


def assert_writable(out_dir: str, files: dict[str, str], chunked: bool, force: bool) -> None:
    """Refuse a write that would overwrite another run's files in the arm's shared directory.

    THE ADDITIVE WRITE MOVES EACH FILE IN WITH `os.replace`, which overwrites silently -- so the
    refusal a one-shot product gets from `OutDir` has to be made here, per file. Three shapes are
    refused:

      * this exact chunk is already there (re-running it is a no-op, see `existing_chunk`; only
        --force overwrites it);
      * a `--rows` chunk into a directory that holds the arm's WHOLE-SET product;
      * a whole-set run into a directory that holds chunks.

    The last two are the shape `common.read_rollouts` refuses on the reading side: one arm
    described twice, where preferring either silently reads a partial as if it were complete.
    """
    outp = Path(out_dir)
    clash = sorted(n for n in files.values() if (outp / n).is_file())
    assert not clash or force, (
        f"{out_dir} already holds {clash} -- this exact chunk has been written. Re-running it is a "
        f"no-op (see `existing_chunk`); overwriting it takes --force."
    )
    if chunked:
        whole = [n for n in chunk_files("").values() if (outp / n).is_file()]
        assert not whole or force, (
            f"{out_dir} holds the WHOLE-SET product {whole} of this arm; a `--rows` chunk beside it "
            f"would make one arm two experiments (common.read_rollouts refuses the same shape). "
            f"Write the chunks to a new arm (`--arm-suffix`) or move the whole-set product aside."
        )
    else:
        parts = sorted(f.name for f in (outp.glob("finals__rows*.jsonl") if outp.is_dir() else []))
        assert not parts or force, (
            f"{out_dir} holds `--rows` chunks {parts[:4]} of this arm; a whole-set run beside them "
            f"would make one arm two experiments. Use --arm-suffix, or read the chunks with "
            f"gcg/collect.py."
        )


def existing_chunk(out_dir: str, files: dict[str, str], sel: list[int], a: dict) -> dict | None:
    """This chunk's own committed product, when it already covers exactly this call. Else None.

    The retry loop cannot tell "the container failed" from "the container finished and the local
    client dropped", and the second case is the one that used to cost a whole second run (or, with
    a one-shot write, a `--force` refusal loop). A chunk that is already on the volume for THIS row
    selection and THIS arm configuration is returned as the result, with no model load and no GPU.
    A summary that disagrees on any of it is NOT reused: it falls through to the ordinary write,
    whose own guard refuses to overwrite it.
    """
    path = Path(out_dir) / files["summary"]
    if not path.is_file():
        return None
    with open(path) as fh:
        prev = json.load(fh)
    want = {k: a[k] for k in ("mode", "init", "iters", "pop", "children", "seq_len", "topk")}
    got = {k: prev.get("config", {}).get(k) for k in want}
    if prev.get("rows") != list(sel) or got != want:
        why = []
        if prev.get("rows") != list(sel):
            why.append(f"rows {prev.get('rows', [])[:4]}... vs {list(sel)[:4]}...")
        if got != want:
            why.append(f"config {({k: (got[k], want[k]) for k in want if got[k] != want[k]})}")
        print(
            f"[gcg] {path} exists but is a DIFFERENT call ({'; '.join(why)}); not reusing it",
            flush=True,
        )
        return None
    tot = prev.get("totals", {})
    print(
        f"[gcg] {path.name} already covers these {len(sel)} directions at this configuration; "
        f"returning the committed product without starting a search",
        flush=True,
    )
    return {
        "arm": prev["arm"],
        "family": prev["family"],
        "n_directions": prev["n_directions"],
        "n_directions_run": 0,
        "sae": prev.get("sae"),
        "mean_final_cos": prev["mean_final_cos"],
        "mean_init_cos": prev["mean_init_cos"],
        "mean_nll": prev["mean_nll"],
        "cand_forwards": tot.get("cand_forwards", 0),
        "filter_reject_rate": tot.get("filter_reject_rate", 0.0),
        "alphabet_size": prev.get("alphabet_size", 0),
        "cos_check_same_batch_max": tot.get("cos_check_same_batch_max", 0.0),
        "cos_check_rebatch_max": tot.get("cos_check_rebatch_max", 0.0),
        "out": out_dir,
        "reused": True,
    }


def resolve_config(cfg, args):
    """The per-arm configuration, mode defaults filled in and every value asserted."""
    mode, init = args.get("mode") or "", args.get("init") or ""
    arm = args.get("arm") or ""
    if arm:
        assert not mode and not init, "--arm already names the mode and the init; pass one or the other"
        parts = arm.split("-", 1)
        assert len(parts) == 2, f"--arm must be '<mode>-<init>', got {arm!r}"
        mode, init = parts
    assert mode in MODES, f"mode must be one of {list(MODES)}, got {mode!r}"
    assert init in INITS, f"init must be one of {list(INITS)}, got {init!r}"
    dflt = MODE_DEFAULTS[mode]
    a = {
        "mode": mode,
        "init": init,
        "iters": int(args.get("iters") or dflt["iters"]),
        "pop": int(args.get("pop") or dflt["pop"]),
        "children": int(args.get("children") or dflt["children"]),
        "lam_grid": args.get("lam_grid") or dflt["lam_grid"],
        "topk": int(args.get("topk") or 512),
        "seq_len": int(args.get("seq_len") or 32),
        "tau": float(args.get("tau") or 0.02),
        "sbatch": int(args.get("sbatch") or 256),
        "restart_every": int(args.get("restart_every") or 0),
        "log_every": int(args.get("log_every") or 10),
        "filter_oversample": float(args.get("filter_oversample") or INIT_OVERSAMPLE[init]),
        "seed": int(args.get("seed") or 0),
        "arm_suffix": (args.get("arm_suffix") or "").strip(),
    }
    assert not a["arm_suffix"] or ARM_SUFFIX_RE.match(a["arm_suffix"]), (
        f"--arm-suffix {a['arm_suffix']!r} becomes a directory name, so it must be lowercase "
        f"alphanumerics separated by single hyphens"
    )
    # The mode/pop invariant is checked BEFORE the grid is parsed, so `--mode epo --pop 1` reports
    # the population it is wrong about rather than the grid length that follows from it.
    if mode == "gcg":
        assert a["pop"] == 1, f"--mode gcg is the loop at pop 1, got pop {a['pop']}"
    else:
        assert a["pop"] > 1, f"--mode epo needs a population, got pop {a['pop']}"
    lams = parse_lam_grid(a["lam_grid"], a["pop"])
    if mode == "gcg":
        assert all(x == 0.0 for x in lams), f"--mode gcg is lambda 0, got {lams}"
    # A restart re-initialises the stalled member from a RANDOM string, which silently converts a
    # corpus-init arm into a partly-random-init one halfway through while the finals still say
    # `init: corpus`. Asserted rather than left to hold by accident (both measured arms use 0).
    assert not (a["restart_every"] and a["pop"] > 1 and init != "random32"), (
        f"--restart-every {a['restart_every']} replaces the least-progressing member with a RANDOM "
        f"string, discarding the --init {init} start that defines this arm; pass --restart-every 0"
    )
    assert 2 <= a["seq_len"] <= MAX_REENC_TOK, (
        f"--seq-len {a['seq_len']} must be in [2, {MAX_REENC_TOK}] (the NLL needs a context token "
        f"and common.score_ids refuses a row above the re-encode truncation)"
    )
    assert a["iters"] > 0 and a["children"] > 0 and a["topk"] > 0
    return a, lams, arm_name(mode, init, a["arm_suffix"])


class Targets(NamedTuple):
    """What one held-out set hands the search: the direction it optimises, and the one it is scored on.

    ONE object rather than four return values on purpose -- `unit_smoke.check_return_arities` pins
    `_load_targets` at two, and an arity is the seam that broke this file's siblings before
    (`RETURN_ARITY`, measured the expensive way on 2026-09-21).
    """

    dirs: object          # [N, d] fp32 unit -- the OBJECTIVE's target, resolved by --mu
    dirs_centred: object  # [N, d] fp32 unit(act - score_mu), or None when no mean is available
    centrable: object     # [N] bool -- False where the family has no mean (an encoder column)
    mu: object            # [d] float32 numpy, the mean BOTH sides of cos_centred use, or None
    mu_spec: object       # how --mu spells the objective's mean (a path, or None)
    score_mu_spec: object # how the SCORING mean is spelled (M0a's constant), or None


def _load_targets(cfg, args, notes=None):
    """(rows meta, Targets) for the held-out set.

    The OBJECTIVE is still the uncentred cosine (module docstring) -- that half is untouched. What
    is resolved here is the other half, the TARGET: `vecs.f16` stopped being a fixed object on
    2026-09-21 (a raw set stores unit(act) and derives the rest), so the search's ceiling is
    computed against whichever direction `--mu` names. This file has no `--maem` in scope, so on
    a raw set common.mu_for refuses rather than defaulting; on the legacy sets every
    published gcg number reproduces because the set's own stored convention is the default.

    SECOND, since 2026-09-23: the directions of the CENTRED rescoring, `unit(act - score_mu)` with
    `score_mu` the one scoring constant (`score_mu_spec`). They are a second `dirs_for` call rather
    than a copy of the first, so the printed column is centred on the scoring mean whatever `--mu`
    the search was run at -- and when the two agree, which is the production case, the second call
    is the same tensor by construction. A family that is not `centrable` (an encoder column has no
    mean) gets no centred direction at all and no `cos_centred`: `cos(h - mu, encoder column)` is
    the one-sided number this column exists to replace, not a centred one.
    """
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    d_model = cfg["bases"][base]["d"]
    hdir = C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{hdir}/ids.jsonl")
    mu, _ = C.mu_for(cfg, base, hdir, args, "", root, notes)
    vecs = C.dirs_for(cfg, base, hdir, mu, root, notes)
    assert vecs.shape == (len(rows), d_model), f"{hdir}: dirs_for returned {vecs.shape}"
    v = torch.as_tensor(np.asarray(vecs), dtype=torch.float32)
    v = torch.nn.functional.normalize(v, dim=-1)

    cen = np.asarray(
        [bool(cfg["family_kinds"][r["family"]]["centrable"]) for r in rows], dtype=bool
    )
    smu_spec, v_cen, smu = score_mu_spec(cfg, base), None, None
    if cen.any():
        smu = C.load_mu(cfg, base, smu_spec, root)
        cvecs = C.dirs_for(cfg, base, hdir, smu_spec, root, None)
        assert cvecs.shape == (len(rows), d_model), f"{hdir}: dirs_for returned {cvecs.shape}"
        v_cen = torch.nn.functional.normalize(
            torch.as_tensor(np.asarray(cvecs), dtype=torch.float32), dim=-1
        )
        if notes is not None:
            notes.append(
                f"cos_centred on every final of a centrable family: max over kept tokens of "
                f"cos(unit(h - mu), unit(act - mu)) at mu={C.mu_label(smu_spec, base, root)} "
                f"(the SCORING mean, on both sides), rescored through common.score_ids from the "
                f"same forward as `cos`. The objective the search optimised is `cos`, uncentred, "
                f"at mu={C.mu_label(mu, base, root)}"
            )
    return rows, Targets(v, v_cen, cen, smu, mu, smu_spec)


def _load_scan_top(cfg, args, n_rows):
    """`{row: {size: [[doc, start, argmax, cos], ...]}}` from the scan's topk.jsonl."""
    cn = args.get("corpus_name") or ""
    path = f"{C.scan_dir(args['base'], args['heldout'], args['root'], cn)}/topk.jsonl"
    import os

    assert os.path.exists(path), (
        f"--init corpus needs the corpus scan at {path}: run `--product scan --base "
        f"{args['base']} --set {args['heldout']}` first (and with --force if the held-out set was "
        f"re-drawn since -- a scan built against different vectors is silently wrong)"
    )
    top: dict[int, dict[int, list]] = {}
    for r in C.read_jsonl(path):
        top.setdefault(int(r["row"]), {})[int(r["size"])] = r["top"]
    assert len(top) >= n_rows, f"{path} covers {len(top)} target rows, the set has {n_rows}"
    return top


def run(cfg, args):
    import torch

    t_start = time.time()
    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product gcg needs --base"
    a, lams, arm = resolve_config(cfg, args)
    read_layer = cfg["bases"][base]["read_layer"]

    cen_notes: list[str] = []
    rows_meta, tgt = _load_targets(cfg, args, notes=cen_notes)
    dirs = tgt.dirs
    # --rows indexes WITHIN the family, not the concatenated set: the held-out set lays the
    # families out end to end (realact 0-511, random 512-1023, sae 1024-1535 at 512 each), so
    # `--family sae --rows 0-7` is global rows 1024-1031. Every output row carries BOTH -- `row`
    # is the global index everything else in the pipeline joins on (scan topk, score per_target),
    # `family_row` is the index this flag named.
    family = args.get("family") or "realact"
    fam_rows = [i for i, r in enumerate(rows_meta) if r["family"] == family]
    assert fam_rows, (
        f"held-out set {set_name} on {base} has no `{family}` rows; it has "
        f"{sorted({r['family'] for r in rows_meta})}"
    )
    sel_local = C.parse_rows(args.get("rows") or "", len(fam_rows))
    sel = [fam_rows[i] for i in sel_local]
    local_of = dict(zip(sel, sel_local, strict=True))
    fams = sorted({rows_meta[i]["family"] for i in sel})
    assert fams == [family], f"row selection crossed families: {fams}"

    # THE CHUNK. A call given `--rows` is one chunk of an arm and writes its own four files into the
    # arm's ONE directory through M0a's additive write; a call over the whole family keeps the
    # historical single-product spelling and the one-shot write. `rows_spec` comes from the parsed
    # selection, not from the string the caller typed, so `--rows 0-3` and `--rows 0,1,2,3` are one
    # chunk and the second one finds the first's product instead of writing a second copy.
    out_dir = C.gcg_dir(base, set_name, family, arm, root)
    chunked = bool((args.get("rows") or "").strip())
    rows_spec = rows_spec_of(sel_local) if chunked else ""
    files = chunk_files(rows_spec)
    if not args.get("force"):
        done = existing_chunk(out_dir, files, sel, a)
        if done is not None:
            return done

    # --resume-from: finish an arm whose call died (e.g. on the Modal function timeout) from its KEPT
    # temp dir instead of re-running it. Only whole directions are carried: a direction's three
    # streams are written together after it returns, and the checks below refuse anything else.
    # `sel` stays the FULL selection, so every per-arm number below is over all of it, and the loop
    # skips the carried rows. Validated here, before the model load, so a wrong path costs seconds.
    # The guard compares what finals.jsonl records (family, mode, init, iters, seq_len, lambda per
    # member); topk/tau/children/oversample/seed are not in the finals and are NOT checked.
    resume_from = (args.get("resume_from") or "").rstrip("/")
    if not resume_from and not args.get("no_auto_resume"):
        resume_from = find_partial(out_dir, files, a["pop"])
    done_rows: set[int] = set()
    prior: dict[str, list[dict]] = {}
    if resume_from:
        rdir = Path(resume_from)
        assert rdir.resolve() != Path(out_dir).resolve(), (
            f"--resume-from {rdir} is this call's own output directory; resume from a kept staging "
            f"directory (`<arm>.tmp-<date>-<pid>-<hex>`, printed by `[outdir] FAILED`) or from the "
            f"`<arm>.carry-<stamp>` directory the automatic carry writes"
        )
        for k in STREAMS:
            assert (rdir / files[k]).is_file(), f"--resume-from {rdir} has no {files[k]}"
            prior[k] = read_jsonl_tolerant(rdir / files[k], "resume")
        done_rows = {int(f["row"]) for f in prior["finals"]}
        extra = sorted(done_rows - set(sel))
        assert not extra, (
            f"--resume-from {rdir} carries rows {extra[:8]} outside this --rows selection; the "
            f"per-arm summary is over the selection, so pass a --rows that covers them"
        )
        for k in STREAMS[1:]:
            got = {int(r["row"]) for r in prior[k]}
            assert got == done_rows, (
                f"--resume-from {rdir}: {files[k]} and {files['finals']} disagree on rows "
                f"{sorted(got ^ done_rows)[:8]} -- a partly written direction"
            )
        want = {"family": family, "mode": a["mode"], "init": a["init"], "iters": a["iters"],
                "seq_len": a["seq_len"]}
        for row in sorted(done_rows):
            fin = [f for f in prior["finals"] if f["row"] == row]
            assert sorted(f["member"] for f in fin) == list(range(a["pop"])), (
                f"--resume-from {rdir}: row {row} has members {sorted(f['member'] for f in fin)}, "
                f"want 0..{a['pop'] - 1}"
            )
            for f in fin:
                got = {k: f[k] for k in want}
                assert got == want and abs(f["lam"] - lams[f["member"]]) < 1e-9, (
                    f"--resume-from {rdir}: row {row} member {f['member']} was run as {got} at "
                    f"lambda {f['lam']}; this call is {want} at lambda {lams[f['member']]}"
                )
        # EVERY direction carried is not an error: a container killed between its last direction and
        # its commit leaves exactly that, and the call finishes by committing the carried streams.
        # It still costs a model load, which is why it says so.
        if done_rows == set(sel):
            print(
                f"[gcg] resume: all {len(sel)} directions are already in {rdir}; this call only "
                f"commits them to {out_dir} and runs no search",
                flush=True,
            )
        print(
            f"[gcg] resume: {len(done_rows)} of {len(sel)} directions carried from {rdir} "
            f"({len(prior['finals'])} finals, {len(prior['trajectory'])} trajectory, "
            f"{len(prior['top64'])} top64 rows); {len(sel) - len(done_rows)} to run",
            flush=True,
        )
    n_todo = len(sel) - len(done_rows)
    print(
        f"[gcg] arm {arm} on {base}/{set_name}: family {family}, {len(sel)} directions "
        f"local {sel_local[:8]}{'...' if len(sel) > 8 else ''} = global {sel[:8]}"
        f"{'...' if len(sel) > 8 else ''} | pop {a['pop']} x children "
        f"{a['children']} = {a['pop'] * a['children']} cands/iter x {a['iters']} iters | "
        f"T={a['seq_len']} topk={a['topk']} tau={a['tau']} oversample={a['filter_oversample']} "
        f"lams {[round(x, 4) for x in lams]}",
        flush=True,
    )

    sae_ctx = None
    if family in C.SAE_FAMILIES:
        # WHICH dictionary the `sae` family's feature ids index: --sae, or the base's only SAE.
        # A set carrying two dictionaries under one family label must also RESTRICT the rows --
        # `--family sae --rows 0-7` would otherwise be the first eight rows of both, and a feature
        # id of the other dictionary is a valid index into this one.
        sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
        hdir = C.heldout_dir(base, set_name, root)
        keyed = {
            r["row"]
            for r in C.sae_rows_of(
                rows_meta, sae_key, (family,),
                declared=C.declared_sae_key(cfg, hdir, root), where=hdir,
            )
        }
        crossed = sorted(set(sel) - keyed)
        assert not crossed, (
            f"--family {family} --rows {args.get('rows') or 'all'} selected {len(crossed)} rows "
            f"that are NOT features of --sae {sae_key} (first 8: {crossed[:8]}). This set carries "
            f"dictionaries "
            f"{sorted({r.get('sae_key', '(unkeyed)') for r in rows_meta if r['family'] == family})}; "
            f"give --rows the local range of the one you mean."
        )
        sae_ctx = {"key": sae_key, "feature_id": None}

    init_ctx: dict = {}
    if a["init"] == "corpus":
        toks, docs = C.load_corpus(base, root)
        init_ctx = {"scan_top": _load_scan_top(cfg, args, len(rows_meta)), "toks": toks, "docs": docs}

    model, tok = C.load_base(cfg, base)  # the CLEAN base: nothing here loads a MAEM
    assert tok.is_fast, "the retokenisation filter runs every iteration; it needs a fast tokenizer"
    # No parameter needs a gradient: only the one-hot leaf does, and without this the backward
    # allocates a full copy of the model's parameter grads on the first iteration.
    model.requires_grad_(False)
    assert hasattr(model, "model"), "the NLL path calls the transformer body as `model.model`"
    mdtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    sink_id = C.sink_token_id(tok)

    alpha, n_vocab = ascii_alphabet(tok, base)
    alpha_sp = space_prefixed(tok, alpha)
    alpha_t = torch.as_tensor(alpha, device=dev)
    assert sink_id not in set(int(x) for x in alpha), (
        f"the sink id {sink_id} survived the alphabet filter -- it must not, the scorer prepends "
        f"its own sink and a string containing it would be scored on a stream nothing else builds"
    )
    # W_E in fp32: the one-hot relaxation's forward and backward both run through it, and that
    # matmul in bf16 costs ~3 decimal digits on a ranking over ~100k ids.
    w_e32 = input_embeddings(model).weight.detach().float()

    if sae_ctx is not None:
        # ONLY THE COLUMNS THIS RUN READS. The activation block below calls `common.sae_encode` on
        # one feature per direction, i.e. `relu((h - b_dec) @ W_enc[:, f] + b_enc[f])`, and the
        # objective is the cosine to a direction that came off `vecs.f16` -- so W_dec is never
        # touched here and neither are the other 2^21 - len(sel) encoder columns. Loading the whole
        # dictionary in fp32 on the device cost 43 GB of W_dec alone and OOMed an H200 at setup on
        # `--sae qwen36-27b/dict2m` (MEASURED 2026-09-21: the fp32 unembedding's 4.74 GiB could not
        # be allocated with 135.55 GiB in use), for a matrix no line of this product reads.
        # `load_sae_columns` reads the encoder on the CPU and moves only the slice; `sae_encode`
        # still takes DICTIONARY ids, so nothing downstream changes meaning.
        want = sorted({int(rows_meta[r]["id"]) for r in sel})
        sae_ctx["sae"] = C.load_sae_columns(
            C.sae_path(cfg, sae_ctx["key"]), cfg["bases"][base]["d"], want, device=str(dev),
            dtype=torch.float32,
        )
        print(
            f"[gcg] sae {sae_ctx['key']}: {sae_ctx['sae'].d_sae} features, learned gate "
            f"{sae_ctx['sae'].threshold:.4f} -- {sae_ctx['sae'].n_cols} encoder column(s) on the "
            f"device, no decoder; activation is RECORDED on the finals, never optimised (the "
            f"objective stays the cosine to the encoder column)",
            flush=True,
        )

    assert_writable(out_dir, files, chunked, bool(args.get("force")))

    inputs = {
        "heldout": C.heldout_dir(base, set_name, root),
        "family": family,
        "rows": f"{args.get('rows') or 'all'} of family {family} (global {sel[0]}..{sel[-1]})",
        "directions": len(sel),
        "read_layer": read_layer,
        "arm": arm,
        "chunk": f"{files['finals']} (--rows {rows_spec})" if chunked else "whole family, one product",
        "score_mu": C.mu_label(tgt.score_mu_spec, base, root) if tgt.mu is not None else "none",
    }
    if sae_ctx is not None:
        inputs["sae"] = f"{sae_ctx['key']} (gate {sae_ctx['sae'].threshold:.4f})"
    if a["init"] == "corpus":
        inputs["scan"] = C.scan_dir(base, set_name, root, args.get("corpus_name") or "")
        inputs["corpus"] = C.corpus_dir(base, root)
    if resume_from:
        inputs["resumed_from"] = (
            f"{resume_from} ({len(done_rows)} of {len(sel)} directions carried, not re-run here)"
        )

    # `prior` is keyed by the LOGICAL stream name (STREAMS), not by the file name -- the file name
    # is per chunk. Reading it with the old `finals.jsonl` key silently returned nothing, which
    # would have put the carried directions in the FILES and left them out of every aggregate.
    all_finals: list[dict] = list(prior.get("finals", []))
    all_top: list[dict] = list(prior.get("top64", []))
    all_traj: list[dict] = list(prior.get("trajectory", []))
    runs: dict[str, dict] = {}
    # keep_existing = ADDITIVE (M0a, 2026-09-23): the chunks of one arm share its directory, each
    # staging in a temp of its own and moving in only its own four files. A whole-family run is a
    # one-shot product and keeps the temp-and-rename it always had.
    with C.outdir(out_dir, args, inputs=inputs, keep_existing=chunked) as od:
        C.note_convention(od, cen_notes)
        # Streamed, not buffered: an 8-direction epo run is ~20 GPU-minutes and a crash at
        # direction 7 must not throw away the six that finished (the staging dir is kept).
        # On a resume the carried streams are copied in byte for byte and appended to.
        for k in prior:
            shutil.copyfile(Path(resume_from) / files[k], od.file(files[k]))
        mode = "a" if prior else "w"
        fh_fin = open(od.file(files["finals"]), mode)
        fh_tr = open(od.file(files["trajectory"]), mode)
        fh_top = open(od.file(files["top64"]), mode)
        try:
            n_run = 0
            for row in sel:
                if row in done_rows:
                    continue
                fam = rows_meta[row]["family"]
                # crc32, not hash(): Python's string hash is salted per process, so hash(fam) would
                # make --seed reproduce nothing across runs.
                rng = np.random.default_rng([a["seed"], zlib.crc32(fam.encode()), row])
                if sae_ctx is not None:
                    sae_ctx["feature_id"] = int(rows_meta[row]["id"])
                fin, tops, traj, tm = run_direction(
                    a, model, tok, dev, alpha, alpha_t, w_e32, dirs[row], row, fam, lams, rng,
                    arm, sink_id, mdtype, read_layer, alpha_sp, init_ctx,
                    rows_meta[row].get("span_text"), sae_ctx, local_of[row],
                    d_centred_cpu=(
                        None if tgt.dirs_centred is None or not tgt.centrable[row]
                        else tgt.dirs_centred[row]
                    ),
                    mu_vec=None if not tgt.centrable[row] else tgt.mu,
                )
                for r in fin:
                    fh_fin.write(json.dumps(r) + "\n")
                for r in traj:
                    fh_tr.write(json.dumps(r) + "\n")
                for r in tops:
                    fh_top.write(json.dumps(r) + "\n")
                for fh in (fh_fin, fh_tr, fh_top):
                    fh.flush()
                all_finals += fin
                all_top += tops
                all_traj += traj
                runs[f"{fam}:{row}"] = {
                    "row": row, "family": fam, "timings": tm,
                    "finals": [
                        {k: v for k, v in f.items() if k not in _BULKY_FINAL_KEYS} for f in fin
                    ],
                }
                n_run += 1
                el = time.time() - t_start
                print(
                    f"[gcg] DONE {fam}:{row} best cos {max(f['cos'] for f in fin):.4f} "
                    f"(init {fin[0]['init_cos']:.4f}) | {tm['cand_forwards']} cands in "
                    f"{tm['gpu_seconds']:.0f} gpu-s = {tm['cand_per_s_total']:.0f} cand/s | reject "
                    f"{tm['filter_reject_rate']:.3f} | {len(done_rows) + n_run}/{len(sel)} dirs, "
                    f"eta {(n_todo - n_run) * el / n_run / 60:.1f} min",
                    flush=True,
                )
        finally:
            for fh in (fh_fin, fh_tr, fh_top):
                fh.close()
        # OutDir.write_jsonl builds these entries for files it writes itself; these three are
        # streamed above, so their index entries are registered by hand.
        for name, n in (
            (files["finals"], len(all_finals)),
            (files["trajectory"], len(all_traj)),
            (files["top64"], len(all_top)),
        ):
            od.index[name] = {"kind": "jsonl", "rows": n, "bytes": od.file(name).stat().st_size}

        best = np.array([f["cos"] for f in all_finals], np.float64)
        init_c = np.array([f["init_cos"] for f in all_finals], np.float64)
        nlls = np.array([f["nll"] for f in all_finals], np.float64)
        per_dir_best = [max(f["cos"] for f in all_finals if f["row"] == r) for r in sel]
        per_dir_init = [max(f["init_cos"] for f in all_finals if f["row"] == r) for r in sel]
        n_distinct = [
            len({tuple(t["ids"]) for t in all_top if t["row"] == r}) for r in sel
        ]
        tot = {
            k: float(sum(runs[r]["timings"][k] for r in runs))
            for k in ("grad_s", "cand_s", "fwd_s", "nll_s", "misc_s", "gpu_seconds", "wall_s")
        }
        tot["cand_forwards"] = int(sum(runs[r]["timings"]["cand_forwards"] for r in runs))
        tot["cand_per_s_total"] = tot["cand_forwards"] / max(1e-9, tot["gpu_seconds"])
        tot["cand_per_s_fwd"] = tot["cand_forwards"] / max(1e-9, tot["fwd_s"])
        drawn = float(sum(runs[r]["timings"]["filter_drawn"] for r in runs))
        kept = float(sum(runs[r]["timings"]["filter_kept"] for r in runs))
        tot["filter_reject_rate"] = 1.0 - kept / max(1.0, drawn)
        # `default=`: a call that carried every direction ran no CHECK of its own, and an empty
        # max() would turn "there was nothing left to do" into a crash after the model load.
        tot["cos_check_same_batch_max"] = max(
            (runs[r]["timings"]["cos_check_same_batch"] for r in runs), default=0.0
        )
        tot["cos_check_rebatch_max"] = max(
            (runs[r]["timings"]["cos_check_rebatch"] for r in runs), default=0.0
        )

        sae_rows = [f for f in all_finals if "sae_peak_act" in f]
        cen_vals = [f["cos_centred"] for f in all_finals if f.get("cos_centred") is not None]

        def _centred_of_reported(r):
            """The centred cosine OF THE FINAL THIS ROW REPORTS -- the member with the best `cos`.

            Not `max(cos_centred)` over the members: the arm's number is the string the objective
            selected, and rescoring it is the point (spec §1.4, "what is optimised is not what is
            reported"). Taking the best centred value instead would be a second, centred search
            over the Pareto front, which is exactly the bound the paper says it does NOT have.
            """
            fs = [f for f in all_finals if f["row"] == r]
            best_f = max(fs, key=lambda f: f["cos"])
            return best_f.get("cos_centred")

        per_dir_cen = [_centred_of_reported(r) for r in sel]
        summary = {
            "base": base, "set": set_name, "family": family, "arm": arm, "config": a,
            "arm_base": arm_name(a["mode"], a["init"]), "arm_suffix": a["arm_suffix"],
            "rows_spec": rows_spec, "chunked": chunked,
            "files": {k: files[k] for k in (*STREAMS, "summary")},
            # The centred rescoring: the mean, and the column the paper prints. `score_mu` is the
            # ONE scoring constant (gcg.score_mu_spec); `mu` is what the SEARCH optimised against.
            "score_mu": C.mu_label(tgt.score_mu_spec, base, root) if tgt.mu is not None else None,
            "mu": C.mu_label(tgt.mu_spec, base, root),
            "n_cos_centred": len(cen_vals),
            "cos_centred_is": (
                "the centred rescoring of the member with the best uncentred cos, per direction"
            ),
            "mean_final_cos_centred": float(np.mean(cen_vals)) if cen_vals else None,
            "mean_per_dir_best_cos_centred": (
                float(np.mean([x for x in per_dir_cen if x is not None]))
                if any(x is not None for x in per_dir_cen) else None
            ),
            "per_dir_best_cos_centred": per_dir_cen,
            "lams": lams, "rows": sel, "family_rows": sel_local, "families": fams,
            "read_layer": read_layer, "d": cfg["bases"][base]["d"],
            "vocab": int(n_vocab),
            "alphabet_size": int(len(alpha)),
            "alphabet_expected": ALPHABET_EXPECT.get(base),
            "alphabet_space_prefixed": int(len(alpha_sp)),
            "sink_id": int(sink_id),
            "score_max_length": C.SCORE_MAX_LENGTH,
            "score_chunk": C.SCORE_CHUNK,
            "candidate_budget_per_iter": a["pop"] * a["children"],
            "n_directions": len(sel),
            "n_directions_run": n_todo,
            "resumed_from": resume_from or None,
            "resumed_rows": sorted(done_rows),
            "mean_final_cos": float(best.mean()),
            "mean_init_cos": float(init_c.mean()),
            "mean_nll": float(nlls.mean()),
            # The number to quote: mean over TARGETS of each target's best member, with its
            # standard error. `mean_final_cos` above averages over (target, member) pairs, which
            # for an `epo` arm mixes three lambdas and is not the arm's reachability figure.
            "mean_per_dir_best_cos": float(np.mean(per_dir_best)),
            "se_per_dir_best_cos": (
                float(np.std(per_dir_best, ddof=1) / np.sqrt(len(per_dir_best)))
                if len(per_dir_best) > 1
                else None
            ),
            "mean_per_dir_init_cos": float(np.mean(per_dir_init)),
            "se_per_dir_init_cos": (
                float(np.std(per_dir_init, ddof=1) / np.sqrt(len(per_dir_init)))
                if len(per_dir_init) > 1
                else None
            ),
            "per_dir_best_cos": [round(float(x), 6) for x in per_dir_best],
            "per_dir_init_cos": [round(float(x), 6) for x in per_dir_init],
            "distinct_top_strings_per_dir": n_distinct,
            "by_lam": {
                f"{lam:.4g}": {
                    "n": int(sum(1 for f in all_finals if f["member"] == m)),
                    "cos_mean": float(np.mean([f["cos"] for f in all_finals if f["member"] == m])),
                    "nll_mean": float(np.mean([f["nll"] for f in all_finals if f["member"] == m])),
                }
                for m, lam in enumerate(lams)
            },
            "sae": (
                None
                if sae_ctx is None
                else {
                    "key": sae_ctx["key"],
                    "threshold": round(float(sae_ctx["sae"].threshold), 6),
                    "features": [int(rows_meta[r]["id"]) for r in sel],
                    "mean_peak_act": float(np.mean([f["sae_peak_act"] for f in sae_rows])),
                    "mean_act_at_argmax": float(np.mean([f["sae_act_at_argmax"] for f in sae_rows])),
                    "frac_fired": float(np.mean([f["sae_fired"] for f in sae_rows])),
                    # only defined where the feature actually fires somewhere (peak_pos >= 0)
                    "n_rows_peak_defined": int(
                        sum(1 for f in sae_rows if f["sae_peak_pos"] >= 0)
                    ),
                    "frac_peak_at_cos_argmax": (
                        float(
                            np.mean(
                                [
                                    f["sae_peak_pos"] == f["argmax"]
                                    for f in sae_rows
                                    if f["sae_peak_pos"] >= 0
                                ]
                            )
                        )
                        if any(f["sae_peak_pos"] >= 0 for f in sae_rows)
                        else None
                    ),
                }
            ),
            "runs": runs,
            "totals": tot,
            "wall_min": (time.time() - t_start) / 60.0,
        }
        od.write_json(files["summary"], summary)

        od.section(
            "Objective",
            [
                "`L_lambda(x) = cos(x) - lambda * nll(x)` at a FIXED string length of "
                f"T={a['seq_len']} ids.",
                "",
                f"- `cos(x)` is the max over KEPT tokens (sink excluded, no norm filter) of "
                f"`cos(unit(h_t), d)` at read layer {read_layer}, through `common.score_ids` -- "
                f"the same function `score.py` scores rollouts with. UNCENTRED, fp32.",
                "- `nll(x)` is the mean per-token NLL of the string's own ids under the clean base "
                "(teacher forcing, no prompt, no sink), fp32 lm_head, self-checked against "
                "`model(...).logits` on the first batch.",
                f"- arm `{arm}`: mode `{a['mode']}` (pop {a['pop']} x children {a['children']} = "
                f"{a['pop'] * a['children']} candidates/iter x {a['iters']} iters), init "
                f"`{a['init']}`, lambdas {[round(x, 4) for x in lams]}, topk {a['topk']}, tau "
                f"{a['tau']}, sbatch {a['sbatch']}, oversample {a['filter_oversample']}, seed "
                f"{a['seed']}.",
            ],
        )
        od.note(
            f"{len(sel)} `{family}` directions, family rows {args.get('rows') or 'all'} (global "
            f"{sel[0]}..{sel[-1]}) of {set_name}: mean final cos **{best.mean():.4f}**, mean init "
            f"cos **{init_c.mean():.4f}**, mean NLL {nlls.mean():.4f}"
        )
        if sae_ctx is not None:
            sm = summary["sae"]
            od.note(
                f"sae activation, RECORDED not optimised (objective = cos to the encoder column, "
                f"gate {sm['threshold']:.4f} from the checkpoint): mean peak pre-gate act over "
                f"kept tokens **{sm['mean_peak_act']:.4f}**, mean act at the cosine's argmax "
                f"{sm['mean_act_at_argmax']:.4f}, **fraction fired {sm['frac_fired']:.3f}**, "
                f"peak and cos-argmax coincide on "
                + (
                    "n/a (the feature is dead on every final)"
                    if sm["frac_peak_at_cos_argmax"] is None
                    else f"{sm['frac_peak_at_cos_argmax']:.3f} of the "
                    f"{sm['n_rows_peak_defined']} finals where it fires somewhere"
                )
            )
        if summary["mean_per_dir_best_cos_centred"] is not None:
            od.note(
                f"CENTRED RESCORING (the column the paper prints): mean over directions of the "
                f"reported final's `cos_centred` = "
                f"**{summary['mean_per_dir_best_cos_centred']:.4f}**, against "
                f"{summary['mean_per_dir_best_cos']:.4f} for the uncentred objective the search "
                f"optimised. Both come off ONE `common.score_ids` forward per direction "
                f"(`exact_cos`), at score_mu={summary['score_mu']} on both sides; no separate "
                f"`score` pass is run and the search never saw the centred number"
            )
        od.note(
            f"alphabet {len(alpha)} ids of a {n_vocab}-id vocabulary (expected "
            f"{ALPHABET_EXPECT.get(base)}), {len(alpha_sp)} of them space-prefixed"
        )
        od.note(
            f"retokenisation filter: {tot['filter_reject_rate']:.3f} of {int(drawn)} drawn "
            f"candidates rejected at --filter-oversample {a['filter_oversample']}; "
            f"{sum(runs[r]['timings']['topup_capped_iters'] for r in runs)} iterations hit the "
            f"{MAX_TOPUP_ROUNDS}-round top-up cap"
        )
        od.note(
            f"CHECK: the loop's cos vs a fresh common.score_ids call, max over all finals -- "
            f"{tot['cos_check_same_batch_max']:.2e} at the loop's own sbatch "
            f"(hard bound {COS_TOL_SAME_BATCH_HARD:.0e}) and "
            f"{tot['cos_check_rebatch_max']:.2e} at the pipeline's SCORE_CHUNK={C.SCORE_CHUNK} "
            f"(bound {COS_TOL_REBATCH:.0e})"
        )
        od.note(
            f"{tot['cand_forwards']} candidate forwards in {tot['gpu_seconds']:.0f} gpu-s = "
            f"{tot['cand_per_s_total']:.0f} cand/s total, {tot['cand_per_s_fwd']:.0f} fwd-only "
            f"(grad {tot['grad_s']:.0f}s cand {tot['cand_s']:.0f}s fwd {tot['fwd_s']:.0f}s nll "
            f"{tot['nll_s']:.0f}s misc {tot['misc_s']:.0f}s)"
        )
        if resume_from:
            od.note(
                f"RESUMED from `{resume_from}`: {len(done_rows)} of {len(sel)} directions were "
                f"carried over byte for byte and NOT re-run in this call. The mean cos / init / NLL "
                f"above are over all {len(sel)}; the CHECK maxima, the candidate-forward and timing "
                f"totals, `runs` in summary.json and the wall/cost of this README cover only the "
                f"{n_todo} directions run here. The carried directions' cost belongs to the call "
                f"that wrote {resume_from} and is not in this README"
            )
        od.note(
            f"cost per direction: ${od.cost_usd() / max(1, n_todo):.4f} "
            f"({od.wall() / max(1, n_todo):.0f} s/dir over the {n_todo} direction(s) run in this call)"
        )
        if a["init"] == "corpus":
            od.note(
                "init: the scan's top-1 corpus window at the largest corpus size, cut to the "
                f"{a['seq_len']} tokens ending at max(argmax, {a['seq_len'] - 1}) and then "
                "roundtrip-repaired; `init_window_cos` in finals.jsonl is the WHOLE window's scan "
                "cosine (pre-cut, pre-repair) and `init_cos` is the exact cosine of the cut the "
                f"search actually starts from. {sum(runs[r]['timings']['init_short_windows'] for r in runs)} "
                "top-k windows were skipped for being shorter than T."
            )
        od.note(
            "trajectory.jsonl logs every "
            f"{a['log_every']} iterations per member; top64.jsonl is the {TOP_KEEP} best DISTINCT "
            "candidate strings each member saw over its whole run, with the cosine that was "
            "selected on and an NLL computed in one batch at the end"
        )

    print(
        f"[gcg] TOTAL arm {arm}: {len(sel)} dirs, mean final cos {best.mean():.4f}, mean init cos "
        f"{init_c.mean():.4f}, mean nll {nlls.mean():.4f}, {tot['cand_forwards']} candidate "
        f"forwards, {(time.time() - t_start) / 60:.1f} min",
        flush=True,
    )
    return {
        "arm": arm,
        "family": family,
        "n_directions": len(sel),
        "n_directions_run": n_todo,
        "sae": summary["sae"],
        "mean_final_cos": float(best.mean()),
        "mean_init_cos": float(init_c.mean()),
        "mean_nll": float(nlls.mean()),
        "cand_forwards": tot["cand_forwards"],
        "filter_reject_rate": tot["filter_reject_rate"],
        "alphabet_size": int(len(alpha)),
        "cos_check_same_batch_max": tot["cos_check_same_batch_max"],
        "cos_check_rebatch_max": tot["cos_check_rebatch_max"],
        "out": out_dir,
    }
