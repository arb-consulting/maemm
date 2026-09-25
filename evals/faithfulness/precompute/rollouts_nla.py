# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.6"]
# ///
"""Product `rollouts_nla`: the NLA activation-verbalizer baseline on the HF `generate` path.

    <root>/maemms/<base>/<nla>/rollouts/<set>.jsonl                one row per (target, rollout)
    <root>/maemms/<base>/<nla>/rollouts/README.md                  + index.json, as rollouts_hf
    <root>/maemms/<base>/<nla>/variants/<set>__amp-<amp>/          a NON-default `--amp` run:
        rollouts.jsonl + rollouts.summary.json                     `score --rollouts-dir <that>`
    <root>/maemms/<base>/<nla>/README.md                           the identity card, written once

**What the NLA is.** A Natural Language Autoencoder verbalizer (any NLA implementation): the NLA checkpoint, a FULL merged bf16
`Qwen3_5ForCausalLM` trained -- warm-start SFT on `qwen3-8b-nla-L24` explanations, then GRPO
against a reconstruction reward -- to read a layer-42 activation injected at a marker token and
answer `<explanation>...</explanation>` with 2-3 snippets describing it. It is the paper's
"somebody else already built an activation-to-text model" baseline against the MAEMMs, and it goes
through the SAME scorer they do: this product writes rollout rows and computes no cosine at all.

**The injection contract**, from the checkpoint's own sidecar and the NLA implementation:

  * input = the RAW layer-42 block-output residual, `hidden_states[43]`, `norm: none` in
    `nla_meta.yaml extraction` -- no centring, no scaling of any kind at the model's side;
  * the hook is a Karvonen norm-matched ADD at the OUTPUT of decoder block 1
    (`nla/injection.py:karvonen_inject_in_residual`, `nla/utils/hooks.py:register_karvonen_hook`
    with `layer_idx=1`): `out[b,p] = h_p + ||h_p|| * v/||v||`. That is exactly
    `common.make_inject_hook(vecs, [[marker]], coeff=1.0, ...)` on `common.get_layer(model, 1)`,
    which normalises `vecs` itself and skips decode steps -- i.e. the MAEMMs' own hook, at the
    MAEMMs' own layer and coefficient, with a different marker;
  * the marker is `㈜` (id 158983) and it is injected ONLY where `ids[p-1] == 29` and
    `ids[p+1] == 510` (the `<concept>`/`</concept>` tags around it). Those neighbours are part of
    the contract, not a diagnostic, and `nla_prompt_ids` asserts all three. NOTE the marker is
    NOT the last prompt token, which is where this product's prompt handling parts company with
    `rollouts_hf` (whose `mpos == len(prompt) - 1` assert would fire here);
  * the prompt is ONE user message -- the sidecar's `prompt_templates.actor` with
    `{injection_char}` replaced by the marker -- through `apply_chat_template(...,
    add_generation_prompt=True)`. The `enable_thinking` argument is OURS and is a DIVERGENCE, not
    part of the contract; see the divergences below. (The reference script renders with
    `tokenize=False` and then `tok.encode(text, add_special_tokens=False)`, which gives the same
    ids as its own `tokenize=True` rendering -- an equivalence about ITS rendering, not about ours.)

**Why the input AMPLITUDE only matters through `mu`.** The hook normalises `v` before it scales by
`||h_p||`, so multiplying the whole input vector by any positive constant changes nothing the model
sees -- only its DIRECTION reaches block 1. But our held-out `realact` rows are `unit(X[p] - mu)`
(README "Methods": the centring happens once, in `targets`), while the verbalizer was trained on
the uncentred `X[p]`. Adding `mu` back tilts the direction, and HOW FAR it tilts depends on the
size of the `u` component relative to `||mu||` = 67.93 on this base. So the amplitude is not a
no-op after all: it is the mixing ratio. `--amp` names the convention:

**Since 2026-09-21 this is contract-dependent, and `mu`/`exact` are REFUSED on a raw set**
(`check_amp_storage`). The sentence above describes a `storage: unit` set, where the stored row is
`unit(X[p] - mu)`. On a `storage: raw` set -- every `2026-09-21_v3_*` block with an act.f32 --
`dirs_for` returns `unit(X[p])` at the NLA's `mu: null`, so the direction is ALREADY the one the
verbalizer was trained on and there is nothing to add back. There `raw` is not a compromise, it
is the right answer, and its amplitude is a true no-op (the hook normalises `v`).

    raw     x = r*u          THE DEFAULT (2026-09-21). The direction the SCORER's target
                             is, fed as is; no mu anywhere. It is also what every MAEMM arm is
                             injected with, so the NLA column is read against them on the same
                             input -- at the cost that it is NOT what the NLA was trained to
                             read, which is what the two variants below are for.
    mu      x = mu + r*u     the uncentred reconstruction at a typical corpus amplitude r.
    exact   x = mu + t*u     the uncentred reconstruction at THIS row's own recorded raw norm:
                             t > 0 solving ||mu + t*u|| = act_norm, the `||X[p]||` before centring
                             that `targets.py:114,139` stored on every realact row. Falls back to
                             `mu` (and says which rows, in the summary) for a row that carries no
                             `act_norm` -- every `sae`, `random` and `sae2m_enc` row, which are
                             encoder columns and Gaussian draws with no amplitude of their own.

`r` is `nla.amp_r`: `median` resolves to the read layer's q[0.5] of
`base/<base>/stats/resid_norm_quantiles.json` (93.259 on `qwen36-27b`, against a `mu` of 67.93 and
the card's own `example_activations.parquet` norms of min 67.7 / median 88.3 / max 116.8), or a
number is taken as is. The DEFAULT amp (`raw`) writes the ordinary accumulating
`rollouts/<set>.jsonl`; any other amp writes `variants/<set>__amp-<amp>/rollouts.jsonl`, which
`score --rollouts-dir` reads. One scoring path, one rollouts schema, several inputs.

**Seeding** is `rollouts_hf`'s rule verbatim (`common.gen_seed_for`): the (target, rollout) grid is
flattened target-major, cut into `gen_rows` chunks, and each chunk seeded
`rollouts.seed * 1000 + flat_of_its_first_row` immediately before `generate`, with the seed stored
on every row it produced.

**DIVERGENCES from the model card's reference script** (`scripts/show_nla_generations.py`),
deliberate and recorded in every summary:

  * it decodes GREEDILY at `max_new_tokens=200`; we SAMPLE, with the pipeline's SHARED
    `rollouts:` constants (T 1.0, top_p 1.0, top_k off) -- the same as every MAEMM arm. The
    checkpoint's `generation_config.json` ships top_p 0.95 / top_k 20; that is recorded on every
    summary but NOT used (2026-09-21): it is not a convention anywhere, and using it
    would make this column differ from the others in two ways at once. `n` texts per target need
    sampling to differ at all, which is why not greedy;
  * **`enable_thinking=False`, which the reference does NOT pass.**
    `nla/utils/prompts.py:build_prompt_text:12` and `scripts/show_nla_generations.py` call
    `apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)` with no
    `enable_thinking`, so the Qwen3.6 template takes its `else` branch and the prompt ends
    `<|im_start|>assistant\n<think>\n` -- an OPEN think block. Ours ends
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`, i.e. the block already closed and empty.
    MEASURED on the real tokenizer (review, 2026-09-20): reference 110 tokens, ours 112, marker at
    93 EITHER WAY -- the difference is entirely in the generation prefix after the marker. Two
    reasons for ours: the AV-SFT rendering (`nla/schema.py:205-225`, which appends a trailing
    assistant message) emits `<|im_start|>assistant\n<think>\n` + trimmed reasoning +
    `\n</think>\n\n`, which with empty reasoning is EXACTLY our prefix; and the sibling model
    cards (`qwen3.6-27b-nla-rl`, `nla-qwen36-27b-matryoshka`) say to pass `enable_thinking=False`,
    while the `-av` card is silent on it. The A/B -- the same rows under the reference's open
    `<think>` block -- is UNMEASURED. Recorded as `enable_thinking` in every summary and on the
    identity card so the choice is visible rather than implicit;
  * it generates one row at a time; we batch `gen_rows` rows per call, every row carrying the
    IDENTICAL prompt so the batch is rectangular and UNPADDED (asserted -- padding would move the
    marker away from its two required neighbours and the injection would land nowhere).

**LENGTH: this arm runs at the checkpoint's NATIVE 200 tokens, and is SCORED in a 256-token
window** (2026-09-21). `nla.max_new` is 200 -- the card's own invocation
(`--max-new-tokens 200`) and the reference script's default -- because the verbalizer's
`<explanation>` answers are shaped for that length and reporting this baseline on the first 64
tokens would report a fraction of its output. The pipeline's re-encode truncation is
`common.SCORE_MAX_LENGTH` = 95, which would score LESS THAN HALF of such a text, so this arm
carries its own: `nla.score_max_tokens` = 256 goes into the rollouts summary as `score_max_length`,
`score` re-encodes at it (`common.encode_for_score(..., max_length)`) and records it in its
`rows.json`, and `common.score_width_of` is how every reader of the stored arrays gets the width.
256 rather than 201 leaves room for re-tokenization expansion -- a decoded rollout does not always
re-encode to the same id count (README, checklist item 8).

That is a STATED DEVIATION from the one protocol every other arm shares, and it costs two things
that are named wherever the number appears rather than buried: a cosine from this arm is a max over
a WIDER window than a MAEMM's, and its generation length is not the MAEMMs' 64. Both are
properties of the baseline being somebody else's model at its own operating point.

**Out of scope here**, and named so nobody reads a missing number as a negative one: the AR critic
-- the reconstruction / FVE half of the autoencoder, which is a second checkpoint and a second
objective. Nothing in this file computes a cosine or an FVE.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# `uv run precompute/rollouts_nla.py --selftest` runs this file as a script, where evals/faithfulness/ is
# not on the path; unit_smoke.py does the same for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import precompute.common as C  # noqa: E402
import precompute.rollouts_hf as rollouts_hf  # noqa: E402

# The input-amplitude conventions, documented in the module docstring above. The tuple itself lives
# in common because `load_config` validates `nla.amp` and must not import a product module.
AMP_MODES = C.AMP_MODES

# nla/schema.py:28-33, verbatim. The actor's whole payload is what sits between these tags; the
# `.strip()` in extract_explanation is upstream too (schema.py:45-54), and it matters there because the
# critic was trained on `<text>foo</text>`, not `<text>\nfoo\n</text>`.
EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}(.*?){re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)

# The engine label on every row and in the summary: this is the HF `generate` path, exactly the one
# `rollouts_hf` uses, so `score` reads these rows with no engine special-casing at all.
ENGINE = "hf"
# `summary["prompt"]`: not a key of common.PROMPTS (this product builds its own prompt from the
# checkpoint's sidecar), but the field has to name something, and score.py only prints it.
PROMPT_NAME = "nla_av"


# ---------------------------------------------------------------------------------------------
# pure pieces -- no torch, no transformers, no volume; all of them covered by --selftest
# ---------------------------------------------------------------------------------------------


def explanation_token_mask(pieces: list[str]) -> tuple[list[bool], str]:
    """Which tokens of an NLA decode fall inside its `<explanation>` BODY, and how it was found.

    The mask any LLM-judge-facing consumer uses. 2026-09-21: some judges were being handed
    the FULL NLA output -- chat preamble, the `<explanation>` tags themselves, anything after the
    close -- rather than the verbalizer's answer. The tags are the NLA's output FORMAT, not
    content; an LLM judge reading them is reading scaffolding no other arm has.

    Token-level rather than a string slice, because the example arm renders per-token SAE
    activation marks and the mask has to stay aligned with them. `pieces` is one decoded string
    per token (autointerp.build.token_pieces). A token is kept iff it lies entirely inside the
    body, so a token straddling a tag boundary is dropped rather than half-shown.

    status:
      "closed"   `<explanation>...</explanation>` both present -- the normal case
      "unclosed" the opening tag but no close, i.e. the answer ran into max_new. Everything after
                 the opening tag is kept. This is the case the old fallback got wrong: it handed
                 the judge the ENTIRE raw decode, opening tag and preamble included.
      "none"     no opening tag at all. Every token is kept and the status says so; there is
                 nothing to cut on, and dropping the row would shrink this arm's feature set
                 relative to the others in a paired comparison.

    The cosine score is deliberately NOT affected: it is taken over the full decode by the one
    scoring path every product shares (see `extract_explanation`).
    """
    text = "".join(pieces)
    start = text.find(EXPLANATION_OPEN)
    if start < 0:
        return [True] * len(pieces), "none"
    body_lo = start + len(EXPLANATION_OPEN)
    close = text.find(EXPLANATION_CLOSE, body_lo)
    body_hi, status = (close, "closed") if close >= 0 else (len(text), "unclosed")
    mask, pos = [], 0
    for p in pieces:
        lo, hi = pos, pos + len(p)
        mask.append(lo >= body_lo and hi <= body_hi)
        pos = hi
    return mask, status


def explanation_body(text: str) -> tuple[str, str]:
    """(body, status) for a whole NLA decode -- the string counterpart of the token mask.

    Same three statuses. Replaces the old "the body, or ALL of it" fallback, which passed an
    unclosed decode to the judge with its opening tag and preamble intact.
    """
    start = text.find(EXPLANATION_OPEN)
    if start < 0:
        return text.strip(), "none"
    body_lo = start + len(EXPLANATION_OPEN)
    close = text.find(EXPLANATION_CLOSE, body_lo)
    if close < 0:
        return text[body_lo:].strip(), "unclosed"
    return text[body_lo:close].strip(), "closed"


def extract_explanation(text: str) -> str | None:
    """nla/schema.py:45-54, verbatim: the first `<explanation>...</explanation>` body, stripped.

    INFORMATIONAL on every row. What is scored is `text`, the full decode -- the NLA's tags are
    its own output format and stripping them would be a second, NLA-only text convention in a
    pipeline whose whole point is that one scorer sees every product's text unchanged.
    """
    m = EXPLANATION_RE.search(text)
    return m.group(1).strip() if m else None


def check_amp_storage(amp: str, storage: str) -> None:
    """`--amp mu` and `--amp exact` are the UN-CENTRING variants: they are wrong on a raw set.

    Both build `x = mu + c*u`, which is only "the uncentred activation" when `u` is a CENTRED
    direction -- `unit(act - mu)`, which is what every set drawn before 2026-09-21 stored. On a
    `storage: raw` set `common.dirs_for` returns `u = unit(act)` at the NLA's `mu: null`, already
    uncentred, so adding `mu` a second time tilts the direction away from the activation instead
    of recovering it. Nothing downstream can see that: `x` still has a plausible norm and the
    hook still normalises it, so the run produces ordinary-looking rollouts of the WRONG vector.

    `raw` is correct under both contracts and is the default, so this refuses rather than
    silently picking: on a raw set the direction IS the activation's and no reconstruction is
    called for; on a centred set `mu`/`exact` remain the way to undo the centring.

    (`raw`'s AMPLITUDE is not a correctness question on either contract: the hook is
    `h_p + ||h_p|| * v/||v||`, so it normalises `v` and the scale never reaches the model.
    MEASURED 2026-09-21 on the 512 `2026-09-21_v3_realact` rows: scaling by each row's own
    `act_norm` instead of the corpus median `r` moves what the hook feeds the model by
    max 1.2e-07 -- one float32 ulp -- at min cos 0.99999982. So the two are the same run.)
    """
    if amp in ("mu", "exact") and storage == "raw":
        raise AssertionError(
            f"--amp {amp!r} on a `storage: raw` set: both reconstruct an uncentred activation as "
            f"`mu + c*u` from a CENTRED direction, but a raw set's rows already are "
            f"`unit(act)` (common.dirs_for at the NLA's `mu: null`), so this would add `mu` to an "
            f"already-uncentred direction and tilt it. Use `--amp raw`, which is the default and "
            f"is correct here; `mu`/`exact` are for a `storage: unit` set that carries a mean."
        )


def build_inputs(u: np.ndarray, rows_meta: list[dict], mu: np.ndarray | None, amp: str, r: float):
    """(x [N, d] float32, info [N]) -- the vectors handed to the injection hook, one per target.

    `u` is [N, d] float32 UNIT rows (the held-out directions, in the selected rows' order) and
    `rows_meta` is their ids.jsonl records, in the same order. See the module docstring for what
    each `amp` means and why the amplitude is a mixing ratio rather than a scale.

    The `exact` solve: with `||u|| = 1`, `||mu + t*u||^2 = t^2 + 2t(mu.u) + ||mu||^2`, so
    `||mu + t*u|| = a` has the roots `t = -(mu.u) +- sqrt((mu.u)^2 + a^2 - ||mu||^2)`. The
    discriminant is negative exactly when `a < dist(0, the line mu + R*u)`, i.e. when the row's
    recorded raw norm is smaller than anything reachable on that line -- possible for a genuinely
    small activation, so it is a FALLBACK with a reason on the row, not an assert.

    WHICH ROOT: always the larger, `-b + sqrt(...)`. The roots multiply to `||mu||^2 - a^2`, so
    when `a > ||mu||` they have opposite signs and the larger one is the UNIQUE positive solution
    -- no choice to make. When `a < ||mu||` AND `mu.u < 0` they are both positive and the two
    inputs `mu + t*u` differ in direction, which is the only thing the model sees; the row is then
    flagged `exact_ambiguous: True` (and counted in the summary) rather than silently resolved.
    MEASURED on `2026-09-16_v1`: 7 of 512 realact rows have `act_norm < ||mu||` at all (min 62.1
    against `||mu||` 67.93), so this is a rare tail, not the common case. `fallback` stays None --
    the solve HOLDS, it is the uniqueness that does not.

    Every info dict carries `amp` (what was asked for), `amp_used` (what this row got), `r` (the
    coefficient actually used: `t` for an exact row, `r` otherwise), `in_norm` (`||x||`),
    `cos_in_dir` (`cos(x, u)`: how far adding mu tilted the direction, which is the only thing the
    model sees), `fallback` (None, "no_act_norm" or "discriminant") and `exact_ambiguous`.
    """
    assert amp in AMP_MODES, f"unknown amp {amp!r}, want one of {list(AMP_MODES)}"
    assert u.ndim == 2, f"u must be [N, d], got {u.shape}"
    n, d = u.shape
    assert len(rows_meta) == n, f"{n} direction rows but {len(rows_meta)} ids.jsonl records"
    assert r > 0, f"amp_r resolved to {r}, which is not a positive amplitude"
    un = np.linalg.norm(u.astype(np.float64), axis=-1)
    assert np.allclose(un, 1.0, atol=1e-3), (
        f"build_inputs needs UNIT direction rows (the exact solve assumes ||u|| = 1); got norms in "
        f"[{un.min():.6f}, {un.max():.6f}]"
    )
    if amp in ("mu", "exact"):
        assert mu is not None and mu.shape == (d,), (
            f"amp {amp!r} reconstructs an UNCENTRED activation and so needs stats/mu.f32 [{d}], "
            f"got {None if mu is None else mu.shape}"
        )
    mu64 = np.zeros(d, dtype=np.float64) if mu is None else mu.astype(np.float64)
    mu_sq = float(mu64 @ mu64)

    x = np.zeros((n, d), dtype=np.float32)
    info: list[dict] = []
    for i in range(n):
        ui = u[i].astype(np.float64)
        fallback, used, coef, ambiguous = None, amp, float(r), False
        if amp == "exact":
            a = rows_meta[i].get("act_norm")
            if a is None or not np.isfinite(float(a)) or float(a) <= 0:
                # sae / random / sae2m_enc rows: an encoder column or a Gaussian draw has no raw
                # activation norm of its own, so there is nothing to solve for.
                used, fallback = "mu", "no_act_norm"
            else:
                b = float(mu64 @ ui)
                disc = b * b + float(a) ** 2 - mu_sq
                t = (-b + float(np.sqrt(disc))) if disc >= 0 else 0.0
                if disc < 0 or t <= 0:
                    used, fallback = "mu", "discriminant"
                else:
                    coef = t
                    # the SMALLER root -b - sqrt(disc) is positive too exactly when b < 0 and
                    # a < ||mu||: two different inputs satisfy the same norm constraint and we
                    # take the larger. Flagged, not resolved -- see the docstring.
                    ambiguous = bool(b < 0 and float(a) ** 2 < mu_sq)
        row = (coef * ui) if used == "raw" else (mu64 + coef * ui)
        x[i] = row.astype(np.float32)
        nrm = float(np.linalg.norm(row))
        info.append(
            {
                "amp": amp,
                "amp_used": used,
                "r": round(coef, 6),
                "in_norm": round(nrm, 4),
                "cos_in_dir": round(float(row @ ui / max(nrm, 1e-12)), 6),
                "fallback": fallback,
                "exact_ambiguous": ambiguous,
            }
        )
    norms = np.array([rec["in_norm"] for rec in info])
    assert np.isfinite(x).all() and np.isfinite(norms).all(), (
        f"build_inputs produced non-finite values under amp {amp!r} (r={r}): "
        f"{int((~np.isfinite(x)).sum())} of {x.size} entries"
    )
    assert (norms > 0).all(), (
        f"build_inputs produced {int((norms <= 0).sum())} zero-norm inputs under amp {amp!r}; the "
        f"hook divides by ||v||, so such a row would inject a NaN at the marker"
    )
    return x, info


def resolve_r(cfg: dict, base: str, root: str, amp_r) -> tuple[float, str]:
    """(r, source) for `nla.amp_r`. A number is itself; "median" reads the pipeline's own quantiles.

    "median" = the read layer's q[0.50] of `base/<base>/stats/resid_norm_quantiles.json`, the block
    OUTPUT residual norms `stats` measured over the 64/16 window scan (93.259 at layer 42 of
    qwen36-27b, mean 93.13). Read out of the file rather than written into the config so it cannot
    drift from the corpus it describes; the layer field and `block_output` are asserted because
    `layers` is a list indexed by layer and a shifted list would silently give the wrong amplitude.
    """
    if amp_r != "median":
        r = float(amp_r)
        assert r > 0, f"nla.amp_r must be a positive number, got {amp_r!r}"
        return r, f"config.yaml nla.amp_r = {r:g}"
    layer = int(cfg["bases"][base]["read_layer"])
    path = f"{C.stats_dir(base, root)}/resid_norm_quantiles.json"
    assert os.path.exists(path), (
        f"no {path}: `nla.amp_r: median` takes the amplitude from the `stats` product's measured "
        f"residual-norm quantiles, so `stats` must have run on {base!r} (or set a number)"
    )
    with open(path) as fh:
        q = json.load(fh)
    assert q.get("block_output") is True, (
        f"{path} does not declare block_output: true -- the injection contract is about the block "
        f"OUTPUT residual and an input-side quantile is a different number"
    )
    qs = list(q["quantiles"])
    assert 0.5 in qs, f"{path} reports quantiles {qs}, which do not include the median 0.5"
    idx = qs.index(0.5)
    rec = q["layers"][layer]
    assert rec["layer"] == layer, (
        f"{path} layers[{layer}] says layer={rec['layer']}: the list is indexed BY layer, so a "
        f"mismatch means the file was written for a different model"
    )
    r = float(rec["q"][idx])
    assert r > 0, f"{path} layers[{layer}].q[{idx}] is {r}, not a positive norm"
    return r, f"{path} layers[{layer}].q[{idx}] (quantile 0.50 of the block-output residual norm)"


def nla_prompt_ids(tok, spec: dict) -> tuple[list[int], int]:
    """(ids, marker_pos) for the verbalizer's ONE user message. Asserts the whole marker contract.

    The sidecar's `prompt_templates.actor` with `{injection_char}` -> the marker char, through
    `common._chat_ids` (= `apply_chat_template(..., add_generation_prompt=True,
    enable_thinking=False)`). Unlike every MAEMM prompt the marker is NOT the last token: it sits
    between the `<concept>` / `</concept>` tags whose ids the hook checks, and the generation
    prompt follows it.
    """
    nla = spec["nla"]
    marker, mid_want = nla["marker"], int(nla["marker_id"])
    content = nla["template"].strip().replace("{injection_char}", marker)
    ids = C._chat_ids(tok, content, add_gen=True)
    mid = list(tok.encode(marker, add_special_tokens=False))
    assert mid == [mid_want], (
        f"the marker {marker!r} encodes to {mid} under this tokenizer, but the checkpoint's "
        f"nla_meta.yaml says injection_token_id {mid_want}: a tokenizer/template mismatch would "
        f"put the injection nowhere and the model would read the literal character"
    )
    found = [i for i, t in enumerate(ids) if t == mid_want]
    assert len(found) == 1, (
        f"expected exactly one marker token {mid_want} in the {len(ids)}-token prompt, found "
        f"{len(found)} at {found[:8]}; the hook injects one vector per valid site"
    )
    mpos = found[0]
    assert 0 < mpos < len(ids) - 1, (
        f"the marker is at {mpos} of {len(ids)} tokens: it needs a token on BOTH sides (the "
        f"neighbour check below, and nla/injection.py skips p == 0 and p == len-1 outright), and "
        f"the generation prompt must follow it -- at the very end there is no assistant header"
    )
    left, right = int(nla["left_id"]), int(nla["right_id"])
    assert ids[mpos - 1] == left and ids[mpos + 1] == right, (
        f"the marker at {mpos} has neighbours ({ids[mpos - 1]}, {ids[mpos + 1]}) but the sidecar "
        f"requires ({left}, {right}): nla/injection.py:karvonen_inject_in_residual injects ONLY at "
        f"a site with those neighbours, so the direction would never reach the model"
    )
    return ids, mpos


def rows_from_generation(chunk, new_ids, stop, tok, seed: int, rows_meta: list[dict], info: dict):
    """The rollouts rows of one generate call. `info` maps a TARGET ROW to its build_inputs record.

    The schema is `rollouts_hf`'s verbatim -- row, family, k, text, ids (the generated ids only,
    trimmed at the first stop token which is KEPT, train/rl/rl.py:82-90), n_tok, finished, engine, seed
    -- plus seven NLA-only fields (amp, amp_used, r, in_norm, cos_in_dir, exact_ambiguous,
    explanation). score.py reads the former and ignores the latter, which is what keeps this
    product on the one scoring path. `cos_in_dir` is on the ROW, not only in the summary, because
    it is the per-target number an analysis would regress the score against: how far adding mu
    tilted this row's input away from the direction the scorer measures against.

    `text` is the FULL decode of the trimmed ids, never the extracted explanation.
    """
    out = []
    for (r, k), g in zip(chunk, new_ids, strict=True):
        trimmed = C.trim_at_stop(list(g), stop)
        text = tok.decode(trimmed, skip_special_tokens=True)
        rec = info[r]
        out.append(
            {
                "row": r,
                "family": rows_meta[r]["family"],
                "k": k,
                "text": text,
                "ids": [int(t) for t in trimmed],
                "n_tok": len(trimmed),
                "finished": bool(trimmed[-1] in stop),
                "engine": ENGINE,
                "seed": seed,
                "amp": rec["amp"],
                "amp_used": rec["amp_used"],
                "r": rec["r"],
                "in_norm": rec["in_norm"],
                "cos_in_dir": rec["cos_in_dir"],
                "exact_ambiguous": rec["exact_ambiguous"],
                "explanation": extract_explanation(text),
            }
        )
    return out


def _quantiles(v) -> list[float]:
    """[min, q25, q50, q75, max] of a 1-D array, rounded -- the shape every summary reports."""
    return [
        round(float(x), 6)
        for x in (v.min(), np.quantile(v, 0.25), np.quantile(v, 0.50), np.quantile(v, 0.75), v.max())
    ]


def template_sha256(spec: dict) -> str:
    """sha256 of the actor template AS CONFIGURED (stripped, placeholder NOT substituted).

    It goes in the summary so a later run that changed one character of the prompt is visible in a
    diff of two summary.json files, without either of them carrying 520 characters of prose.
    """
    return hashlib.sha256(spec["nla"]["template"].strip().encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# the checkpoint's own sidecar
# ---------------------------------------------------------------------------------------------


def check_sidecar(snapshot_dir: str, spec: dict, read_layer: int, d: int) -> tuple[dict, dict]:
    """Assert `config.yaml`'s `nla:` block still matches the checkpoint. (sidecar, shipped sampling).

    `nla_meta.yaml` is the injection contract as the checkpoint's authors published it, and
    `generation_config.json` is the sampling they shipped it with. Everything in the config was
    copied from these two files by hand, so this is the gate that stops the copy from drifting --
    a re-fetched revision with a different marker id or a different `top_k` must fail here, before
    an H200 spends an hour generating with the wrong contract.
    """
    import yaml

    nla = spec["nla"]
    path = os.path.join(snapshot_dir, "nla_meta.yaml")
    assert os.path.exists(path), (
        f"no {path}: a `type: nla` checkpoint must ship its nla_meta.yaml sidecar, which is the "
        f"injection contract (marker id, its neighbours, the extraction layer, the actor prompt)"
    )
    with open(path) as fh:
        side = yaml.safe_load(fh)

    ext, tokens = side["extraction"], side["tokens"]
    assert int(ext["layer_index"]) == read_layer, (
        f"{path} was trained on layer {ext['layer_index']} but this base reads at {read_layer}: "
        f"the held-out directions are layer-{read_layer} objects and the verbalizer would be shown "
        f"activations from a layer it never saw"
    )
    assert int(ext["d_model"]) == d, f"{path} says d_model={ext['d_model']}, the base has d={d}"
    assert ext["norm"] == "none", (
        f"{path} says extraction.norm={ext['norm']!r}, not 'none': this product feeds the RAW "
        f"residual (see the module docstring's amp conventions) and would have to rescale first"
    )
    assert tokens["injection_char"] == nla["marker"], (
        f"{path} injects at {tokens['injection_char']!r}, config.yaml says {nla['marker']!r}"
    )
    for field, key in (
        ("injection_token_id", "marker_id"),
        ("injection_left_neighbor_id", "left_id"),
        ("injection_right_neighbor_id", "right_id"),
    ):
        assert int(tokens[field]) == int(nla[key]), (
            f"{path} {field}={tokens[field]}, config.yaml nla.{key}={nla[key]}"
        )
    actor = side["prompt_templates"]["actor"]
    assert actor.strip() == nla["template"].strip(), (
        f"{path} prompt_templates.actor differs from config.yaml nla.template (sidecar "
        f"{len(actor.strip())} chars, config {len(nla['template'].strip())}): the template is the "
        f"prompt the verbalizer was TRAINED on and must be verbatim"
    )

    gpath = os.path.join(snapshot_dir, "generation_config.json")
    assert os.path.exists(gpath), f"no {gpath}: the shipped sampling constants are part of the recipe"
    with open(gpath) as fh:
        gen = json.load(fh)
    # RECORDED, not used (2026-09-21). The checkpoint ships top_p 0.95 / top_k 20 in
    # its generation_config.json, but that is not a convention anywhere -- not the NLA paper, not
    # the model card's reference script (which decodes greedily), not our pipeline. Sampling the
    # NLA with its own truncation while every other arm samples the full distribution would make
    # this column differ from the MAEMM columns in TWO ways at once. So the NLA arm samples with
    # the SHARED `rollouts:` block like everything else; what the checkpoint shipped is printed
    # and written to the summary so the choice is visible, not silent.
    shipped = {k: gen.get(k) for k in ("do_sample", "temperature", "top_p", "top_k")}
    print(
        f"[nla] sidecar {path}: layer {ext['layer_index']}, d {ext['d_model']}, norm "
        f"{ext['norm']!r}, marker {tokens['injection_token_id']} between "
        f"({tokens['injection_left_neighbor_id']}, {tokens['injection_right_neighbor_id']}); "
        f"checkpoint ships {shipped}; NOT used -- the NLA arm samples with the shared "
        f"rollouts: block like every other arm",
        flush=True,
    )
    # RETURNED, not just printed. "recorded on the summary" was the stated reason for keeping the
    # shipped constants at all, and until this returned them it was true of a print() and of
    # nothing else -- while the asserts that used to catch a re-fetched revision drifting on
    # top_k were removed in the same change. A claim about a record needs the record.
    return side, shipped


def write_nla_readme(cfg, args, maemm_key: str, sha: dict, side: dict, prompt, mpos: int) -> str:
    """`<root>/maemms/<base>/<nla>/README.md`, the identity card. Written once; --force rewrites.

    Modelled on `rollouts_hf.write_maemm_readme` but it cannot be that function: the fields that
    matter here (revision, sidecar extraction block, marker neighbours, the amp conventions, the
    card's own generation budget) do not exist on a MAEMM entry, and the fields that matter there
    (prompt name from common.PROMPTS, train_max_new, adapter subdir) do not exist here.
    """
    spec = cfg["maemms"][maemm_key]
    nla = spec["nla"]
    # FROM cfg, the same place `run` takes it. `samp` was a free name here after the shared-
    # sampling change -- the `## Sampling` bullet below reads it, the function
    # never bound it, and nothing at module scope defines it. Every NLA run that wrote a README
    # would have raised NameError after the generation it had already paid for.
    samp = {k: cfg["rollouts"][k] for k in C.NLA_SAMPLING_KEYS}
    path = f"{C.maemm_dir(maemm_key, args['root'])}/README.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not args.get("force"):
        print(f"[nla] README already at {path}; leaving it (pass --force to rewrite)", flush=True)
        return path
    ext = side["extraction"]
    inj = spec["inject"]
    lines = [
        f"# {maemm_key}",
        "",
        "**The NLA activation-verbalizer BASELINE, not a MAEMM.** Given an activation injected at "
        "a marker token it writes `<explanation>`-tagged snippets describing what that activation "
        "represents (any NLA implementation). It lives under "
        "`maemms/` because it shares the layout and the ONE clean-base scoring path; it shares "
        "neither the prompt nor the marker, and `rollouts_nla` is its only generator.",
        "",
        f"- source: {spec['hf']} (HF repo id)",
        f"- revision: `{spec['revision']}` (asserted against the resolved snapshot's directory name)",
        f"- type: {spec['type']}",
        f"- resolved weights: {sha['path']}",
        f"- written by: `{' '.join(args.get('argv') or [])}`",
        f"- repo commit: {args.get('repo_commit', '?')[:12]}",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        "",
        "## Injection convention",
        "",
        f"- inject layer: {inj['layer']} (decoder block OUTPUT), mode add, coeff {inj['coef']} -- "
        "the Karvonen norm-matched add `h[p] += unit(v) * ||h[p]|| * coeff` "
        "(`nla/injection.py:karvonen_inject_in_residual`), which is exactly "
        "`common.make_inject_hook`, at PREFILL only.",
        f"- marker: {nla['marker']!r} (id {nla['marker_id']}) at position {mpos} of "
        f"{len(prompt)} prompt tokens, occurring exactly once, with the required neighbours "
        f"{nla['left_id']} (left) and {nla['right_id']} (right). The marker is NOT the last prompt "
        "token: the `</concept>` tag and the generation prompt follow it.",
        "- prompt: ONE user message, `nla_meta.yaml prompt_templates.actor` with "
        "`{injection_char}` replaced by the marker, through `apply_chat_template("
        "add_generation_prompt=True, enable_thinking=False)`.",
        "- **`enable_thinking=False` is OUR choice, not the reference's.** "
        "`nla/utils/prompts.py:build_prompt_text` and `scripts/show_nla_generations.py` pass no "
        "`enable_thinking` at all, so their prompt ends with an OPEN `<think>\\n` block (110 "
        "tokens on this tokenizer, against our 112; the marker sits at 93 either way). Ours "
        "matches the AV-SFT rendering (`nla/schema.py`, trailing assistant message -> "
        "`<think>\\n\\n</think>\\n\\n`) and what the sibling `-rl` / matryoshka cards "
        "instruct; the `-av` card is silent. The A/B is UNMEASURED.",
        f"- template sha256: `{template_sha256(spec)}` (stripped, placeholder not substituted)",
        "- ONLY the direction reaches the model: the hook normalises `v` before scaling it by "
        "`||h||`. See the amp conventions below for what that means for the input.",
        "",
        "## The checkpoint's sidecar (`nla_meta.yaml`, `extraction`)",
        "",
        *[f"- {k}: {v}" for k, v in ext.items() if not isinstance(v, dict)],
        f"- dataset_id: {side.get('dataset_id')}",
        f"- stage: {side.get('stage')}, rows {side.get('row_count')}, created {side.get('created_at')}",
        "",
        "## Sampling",
        "",
        f"- sampling: the SHARED rollouts: block (T {samp['temperature']}, top_p {samp['top_p']}, "
        f"top_k {samp['top_k']}) as every MAEMM arm. What the checkpoint ships in "
        f"generation_config.json is on the rollouts summary as `shipped_generation_config`; "
        "recorded, not used.",
        f"- max_new {nla['max_new']}: the checkpoint's NATIVE length (2026-09-21) -- the "
        f"card's own invocation is `--max-new-tokens {nla['card_max_new']}` and that is the "
        "reference script's default too. NOT the pipeline's `rollouts.max_new`, which every MAEMM "
        "arm uses, so this arm's generation length is not comparable with theirs.",
        f"- scored in a **{nla['score_max_tokens']}-token window**, not the pipeline's "
        f"`common.SCORE_MAX_LENGTH` = {C.SCORE_MAX_LENGTH}: a 95-token cut would score less than "
        f"half of a {nla['max_new']}-token answer. `rollouts_nla` puts it on the rollouts summary "
        "as `score_max_length`, `score` re-encodes at it and writes it into its `rows.json`, and "
        "`common.score_width_of` is how a reader gets the stored width. A cosine from this arm is "
        "therefore a max over a WIDER window than every other arm's -- a stated deviation from the "
        "one scoring protocol, not an oversight.",
        "",
        "## Input amplitude (`--amp`)",
        "",
        "The verbalizer was trained on the RAW, uncentred layer-42 residual; our held-out "
        "`realact` directions are `unit(X[p] - mu)`. Because the hook normalises, the amplitude is "
        "not a scale but a MIXING RATIO between `mu` and the direction:",
        "",
        "- `raw` -- `x = r*u`, no `mu` at all: the scorer's own target, which is not what this "
        "model was trained to read.",
        "- `mu` -- `x = mu + r*u`, the uncentred reconstruction at a typical corpus amplitude.",
        "- `exact` -- `x = mu + t*u` with `t` solving `||mu + t*u|| = act_norm`, the row's own raw "
        "norm as `targets` recorded it; falls back to `mu` for a row that has none (every `sae`, "
        "`random` and `sae2m_enc` row).",
        "",
        f"`r` comes from `nla.amp_r` ({nla['amp_r']!r}). The DEFAULT amp ({nla['amp']!r}) writes "
        "`rollouts/<set>.jsonl`; any other writes `variants/<set>__amp-<amp>/rollouts.jsonl`, for "
        "`score --rollouts-dir`.",
        "",
        "## Weight identity",
        "",
        f"- path: {sha['path']}",
        f"- combined sha256: `{sha['sha256']}`",
        f"- kind: {sha.get('kind', 'streamed sha256 of every *.safetensors / *.bin / *.pt')}",
        "",
    ]
    if "shards" in sha:
        lines += [f"- index.json sha256: `{sha['index_sha256']}`", "", "| shard | bytes |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in sha["shards"].items()]
    else:
        lines += ["| file | bytes | sha256 |", "|---|---|---|"]
        lines += [f"| `{k}` | {v['bytes']} | `{v['sha256']}` |" for k, v in sha["files"].items()]
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[nla] wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------------------------
# the product
# ---------------------------------------------------------------------------------------------


def run(cfg, args):
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base, "product rollouts_nla needs --base"
    assert maemm, "product rollouts_nla needs --maemm"
    assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    key_base, _ = C.split_key(maemm, "maemm")
    assert key_base == base, f"maemm {maemm!r} is on base {key_base!r}, not {base!r}"
    spec = cfg["maemms"][maemm]
    assert spec["type"] == "nla", (
        f"maemm {maemm!r} is type {spec['type']!r}, not 'nla': use rollouts_hf (or rollouts_vllm) "
        f"-- this product builds the verbalizer's own prompt and marker, which no MAEMM has"
    )

    nla = spec["nla"]
    rl = cfg["rollouts"]
    n = int(args.get("n") or nla["n"])
    max_new = int(args.get("max_new") or nla["max_new"])
    score_max_length = int(nla["score_max_tokens"])
    # NOT bounded by rollouts.max_new: this arm carries its own scoring window (see the module
    # docstring). The rule is the same one SCORE_MAX_LENGTH enforces everywhere else -- the
    # window must hold the whole generation plus the sink -- applied to that window.
    assert max_new <= score_max_length - 1, (
        f"--max-new {max_new} exceeds this arm's scoring window nla.score_max_tokens "
        f"{score_max_length} minus the sink: the tail would never be scored"
    )
    # The SHARED sampling constants -- the same T / top_p / top_k every MAEMM arm uses.
    # ... with ONE per-MAEMM override: `nla.min_new`. The shared block's 16 forces this
    # verbalizer past its own stop token, and the shared block is never edited because that would
    # re-point every rollout product in the pipeline; so the key lives on the `nla:` sub-block and
    # the shared value is the fallback (config.yaml maemms.<nla>.nla.min_new).
    samp = {"temperature": rl["temperature"], "top_p": rl["top_p"], "top_k": rl["top_k"],
            "min_new": nla["min_new"] if "min_new" in nla else rl["min_new"]}
    min_new = int(samp["min_new"])
    base_seed = int(rl["seed"])
    inj_layer, coef = int(spec["inject"]["layer"]), float(spec["inject"]["coef"])
    gen_rows = int(args.get("gen_rows") or rollouts_hf.GEN_ROWS[base])
    assert gen_rows > 0, f"--gen-rows must be positive, got {gen_rows}"
    amp = str(args.get("amp") or nla["amp"])
    assert amp in AMP_MODES, f"--amp {amp!r} is not one of {list(AMP_MODES)}"

    # The DEFAULT amp is the product's own output and accumulates in rollouts/ exactly as
    # rollouts_hf's does; any other amp is a VARIANT and gets its own one-shot directory in the
    # `score --rollouts-dir` layout, so a sweep over amps cannot overwrite the headline run and
    # cannot be mistaken for it either.
    variant = amp != nla["amp"]
    if variant:
        out_dir = C.nla_variant_dir(maemm, set_name, amp, root)
        path = f"{out_dir}/rollouts.jsonl"
        stem, summary_name = "rollouts", "rollouts.summary.json"
    else:
        out_dir = C.rollouts_dir(maemm, root)
        # Through common.rollout_stem, like rollouts_hf and rollouts_vllm: the HF-shaped stem IS
        # the bare set name, so spelling it here quietly ignored `--run-tag` and two runs of one
        # checkpoint on one set differing only in --mu would both claim `<set>.jsonl`.
        stem = C.rollout_chunk_stem(
            C.rollout_stem(set_name, "hf", args.get("run_tag") or ""), args.get("rows", "")
        )
        path = f"{out_dir}/{stem}.jsonl"
        summary_name = f"{stem}.summary.json"
    assert args.get("force") or not os.path.exists(path), (
        f"{path} already exists; refusing to overwrite without --force"
    )

    check_amp_storage(amp, C.set_storage(cfg, C.heldout_dir(base, set_name, root), root)["storage"])

    cen_notes: list[str] = []
    rows_meta, dirs, dirs_src = rollouts_hf.load_dirs(cfg, args, device="cpu", notes=cen_notes)
    sel = C.parse_rows(args.get("rows", ""), len(rows_meta))
    u = dirs[sel].numpy().astype(np.float32)
    mu = C.stats_mu(cfg, base, root) if amp in ("mu", "exact") else None
    r, r_src = resolve_r(cfg, base, root, nla["amp_r"])
    x, info = build_inputs(u, [rows_meta[i] for i in sel], mu, amp, r)
    info_by_row = dict(zip(sel, info, strict=True))
    amp_used_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    for rec in info:
        amp_used_counts[rec["amp_used"]] = amp_used_counts.get(rec["amp_used"], 0) + 1
        if rec["fallback"]:
            fallback_counts[rec["fallback"]] = fallback_counts.get(rec["fallback"], 0) + 1
    in_norms = np.array([rec["in_norm"] for rec in info], dtype=np.float64)
    cos_dirs = np.array([rec["cos_in_dir"] for rec in info], dtype=np.float64)
    n_ambiguous = sum(1 for rec in info if rec["exact_ambiguous"])
    print(
        f"[nla] {maemm} on {len(sel)} of {len(rows_meta)} targets x {n} texts = {len(sel) * n} "
        f"rows, {gen_rows} per generate call, dirs from {dirs_src}; amp {amp} (r={r:.4f} "
        f"[{r_src}]), amp_used {amp_used_counts}, fallbacks {fallback_counts or 'none'}, "
        f"{n_ambiguous} rows with a second positive root; "
        f"writing {'VARIANT ' if variant else ''}{path}",
        flush=True,
    )

    weights = C.maemm_weights_path(cfg, maemm)
    assert os.path.basename(weights) == spec["revision"], (
        f"maemm {maemm!r} resolved to {weights}, whose snapshot directory is "
        f"{os.path.basename(weights)!r} and not the pinned revision {spec['revision']!r}: the HF "
        f"cache holds a different commit of {spec['hf']} than config.yaml names"
    )
    side, shipped = check_sidecar(weights, spec, int(cfg["bases"][base]["read_layer"]),
                                  int(cfg["bases"][base]["d"]))
    # `load_maemm` takes its non-lora branch for anything whose type is not "lora": tokenizer +
    # AutoModelForCausalLM from the snapshot, bf16, sdpa -- which is exactly how the card says to
    # load this merged checkpoint. It returns the config `type` verbatim, so `kind` is "nla" here.
    model, tok, kind = C.load_maemm(cfg, base, maemm)
    assert kind == "nla", f"load_maemm returned kind {kind!r} for a type: nla entry"
    prompt, mpos = nla_prompt_ids(tok, spec)
    stop = C.eos_ids(tok, model)
    sub = C.get_layer(model, inj_layer)

    # OBSERVATION ONLY, and unlike rollouts_hf.marker_check there is NO assert to make: the
    # comparison there is served-vs-clean-base at the SAME marker and prompt, and this checkpoint
    # is fully merged (no adapter to disable) with a prompt and marker no clean base was ever
    # measured at. bases.<base>.marker_norm_base is the MAEMM prompt's number and is not
    # comparable. What proves the weights loaded is the revision assert and the weight sha above.
    hn_src = "none: the NLA marker/prompt has no clean-base reference"
    if args.get("no_marker_check"):
        # The flag means the same thing here as in rollouts_hf -- skip the extra forwards -- even
        # though here they cost one prompt and prove nothing on their own.
        hn_served = None
        hn_src = "skipped (--no-marker-check)"
    else:
        hn_served = C.marker_norm(model, prompt, mpos, inj_layer, adapter=True)
    print(
        f"[nla] prompt {len(prompt)} tokens, marker {nla['marker_id']} at {mpos}; marker ||h|| at "
        f"layer {inj_layer} = "
        + ("not measured" if hn_served is None else f"{hn_served:.3f}")
        + f" (observation only, {hn_src})",
        flush=True,
    )

    pairs = [(row, k) for row in sel for k in range(n)]
    x_by_row = {row: x[i] for i, row in enumerate(sel)}
    out_rows: list[dict] = []
    t0, gen_tok, n_calls = time.time(), 0, 0
    for s in range(0, len(pairs), gen_rows):
        chunk = pairs[s : s + gen_rows]
        seed = C.gen_seed_for(base_seed, chunk[0][0], chunk[0][1], n)
        vecs = [torch.from_numpy(x_by_row[row]).unsqueeze(0) for row, _ in chunk]
        hook = C.make_inject_hook(vecs, [[mpos]] * len(chunk), coef, "cuda", torch.bfloat16)
        ids = torch.tensor([list(prompt)] * len(chunk), dtype=torch.long, device="cuda")
        am = torch.ones_like(ids)
        assert ids.shape == (len(chunk), len(prompt)) and bool(am.all()), (
            f"every row shares the identical {len(prompt)}-token prompt, so the batch must be "
            f"rectangular and unpadded (padding moves the marker off its required neighbours and "
            f"the injection lands nowhere); got {tuple(ids.shape)}"
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with C.hooked(sub, hook), torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=am,
                do_sample=True,
                temperature=float(samp["temperature"]),
                top_p=float(samp["top_p"]),
                top_k=int(samp["top_k"]),
                min_p=0.0,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, len(prompt) :]
        gen_tok += int(new.numel())
        n_calls += 1
        out_rows += rows_from_generation(chunk, new.tolist(), stop, tok, seed, rows_meta, info_by_row)
        if n_calls % 10 == 1 or s + gen_rows >= len(pairs):
            el = time.time() - t0
            print(
                f"[nla] {len(out_rows)}/{len(pairs)} rows | {gen_tok / max(el, 1e-9):.0f} "
                f"gen tok/s | {el:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0

    # The "not all identical" guard rollouts_hf has. TWO things are different here and neither is
    # fixed: nla.n defaults to 4, not 64, and the output is a short templated
    # <explanation>...</explanation>, so two of four samples colliding is a great deal more likely
    # than it is at n=64 over free-form rollouts -- this can fire on a run that is perfectly fine.
    # And it runs AFTER the whole generate loop, so a trip costs the run's full GPU time. Left as
    # is deliberately: a shared-seed / greedy / no-injection bug has exactly this signature, and a
    # loud stop is the point.
    if n > 1:
        by_row: dict[int, list[dict]] = {}
        for rec in out_rows:
            by_row.setdefault(rec["row"], []).append(rec)
        for row, recs in by_row.items():
            assert len({y["text"] for y in recs}) > 1, (
                f"target row {row} ({recs[0]['family']}): all {len(recs)} texts decoded to the "
                f"IDENTICAL string {recs[0]['text']!r} -- expected sampling at "
                f"T={samp['temperature']} to differ; a shared-seed / greedy / no-injection bug "
                f"looks exactly like this"
            )
    n_tok = [y["n_tok"] for y in out_rows]
    expl_rate = float(np.mean([y["explanation"] is not None for y in out_rows]))
    summary = {
        "maemm": maemm,
        "base": base,
        "set": set_name,
        "dirs_from": dirs_src,
        "rows": sel,
        "n_targets": len(sel),
        "n": n,
        "bo": n,
        "seed": base_seed,
        "seed_rule": "rollouts.seed * 1000 + (row * n + k) of the generate call's first row",
        "engine": ENGINE,
        "kind": "nla",
        "prompt": PROMPT_NAME,
        "prompt_tokens": len(prompt),
        # OURS, not the checkpoint's: the reference script passes no enable_thinking and gets an
        # OPEN <think> block (110 tokens against our 112, marker at 93 either way). Recorded so
        # the choice is visible in the summary rather than implicit in the code.
        "enable_thinking": False,
        "marker_pos": mpos,
        "marker_id": int(nla["marker_id"]),
        "inject_layer": inj_layer,
        "inject_coef": coef,
        "temperature": float(samp["temperature"]),
        "top_p": float(samp["top_p"]),
        "top_k": int(samp["top_k"]),
        "min_p": 0.0,
        "max_new": max_new,
        "min_new": min_new,
        # What the CHECKPOINT ships in its generation_config.json, which is NOT what it was run
        # with: this arm samples under the shared `rollouts:` block like every other. Here so the
        # divergence is on the record rather than in a print the log rotates away.
        "shipped_generation_config": shipped,
        "sampling_source": "config.yaml rollouts: (shared with every MAEMM arm)",
        # `score` reads this off the summary and re-encodes at it instead of the protocol's
        # SCORE_MAX_LENGTH, then records it in its own rows.json (common.score_width_of). It
        # travels with the ROLLOUTS so the scorer cannot be pointed at a window this generation
        # never agreed to.
        "score_max_length": score_max_length,
        "gen_rows": gen_rows,
        "marker_norm_served": None if hn_served is None else round(hn_served, 4),
        "marker_norm_clean_base": None,
        "marker_norm_base_source": hn_src,
        "eos_ids": sorted(stop),
        "mean_n_tok": round(float(np.mean(n_tok)), 3),
        "eos_rate": round(float(np.mean([y["finished"] for y in out_rows])), 4),
        "gen_tok_per_s": round(gen_tok / max(elapsed, 1e-9), 1),
        "generate_seconds": round(elapsed, 1),
        "generate_calls": n_calls,
        # --- NLA-only -------------------------------------------------------------------------
        "nla_hf": spec["hf"],
        "nla_revision": spec["revision"],
        "snapshot": weights,
        "amp": amp,
        "amp_r": nla["amp_r"],
        "amp_r_source": r_src,
        "amp_r_resolved": round(r, 4),
        "mu_norm": None if mu is None else round(float(np.linalg.norm(mu)), 4),
        "amp_used_counts": amp_used_counts,
        "fallback_counts": fallback_counts,
        "in_norm_quantiles": _quantiles(in_norms),
        # cos(x, u): how far adding mu tilted the input away from the direction the SCORER
        # measures against. 1.0 everywhere under `raw`; below 1 under `mu` / `exact`.
        "cos_in_dir_quantiles": _quantiles(cos_dirs),
        "exact_ambiguous_count": n_ambiguous,
        "template_sha256": template_sha256(spec),
        "explanation_rate": round(expl_rate, 4),
        "card_max_new": int(nla["card_max_new"]),
    }

    sha = C.sha256_of_index(weights)
    summary["weight_sha256"] = sha["sha256"]
    write_nla_readme(cfg, args, maemm, sha, side, prompt, mpos)

    inputs = {
        "maemm": f"{maemm} (NLA verbalizer, {spec['hf']} @ {spec['revision'][:12]})",
        "dirs": dirs_src,
        "targets": f"{len(sel)} of {len(rows_meta)} rows",
        "n": n,
        "amp": f"{amp} (r={r:.4f} from {r_src})",
        "weight sha256": sha["sha256"],
    }
    # The shared `rollouts/` directory is ACCUMULATING and additive (common.OutDir); a
    # `--amp` variant gets its own one-shot directory and stays on the rename path.
    keep = not variant
    with C.outdir(out_dir, args, inputs=inputs, keep_existing=keep) as od:
        C.note_convention(od, cen_notes)
        od.write_jsonl(f"{stem}.jsonl", out_rows)
        od.write_json(summary_name, summary)
        od.note(
            f"`{stem}.jsonl`: one row per (target, text) -- row, family, k, text, ids (the "
            "GENERATED ids only, trimmed at the first stop token which is KEPT, train/rl/rl.py:82-90), "
            "n_tok, finished, engine, seed -- EXACTLY the schema rollouts_hf writes, plus the "
            "NLA-only amp, amp_used, r, in_norm and explanation, which score.py ignores. `text` is "
            "decode(ids, skip_special_tokens=True); the prompt is not part of either field."
        )
        od.note(
            f"sampling: the CHECKPOINT's own generation_config.json -- T={samp['temperature']} "
            f"top_p={samp['top_p']} top_k={samp['top_k']} min_p=0 min_new={min_new} "
            f"max_new={max_new}, the CHECKPOINT'S NATIVE length (the reference script decodes "
            f"GREEDILY at the same {nla['card_max_new']}; see ../README.md), {n} texts per target, "
            f"{gen_rows} "
            "rows per generate call, all rows of a call sharing the identical prompt so the batch "
            "is UNPADDED (asserted -- padding moves the marker off its required neighbours)"
        )
        od.note(
            f"seed rule: {summary['seed_rule']} (base seed {base_seed}); set with torch.manual_seed "
            "+ torch.cuda.manual_seed_all immediately before each generate and stored on every row"
        )
        od.note(
            f"input amplitude: amp={amp} (default {nla['amp']!r}), r={r:.4f} from {r_src}, "
            f"||mu||={summary['mu_norm']}. amp_used {amp_used_counts}, fallbacks "
            f"{fallback_counts or 'none'}; ||x|| quantiles [min, q25, q50, q75, max] = "
            f"{summary['in_norm_quantiles']}. ONLY the direction reaches the model (the hook "
            f"normalises), so the amplitude acts as the mixing ratio between mu and the target "
            f"direction, and cos(x, u) -- on every row and quantiled here as "
            f"{summary['cos_in_dir_quantiles']} -- is how far that mixing tilted the input away "
            f"from the direction the SCORER measures against. {n_ambiguous} row(s) had a SECOND "
            f"positive root (act_norm < ||mu|| and mu.u < 0; the larger root is taken and the row "
            f"carries exact_ambiguous). See ../README.md for what each amp means."
        )
        od.note(
            f"marker ||h|| at inject layer {inj_layer} under the served verbalizer: "
            + ("NOT MEASURED" if hn_served is None else f"{hn_served:.4f}")
            + f". OBSERVATION ONLY and NOT compared: {hn_src}. The merged checkpoint "
            f"has no adapter to disable, and bases.{base}.marker_norm_base was measured at the "
            f"MAEMM prompt's marker, a different token at a different position. What proves these "
            f"weights loaded is the pinned revision and the sha below."
        )
        od.note(
            f"weights: {sha['path']} sha256 {sha['sha256']} "
            f"({sha.get('kind', 'streamed content hash')}); identity card in ../README.md"
        )
        od.note(
            f"SCORING WINDOW: these rows carry `score_max_length` {score_max_length} in the "
            f"summary, so `score` re-encodes them at {score_max_length} tokens instead of the "
            f"pipeline's common.SCORE_MAX_LENGTH={C.SCORE_MAX_LENGTH} and writes [N, n, "
            f"{score_max_length + 1}] arrays (common.score_width_of reads the width back). A "
            f"{C.SCORE_MAX_LENGTH}-token cut would score less than half of a {max_new}-token "
            f"answer. A cosine from this arm is therefore a max over a WIDER window than every "
            f"other arm's -- a STATED deviation from the one scoring protocol, and the reason the "
            f"generation length is not the MAEMMs' either."
        )
        od.note(
            f"`explanation`: the first <explanation>...</explanation> body (nla/schema.py:45-54), "
            f"INFORMATIONAL -- {expl_rate:.1%} of rows carry a closed tag. What is SCORED is "
            f"`text`, the full decode, never the stripped explanation."
        )
        od.note(
            f"throughput: {summary['gen_tok_per_s']} generated tok/s over {elapsed:.0f}s in "
            f"{n_calls} generate calls; mean {summary['mean_n_tok']} kept tokens per row, "
            f"eos rate {summary['eos_rate']}"
        )
        od.note(
            "this directory ACCUMULATES one <set>.jsonl + <set>.summary.json per run; index.json "
            "lists every set present, while the header above describes the MOST RECENT run only"
            if not variant
            else f"a VARIANT directory: `--amp {amp}` is not this entry's default "
            f"({nla['amp']!r}), so the run lands here in the `score --rollouts-dir` layout rather "
            f"than in the accumulating rollouts/ directory. Score it with "
            f"`--product score --rollouts-dir {out_dir}`."
        )
    return {
        "out": path,
        "rows": len(out_rows),
        "targets": len(sel),
        "amp": amp,
        "explanation_rate": summary["explanation_rate"],
        **{k: summary[k] for k in ("marker_norm_served", "marker_norm_clean_base", "gen_tok_per_s")},
        "mean_n_tok": summary["mean_n_tok"],
        "eos_rate": summary["eos_rate"],
        "weight_sha256": sha["sha256"],
    }


# ---------------------------------------------------------------------------------------------
# --selftest: the pure pieces against independent computations (no torch, no weights, no network)
# ---------------------------------------------------------------------------------------------


class _StubTok:
    """Enough tokenizer for the prompt and decode paths: a fixed template, a fixed marker id.

    `apply_chat_template` returns HEADER + the content's ids + TAIL, where the content's ids come
    from a toy encoder that maps the marker char to `marker_id` and every other character to its
    ordinal. That is enough for nla_prompt_ids' whole contract (single occurrence, both
    neighbours, not the last token) and lets the neighbour asserts be provoked on demand.
    """

    def __init__(self, marker="M", marker_id=900, left=29, right=510, tail=(7, 8)):
        self.marker, self.marker_id, self.left, self.right, self.tail = marker, marker_id, left, right, tail
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [self.marker_id if ch == self.marker else ord(ch) for ch in text]

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        assert tokenize and add_generation_prompt and not enable_thinking
        body = []
        for ch in msgs[0]["content"]:
            body.append(self.marker_id if ch == self.marker else ord(ch))
        return [1, 2, *body, *self.tail]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in (0, 3)))


def _selftest_amp_storage():
    """`mu`/`exact` are refused on a raw set and allowed on a centred one; `raw` always passes."""
    for amp in ("mu", "exact"):
        try:
            check_amp_storage(amp, "raw")
        except AssertionError as e:
            assert "already-uncentred" in str(e), f"wrong refusal text for {amp}: {e}"
        else:
            raise AssertionError(f"--amp {amp} was accepted on a `storage: raw` set")
        check_amp_storage(amp, "unit")   # still the right tool on the contract it was written for
    for storage in ("raw", "unit", "dirs_only"):
        check_amp_storage("raw", storage)


def _selftest_build_inputs():
    rng = np.random.default_rng(20260920)
    d, n = 64, 24
    mu = rng.normal(size=d).astype(np.float32) * 3.0
    u = rng.normal(size=(n, d)).astype(np.float32)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    mu_norm = float(np.linalg.norm(mu))
    # act_norm comfortably above ||mu||, so every row has a solution; two rows carry none.
    meta = [{"family": "realact", "act_norm": float(mu_norm * (1.2 + 0.4 * i / n))} for i in range(n)]
    meta[3].pop("act_norm")
    meta[7]["act_norm"] = None

    x, info = build_inputs(u, meta, mu, "exact", 10.0)
    solved = [i for i, rec in enumerate(info) if rec["amp_used"] == "exact"]
    assert len(solved) == n - 2, f"expected {n - 2} solved rows, got {len(solved)}"
    for i in solved:
        want = meta[i]["act_norm"]
        got = float(np.linalg.norm(x[i].astype(np.float64)))
        assert abs(got - want) < 1e-4 * max(want, 1.0), f"row {i}: ||mu + t u|| = {got}, want {want}"
    assert info[3]["fallback"] == "no_act_norm" and info[3]["amp_used"] == "mu", info[3]
    assert info[7]["fallback"] == "no_act_norm", info[7]
    assert info[3]["r"] == 10.0, "a no_act_norm row falls back to mu AT r"

    assert not any(rec["exact_ambiguous"] for rec in info), (
        "every act_norm here is > ||mu||, so each solve has a UNIQUE positive root"
    )

    # act_norm < ||mu|| AND mu.u < 0 -> the smaller root is positive too. Construct it: take a
    # direction with mu.u < 0 and a norm between the line's closest approach and ||mu||.
    u0 = -mu.astype(np.float64) / mu_norm
    b0 = float(mu.astype(np.float64) @ u0)
    assert b0 < 0, "u0 was built to point against mu"
    amb = [{"family": "realact", "act_norm": 0.5 * mu_norm}]  # closest approach is 0 here
    xa, ia = build_inputs(u0.astype(np.float32)[None, :], amb, mu, "exact", 10.0)
    assert ia[0]["amp_used"] == "exact" and ia[0]["fallback"] is None, ia[0]
    assert ia[0]["exact_ambiguous"] is True, "a < ||mu|| with mu.u < 0 has TWO positive roots"
    got = float(np.linalg.norm(xa[0].astype(np.float64)))
    assert abs(got - 0.5 * mu_norm) < 1e-3 * mu_norm, f"the solve must still hold: {got}"
    t_big, t_small = ia[0]["r"], 2.0 * (-b0) - ia[0]["r"]
    assert t_big > t_small > 0, f"the LARGER root must be the one taken: {t_big} vs {t_small}"

    # a row whose recorded norm is BELOW the line's closest approach: discriminant < 0
    close = [dict(meta[0]) for _ in range(1)]
    close[0]["act_norm"] = 1e-3
    xs, infos = build_inputs(u[:1], close, mu, "exact", 10.0)
    assert infos[0]["fallback"] == "discriminant" and infos[0]["amp_used"] == "mu", infos[0]
    assert infos[0]["exact_ambiguous"] is False, "a fallback row solved nothing, ambiguously or not"
    assert abs(float(np.linalg.norm(xs[0] - (mu + 10.0 * u[0])))) < 1e-3, "fallback must be mu + r*u"

    # `raw` twice: once with no mu at all and once with mu IN HAND, because a "raw" branch that
    # forgot to skip mu is invisible when mu is None (it adds a zero vector).
    for passed_mu in (None, mu):
        xr, ir = build_inputs(u, meta, passed_mu, "raw", 7.5)
        assert np.allclose(np.linalg.norm(xr, axis=-1), 7.5, atol=1e-3), "raw: ||x|| == r"
        assert all(abs(rec["cos_in_dir"] - 1.0) < 1e-6 for rec in ir), "raw: x is parallel to u"
        assert np.allclose(xr, 7.5 * u, atol=1e-3), "raw: x is r*u exactly -- no mu anywhere"
        assert {rec["amp_used"] for rec in ir} == {"raw"}
        assert {rec["r"] for rec in ir} == {7.5} and {rec["fallback"] for rec in ir} == {None}

    xm, im = build_inputs(u, meta, mu, "mu", 7.5)
    for i in range(n):
        want = float(np.linalg.norm(mu.astype(np.float64) + 7.5 * u[i].astype(np.float64)))
        assert abs(float(np.linalg.norm(xm[i].astype(np.float64))) - want) < 1e-3
        assert im[i]["amp_used"] == "mu" and im[i]["fallback"] is None
        assert im[i]["cos_in_dir"] < 1.0, "adding mu must tilt the direction away from u"

    for bad, want in (("nonsense", "unknown amp"), ("exact", "stats/mu.f32")):
        try:
            build_inputs(u, meta, None if bad == "exact" else mu, bad, 1.0)
        except AssertionError as e:
            assert want in str(e), f"amp {bad!r}: wrong assert fired: {e}"
        else:
            raise AssertionError(f"build_inputs accepted amp={bad!r}")
    try:
        build_inputs(u * 3.0, meta, mu, "mu", 1.0)
    except AssertionError as e:
        assert "UNIT direction rows" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("build_inputs accepted non-unit rows")


def _selftest_extract_explanation():
    assert extract_explanation("<explanation>  a b  </explanation>") == "a b", "hit + strip"
    assert extract_explanation("no tags here") is None, "miss"
    assert extract_explanation("<explanation>open but never closed") is None, "unclosed tag is a miss"
    assert extract_explanation("x\n<explanation>\nl1\nl2\n</explanation>\ny") == "l1\nl2", "DOTALL"
    assert extract_explanation("<explanation>a</explanation><explanation>b</explanation>") == "a", "first"


def _selftest_resolve_r():
    import tempfile

    cfg = {"bases": {"b": {"read_layer": 2, "d": 8}}}
    qjson = {
        "quantiles": [0.01, 0.25, 0.50, 0.75, 0.99],
        "block_output": True,
        "layers": [{"layer": i, "q": [1.0, 2.0, 10.0 + i, 4.0, 5.0]} for i in range(4)],
    }
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/base/b/stats")
        with open(f"{td}/base/b/stats/resid_norm_quantiles.json", "w") as fh:
            json.dump(qjson, fh)
        r, src = resolve_r(cfg, "b", td, "median")
        assert r == 12.0, f"layer 2's q[index of 0.50] is 12.0, got {r}"
        assert "resid_norm_quantiles.json" in src and "0.50" in src, src

        # a layers list written for a different model: the `layer` field no longer matches its index
        bad = dict(qjson, layers=[{"layer": i + 1, "q": q["q"]} for i, q in enumerate(qjson["layers"])])
        with open(f"{td}/base/b/stats/resid_norm_quantiles.json", "w") as fh:
            json.dump(bad, fh)
        try:
            resolve_r(cfg, "b", td, "median")
        except AssertionError as e:
            assert "indexed BY layer" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("resolve_r accepted a shifted layers list")

        # block_output must be declared: an input-side quantile is a different number
        bad2 = dict(qjson, block_output=False)
        with open(f"{td}/base/b/stats/resid_norm_quantiles.json", "w") as fh:
            json.dump(bad2, fh)
        try:
            resolve_r(cfg, "b", td, "median")
        except AssertionError as e:
            assert "block_output" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("resolve_r accepted block_output: false")

    r, src = resolve_r(cfg, "b", "/nonexistent", 93.259)
    assert r == 93.259 and "config.yaml" in src, (r, src)


def _selftest_nla_prompt_ids():
    tok = _StubTok()
    spec = {
        "nla": {
            "marker": "M",
            "marker_id": 900,
            "left_id": ord("<"),
            "right_id": ord(">"),
            "template": "read this: <{injection_char}> please",
        }
    }
    ids, mpos = nla_prompt_ids(tok, spec)
    assert ids[mpos] == 900 and ids[mpos - 1] == ord("<") and ids[mpos + 1] == ord(">")
    assert mpos < len(ids) - 1, "the generation prompt must follow the marker"

    for template, want in (
        ("read this: [{injection_char}] please", "requires"),  # wrong neighbours
        ("{injection_char} and {injection_char}", "exactly one marker"),  # two markers
        ("no marker at all", "exactly one marker"),
    ):
        try:
            nla_prompt_ids(tok, {"nla": dict(spec["nla"], template=template)})
        except AssertionError as e:
            assert want in str(e), f"template {template!r}: wrong assert fired: {e}"
        else:
            raise AssertionError(f"nla_prompt_ids accepted {template!r}")

    # the marker as the LAST content token: the stub's tail keeps it legal, so shorten the tail
    bare = _StubTok(tail=())
    try:
        nla_prompt_ids(bare, {"nla": dict(spec["nla"], template="<{injection_char}")})
    except AssertionError as e:
        assert "needs a token on BOTH sides" in str(e) or "requires" in str(e), f"wrong assert: {e}"
    else:
        raise AssertionError("nla_prompt_ids accepted a trailing marker")

    # a tokenizer whose marker id disagrees with the sidecar
    try:
        nla_prompt_ids(_StubTok(marker_id=42), spec)
    except AssertionError as e:
        assert "nla_meta.yaml says injection_token_id" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("nla_prompt_ids accepted a marker-id mismatch")


def _selftest_rows_from_generation():
    tok = _StubTok()
    stop = {ord("Z")}
    rows_meta = [{"row": 0, "family": "realact"}, {"row": 1, "family": "sae"}]
    info = {
        0: {
            "amp": "exact",
            "amp_used": "exact",
            "r": 25.0,
            "in_norm": 93.2,
            "cos_in_dir": 0.7312,
            "fallback": None,
            "exact_ambiguous": False,
        },
        1: {
            "amp": "exact",
            "amp_used": "mu",
            "r": 10.0,
            "in_norm": 68.1,
            "cos_in_dir": 0.1455,
            "fallback": "no_act_norm",
            "exact_ambiguous": True,
        },
    }
    body = [ord(c) for c in "<explanation>ab</explanation>"]
    gen = [
        [*body, ord("Z"), ord("q"), ord("q")],  # stops, with a tail the trim must drop
        [ord("n"), ord("o")],  # never stops
    ]
    out = rows_from_generation([(0, 0), (1, 3)], gen, stop, tok, 1234000, rows_meta, info)
    assert [y["row"] for y in out] == [0, 1] and [y["k"] for y in out] == [0, 3]
    assert out[0]["n_tok"] == len(body) + 1 and out[0]["finished"] is True, out[0]
    assert out[0]["ids"][-1] == ord("Z"), "the stop token is KEPT"
    assert out[0]["text"] == "<explanation>ab</explanation>Z", "text is the FULL decode"
    assert out[0]["explanation"] == "ab"
    assert out[1]["finished"] is False and out[1]["explanation"] is None and out[1]["text"] == "no"
    assert out[0]["family"] == "realact" and out[1]["family"] == "sae"
    for y, src in zip(out, (info[0], info[1]), strict=True):
        assert y["engine"] == "hf" and y["seed"] == 1234000
        for key in ("amp", "amp_used", "r", "in_norm", "cos_in_dir", "exact_ambiguous"):
            assert key in y, f"row is missing the NLA field {key!r}"
            assert y[key] == src[key], f"row {key} is {y[key]!r}, build_inputs said {src[key]!r}"


SELFTESTS = (
    _selftest_amp_storage,
    _selftest_build_inputs,
    _selftest_extract_explanation,
    _selftest_resolve_r,
    _selftest_nla_prompt_ids,
    _selftest_rows_from_generation,
)


def selftest() -> list[str]:
    """Run every pure-piece check. Imported by unit_smoke.check_rollouts_nla_selftest too."""
    names = []
    for fn in SELFTESTS:
        t0 = time.time()
        fn()
        print(f"[nla-selftest] ok  {fn.__name__:<32} {time.time() - t0:5.2f}s", flush=True)
        names.append(fn.__name__)
    print(f"[nla-selftest] {len(names)}/{len(SELFTESTS)} checks passed", flush=True)
    return names


if __name__ == "__main__":
    assert "--selftest" in sys.argv, (
        "this module is a Modal product (`--product rollouts_nla`); the only thing it does "
        "standalone is `uv run precompute/rollouts_nla.py --selftest`"
    )
    selftest()
