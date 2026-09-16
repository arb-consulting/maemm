"""Stage `run`: the LLM half -- Delphi's explainer, then its detection and fuzzing scorers.

    <root>/runs/<date>_autointerp-<tag>/
        cache/<sha256>.json   ONE file per (job, prompt): the response text and its usage. The
                              cache is the run's durable state -- every product below is a pure
                              function of it, so a container that dies is resumed by re-launching
                              and costs nothing for the calls already made.
        explain/              explanations.jsonl, costs.json
        detection/            batches.jsonl, per_feature.jsonl, costs.json
        fuzzing/              batches.jsonl, per_feature.jsonl, costs.json
        summary/              scores.jsonl (feature x arm x scorer), costs.json, features.json

Everything between `# ---- Delphi` markers is TRANSCRIBED from EleutherAI/delphi pinned to
DELPHI_COMMIT, by way of repo-maemm/eval/autointerp_detection.py:229-450, which records the raw
file URLs and the fetch date. The prompts are copied rather than imported: that file lives in
another worktree, is read-only, and carries `mxf`/torch imports this CPU container must not need.

THE API IS ANTHROPIC'S MESSAGES API, DIRECTLY (Tomas, 2026-09-16), not OpenRouter. Two paths,
`--path sync` (bounded thread pool) and `--path batch` (one Message Batch per stage, half price);
the pilot measures batch latency so the full run can choose. See the `# ---- Anthropic client`
header for the two things the move changes, the larger of which is that `temperature` NO LONGER
EXISTS on this API for this model generation, so the design's "temperature 0" is unachievable and
the A7 null arm is the only noise floor left.

The API key arrives as the Modal secret `anthropic` (env `ANTHROPIC_API_KEY`) and is never
printed, never written to the volume and never put in a README. Only call counts, token counts and
dollars are recorded -- and the dollars are COMPUTED from the token counts at a rate table that
goes into costs.json beside them, because the Anthropic API returns no cost field.

Two phases, in this order for a reason: EVERY feature is explained first, then features are scored
one at a time. The floor arm (A6) scores a feature's test set with a DIFFERENT feature's
description, so it cannot run inside a feature-at-a-time loop that has not written that
description yet.

Arms `run` adds to the build's, neither of which has an explainer call of its own:
  * `R-shuffled` (A6) -- the floor: another feature's C16 description, under a fixed derangement.
  * `C16-draw2` (A7) -- the null: C16's own description on the second, disjoint test draw. The
    per-feature C16 - C16-draw2 difference is the test-set sampling noise every contrast is
    exposed to, and it replaces the temperature-0 repeat, which measured judge jitter instead.

Cost control, two layers, BOTH now acting before money is spent. Every stage is PROJECTED first
(`Claude.project`: `messages.count_tokens` on a sample of the uncached jobs, times the rate table),
and the stage is refused if the projection exceeds `autointerp.stop_above_usd` without
`--approved` -- that is the "anything over $100 gets reported before it runs" rule made mechanical
rather than left to notice. `autointerp.max_cost_usd` is then a hard ceiling checked before every
individual submission on the sync path, and once per batch on the batch path.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

import precompute.common as C

# ---- Delphi (EleutherAI) protocol ------------------------------------------------------------
# Transcribed from EleutherAI/delphi @ DELPHI_COMMIT via
# related-work/2026-09-13_autointerp-methods-and-prompts.md §3.1-3.3 and
# related-work/2026-09-15_delphi-updates-and-negatives.md, by way of
# repo-maemm/eval/autointerp_detection.py:229-450. Nothing here is written from memory.
#
#   explainer prompts  https://raw.githubusercontent.com/EleutherAI/delphi/
#                      4fea06e6e8b68eeaf302474325fca13df95c5d6f/delphi/explainers/default/prompts.py
#   detection prompt   .../delphi/scorers/classifier/prompts/detection_prompt.py
#   fuzz prompt        .../delphi/scorers/classifier/prompts/fuzz_prompt.py
DELPHI_COMMIT = "4fea06e6e8b68eeaf302474325fca13df95c5d6f"  # main, 2026-08-25
DELPHI_REPO = "https://github.com/EleutherAI/delphi"

DELPHI_EXPLAINER_SYSTEM = (
    "You are a meticulous AI researcher conducting an important investigation into patterns "
    "found in language. Your task is to analyze text and provide an explanation that thoroughly "
    "encapsulates possible patterns found in it.\n"
    "Guidelines:\n\n"
    "You will be given a list of text examples on which special words are selected and between "
    "delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire "
    "sequence of tokens will be contained between delimiters <<just like this>>. How important "
    "each token is for the behavior is listed after each example in parentheses.\n\n"
    "- Try to produce a concise final description. Simply describe the text latents that are "
    "common in the examples, and what patterns you found.\n"
    "- If the examples are uninformative, you don't need to mention them. Don't focus on giving "
    "examples of important tokens, but try to summarize the patterns found in the examples.\n"
    "- Do not mention the marker tokens (<< >>) in your explanation.\n"
    "- Do not make lists of possible explanations. Keep your explanations short and concise.\n"
    "- The last line of your response must be the formatted explanation, using [EXPLANATION]:"
)

# prompts.py::EXAMPLE_1 / EXAMPLE_1_ACTIVATIONS / EXAMPLE_1_EXPLANATION, verbatim, sent as a real
# user/assistant turn pair the way Delphi sends it. The comma form ("over", 5) here against the
# colon form ("over" : 5) the real examples use is Delphi's own inconsistency and is preserved.
DELPHI_EXPLAINER_FEWSHOT = [
    {
        "role": "user",
        "content": (
            "Example 1:  and he was <<over the moon>> to find\n"
            'Activations: ("over", 5), (" the", 6), (" moon", 9)\n'
            "Example 2:  we'll be laughing <<till the cows come home>>! Pro\n"
            'Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)\n'
            "Example 3:  thought Scotland was boring, but really there's more <<than meets the "
            "eye>>! I'd\n"
            'Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)'
        ),
    },
    {
        "role": "assistant",
        "content": "[EXPLANATION]: Common idioms in text conveying positive sentiment.",
    },
]

# scorers/classifier/prompts/detection_prompt.py::DSCORER_SYSTEM_PROMPT, verbatim.
DELPHI_DETECTION_SYSTEM = (
    "You are an intelligent and meticulous linguistics researcher.\n\n"
    'You will be given a certain latent of text, such as "male pronouns" or "text with '
    'negative sentiment".\n\n'
    "You will then be given several text examples. Your task is to determine which examples "
    "possess the latent.\n\n"
    "For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens "
    "are mislabeled. You must return your response in a valid Python list. Do not return "
    "anything else besides a Python list."
)

# scorers/classifier/prompts/fuzz_prompt.py::FUZZ_SYSTEM_PROMPT, verbatim (fetched 2026-09-13,
# recorded at related-work/2026-09-13_autointerp-methods-and-prompts.md:411-424). Delphi's fuzzing
# FEW-SHOT turns are NOT in our transcription, so the fuzzing scorer runs ZERO-SHOT here and the
# detection scorer keeps its three verbatim shots. That asymmetry is stated in every fuzzing
# README: it is a reason fuzzing and detection numbers are not comparable to each other, not a
# reason either is wrong.
DELPHI_FUZZ_SYSTEM = (
    "You are an intelligent and meticulous linguistics researcher.\n\n"
    'You will be given a certain latent of text, such as "male pronouns" or "text with negative '
    "sentiment\". You will be given a few examples of text that contain this latent. Portions of "
    "the sentence which strongly represent this latent are between tokens << and >>.\n\n"
    "Some examples might be mislabeled. Your task is to determine if every single token within "
    "<< and >> is correctly labeled. Consider that all provided examples could be correct, none "
    "of the examples could be correct, or a mix. An example is only correct if every marked token "
    "is representative of the latent\n\n"
    "For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens "
    "are mislabeled. You must return your response in a valid Python list. Do not return anything "
    "else besides a Python list."
)

# scorers/classifier/prompts/detection_prompt.py::DSCORER_EXAMPLE_{ONE,TWO,THREE} and their
# responses, verbatim. The mojibake (`Ċ`, `âĢĻ`, `<|endoftext|>`) is GPT-2 byte-level token
# artifact in Delphi's own source and stays. The three answers are 2/5, 0/5 and 5/5 positive: the
# shots teach the OUTPUT FORMAT and deliberately not a balanced prior.
DELPHI_DETECTION_FEWSHOT = [
    {"role": "user", "content": """Latent explanation: Words related to American football positions, specifically the tight end position.

Test examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: names of months used in The Lord of the Rings:ĊĊâĢľâĢ¦the
Example 2: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 4: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
"""},  # noqa: E501
    {"role": "assistant", "content": """[1,0,0,0,1]"""},
    {"role": "user", "content": """Latent explanation: The word "guys" in the phrase "you guys".

Test examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle
Example 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because
Example 3:Ċ<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre
Example 4: the<|endoftext|>ľI want to remind you all that 10 days ago (director Massimil
"""},  # noqa: E501
    {"role": "assistant", "content": """[0,0,0,0,0]"""},
    {"role": "user", "content": """Latent explanation: "of" before words that start with a capital letter.

Test examples:

Example 0: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 2:ĊĊDeborah Sathe, Head of Talent Development and Production at Film London,
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
"""},  # noqa: E501
    {"role": "assistant", "content": """[1,1,1,1,1]"""},
]


def parse_delphi_explanation(txt: str) -> str:
    """Delphi asks for '[EXPLANATION]: ...' as the LAST line: take what follows the last marker.

    With no marker, fall back to the whole response -- a model that answered without the tag still
    answered. "" means nothing usable, which the caller records as a gap.
    """
    if not txt:
        return ""
    i = txt.rfind("[EXPLANATION]:")
    body = txt[i + len("[EXPLANATION]:") :] if i >= 0 else txt
    return " ".join(body.split()).strip()


def delphi_scorer_prompt(explanation: str, texts: list[str]) -> str:
    """Delphi's per-query user message, `GENERATION_PROMPT` filled with 0-based `Example i:` lines.

    The 0-based numbering and the line shape come from the fifteen few-shot lines above. Two
    inconsistencies of Delphi's own are preserved: the shots head their block `Test examples:`
    while the template says `Text examples:`, and the detection system prompt asks "which examples
    possess the latent" but defines 1/0 as "correctly labeled"/"mislabeled" (wording inherited from
    the fuzzing prompt, where it is the right question).
    """
    block = "\n".join(f"Example {i}: {t}" for i, t in enumerate(texts))
    return f"Latent explanation: {explanation}\n\nText examples:\n\n{block}\n"


_DELPHI_BOOL = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}


def parse_delphi_scores(txt: str, n: int):
    """Delphi's classifier answer -> n ints in {0,1}, or None when it cannot be read.

    MEASURED 2026-09-15 on the live probe: Sonnet 5 narrates before answering on roughly one batch
    in six. So EVERY bracketed group is tried and the LAST one of exactly length n wins, which
    passes over a quoted list inside the narration. A list of the WRONG length is refused, not
    padded: padding would score real items against answers the judge never gave.
    """
    if not txt:
        return None
    best = None
    for m in re.finditer(r"\[[^\[\]]*\]", txt, re.S):
        body = m.group()[1:-1].strip()
        if not body:
            continue
        vals = []
        for part in body.split(","):
            v = _DELPHI_BOOL.get(part.strip().strip("'\"").lower())
            if v is None:
                vals = None
                break
            vals.append(v)
        if vals is not None and len(vals) == n:
            best = vals
    return best


# ---- Anthropic client (Messages API, direct; Tomas 2026-09-16, replacing OpenRouter) ---------
#
# Two paths, both through the official `anthropic` SDK:
#   "sync"  -- messages.create through a bounded thread pool. Latency is one request.
#   "batch" -- messages.batches, ONE batch per stage over every feature, at HALF PRICE. Latency is
#              the batch's, which is why the pilot measures it before the full run commits.
#
# TWO THINGS THE DIRECT API CHANGES, both MEASURED 2026-09-16 against `claude-sonnet-5`:
#
#  1. `temperature` IS GONE. Not "rejected": `anthropic` 1.6.0's `messages.create()` has no such
#     parameter (`TypeError: unexpected keyword argument 'temperature'`), because sampling
#     parameters were removed from the API for this model generation. The design's "temperature 0"
#     is therefore NOT ACHIEVABLE on this surface and every call runs at the model's own sampling.
#     OpenRouter accepted the parameter, which is what hid this. Consequences, stated rather than
#     absorbed: run-to-run variation is now real, and the A7 null arm -- the same C16 description
#     scored on a second disjoint test draw -- is the only noise floor this evaluation has.
#  2. Thinking is on by default on this generation, so `thinking: {"type": "disabled"}` is sent
#     explicitly. It is the direct equivalent of OpenRouter's `reasoning: {"enabled": False}` and
#     is accepted (verified by `check_model` on every run).
#
# Cost is no longer a field in the response: the Anthropic API returns token counts only. It is
# computed here from a rate table, and BOTH the rates and the token counts go into costs.json, so
# the dollar figure is auditable rather than asserted.

ANTHROPIC_MODEL = "claude-sonnet-5"
# $ per MILLION tokens, Anthropic first-party API. Cache writes are 1.25x input at the 5-minute
# TTL and cache reads 0.1x input (`shared/prompt-caching.md` of the claude-api skill, cached
# 2026-06-24). The Batches API is 50% of every one of these, applied by `_cost`.
RATES = {
    "claude-sonnet-5": {"in": 2.00, "out": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-opus-5": {"in": 5.00, "out": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}
BATCH_DISCOUNT = 0.5
# Anthropic's own limits are 100,000 requests and 256 MB per batch. The request COUNT is not what
# binds us -- the BYTES are: a detection request carries Delphi's three few-shot turns and five
# 64-token windows, ~6 KB of JSON, so the full run's detection stage (512 features x 9 arms x 8
# five-item prompts = 36,864 requests) would be ~220 MB in one batch, inside the limit but with no
# margin. A stage is therefore cut into chunks of at most BATCH_MAX_REQUESTS, all submitted before
# any is polled, so the chunks queue in parallel and the stage costs one queue wait rather than k.
BATCH_MAX_REQUESTS = 8000
BATCH_POLL_S = 20.0
# Output tokens a call of each kind actually writes, MEASURED 2026-09-16 over 576 explainer and
# ~11,800 scorer calls. Used ONLY by `Claude.project`, which says so in its `note`.
EXPECTED_OUT = {"explain": 300, "score": 24}


def _cost(usage: dict, model: str, batch: bool) -> float:
    r = RATES.get(model)
    assert r, f"no rate table entry for {model!r}; add one rather than guessing a price"
    c = (
        usage.get("in", 0) * r["in"]
        + usage.get("out", 0) * r["out"]
        + usage.get("cache_write", 0) * r["cache_write"]
        + usage.get("cache_read", 0) * r["cache_read"]
    ) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


def _usage_of(u, batch: bool, model: str) -> dict:
    out = {
        "in": int(getattr(u, "input_tokens", 0) or 0),
        "out": int(getattr(u, "output_tokens", 0) or 0),
        "cache_write": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        "cache_read": int(getattr(u, "cache_read_input_tokens", 0) or 0),
    }
    out["cost"] = _cost(out, model, batch)
    out["batch"] = batch
    return out


class Claude:
    """Thread-safe wrapper over the Anthropic Messages API, with a batch path.

    Retries are the SDK's own (connection errors, 408, 409, 429 and every 5xx including 529, with
    exponential backoff) rather than a hand-rolled loop: `max_retries` is raised from the default 2
    to 8 because this run submits tens of thousands of requests at concurrency 32 and a 429 burst
    is expected rather than exceptional.
    """

    def __init__(self, model: str, api_key: str, timeout_s: float = 90.0, max_retries: int = 8,
                 cache_prompt: bool = True, on_commit=None):
        import anthropic

        self.model = model
        self.cache_prompt = cache_prompt
        # so a batch ledger written mid-stage survives a container that is about to be SIGTERMed
        self.on_commit = on_commit
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s,
                                           max_retries=max_retries)
        self._lock = threading.Lock()
        self.totals = {"calls": 0, "fails": 0, "in": 0, "out": 0,
                       "cache_write": 0, "cache_read": 0, "cost": 0.0}

    def params(self, system, user, max_tokens, fewshot=None) -> dict:
        """The exact request body, so the cache key is taken over what is actually sent.

        A `cache_control` breakpoint goes on the LAST few-shot turn, i.e. after the stable prefix
        (system + Delphi's verbatim shots) and before the per-call user message. Sonnet 5's minimum
        cacheable prefix is 1024 tokens; the detection prefix is close to it, so this may or may not
        create an entry -- `usage.cache_creation_input_tokens` says which, and costs.json reports
        it rather than assuming. Nothing is added to the prompt to reach the minimum: the prompts
        are Delphi's, verbatim.
        """
        msgs = []
        for i, turn in enumerate(fewshot or []):
            content = turn["content"]
            if self.cache_prompt and i == len(fewshot) - 1:
                content = [{"type": "text", "text": content,
                            "cache_control": {"type": "ephemeral"}}]
            msgs.append({"role": turn["role"], "content": content})
        msgs.append({"role": "user", "content": user})
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": msgs,
            # Sonnet 5 thinks by default; thinking would eat max_tokens and change the answer
            # format. No `temperature`: the parameter does not exist on this API (see the header).
            "thinking": {"type": "disabled"},
        }
        if self.cache_prompt and not fewshot:
            body["system"] = [{"type": "text", "text": system,
                               "cache_control": {"type": "ephemeral"}}]
        return body

    def complete(self, system, user, max_tokens, fewshot=None):
        """One synchronous call -> (text, usage). Raises on a failure the SDK could not retry."""
        try:
            r = self._client.messages.create(**self.params(system, user, max_tokens, fewshot))
        except Exception:
            self._account(None)
            raise
        usage = _usage_of(r.usage, batch=False, model=self.model)
        usage["stop_reason"] = r.stop_reason
        self._account(usage)
        return "".join(b.text for b in r.content if b.type == "text"), usage

    # -- batch path ---------------------------------------------------------------------------

    def run_batch(self, jobs: list[dict], label: str, ledger: str = ""):
        """Submit `jobs` as Message Batches and block until all of them end. -> {job key: record}.

        All chunks are submitted BEFORE any is polled, so they queue in parallel and the stage
        waits one queue latency rather than one per chunk.

        `custom_id` is a positional token (`c00r000123`), not the job key: job keys carry `|` and
        are longer than the id format allows, and results come back in ANY order, so they are keyed
        back by that token rather than by position in the results stream. The id is derived from
        the job's POSITION in `jobs`, so a re-attach must rebuild `by_id` from the same list -- it
        does, because the ledger is keyed by a hash of the job keys.

        RE-ATTACH, and the measurement that forced it. MEASURED 2026-09-16: a Modal container
        polling a batch was terminated at 1223 s with `Runner terminated (SIGTERM), exit code:
        143`, Modal re-scheduled the input, and this function submitted a SECOND batch for the same
        six requests -- the prompt cache could not help, because it is only written once a batch
        ends. At the full run's 36,864 requests that is a ~$56 double charge and two batches racing
        for the same work. So the batch ids are written to `ledger` BEFORE the first poll, and a
        restart re-attaches to them instead of submitting again.
        """
        from anthropic.types.messages.batch_create_params import Request

        t0 = time.time()
        chunks = [jobs[i : i + BATCH_MAX_REQUESTS] for i in range(0, len(jobs), BATCH_MAX_REQUESTS)]
        by_id: dict[str, dict] = {}
        for ci, chunk in enumerate(chunks):
            for i, j in enumerate(chunk):
                by_id[f"c{ci:02d}r{i:06d}"] = j
        ids: list[str] = []
        prior = None
        if ledger and os.path.exists(ledger):
            try:
                with open(ledger) as fh:
                    prior = json.load(fh)
            except json.JSONDecodeError:
                prior = None
        if prior and int(prior.get("n", -1)) == len(jobs):
            ids = list(prior["batch_ids"])
            ok = True
            for bid in ids:
                try:
                    self._client.messages.batches.retrieve(bid)
                except Exception as e:  # noqa: BLE001 -- an id we cannot retrieve is not reusable
                    print(f"[{label}] ledger names batch {bid} but it cannot be retrieved "
                          f"({type(e).__name__}); submitting fresh", flush=True)
                    ok = False
                    break
            if ok:
                print(f"[{label}] RE-ATTACHING to {len(ids)} batch(es) from {ledger} -- not "
                      f"resubmitting {len(jobs)} requests", flush=True)
            else:
                ids = []
        if not ids:
            for ci, chunk in enumerate(chunks):
                reqs = [
                    Request(custom_id=f"c{ci:02d}r{i:06d}",
                            params=self.params(j["system"], j["user"], j["max_tokens"],
                                               j.get("fewshot")))
                    for i, j in enumerate(chunk)
                ]
                b = self._client.messages.batches.create(requests=reqs)
                ids.append(b.id)
                print(f"[{label}] batch {b.id} submitted, {len(reqs)} requests "
                      f"(chunk {ci + 1}/{len(chunks)})", flush=True)
            if ledger:
                os.makedirs(os.path.dirname(ledger), exist_ok=True)
                tmp = f"{ledger}.tmp"
                with open(tmp, "w") as fh:
                    json.dump({"batch_ids": ids, "n": len(jobs), "label": label,
                               "submitted": time.time()}, fh)
                os.replace(tmp, ledger)
                if self.on_commit:
                    self.on_commit()
        pending = set(ids)
        while pending:
            time.sleep(BATCH_POLL_S)
            for bid in list(pending):
                b = self._client.messages.batches.retrieve(bid)
                if b.processing_status == "ended":
                    pending.discard(bid)
                    print(f"[{label}] batch {bid} ended | {time.time() - t0:.0f}s", flush=True)
                else:
                    print(f"[{label}] batch {bid} {b.processing_status} {b.request_counts} | "
                          f"{time.time() - t0:.0f}s", flush=True)
        out: dict[str, dict] = {}
        n_err = 0
        for bid in ids:
            for res in self._client.messages.batches.results(bid):
                j = by_id.get(res.custom_id)
                if j is None:
                    continue
                if res.result.type != "succeeded":
                    n_err += 1
                    self._account(None)
                    continue
                m = res.result.message
                usage = _usage_of(m.usage, batch=True, model=self.model)
                usage["stop_reason"] = m.stop_reason
                self._account(usage)
                out[j["key"]] = {
                    "text": "".join(b.text for b in m.content if b.type == "text"),
                    "usage": usage,
                }
        wall = time.time() - t0
        print(f"[{label}] {len(ids)} batch(es) ENDED: {len(out)} ok, {n_err} failed, {wall:.0f}s "
              f"({wall / max(1, len(jobs)) * 1000:.0f} ms/request amortised)", flush=True)
        return out, {"batch_ids": ids, "chunks": len(chunks), "wall_s": round(wall, 1),
                     "requests": len(jobs), "errors": n_err}

    # -- projection ---------------------------------------------------------------------------

    def project(self, jobs: list[dict], batch: bool, sample: int = 12) -> dict:
        """Projected cost of `jobs` BEFORE any of them is submitted.

        `messages.count_tokens` on a sample gives the input side exactly; the output side is
        estimated from `max_tokens` scaled by a measured ratio, which is the only estimated
        quantity here and is stated as such. This is what makes "report anything over $100 before
        it runs" mechanical rather than a matter of noticing.
        """
        if not jobs:
            return {"jobs": 0, "usd": 0.0}
        step = max(1, len(jobs) // sample)
        picked = jobs[::step][:sample]
        tot_in = 0
        for j in picked:
            p = self.params(j["system"], j["user"], j["max_tokens"], j.get("fewshot"))
            ct = self._client.messages.count_tokens(
                model=p["model"], system=p["system"], messages=p["messages"],
                thinking=p["thinking"],
            )
            tot_in += int(ct.input_tokens)
        mean_in = tot_in / len(picked)
        # The output side is the only estimated quantity here. It keys off the job's declared
        # KIND, not off `max_tokens`: MEASURED 2026-09-16, keying off `max_tokens <= 400` broke
        # silently the moment `explainer_max_tokens` went 300 -> 600 under A10, because that pushed
        # the explainer into the scorer's branch and assumed 30 output tokens where it writes ~290.
        # The explain stage then projected $3.38 against an actual $4.87. Per-call medians from
        # that same run: explainer 290 output tokens, scorer 20.
        mean_out = sum(EXPECTED_OUT.get(j.get("kind", "score"), 24) for j in picked) / len(picked)
        per = _cost({"in": mean_in, "out": mean_out}, self.model, batch)
        return {
            "jobs": len(jobs), "sampled": len(picked),
            "mean_input_tokens": round(mean_in, 1),
            "assumed_output_tokens": round(mean_out, 1),
            "usd_per_call": round(per, 6),
            "usd": round(per * len(jobs), 2),
            "path": "batch" if batch else "sync",
            "note": "input from messages.count_tokens on a sample; output assumed, see project()",
        }

    def _account(self, usage):
        with self._lock:
            self.totals["calls"] += 1
            if usage is None:
                self.totals["fails"] += 1
            else:
                for k in ("in", "out", "cache_write", "cache_read", "cost"):
                    self.totals[k] += usage[k]

    def snapshot(self):
        with self._lock:
            return dict(self.totals)


# ---- the cache -------------------------------------------------------------------------------


class Cache:
    """One json file per (job key, request body). The run's only durable state.

    The key hashes the job key AND the body, so an edited prompt, a changed model or a different
    temperature all miss rather than silently reusing a response from another protocol; and two
    jobs with an identical body (the `C16-rep` drift arm) still get their own call, which is the
    whole point of that arm.
    """

    def __init__(self, path: str, on_commit=None, commit_every: int = 500):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.on_commit = on_commit
        self.commit_every = commit_every
        self._n_since = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(job_key: str, body: dict) -> str:
        payload = json.dumps({"job": job_key, "body": body}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _file(self, k: str) -> str:
        """Sharded by the key's first two hex characters.

        The full 512-feature run makes ~68,000 calls and therefore ~68,000 cache files. One
        directory with that many entries is a volume-commit and directory-listing problem on a
        FUSE mount; 256 shards of ~270 files each is not.
        """
        d = os.path.join(self.path, k[:2])
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{k}.json")

    def get(self, k: str):
        p = self._file(k)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as fh:
                rec = json.load(fh)
        except json.JSONDecodeError:  # a container killed mid-write
            return None
        with self._lock:
            self.hits += 1
        return rec

    def put(self, k: str, rec: dict):
        p = self._file(k)
        tmp = f"{p}.tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, p)
        with self._lock:
            self.misses += 1
            self._n_since += 1
            due = self._n_since >= self.commit_every
            if due:
                self._n_since = 0
        if due and self.on_commit is not None:
            self.on_commit()


# ---- the driver ------------------------------------------------------------------------------


def rates(labels, preds):
    """(balanced accuracy, TPR, TNR). NaN wherever a class is absent from the scored items."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(preds, dtype=int)
    if not len(y):
        return float("nan"), float("nan"), float("nan")
    pos, neg = y == 1, y == 0
    tpr = float((p[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((p[neg] == 0).mean()) if neg.any() else float("nan")
    if not pos.any() or not neg.any():
        return float("nan"), tpr, tnr
    return float(0.5 * (tpr + tnr)), tpr, tnr


def _nr(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _submit(cl: Claude, cache: Cache, jobs: list[dict], label: str, max_cost_usd: float,
            concurrency: int, path: str, ledger: str = ""):
    """Run `jobs` through the cache and the client. Returns ({job key: record}, info).

    Cached jobs never reach the network, in either path. The sync path checks the cost ceiling
    before EVERY submission, so it is a real ceiling on what this call can spend (up to the
    requests already in flight); the batch path checks it once, before submitting, because a batch
    is all-or-nothing.
    """
    out: dict[str, dict] = {}
    todo = []
    for j in jobs:
        k = Cache.key(j["key"], cl.params(j["system"], j["user"], j["max_tokens"], j.get("fewshot")))
        rec = cache.get(k)
        if rec is not None:
            out[j["key"]] = rec
        else:
            todo.append((j, k))
    info = {"path": path, "jobs": len(jobs), "cached": len(jobs) - len(todo), "sent": len(todo)}
    if not todo:
        print(f"[{label}] {len(jobs)}/{len(jobs)} from cache, nothing sent", flush=True)
        return out, {**info, "stopped": False}

    t0 = time.time()
    if path == "batch":
        if cl.snapshot()["cost"] >= max_cost_usd:
            print(f"[{label}] COST CAP ${max_cost_usd:.2f} reached, batch not submitted", flush=True)
            return out, {**info, "stopped": True}
        got, binfo = cl.run_batch([j for j, _ in todo], label, ledger=ledger)
        for j, k in todo:
            rec = got.get(j["key"])
            if rec is not None:
                cache.put(k, rec)
                out[j["key"]] = rec
        info.update(binfo)
        s = cl.snapshot()
        print(f"[{label}] {len(out)}/{len(jobs)} | ${s['cost']:.4f} | {time.time() - t0:.0f}s",
              flush=True)
        return out, {**info, "stopped": False}

    stopped = False
    errs: list[str] = []
    pend, it = set(), iter(todo)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        while True:
            while len(pend) < concurrency and not stopped:
                if cl.snapshot()["cost"] >= max_cost_usd:
                    stopped = True
                    print(f"[{label}] COST CAP ${max_cost_usd:.2f} reached, no further requests",
                          flush=True)
                    break
                try:
                    j, k = next(it)
                except StopIteration:
                    break

                def call(job=j, key=k):
                    text, usage = cl.complete(
                        job["system"], job["user"], job["max_tokens"], job.get("fewshot")
                    )
                    return job["key"], key, {"text": text, "usage": usage}

                pend.add(ex.submit(call))
            if not pend:
                break
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for f in fin:
                try:
                    jk, k, rec = f.result()
                except Exception as e:  # noqa: BLE001 -- one dead call is not fatal
                    errs.append(f"{type(e).__name__}: {str(e)[:160]}")
                    continue
                cache.put(k, rec)
                out[jk] = rec
    s = cl.snapshot()
    wall = time.time() - t0
    print(
        f"[{label}] {len(out)}/{len(jobs)} ({info['cached']} cached) | ${s['cost']:.4f} | "
        f"{wall:.0f}s | {len(errs)} errors{' | STOPPED AT CAP' if stopped else ''}",
        flush=True,
    )
    if errs:
        print(f"[{label}] first errors: {errs[:3]}", flush=True)
    return out, {**info, "wall_s": round(wall, 1), "errors": len(errs), "stopped": stopped}


def _feature_rows(build_dir: str, feat: int):
    """(meta, {arm: row}, draw-1 test items, draw-2 test items) for one feature's build file."""
    rows = C.read_jsonl(f"{build_dir}/{feat}.jsonl")
    meta = rows[0]
    assert meta["kind"] == "meta", f"{build_dir}/{feat}.jsonl does not start with its meta row"
    arms = {r["arm"]: r for r in rows if r["kind"] == "arm"}
    t1 = [r for r in rows if r["kind"] == "test"]
    t2 = [r for r in rows if r["kind"] == "test2"]
    assert t1, f"{build_dir}/{feat}.jsonl has no `test` rows"
    return meta, arms, t1, t2


def derangement(items: list, seed: int) -> dict:
    """A fixed permutation with NO fixed point: the floor arm's "a different random feature".

    Amendment A6. A plain shuffle would leave roughly one feature in e scoring against its own
    description, which is exactly the case the floor arm exists to exclude.
    """
    assert len(items) >= 2, "a floor arm needs at least two features to permute between"
    rng = random.Random(seed)
    order = list(items)
    for _ in range(64):
        rng.shuffle(order)
        if all(a != b for a, b in zip(items, order, strict=True)):
            return dict(zip(items, order, strict=True))
    # Deterministic fallback: a rotation by one is a derangement for any length >= 2.
    return dict(zip(items, items[1:] + items[:1], strict=True))


def check_model(cl: Claude, cache: Cache) -> dict:
    """Amendment A12, as it survives the move off OpenRouter: one real call proving the request
    shape this run uses is accepted, and a recorded statement of what changed.

    It is a real request rather than a dry run because the failures it guards against -- a model id
    that does not exist, `thinking: {"type": "disabled"}` refused on this generation -- are
    server-side. It goes through the cache like everything else, so a resume pays nothing.
    """
    job = {
        "key": "check|model",
        "system": "You are a test harness. Reply with exactly the word OK.",
        "user": "Reply with exactly the word OK.",
        "max_tokens": 16,
    }
    p = cl.params(job["system"], job["user"], job["max_tokens"])
    assert p["thinking"] == {"type": "disabled"}, "thinking must be explicitly disabled"
    assert "temperature" not in p, (
        "`temperature` must NOT be in the request: the parameter does not exist on the Anthropic "
        "Messages API for this model generation (MEASURED 2026-09-16, anthropic 1.6.0 raises "
        "TypeError). See the header of this file for what that costs us."
    )
    k = Cache.key(job["key"], p)
    rec = cache.get(k)
    if rec is None:
        text, usage = cl.complete(job["system"], job["user"], job["max_tokens"])
        rec = {"text": text, "usage": usage}
        cache.put(k, rec)
    out = {
        "model": cl.model,
        "thinking": "disabled",
        "temperature": "NOT SENT -- parameter removed from the API for this model generation",
        "accepted": True,
        "reply": rec["text"].strip()[:40],
        "usage": rec["usage"],
        "rates_usd_per_mtok": RATES[cl.model],
    }
    print(f"[run] A12 model check: {cl.model} accepted thinking=disabled; reply "
          f"{out['reply']!r}; temperature is NOT a parameter of this API", flush=True)
    return out


def run(cfg, args):
    base, root, set_name = args["base"], args["root"], args["heldout"]
    ac = cfg["autointerp"]
    build_name = args.get("build_dir") or time.strftime("%Y-%m-%d") + "_build"
    build_dir = f"{C.base_dir(base, root)}/autointerp/{set_name}/{build_name}"
    assert os.path.exists(f"{build_dir}/build.json"), (
        f"no build at {build_dir}: run `--stage build` first (or pass --build-dir <name>)"
    )
    binfo = json.load(open(f"{build_dir}/build.json"))
    fmeta = {f["feature"]: f for f in json.load(open(f"{build_dir}/features.json"))["features"]}
    feats = list(fmeta)

    scorers = [s for s in (args.get("scorers") or "detection,fuzzing").split(",") if s]
    for s in scorers:
        assert s in ("detection", "fuzzing"), f"unknown scorer {s!r}"
    model = args.get("model") or ac["scorer_model"]
    path = args.get("path") or ac["path"]
    assert path in ("sync", "batch"), f"--path must be 'sync' or 'batch', got {path!r}"
    concurrency = int(args.get("concurrency") or ac["concurrency"])
    max_cost = float(args.get("max_cost_usd") or ac["max_cost_usd"])
    stop_above = float(args.get("stop_above_usd") or ac["stop_above_usd"])
    approved = bool(args.get("approved"))
    batch = path == "batch"
    key = os.environ.get("ANTHROPIC_API_KEY")
    assert key, (
        "no ANTHROPIC_API_KEY in the environment: the Modal secret `anthropic` must be mounted on "
        "this function (autointerp/modal_app.py LLM_SECRETS)"
    )
    assert len(key) > 8, "ANTHROPIC_API_KEY is present but implausibly short"
    batch_int = int(ac["scorer_batch"])
    floor_arm, floor_src, draw2_arm = str(ac["floor_arm"]), str(ac["floor_source_arm"]), "C16-draw2"
    judge_arm = str(ac.get("judge_floor_arm") or "C16-judge2")

    run_name = args.get("run_dir") or f"{time.strftime('%Y-%m-%d')}_autointerp-{base.split('-')[-1]}"
    run_root = f"{root}/runs/{run_name}"
    cache = Cache(f"{run_root}/cache", on_commit=args.get("on_commit"))
    cl = Claude(model, key, timeout_s=float(args.get("timeout_s") or ac["timeout_s"]),
                on_commit=args.get("on_commit"))

    def ledger_for(stage_label: str, jobs: list[dict]) -> str:
        """One ledger file per (stage, exact job set), so a re-attach can only match its own work."""
        h = hashlib.sha256("\n".join(sorted(j["key"] for j in jobs)).encode()).hexdigest()[:16]
        return f"{run_root}/batches/{stage_label}-{h}.json"

    arm_names = list(binfo["arms"])
    if args.get("arms"):
        arm_names = [a for a in args["arms"].split(",") if a]
    if args.get("rows"):
        want = set(C.parse_rows(args["rows"], 1 << 30))
        feats = [f for f in feats if fmeta[f]["row"] in want] or feats
    print(
        f"[run] {len(feats)} features x {len(arm_names)} explainer arms (+ {floor_arm}, "
        f"{judge_arm}, {draw2_arm}) x {scorers} | model {model} | path {path} | cap ${max_cost:.2f} | "
        f"report-above ${stop_above:.2f}{' (APPROVED)' if approved else ''} | build {build_dir}",
        flush=True,
    )
    if args.get("dry_run"):
        _meta, arms, tests, _t2 = _feature_rows(build_dir, feats[0])
        print(json.dumps({
            "explain": cl.params(DELPHI_EXPLAINER_SYSTEM, arms[arm_names[0]]["block"],
                                 int(ac["explainer_max_tokens"]), DELPHI_EXPLAINER_FEWSHOT),
            "detection": cl.params(
                DELPHI_DETECTION_SYSTEM,
                delphi_scorer_prompt("<explanation>", [t["text"] for t in tests[:batch_int]]),
                int(ac["scorer_max_tokens"]), DELPHI_DETECTION_FEWSHOT),
        })[:6000], flush=True)
        return {"dry_run": True, "features": len(feats), "arms": arm_names, "path": path}

    model_check = check_model(cl, cache)
    projections: dict[str, dict] = {}
    stage_info: dict[str, dict] = {}
    stopped_at = None

    def gate(jobs, stage):
        """Project a stage BEFORE submitting it; refuse to spend over `stop_above` unapproved."""
        nonlocal stopped_at
        todo = [
            j for j in jobs
            if cache.get(Cache.key(j["key"], cl.params(j["system"], j["user"], j["max_tokens"],
                                                       j.get("fewshot")))) is None
        ]
        pr = cl.project(todo, batch)
        pr["stage"] = stage
        pr["arms"] = sorted({j.get("arm", "-") for j in jobs})
        pr["usd_per_arm"] = round(pr["usd"] / max(1, len(pr["arms"])), 2)
        projections[stage] = pr
        print(f"[run] PROJECTION {stage}: {pr['jobs']} uncached calls -> ${pr['usd']:.2f} "
              f"(${pr['usd_per_arm']:.2f}/arm, {pr['mean_input_tokens']:.0f} input tok/call, "
              f"path {pr['path']})", flush=True)
        if pr["usd"] > stop_above and not approved:
            stopped_at = (
                f"stage {stage} projects ${pr['usd']:.2f}, above the ${stop_above:.2f} "
                f"report-first threshold; re-run with --approved once it has been reported"
            )
            print(f"[run] STOPPING: {stopped_at}", flush=True)
            return False
        if pr["usd"] > max_cost:
            stopped_at = f"stage {stage} projects ${pr['usd']:.2f}, above the ${max_cost:.2f} cap"
            print(f"[run] STOPPING: {stopped_at}", flush=True)
            return False
        return True

    # ---- PHASE 1: explain every feature x every explainer arm -------------------------------
    # All of it before any scoring, because the floor arm needs ANOTHER feature's description and
    # a feature-at-a-time loop cannot supply one that has not been written yet.
    expl_rows: list[dict] = []
    expl: dict[tuple[int, str], str] = {}
    n_trunc = 0
    jobs = []
    for feat in feats:
        _meta, arms, _t1, _t2 = _feature_rows(build_dir, feat)
        for a in arm_names:
            if a not in arms:
                continue
            jobs.append({
                "key": f"explain|{feat}|{a}", "arm": a, "feat": feat, "n": arms[a]["n"],
                "kind": "explain",
                "system": DELPHI_EXPLAINER_SYSTEM, "fewshot": DELPHI_EXPLAINER_FEWSHOT,
                "user": arms[a]["block"], "max_tokens": int(ac["explainer_max_tokens"]),
            })
    got: dict[str, dict] = {}
    if gate(jobs, "explain"):
        got, info = _submit(cl, cache, jobs, "explain", max_cost, concurrency, path,
                            ledger=ledger_for("explain", jobs))
        stage_info["explain"] = info
        # A10: an explainer answer cut off at max_tokens is an ERROR, not a shorter explanation.
        # Retry ONCE at double the budget under its own job key, then raise.
        retry = [
            {**j, "key": j["key"] + "|retry", "max_tokens": 2 * j["max_tokens"]}
            for j in jobs
            if (got.get(j["key"]) or {}).get("usage", {}).get("stop_reason") == "max_tokens"
        ]
        n_trunc = len(retry)
        if retry:
            print(f"[run] A10: {n_trunc} explainer answers hit max_tokens; retrying at "
                  f"{2 * int(ac['explainer_max_tokens'])}", flush=True)
            got2, _ = _submit(cl, cache, retry, "explain-retry", max_cost, concurrency, path,
                              ledger=ledger_for("explain-retry", retry))
            still = []
            for j in retry:
                rec = got2.get(j["key"])
                if rec is None or rec.get("usage", {}).get("stop_reason") == "max_tokens":
                    still.append(j["key"])
                else:
                    got[j["key"].removesuffix("|retry")] = rec
            assert not still, (
                f"A10: {len(still)} explainer answers were STILL truncated at "
                f"{2 * int(ac['explainer_max_tokens'])} tokens ({still[:3]}). A truncated "
                f"explanation is not a shorter explanation -- raise "
                f"autointerp.explainer_max_tokens and re-run."
            )
        if info.get("stopped"):
            stopped_at = stopped_at or "cost cap during explain"
    for j in jobs:
        rec = got.get(j["key"])
        text = rec["text"] if rec else ""
        e = parse_delphi_explanation(text)
        expl[(j["feat"], j["arm"])] = e
        expl_rows.append({
            "feature": j["feat"], "arm": j["arm"], "n_examples": j["n"], "explanation": e,
            "ok": bool(e), "raw_len": len(text), "usage": (rec or {}).get("usage", {}),
        })
    explain_cost = cl.snapshot()["cost"]
    n_empty = sum(1 for r in expl_rows if not r["ok"])
    print(f"[run] explained {len(expl_rows)} (feature, arm) pairs, {n_empty} empty, "
          f"{n_trunc} truncated-and-retried, ${explain_cost:.4f}", flush=True)

    perm = derangement(feats, int(ac["shuffle_seed"]))

    # ---- PHASE 2: score. ONE job list per scorer over EVERY feature, so the batch path has a
    # batch worth submitting and the sync path keeps the thread pool saturated.
    batch_rows: dict[str, list[dict]] = {s: [] for s in scorers}
    scores: list[dict] = []
    plans: dict[int, list[tuple[str, str, list]]] = {}
    for feat in feats:
        _meta, arms, t1, t2 = _feature_rows(build_dir, feat)
        plan = [(a, expl.get((feat, a), ""), t1) for a in arm_names if a in arms]
        plan.append((floor_arm, expl.get((perm[feat], floor_src), ""), t1))
        if expl.get((feat, floor_src)):
            # The JUDGE-ONLY floor: the same description on the SAME draw-1 items. Its job key
            # differs from C16's, so the cache treats it as a separate call and it really is a
            # second judgement -- which is a measurement now that no temperature parameter exists
            # and nothing is deterministic. `C16-draw2` then measures judge AND draw variation
            # together, and the difference between the two floors is the draw half.
            plan.append((judge_arm, expl.get((feat, floor_src), ""), t1))
        if t2 and expl.get((feat, floor_src)):
            plan.append((draw2_arm, expl.get((feat, floor_src), ""), t2))
        plans[feat] = plan

    def groups_of(items):
        return [list(range(i, min(i + batch_int, len(items)))) for i in range(0, len(items), batch_int)]

    for scorer in scorers:
        if stopped_at:
            break
        field = "text" if scorer == "detection" else "text_fuzz"
        system = DELPHI_DETECTION_SYSTEM if scorer == "detection" else DELPHI_FUZZ_SYSTEM
        fewshot = DELPHI_DETECTION_FEWSHOT if scorer == "detection" else None
        jobs = []
        for feat in feats:
            for a, e, items in plans[feat]:
                if not e:
                    continue
                for bi, g in enumerate(groups_of(items)):
                    jobs.append({
                        "key": f"{scorer}|{feat}|{a}|{bi}", "arm": a, "feat": feat,
                        "kind": "score",
                        "system": system, "fewshot": fewshot,
                        "user": delphi_scorer_prompt(e, [items[i][field] for i in g]),
                        "max_tokens": int(ac["scorer_max_tokens"]),
                    })
        if not gate(jobs, scorer):
            break
        got, info = _submit(cl, cache, jobs, scorer, max_cost, concurrency, path,
                            ledger=ledger_for(scorer, jobs))
        stage_info[scorer] = info
        if info.get("stopped"):
            stopped_at = stopped_at or f"cost cap during {scorer}"
        for feat in feats:
            gate_v = float(_feature_rows(build_dir, feat)[0]["gate"])
            arms = _feature_rows(build_dir, feat)[1]
            for a, e, items in plans[feat]:
                gs = groups_of(items)
                labels, preds, srcs = [], [], []
                n_batches = n_parsed = 0
                for bi, g in enumerate(gs):
                    rec = got.get(f"{scorer}|{feat}|{a}|{bi}")
                    if rec is None:
                        continue
                    n_batches += 1
                    vals = parse_delphi_scores(rec["text"], len(g))
                    batch_rows[scorer].append({
                        "feature": feat, "arm": a, "batch": bi,
                        "items": [items[i]["i"] for i in g],
                        "labels": [items[i]["label"] for i in g],
                        "preds": vals, "parsed": vals is not None,
                        "usage": rec.get("usage", {}),
                    })
                    if vals is None:
                        continue  # unparsed batches are DROPPED, never imputed
                    n_parsed += 1
                    labels += [items[i]["label"] for i in g]
                    preds += vals
                    srcs += [items[i]["src"] for i in g]
                if not n_batches:
                    continue
                acc, tpr, tnr = rates(labels, preds)
                # Amendment A5: the negative side is half zero-activation randoms and half
                # near-miss windows, and they are NOT the same test. Reported separately, from the
                # same answers, so a result that lives entirely on one half cannot hide.
                zi = [i for i, sr in enumerate(srcs) if not str(sr).startswith("nearmiss")]
                ni = [i for i, sr in enumerate(srcs)
                      if str(sr).startswith("nearmiss") or labels[i] == 1]
                z_acc, _zt, z_tnr = rates([labels[i] for i in zi], [preds[i] for i in zi])
                n_acc, _nt, n_tnr = rates([labels[i] for i in ni], [preds[i] for i in ni])
                scores.append({
                    "feature": feat, "arm": a, "scorer": scorer,
                    "bal_acc": _nr(acc), "tpr": _nr(tpr), "tnr": _nr(tnr),
                    "bal_acc_zero_neg": _nr(z_acc), "tnr_zero": _nr(z_tnr),
                    "bal_acc_nearmiss_neg": _nr(n_acc), "tnr_nearmiss": _nr(n_tnr),
                    "acc": _nr(float(np.mean(np.asarray(labels) == np.asarray(preds))))
                    if labels else None,
                    "n_items": len(labels), "n_batches": n_batches, "n_parsed": n_parsed,
                    "n_pos": int(sum(labels)),
                    "n_neg_nearmiss": int(sum(1 for sr in srcs if str(sr).startswith("nearmiss"))),
                    "draw": 2 if a == draw2_arm else 1,
                    "role": ("floor" if a == floor_arm else "judge_null" if a == judge_arm
                             else "draw_null" if a == draw2_arm else "arm"),
                    "n_examples": arms[a]["n"] if a in arms else 0,
                    "explanation_ok": bool(e),
                    "explanation_of": perm[feat] if a == floor_arm else feat,
                    "path": path, "gate": gate_v,
                    "stratum": fmeta[feat]["stratum"],
                    "fire_fraction": fmeta[feat]["fire_fraction"],
                    "corpus_peak": fmeta[feat]["corpus_peak"],
                    "density": fmeta[feat]["density"],
                })

    # ---- products ---------------------------------------------------------------------------
    s = cl.snapshot()
    cum = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0, "cost": 0.0, "calls": 0}
    by_arm: dict[str, dict] = {}
    by_stage: dict[str, dict] = {}
    for rows, stage in [(expl_rows, "explain")] + [(batch_rows[x], x) for x in scorers]:
        for r in rows:
            u = r.get("usage") or {}
            for d in (by_arm.setdefault(r["arm"], {"in": 0, "out": 0, "cache_write": 0,
                                                   "cache_read": 0, "cost": 0.0, "calls": 0,
                                                   "paths": set()}),
                      by_stage.setdefault(stage, {"in": 0, "out": 0, "cache_write": 0,
                                                  "cache_read": 0, "cost": 0.0, "calls": 0,
                                                  "paths": set()}),
                      cum):
                for k2 in ("in", "out", "cache_write", "cache_read", "cost"):
                    d[k2] += u.get(k2, 0)
                d["calls"] += 1
                if isinstance(d.get("paths"), set):
                    d["paths"].add("batch" if u.get("batch") else "sync")

    def rnd(d):
        return {k: (sorted(v) if isinstance(v, set) else round(v, 6) if isinstance(v, float) else v)
                for k, v in d.items()}

    n_scored = len({r["feature"] for r in scores})
    costs = {
        "api": "anthropic-messages",
        "model": model,
        "path": path,
        "temperature": "NOT SENT -- removed from the Anthropic Messages API for this model "
                       "generation (MEASURED 2026-09-16, anthropic 1.6.0 raises TypeError)",
        "rates_usd_per_mtok": RATES[model],
        "batch_discount": BATCH_DISCOUNT,
        "model_check": model_check,
        "this_call": s,
        "cumulative_over_cache": rnd(cum),
        "per_arm": {a: rnd(d) for a, d in sorted(by_arm.items())},
        "per_stage": {a: rnd(d) for a, d in sorted(by_stage.items())},
        "stage_info": stage_info,
        "projections": projections,
        "explainer_truncated_and_retried": n_trunc,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "features_done": n_scored,
        "features_total": len(feats),
        "max_cost_usd": max_cost,
        "stop_above_usd": stop_above,
        "approved": approved,
        "projection_usd": round(sum(p["usd"] for p in projections.values()), 2),
        "stopped_at": stopped_at,
    }
    print(f"[run] costs {json.dumps(costs['this_call'])} | cumulative ${cum['cost']:.4f} over "
          f"{cum['calls']} calls", flush=True)

    common_inputs = {
        "build": build_dir, "features": f"{n_scored} of {len(feats)}",
        "arms": ",".join([*arm_names, floor_arm, judge_arm, draw2_arm]),
        "api": "anthropic-messages", "model": model, "path": path,
        "delphi_commit": DELPHI_COMMIT,
    }
    # These directories are a PURE FUNCTION of cache/, so a resumed run rewrites them and
    # force=True is not destroying a measurement -- the measurement is the cache.
    def out(name, status="ok"):
        return C.outdir(f"{run_root}/{name}", {**args, "force": True},
                        inputs=common_inputs, status=status)

    status = "stopped_early" if stopped_at else "ok"
    with out("explain", status) as od:
        od.write_jsonl("explanations.jsonl", expl_rows)
        od.write_json("costs.json", costs)
        od.note(
            f"Delphi explainer, verbatim system prompt + the one few-shot user/assistant pair, "
            f"max_tokens {int(ac['explainer_max_tokens'])}, thinking disabled, NO temperature "
            f"(the parameter no longer exists on this API -- see costs.json). The answer is the "
            f"text after the LAST `[EXPLANATION]:`. {n_empty} of {len(expl_rows)} came back empty. "
            f"AMENDMENT A10: {n_trunc} answers hit max_tokens and were retried ONCE at double the "
            f"budget; a still-truncated answer raises rather than being kept as a short one."
        )
        od.note(
            f"three arms have NO explainer call of their own: `{floor_arm}` (A6, a DIFFERENT "
            f"feature's `{floor_src}` description under a fixed derangement -- the interpretability "
            f"floor), `{judge_arm}` (the same description on the same draw-1 items, scored again -- "
            f"the JUDGE-ONLY noise floor), and `{draw2_arm}` (A7, the same description on the "
            f"second disjoint draw -- judge AND draw variation together). The difference between "
            f"the last two is the test-set-draw half of the noise."
        )
    for scorer in scorers:
        rows = batch_rows[scorer]
        n_bad = sum(1 for r in rows if not r["parsed"])
        with out(scorer, status) as od:
            od.write_jsonl("batches.jsonl", rows)
            od.write_jsonl("per_feature.jsonl", [r for r in scores if r["scorer"] == scorer])
            od.write_json("costs.json", costs)
            od.note(
                f"Delphi {scorer} scorer, {batch_int} items per prompt, binary answers, "
                f"max_tokens {int(ac['scorer_max_tokens'])}, via the Anthropic Messages API on the "
                f"`{path}` path. The parser takes the LAST bracketed group of exactly the batch "
                f"length; a wrong-length or unreadable answer DROPS the batch rather than padding "
                f"it. {n_bad} of {len(rows)} batches ({n_bad / max(1, len(rows)):.2%}) were "
                f"unparsed and dropped."
            )
            od.note(
                "detection sees PLAIN text (Delphi's highlighted=False) with the three verbatim "
                "few-shot turns; fuzzing sees the << >>-marked text ZERO-SHOT, because Delphi's "
                "fuzzing few-shots are not in our transcription. The two numbers are therefore "
                "comparable ACROSS ARMS but not to each other."
            )
            od.note(
                "metric per (feature, arm): balanced accuracy = mean(TPR, TNR) over the items of "
                "PARSED batches only. `bal_acc_zero_neg` and `bal_acc_nearmiss_neg` are the same "
                "answers with the negative side restricted to each half of the A5 mix."
            )
    with out("summary", status) as od:
        od.write_jsonl("scores.jsonl", scores)
        od.write_json("costs.json", costs)
        od.write_json("features.json", {"features": [fmeta[f] for f in feats]})
        od.write_json("build.json", binfo)
        od.write_json("floor_permutation.json",
                      {"floor_arm": floor_arm, "source_arm": floor_src,
                       "seed": int(ac["shuffle_seed"]),
                       "map": {str(k): v for k, v in perm.items()}})
        od.note(
            f"scores.jsonl is one row per (feature, arm, scorer): bal_acc, its two negative-half "
            f"restrictions, tpr/tnr, item counts, the arm's ACTUAL example count, which test draw "
            f"it used, whose description it used, which API path served it, and the feature "
            f"covariates. {len(scores)} rows over {n_scored} features."
        )
        od.note(
            f"`{floor_arm}` is the floor: each feature's test set scored with ANOTHER feature's "
            f"`{floor_src}` description under a fixed derangement (no fixed point). "
            f"`{draw2_arm}` is the null: C16's own description on the second, disjoint test draw."
        )
        od.note(
            f"API: Anthropic Messages, model {model}, path `{path}`"
            + (f" (batch = {BATCH_DISCOUNT:.0%} of list price)" if batch else "")
            + f". Cost is COMPUTED from token counts at {RATES[model]} $/MTok -- the Anthropic API "
            f"returns no cost field -- so both the rates and the counts are in costs.json and the "
            f"dollar figure is auditable. This call ${s['cost']:.4f} over {s['calls']} calls "
            f"({s['fails']} failures); cumulative over the cache ${cum['cost']:.4f} over "
            f"{cum['calls']} calls."
        )
        od.note(
            "temperature is NOT SENT: `anthropic` 1.6.0's messages.create() has no such parameter "
            "for this model generation (MEASURED 2026-09-16). The design's 'temperature 0' is not "
            "achievable on this surface, so run-to-run variation is real and the A7 null arm is "
            "the only noise floor this evaluation has."
        )
        if stopped_at:
            od.note(f"STOPPED EARLY: {stopped_at}")

    return {
        "out": run_root, "path": path,
        "features_done": n_scored, "features_total": len(feats),
        "arms": [*arm_names, floor_arm, judge_arm, draw2_arm], "scorers": scorers,
        "explainer_truncated": n_trunc, "explanations_empty": n_empty,
        "cost_this_call": round(s["cost"], 4), "cost_cumulative": round(cum["cost"], 4),
        "calls_this_call": s["calls"], "cache_hits": cache.hits,
        "projections": {k: v["usd"] for k, v in projections.items()},
        "batch_walls": {k: v.get("wall_s") for k, v in stage_info.items()},
        "stopped_at": stopped_at,
    }
