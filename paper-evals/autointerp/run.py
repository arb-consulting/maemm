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
file URLs and the fetch date. The OpenRouter client is the same file's `_OpenRouter`
(:1825-1911). Both are copied rather than imported: that file lives in another worktree, is
read-only, and carries `mxf`/torch imports this CPU container must not need.

The API key arrives as the Modal secret `openrouter` (env `OPENROUTER_API_KEY`) and is never
printed, never written to the volume and never put in a README. Only call counts, token counts and
dollars are recorded -- and the dollars come from each response's own `usage.cost`, never from the
key's usage delta, because the key is shared
(experiments/2026-09-11_autointerp-64feat-plan.md:153).

Two phases, in this order for a reason: EVERY feature is explained first, then features are scored
one at a time. The floor arm (A6) scores a feature's test set with a DIFFERENT feature's
description, so it cannot run inside a feature-at-a-time loop that has not written that
description yet.

Arms `run` adds to the build's, neither of which has an explainer call of its own:
  * `R-shuffled` (A6) -- the floor: another feature's C16 description, under a fixed derangement.
  * `C16-draw2` (A7) -- the null: C16's own description on the second, disjoint test draw. The
    per-feature C16 - C16-draw2 difference is the test-set sampling noise every contrast is
    exposed to, and it replaces the temperature-0 repeat, which measured judge jitter instead.

Cost control, two layers. `autointerp.max_cost_usd` is a HARD ceiling checked before every
submission (the lifted `--max-cost-usd` behaviour). On top of it, the first
`autointerp.probe_features` features are SCORED and the whole run's cost is PROJECTED from them
plus the already-paid explainer bill; if the projection exceeds the ceiling the run stops there
and says so, rather than discovering the overrun at feature 60.
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


# ---- OpenRouter client (repo-maemm/eval/autointerp_detection.py:1800-1911) --------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 409, 429, 529}


class LLMError(Exception):
    def __init__(self, msg, retryable):
        super().__init__(msg)
        self.retryable = retryable


class OpenRouter:
    """Thread-safe. complete(...) -> (text, usage); usage.cost is the provider's own number.

    Retries 408/409/429/529, every 5xx and network errors, with exponential backoff capped at 60 s
    plus jitter. Provider errors that arrive as HTTP 200 with an error body are caught too.

    DIVERGENCE from the lifted client, deliberate: `temperature` IS sent (the design fixes it at
    0). The lifted version never sent it because its default judge was Opus 5 through the Anthropic
    Batches API, which 400s on temperature with thinking enabled; here thinking is disabled
    explicitly and the model is Sonnet 5 through OpenRouter.
    """

    def __init__(self, model, api_key, timeout_s=90.0, max_attempts=5, temperature=0.0):
        self.model, self.timeout_s, self.max_attempts = model, timeout_s, max_attempts
        self.temperature = temperature
        self._h = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "maemm-paper-evals-autointerp",
        }
        self._lock = threading.Lock()
        self.totals = {"calls": 0, "fails": 0, "in": 0, "out": 0, "cost": 0.0}

    def _post(self, body):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(body).encode(), headers=self._h, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except Exception:  # noqa: BLE001 -- a non-JSON error body is still a status
                return e.code, {}

    def body_for(self, system, user, max_tokens, fewshot=None):
        """The exact request body, so the cache key can be taken over what is actually sent."""
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            # Sonnet 5 / Opus 5 think by default; thinking would silently eat max_tokens.
            "reasoning": {"enabled": False},
            "messages": (
                [{"role": "system", "content": system}]
                + list(fewshot or [])
                + [{"role": "user", "content": user}]
            ),
            "usage": {"include": True},
        }

    def _once(self, system, user, max_tokens, fewshot=None):
        status, data = self._post(self.body_for(system, user, max_tokens, fewshot))
        if status != 200:
            raise LLMError(f"HTTP {status}: {str(data)[:200]}", status in RETRY_STATUS or status >= 500)
        err = data.get("error") if isinstance(data, dict) else None
        if err:  # provider errors arrive as 200 + error body
            code = int(err.get("code") or 500)
            raise LLMError(f"provider {code}: {str(err)[:200]}", code in RETRY_STATUS or code >= 500)
        try:
            ch = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"malformed response: {str(data)[:200]}", True) from None
        if ch.get("finish_reason") == "content_filter":
            raise LLMError("refusal/content_filter", False)
        u = data.get("usage") or {}
        return (ch["message"].get("content") or ""), {
            "in": int(u.get("prompt_tokens") or 0),
            "out": int(u.get("completion_tokens") or 0),
            "cost": float(u.get("cost") or 0.0),
            "finish_reason": ch.get("finish_reason"),
        }

    def complete(self, system, user, max_tokens, fewshot=None):
        delay = 1.0
        for attempt in range(self.max_attempts):
            try:
                text, usage = self._once(system, user, max_tokens, fewshot)
            except LLMError as e:
                if not e.retryable or attempt == self.max_attempts - 1:
                    self._account(None)
                    raise
            except Exception as e:  # noqa: BLE001 -- network/timeout: retry
                if attempt == self.max_attempts - 1:
                    self._account(None)
                    raise LLMError(f"{type(e).__name__}: {e}", True) from None
            else:
                self._account(usage)
                return text, usage
            time.sleep(min(60.0, delay) + random.uniform(0.0, 0.5))
            delay *= 2
        raise LLMError("unreachable", False)

    def _account(self, usage):
        with self._lock:
            self.totals["calls"] += 1
            if usage is None:
                self.totals["fails"] += 1
            else:
                for k in ("in", "out", "cost"):
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


def _submit(cl: OpenRouter, cache: Cache, jobs: list[dict], label: str, max_cost_usd: float,
            concurrency: int):
    """Run `jobs` through the cache and the client. Returns ({job key: record}, stopped_early).

    Cached jobs never reach the network. The cost cap is checked before EVERY submission, so it is
    a real ceiling on what this call can spend (up to the requests already in flight).
    """
    out: dict[str, dict] = {}
    todo = []
    for j in jobs:
        body = cl.body_for(j["system"], j["user"], j["max_tokens"], j.get("fewshot"))
        k = Cache.key(j["key"], body)
        rec = cache.get(k)
        if rec is not None:
            out[j["key"]] = rec
        else:
            todo.append((j, k))
    if not todo:
        return out, False
    stopped = False
    errs: list[str] = []
    t0 = time.time()
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
                    errs.append(str(e)[:200])
                    continue
                cache.put(k, rec)
                out[jk] = rec
    s = cl.snapshot()
    print(
        f"[{label}] {len(out)}/{len(jobs)} ({len(jobs) - len(todo)} cached) | ${s['cost']:.4f} | "
        f"{time.time() - t0:.0f}s | {len(errs)} errors{' | STOPPED AT CAP' if stopped else ''}",
        flush=True,
    )
    if errs:
        print(f"[{label}] first errors: {errs[:3]}", flush=True)
    return out, stopped


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


def check_temperature(cl: OpenRouter, cache: Cache) -> dict:
    """Amendment A12: one call confirming OpenRouter accepts `temperature` with reasoning disabled.

    Sent through the cache like every other call, so it costs nothing on a resume. It is a real
    request rather than a dry run because the failure it guards against -- a 400 on `temperature`,
    which is what the Anthropic Batches path does for Opus 5 with thinking on -- is a server-side
    rejection that only a real request can surface.
    """
    job = {
        "key": "check|temperature",
        "system": "You are a test harness. Reply with exactly the word OK.",
        "user": "Reply with exactly the word OK.",
        "max_tokens": 16,
    }
    body = cl.body_for(job["system"], job["user"], job["max_tokens"])
    assert body["temperature"] == cl.temperature and body["reasoning"] == {"enabled": False}, (
        "the request body does not carry temperature with reasoning disabled"
    )
    k = Cache.key(job["key"], body)
    rec = cache.get(k)
    if rec is None:
        text, usage = cl.complete(job["system"], job["user"], job["max_tokens"])
        rec = {"text": text, "usage": usage}
        cache.put(k, rec)
    out = {
        "model": cl.model,
        "temperature": cl.temperature,
        "reasoning": "disabled",
        "accepted": True,
        "reply": rec["text"].strip()[:40],
        "usage": rec["usage"],
    }
    print(f"[run] A12 temperature check: {cl.model} accepted temperature="
          f"{cl.temperature} with reasoning disabled; reply {out['reply']!r}", flush=True)
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
    concurrency = int(args.get("concurrency") or ac["concurrency"])
    max_cost = float(args.get("max_cost_usd") or ac["max_cost_usd"])
    probe_n = int(args.get("probe_features") or ac["probe_features"])
    batch = int(ac["scorer_batch"])
    floor_arm = str(ac["floor_arm"])
    floor_src = str(ac["floor_source_arm"])
    draw2_arm = "C16-draw2"
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, (
        "no OPENROUTER_API_KEY in the environment: the Modal secret `openrouter` must be mounted "
        "on this function (autointerp/modal_app.py LLM_SECRETS)"
    )
    assert len(key) > 8, "OPENROUTER_API_KEY is present but implausibly short"

    run_name = args.get("run_dir") or f"{time.strftime('%Y-%m-%d')}_autointerp-{base.split('-')[-1]}"
    run_root = f"{root}/runs/{run_name}"
    cache = Cache(f"{run_root}/cache", on_commit=args.get("on_commit"))
    cl = OpenRouter(
        model,
        key,
        timeout_s=float(args.get("timeout_s") or ac["timeout_s"]),
        temperature=float(ac["temperature"]),
    )

    arm_names = list(binfo["arms"])
    if args.get("arms"):
        arm_names = [a for a in args["arms"].split(",") if a]
    print(
        f"[run] {len(feats)} features x {len(arm_names)} explainer arms (+ {floor_arm}, "
        f"{draw2_arm}) x {scorers} | model {model} | cap ${max_cost:.2f} | build {build_dir}",
        flush=True,
    )
    if args.get("dry_run"):
        _meta, arms, tests, _t2 = _feature_rows(build_dir, feats[0])
        print(json.dumps({
            "explain": cl.body_for(DELPHI_EXPLAINER_SYSTEM, arms[arm_names[0]]["block"],
                                   int(ac["explainer_max_tokens"]), DELPHI_EXPLAINER_FEWSHOT),
            "detection": cl.body_for(
                DELPHI_DETECTION_SYSTEM,
                delphi_scorer_prompt("<explanation>", [t["text"] for t in tests[:batch]]),
                int(ac["scorer_max_tokens"]), DELPHI_DETECTION_FEWSHOT),
        })[:4000], flush=True)
        return {"dry_run": True, "features": len(feats), "arms": arm_names}

    temp_check = check_temperature(cl, cache)

    # ---- PHASE 1: explain every feature x every explainer arm -------------------------------
    # All of it before any scoring, because the floor arm needs ANOTHER feature's description and
    # feature-by-feature ordering cannot supply one that has not been written yet.
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
                "key": f"explain|{feat}|{a}",
                "system": DELPHI_EXPLAINER_SYSTEM,
                "fewshot": DELPHI_EXPLAINER_FEWSHOT,
                "user": arms[a]["block"],
                "max_tokens": int(ac["explainer_max_tokens"]),
                "feat": feat,
                "arm": a,
                "n": arms[a]["n"],
            })
    got, stopped = _submit(cl, cache, jobs, "explain", max_cost, concurrency)
    # A10: an explainer answer cut off at max_tokens is an ERROR, not a shorter explanation. Retry
    # ONCE at double the budget under its own job key, then raise.
    retry = []
    for j in jobs:
        rec = got.get(j["key"])
        if rec and (rec.get("usage") or {}).get("finish_reason") == "length":
            n_trunc += 1
            retry.append({**j, "key": j["key"] + "|retry", "max_tokens": 2 * j["max_tokens"]})
    if retry:
        print(f"[run] A10: {len(retry)} explainer answers hit max_tokens; retrying at "
              f"{2 * int(ac['explainer_max_tokens'])}", flush=True)
        got2, _ = _submit(cl, cache, retry, "explain-retry", max_cost, concurrency)
        still = []
        for j in retry:
            rec = got2.get(j["key"])
            if rec is None or (rec.get("usage") or {}).get("finish_reason") == "length":
                still.append(j["key"])
            else:
                got[j["key"].removesuffix("|retry")] = rec
        assert not still, (
            f"A10: {len(still)} explainer answers were STILL truncated at "
            f"{2 * int(ac['explainer_max_tokens'])} tokens ({still[:3]}). A truncated explanation "
            f"is not a shorter explanation -- raise autointerp.explainer_max_tokens and re-run."
        )
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

    # ---- PHASE 2: score, feature by feature so the projection gate can stop the run ----------
    batch_rows: dict[str, list[dict]] = {s: [] for s in scorers}
    scores: list[dict] = []
    stopped_at = "cost cap during explain" if stopped else None
    projection = None
    done = 0
    for feat in feats:
        if stopped_at:
            break
        _meta, arms, t1, t2 = _feature_rows(build_dir, feat)
        gate = float(_meta["gate"])
        # (arm label, explanation, items). The two scorer-only pseudo-arms are here and nowhere
        # else: neither has an explainer call of its own.
        plan = [(a, expl.get((feat, a), ""), t1) for a in arm_names if a in arms]
        plan.append((floor_arm, expl.get((perm[feat], floor_src), ""), t1))
        if t2 and (feat, "C16") in expl:
            plan.append((draw2_arm, expl.get((feat, "C16"), ""), t2))
        for scorer in scorers:
            field = "text" if scorer == "detection" else "text_fuzz"
            system = DELPHI_DETECTION_SYSTEM if scorer == "detection" else DELPHI_FUZZ_SYSTEM
            fewshot = DELPHI_DETECTION_FEWSHOT if scorer == "detection" else None
            jobs = []
            for a, e, items in plan:
                if not e:
                    continue
                groups = [list(range(i, min(i + batch, len(items))))
                          for i in range(0, len(items), batch)]
                for bi, g in enumerate(groups):
                    jobs.append({
                        "key": f"{scorer}|{feat}|{a}|{bi}",
                        "system": system,
                        "fewshot": fewshot,
                        "user": delphi_scorer_prompt(e, [items[i][field] for i in g]),
                        "max_tokens": int(ac["scorer_max_tokens"]),
                    })
            got, stop = _submit(cl, cache, jobs, f"{scorer} f{feat}", max_cost, concurrency)
            for a, e, items in plan:
                groups = [list(range(i, min(i + batch, len(items))))
                          for i in range(0, len(items), batch)]
                labels, preds, srcs = [], [], []
                n_batches = n_parsed = 0
                for bi, g in enumerate(groups):
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
                acc, tpr, tnr = rates(labels, preds)
                # Amendment A5: the negative side is half zero-activation randoms and half
                # near-miss windows, and they are NOT the same test. Reported separately, from the
                # same answers, so a result that lives entirely on one half cannot hide.
                zi = [i for i, sr in enumerate(srcs) if not str(sr).startswith("nearmiss")]
                ni = [i for i, sr in enumerate(srcs) if str(sr).startswith("nearmiss")
                      or labels[i] == 1]
                z_acc, _z_tpr, z_tnr = rates([labels[i] for i in zi], [preds[i] for i in zi])
                n_acc, _n_tpr, n_tnr = rates([labels[i] for i in ni], [preds[i] for i in ni])
                row = {
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
                    "n_examples": arms[a]["n"] if a in arms else 0,
                    "explanation_ok": bool(e),
                    "explanation_of": perm[feat] if a == floor_arm else feat,
                    "gate": gate,
                    "stratum": fmeta[feat]["stratum"],
                    "fire_fraction": fmeta[feat]["fire_fraction"],
                    "corpus_peak": fmeta[feat]["corpus_peak"],
                    "density": fmeta[feat]["density"],
                }
                scores.append(row)
            if stop:
                stopped_at = f"cost cap during {scorer} on feature {feat}"
                break
        done += 1
        if stopped_at:
            break
        if done == probe_n and len(feats) > probe_n:
            score_cost = cl.snapshot()["cost"] - explain_cost
            projection = explain_cost + score_cost / done * len(feats)
            print(
                f"[run] PROJECTION from {done} scored features: explain ${explain_cost:.4f} + "
                f"scoring ${score_cost:.4f} so far -> ${projection:.2f} for {len(feats)} "
                f"(cap ${max_cost:.2f})",
                flush=True,
            )
            if projection > max_cost:
                stopped_at = (
                    f"projected ${projection:.2f} from the first {done} scored features exceeds "
                    f"the ${max_cost:.2f} cap"
                )
                break

    # ---- products ---------------------------------------------------------------------------
    s = cl.snapshot()
    cum = {"in": 0, "out": 0, "cost": 0.0, "calls": 0}
    by_arm: dict[str, dict] = {}
    by_stage: dict[str, dict] = {}
    for rows, stage in [(expl_rows, "explain")] + [(batch_rows[x], x) for x in scorers]:
        for r in rows:
            u = r.get("usage") or {}
            for d in (by_arm.setdefault(r["arm"], {"in": 0, "out": 0, "cost": 0.0, "calls": 0}),
                      by_stage.setdefault(stage, {"in": 0, "out": 0, "cost": 0.0, "calls": 0}),
                      cum):
                for k2 in ("in", "out", "cost"):
                    d[k2] += u.get(k2, 0)
                d["calls"] += 1
    rnd = lambda d: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    costs = {
        "model": model,
        "temperature": float(ac["temperature"]),
        "temperature_check": temp_check,
        "this_call": s,
        "cumulative_over_cache": rnd(cum),
        "per_arm": {a: rnd(d) for a, d in sorted(by_arm.items())},
        "per_stage": {a: rnd(d) for a, d in sorted(by_stage.items())},
        "explainer_truncated_and_retried": n_trunc,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "features_done": done,
        "features_total": len(feats),
        "max_cost_usd": max_cost,
        "projection_usd": None if projection is None else round(projection, 2),
        "stopped_at": stopped_at,
    }
    print(f"[run] costs {json.dumps(costs['this_call'])} | cumulative ${cum['cost']:.4f} over "
          f"{cum['calls']} calls", flush=True)

    common_inputs = {
        "build": build_dir,
        "features": f"{done} of {len(feats)}",
        "arms": ",".join([*arm_names, floor_arm, draw2_arm]),
        "model": model,
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
            f"temperature {float(ac['temperature'])}, max_tokens {int(ac['explainer_max_tokens'])}. "
            f"The answer is the text after the LAST `[EXPLANATION]:`; a response without the tag "
            f"falls back to the whole body. {n_empty} of {len(expl_rows)} came back empty. "
            f"AMENDMENT A10: {n_trunc} answers hit max_tokens and were retried ONCE at double the "
            f"budget; a still-truncated answer raises rather than being kept as a short one."
        )
        od.note(
            f"the floor arm `{floor_arm}` (A6) and `{draw2_arm}` (A7) have NO explainer call of "
            f"their own: the first reuses a different feature's `{floor_src}` description under a "
            f"fixed derangement, the second reuses this feature's C16 description on the second, "
            f"disjoint test draw."
        )
    for scorer in scorers:
        rows = batch_rows[scorer]
        n_bad = sum(1 for r in rows if not r["parsed"])
        with out(scorer, status) as od:
            od.write_jsonl("batches.jsonl", rows)
            od.write_jsonl("per_feature.jsonl", [r for r in scores if r["scorer"] == scorer])
            od.write_json("costs.json", costs)
            od.note(
                f"Delphi {scorer} scorer, {batch} items per prompt, binary answers, "
                f"max_tokens {int(ac['scorer_max_tokens'])}. The parser takes the LAST bracketed "
                f"group of exactly the batch length; a wrong-length or unreadable answer DROPS the "
                f"batch rather than padding it. {n_bad} of {len(rows)} batches "
                f"({n_bad / max(1, len(rows)):.2%}) were unparsed and dropped."
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
                "answers with the negative side restricted to each half of the A5 mix, so a "
                "result that lives entirely on one half cannot hide in the pooled number."
            )
    with out("summary", status) as od:
        od.write_jsonl("scores.jsonl", scores)
        od.write_json("costs.json", costs)
        od.write_json("features.json", {"features": [fmeta[f] for f in feats[:done]]})
        od.write_json("build.json", binfo)
        od.write_json("floor_permutation.json",
                      {"floor_arm": floor_arm, "source_arm": floor_src,
                       "seed": int(ac["shuffle_seed"]),
                       "map": {str(k): v for k, v in perm.items()}})
        od.note(
            f"scores.jsonl is one row per (feature, arm, scorer): bal_acc, its two negative-half "
            f"restrictions, tpr/tnr, item counts, the arm's ACTUAL example count, which test draw "
            f"it used, whose description it used, and the feature covariates. {len(scores)} rows "
            f"over {done} features."
        )
        od.note(
            f"`{floor_arm}` is the floor: each feature's test set scored with ANOTHER feature's "
            f"`{floor_src}` description under a fixed derangement (no fixed point). "
            f"`{draw2_arm}` is the null: C16's own description on the second, disjoint test draw, "
            f"so the per-feature C16 - C16-draw2 difference is the test-set sampling noise every "
            f"contrast is exposed to."
        )
        if stopped_at:
            od.note(f"STOPPED EARLY: {stopped_at}")
        od.note(
            f"cost, from each response's own usage.cost: this call ${s['cost']:.4f} over "
            f"{s['calls']} calls ({s['fails']} failures); cumulative over the cache "
            f"${cum['cost']:.4f} over {cum['calls']} calls. Per arm and per stage in costs.json."
        )

    return {
        "out": run_root,
        "features_done": done,
        "features_total": len(feats),
        "arms": [*arm_names, floor_arm, draw2_arm],
        "scorers": scorers,
        "explainer_truncated": n_trunc,
        "explanations_empty": n_empty,
        "cost_this_call": round(s["cost"], 4),
        "cost_cumulative": round(cum["cost"], 4),
        "calls_this_call": s["calls"],
        "cache_hits": cache.hits,
        "projection_usd": costs["projection_usd"],
        "stopped_at": stopped_at,
    }
