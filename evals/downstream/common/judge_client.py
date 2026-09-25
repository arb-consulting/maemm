"""The judge transport every judged evaluation shares: a budget `Ledger`, two retrying HTTP clients
(OpenRouter chat completions and the Anthropic Messages API) with one `(text, usage)` shape, the cached,
resumable `run_requests`, and `decode_json_object` for structured replies. A judge stage runs::

    client = client_for(spec, key_for(spec))
    run_requests(run, judge_log(spec.name, "identify"), [with_judge(r, spec) for r in reqs], client, ledger)
"""

import hashlib, json, os, re, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TITLE = "maemm-eval"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
KEY_VARIABLES = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class BudgetExceeded(RuntimeError):
    pass


class JudgeError(RuntimeError):
    pass


class Retryable(RuntimeError):
    pass


class AuthError(JudgeError):
    """The endpoint refused the key or account (HTTP 401, 402 or 403, in `code`), so every further request
    would fail too; `run_requests` stops its pass on the first one."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

    def __reduce__(self):
        # keep `code` across pickling (a call run in another process re-raises by pickle)
        return type(self), (self.code, str(self))


class ModerationBlocked(JudgeError):
    """A 403 naming moderation: the provider refused this input. Deterministic and per-request, unlike
    `AuthError`."""


RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
AUTH_CODES = {401, 402, 403}
AUTH_MEANS = {401: "the key was not accepted", 402: "the account is out of credit",
              403: "the key has no access to this model or provider"}


def _refusal_of(code, detail, message):
    """The exception for a non-retryable status `code`; `detail` is the full error body."""
    if code == 403 and ("moderation" in detail.lower() or "flagged" in detail.lower()):
        return ModerationBlocked(message)
    if code in AUTH_CODES:
        return AuthError(code, message)
    return JudgeError(message)


def request_key(req):
    """The digest a request is logged and resumed under: model, both texts, token cap, and the `provider`,
    `sampling` and `transport` that `with_judge` bound. KeyError on an unaddressed request."""
    return hashlib.sha256(
        json.dumps([req["model"], req["system"], req["user"], req["max_tokens"], req["provider"], req["sampling"],
                    req["transport"]],
                   ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:24]


@dataclass(frozen=True)
class JudgeSpec:
    """One judge. `provider` is OpenRouter's routing pin (None for Anthropic), `sampling` request parameters
    sent verbatim, `max_tokens` an int or a dict by request `kind`, `rates` (input, output) US$ per million
    tokens, `name` the log directory. All but `name`, `label` and `rates` enter `request_key`."""

    name: str
    model: str
    provider: dict
    sampling: dict
    max_tokens: object
    label: str
    rates: tuple
    transport: str = "openrouter"

    def __post_init__(self):
        if self.transport not in KEY_VARIABLES:
            raise ValueError(f"judge {self.name!r} names transport {self.transport!r}; one of {sorted(KEY_VARIABLES)}")

    def max_tokens_for(self, kind):
        """The token cap for `kind`; KeyError when a dict has none (there is no default)."""
        return self.max_tokens if isinstance(self.max_tokens, int) else self.max_tokens[kind]


def judge_log(judge, instrument):
    """`judges/<judge>/<instrument>.jsonl`: one log per judge, so two judges never share a resume cache."""
    return f"judges/{judge}/{instrument}.jsonl"


def key_for(spec):
    """`spec`'s API key from its transport's variable (`KEY_VARIABLES`); RuntimeError when unset."""
    var = KEY_VARIABLES[spec.transport]
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(f"{var} is not set: judge {spec.name!r} ({spec.label}) is asked through the "
                           f"{spec.transport} transport, which reads its key from that variable")
    return key


def client_for(spec, api_key, url=OPENROUTER_URL, title=DEFAULT_TITLE):
    """The client for `spec`'s transport; `url` and `title` apply to OpenRouter only."""
    if spec.transport == "anthropic":
        return AnthropicClient(api_key, spec.provider, sampling=spec.sampling)
    return OpenRouterClient(api_key, spec.provider, sampling=spec.sampling, url=url, title=title)


def with_judge(req, spec):
    """A copy of `req` addressed to `spec`: its model, token cap, `provider`, `sampling` and `transport`."""
    return dict(req, model=spec.model, max_tokens=spec.max_tokens_for(req["kind"]),
                provider=spec.provider, sampling=spec.sampling, transport=spec.transport)


# a bare refusal, judged by the reply's opening only; a reply that merely lacks JSON stays "parse_fail"
_REFUSAL_RX = re.compile(
    r"^\s*(i'?m sorry|i (?:cannot|can't|am not able|'m not able|am unable|'m unable)|as an ai\b|i must decline|i won'?t\b)",
    re.I,
)


def looks_like_refusal(text):
    return bool(_REFUSAL_RX.match((text or "").strip()))


_DECODER = json.JSONDecoder()
# a string value followed by a gloss and a stray quote, e.g. "quote": "城市" (city)"; never valid JSON
_GLOSSED_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"\s*\([^()]*\)\s*"?')


def _objects(text):
    """Every JSON object that starts at some "{" of `text` and decodes, in order of appearance."""
    for k, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(text, k)
        except ValueError:
            continue
        yield obj


def decode_json_object(text, required_key=None, accept=None):
    """`(obj, repaired)`: the first JSON object in `text` that `accept` admits (default: has `required_key`,
    or any object), and whether the glossed-quote repair was needed; the unrepaired text is tried first."""
    text = text or ""
    admits = accept or (lambda o: required_key is None or required_key in o)
    for repaired, attempt in ((False, text), (True, _GLOSSED_STRING.sub(r'"\1"', text))):
        for obj in _objects(attempt):
            if isinstance(obj, dict) and admits(obj):
                return obj, repaired
    return None, False


class Ledger:
    def __init__(self, cap_usd, rates):
        self.cap, self.rates, self.lock = float(cap_usd), rates, threading.Lock()
        self.spent = 0.0
        self.reserved = 0.0
        self.n = 0
        self.tin = 0
        self.tout = 0

    def _rate(self, model):
        return self.rates[model]

    def estimate(self, req):
        rin, rout = self._rate(req["model"])
        est_in = len(req["system"] + req["user"]) / 3.5 * 1.1
        return est_in * rin / 1e6 + req["max_tokens"] * rout / 1e6

    def reserve(self, req):
        with self.lock:
            e = self.estimate(req)
            if self.spent + self.reserved + e > self.cap:
                raise BudgetExceeded(
                    f"judge cap US${self.cap:.2f} would be exceeded (spent {self.spent:.3f}, reserved {self.reserved:.3f})"
                )
            self.reserved += e
            return e

    def settle(self, req, usage):
        rin, rout = self._rate(req["model"])
        cost = usage["input_tokens"] * rin / 1e6 + usage["output_tokens"] * rout / 1e6
        with self.lock:
            self.reserved = max(0.0, self.reserved - self.estimate(req))
            self.spent += cost
            self.n += 1
            self.tin += usage["input_tokens"]
            self.tout += usage["output_tokens"]
        return cost

    def release(self, req):
        with self.lock:
            self.reserved = max(0.0, self.reserved - self.estimate(req))

    def state(self):
        # locked: worker threads may be settling during a mid-stage checkpoint
        with self.lock:
            return {
                "cap_usd": self.cap,
                "spent_usd": self.spent,
                "reserved_usd": self.reserved,
                "requests": self.n,
                "input_tokens": self.tin,
                "output_tokens": self.tout,
            }

    def save(self, path):
        # atomic, so a kill mid-write cannot leave a truncated ledger
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as h:
            json.dump(self.state(), h, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, cap, rates):
        L = cls(cap, rates)
        if os.path.exists(path):
            with open(path) as h:
                s = json.load(h)
            L.spent, L.n, L.tin, L.tout = s["spent_usd"], s["requests"], s["input_tokens"], s["output_tokens"]
        return L


def _check_addressed(req, transport, provider, sampling):
    """Refuse a request addressed with another transport, provider or sampling than this client's."""
    for block, mine in (("transport", transport), ("provider", provider), ("sampling", sampling)):
        if req.get(block) != mine:
            raise ValueError(f"request addressed with {block} {req.get(block)!r}, but this client sends "
                             f"{mine!r}: address a request with `with_judge(req, spec)` and send it "
                             f"through `client_for(spec, ...)` of the same spec")


class OpenRouterClient:
    TRANSPORT = "openrouter"

    def __init__(self, api_key, provider, timeout_s=120, sampling=None, url=OPENROUTER_URL, title=DEFAULT_TITLE):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": title,
        }
        self.provider, self.timeout, self.url = provider, timeout_s, url
        # the spec's request parameters, sent verbatim (an unsupported one is silently ignored upstream)
        self.sampling = dict(sampling or {})

    def body(self, req):
        _check_addressed(req, self.TRANSPORT, self.provider, self.sampling)
        return {
            "model": req["model"],
            "max_tokens": req["max_tokens"],
            **self.sampling,
            "provider": self.provider,
            "usage": {"include": True},
            "messages": self.messages(req),
        }

    @staticmethod
    def messages(req):
        """The user message, preceded by a system message only when there is system text."""
        user = {"role": "user", "content": req["user"]}
        return [{"role": "system", "content": req["system"]}, user] if req["system"] else [user]

    def _once(self, req):
        raw = _post(self.url, self.headers, self.body(req), self.timeout, _refusal_of)
        try:
            out = json.loads(raw)
            if out.get("error"):
                code = int((out["error"] or {}).get("code") or 0)
                if code in RETRY_CODES or code == 0:
                    raise Retryable(f"openrouter {code}")
                # the same families an HTTP status has, for an error that arrives inside a 200
                raise _refusal_of(code, str(out["error"]), f"openrouter error {code}: {str(out['error'])[:160]}")
            ch = (out.get("choices") or [{}])[0]
            u = out.get("usage") or {}
            text = (ch.get("message") or {}).get("content") or ""
            details = u.get("completion_tokens_details") or {}
            usage = {
                "input_tokens": int(u.get("prompt_tokens") or 0),
                "output_tokens": int(u.get("completion_tokens") or 0),
                "finish": ch.get("finish_reason"),
                "cost_reported": u.get("cost"),
                # billed inside completion_tokens; recorded to show reasoning stayed off
                "reasoning_tokens": int((details.get("reasoning_tokens") if isinstance(details, dict) else 0) or 0),
                "served_model": out.get("model"),
                "served_provider": out.get("provider"),
            }
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # a malformed body is treated as a transient transport failure and retried
            raise Retryable(f"malformed response: {e}") from e
        return text, usage

    def call(self, req):
        return _call(self._once, req)


_ANTHROPIC_AUTH_TYPES = {"authentication_error": 401, "permission_error": 403, "billing_error": 402}
_ANTHROPIC_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"}


def _anthropic_refusal(code, detail, message):
    """`_refusal_of` for Anthropic, also reading `_ANTHROPIC_AUTH_TYPES` from the error body."""
    try:
        etype = (json.loads(detail).get("error") or {}).get("type")
    except (json.JSONDecodeError, AttributeError):
        etype = None
    if etype in _ANTHROPIC_AUTH_TYPES:
        return AuthError(code if code in AUTH_CODES else _ANTHROPIC_AUTH_TYPES[etype], message)
    return _refusal_of(code, detail, message)


class AnthropicClient:
    """`OpenRouterClient`'s interface against the Anthropic Messages API (nothing is routed)."""

    TRANSPORT = "anthropic"

    def __init__(self, api_key, provider=None, timeout_s=120, sampling=None, url=ANTHROPIC_URL):
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        self.provider, self.timeout, self.url = provider, timeout_s, url
        self.sampling = dict(sampling or {})

    def body(self, req):
        _check_addressed(req, self.TRANSPORT, self.provider, self.sampling)
        body = {"model": req["model"], "max_tokens": req["max_tokens"], **self.sampling,
                "messages": self.messages(req)}
        if req["system"]:
            body["system"] = req["system"]
        return body

    @staticmethod
    def messages(req):
        """The user turn alone; the system text goes in the top-level `system` field."""
        return [{"role": "user", "content": req["user"]}]

    def _once(self, req):
        raw = _post(self.url, self.headers, self.body(req), self.timeout, _anthropic_refusal)
        try:
            out = json.loads(raw)
            text = "".join(block.get("text") or "" for block in out["content"] if block.get("type") == "text")
            u = out.get("usage") or {}
            stop = out.get("stop_reason")
            usage = {
                "input_tokens": int(u.get("input_tokens") or 0),
                "output_tokens": int(u.get("output_tokens") or 0),
                "finish": _ANTHROPIC_FINISH.get(stop, stop),
                "stop_reason": stop,
                "cost_reported": None,
                "reasoning_tokens": 0,
                "served_model": out.get("model"),
                "served_provider": "anthropic",
            }
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            raise Retryable(f"malformed response: {e}") from e
        return text, usage

    def call(self, req):
        return _call(self._once, req)


def _post(url, headers, body, timeout, refusal):
    """POST `body` as JSON and return the response text. Retry codes, dropped connections and timeouts
    raise `Retryable`; other statuses raise `refusal(code, detail, message)`."""
    data = json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        if e.code in RETRY_CODES:
            raise Retryable(f"HTTP {e.code}")
        detail = e.read()
        raise refusal(e.code, detail.decode(errors="replace"), f"HTTP {e.code}: {detail[:200]!r}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise Retryable(str(e)[:80])


def _call(once, req):
    """`once(req)` with two retries after `Retryable` (2 s, 4 s), then `JudgeError("transport exhausted")`."""
    for attempt in range(3):
        try:
            return once(req)
        except Retryable as e:
            if attempt == 2:
                raise JudgeError(f"transport exhausted: {e}")
            time.sleep(2.0 * (attempt + 1))


#: Error kinds of a request never really asked, re-asked on resume. Refusals, content filters, moderation
#: blocks and parse failures are answers: counted as missing, not re-asked.
UNASKED_KINDS = ("budget", "transport", "auth")


def moderation_blocked(r):
    """Whether record `r` is a moderation block, a deterministic answer that is never re-asked."""
    err = str(r.get("error") or "")
    return r.get("status") == "unavailable" and (
        r.get("error_kind") == "moderation" or ("HTTP 403" in err and "moderation" in err))


def unasked(records):
    """`{"budget": n, "transport": n, "auth": n}`: how many `records` were never really asked, by reason."""
    counts = dict.fromkeys(UNASKED_KINDS, 0)
    for r in records:
        kind = (r or {}).get("error_kind")
        if kind in counts:
            counts[kind] += 1
    return counts


def unasked_detail(counts):
    """`unasked` counts as the phrase every refusal message uses."""
    return (f"{counts['budget']:,} budget-refused, {counts['transport']:,} failed in transport, "
            f"{counts['auth']:,} turned away with the key (HTTP 401/402/403)")


def _is_retryable_record(r):
    """Whether a logged record is re-asked: an `UNASKED_KINDS` outcome, or a parse_fail not yet retried."""
    status = r.get("status")
    if status == "unavailable" and r.get("error_kind") in UNASKED_KINDS:
        return True
    if status == "parse_fail" and (r.get("attempts") or 0) < 2:
        return True
    return False


def _meta_id(meta):
    return json.dumps(meta, sort_keys=True, ensure_ascii=False)


# request fields copied onto each record, so a verdict can be re-scored from the log alone
RECORD_FIELDS = ("targets", "samples", "readout")

NOBODY_SERVED = {"served_model": None, "served_provider": None}


def served_by(root, judges):
    """`{judge: {(served model, served provider): n}}` over the answered records of `<root>/judges/<judge>/`."""
    out = {}
    for judge in judges:
        pairs, folder = {}, os.path.join(str(root), "judges", judge)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as h:
                for line in h:
                    r = json.loads(line) if line.strip() else {}
                    if r.get("usage"):
                        pair = (r.get("served_model"), r.get("served_provider"))
                        pairs[pair] = pairs.get(pair, 0) + 1
        out[judge] = pairs
    return out


def served_line(root, specs):
    """A report line naming, per judge, every (model, provider) pair that answered and how often."""
    def pair(model, provider, n):
        who = "not named by the response" if model is None and provider is None else (
            f"`{model}` via {provider}" if provider is not None else f"`{model}`, provider not named")
        return f"{who} × {n:,}"

    parts = []
    for judge, pairs in served_by(root, specs).items():
        ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], str(kv[0])))
        parts.append(f"`{judge}` (asked for `{specs[judge].model}`): "
                     + (", ".join(pair(m, p, n) for (m, p), n in ranked) if ranked else "no answered request"))
    return "answered by, as each response names its model and provider: " + "; ".join(parts)


def run_requests(run, rel, reqs, client, ledger, workers=16, status_fn=None, checkpoint=None,
                 checkpoint_every=200, is_retryable=None, record_fields=RECORD_FIELDS):
    """Send `reqs` through `client` under `ledger`, appending one record per request and requester `meta`
    to the JSONL `rel` and skipping keys already answered; returns `{key: record}`. `status_fn(kind, text,
    finish)` classifies replies; a parse_fail or empty reply is re-sent once. On an `AuthError` nothing more
    is sent, the rest are recorded as "auth", and it is raised after a final checkpoint."""
    status = status_fn or (lambda kind, text, finish: "ok" if (text or "").strip() else "unavailable")
    retryable = is_retryable or _is_retryable_record
    # append-only log: the last non-retryable record per key wins
    done = {}
    seen = set()
    for r in run.read_jsonl(rel):
        if retryable(r):
            continue
        done[r["key"]] = r
        seen.add((r["key"], _meta_id(r.get("meta"))))
    # identical requests are sent once but recorded once per requester's meta
    by_key = {}
    for r in reqs:
        k = request_key(r)
        if k in done:
            mid = (k, _meta_id(r.get("meta")))
            if mid not in seen:
                run.append_jsonl(rel, dict(done[k], meta=r.get("meta")))
                seen.add(mid)
            continue
        by_key.setdefault(k, []).append(r)
    todo = [rs[0] for rs in by_key.values()]
    lock = threading.Lock()

    def _dispatch(req):
        """Reserve, call, settle, classify. Raises BudgetExceeded or JudgeError (reservation released)."""
        ledger.reserve(req)
        try:
            text, usage = client.call(req)
        except JudgeError:
            ledger.release(req)
            raise
        cost = ledger.settle(req, usage)
        usage = dict(usage)
        return {
            "status": status(req["kind"], text, usage.get("finish")),
            "text": text,
            "served_model": usage.pop("served_model", None),
            "served_provider": usage.pop("served_provider", None),
            "usage": usage,
            "cost_usd": cost,
        }

    halt = {}  # {"error": the first AuthError of this pass}; empty while the key is being accepted

    def _unanswered(base, e, attempts):
        """The record of a call that raised, tagged "auth" (which halts the pass), "moderation" or "transport"."""
        msg = str(e)[:200]
        rec = dict(base, status="unavailable", error=msg, text=None, usage=None, **NOBODY_SERVED, cost_usd=0.0,
                   attempts=attempts)
        if isinstance(e, AuthError):
            halt.setdefault("error", e)
            rec["error_kind"] = "auth"
        elif isinstance(e, ModerationBlocked):
            rec["error_kind"] = "moderation"
        elif msg.startswith("transport exhausted"):
            rec["error_kind"] = "transport"
        return rec

    def one(req):
        key = request_key(req)
        base = {"key": key, "model": req["model"], "meta": req.get("meta"), "kind": req["kind"]}
        for field in record_fields:
            base[field] = req.get(field)
            if field == "samples":
                base["n_samples"] = len(req.get("samples") or [])
        attempts = 0
        if halt:
            msg = f"not sent: the pass stopped on HTTP {halt['error'].code}"
            return dict(base, status="unavailable", error=msg, text=None, usage=None, **NOBODY_SERVED,
                        cost_usd=0.0, attempts=attempts, error_kind="auth")
        try:
            attempts += 1
            result = _dispatch(req)
        except BudgetExceeded as e:
            return dict(
                base,
                status="unavailable",
                error=str(e),
                error_kind="budget",
                text=None,
                usage=None,
                **NOBODY_SERVED,
                cost_usd=0.0,
                attempts=attempts,
            )
        except JudgeError as e:
            return _unanswered(base, e, attempts)
        # one retry for a parse_fail or empty reply; a content-filtered reply would be blocked again
        filtered = (result.get("usage") or {}).get("finish") == "content_filter"
        if not filtered and (
            result["status"] == "parse_fail"
            or (result["status"] == "unavailable" and not (result["text"] or "").strip())
        ):
            try:
                attempts += 1
                result = _dispatch(req)
            except BudgetExceeded:
                attempts -= 1  # the retry was never dispatched; keep the first outcome and its attempt count
            except JudgeError as e:
                return _unanswered(base, e, attempts)
        rec = dict(
            base,
            status=result["status"],
            text=result["text"],
            usage=result["usage"],
            served_model=result["served_model"],
            served_provider=result["served_provider"],
            cost_usd=result["cost_usd"],
            attempts=attempts,
        )
        if result["status"] == "unavailable" and not (result["text"] or "").strip():
            rec["error"] = "empty content"
            rec["error_kind"] = "empty"
        return rec

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in todo]):
            rec = fut.result()
            with lock:
                run.append_jsonl(rel, rec)
                done[rec["key"]] = rec
                for dup in by_key.get(rec["key"], [])[1:]:
                    run.append_jsonl(rel, dict(rec, meta=dup.get("meta")))
                completed += 1
                if checkpoint is not None and checkpoint_every and completed % checkpoint_every == 0:
                    checkpoint()
    if halt:
        if checkpoint is not None:
            checkpoint()
        stopped = halt["error"]
        n = sum(1 for r in todo if done[request_key(r)].get("error_kind") == "auth")
        raise AuthError(
            stopped.code,
            f"{rel}: the pass stopped on HTTP {stopped.code} ({AUTH_MEANS[stopped.code]}) -- {stopped}. "
            f"{n:,} of the {len(todo):,} requests it had to send have no answer for it: nothing was sent "
            "after the first such response, each is recorded with error_kind \"auth\", and nothing is complete. "
            "Run the stage again under a key the endpoint accepts; it asks exactly those requests.")
    return done
