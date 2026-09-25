"""Stage `frontier_context_fluency` (methodology §6): every judged text's mean per-token log-likelihood
under `google/gemma-4-31B` and its gap to the source passage's, written to `scores/context_fluency.jsonl`
(one record per distinct text, then one per pair). `build_scorer` is the only GPU call.
"""

import hashlib
import time

import numpy as np

from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance
from evals.downstream.rollout_coherence import config as C
from evals.downstream.rollout_coherence.runs import stage_hashes

# Literal special-token spellings a judged text can contain as ordinary characters; tokenised with
# `split_special_tokens=True` they are scored as text. Their counts are reported.
SPECIAL_LITERALS = ("<mask>", "<s>", "</s>", "<pad>", "<unk>", "<bos>", "<eos>",
                    "<start_of_turn>", "<end_of_turn>")


def text_id(text):
    return "t" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dedupe_texts(pairs):
    """(distinct strings in first-appearance order, {text_id: index}): identical strings are scored once."""
    texts, index = [], {}
    for p in pairs:
        for s in (p["x"], p["partner"]):
            k = text_id(s)
            if k not in index:
                index[k] = len(texts)
                texts.append(s)
    return texts, index


def encode_text(tok, text, bos_id):
    """`<bos>` + the text, no chat template, no truncation, special-token spellings kept as text. Refuses
    anything but exactly one leading BOS, since a missing or second one changes the likelihood."""
    ids = list(tok.encode(text, add_special_tokens=True, split_special_tokens=True))
    if not ids or ids[0] != bos_id:
        raise ValueError(f"tokenizer did not prepend BOS {bos_id}: got {ids[:4]!r} for {text[:40]!r}")
    if bos_id in ids[1:]:
        raise ValueError(f"BOS {bos_id} appears again inside the text at {ids.index(bos_id, 1)}: {text[:40]!r}")
    return ids


def ll_summary(per_token):
    """{ll, n_tokens} from one text's per-token log-probabilities (`ll` None for an empty text)."""
    vals = [float(v) for v in per_token]
    return {"ll": float(np.mean(vals)) if vals else None, "n_tokens": len(vals)}


def check_ids_in_vocab(ids, vocab_size, text_ids):
    """Raise, naming the text, when a text's max token id is outside the vocabulary (else a device assert)."""
    for tid, m in zip(text_ids, ids):
        if m is not None and m >= vocab_size:
            raise ValueError(f"text {tid!r} tokenises to max id {m}, but vocab_size is {vocab_size} "
                             f"(valid ids are 0..{vocab_size - 1}): a literal special-token string likely "
                             f"parsed as a real special-token id")


def count_special_literals(texts):
    """Occurrences of each `SPECIAL_LITERALS` spelling across the distinct texts."""
    counts = {lit: 0 for lit in SPECIAL_LITERALS}
    n_with_any = 0
    for t in texts:
        hit = False
        for lit in SPECIAL_LITERALS:
            c = t.count(lit)
            if c:
                counts[lit] += c
                hit = True
        n_with_any += hit
    return {"counts": counts, "n_texts_with_any": n_with_any, "n_texts": len(texts)}


def plan_batches(lengths, batch_tokens, max_batch):
    """Indices grouped into scoring batches, sorted by length so a batch pads only to its own spread; a
    batch closes at `batch_tokens` padded tokens or `max_batch` items."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out, cur, cur_max = [], [], 0
    for i in order:
        wide = max(cur_max, lengths[i])
        if cur and (wide * (len(cur) + 1) > batch_tokens or len(cur) >= max_batch):
            out.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = wide
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- the GPU scorer

class TorchTextLM:
    """The loaded scorer as two calls: hidden states from the text tower, and the output embedding applied
    to a flat [N, H] selection of positions, so logits are only ever built for the scored positions."""

    def __init__(self, body, head, device, dtype, softcap=None):
        self.body, self.head, self.device, self.dtype, self.softcap = body, head, device, dtype, softcap

    def hidden(self, input_ids, attention_mask):
        out = self.body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def logits(self, h):
        """[N, H] -> [N, V] in float32, soft-capped when the config declares a cap."""
        import torch

        lg = self.head(h).float()
        if self.softcap:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        return lg


class GemmaScorer:
    """Per-token log-likelihood of standalone texts under the pinned scorer."""

    def __init__(self, adapter, tok, *, bos_id, pad_id, batch_tokens=C.FLUENCY_BATCH_TOKENS,
                 max_batch=C.FLUENCY_MAX_BATCH, logit_chunk=C.FLUENCY_LOGIT_CHUNK, log=print):
        self.adapter, self.tok, self.bos_id, self.pad_id = adapter, tok, bos_id, pad_id
        self.batch_tokens, self.max_batch, self.logit_chunk = batch_tokens, max_batch, logit_chunk
        self.log = log
        self.bos_ok = False          # set once a text has passed encode_text's one-leading-BOS check
        self.n_batches = 0

    def ll(self, texts):
        ids = [encode_text(self.tok, t, self.bos_id) for t in texts]
        self.bos_ok = True
        return [ll_summary(v) for v in self._per_token(ids)]

    def _per_token(self, ids_list):
        out = [None] * len(ids_list)
        batches = plan_batches([len(x) for x in ids_list], self.batch_tokens, self.max_batch)
        for k, batch in enumerate(batches):
            vals = self._forward(ids_list, batch)
            for j, i in enumerate(batch):
                out[i] = vals[j]
            self.n_batches += 1
            if (k + 1) % 20 == 0 or k + 1 == len(batches):
                self.log(f"  [fluency] batch {k + 1}/{len(batches)}", flush=True)
        return out

    def _forward(self, ids_list, batch):
        """Log-probabilities of every token after the BOS. Right padding: under a causal mask a trailing pad
        is never attended to, so each row scores as it would unpadded."""
        import torch

        vocab_size = self.adapter.head.weight.shape[0]
        check_ids_in_vocab([max(ids_list[i]) if ids_list[i] else -1 for i in batch], vocab_size, list(batch))
        dev = self.adapter.device
        width = max(len(ids_list[i]) for i in batch)
        inp = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            ids = ids_list[i]
            inp[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[j, :len(ids)] = 1
        inp, mask = inp.to(dev), mask.to(dev)
        with torch.no_grad():
            h = self.adapter.hidden(inp, mask)                   # [B, L, H]
            rows, cols, tgt, counts = [], [], [], []
            for j, i in enumerate(batch):
                ids = ids_list[i]
                for t in range(1, len(ids)):
                    rows.append(j)
                    cols.append(t - 1)                           # the position that predicts token t
                    tgt.append(ids[t])
                counts.append(max(len(ids) - 1, 0))
            if not rows:
                return [[] for _ in batch]
            sel = h[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)]     # [N, H]
            tt = torch.tensor(tgt, device=dev)
            vals = []
            for k in range(0, sel.shape[0], self.logit_chunk):
                lg = self.adapter.logits(sel[k:k + self.logit_chunk])                   # [n, V] float32
                lse = torch.logsumexp(lg, dim=-1)
                tl = lg.gather(1, tt[k:k + self.logit_chunk, None]).squeeze(1)
                vals.append((tl - lse).to("cpu"))
                del lg
            flat = torch.cat(vals).tolist()
        out, at = [], 0
        for n in counts:
            out.append(flat[at:at + n])
            at += n
        return out


# ---------------------------------------------------------------- loading

def _body_and_head(mdl):
    """(text tower, output embedding, the attribute path taken) of a multimodal wrapper or a causal LM."""
    for path in ("model.language_model", "language_model.model", "language_model", "model"):
        obj = mdl
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "embed_tokens"):
            head = mdl.get_output_embeddings()
            if head is None:
                raise ValueError(f"{type(mdl).__name__}.get_output_embeddings() is None: the checkpoint "
                                 f"loaded without an LM head, so no likelihood can be scored from it")
            return obj, head, path
    raise ValueError(f"no text tower with embed_tokens under {type(mdl).__name__}")


def _free_vision_tower(mdl):
    """Drop the multimodal checkpoint's vision modules before any batch runs."""
    freed = []
    for owner in (mdl, getattr(mdl, "model", None)):
        if owner is None:
            continue
        for name in ("vision_tower", "multi_modal_projector", "vqmodel"):
            if getattr(owner, name, None) is not None:
                setattr(owner, name, None)
                freed.append(name)
    return freed


def load_scorer(device, model, revision):
    """(TorchTextLM, tokenizer, info): the language model (falling back to the multimodal wrapper, recorded
    in `info["load_path"]`), checked for vocabulary, head width, special ids and a probe forward."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model, revision=revision)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    vocab = getattr(text_cfg, "vocab_size", None)
    if vocab != 262144:
        raise ValueError(f"{model} text vocab is {vocab}, expected 262144: wrong checkpoint or wrong config")
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    kw = dict(revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model, **kw)
        load_path = "AutoModelForCausalLM"
    except (ValueError, KeyError, TypeError) as e:
        import transformers

        cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
        if cls is None:
            raise
        mdl = cls.from_pretrained(model, **kw)
        load_path = f"Gemma4ForConditionalGeneration (AutoModelForCausalLM raised {type(e).__name__})"
    mdl.eval()
    freed = _free_vision_tower(mdl)
    if freed and torch.cuda.is_available():
        torch.cuda.empty_cache()
    body, head, body_path = _body_and_head(mdl)
    softcap = getattr(text_cfg, "final_logit_softcapping", None)
    adapter = TorchTextLM(body, head, device, torch.bfloat16, softcap=softcap)

    hidden = getattr(text_cfg, "hidden_size", None)
    if head.weight.shape[0] != vocab:
        raise ValueError(f"output embedding is [{head.weight.shape[0]}, ...], expected vocab {vocab}")
    if hidden is not None and head.weight.shape[1] != hidden:
        raise ValueError(f"output embedding width {head.weight.shape[1]} != hidden size {hidden}")
    for name, got, want in (("bos", tok.bos_token_id, 2), ("eos", tok.eos_token_id, 1),
                            ("pad", tok.pad_token_id, 0)):
        if got != want:
            raise ValueError(f"tokenizer {name}_token_id is {got}, expected {want}")
    with torch.no_grad():
        probe = torch.tensor([[tok.bos_token_id, tok.eos_token_id, tok.bos_token_id]], device=device)
        lg = adapter.logits(adapter.hidden(probe, torch.ones_like(probe))[0])
    if lg.shape != (3, vocab) or not bool(torch.isfinite(lg).all()):
        raise ValueError(f"probe forward produced {tuple(lg.shape)} logits, finite={bool(torch.isfinite(lg).all())}")

    info = {"model": model, "revision": revision, "load_path": load_path, "body_path": body_path,
            "model_type": getattr(cfg, "model_type", None), "vocab_size": int(vocab),
            "hidden_size": int(hidden) if hidden is not None else None, "dtype": str(torch.bfloat16),
            "attn_implementation": "sdpa", "device": str(device), "vision_freed": freed,
            "final_logit_softcapping": softcap,
            "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
            "n_params": int(sum(p.numel() for p in mdl.parameters()))}
    return adapter, tok, info


def build_scorer(args):
    """(scorer, info): the only GPU-touching call of the fluency stages."""
    adapter, tok, info = load_scorer(args.device, C.FLUENCY_SCORER["model"], C.FLUENCY_SCORER["revision"])
    return GemmaScorer(adapter, tok, bos_id=tok.bos_token_id, pad_id=tok.pad_token_id), info


# ---------------------------------------------------------------- the stages' logic

def score_run(pairs, scorer, *, log=print):
    """(text_records, pair_records, meta) for one pair manifest; every model call goes through `scorer`."""
    t_all = time.time()
    seconds = {}
    if not pairs:
        raise ValueError("no pairs to score: run the pair stage first")
    texts, tindex = dedupe_texts(pairs)
    uses, roles = {}, {}
    for p in pairs:
        for role, s in (("test", p["x"]), ("partner", p["partner"])):
            k = text_id(s)
            uses[k] = uses.get(k, 0) + 1
            roles.setdefault(k, set()).add(role)
    log(f"[fluency] {len(pairs)} pairs, {len(texts)} distinct texts", flush=True)

    t0 = time.time()
    ll = scorer.ll(texts)
    seconds["ll"] = time.time() - t0
    if len(ll) != len(texts):
        raise ValueError(f"scorer returned {len(ll)} records for {len(texts)} texts")

    # The most-reused text scored again alone: a difference is padding or batching leaking into the score.
    t0 = time.time()
    top = max(range(len(texts)), key=lambda i: uses.get(text_id(texts[i]), 0))
    repeat = scorer.ll([texts[top]])[0]
    seconds["ll_repeat"] = time.time() - t0
    ll_repeat_abs_diff = (None if repeat["ll"] is None or ll[top]["ll"] is None
                          else abs(float(repeat["ll"]) - float(ll[top]["ll"])))

    text_records = []
    for i, s in enumerate(texts):
        k, r = text_id(s), ll[i]
        text_records.append({"kind": "text", "text_id": k, "n_tokens": r["n_tokens"], "ll": r["ll"],
                             "n_uses": uses.get(k, 0), "roles": sorted(roles.get(k, ()))})

    pair_records = []
    for p in pairs:
        xk, pk = text_id(p["x"]), text_id(p["partner"])
        tx, tp = ll[tindex[xk]], ll[tindex[pk]]
        pair_records.append({
            "kind": "pair", "pid": p["pid"], "group": p["group"], "i": p["i"], "sample": p["sample"],
            "x_id": xk, "partner_id": pk,
            "ll_test": tx["ll"], "ll_partner": tp["ll"],
            "ll_gap": None if tx["ll"] is None or tp["ll"] is None else tx["ll"] - tp["ll"],
            "n_tokens_test": tx["n_tokens"], "n_tokens_partner": tp["n_tokens"],
        })

    seconds["total"] = time.time() - t_all
    meta = {"n_pairs": len(pair_records), "n_texts": len(text_records), "seconds": seconds, "checks": _checks(pair_records, texts, ll_repeat_abs_diff, scorer)}
    log(f"[fluency] done in {seconds['total']:.0f}s: {meta['checks']}", flush=True)
    return text_records, pair_records, meta


def _checks(pair_records, texts, ll_repeat_abs_diff, scorer):
    """Methodology §6's sanity checks, recorded rather than asserted."""
    lo, hi = C.LL_PARTNER_SANE
    part = [r["ll_partner"] for r in pair_records if r["ll_partner"] is not None]
    return {
        "bos_ok": bool(getattr(scorer, "bos_ok", False)),
        "dedupe_exact": {"n_text_slots": 2 * len(pair_records), "n_texts": len(texts)},
        "ll_repeat_abs_diff": ll_repeat_abs_diff,
        "ll_partner": {
            "mean": float(np.mean(part)) if part else None,
            "p05": float(np.percentile(part, 5)) if part else None,
            "p95": float(np.percentile(part, 95)) if part else None,
            "min": float(np.min(part)) if part else None, "max": float(np.max(part)) if part else None,
            "in_range": float(np.mean([lo <= v <= hi for v in part])) if part else None,
            "range": [lo, hi],
        },
        "special_literals": count_special_literals(texts),
    }


# ---------------------------------------------------------------- the stages

def fluency_config_hash(run, upstream):
    """The scorer's pins and batching, chained to the pair stage whose manifest is scored."""
    return config_hash({"scorer": C.FLUENCY_SCORER, "batch_tokens": C.FLUENCY_BATCH_TOKENS,
                        "max_batch": C.FLUENCY_MAX_BATCH, "logit_chunk": C.FLUENCY_LOGIT_CHUNK,
                        "scoring": C.SCORING_VERSION,
                        "upstream": stage_hashes(run, [upstream])})


def run_fluency_pass(args, run, stage, manifest, out, upstream):
    """`score_run` over one manifest's pairs, written whole to `out`."""
    chash = fluency_config_hash(run, upstream)
    if stage_done(run, stage, chash) and not args.force:
        print(f"[{stage}] up to date", flush=True)
        return
    pairs = run.read_json(manifest)["pairs"]
    t0 = time.time()
    scorer, info = build_scorer(args)
    load_seconds = time.time() - t0
    text_records, pair_records, meta = score_run(pairs, scorer)
    run.write_jsonl(out, text_records + pair_records)
    meta["seconds"]["load"] = load_seconds
    write_provenance(run, {f"{stage}_scorer": info, f"{stage}_checks": meta["checks"],
                           f"{stage}_seconds": meta["seconds"]}, stage=stage)
    mark_stage(run, stage, chash, {"n_pairs": meta["n_pairs"], "n_texts": meta["n_texts"],
                                   "load_path": info.get("load_path"), "phase_seconds": meta["seconds"]},
               started=t0)
    print(f"[{stage}] {meta['n_texts']} texts, {meta['n_pairs']} pairs", flush=True)


FLUENCY_OUT = "scores/context_fluency.jsonl"


def stage_frontier_context_fluency(args, run):
    """`frontier/context/pairs.json` into `scores/context_fluency.jsonl`."""
    run_fluency_pass(args, run, "frontier_context_fluency", "frontier/context/pairs.json", FLUENCY_OUT,
                     "frontier_context_pairs")
