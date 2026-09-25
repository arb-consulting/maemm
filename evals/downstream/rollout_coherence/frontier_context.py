"""Stage `frontier_context` (methodology §3-§4): the matched targets and every text the figure reads,
scored against them. Methods (`targets`, `retrieval`, `maem`, `continuation`, `nla`), each with its own
stage record, resume key and artifacts under `frontier/context/`; `targets` runs first, and every
invocation checks that a source passage scores a centred cosine of 1 against its own target.
"""

import functools
import time

import numpy as np

from evals.downstream.common import retrieval as R
from evals.downstream.common.background import load_centring_mean
from evals.downstream.common.model_io import (direction, free_model, load_base, load_inverter,
                                  refuse_unless_generation_agrees, stop_token_ids)
from evals.downstream.common.nla import nla_reader as N
from evals.downstream.common.retrieval import bank_cos
from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance
from evals.downstream.common.scorer import SCORE_MAX_LENGTH, new_filter_stats, reencode
from evals.downstream.rollout_coherence import config as C
from evals.downstream.rollout_coherence.documents import source_passage_ids
from evals.downstream.rollout_coherence.frontier_corpus import CORPUS_JSON, CORPUS_NPZ
from evals.downstream.rollout_coherence.model import filter_record, generate_continuation, generate_injected, source_prefix_ids
from evals.downstream.rollout_coherence.runs import stage_hashes

# a source text's centred cosine against its own matched target is 1 up to bf16 / float32 rounding
MATCHED_CEILING_TOL = 1e-3

TARGETS_NPZ = "frontier/context/targets.npz"
TARGETS_JSON = "frontier/context/targets.json"
SELFCHECK = "frontier/context/selfcheck/{}.json"
SCORES = "frontier/context/scores/{}.jsonl"
TEXTS = "frontier/context/texts/{}.jsonl"
PAIRS = "frontier/context/pairs.json"
RETRIEVAL_PART = "frontier/context/retrieval_scores.part{k}of{n}.npz"
RETRIEVAL_PART_STAGE = "frontier_context_retrieval_part{k}of{n}"
SCORED_AS = {m: ("source" if m == "targets" else m) for m in C.CONTEXT_METHODS}
ARTIFACTS = {m: SCORES.format(SCORED_AS[m]) for m in C.CONTEXT_METHODS}


# ---------------------------------------------------------------- the re-read protocol

REENCODE_MAX_TOKENS = SCORE_MAX_LENGTH
reencode_masked = functools.partial(reencode, sbatch=C.BANK_BATCH, norm_filter=False)
NATIVE_REENCODE = functools.partial(reencode_masked, max_length=C.NATIVE_MAX_TOKENS)
#: the two re-read windows: (re-encoder, batch, how a failure names it)
WINDOWS = {"short": (None, C.BANK_BATCH, "the shared re-read window"),
           "long": (NATIVE_REENCODE, C.NATIVE_BATCH, f"the {C.NATIVE_MAX_TOKENS}-token re-read window")}


def build_reader(args):
    """The pinned base model and tokenizer."""
    return load_base(args.device, C.MODEL, C.MODEL_REVISION)


def window_of(reencode=None):
    """The token window `reencode` reads in (its bound `max_length`, else the scorer's)."""
    bound = getattr(reencode, "keywords", None) or {}
    return int(bound.get("max_length", REENCODE_MAX_TOKENS))


def bank_reread(texts, dirs, mdl, tok, device, batch=C.BANK_BATCH, reencode=None, mu=None, stats=None):
    """`bank_cos` with the filter tally `stats` bound to whichever generator does the read."""
    if stats is None:
        return bank_cos(texts, dirs, mdl, tok, device, batch=batch, reencode=reencode, mu=mu)
    if reencode is None:
        return bank_cos(texts, dirs, mdl, tok, device, batch=batch, mu=mu, stats=stats)
    return bank_cos(texts, dirs, mdl, tok, device, batch=batch, mu=mu,
                    reencode=functools.partial(reencode, stats=stats))


# ---------------------------------------------------------------- what to run, and what is already done

def methods_flag(args, attr, flag, known):
    """The methods a `--context-methods`-style flag names (default: all of `known`), checked."""
    raw = [m.strip() for m in getattr(args, attr, ",".join(known)).split(",")]
    asked = [m for m in raw if m]
    bad = [m for m in asked if m not in known]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; {flag} accepts {list(known)}")
    return asked


def context_methods(args):
    """`--context-methods` as a list, in the order this stage runs them (`C.CONTEXT_METHODS`)."""
    asked = methods_flag(args, "_context_methods", "--context-methods", C.CONTEXT_METHODS)
    return [m for m in C.CONTEXT_METHODS if m in asked]


def method_done(run, args, stage, chash, artifact):
    """Recorded under its current key with its artifact on disk, and not `--force`d."""
    return not args.force and stage_done(run, stage, chash) and run.exists(artifact)


def nla_pins_record(pins):
    """The verbalizer's pins as a JSON-plain dict (`asdict` cannot copy the frozen sidecar hashes)."""
    return {"repo": pins.repo, "revision": pins.revision, "max_new": pins.max_new, "min_new": pins.min_new,
            "trunc": pins.trunc, "temp": pins.temp, "top_p": pins.top_p, "top_k": pins.top_k,
            "min_p": pins.min_p, "enable_thinking": pins.enable_thinking,
            "score_max_length": pins.score_max_length, "gen_chunk": pins.gen_chunk,
            "inject_layer": pins.inject_layer, "n_samples": pins.n_samples,
            "sidecar_sha256": dict(pins.sidecar_sha256),
            "pins_record": N.pins_record(pins)}


def method_config_hash(run, args, method, n):
    """This method's resume key, chained to the records it is built from."""
    common = {"method": method, "model": C.MODEL_REVISION, "scoring": C.SCORING_VERSION, "n": int(n),
              "window_tokens": C.WINDOW_TOKENS, "window": REENCODE_MAX_TOKENS, "smoke": bool(args.smoke)}
    upstream = ["capture"] if method == "targets" else ["frontier_context_targets"]
    if method == "retrieval":
        common["retrieval"] = {"sizes": [list(s) for s in C.corpus_sizes(args.smoke)[0]],
                               "spec": C.SEARCH_CORPUS.record(), "bank_batch": C.BANK_BATCH}
        upstream.append("frontier_corpus")
    elif method in ("maem", "continuation"):
        common["arm"] = {"inverter": C.INVERTER_REVISION if method == "maem" else None,
                         "sampling": C.SAMPLING, "n_samples": C.n_samples(method), "gen_chunk": C.GEN_CHUNK}
        if method != "continuation":
            common["arm"].update(inject_layer=C.INJECT_LAYER, steer_coeff=C.STEER_COEFF)
        common["seed"] = C.GEN_SEED + C.SEED_OFFSET[method]
    elif method == "nla":
        common["nla"] = nla_pins_record(C.NLA)
        common["native"] = {"max_tokens": C.NATIVE_MAX_TOKENS, "batch": C.NATIVE_BATCH}
        common["seed"] = C.GEN_SEED + C.SEED_OFFSET["nla"]
    common["upstream"] = stage_hashes(run, upstream)
    return config_hash(common)


def context_method_done(run, args, method, n):
    return method_done(run, args, f"frontier_context_{method}", method_config_hash(run, args, method, n),
                       ARTIFACTS[method])


def retrieval_shard(args, n_windows):
    """`(k, n, the window block this container scores)` from `--shard k/n` (default: the whole corpus)."""
    k, n = (int(x) for x in str(getattr(args, "shard", "0/1")).split("/"))
    return k, n, R.shard_block(n_windows, k, n)


def merge_requested(args):
    """`--merge`: the merge call of the sharded forward, where a missing part is an error."""
    return bool(getattr(args, "_merge", False))


# ---------------------------------------------------------------- the last-token read

def last_token_batch(h, keep, mask):
    """`(h_last [b, d], n_tokens [b], last_kept [b])` of one right-padded `[sink, tokens, pad...]` batch:
    the last content token is the last attended index."""
    import torch

    n = mask.sum(1).to(torch.long)                       # sink + the text's tokens
    pos = (n - 1).clamp(min=0)
    rows = torch.arange(h.shape[0], device=h.device)
    return (h[rows, pos].float().cpu().numpy().astype(np.float32),
            (n - 1).cpu().numpy().astype(np.int64),
            keep[rows, pos].cpu().numpy().astype(bool))


def read_targets(texts, mdl, tok, device, reencode=None, batch=C.BANK_BATCH):
    """`(h [n, d], n_tokens [n], last_kept [n])`: each text's last-token residual, read through the re-read
    generator at `bank_cos`'s batch so a later re-read reproduces it; `last_kept` records whether the norm
    filter would keep that token."""
    import torch

    gen = functools.partial(reencode_masked, norm_filter=True) if reencode is None else reencode
    h = np.zeros((len(texts), 0), dtype=np.float32)
    n_tokens = np.zeros(len(texts), dtype=np.int64)
    last_kept = np.zeros(len(texts), dtype=bool)
    with torch.no_grad():
        for s, hb, keep, mask in gen(texts, mdl, tok, device, sbatch=batch):
            rows, lens, kept = last_token_batch(hb, keep, mask)
            if h.shape[1] == 0:
                h = np.zeros((len(texts), rows.shape[1]), dtype=np.float32)
            h[s:s + rows.shape[0]] = rows
            n_tokens[s:s + rows.shape[0]] = lens
            last_kept[s:s + rows.shape[0]] = kept
    return h, n_tokens, last_kept


# ---------------------------------------------------------------- the records

def score_record(method, i, sid, sample, cos_raw, cos_centred, point=None, n_tokens=None, text=None,
                 window_id=None, search_cos=None, score_max_length=REENCODE_MAX_TOKENS):
    """One line of `scores/<method>.jsonl` (`cos_*` None for a blank text). `point`, `text`, `window_id`
    and `search_cos` are retrieval's; the search score selects a window and is never on the axis."""
    return {"method": method, "i": int(i), "id": sid, "sample": sample, "point": point,
            "cos_raw": cos_raw, "cos_centred": cos_centred, "n_tokens": n_tokens, "text": text,
            "window_id": window_id, "search_cos": search_cos, "score_max_length": int(score_max_length)}


class Context:
    """What one invocation's methods share: the clean base, `mu`, the activations and the checked targets."""

    def __init__(self, args, run, doc, mdl, tok, mu):
        self.args, self.run, self.doc = args, run, doc
        self.mdl, self.tok = mdl, tok
        self.device = args.device
        self.mu = mu
        self.idx = list(range(len(doc["sources"])))
        self._targets = None
        self._texts = None
        self.checks = {}
        self.tally = new_filter_stats()

    def new_tally(self):
        self.tally = new_filter_stats()
        return self.tally

    def sid(self, i):
        return self.doc["sources"][int(i)]["id"]

    def source_texts(self):
        """Each activation's `C.WINDOW_TOKENS`-token source tail, the text its target is read from."""
        from evals.downstream.rollout_coherence.frontier_pairs import source_cut

        if self._texts is None:
            self._texts = [source_cut(s, C.WINDOW_TOKENS, self.tok)[0] for s in self.doc["sources"]]
        return self._texts

    def set_targets(self, targets):
        self._targets = targets
        self.verify_matched()

    def targets(self):
        if self._targets is None:
            if not self.run.exists(TARGETS_NPZ):
                raise RuntimeError(
                    f"{TARGETS_NPZ} is missing: every method but `targets` scores against the re-captured "
                    f"targets, so run `python -m evals.downstream.rollout_coherence frontier_context --context-methods "
                    f"targets` first")
            self._targets = load_targets(self.run, self.idx)
            self.verify_matched()
        return self._targets

    def verify_matched(self, window="short"):
        """Raise unless each source text scores a centred cosine of 1 (within `MATCHED_CEILING_TOL`) against
        its own target, once per window. Returns `(check, raw, centred)`."""
        if window in self.checks:
            return self.checks[window]
        tgt = self._targets
        texts = self.source_texts()
        reenc, batch, where = WINDOWS[window]
        raw, centred = bank_cos(texts, tgt["dirs"], self.mdl, self.tok, self.device, batch=batch,
                                reencode=reenc, mu=self.mu)
        diag_raw = np.array([raw[t, t] for t in range(len(texts))], dtype=np.float64)
        diag = np.array([centred[t, t] for t in range(len(texts))], dtype=np.float64)
        kept = np.asarray(tgt["last_kept"], dtype=bool)
        bad = int(np.sum(diag < 1.0 - MATCHED_CEILING_TOL))
        check = {"window": window, "n": len(texts), "n_last_kept": int(kept.sum()),
                 "min_centred": float(diag.min()) if diag.size else None,
                 "mean_centred": float(diag.mean()) if diag.size else None,
                 "tol": MATCHED_CEILING_TOL, "below_tolerance": bad, "ok": bad == 0}
        self.checks[window] = (check, diag_raw, diag)
        print(f"[frontier_context] matched self-check in {where}: centred cosine of a source text against "
              f"its own target >= {check['min_centred']} on all {check['n']} activations (the norm filter, "
              f"which no score applies, would keep the target's token on {check['n_last_kept']})", flush=True)
        if not check["ok"]:
            raise RuntimeError(
                f"{bad} of {check['n']} source texts do not reproduce their own matched target in {where} "
                f"(centred cosine below {1.0 - MATCHED_CEILING_TOL}): the re-read is not the read the targets "
                f"were captured through, so no cosine here means what methodology §4 says it means")
        return self.checks[window]


def load_targets(run, idx):
    """`frontier/context/targets.npz` as a dict; raises unless it was captured for exactly `idx`."""
    z = np.load(run.file(TARGETS_NPZ))
    got = [int(i) for i in z["idx"]]
    if got != [int(i) for i in idx]:
        raise ValueError(f"{TARGETS_NPZ} holds {len(got)} activation rows starting at {got[:3]}; this "
                         f"invocation runs over {len(idx)} starting at {list(idx)[:3]}. Re-capture the "
                         f"targets (--context-methods targets --force)")
    return {"idx": np.asarray(got, dtype=np.int64), "h": z["h"].astype(np.float32),
            "dirs": z["dirs"].astype(np.float32), "n_tokens": z["n_tokens"],
            "last_kept": z["last_kept"].astype(bool), "ceiling_raw": z["ceiling_raw"]}


def score_rows(ctx, rows, method, reencode=None, batch=C.BANK_BATCH):
    """Score records of `rows` (`i`, `id`, `sample`, `text`, `n_tokens`) against their own targets."""
    if not rows:
        return []
    raw, centred = bank_reread([r["text"] for r in rows], ctx.targets()["dirs"], ctx.mdl, ctx.tok, ctx.device,
                               batch=batch, reencode=reencode, mu=ctx.mu, stats=ctx.tally)
    out = []
    for t, r in enumerate(rows):
        j = int(r["i"])
        blank = not r["text"].strip()
        out.append(score_record(method, r["i"], r["id"], r["sample"],
                                None if blank else float(raw[t, j]), None if blank else float(centred[t, j]),
                                n_tokens=r.get("n_tokens"), score_max_length=window_of(reencode)))
    return out


# ---------------------------------------------------------------- the verbalizer

def read_with_verbalizer(h, device, seed, row_ids=None):
    """`(rows, info)`: the NLA verbalizer's greedy and sampled explanations of the raw `h`, rows
    `[(row, "greedy" | sample, record)]`. The verbalizer is freed before anything is re-read."""
    if C.NLA.trunc != C.WINDOW_TOKENS:
        raise ValueError(f"the NLA reader truncates at {C.NLA.trunc} tokens and this package's budget is "
                         f"{C.WINDOW_TOKENS}: the 64-token view would not be budget-matched to the others")
    con = N.contract(N.load_sidecar(pins=C.NLA))
    snapshot = N.download_checkpoint(pins=C.NLA)
    # held to the pins; the snapshot's own sampling config is recorded, never applied
    checkpoint = N.check_checkpoint(snapshot, C.NLA)
    verb, vtok = N.load_verbalizer(device, snapshot, C.NLA)
    try:
        ids, mpos = N.prompt_ids(vtok, con, pins=C.NLA)
        t0 = time.time()
        samples, greedy = N.generate_explanations(verb, vtok, np.asarray(h, dtype=np.float32), ids, mpos,
                                                  device, seed=seed, pins=C.NLA, row_ids=row_ids)
        t_gen = time.time() - t0
    finally:
        verb = free_model(verb)
    rows = []
    for j in range(len(greedy)):
        rows.append((j, "greedy", greedy[j]))
        for s in range(len(samples[j])):
            rows.append((j, s, samples[j][s]))
    return rows, {"seed": seed, "prompt_ids": [int(t) for t in ids], "marker": int(mpos),
                  "prompt_len": len(ids), "nla": nla_pins_record(C.NLA), "checkpoint": checkpoint,
                  "gen_seconds": t_gen}


def scored_text(full, judged):
    """The generation as decoded, or blank when its judge view is blank (methodology §4)."""
    return full if judged.strip() else ""


def verbalizer_record(i, sid, sample, r, tok):
    """One line of `texts/nla.jsonl`: judge views `text*`, decoded `full_text*` (what cosines score), and
    the 64-token view's re-tokenised length, which its partner is cut to."""
    return {"i": int(i), "id": sid, "method": "nla", "sample": sample,
            "text": r["text"], "text_trunc": r["text_trunc"],
            "full_text": r["full_text"], "full_text_trunc": r["full_text_trunc"],
            "n_tokens": int(r["n_tokens"]),
            "n_trunc_tokens": len(tok.encode(r["text_trunc"], add_special_tokens=False)),
            "closed": bool(r["closed"]), "closed_trunc": bool(r["closed_trunc"]),
            "eos": bool(r["eos"]), "capped": bool(r["capped"]), "seed": r.get("seed")}


def verbalizer_stats(recs, stage):
    """The verbalizer's diagnostics over the sampled rows; a low close rate is printed, never raised."""
    sampled = [r for r in recs if r["sample"] != "greedy"]
    close_rate = N.close_rate(sampled)
    stats = {
        "close_rate": close_rate,
        "close_rate_trunc": (float(np.mean([r["closed_trunc"] for r in sampled])) if sampled else None),
        "mean_generated_tokens": float(np.mean([r["n_tokens"] for r in sampled])) if sampled else None,
        "mean_trunc_tokens": float(np.mean([r["n_trunc_tokens"] for r in sampled])) if sampled else None,
        "capped_share": float(np.mean([r["capped"] for r in sampled])) if sampled else None,
        "distinct_greedy": len({r["text"] for r in recs if r["sample"] == "greedy"}),
        "blank": sum(1 for r in recs if not r["text_trunc"].strip()),
    }
    if close_rate is not None and close_rate < C.NLA.min_close_rate:
        print(f"[{stage}] NOTE: NLA close-tag rate {close_rate:.2f} < {C.NLA.min_close_rate}; the "
              f"unclosed explanations are read whole and the rate is reported", flush=True)
    return stats


# ---------------------------------------------------------------- the methods

def method_targets(args, run, ctx):
    """Re-capture every target from its 64-token source tail (methodology §3), with its raw ceiling, and
    write the source passage's own cosines."""
    idx, texts = ctx.idx, ctx.source_texts()
    t0 = time.time()
    h, n_tokens, last_kept = read_targets(texts, ctx.mdl, ctx.tok, ctx.device)
    if h.shape[0] != len(idx):
        raise ValueError(f"read {h.shape[0]} targets for {len(idx)} activations")
    dirs = np.stack([direction(h[j], ctx.mu) for j in range(len(idx))]).astype(np.float32)
    ceiling = np.sum(h / np.linalg.norm(h, axis=1, keepdims=True) * dirs, axis=1).astype(np.float32)
    np.savez(run.file(TARGETS_NPZ), idx=np.asarray(idx, dtype=np.int64), h=h, dirs=dirs,
             n_tokens=n_tokens, last_kept=last_kept, ceiling_raw=ceiling)
    rec = {"n": len(idx), "window_tokens": C.WINDOW_TOKENS, "window": REENCODE_MAX_TOKENS,
           "n_last_kept": int(last_kept.sum()), "last_kept_share": float(np.mean(last_kept)),
           "mean_n_tokens": float(np.mean(n_tokens)), "max_n_tokens": int(np.max(n_tokens)),
           "n_at_window": int(np.sum(n_tokens >= REENCODE_MAX_TOKENS)),
           "mean_ceiling_raw": float(np.mean(ceiling)), "read_seconds": time.time() - t0}
    run.write_json(TARGETS_JSON, rec)
    ctx.set_targets(load_targets(run, idx))
    _check, raw, centred = ctx.verify_matched()
    run.write_jsonl(SCORES.format("source"), [
        score_record("source", i, ctx.sid(i), "passage", float(raw[j]), float(centred[j]),
                     n_tokens=len(source_passage_ids(ctx.doc["sources"][i]["ids"], ctx.doc["sources"][i]["pos"],
                                                     C.WINDOW_TOKENS)))
        for j, i in enumerate(idx)])
    print(f"[frontier_context] targets: {len(idx)} activations re-captured from their own source tail; "
          f"mean raw ceiling {rec['mean_ceiling_raw']:.3f}; the norm filter would keep the target's token "
          f"for {rec['n_last_kept']}/{len(idx)}", flush=True)
    return rec


def finish_retrieval(run, ctx, sizes, corpus, scores, n_shards, part_seconds):
    """Top-1 window per corpus size from the merged scores, its decoded text re-read and written to
    `scores/retrieval.jsonl`."""
    m = len(ctx.idx)
    t0 = time.time()
    ranks = R.rank_retrieval(scores, [n for _, n in sizes], 1, corpus)
    selected = sorted({int(ranks[n][j][0]) for _label, n in sizes for j in range(m)})
    text_of = {k: ctx.tok.decode(corpus.window_ids(k), skip_special_tokens=False) for k in selected}
    texts = list(dict.fromkeys(text_of.values()))
    raw, centred = bank_reread(texts, ctx.targets()["dirs"], ctx.mdl, ctx.tok, ctx.device, batch=C.BANK_BATCH,
                               mu=ctx.mu, stats=ctx.tally)
    row_of = {t: u for u, t in enumerate(texts)}
    recs, dupes = [], {}
    for label, n in sizes:
        dupes[label] = R.near_duplicates([scores[ranks[n][j][0], j] for j in range(m)],
                                         C.SEARCH_CORPUS.near_duplicate_cos)
        for j, i in enumerate(ctx.idx):
            k = int(ranks[n][j][0])
            u = row_of[text_of[k]]
            recs.append(score_record("retrieval", i, ctx.sid(i), "top1", float(raw[u, j]),
                                     float(centred[u, j]), point=label, n_tokens=len(corpus.window_ids(k)),
                                     text=text_of[k], window_id=corpus.window_id(k),
                                     search_cos=float(scores[k, j])))
    run.write_jsonl(SCORES.format("retrieval"), recs)
    rec = {"n": len(recs), "n_act": m, "n_windows": len(corpus), "sizes": [[label, n] for label, n in sizes],
           "n_shards": int(n_shards), "n_reread_texts": len(texts), "near_duplicates": dupes,
           "near_duplicate_cos": C.SEARCH_CORPUS.near_duplicate_cos,
           "bank_seconds": float(part_seconds), "merge_seconds": time.time() - t0}
    print(f"[frontier_context] retrieval: {len(corpus)} windows x {m} directions over {n_shards} part(s), "
          f"{len(recs)} records, near-duplicates {dupes}", flush=True)
    return rec


def method_retrieval(args, run, ctx):
    """Score this container's corpus block and merge once every block exists; None while parts are
    outstanding."""
    corpus = R.load_corpus(run, CORPUS_NPZ, C.SEARCH_CORPUS)
    _meta, sizes = R.corpus_sizes(run, CORPUS_JSON, len(corpus))
    n_windows = len(corpus)
    k, n, sl = retrieval_shard(args, n_windows)
    base = method_config_hash(run, args, "retrieval", len(ctx.idx))
    chash = R.part_config_hash(base, k, n, sl)
    stage = RETRIEVAL_PART_STAGE.format(k=k, n=n)
    bank = ctx.targets()["dirs"]
    if not method_done(run, args, stage, chash, RETRIEVAL_PART.format(k=k, n=n)):
        R.score_corpus_part(run, corpus, args.device, ctx.mdl, ctx.tok, bank, k, n, sl,
                            chash, RETRIEVAL_PART, RETRIEVAL_PART_STAGE, "frontier_context")
    else:
        print(f"[frontier_context] retrieval shard {k}/{n} up to date", flush=True)
    done, waiting = R.part_states(run, n, n_windows, RETRIEVAL_PART, RETRIEVAL_PART_STAGE, base)
    if waiting:
        R.refuse_or_wait(merge_requested(args), "frontier_context",
                         "frontier_context --context-methods retrieval", n, done, waiting)
        return None
    part_seconds = sum(float(run.read_json(f"stages/{RETRIEVAL_PART_STAGE.format(k=j, n=n)}.json")
                             .get("bank_seconds") or 0.0) for j in range(n))
    scores = R.merge_parts(run, n, n_windows, bank.shape[0], RETRIEVAL_PART)["scores"]
    return finish_retrieval(run, ctx, sizes, corpus, scores, n, part_seconds)


def build_inverter_prompt(tok):
    """`(prompt ids, marker position)` of the inverter prompt (`maem.prompts`)."""
    from maem.prompts import build_prompt_ids

    ids, mpos = build_prompt_ids(tok)
    return ids, mpos[0]


def write_generations(run, ctx, arm, samples, greedy, n_samples):
    """One generating method's texts, and their score records, written; returns both."""
    rows = []
    for j, i in enumerate(ctx.idx):
        for sample, r in [("greedy", greedy[j])] + [(s, samples[j][s]) for s in range(n_samples)]:
            rows.append({"i": int(i), "id": ctx.sid(i), "arm": arm, "sample": sample, "text": r["text"],
                         "n_gen": int(r["n_gen"]), "capped": bool(r["capped"]), "seed": r.get("seed")})
    run.write_jsonl(TEXTS.format(arm), rows)
    recs = score_rows(ctx, [{**r, "n_tokens": r["n_gen"]} for r in rows], arm)
    run.write_jsonl(SCORES.format(arm), recs)
    return rows, recs


def generation_record(ctx, arm, seed, rows, recs, t_gen, **extra):
    """The stage record's generation diagnostics, printed."""
    greedy_texts = [r["text"] for r in rows if r["sample"] == "greedy"]
    scored = [r["cos_centred"] for r in recs if r["cos_centred"] is not None]
    rec = {"n": len(recs), "n_act": len(ctx.idx), "arm": arm, "seed": seed, **extra,
           "sampling": dict(C.SAMPLING), "n_samples": C.n_samples(arm),
           "distinct_greedy": len(set(greedy_texts)),
           "blank": sum(1 for r in rows if not r["text"].strip()),
           "capped": sum(1 for r in rows if r["capped"]),
           "mean_cos_centred": float(np.mean(scored)) if scored else None, "gen_seconds": t_gen}
    print(f"[frontier_context] {arm}: {len(rows)} texts in {t_gen:.0f}s; {rec['distinct_greedy']}/"
          f"{len(ctx.idx)} distinct greedy; mean centred cosine {rec['mean_cos_centred']}", flush=True)
    return rec


def method_maem(args, run, ctx):
    """The inverter with the matched direction injected (methodology §4), freed before the re-read."""
    arm = "maem"
    tgt = ctx.targets()
    n_samples = C.n_samples(arm)
    prompt_ids, marker = build_inverter_prompt(ctx.tok)
    stops = stop_token_ids(ctx.tok, ctx.mdl)
    seed = C.GEN_SEED + C.SEED_OFFSET[arm]
    inverter = load_inverter(ctx.device, C.INVERTER, C.INVERTER_REVISION)
    t0 = time.time()
    try:
        # `stops` is read off the clean base, so both models must share a generation config
        refuse_unless_generation_agrees(ctx.mdl, inverter)
        samples, greedy = generate_injected(
            inverter, ctx.tok, tgt["dirs"], prompt_ids, marker, ctx.device,
            n_samples, seed, True, C.GEN_CHUNK, C.SAMPLING, C.INJECT_LAYER, C.STEER_COEFF, stop_ids=stops,
            row_ids=list(ctx.idx))
    finally:
        inverter = free_model(inverter)
    t_gen = time.time() - t0
    rows, recs = write_generations(run, ctx, arm, samples, greedy, n_samples)
    return generation_record(ctx, arm, seed, rows, recs, t_gen,
                             prompt_ids=[int(t) for t in prompt_ids], marker=int(marker), stop_ids=list(stops),
                             inverter_revision=C.INVERTER_REVISION)


def method_continuation(args, run, ctx):
    """The clean base continuing each activation's source prefix, no injection."""
    n_samples = C.n_samples("continuation")
    stops = stop_token_ids(ctx.tok, ctx.mdl)
    seed = C.GEN_SEED + C.SEED_OFFSET["continuation"]
    prompts = [source_prefix_ids(ctx.doc["sources"][i]) for i in ctx.idx]
    t0 = time.time()
    samples, greedy = generate_continuation(ctx.mdl, ctx.tok, prompts, ctx.device, n_samples, seed, True,
                                            C.GEN_CHUNK, C.SAMPLING, stop_ids=stops, row_ids=list(ctx.idx))
    t_gen = time.time() - t0
    rows, recs = write_generations(run, ctx, "continuation", samples, greedy, n_samples)
    return generation_record(ctx, "continuation", seed, rows, recs, t_gen, prompt_kind="source_prefix",
                             stop_ids=list(stops))


def method_nla(args, run, ctx):
    """The NLA verbalizer on the matched raw `h`, scored at its 64-token view (`nla`) and whole
    (`nla_native`, through the long window)."""
    tgt = ctx.targets()
    ctx.verify_matched("long")
    rows_in, info = read_with_verbalizer(np.asarray(tgt["h"], dtype=np.float32), ctx.device,
                                         C.GEN_SEED + C.SEED_OFFSET["nla"], row_ids=list(ctx.idx))
    rows = [verbalizer_record(ctx.idx[j], ctx.sid(ctx.idx[j]), sample, r, ctx.tok) for j, sample, r in rows_in]
    run.write_jsonl(TEXTS.format("nla"), rows)
    trunc_rows = [{"i": r["i"], "id": r["id"], "sample": r["sample"],
                   "text": scored_text(r["full_text_trunc"], r["text_trunc"]),
                   "n_tokens": r["n_trunc_tokens"]} for r in rows]
    native_rows = [{"i": r["i"], "id": r["id"], "sample": r["sample"],
                    "text": scored_text(r["full_text"], r["text"]), "n_tokens": r["n_tokens"]} for r in rows]
    recs = score_rows(ctx, trunc_rows, "nla")
    recs += score_rows(ctx, native_rows, "nla_native", reencode=NATIVE_REENCODE, batch=C.NATIVE_BATCH)
    run.write_jsonl(SCORES.format("nla"), recs)
    stats = verbalizer_stats(rows, "frontier_context")
    retok = [len(ctx.tok.encode(r["full_text"], add_special_tokens=False)) for r in rows]
    rec = {"n": len(recs), "n_texts": len(rows), "n_act": len(ctx.idx), **info, **stats,
           "n_truncated_native": sum(1 for n in retok if n > C.NATIVE_MAX_TOKENS),
           "native_max_tokens": C.NATIVE_MAX_TOKENS, "native_batch": C.NATIVE_BATCH}
    print(f"[frontier_context] nla: {len(rows)} explanations of the matched activations in "
          f"{info['gen_seconds']:.0f}s; close rate {stats['close_rate']}; distinct greedy "
          f"{stats['distinct_greedy']}/{len(ctx.idx)}", flush=True)
    return rec


METHODS = {"targets": method_targets, "retrieval": method_retrieval,
           "maem": method_maem, "continuation": method_continuation, "nla": method_nla}


# ---------------------------------------------------------------- the stage

def stage_frontier_context(args, run):
    """Run every requested method not already done, sharing one model load and one set of checks."""
    doc = run.read_json("data/documents.json")
    n = len(doc["sources"])
    if not n:
        raise ValueError("data/documents.json holds no sources: run `capture` first")
    asked = context_methods(args)
    pending = [m for m in asked if not context_method_done(run, args, m, n)]
    for m in asked:
        if m not in pending:
            print(f"[frontier_context] {m} up to date", flush=True)
    if not pending:
        return
    mu = load_centring_mean()
    mdl, tok = build_reader(args)
    ctx = Context(args, run, doc, mdl, tok, mu)
    for m in asked:
        # done-check and key recomputed here: `targets` may just have been written by this loop
        if context_method_done(run, args, m, n):
            continue
        chash = method_config_hash(run, args, m, n)
        started = time.time()
        tally = ctx.new_tally()
        rec = METHODS[m](args, run, ctx)
        if rec is None:
            # an incomplete retrieval split: its part is recorded, the method is not
            print(f"[frontier_context] {m} is not complete in this invocation; no stage record written",
                  flush=True)
            continue
        if m != "targets":
            # `targets` scores through the check alone, which keeps no tally
            rec = {**rec, "norm_filter": filter_record(tally)}
        checks = {w: c[0] for w, c in ctx.checks.items()}
        run.write_json(SELFCHECK.format(m), checks)
        mark_stage(run, f"frontier_context_{m}", chash, {**rec, "matched_ceiling": checks}, started=started)
        # per-method provenance name: methods run in parallel containers
        write_provenance(run, {f"frontier_context_{m}": rec, f"frontier_context_{m}_selfcheck": checks},
                         stage=f"frontier_context_{m}")
