"""Stage `frontier_context_pairs` (methodology §5): every judged text against its source passage, cut to
the text's re-tokenised length, as the manifest `frontier/context/pairs.json`. Groups: `retrieval@<label>`,
the greedy and draws 0-7 of `maemm`, `continuation`, `nla` and `nla_native`, and the best-of-k selections
`<arm>_k<k>`. A blank text is skipped. CPU + tokenizer only."""
import time

from evals.downstream.common import retrieval as R
from evals.downstream.rollout_coherence import config as C
from evals.downstream.rollout_coherence import frontier_context as FX
from evals.downstream.rollout_coherence import frontier_corpus as FCORP
from evals.downstream.rollout_coherence.documents import load_tokenizer, source_passage_ids
from evals.downstream.rollout_coherence.model import best_of_8
from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance
from evals.downstream.rollout_coherence.runs import stage_hashes


def n_retok(text, tok):
    return len(tok.encode(text, add_special_tokens=False))


def source_cut(src, n, tok):
    """`(the last n document tokens through the read position, decoded; whether the document ran short)`."""
    ids = source_passage_ids(src["ids"], src["pos"], n)
    return tok.decode(ids, skip_special_tokens=False), len(ids) < n


def skipped_by_group(skipped):
    out = {}
    for s in skipped:
        g = s.get("group") or s["pid"].split("/", 1)[0]
        out[g] = out.get(g, 0) + 1
    return out


def window_short_by_group(pairs):
    out = {}
    for p in pairs:
        out[p["group"]] = out.get(p["group"], 0) + int(bool(p["window_short"]))
    return out


def _sample_order(sample):
    """Sort key: greedy, then samples 0..7 (a texts file's order), then selections `k16`.. by k."""
    if sample == "greedy":
        return -1
    text = str(sample)
    return int(text[1:]) if text.startswith("k") else int(text)


def add_one(pairs, skipped, src, group, row, tok):
    """Append one text's pair (`row`: `sample`, `text`, `n_gen`, `reread_cos`, `extra`), or a skip record
    to `skipped` when the text is blank."""
    pid = f"{group}/{src['id']}/{row['sample']}"
    if not row["text"].strip():
        skipped.append({"pid": pid, "group": group, "i": src["i"], "id": src["id"],
                        "sample": row["sample"], "reason": "empty_text"})
        return
    n = n_retok(row["text"], tok)
    partner, short = source_cut(src, n, tok)
    pairs.append({"pid": pid, "group": group, "i": src["i"], "id": src["id"], "sample": row["sample"],
                  "n_gen": row["n_gen"], "n_retok": n, "reread_cos": row["reread_cos"], "x": row["text"],
                  "partner": partner, "window_short": short, "extra": row["extra"]})


def add_nine(pairs, skipped, src, group, rows, tok):
    """Append one activation's greedy plus `C.N_SAMPLES` samples of one group; raises unless all are present."""
    want = {"greedy"} | set(range(C.N_SAMPLES))
    if len(rows) != C.N_SAMPLES + 1 or {r["sample"] for r in rows} != want:
        raise ValueError(f"{group}: {src['id']} has samples "
                         f"{sorted(map(str, (r['sample'] for r in rows)))}; "
                         f"expected greedy and 0..{C.N_SAMPLES - 1}")
    for row in sorted(rows, key=lambda r: _sample_order(r["sample"])):
        add_one(pairs, skipped, src, group, row, tok)


def best_of_k_cosines(recs, k):
    """`(raw, centred)` exact best-of-k means over all of an arm's scored draws (an extended point's x);
    None where a draw lacks the value."""
    from evals.downstream.rollout_coherence.frontier_analysis import best_of_k_weights

    ranked = sorted((r for r in recs if r["sample"] != "greedy" and r["reread_cos"] is not None),
                    key=lambda r: (-float(r["reread_cos"]), int(r["sample"])))
    if not ranked:
        return None, None
    w = best_of_k_weights(len(ranked), k)
    raw = float(sum(float(r["reread_cos"]) * wi for r, wi in zip(ranked, w)))
    cen = [r.get("cos_centred") for r in ranked]
    centred = None if any(c is None for c in cen) else float(sum(float(c) * wi for c, wi in zip(cen, w)))
    return raw, centred


def selection_rows(recs, k):
    """One activation's best-of-k selected text (highest raw cosine of the first `k` draws) as a row."""
    first = sorted((r for r in recs if r["sample"] != "greedy" and int(r["sample"]) < k),
                   key=lambda r: int(r["sample"]))
    if len(first) != k:
        raise ValueError(f"best-of-{k} selection wants samples 0..{k - 1}, found "
                         f"{sorted(int(r['sample']) for r in first)}")
    pick = first[best_of_8(first)]
    raw, centred = best_of_k_cosines(recs, k)
    # `reread_cos` is the selected text's own cosine; the plotted x is the exact estimator over every draw
    return {"sample": f"k{k}", "text": pick["text"], "n_gen": pick["n_gen"],
            "reread_cos": pick["reread_cos"],
            "extra": {"selected_sample": int(pick["sample"]), "k": k, "capped": pick.get("capped"),
                      "cos_centred": pick.get("cos_centred"),
                      "n_draws": sum(1 for r in recs if r["sample"] != "greedy"),
                      "cos_best_of_k": raw, "cos_centred_best_of_k": centred}}


#: the judged groups besides retrieval: texts file, judged field, length field, scores file, method, extra.
CONTEXT_GROUPS = {
    "maemm": {"texts": "maemm", "text": "text", "n_gen": "n_gen", "scores": "maemm", "method": "maemm",
              "extra": {"capped": "capped"}},
    "continuation": {"texts": "continuation", "text": "text", "n_gen": "n_gen", "scores": "continuation",
                     "method": "continuation", "extra": {"capped": "capped"}},
    "nla": {"texts": "nla", "text": "text_trunc", "n_gen": "n_trunc_tokens", "scores": "nla",
            "method": "nla", "extra": {"closed": "closed_trunc", "n_tokens_native": "n_tokens"}},
    "nla_native": {"texts": "nla", "text": "text", "n_gen": "n_tokens", "scores": "nla",
                   "method": "nla_native", "extra": {"closed": "closed", "n_tokens_native": "n_tokens"}},
}


def selection_arms():
    """`C.EXTENDED_ARMS` that are judged groups: they carry the extended ladder."""
    return tuple(a for a in C.EXTENDED_ARMS if a in CONTEXT_GROUPS)


def context_groups(texts, scores):
    """The judged groups, `retrieval` first; raises, naming the missing files, if any group lacks them."""
    missing = []
    if scores.get("retrieval") is None:
        missing.append(f"retrieval ({FX.SCORES.format('retrieval')})")
    for group, spec in CONTEXT_GROUPS.items():
        if texts.get(spec["texts"]) is None:
            missing.append(f"{group} ({FX.TEXTS.format(spec['texts'])})")
        elif scores.get(spec["scores"]) is None:
            missing.append(f"{group} ({FX.SCORES.format(spec['scores'])})")
    if missing:
        raise RuntimeError(
            "frontier_context_pairs: the judged manifest needs every group and this run directory is "
            "missing " + "; ".join(missing) + ". Run `python -m evals.downstream.rollout_coherence frontier_context` "
            "until every method has its stage record, then build the manifest.")
    return ["retrieval"] + list(CONTEXT_GROUPS)


def _cosines(recs, method):
    """`{(activation, sample): (raw, centred)}` over one method's score records."""
    out = {}
    for r in recs:
        if r["method"] != method:
            continue
        key = (int(r["i"]), r["sample"])
        if key in out:
            raise ValueError(f"{method}: two score records for {r['id']} sample {r['sample']}")
        out[key] = (r["cos_raw"], r["cos_centred"])
    return out


def _retrieval_spans(recs, sizes):
    """`{(activation, size label): record}` over retrieval's score records."""
    want = {label for label, _n in sizes}
    out = {}
    for r in recs:
        if r["point"] not in want:
            raise ValueError(f"retrieval: a score record at corpus size {r['point']!r}, which the corpus "
                             f"does not list ({sorted(want)})")
        key = (int(r["i"]), r["point"])
        if key in out:
            raise ValueError(f"retrieval: two records for {r['id']} at size {r['point']!r}")
        out[key] = r
    return out


def _retrieval_pairs(src, by_size, sizes, tok, pairs, skipped):
    """One pair per corpus size for one activation, from the window the search selected."""
    for label, _n_windows in sizes:
        rec = by_size.get((src["i"], label))
        if rec is None:
            raise ValueError(f"retrieval: no window for {src['id']} at size {label!r}")
        add_one(pairs, skipped, src, f"retrieval@{label}",
                {"sample": "top1", "text": rec["text"] or "", "n_gen": rec["n_tokens"],
                 "reread_cos": rec["cos_raw"],
                 "extra": {"size": label, "window_id": rec["window_id"], "cos_centred": rec["cos_centred"]}},
                tok)


def build_context_pairs(doc, texts, scores, sizes, tok):
    """`(pairs, skipped)` for every judged group; `texts` / `scores` map a file name to its records. Every
    group must cover every activation of `doc`."""
    groups = context_groups(texts, scores)
    pairs, skipped = [], []
    by_size = _retrieval_spans(scores["retrieval"], sizes)
    idx = {int(s["i"]) for s in doc["sources"]}
    built, covered = {}, [("retrieval", {i for i, _label in by_size})]
    for group in groups:
        if group == "retrieval":
            continue
        spec = CONTEXT_GROUPS[group]
        per = {}
        for r in texts[spec["texts"]]:
            per.setdefault(int(r["i"]), []).append(r)
        built[group] = (spec, per, _cosines(scores[spec["scores"]], spec["method"]))
        covered.append((group, set(per)))
    for name, got in covered:
        if got != idx:
            raise ValueError(f"{name} covers {len(got)} activations and data/documents.json holds "
                             f"{len(idx)}: every group is reported over the same activations")
    for src in doc["sources"]:
        _retrieval_pairs(src, by_size, sizes, tok, pairs, skipped)
        for group, (spec, per, cos) in built.items():
            rows = []
            for r in per.get(int(src["i"]), []):
                key = (int(r["i"]), r["sample"])
                if key not in cos:
                    raise ValueError(f"{group}: no cosine for {src['id']} sample {r['sample']}; "
                                     f"{FX.SCORES.format(spec['scores'])} and "
                                     f"{FX.TEXTS.format(spec['texts'])} describe different generations")
                raw, centred = cos[key]
                rows.append({"sample": r["sample"], "text": r[spec["text"]], "n_gen": r[spec["n_gen"]],
                             "reread_cos": raw,
                             "extra": {**{k: r[v] for k, v in spec["extra"].items()}, "cos_centred": centred}})
            # greedy + first MAIN_K draws judged whole; an extended arm adds its best-of-k selections
            add_nine(pairs, skipped, src, group,
                     [r for r in rows if r["sample"] == "greedy" or int(r["sample"]) < C.MAIN_K], tok)
            if group in selection_arms():
                flat = [{**r, **(r["extra"] or {})} for r in rows]
                for k in C.EXTENDED_K:
                    add_one(pairs, skipped, src, C.selection_group(group, k), selection_rows(flat, k), tok)
    if len({p["pid"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate pair id")
    return pairs, skipped


def pairs_per_activation(n_sizes):
    """Pairs + skips the manifest holds per activation."""
    return n_sizes + (C.N_SAMPLES + 1) * len(CONTEXT_GROUPS) + len(C.EXTENDED_K) * len(selection_arms())


def frontier_context_pairs_config_hash(run, args):
    """Chained to each method's stage record, named one by one: a `frontier_context_` prefix scan would
    include this stage's own record and it would never read as done."""
    sizes, n_windows = C.corpus_sizes(args.smoke)
    return config_hash({"sizes": [list(s) for s in sizes], "corpus_windows": n_windows,
                        "window_tokens": C.WINDOW_TOKENS, "n_samples": C.N_SAMPLES,
                        "scoring": C.SCORING_VERSION, "groups": list(CONTEXT_GROUPS),
                        "best_of_k": list(C.BEST_OF_K),
                        "samples_by_arm": {a: C.n_samples(a) for a in selection_arms()},
                        "upstream": stage_hashes(run, [f"frontier_context_{m}" for m in C.CONTEXT_METHODS])})


def _records(run, template, names):
    """`{name: its records, or None where the file is not in this run directory}`."""
    return {name: (run.read_jsonl(template.format(name)) if run.exists(template.format(name)) else None)
            for name in names}


def stage_frontier_context_pairs(args, run):
    chash = frontier_context_pairs_config_hash(run, args)
    if stage_done(run, "frontier_context_pairs", chash) and not args.force:
        print("[frontier_context_pairs] up to date", flush=True)
        return
    started = time.time()
    texts = _records(run, FX.TEXTS, sorted({s["texts"] for s in CONTEXT_GROUPS.values()}))
    scores = _records(run, FX.SCORES, sorted({s["scores"] for s in CONTEXT_GROUPS.values()} | {"retrieval"}))
    groups = context_groups(texts, scores)
    # sizes from the ranked corpus file, not this invocation's config
    sizes = R.corpus_sizes(run, FCORP.CORPUS_JSON)[1]
    tok = load_tokenizer()
    doc = run.read_json("data/documents.json")
    pairs, skipped = build_context_pairs(doc, texts, scores, sizes, tok)
    n = len(doc["sources"])
    per_activation = pairs_per_activation(len(sizes))
    if len(pairs) + len(skipped) != per_activation * n:
        raise ValueError(f"{len(pairs)} pairs + {len(skipped)} skipped for {n} activations "
                         f"({per_activation} expected per activation over {groups})")
    run.write_json(FX.PAIRS, {"config": {
        "n": n, "n_pairs": len(pairs), "n_skipped": len(skipped), "methods": list(groups),
        "sizes": [list(s) for s in sizes], "selection_arms": list(selection_arms()),
        "selection_k": list(C.EXTENDED_K), "skipped_by_group": skipped_by_group(skipped),
        "n_window_short": window_short_by_group(pairs), "skipped": skipped}, "pairs": pairs})
    write_provenance(run, {"frontier_context_pairs_groups": list(groups),
                           "frontier_context_pairs_sizes": [list(s) for s in sizes]},
                     stage="frontier_context_pairs")
    mark_stage(run, "frontier_context_pairs", chash, {"n_pairs": len(pairs), "n_skipped": len(skipped),
                                                      "methods": list(groups), "n_act": n}, started=started)
    print(f"[frontier_context_pairs] {len(pairs)} pairs, {len(skipped)} skipped over groups {groups}",
          flush=True)
