"""Ten-way forced-choice identification of every family's bundles, on the AxBench package's instrument
(methodology.md, "Instrument"). Writes `scores/mc10_main.json` and `tables/mc10_*.csv`.
"""
import csv
import os
import random
import re
from collections import Counter, defaultdict

from evals.downstream.common.judges import REFERENCE_JUDGE
from evals.downstream.common.runs import RunDir
from evals.downstream.common.stats import boot_indices, boot_mean, paired_diff, wilson
from evals.downstream.steering_vector_inversion.artifacts import seed_for
from evals.downstream.steering_vector_inversion.judge import identification_prompt, parse_identification

from . import arms as A
from . import config as C
from . import judge as J
from . import plain_steer_arm

INSTRUMENT = "mc10"
KIND = "identify"
CANDIDATE_BEHAVIOURS = 5               # both poles of each
N_CANDIDATES = 2 * CANDIDATE_BEHAVIOURS
CHANCE = 1.0 / N_CANDIDATES
MAIN = "main"                          # the one candidate list, the tables' `list` column

READERS = ("maemm", "nla_native", "retrieval", "jlens")
STEERED_ARMS = plain_steer_arm.ARMS
DEFAULT_ARMS = (*READERS, *STEERED_ARMS, "base_l1", "shuffled", "heldout_matching")
SHUFFLED = "shuffled"
SHUFFLED_READER = "maemm"
CONTROL_ARMS = ("heldout_matching", "base_l1")   # judged first, for the gate
PAIRED_ARMS = (*READERS, plain_steer_arm.P.TABLE_ARM, "base_l1")
SUMMARY_ARM = "maemm"
SUMMARY_AGAINST = ("nla_native", "retrieval", "jlens", *STEERED_ARMS, "base_l1", "shuffled")

FAMILIES = "rollouts/families.json"
TRUTH = "scores/truth.json"
SCORES = "scores/mc10_{list}.json"

CELLS_CSV = "tables/mc10_cells.csv"
ARMS_CSV = "tables/mc10_arms.csv"
PAIRED_CSV = "tables/mc10_paired.csv"
SHUFFLED_CSV = "tables/mc10_shuffled.csv"
MISSINGNESS_CSV = "tables/mc10_missingness.csv"
SUMMARY_CSV = "tables/mc10_summary.csv"

CI_BUNDLES = "wilson95_bundles"
CI_VECTORS = "bootstrap_vectors_10000_seed0"
N_BOOT, BOOT_SEED = 10000, 0
MIN_VECTORS_CI = 5
MIN_VECTORS_PAIRED = 3
MIN_VECTORS_SUMMARY = 5


def vector_of(family_id):
    return str(family_id).rpartition("|")[0]


def arm_of(family_id):
    return str(family_id).rpartition("|")[2]


def opposite_pole(description_id):
    """`"persona:extraversion/+"` -> `"persona:extraversion/-"`; None when the id names no pole."""
    behaviour, sep, pole = str(description_id).rpartition("/")
    if not sep or pole not in ("+", "-"):
        return None
    return f"{behaviour}/{'-' if pole == '+' else '+'}"


def vector_kind(vector_id):
    """`bipo:persona`, `heldout:persona`, or `random` for a corpus-search-only random query."""
    head, _, tail = str(vector_id).partition("/")
    if head == "random":
        return head
    kind = {"bipo": "bipo", "heldout": "heldout"}.get(tail.split("/")[0]) if head and tail else None
    return f"{kind}:{head.partition(':')[0]}" if kind else ""


def behaviour_of_description(description_id):
    return str(description_id).rpartition("/")[0]


def behaviour_of(vector_id):
    head, _, tail = str(vector_id).partition("/")
    return head if head and tail else None


def pool():
    """The candidate sentences: every frozen persona description."""
    return dict(C.persona_descriptions())


def mirror_pairs(pool):
    """`{behaviour: (both pole ids)}` over the behaviours the pool describes at both poles."""
    by_behaviour = defaultdict(list)
    for name in pool:
        by_behaviour[behaviour_of_description(name)].append(name)
    return {b: tuple(sorted(ids)) for b, ids in by_behaviour.items() if len(ids) == 2}


def candidates(truth_ids, vector_id, pool, seed=C.DATA_SEED):
    """`{"ids", "answer"}`: the ten candidates (the target's behaviour and four others, both poles each) in an
    order seeded from the vector's behaviour alone, and the target's 1-based index; None unless the truth is
    exactly one description of the pool."""
    listed = [str(t) for t in (truth_ids or [])]
    if len(listed) != 1 or listed[0] not in pool:
        return None
    target = listed[0]
    pairs = mirror_pairs(pool)
    if len(pairs) < CANDIDATE_BEHAVIOURS:
        raise ValueError(f"The candidate pool describes {len(pairs)} behaviours at both poles; "
                         f"{CANDIDATE_BEHAVIOURS} are needed")
    behaviour = behaviour_of_description(target)
    if behaviour not in pairs:
        raise ValueError(f"The candidate pool holds one pole of {behaviour!r} and a list is whole pairs")
    chosen = [behaviour]
    rng = random.Random(seed_for(seed, "mc10-candidates", behaviour_of(vector_id) or str(vector_id)))
    others = sorted(b for b in pairs if b not in chosen)
    rng.shuffle(others)
    chosen += others[:CANDIDATE_BEHAVIOURS - len(chosen)]
    ids = [name for b in chosen for name in pairs[b]]
    rng.shuffle(ids)
    return {"ids": ids, "answer": ids.index(target) + 1}


def _matcher(family_filter):
    """A predicate over family ids from a callable, a regular expression, or None."""
    if family_filter is None:
        return None
    if isinstance(family_filter, str):
        pattern = re.compile(family_filter)
        return lambda family_id: bool(pattern.search(str(family_id)))
    return lambda family_id: bool(family_filter(str(family_id)))


def donor_vector(vector_id, behaviour):
    """The vector of `behaviour` built the way `vector_id` was: the same kind, training seed and pole."""
    _head, _sep, rest = str(vector_id).partition("/")
    return f"{behaviour}/{rest}"


def shuffled_sources(families, truth, pool, seed=C.DATA_SEED):
    """`{shuffled family id: {"family", "vector_id", "donor", "truth"}}`: vector `v`'s own question asked of
    MAEMM's bundles for the first behaviour of `v`'s list (in a seeded order) whose family exists."""
    out = {}
    for family_id in sorted(families, key=str):
        family = families[family_id]
        if family.get("arm") != SHUFFLED_READER:
            continue
        vector_id = family.get("vector_id") or vector_of(family_id)
        spec = candidates(truth.get(family_id), vector_id, pool)
        if spec is None:
            continue
        target = spec["ids"][int(spec["answer"]) - 1]
        own = behaviour_of_description(target)
        others = sorted({behaviour_of_description(i) for i in spec["ids"]} - {own})
        random.Random(seed_for(seed, "mc10-shuffled", MAIN, vector_id)).shuffle(others)
        for behaviour in others:
            donor = donor_vector(vector_id, behaviour)
            donor_family = families.get(f"{donor}|{SHUFFLED_READER}")
            if donor_family is not None:
                out[f"{vector_id}|{SHUFFLED}"] = {"family": donor_family, "vector_id": vector_id,
                                                  "donor": donor, "truth": [target]}
                break
    return out


def requests(root, arms, family_filter=None, snippets=C.BUNDLE_SIZE):
    """One request per (family, bundle) of `arms` passing `family_filter`, in order: the bundle's texts and
    the ten candidate sentences. Ids and the answer travel in `meta`, which is never sent."""
    run = RunDir(root)
    sentences_of = pool()
    families = run.read_json(FAMILIES)
    truth = run.read_json(TRUTH) if run.exists(TRUTH) else {}
    keep = set(arms)
    matches = _matcher(family_filter)
    # (family, vector the question is about, its truth, donor)
    sources = {fid: (fam, fam.get("vector_id") or vector_of(fid), truth.get(fid), None)
               for fid, fam in families.items()}
    if SHUFFLED in keep:
        for fid, src in shuffled_sources(families, truth, sentences_of).items():
            sources[fid] = (src["family"], src["vector_id"], src["truth"], src["donor"])
    out = []
    for family_id in sorted(sources, key=str):
        family, vector_id, truths, donor = sources[family_id]
        arm = SHUFFLED if donor is not None else family.get("arm")
        if arm not in keep or (matches is not None and not matches(family_id)):
            continue
        spec = candidates(truths, vector_id, sentences_of)
        if spec is None:
            continue
        sentences = [sentences_of[name] for name in spec["ids"]]
        meta_extra = {} if donor is None else {"donor": donor}
        for bundle in A.bundles(family):
            texts = [J.clean_sample(t) for t in bundle["texts"][:snippets]]
            out.append({"kind": KIND, "system": "",
                        "user": identification_prompt(texts, sentences),
                        "meta": {"family_id": family_id, "bundle": bundle["bundle"], "list": MAIN,
                                 "candidate_ids": list(spec["ids"]), "answer": spec["answer"], **meta_extra}})
    return out


def _row(meta, record, judge_name):
    """One scored bundle: `{"family_id", "bundle", "list", "judge", "status", "choice", "correct"}`; a
    bundle with no usable reply is not correct, and `status` says why."""
    record = record or {}
    ids = list(meta.get("candidate_ids") or [])
    answer = meta.get("answer")
    status = record.get("status") or "unavailable"
    choice = None
    if status == "ok":
        value, verdict = parse_identification(record.get("text") or "", record.get("refusal"))
        if verdict == "valid" and value is not None and 1 <= int(value) <= len(ids):
            choice = ids[int(value) - 1]
        else:
            status = "refused" if verdict == "refusal" else "parse_fail"
    row = {"family_id": meta.get("family_id"), "bundle": meta.get("bundle"),
           "list": meta.get("list") or MAIN, "judge": judge_name, "status": status, "choice": choice,
           "correct": None if answer is None else (choice is not None and choice == ids[int(answer) - 1])}
    if meta.get("donor"):
        row["donor"] = meta["donor"]
    return row


def _merge(run, rel, new, key_fields):
    """`new` merged into the JSON list at `rel` on `key_fields`, written back sorted."""
    old = run.read_json(rel) if run.exists(rel) else []
    keyed = {tuple(str(r.get(k)) for k in key_fields): r for r in old}
    keyed.update({tuple(str(r.get(k)) for k in key_fields): r for r in new})
    rows = [keyed[k] for k in sorted(keyed)]
    run.write_json(rel, rows)
    return rows


def run(root, judged, judges=None, arms=None, family_filter=None):
    """The ten-way choice per judge over `arms` (default `DEFAULT_ARMS`); the merged rows of
    `scores/mc10_main.json`. `judged(instrument, requests, judge_name)` returns one record per request."""
    judges = tuple(C.JUDGES) if judges is None else tuple(judges)
    run_dir = RunDir(root)
    reqs = requests(root, DEFAULT_ARMS if arms is None else arms, family_filter)
    families = {r["meta"]["family_id"] for r in reqs}
    rows = []
    for judge_name in judges:
        print(f"[{INSTRUMENT}] {judge_name}: {len(reqs)} requests over {len(families)} families x "
              f"{N_CANDIDATES} candidates, projected US${J.estimate(reqs, judge_name):.2f}", flush=True)
        records = list(judged(f"{INSTRUMENT}_{MAIN}", reqs, judge_name)) if reqs else []
        if len(records) != len(reqs):
            raise ValueError(f"{INSTRUMENT}/{judge_name}: {len(records)} records for {len(reqs)} requests; "
                             f"`judged` must return one record per request, in request order")
        rows += [_row(req["meta"], rec, judge_name) for req, rec in zip(reqs, records)]
    return _merge(run_dir, SCORES.format(list=MAIN), rows, ("family_id", "bundle", "judge"))


CELL_COLUMNS = ("list", "family_id", "vector_id", "arm", "kind", "behaviour", "pair_source", "judge",
                "truth", "n_bundles", "n_answered", "k_correct", "rate", "ci_lower", "ci_upper",
                "ci_method", "n_opposite", "modal_wrong")
ARM_COLUMNS = ("list", "arm", "kind", "judge", "n_vectors", "mean_rate", "ci_lower", "ci_upper",
               "ci_method", "min_rate", "max_rate", "chance", "rates_by_vector")
PAIRED_COLUMNS = ("list", "arm_a", "arm_b", "kind", "judge", "n_vectors", "mean_diff", "ci_lower",
                  "ci_upper", "ci_method", "n_a_greater", "n_b_greater", "n_tied")
SUMMARY_COLUMNS = ("list", "judge", "row_type", "kind", "kind_b", "arm_a", "arm_b", "n_vectors", "value",
                   "ci_lower", "ci_upper", "ci_method", "chance")
SHUFFLED_COLUMNS = ("list", "kind", "judge", "n_vectors", "n_bundles", "target_rate", "donor_rate",
                    "donor_behaviour_rate", "chance")

KIND_ORDER = ("bipo:persona", "heldout:persona")
POOLED = J.POOLED


def _kind_rank(kind):
    return (KIND_ORDER.index(kind), "") if kind in KIND_ORDER else (len(KIND_ORDER), str(kind))


def _behaviour_rank(behaviour):
    order = list(C.BEHAVIOURS)
    return (order.index(behaviour), "") if behaviour in order else (len(order), behaviour or "")


def _arm_rank(arm):
    return (DEFAULT_ARMS.index(arm), "") if arm in DEFAULT_ARMS else (len(DEFAULT_ARMS), str(arm))


def _judge_rank(judge_name):
    order = list(C.JUDGES)
    return (order.index(judge_name), "") if judge_name in order else (len(order), str(judge_name))


def _judges_in(names):
    return sorted({n for n in names if n is not None}, key=_judge_rank)


def _blank(value):
    return value is None or value == "" or (isinstance(value, float) and value != value)


def _rate(numerator, denominator):
    return (numerator / denominator) if denominator else float("nan")


def _modal(values):
    counts = Counter(v for v in values if v is not None)
    return min(counts, key=lambda v: (-counts[v], str(v))) if counts else ""


def _by_vector(pairs, digits=3):
    """`vector=rate` pairs in one CSV cell."""
    return "|".join(f"{v}={'n/a' if _blank(r) else format(r, f'.{digits}f')}" for v, r in pairs)


def _write_csv(root, rel, columns, rows):
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in rows:
            writer.writerow(["" if _blank(r.get(c)) else
                             (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in columns])
    return path


def _read_csv(root, rel):
    path = os.path.join(str(root), rel)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _grouped(rows):
    out = defaultdict(dict)
    for r in rows:
        out[(r.get("family_id"), r.get("judge"))][r.get("bundle")] = r
    return out


def _target(truth, family_id, sentences):
    listed = [str(t) for t in (truth.get(family_id) or [])]
    return listed[0] if len(listed) == 1 and listed[0] in sentences else None


def _cell_rows(rows, truth, sentences):
    """One row per (scorable family, judge): k of all n bundles correct, `n_answered`, `n_opposite` (the
    target's opposite pole chosen) and `modal_wrong` (the description preferred instead)."""
    out = []
    for (family_id, judge_name), got in _grouped(rows).items():
        vector_id = vector_of(family_id)
        target = _target(truth, family_id, sentences)
        if target is None:
            continue
        bundles = sorted(got, key=str)
        choices = [got[b].get("choice") for b in bundles]
        answered = sum(got[b].get("status") == "ok" for b in bundles)
        k = sum(bool(got[b].get("correct")) for b in bundles)
        n = len(bundles)
        opposite = opposite_pole(target)
        rate, lo, hi = wilson(k, n) if n else (float("nan"),) * 3
        out.append({
            "list": MAIN, "family_id": family_id, "vector_id": vector_id, "arm": arm_of(family_id),
            "kind": vector_kind(vector_id), "behaviour": behaviour_of_description(target),
            "pair_source": "persona_statements",
            "judge": judge_name, "truth": target, "n_bundles": n, "n_answered": answered,
            "k_correct": k, "rate": rate, "ci_lower": lo, "ci_upper": hi, "ci_method": CI_BUNDLES,
            "n_opposite": sum(c == opposite for c in choices),
            "modal_wrong": _modal([c for c in choices if c is not None and c != target]),
        })
    return sorted(out, key=lambda r: (_behaviour_rank(r["behaviour"]), _kind_rank(r["kind"]),
                                      _arm_rank(r["arm"]), str(r["family_id"]), _judge_rank(r["judge"])))


def _interval(values, minimum=MIN_VECTORS_CI):
    """(mean, lo, hi, ci_method): a percentile bootstrap over vectors, empty below `minimum` of them."""
    n = len(values)
    if not n:
        return float("nan"), float("nan"), float("nan"), ""
    mean = sum(values) / n
    if n < minimum:
        return mean, float("nan"), float("nan"), ""
    _est, lo, hi = boot_mean(values, boot_indices(n, N_BOOT, BOOT_SEED))
    return mean, lo, hi, CI_VECTORS


def _arm_rows(cells):
    groups = defaultdict(list)
    for r in cells:
        if not _blank(r.get("rate")):
            groups[(r["list"], r["arm"], r["kind"], r["judge"])].append((r["vector_id"], r["rate"]))
    out = []
    for (listing, arm, kind, judge_name), pairs in groups.items():
        pairs = sorted(pairs)
        rates = [p[1] for p in pairs]
        mean, lo, hi, method = _interval(rates)
        out.append({"list": listing, "arm": arm, "kind": kind, "judge": judge_name, "n_vectors": len(pairs),
                    "mean_rate": mean, "ci_lower": lo, "ci_upper": hi, "ci_method": method,
                    "min_rate": min(rates), "max_rate": max(rates), "chance": CHANCE,
                    "rates_by_vector": _by_vector(pairs)})
    return sorted(out, key=lambda r: (r["list"], _arm_rank(r["arm"]), _kind_rank(r["kind"]),
                                      _judge_rank(r["judge"])))


def _paired_rows(cells, arms=PAIRED_ARMS, minimum=MIN_VECTORS_PAIRED):
    """Per (arm pair, kind, judge): the mean paired difference over shared vectors, its interval and sign
    counts; no row below `minimum` shared vectors."""
    rates = {(r["arm"], r["kind"], r["judge"], r["vector_id"]): r["rate"]
             for r in cells if not _blank(r.get("rate"))}
    kinds = sorted({r["kind"] for r in cells}, key=_kind_rank)
    judges = _judges_in({r["judge"] for r in cells})
    here = lambda arm, kind, judge_name: {v for (a, k, j, v) in rates if (a, k, j) == (arm, kind, judge_name)}
    out = []
    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1:]:
            for kind in kinds:
                for judge_name in judges:
                    shared = sorted(here(arm_a, kind, judge_name) & here(arm_b, kind, judge_name))
                    if len(shared) < minimum:
                        continue
                    a = [rates[(arm_a, kind, judge_name, v)] for v in shared]
                    b = [rates[(arm_b, kind, judge_name, v)] for v in shared]
                    est, lo, hi, n = paired_diff(a, b, boot_indices(len(shared), N_BOOT, BOOT_SEED))
                    out.append({"list": MAIN, "arm_a": arm_a, "arm_b": arm_b, "kind": kind,
                                "judge": judge_name, "n_vectors": n, "mean_diff": est, "ci_lower": lo,
                                "ci_upper": hi, "ci_method": CI_VECTORS,
                                "n_a_greater": sum(x > y for x, y in zip(a, b)),
                                "n_b_greater": sum(x < y for x, y in zip(a, b)),
                                "n_tied": sum(x == y for x, y in zip(a, b))})
    return out


def _summary_rows(cells):
    """`arm` rows (each arm's mean per-vector rate with its interval) and `paired` rows (`maemm` minus each
    arm of `SUMMARY_AGAINST` over shared vectors, at least `MIN_VECTORS_SUMMARY`)."""
    judges = _judges_in({r["judge"] for r in cells})
    kinds = sorted({r["kind"] for r in cells}, key=_kind_rank)
    rates = defaultdict(list)          # (judge, kind, arm, vector) -> rates
    for r in cells:
        if not _blank(r.get("rate")):
            rates[(r["judge"], r["kind"], r["arm"], r["vector_id"])].append(r["rate"])
    mean_of = lambda values: sum(values) / len(values)
    here = lambda judge_name, kind, arm: {v for (j, k, a, v) in rates if (j, k, a) == (judge_name, kind, arm)}
    out = []
    for judge_name in judges:
        for kind in kinds:
            arms = sorted({a for (j, k, a, _v) in rates if (j, k) == (judge_name, kind)}, key=_arm_rank)
            for arm in arms:
                values = [mean_of(rates[(judge_name, kind, arm, v)]) for v in sorted(here(judge_name, kind, arm))]
                mean, lo, hi, method = _interval(values, MIN_VECTORS_SUMMARY)
                out.append({"list": MAIN, "judge": judge_name, "row_type": "arm", "kind": kind, "kind_b": "",
                            "arm_a": arm, "arm_b": "", "n_vectors": len(values), "value": mean,
                            "ci_lower": lo, "ci_upper": hi, "ci_method": method, "chance": CHANCE})
            if SUMMARY_ARM not in arms:
                continue
            for arm_b in SUMMARY_AGAINST:
                shared = sorted(here(judge_name, kind, SUMMARY_ARM) & here(judge_name, kind, arm_b))
                if len(shared) < MIN_VECTORS_SUMMARY:
                    continue
                a = [mean_of(rates[(judge_name, kind, SUMMARY_ARM, v)]) for v in shared]
                b = [mean_of(rates[(judge_name, kind, arm_b, v)]) for v in shared]
                est, lo, hi, n = paired_diff(a, b, boot_indices(len(shared), N_BOOT, BOOT_SEED))
                out.append({"list": MAIN, "judge": judge_name, "row_type": "paired", "kind": kind,
                            "kind_b": "", "arm_a": SUMMARY_ARM, "arm_b": arm_b, "n_vectors": n, "value": est,
                            "ci_lower": lo, "ci_upper": hi, "ci_method": CI_VECTORS, "chance": None})
    return out


def _loaded(root):
    """`(scored rows, truth, candidate sentences)`, or None when nothing has been judged."""
    run_dir = RunDir(root)
    rel = SCORES.format(list=MAIN)
    rows = run_dir.read_json(rel) if run_dir.exists(rel) else []
    if not rows:
        return None
    truth = dict(run_dir.read_json(TRUTH) if run_dir.exists(TRUTH) else {})
    for r in rows:
        fid = r.get("family_id")
        if arm_of(fid) == SHUFFLED and fid not in truth:
            truth[fid] = truth.get(f"{vector_of(fid)}|{SHUFFLED_READER}")
    return rows, truth, pool()


def _shuffled_rows(rows, truth):
    """Per (kind, judge), bundle shares of the shuffled arm naming the vector's own truth, the donor's
    description, and either pole of the donor's behaviour."""
    groups = defaultdict(lambda: {"vectors": set(), "n": 0, "target": 0, "donor": 0, "donor_behaviour": 0})
    for r in rows:
        if arm_of(r.get("family_id")) != SHUFFLED or not r.get("donor"):
            continue
        donor_truth = [str(t) for t in (truth.get(f"{r['donor']}|{SHUFFLED_READER}") or [])]
        g = groups[(vector_kind(vector_of(r["family_id"])), r.get("judge"))]
        g["vectors"].add(vector_of(r["family_id"]))
        g["n"] += 1
        g["target"] += bool(r.get("correct"))
        choice = r.get("choice")
        g["donor"] += bool(donor_truth) and choice == donor_truth[0]
        g["donor_behaviour"] += (bool(donor_truth) and choice is not None
                                 and behaviour_of_description(choice) == behaviour_of_description(donor_truth[0]))
    return [{"list": MAIN, "kind": kind, "judge": judge_name, "n_vectors": len(g["vectors"]),
             "n_bundles": g["n"], "target_rate": _rate(g["target"], g["n"]),
             "donor_rate": _rate(g["donor"], g["n"]), "donor_behaviour_rate": _rate(g["donor_behaviour"], g["n"]),
             "chance": CHANCE}
            for (kind, judge_name), g in sorted(groups.items(), key=lambda kv: (_kind_rank(kv[0][0]),
                                                                               _judge_rank(kv[0][1])))]


def tables(root):
    """Every table of the ten-way choice from the saved scores and truth; `{name: rows}`."""
    names = ("mc10_summary", "mc10_cells", "mc10_arms", "mc10_paired", "mc10_missingness", "mc10_shuffled")
    loaded = _loaded(root)
    if loaded is None:
        return {name: [] for name in names}
    rows, truth, all_pools = loaded
    cells = _cell_rows(rows, truth, all_pools)
    out = {"mc10_summary": _summary_rows(cells), "mc10_cells": cells, "mc10_arms": _arm_rows(cells),
           "mc10_paired": _paired_rows(cells),
           "mc10_missingness": [{"list": MAIN, **row}
                                for row in J.missingness(f"{INSTRUMENT}_{MAIN}", rows, arm_of)],
           "mc10_shuffled": _shuffled_rows(rows, truth)}
    _write_csv(root, SUMMARY_CSV, SUMMARY_COLUMNS, out["mc10_summary"])
    _write_csv(root, CELLS_CSV, CELL_COLUMNS, out["mc10_cells"])
    _write_csv(root, ARMS_CSV, ARM_COLUMNS, out["mc10_arms"])
    _write_csv(root, PAIRED_CSV, PAIRED_COLUMNS, out["mc10_paired"])
    _write_csv(root, MISSINGNESS_CSV, ("list", *J.MISSINGNESS_COLUMNS), out["mc10_missingness"])
    _write_csv(root, SHUFFLED_CSV, SHUFFLED_COLUMNS, out["mc10_shuffled"])
    return out


ARM_LABELS = {"maemm": "MAEMM", "nla_native": "NLA verbalizer", "retrieval": "Corpus search",
              "jlens": "Jacobian lens", "base_l1": "Untrained base",
              "shuffled": "Shuffled (MAEMM, another behaviour)", "heldout_matching": "Held-out statements",
              **{arm: f"Steered model, s={arm.partition('@')[2]}" for arm in STEERED_ARMS}}
KIND_LABELS = {"bipo:persona": "learned (persona)", "heldout:persona": "held-out (persona)"}


def _fmt(value, digits=3):
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if number != number else f"{number:.{digits}f}"


def _md(headers, rows):
    escape = lambda c: ("" if c is None else str(c)).replace("|", "\\|")
    out = ["| " + " | ".join(escape(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(escape(c) for c in r) + " |")
    return out


def _require(root, rel, what):
    rows = _read_csv(root, rel)
    if not rows:
        raise FileNotFoundError(f"{rel} is missing; {what}")
    return rows


def section(root):
    """The report's markdown for this instrument, from the saved CSVs."""
    lines = ["## Ten-way identification from the rollouts", "",
             "A judge reads eight texts of one family and ten candidate descriptions and returns the number "
             f"of the one they express, on the pinned instrument of `evals.downstream.steering_vector_inversion`. Chance "
             f"is {CHANCE:.0%}. The ten are {CANDIDATE_BEHAVIOURS} behaviours with both poles of each, the "
             "target's among them, seeded from the behaviour, so every reader of a vector answers the "
             f"identical question. `{REFERENCE_JUDGE}` is the judge.", ""]
    return "\n".join(lines + _summary_block(root) + _paired_block(root) + _shuffled_block(root)
                     + _missingness_block(root)).rstrip("\n") + "\n"


def _summary_block(root):
    lines = ["### 1. Identification rate by reader", "",
             "The mean over per-vector identification rates, with a percentile bootstrap interval over "
             "vectors (10,000 resamples, seed 0). A bundle with no usable answer counts as no "
             f"identification. Chance is {CHANCE:.0%}.", ""]
    rows = [r for r in _require(root, SUMMARY_CSV, "run the mc10 stages") if r.get("row_type") == "arm"]
    body = [(r.get("list"), KIND_LABELS.get(r.get("kind"), r.get("kind")),
             ARM_LABELS.get(r.get("arm_a"), r.get("arm_a")), r.get("judge"), r.get("n_vectors"),
             _fmt(r.get("value")),
             f"[{_fmt(r.get('ci_lower'))}, {_fmt(r.get('ci_upper'))}]" if r.get("ci_lower") else "n/a")
            for r in rows]
    return lines + _md(["List", "Vector", "Reader", "Judge", "n vectors", "Rate", "95% CI"], body) + [
        "", f"[Full table, with the paired rows]({SUMMARY_CSV})",
        f" · [Per-family cells]({CELLS_CSV}) · [Per-arm pools]({ARMS_CSV})", ""]


def _paired_block(root):
    lines = ["### 2. The readers against each other, vector by vector", "",
             "The mean difference between two readers' rates on the same vectors, with a paired bootstrap "
             f"interval and sign counts; fewer than {MIN_VECTORS_PAIRED} shared vectors give no row.", ""]
    rows = _read_csv(root, PAIRED_CSV)
    if rows is None:
        raise FileNotFoundError(f"{PAIRED_CSV} is missing; run the mc10 stages")
    if not rows:
        return lines + [f"No paired row: no kind holds {MIN_VECTORS_PAIRED} vectors in both arms of a pair "
                        "in this run.", "", f"[Full table]({PAIRED_CSV})", ""]
    body = [(r.get("list"),
             f"{ARM_LABELS.get(r.get('arm_a'), r.get('arm_a'))} - "
             f"{ARM_LABELS.get(r.get('arm_b'), r.get('arm_b'))}",
             KIND_LABELS.get(r.get("kind"), r.get("kind")), r.get("judge"), r.get("n_vectors"),
             _fmt(r.get("mean_diff")), f"[{_fmt(r.get('ci_lower'))}, {_fmt(r.get('ci_upper'))}]",
             f"{r.get('n_a_greater')}/{r.get('n_b_greater')}/{r.get('n_tied')}") for r in rows]
    return lines + _md(["List", "Readers", "Vector", "Judge", "n vectors", "Mean difference", "95% CI",
                        "A>B / B>A / tied"], body) + ["", f"[Full table]({PAIRED_CSV})", ""]


def _shuffled_block(root):
    rows = _read_csv(root, SHUFFLED_CSV)
    if not rows:
        return []
    lines = ["### 3. Does the judge read the text? The shuffled check", "",
             "Each vector's own question asked of MAEMM's bundles for another behaviour on its list (the "
             "donor). A judge that reads the text names the donor, so the rate against the vector's own "
             f"truth sits below the {CHANCE:.0%} line by construction.", ""]
    body = [(r.get("list"), KIND_LABELS.get(r.get("kind"), r.get("kind")), r.get("judge"), r.get("n_vectors"),
             r.get("n_bundles"), _fmt(r.get("target_rate")), _fmt(r.get("donor_rate")),
             _fmt(r.get("donor_behaviour_rate"))) for r in rows]
    return lines + _md(["List", "Vector", "Judge", "n vectors", "n bundles", "Own truth (should be low)",
                        "Donor's description", "Donor's behaviour, either pole"], body) + [
        "", f"[Full table]({SHUFFLED_CSV})", ""]


def _missingness_block(root):
    lines = ["### 4. What each judge did not answer", "",
             "Per judge and reader: the requests sent, the ones answered, and how the rest failed. An "
             "unanswered bundle counts as no identification.", ""]
    rows = [r for r in _require(root, MISSINGNESS_CSV, "run the mc10 stages") if r.get("arm") != POOLED]
    body = [(r.get("list"), r.get("judge"), ARM_LABELS.get(r.get("arm"), r.get("arm")),
             r.get("n_requests"), r.get("ok"), r.get("refused"), r.get("content_filter"),
             r.get("parse_fail"), r.get("truncated"), r.get("unavailable")) for r in rows]
    return lines + _md(["List", "Judge", "Reader", "n requests", "ok", "refused", "content filter",
                        "parse fail", "truncated", "unavailable"], body) + [
        "", f"[Full table, with the pooled rows]({MISSINGNESS_CSV})", ""]
