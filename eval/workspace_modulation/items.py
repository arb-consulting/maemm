"""Items: the paper's directed-modulation materials built into 46 x 5 items, the carrier cells, targets,
exclusions and donors (methodology, population)."""

import hashlib, json, os, re, time

import numpy as np

from eval.common.matcher import normalise, whole_word_hit
from eval.common.runs import mark_stage, stage_done
from eval.workspace_modulation import config as C
from eval.workspace_modulation.runs import stage_key, write_provenance

THINK_BLOCK = "<think>\n\n</think>\n\n"


def load_json(fname, sha):
    path = os.path.join(C.DATASETS_DIR, fname)
    with open(path, "rb") as h:
        raw = h.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {sha}")
    return json.loads(raw.decode("utf-8"))


def load_dm():
    return load_json(C.DM_FILE, C.DM_SHA256)


def fill(text, x):
    """The released phrasing with {x} filled and the first letter (not character) capitalised, so "({x})"
    reads as a sentence like the other clauses."""
    s = text.replace("{x}", x)
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def user_turn(carrier, phrasing_text, x):
    return C.FRAME.format(carrier=carrier, phrasing=fill(phrasing_text, x))


def baseline_turn(carrier):
    return C.BASELINE_FRAME.format(carrier=carrier)


def render_chat(tok, user, carrier):
    """User turn + teacher-forced assistant carrier, thinking disabled (Qwen3's template writes an empty think
    block before the carrier). Asserts the carrier sits where expected, one think block, and an open turn."""
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": carrier}]
    chat = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True, enable_thinking=False)
    if not chat.endswith(carrier) or chat.count(THINK_BLOCK) != 1 or chat.count("<|im_end|>") != 1:
        raise ValueError("unexpected chat rendering: " + repr(chat[-120:]))
    ids = [int(i) for i in tok.encode(chat, add_special_tokens=False)]
    strs = [tok.decode([i]) for i in ids]
    return chat, ids, strs


def _spans(strs):
    out, off = [], 0
    for s in strs:
        out.append((off, off + len(s)))
        off += len(s)
    return out


def carrier_positions(chat, strs, carrier):
    a = chat.rindex(carrier)
    b = a + len(carrier)
    pos = [k for k, (s, e) in enumerate(_spans(strs)) if s < b and e > a and k >= 1]
    if not pos or "".join(strs[p] for p in pos).strip() != carrier:
        raise ValueError("carrier tokens do not reproduce the carrier")
    if any("<|" in strs[p] or strs[p].strip() in ("<think>", "</think>") for p in pos):
        raise ValueError("carrier tokens contain a control token")
    return pos


# Irregular plurals of the released members; also read backwards by `already_plural`.
IRREGULAR_PLURALS = {
    "child": "children",
    "person": "people",
    "man": "men",
    "woman": "women",
    "foot": "feet",
    "tooth": "teeth",
    "goose": "geese",
    "mouse": "mice",
    "louse": "lice",
    "ox": "oxen",
    "die": "dice",
    "leaf": "leaves",
    "loaf": "loaves",
    "half": "halves",
    "calf": "calves",
    "knife": "knives",
    "life": "lives",
    "wife": "wives",
    "shelf": "shelves",
    "thief": "thieves",
    "wolf": "wolves",
    "cactus": "cacti",
    "fungus": "fungi",
    "nucleus": "nuclei",
    "analysis": "analyses",
    "crisis": "crises",
}
# Nouns whose plural is the word itself: neither pluralised nor read as a singular.
UNCHANGED_PLURALS = ("sheep", "deer", "fish", "series", "species", "aircraft", "means", "offspring")
# Singular endings in `s`.
SINGULAR_S_ENDINGS = ("ss", "us", "is")


def already_plural(w):
    """Whether a released category member is already plural: a word ending in `s` is plural unless its ending
    is one a singular also has."""
    lw = w.lower()
    if lw in IRREGULAR_PLURALS:
        return False
    if lw in UNCHANGED_PLURALS or lw in set(IRREGULAR_PLURALS.values()):
        return True
    return lw.endswith("s") and not lw.endswith(SINGULAR_S_ENDINGS)


def _plural(w):
    """The plural of one member, or the member unchanged when it already is one. Capitalisation is carried
    over from the input, because the lens asks for single-token forms and the tokenizer is case-sensitive."""
    lw = w.lower()
    if lw in IRREGULAR_PLURALS:
        out = IRREGULAR_PLURALS[lw]
        return out.capitalize() if w[:1].isupper() else out
    if already_plural(w):
        return w
    if lw.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if len(lw) > 1 and lw.endswith("y") and lw[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def topic_forms(members):
    out = []
    for m in members:
        for f in (m, _plural(m)):
            if f not in out:
                out.append(f)
    return out


def arith_forms(answer):
    a = str(answer)
    return [a] + ([C.NUMBER_WORDS[a]] if a in C.NUMBER_WORDS else [])


def naming_targets(family, display, forms):
    """The target forms the naming judge is asked about: for a topic the category name and the member forms
    the word rule reads; for arithmetic the answer as digit and word (never the expression)."""
    return ([display] if family == "topics" else []) + [f for f in forms if f != display]


def whole_word_in(form, text):
    return whole_word_hit(text, [form])


def _numbers(expr):
    return set(re.findall(r"\d+", expr))


def target_in_prompt(family, concept):
    """A member that is also a word of the category name, or an answer that is a number of the expression,
    would make every prompt of the concept carry its own target."""
    if family == "arithmetic":
        return str(concept["answer"]) in _numbers(concept["expr"])
    words = {normalise(w) for w in re.findall(r"[A-Za-z]+", concept["name"])}
    return any(normalise(f) in words for f in topic_forms(concept["members"]))


def admissible(carrier, forms, tok):
    if any(whole_word_in(f, carrier) for f in forms):
        return "target_in_carrier"
    if len(tok.encode(carrier, add_special_tokens=False)) < C.MIN_CARRIER_TOKENS:
        return "carrier_too_short"
    return None


def assign_carriers(n_concepts, carriers, forms_by_k, tok, seed):
    """(chosen carrier, the one it replaced, why) per concept: concept k gets perm[k mod 20], an inadmissible
    carrier (`target_in_carrier`, `carrier_too_short`) is replaced by the next one; None when none is admissible."""
    perm = [int(p) for p in np.random.default_rng(seed).permutation(len(carriers))]
    out = []
    for k in range(n_concepts):
        start = k % len(carriers)
        first = perm[start]
        chosen = None
        for step in range(len(carriers)):
            ci = perm[(start + step) % len(carriers)]
            if admissible(carriers[ci], forms_by_k[k], tok) is None:
                chosen = ci
                break
        reassigned = None if chosen == first else first
        out.append((chosen, reassigned, None if reassigned is None else admissible(carriers[first], forms_by_k[k], tok)))
    return out


def pick_phrasings(dm, n_concepts, seed):
    """One phrasing per concept and instruction group, drawn with one generator in file order of the groups."""
    rng = np.random.default_rng(seed)
    by_group = {}
    for p in dm["phrasings"]:
        by_group.setdefault(p["group"], []).append(p)
    out = []
    for _ in range(n_concepts):
        out.append({g: by_group[g][int(rng.integers(len(by_group[g])))] for g in C.GROUP_TO_CONDITION})
    return out


def draw_donors(concepts, seed):
    """Twenty same-family concepts per concept whose forms are disjoint from its own, drawn without
    replacement where the pool allows and with replacement otherwise; an empty pool raises. The donors are the
    chance line's and the foil pool (`judge.foil_of`)."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, c in concepts.items():
        own = {normalise(f) for f in c["forms"]}
        pool = sorted(
            k
            for k, v in concepts.items()
            if v["family"] == c["family"] and k != key and not (own & {normalise(f) for f in v["forms"]})
        )
        if not pool:
            raise RuntimeError(f"{key}: no same-family donor concept whose forms are disjoint from {sorted(own)}")
        draw = (
            rng.permutation(len(pool))[: C.N_DONORS]
            if len(pool) >= C.N_DONORS
            else rng.integers(len(pool), size=C.N_DONORS)
        )
        out[key] = [pool[int(j)] for j in draw]
    return out


def _smoke_indices(families):
    """--smoke keeps the first SMOKE_CONCEPTS_PER_FAMILY concepts of each family; every draw is still made
    over the whole population, so a smoke item is the item a full run has."""
    seen, out = {}, []
    for k, fam in enumerate(families):
        if seen.get(fam, 0) < C.SMOKE_CONCEPTS_PER_FAMILY:
            seen[fam] = seen.get(fam, 0) + 1
            out.append(k)
    return out


def build_items(tok, smoke=False):
    """Every item of the population and the concept registry; the construction seeds are config's, not --seed."""
    dm = load_dm()
    population = [("topics", t) for t in dm["topic_categories"]] + [("arithmetic", m) for m in dm["math_problems"]]
    forms_by_k = {
        k: (topic_forms(c["members"]) if fam == "topics" else arith_forms(c["answer"]))
        for k, (fam, c) in enumerate(population)
    }
    carriers = list(dm["carrier_sentences"])
    assign = assign_carriers(len(population), carriers, forms_by_k, tok, C.CARRIER_SEED)
    phr = pick_phrasings(dm, len(population), C.PHRASING_SEED)
    concepts = {}
    for k, (fam, c) in enumerate(population):
        display = c["name"] if fam == "topics" else c["expr"]
        forms = forms_by_k[k]
        lens_forms = [
            f
            for f in forms
            if len(tok.encode(" " + f, add_special_tokens=False)) == 1
            or len(tok.encode(f, add_special_tokens=False)) == 1
        ]
        target_display = f"{c['name']} ({', '.join(c['members'])})" if fam == "topics" else " / ".join(forms)
        concepts[str(k)] = {
            "family": fam,
            "forms": forms,
            "lens_forms": lens_forms,
            # `display` is the concept as the instruction names it; `targets` what the judge is asked about
            "display": display,
            "targets": naming_targets(fam, display, forms),
            "target_display": target_display,
        }
    donors = draw_donors(concepts, C.DONOR_SEED)
    selected = _smoke_indices([fam for fam, _ in population]) if smoke else range(len(population))
    items = []
    for k in selected:
        fam, c = population[k]
        key = str(k)
        ci, reassigned, reassigned_why = assign[k]
        excluded_reason = None
        if target_in_prompt(fam, c):
            excluded_reason = "target_in_prompt"
        elif ci is None:
            excluded_reason = "no_admissible_carrier"
        x = c["name"] if fam == "topics" else f"evaluating {c['expr']}"
        for cond in C.CONDITIONS:
            group = next((g for g, cc in C.GROUP_TO_CONDITION.items() if cc == cond), None)
            phrasing = phr[k][group] if group else None
            carrier = carriers[ci if ci is not None else k % len(carriers)]
            user = user_turn(carrier, phrasing["text"], x) if phrasing else baseline_turn(carrier)
            chat, ids, strs = render_chat(tok, user, carrier)
            cpos = carrier_positions(chat, strs, carrier)
            items.append(
                {
                    "i": None,
                    "family": fam,
                    "concept_key": key,
                    "concept": concepts[key]["display"],
                    "instruction": cond,
                    "phrasing": phrasing["name"] if phrasing else None,
                    "carrier": carrier,
                    "carrier_index": ci,
                    "carrier_reassigned_from": reassigned,
                    "carrier_reassigned_reason": reassigned_why,
                    "user": user,
                    "chat": chat,
                    "ids": ids,
                    "token_strs": strs,
                    "carrier_cells": cpos,
                    "forms": concepts[key]["forms"],
                    "lens_forms": concepts[key]["lens_forms"],
                    "single_token": bool(concepts[key]["lens_forms"]),
                    "target_display": concepts[key]["target_display"],
                    "targets": concepts[key]["targets"],
                    "response_type": "pair",
                    "donors": donors[key],
                    "excluded": excluded_reason is not None,
                    "exclusion_reason": excluded_reason,
                }
            )
    for i, it in enumerate(items):
        it["i"] = i
    kept = [x for x in items if not x["excluded"]]
    counts = {}
    for x in kept:
        counts[x["family"]] = counts.get(x["family"], 0) + 1
    excl, excluded_concepts, seen = {}, [], set()
    for x in items:
        if x["excluded"]:
            excl[x["exclusion_reason"]] = excl.get(x["exclusion_reason"], 0) + 1
            if x["concept_key"] not in seen:
                seen.add(x["concept_key"])
                excluded_concepts.append((x["concept_key"], x["concept"], x["exclusion_reason"]))
    cfg = {
        "seeds": {
            "carrier": C.CARRIER_SEED,
            "phrasing": C.PHRASING_SEED,
            "donor": C.DONOR_SEED,
        },
        "smoke": bool(smoke),
        "counts": counts,
        "n_excluded": sum(excl.values()),
        "exclusions": excl,
        "excluded_concepts": excluded_concepts,
        "reassigned": [
            {
                "concept_key": x["concept_key"],
                "concept": x["concept"],
                "from": x["carrier_reassigned_from"],
                "to": x["carrier_index"],
                "reason": x["carrier_reassigned_reason"],
            }
            for x in items
            if x["carrier_reassigned_from"] is not None and x["instruction"] == "focus"
        ],
        "n_cells": {C.CARRIER_BAND: sum(len(x["carrier_cells"]) for x in kept)},
    }
    return {"config": cfg, "concepts": concepts, "items": items}


def cell_table(items):
    """Every carrier cell of every kept item, (i, pos, band), sorted: the row order of the activation store."""
    return sorted((it["i"], p, C.CARRIER_BAND) for it in items for p in it["carrier_cells"])


def load_tokenizer():
    """The pinned tokenizer alone, with load_base's pad-token fallback."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stage_prepare(args, run):
    chash = stage_key("prepare", args, run)
    if stage_done(run, "prepare", chash) and not args.force:
        print("[prepare] up to date")
        return
    started = time.time()
    tok = load_tokenizer()
    doc = build_items(tok, smoke=args.smoke)
    run.write_json("data/items.json", doc)
    write_provenance(run, {"tokenizer": C.MODEL})
    print(
        f"[prepare] {doc['config']['counts']}; excluded {doc['config']['exclusions']}; "
        f"reassigned {len(doc['config']['reassigned'])}; cells {doc['config']['n_cells']}",
        flush=True,
    )
    for key, display, reason in doc["config"]["excluded_concepts"]:
        print(f"[prepare] excluded {key} ({display}): {reason}", flush=True)
    mark_stage(run, "prepare", chash, {"counts": doc["config"]["counts"]}, started=started)
