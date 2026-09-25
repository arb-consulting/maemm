"""Items: the two vendored J-lens sets, readout positions, target forms, exclusions, donors and foils (methodology §2)."""

import hashlib, json, os, time
import numpy as np
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common import matcher as M
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def load_set(family, root=DATASETS_DIR):
    path = os.path.join(root, C.DATASET_FILES[family])
    with open(path, "rb") as h:
        raw = h.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != C.DATASET_SHA256[family]:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {C.DATASET_SHA256[family]}")
    rows = json.loads(raw.decode("utf-8"))["items"]
    if len(rows) != C.EXPECTED_COUNTS[family]:
        raise RuntimeError(f"{family}: {len(rows)} items, pinned {C.EXPECTED_COUNTS[family]}")
    return rows


def prepare_prompt(prompt):
    return prompt.rstrip()


def readout_position(prompt, family, tok, name="?"):
    text = prepare_prompt(prompt)
    ids = [int(i) for i in tok.encode(text, add_special_tokens=False)]
    if len(ids) < 2:
        raise ValueError(f"{family}/{name}: readout would be position 0")
    pos = len(ids) - 1
    token = tok.decode([ids[pos]])
    if family == "association" and not token.endswith("."):
        raise ValueError(f"{family}/{name}: readout token {token!r} is not the final period")
    if family == "multihop" and not token.strip():
        raise ValueError(f"{family}/{name}: readout token {token!r} is whitespace")
    return ids, pos, token


def forms_of(item):
    out = []
    for f in item["intermediates"]:
        if f not in out:
            out.append(f)
        if f.isdigit() and C.NUMBER_WORDS.get(f) and C.NUMBER_WORDS[f] not in out:
            out.append(C.NUMBER_WORDS[f])
    return out


def surface_variants(form):
    seen, out = set(), []
    for v in (form, " " + form, form.capitalize(), " " + form.capitalize(), form.upper(), " " + form.upper()):
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def form_token_ids(forms, tok):
    out = {}
    for f in forms:
        ids = []
        for v in surface_variants(f):
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1 and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        out[f] = {"ids": ids, "single_token": bool(ids)}
    return out


def visible_in_prompt(forms, prompt):
    return M.whole_word_hit(prompt, forms)


def answer_forms(target):
    out = [target]
    if target.isdigit() and C.NUMBER_WORDS.get(target):
        out.append(C.NUMBER_WORDS[target])
    return out


def subgroup(item):
    if item["family"] != "association":
        return None
    return "proper_noun" if item["intermediates"][0][:1].isupper() else "common"


def _disjoint(a, b):
    A = {x.casefold() for x in a}
    B = {x.casefold() for x in b}
    return not (A & B)


def draw_donors(built, seed):
    rng = np.random.default_rng(seed)
    for it in built:
        pool = [
            o["i"]
            for o in built
            if o["family"] == it["family"] and o["i"] != it["i"] and _disjoint(o["forms"], it["forms"])
        ]
        if len(pool) >= C.N_DONORS:
            it["donors"] = [int(x) for x in rng.choice(pool, C.N_DONORS, replace=False)]
            it["donors_with_replacement"] = False
        else:
            it["donors"] = [int(x) for x in rng.choice(pool, C.N_DONORS, replace=True)]
            it["donors_with_replacement"] = True
        it["foil"] = it["donors"][0]


def build_items(tok, seed, smoke=False):
    built, excluded = [], []
    for fam in C.FAMILIES:
        for row in load_set(fam):
            forms = forms_of(row)
            prompt = prepare_prompt(row["prompt"])
            rec = {
                "family": fam,
                "name": row["name"],
                "prompt": prompt,
                "prompt_raw": row["prompt"],
                "forms": forms,
                "answer": row.get("target"),
                "answer_forms": answer_forms(row["target"]) if fam == "multihop" else None,
                "subgroup": subgroup({"family": fam, "intermediates": row["intermediates"]}),
                "excluded": False,
                "exclusion_reason": None,
            }
            if visible_in_prompt(forms, prompt):
                rec.update(excluded=True, exclusion_reason="target visible in prompt as a whole word")
            elif (
                fam == "multihop"
                and any(M.stem_overlap(f, row["target"]) for f in forms)
                and not row["name"].startswith("firstletter-")
            ):
                rec.update(excluded=True, exclusion_reason="bridge and answer overlap under the stem rule")
            if rec["excluded"]:
                # the same keys as a kept record, None where never computed
                rec.update(
                    ids=None,
                    readout_pos=None,
                    readout_token=None,
                    form_ids=None,
                    multi_token=None,
                    i=None,
                    donors=None,
                    foil=None,
                    donors_with_replacement=None,
                )
                excluded.append(rec)
                continue
            ids, pos, token = readout_position(prompt, fam, tok, row["name"])
            fid = form_token_ids(forms, tok)
            rec.update(
                ids=ids,
                readout_pos=pos,
                readout_token=token,
                form_ids=fid,
                multi_token=not any(v["single_token"] for v in fid.values()),
            )
            built.append(rec)
    if smoke:
        keep = []
        for fam in C.FAMILIES:
            keep += [b for b in built if b["family"] == fam][:3]
        built = keep
    for k, b in enumerate(built):
        b["i"] = k
    draw_donors(built, C.DONOR_SEED)
    return {
        "config": {
            "seed": seed,
            "smoke": smoke,
            "donor_seed": C.DONOR_SEED,
            "n_donors": C.N_DONORS,
            "scoring_version": C.SCORING_VERSION,
            "n_kept": len(built),
            "n_excluded": len(excluded),
            "counts": {f: sum(1 for b in built if b["family"] == f) for f in C.FAMILIES},
        },
        "items": built + excluded,
    }


def stage_prepare(args, run):
    chash = stage_key("prepare", args, run)
    if stage_done(run, "prepare", chash) and not args.force:
        print("[prepare] up to date")
        return
    started = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_REVISION)
    doc = build_items(tok, args.seed, smoke=args.smoke)
    run.write_json("data/items.json", doc)
    write_provenance(run, {"tokenizer": C.MODEL})
    kept = [x for x in doc["items"] if not x["excluded"]]
    print(
        f"[prepare] kept {len(kept)} items {doc['config']['counts']}; excluded {doc['config']['n_excluded']}; multi-token lens targets "
        f"{sum(1 for x in kept if x['multi_token'])}",
        flush=True,
    )
    mark_stage(run, "prepare", chash, {"n_kept": len(kept)}, started=started)
