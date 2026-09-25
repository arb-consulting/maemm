"""Each stage's resolved config (`stage_config`), the chained key it resumes on (`stage_key`), the judge
budget and the provenance pins, on top of evals.downstream.common.runs.

A stage's key is its own settings chained to the records of every stage it reads (`stages.DEPENDS`), so a
re-run upstream invalidates exactly what reads it."""

import re

from evals.downstream.common.nla import nla_reader as N
from evals.downstream.common.runs import broken_links, chained_hash, stage_record_names
from evals.downstream.common.runs import write_provenance as _common_write_provenance
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation.stages import DEPENDS, SHARDED, STAGES


def resolve_judge_budget(args):
    """An explicit --judge-budget-usd, else the package's own cap."""
    given = getattr(args, "judge_budget_usd", None)
    return float(given) if given is not None else C.JUDGE_BUDGET_USD


def centring_mean_digest():
    """sha256 of the shipped centring mean (`evals.downstream.common.background.CENTRING_MEAN`)."""
    import hashlib

    from evals.downstream.common import background

    with open(background.CENTRING_MEAN, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stage_config(stage, args):
    """The resolved inputs a stage's output depends on; changing any of them invalidates it. The judge cap is
    in no key: it limits spend and changes no verdict."""
    common = {"scoring_version": C.SCORING_VERSION, "seed": args.seed, "smoke": bool(args.smoke)}
    arms = getattr(args, "arms", ",".join(C.ARM_ORDER))
    unknown = [a for a in arms.split(",") if a and a not in C.ARMS]
    if unknown:
        raise RuntimeError(f"unknown generation arm(s) {unknown}; this package has {list(C.ARM_ORDER)}")
    shard = {"shard": getattr(args, "shard", 0), "n_shards": getattr(args, "n_shards", 1)}
    model = {"model_revision": C.MODEL_REVISION}
    nla = N.pins_record(N.PINS)
    bands = {"read_bands": list(C.READ_BANDS), "mean_pos": C.MEAN_POS}
    judges = {name: [s.model, s.provider, s.sampling] for name, s in C.JUDGES.items()}
    # `corpus`'s whole config is contained in `retrieval`'s, so a rebuilt corpus invalidates every part.
    corpus = {"spec": C.CORPUS, "smoke_corpus_docs": C.SMOKE_CORPUS_DOCS, "top_k": C.TOP_WINDOWS, "metric": "raw"}
    gen = {a: C.ARMS[a] for a in arms.split(",") if a}
    per = {
        "prepare": {
            "seeds": [C.CARRIER_SEED, C.PHRASING_SEED, C.DONOR_SEED],
            "n_donors": C.N_DONORS,
            "datasets": [C.DM_SHA256],
            **model,
        },
        "capture": {"read_layer": C.READ_LAYER, "n_layers": C.N_LAYERS, "compliance_max_new": C.COMPLIANCE_MAX_NEW,
                    **model},
        "cells_mean": {
            "read_layer": C.READ_LAYER,
            "over_band": C.CARRIER_BAND,
            "mean_pos": C.MEAN_POS,
            "norm_ratio_range": list(C.MEAN_NORM_RATIO_RANGE),
        },
        "rollouts": {
            "inverter": [C.INVERTER, C.INVERTER_REVISION],
            "arms": arms,
            "gen": gen,
            "gen_chunk": C.GEN_CHUNK,
            "reread": [C.REREAD_MAX_LENGTH, C.REREAD_NORM_FILTER],
            "injection_check": C.INJECTION_CHECK,
            "mean_arms": list(C.MEAN_ARMS),
            "mean_control_from_band": C.MEAN_CONTROL_FROM_BAND,
            **shard,
            **bands,
            **model,
        },
        "rollouts_merge": {
            "inverter": [C.INVERTER, C.INVERTER_REVISION],
            "arms": arms,
            "gen": gen,
            "mean_arms": list(C.MEAN_ARMS),
            "injection_check": C.INJECTION_CHECK,
            **bands,
            **model,
        },
        "corpus": {"corpus": corpus},
        "retrieval": {"corpus": corpus, "shared_size": C.CORPUS_SHARED_SIZE, **shard, **bands, **model},
        "retrieval_merge": {"corpus": corpus, "shared_size": C.CORPUS_SHARED_SIZE, **bands, **model},
        "patchscope": {
            "arms": list(C.PATCH_ARMS),
            "prompt": [C.PATCH_PROMPT_ID, C.PATCH_PROMPT],
            "layer": C.PATCH_LAYER,
            "rule": C.PATCH_RULE,
            "input": C.PATCH_INPUT[C.PATCH_RULE],
            "alpha": C.PATCH_ALPHA,
            "mu_sha256": centring_mean_digest(),
            "check": [C.PATCH_CHECK_ROWS, C.PATCH_CHECK_MIN_COS, C.PATCH_CHECK_MIN_REL_DELTA],
            "gen": dict(C.PATCH_GEN),
            "gen_rows": C.PATCH_GEN_ROWS,
            "seed_offsets": dict(C.PATCH_SEED_OFFSET),
            "seeding": C.PATCH_SEEDING,
            "floor_from_band": C.PATCH_FLOOR_FROM_BAND,
            **bands,
            **model,
        },
        "lens": {"lens": [C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE], "top_word": C.TOP_WORD, **bands, **model},
        "nla": {"nla": nla, "gen": C.ARMS["nla"], "gen_chunk": N.PINS.gen_chunk, **shard, **bands, **model},
        "nla_merge": {"nla": nla, **bands},
        # both lens proses: layer 42 at both read positions, the eight-layer pool at the final period
        "summarise": {
            "model": C.JUDGES[C.SUMMARISER].model,
            "max_tokens": C.MAX_TOKENS["summary_req"],
            "prompt": [C.PROMPT_SHA256["summariser_system"], C.PROMPT_SHA256["summariser_user"]],
            "lens_band": [C.LENS_BAND, list(C.LENS_BAND_LAYERS)],
            **bands,
        },
        "judge": {
            "judges": judges,
            "max_tokens": C.MAX_TOKENS,
            "prompts": C.PROMPT_SHA256,
            "instrument": {
                "readout_kind": dict(C.READOUT_KIND),
                "vs": list(C.VS),
                "foil_rule": C.FOIL_RULE,
                "donor_seed": C.DONOR_SEED,
            },
            "readers": list(C.JUDGED_READERS),
            "reader_bands": {r: list(b) for r, b in C.READER_BANDS.items()},
            "nla": nla,
            "corpus": corpus,
            "mean_control_from_band": C.MEAN_CONTROL_FROM_BAND,
            **bands,
        },
        "report": {"n_boot": C.N_BOOT, "bootstrap_seed": C.BOOTSTRAP_SEED, "parse_version": C.PARSE_VERSION},
    }
    return {**common, **per[stage]}


_SHARD_SUFFIX = re.compile(r"\.shard(\d+)of(\d+)$")


def base_stage(record):
    """The stage a record belongs to (`rollouts.reg.shard0of2` -> `rollouts`)."""
    return record.split(".", 1)[0]


def records_of(run, stage, n_shards=1):
    """The completed records that stand for `stage`: for a sharded stage, those of the one `n_shards`-way
    split a merge folds."""

    def belongs(name):
        if base_stage(name) != stage:
            return False
        m = _SHARD_SUFFIX.search(name)
        return stage not in SHARDED or (int(m.group(2)) if m else 1) == int(n_shards)

    return stage_record_names(run, belongs)


def _merged_split(run, stage):
    """How many shards merge stage `stage`'s saved record folded, 1 when there is none."""
    rel = f"stages/{stage}.json"
    return int((run.read_json(rel) if run.exists(rel) else {}).get("n_shards") or 1)


def stage_key(stage, args, run, n_shards=None):
    """`stage_config` chained to the records of `stages.DEPENDS[stage]`.

    A missing or broken upstream record refuses (unless --force). `n_shards` is a merge's split; a shard
    record whose chain broke is left out while another record of the split stands."""
    names, missing, broken = [], [], []
    for up in DEPENDS[stage]:
        n = int(n_shards or _merged_split(run, stage)) if up in SHARDED else 1
        found = records_of(run, up, n)
        if not found:
            missing.append(up)
            names.append(up)
            continue
        links = {name: broken_links(run, name) for name in found}
        whole = [name for name in found if not links[name]]
        if up in SHARDED and whole:
            names += whole
        else:
            names += found
            broken += [b for name in found for b in links[name]]
    force = bool(getattr(args, "force", False))
    if missing and not force:
        raise RuntimeError(f"{stage}: " + "; ".join(f"stage {up} has not completed in this run directory"
                                                    for up in missing) + " -- run what it reads first")
    if broken and not force:
        raise RuntimeError(f"{stage}: the records it reads no longer chain to their own upstream -- "
                           f"{'; '.join(dict.fromkeys(broken))}. Run the stage(s) named first again "
                           f"(`all` re-runs exactly what is stale)")
    return chained_hash(stage_config(stage, args), run, names)


def stale_stages(run):
    """The stages whose records no longer chain to the directory as it stands, in stage order."""
    every = stage_record_names(run, lambda n: base_stage(n) in STAGES)
    stale = {base_stage(n) for n in every if broken_links(run, n)}
    return [s for s in STAGES if s in stale]


def write_provenance(run, extra=None, stage=None):
    pins = {
        "model": C.MODEL,
        "model_revision": C.MODEL_REVISION,
        "inverter": [C.INVERTER, C.INVERTER_REVISION],
        "lens": [C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE, C.LENS_SHA256],
        "jlens_commit": C.JLENS_COMMIT,
        "datasets": {C.DM_FILE: C.DM_SHA256},
        "prompts": C.PROMPT_SHA256,
        "judge_profile": C.JUDGE_PROFILE,
    }
    _common_write_provenance(run, extra, stage, common_fields=pins)
