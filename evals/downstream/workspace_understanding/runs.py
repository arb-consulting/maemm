"""This package's resume keys and provenance pins, on top of evals/downstream/common/runs.py.

`stage_config` is a stage's own resolved inputs; `stage_key` chains them to the records of every stage it
reads (`stages.DEPENDS`), so a re-run upstream re-keys exactly what reads it."""

import re

from evals.downstream.workspace_understanding import config as C
from evals.downstream.workspace_understanding.stages import DEPENDS, STAGES
from evals.downstream.common.nla import nla_reader as N
from evals.downstream.common.runs import broken_links, chained_hash, stage_record_names
from evals.downstream.common.runs import write_provenance as _common_write_provenance


def centring_mean_digest():
    """sha256 of the shipped centring mean (`evals.downstream.common.background.CENTRING_MEAN`), in the key of every
    stage that writes a centred direction into a model."""
    import hashlib

    from evals.downstream.common import background

    with open(background.CENTRING_MEAN, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stage_config(stage, args):
    """The resolved inputs a stage's output depends on; a change invalidates the saved output. A shard split
    is not part of it."""
    common = {"scoring_version": C.SCORING_VERSION, "seed": args.seed, "smoke": bool(args.smoke)}
    per = {
        "prepare": {"donor_seed": C.DONOR_SEED, "n_donors": C.N_DONORS, "model_revision": C.MODEL_REVISION},
        "corpus": {
            "corpus": C.CORPUS.record(),
            "n_docs": len(C.corpus_docs(args.smoke)),
            "model_revision": C.MODEL_REVISION,
        },
        "retrieval": {
            "corpus": C.CORPUS.record(),
            "n_docs": len(C.corpus_docs(args.smoke)),
            "read_layer": C.READ_LAYER,
            "metric": C.CORPUS_METRIC,
            "shared_size": C.CORPUS_SHARED_SIZE,
            "model_revision": C.MODEL_REVISION,
        },
        "capture": {
            "read_layer": C.READ_LAYER,
            "n_layers": C.N_LAYERS,
            "diag_max_new": C.DIAG_MAX_NEW,
            "model_revision": C.MODEL_REVISION,
        },
        "lens": {
            "lens": [C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE],
            "top_word": C.TOP_WORD,
            "model_revision": C.MODEL_REVISION,
        },
        "rollouts": {
            "inverter": [C.INVERTER, C.INVERTER_REVISION],
            "gen": [C.N_SAMPLES, C.TEMP, C.MAX_NEW, C.MIN_NEW, C.GEN_CHUNK],
            "seed_offset": C.ARM_SEED_OFFSET["maem"],
            "model_revision": C.MODEL_REVISION,
        },
        "patchscope": {
            "layers": C.PATCH_LAYERS,
            "floor": C.PATCH_FLOOR,
            "prompt": [C.PATCH_PROMPT_ID, C.PATCH_PROMPT],
            "rules": C.PATCH_RULES,
            "input": C.PATCH_INPUT,
            "alpha": C.PATCH_ALPHA,
            "mu_sha256": centring_mean_digest(),
            "check": [C.PATCH_CHECK_ROWS, C.PATCH_CHECK_MIN_COS, C.PATCH_CHECK_MIN_REL_DELTA],
            "gen": [C.N_SAMPLES, C.TEMP, C.MAX_NEW, C.MIN_NEW, C.PATCH_GEN_ROWS],
            "seed_offsets": {a: C.ARM_SEED_OFFSET[a] for a in C.PATCH_ARMS},
            "seeding": C.PATCH_SEEDING,
            "model_revision": C.MODEL_REVISION,
        },
        "summarise": {
            "model": C.JUDGES[C.SUMMARISER].model,
            "max_tokens": C.JUDGES[C.SUMMARISER].max_tokens_for("summary_req"),
            "prompt": C.SUMMARY_USER,
            "band": [C.LENS_BAND, list(C.LENS_BAND_LAYERS)],
        },
        "judge": {
            "judges": {
                name: [spec.model, spec.provider, spec.sampling, spec.max_tokens] for name, spec in C.JUDGES.items()
            },
            "prompts": [C.JUDGE_SYSTEM],
            "conditions": list(C.JUDGED_CONDITIONS),
            # no budget cap: a limit on spend, not an input the verdicts depend on
        },
        "nla": {
            "nla": N.pins_record(),
            "sidecar": dict(N.PINS.sidecar_sha256),
            "gen": [C.N_SAMPLES, C.TEMP, C.MIN_P, N.PINS.min_new, N.PINS.trunc, N.PINS.gen_chunk],
            "tags": [N.PINS.open_tag, N.PINS.close_tag],
            "seed_offset": C.ARM_SEED_OFFSET["nla"],
            "model_revision": C.MODEL_REVISION,
        },
        "reread": {
            "conditions": list(C.REREAD_CONDITIONS),
            "windows": {c: C.REREAD_WINDOW.get(c, C.REREAD_WINDOW_TOKENS) for c in C.REREAD_CONDITIONS},
            "norm_filter": C.REREAD_NORM_FILTER,
            "model_revision": C.MODEL_REVISION,
            "tol": C.REREAD_SELFCHECK_TOL,
        },
        "nla_control": {
            "nla": N.pins_record(),
            "sidecar": dict(N.PINS.sidecar_sha256),
            "kinds": list(C.POSITION_CONTROLS),
            "gen": [C.N_SAMPLES, C.TEMP, C.MIN_P, N.PINS.min_new, N.PINS.trunc, N.PINS.gen_chunk],
            "tags": [N.PINS.open_tag, N.PINS.close_tag],
            "seed_offsets": {k: C.ARM_SEED_OFFSET[f"nla_{k}"] for k in C.POSITION_CONTROLS},
            "read_layer": C.READ_LAYER,
            "model_revision": C.MODEL_REVISION,
        },
        # the inverter pin: the row is the ablation of that model, though it is never loaded here
        "untrained_base": {
            "inverter": [C.INVERTER, C.INVERTER_REVISION],
            "gen": [C.N_SAMPLES, C.TEMP, C.MAX_NEW, C.MIN_NEW, C.GEN_CHUNK],
            "seed_offset": C.ARM_SEED_OFFSET["untrained_base"],
            "model_revision": C.MODEL_REVISION,
        },
        "maem_control": {
            "inverter": [C.INVERTER, C.INVERTER_REVISION],
            "kinds": list(C.POSITION_CONTROLS),
            "gen": [C.N_SAMPLES, C.TEMP, C.MAX_NEW, C.MIN_NEW, C.GEN_CHUNK],
            "seed_offsets": {k: C.ARM_SEED_OFFSET[f"maem_{k}"] for k in C.POSITION_CONTROLS},
            "model_revision": C.MODEL_REVISION,
        },
        "report": {"n_boot": C.N_BOOT, "bootstrap_seed": C.BOOTSTRAP_SEED},
    }
    return {**common, **per[stage]}


# --- the chained resume key -------------------------------------------------------------------------------

#: The pseudo-stage a corpus part is keyed as: `retrieval`'s settings chained to what the search reads. The
#: merge's own record also chains to every part it ranked.
RETRIEVAL_PART = "retrieval_part"
PART_RECORD = "retrieval_part{k}of{n}"


def records_of(run, stage):
    """The completed records that stand for `stage`: its own and every shard's (the loaders merge every
    shard file they find)."""
    shard = re.compile(rf"{re.escape(stage)}_shard\d+of\d+")
    return stage_record_names(run, lambda n: n == stage or shard.fullmatch(n) is not None)


def base_stage(record):
    """The stage a record name belongs to: the name itself, less a shard or a part suffix."""
    m = re.fullmatch(r"(.+?)_(?:shard|part)\d+of\d+", record)
    return m.group(1) if m else record


def _merged_parts(run):
    """How many parts the saved `retrieval` record was ranked from (1 when there is none)."""
    rel = "stages/retrieval.json"
    return int((run.read_json(rel) if run.exists(rel) else {}).get("n_shards") or 1)


def stage_key(stage, args, run, n_parts=None):
    """`stage_config` chained to the records of `stages.DEPENDS[stage]`. Refuses when a dependency has no
    record or its own chain is broken, unless --force. `n_parts` is the split the retrieval merge chains to."""
    part = stage == RETRIEVAL_PART
    own = "retrieval" if part else stage
    names, missing = [], []
    for up in DEPENDS[own]:
        found = records_of(run, up)
        names += found or [up]
        if not found:
            missing.append(up)
    force = bool(getattr(args, "force", False))
    if missing and not force:
        raise RuntimeError(f"{own}: " + "; ".join(f"stage {up} has not completed in this run directory"
                                                  for up in missing) + " -- run what it reads first")
    broken = [b for name in names for b in broken_links(run, name)]
    if broken and not force:
        raise RuntimeError(f"{own}: the records it reads no longer chain to their own upstream -- "
                           f"{'; '.join(dict.fromkeys(broken))}. Run the stage(s) named first again "
                           f"(`all` re-runs exactly what is stale)")
    if own == "retrieval" and not part:
        n = int(n_parts or _merged_parts(run))
        names += [PART_RECORD.format(k=k, n=n) for k in range(n)]
    return chained_hash(stage_config(own, args), run, names)


def stale_stages(run):
    """The stages whose records no longer chain to the directory as it stands, in stage order."""
    every = stage_record_names(run, lambda n: base_stage(n) in STAGES)
    stale = {base_stage(n) for n in every if broken_links(run, n)}
    return [s for s in STAGES if s in stale]


def write_provenance(run, extra=None, stage=None):
    """This package's pins on top of the common git/python/platform fields (evals/downstream/common/runs.py)."""
    pins = {
        "model": C.MODEL,
        "model_revision": C.MODEL_REVISION,
        "inverter": [C.INVERTER, C.INVERTER_REVISION],
        "lens": [C.LENS_REPO, C.LENS_REVISION, C.LENS_FILE, C.LENS_SHA256],
        "datasets": C.DATASET_SHA256,
        "jlens_commit": C.JLENS_COMMIT,
        "judge_profile": C.JUDGE_PROFILE,
    }
    _common_write_provenance(run, extra, stage, common_fields=pins)
