"""The pinned Concept500 data: construction splits, held-out references and ten-candidate lists."""
from pathlib import Path
import hashlib
import time

import numpy as np

from .artifacts import digest, seed_for
from .config import GENRES


def load_tokenizer(config, cache_dir):
    """The pinned tokenizer; it must have an EOS token, the sink every text is read behind."""
    from evals.downstream.common.model_io import load_tokenizer as load

    tokenizer = load(config.model, config.model_revision, cache_dir=str(cache_dir), token=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Standalone scoring requires an EOS sink")
    return tokenizer


def check_generation_configs(config, cache_dir):
    """Refuse a base and inverter whose shipped generation configs differ (a stop id above all), read from
    the two repos' config files before any GPU container starts: the untrained-base control must differ
    from MAEM's arm in the weights alone."""
    from transformers import GenerationConfig

    from evals.downstream.common.model_io import refuse_unless_generation_agrees

    base, inverter = (GenerationConfig.from_pretrained(repo, revision=revision,
                                                       cache_dir=str(cache_dir), token=False)
                      for repo, revision in ((config.model, config.model_revision),
                                             (config.inverter, config.inverter_revision)))
    refuse_unless_generation_agrees(base, inverter)


def prepare_text(text, tokenizer, max_context):
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) < 8:
        return None, "shorter_than_eight_tokens"
    if len(ids) + 1 > max_context:
        return None, "exceeds_full_context"
    return {"text": text, "token_ids": ids, "text_id": digest(ids)}, None


def prepare_concepts(train, test, tokenizer, config, max_context=262144):
    """Per concept: 48 positive and 48 negative construction texts, and the held-out positive test texts
    (deduplicated and disjoint from construction) as references. Parquet row ids are kept."""
    texts, prepared, exclusions = {}, {"train": [], "test": []}, []
    for split, rows in (("train", train), ("test", test)):
        for index, row in enumerate(rows):
            source_id = f"{split}:{index}"
            text, error = prepare_text(row["output"], tokenizer, max_context)
            record = {"source_id": source_id, "concept_id": int(row["concept_id"]),
                      "genre": row["concept_genre"], "category": row["category"],
                      "description": row["output_concept"], "instruction": row["input"],
                      "response": row["output"], "text_id": None if text is None else text["text_id"]}
            prepared[split].append(record)
            if error:
                exclusions.append({"source_id": source_id, "scope": "standalone_text", "reason": error})
            else:
                texts.setdefault(text["text_id"], text)
    concepts = []
    ids = sorted({row["concept_id"] for row in prepared["train"] if row["category"] == "positive"})
    for concept_id in ids:
        positive = [r for r in prepared["train"] if r["concept_id"] == concept_id and r["category"] == "positive"]
        genres = {r["genre"] for r in positive}
        descriptions = {r["description"] for r in positive}
        if len(genres) != 1 or len(descriptions) != 1:
            raise ValueError(f"Inconsistent metadata for concept {concept_id}")
        genre, description = next(iter(genres)), next(iter(descriptions))
        if genre not in GENRES:
            raise ValueError(f"Unknown genre: {genre}")
        negative = [r for r in prepared["train"] if r["category"] == "negative" and r["genre"] == genre]
        def unique(rows):
            seen, result = set(), []
            for row in rows:
                key = row["text_id"]
                if key is not None and key not in seen:
                    result.append(row); seen.add(key)
            return result
        positive = unique(positive)
        positive_ids = {r["text_id"] for r in positive}
        negative = unique([r for r in negative if r["text_id"] not in positive_ids])
        rng = np.random.default_rng(seed_for(config.data_seed, "build", concept_id))
        positive = [positive[i] for i in rng.permutation(len(positive))][:config.build_count]
        negative = [negative[i] for i in rng.permutation(len(negative))][:config.build_count]
        if len(positive) < config.build_count or len(negative) < config.build_count:
            exclusions.append({"concept_id": concept_id, "genre": genre, "scope": "concept", "reason": "insufficient_build_texts"})
            continue
        build_ids = {r["text_id"] for r in positive + negative}
        # Held-out references: the concept's positive test responses, deduplicated by their full tokens and
        # disjoint from the construction texts.
        references, seen = [], set()
        for row in prepared["test"]:
            if row["concept_id"] != concept_id or row["category"] != "positive":
                continue
            response_key = digest(list(tokenizer.encode(row["response"], add_special_tokens=False)))
            if response_key in build_ids or response_key in seen:
                exclusions.append({"concept_id": concept_id, "source_id": row["source_id"],
                                   "scope": "evaluation_text", "reason": "construction_overlap" if response_key in build_ids else "duplicate"})
                continue
            seen.add(response_key)
            references.append(row)
        references = unique(references)
        ref_rng = np.random.default_rng(seed_for(config.data_seed, "reference", concept_id))
        references = [references[i] for i in ref_rng.permutation(len(references))]
        if len(references) < 8:
            exclusions.append({"concept_id": concept_id, "genre": genre, "scope": "concept", "reason": "insufficient_positive_references"})
            continue
        concepts.append({"concept_id": concept_id, "genre": genre, "description": description,
                         "positive": positive, "negative": negative, "references": references})
    return {"concepts": concepts, "texts": texts, "exclusions": exclusions, "starting_concepts": len(ids)}


def candidate_sets(concepts, seed=0):
    """Per concept: ten same-genre candidates (the target, the shuffled arm's donor, eight distractors), the
    donors a seeded derangement within the genre."""
    result = {}
    for genre in GENRES:
        ids = np.array(sorted(c["concept_id"] for c in concepts if c["genre"] == genre))
        if not len(ids):
            continue
        if len(ids) < 10:
            raise ValueError(f"MC-10 requires ten usable {genre} concepts, got {len(ids)}")
        rng = np.random.default_rng(seed_for(seed, "donors", genre))
        # Rejection sampling gives a seeded permutation conditioned on no fixed points.
        donor_ids = rng.permutation(ids)
        while np.any(donor_ids == ids):
            donor_ids = rng.permutation(ids)
        for target, donor in zip(ids.tolist(), donor_ids.tolist()):
            crng = np.random.default_rng(seed_for(seed, "candidates", target))
            distractors = crng.choice([i for i in ids.tolist() if i not in (target, donor)], 8, replace=False).tolist()
            candidates = [target, donor, *distractors]
            crng.shuffle(candidates)
            result[str(target)] = {"donor": donor, "candidates": candidates, "correct_answer": candidates.index(target) + 1}
    return result


def target_ids(concepts, per_genre, seed=0):
    """The concepts this run evaluates: all of them, or a seeded draw of `per_genre` from each genre."""
    if per_genre is None:
        return sorted(c["concept_id"] for c in concepts)
    ids = []
    for genre in GENRES:
        available = sorted(c["concept_id"] for c in concepts if c["genre"] == genre)
        rng = np.random.default_rng(seed_for(seed, "smoke", genre))
        ids.extend(rng.choice(available, min(per_genre, len(available)), replace=False).tolist())
    return sorted(ids)


def prepare(run):
    started = time.time()
    key = run.key("prepare", modules=("data",))
    cached = run.cached("data/prepared.json", key)
    if cached is not None:
        return cached
    import pandas as pd
    from huggingface_hub import hf_hub_download
    cfg = run.config
    cache = run.root / "cache"
    tokenizer = load_tokenizer(cfg, cache)
    frames, sources = {}, {}
    for split in ("train", "test"):
        path = hf_hub_download(cfg.dataset, f"{cfg.dataset_subset}/{split}/data.parquet", repo_type="dataset",
                               revision=cfg.dataset_revision, token=False)
        frames[split] = pd.read_parquet(path).to_dict("records")
        sources[split] = {"revision": cfg.dataset_revision, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    prepared = prepare_concepts(frames["train"], frames["test"], tokenizer, cfg)
    prepared["sources"] = sources
    run.save("data/prepared.json", key, prepared)
    run.stage_done("prepare", started, concepts=len(prepared["concepts"]))
    return prepared
