# Precompute and shared baselines: layout, conventions, and the v2 data migration

*Self-contained handover for the pre-processing / shared-baseline branch (`arb/precompute`,
directory `paper-evals/`). Written 2026-09-18.*

## Purpose

This branch produces the **shared baselines** the MAEMM evaluation paper is read against: the
held-out target sets (realact / random / sae, and an empty J-lens slot), the corpus-retrieval
search at the nested sizes 1M/4M/16M tokens, MAEMM rollouts and their scoring on the clean base,
GCG/EPO reachability ceilings, the Delphi-style SAE autointerp evaluation, and the reconstruction
statistics assembled from all of them. Everything here today is built against the 131k-feature SAE
`ceselder/qwen36-27b-sae-l42-1b` (our config key `qwen36-27b/l42-1b`) and against the primary MAEMM
`qwen36-27b/2026-09-10_rl-8x2048-full`. Celeste's v2 training/eval bundle changes **the SAE** (a new
2,097,152-feature BatchTopK SAE, k = 64, gate 1.6828) and **the checkpoints**; it does not change
the base model (`Qwen/Qwen3.6-27B`) or the residual space (layer 42 block output, d = 5120), which
is why the migration is a new SAE entry plus a new held-out set rather than a new base tree.

## How to read this

- The **layout doc body** below is the design spec of 2026-09-15, reproduced almost verbatim. It is
  the intent, not a description of the current tree. Where a statement is superseded by the v2 data
  an inline `**v2:**` note says so; where it disagrees with what is actually in the branch today the
  disagreement is listed under *Known discrepancies* at the end and the body text is left alone.
- The **authoritative current state** is in the branch, not here: `paper-evals/config.yaml` (every
  model, SAE, MAEMM, target set and sampling constant, with the reasons in comments),
  `paper-evals/README.md` (products, volume layout as built, conventions, measured costs) and
  `paper-evals/SMOKES.md` (a dated row for every run actually executed, with wall, cost, result and
  what disagreed with expectation). When this document and those three disagree, they win.
- References to files under `infra/`, `paper/` or `experiments/` in the original spec were to a
  private notes repository you do not have; they have been dropped or redirected to the branch.

---

# The layout doc body (2026-09-15 spec)

# Precompute: data layout, code layout, plan (2026-09-15, for review before coding)

Scope per Tomáš 2026-09-15 (second message): precompute + rollouts + scoring + stats; vLLM implemented with a parity
test; GCG/EPO only on 64 realact targets, 2×2 arms; no Delphi. The product definitions, sample sizes and costs now
live in `paper-evals/README.md`; this file is the concrete layout.

## 1. Data layout (Modal volume `maemm`, workspace `maemms`)

```
/vol/
  archive/gavento-1/{data,runs}/          verbatim old volume, read-only by convention (copied and verified 2026-09-15, minus acts.f16)
  hf/                                     HF cache; everything fetchable by id lives only here
  base/<base>/                            base = qwen3-8b | qwen36-27b
    corpus/         tokens.i32, docs.jsonl (doc, offset, len, size_tag 1|2|4|8|16, source part/row), README
    heldout/<set>/  ids.jsonl (row, family, natural id, stratum), vecs.f16 [N,d], README with the rebuild command
    stats/          mu.f32, resid_norm_quantiles.json (all layers), README
    acts_1m/        read-layer activations of the 1M-token nested subset, block-start contexts, f16 [1M, d] + index (§5 item 8)
    scan/<set>/     topk.jsonl (target row, size → 64 × (doc, start, argmax, cos)), quantiles.f16 [N, 5 sizes, 5 q]
    sae/<sae>/      fire_counts.i64 [F, 5 sizes], max_act.f16 [F], examples/<feature>.jsonl for tested features
    gcg/<set>/<arm>/  finals.jsonl, trajectory.jsonl, summary.json           (64 realact only, arms in §4)
  maemms/<base>/<maemm>/                  maemm = 2026-09-03_run1-rl, 2026-09-08_rlI-150, 2026-09-10_rl-8x2048-full, …
    README.md       source id or archive path, type lora|full, adapter subdir, injection convention, sha of weights
    rollouts/<set>.jsonl      64 per target: row, k, text, token ids, engine hf|vllm, seed
    scores/<set>/   cos.f16 [N,64,T], norm.f16 [N,64,T], argmax.i16 [N,64], best_act.f16 [N,64,d],
                    sae_idx.i32 / sae_val.f16 (sparse, flat + offsets), per_target.jsonl (mean, best-of-n, length)
  runs/<YYYY-MM-DD>_<slug>/               ad hoc trials (today: 2026-09-15_patchscopes-run1)
```

Rules (restated in `paper-evals/README.md`): one README per directory written by the script that fills it,
temp-and-rename, no overwrite without `--force`, no registries or sidecar hashes. jsonl for anything under ~200 MB,
raw f16/i32 arrays with shapes stated in the README for matrices. One held-out set per base, shared by all its
MAEMMs, named by date: `heldout/2026-09-16_v1`. **v2:** a second set `heldout/2026-09-1x_v2` joins it — same base
directory, same corpus, realact/random rows reproducing byte-for-byte, a new `sae` family drawn from the 2M SAE's
eval split. Sizes per base: corpus 64 MB, heldout 20 MB, scan ~30 MB, sae examples ~100 MB;
per MAEMM ~2 GB (8B) / ~5 GB (27B), dominated by `best_act`.

## 2. Code layout (Celeste's repo, new local branch `arb/precompute` from `origin/master` 09d4a01 (2026-09-15);
   `gcg.py` is copied from our fork's `eval/gcg_search.py` and trimmed)

```
paper-evals/
  README.md            one page: the YAML, the volume layout, the run order, costs
  config.yaml          bases, SAEs, MAEMMs, held-out set spec, corpus spec, modal (volume, GPU per base, secrets)
  precompute/
    common.py          config loader; HF model + adapter loading; injection hook; read-layer capture;
                       the one scoring function (kept mask, cos, norm); jsonl/array io; volume paths
    modal_app.py       one Modal function per product, `--product` CLI; the only Modal file in precompute/
    corpus.py          build the 16M slice + nested tags (or import the archived retrieval slice, §5 Q2)
    targets.py         draw a held-out set: families, strata, seeds → heldout/<set>
    scan.py            the streaming pass: cos top-k + quantiles, SAE counts + examples, mu + norm stats
    rollouts_hf.py     reference generation path (HF generate + hook)
    rollouts_vllm.py   fast path (vLLM + vllm-lens hook), identical output format
    score.py           base forward on rollouts → scores/
  gcg/
    gcg.py             modes gcg / epo, inits random32 / corpus; writes gcg/<set>/<arm>/
    modal_app.py
  reconstruction/
    parity.py          HF vs vLLM (§3)
    stats.py           scores + scan + gcg → per-family tables, best-of-n curves, paired comparisons
    README.md          what each table means
```

Sharing: only `precompute/common.py` is imported across the three directories, for loading, injection and scoring,
so GCG and rollouts cannot drift on the objective. Everything else is a script with a `main`.

`config.yaml` sketch, special-cased per MAEMM as you suggested:

```yaml
bases:
  qwen3-8b:   {hf: Qwen/Qwen3-8B,   read_layer: 27, d: 4096, gpu: H100, eos: [151645, 151643]}
  qwen36-27b: {hf: Qwen/Qwen3.6-27B, read_layer: 42, d: 5120, gpu: H200, eos: [151645, 151643]}
saes:
  qwen3-8b/adamkarvonen-t2: {hf: adamkarvonen/qwen3-8b-saes, subdir: ..., layer: 27}
  qwen36-27b/l42-1b:        {hf: ceselder/qwen36-27b-sae-l42-1b, layer: 42}
maemms:
  qwen3-8b/2026-09-03_run1-rl:      {type: lora, src: archive/gavento-1/runs/run1/rl/final, prompt: ours8b, inject: {layer: 1, coef: 1.0}, train_max_new: 64}
  qwen36-27b/2026-09-08_rlI-150:    {type: lora, hf: ceselder/maemm-qwen36-27b-inverter-rlI-step150, subdir: step_150, prompt: celeste27b, inject: {layer: 1, coef: 1.0}, train_max_new: 96}
  qwen36-27b/2026-09-10_rl-8x2048-full: {type: full, hf: ceselder/qwen36-27b-maemm-inverter-rl-8x2048, prompt: celeste27b, inject: {layer: 1, coef: 1.0}, train_max_new: 192}
heldout:
  2026-09-16_v1: {seed: 20260916, families: {realact: 512, random: 512, sae: 512, jlens: 512}, realact_len: [1, 64], sae_min_fires: 20, sae_strata: 4}
corpus: {dataset: openbmb/Ultra-FineWeb, parts: [p0009, p0010], tokens: 16_000_000, sizes: [1, 2, 4, 8, 16], seed: 20260916}
rollouts: {n: 64, temperature: 1.0, max_new: 64, min_new: 16, seed: 1234}
```

**v2:** a third `saes:` entry is added for the 2M SAE (`qwen36-27b/sae2m-<id>`, `kind: batchtopk, k: 64, gate:
1.6828`, with an external `feature_split`), and a `heldout: 2026-09-1x_v2` block; nothing else in this sketch
changes.

`prompt:` names a small function in `common.py` (ours8b / celeste27b: template, marker token, chat template or not);
that is where the per-MAEMM special casing lives, verified against each MAEMM's training code once.

## 3. vLLM: plan and what the parity test cannot see

Build: `rollouts_vllm.py` on vLLM 0.19 + vllm-lens 1.1 (the versions our driver pins; Celeste's `rl/fast_lens_ext.py`
is the working example of the hook), LoRA via vLLM's LoRA path (r 64), full model as a plain vLLM model, prefix
caching off, `seed` per request. Output format identical to `rollouts_hf.py` so `score.py` does not care.

Parity (`reconstruction/parity.py`), against data we already have (run1's archived dumps, 7 families × 512 × bo4):
1. Greedy logit test: 8 directions, HF hook vs vLLM hook, first-token logit agreement and greedy token match over
   64 tokens. Catches a no-op or misplaced hook outright.
2. Paired per-direction comparison: vLLM bo 16 on 128 directions of each family vs the archived HF scores of the same
   directions; per-direction mean cos correlation, best-of-4 distributions (KS), rollout length and EOS-rate
   distributions, argmax-position distribution.
3. Scoring stays HF in both, so scoring is not a source of difference.

Not visible to a distribution comparison, hence the pairing and the greedy test:
- Directions permuted or prefix-cache KV reused across directions: marginals identical, per-direction pairing drops.
- Wrong sibling checkpoint (e.g., rl step 140 instead of final): a small shift inside noise; only the weight sha in
  the MAEMM README guards it.
- A one-token prompt difference (space before the marker, chat template on/off): mean shift of ~0.01, inside the bo4
  noise at n=512; the greedy test catches it, the distributions do not.
- Sampling parameter drift (top-p, EOS set, min tokens): shows in lengths and EOS rates, not in cos alone, so both
  are compared.
- Hook applied at the right layer but wrong position (e.g., last prompt token instead of the marker): may still score
  well; the greedy test is the only guard.

## 4. GCG / EPO on 64 realact targets

The first 64 rows of the realact family (nested subset). Arms, 2×2:
- mode: `gcg` (pop 1, λ 0, 512 children × 150 iters) and `epo` (pop 3, λ ∈ {0.1, 0.19, 0.37} as the earlier fluent
  arm, 85 children each × 300 iters; both are the configs measured on our earlier 8B GCG runs, and the arm table in
  `paper-evals/README.md` is the current record of them);
- init: `random32` (32 random ASCII tokens) and `corpus` (the scan's top-1 window for the target, truncated to 32
  tokens; new init).
Outputs per arm: finals (string, cos, NLL, per-token cos), top-64 distinct candidates, trajectory.
Cost at measured 8B rates ($0.12 gcg, $0.26 epo per direction): 64 × 0.38 × 2 inits ≈ $49 on 8B; ~$170 on 27B at
3.5×. Proposal: 8B first, 27B after seeing the 8B result. **v2:** unchanged — GCG depends on the base and the
realact rows only.

## 5. Decisions of 2026-09-15 (Tomáš's answers)

1. Branch: `arb/precompute` from Celeste's current master (09d4a01, 2026-09-15). Directory `paper-evals/` (she has
   `eval/`; not combined). Our code is independent of hers; a scan lists the conventions that must match, everything
   else develops separately.
2. Corpus: rebuilt from Ultra-FineWeb p0009-0010 by the new code and verified; nothing promoted from the archive.
   All products are regenerated with the new code; the archive is reference only. **v2:** this corpus is now
   *verified* document-disjoint from the v2 training ranges (see *Disjointness*), not merely disjoint by dataset.
3. Rollouts: max_new 64. Parity: run1 only. J-lens: 27B only.
4. 27B is the primary model; 8B is dev / secondary / ablation. From 2026-09-16: the full 27B model `2026-09-10_rl-8x2048-full` is the primary MAEMM; rlI-150 and 8B run1 are secondary rows kept where cheap.
5. GCG/EPO: 8 realact targets per arm first (2×2 arms), on both bases (~$6 8B, ~$21 27B).
6. Budget now: ~$40 for smoke tests; no full runs until asked.
7. `acts.f16` stays on the old volume only (README note written); the old-workspace secret is deleted.
8. Corpus activations: store the read-layer activations of the 1M-token nested subset only (block-start contexts,
   ~8 GB 8B / ~10 GB 27B, ~$1/month per base) so re-analysis with new target banks, whitening and nearest-neighbour
   checks need no GPU pass; the 16M scan stays streamed (0.5-0.65 TB otherwise).
9. Her 2026-09-15 commit: SAE "fired" = the SAE's learned BatchTopK gate threshold (1.5846 for `l42-1b`; 1.6539 is the older `-l42` SAE), not
   raw act > 1.0; our SAE-activation benchmark uses the same gate. **v2:** the 2M SAE's gate is **1.6828**; the same
   rule applies to it.

10. Conventions audit of her repo (2026-09-15, scan): the sae family direction is `unit(W_enc[:, f])`, the encoder
    column (not the decoder row as the earlier product spec said); realact directions are `unit(act − whiten_mu)` while the
    scorer cosine is uncentred; realact targets come from 512-token windows without BOS, span position p ~ U[16, T),
    shown span L ~ U[16, 64]; injection = block output of layer 1, add `unit(v)·‖h‖·1.0` at the single " ?" marker,
    chat template with thinking off; read layer = block output; re-encode with a BOS sink then drop position 0; norm
    filter 10× median (we store unfiltered and apply it in stats); sampling top_p 1, top_k 0, seed 1234; vLLM prefix
    caching off, her hook swallows errors (we assert none). All kept as-is for comparability.

## 5a. Questions (answered above; kept for the record)

1. Branch base: `origin/master` (PR-able to Celeste, 27B-only code, we add 8B in our scripts) versus our fork's
   `arb/modal-8b` (has the 8B line and `gcg_search.py`, diverged). Default: master, copy `gcg_search.py` in.
2. Corpus: reuse the archived 16M retrieval slice (Ultra-FineWeb p0009-0010, `runs/retrieval`) if it stores token ids
   and doc boundaries, else rebuild from the same parts. Default: rebuild deterministically; it is minutes of CPU.
3. Rollout length: 64 (matches all archived dumps) or 96 (covers rlI's training regime). Still open.
4. J-lens: 27B only until Celeste answers; the 8B set has three families.
5. Parity target: run1 bo4 dumps as above. Fine, or do you want run2 too.
6. 27B full model in vLLM on one H200: 55 GB bf16 + KV; fits. LoRA + full in one script, chosen by `type:`.

## 6. Build order

1. Worktree + branch; `config.yaml`; `common.py` with the HF path and the scoring function; unit smoke on 8B.
2. `corpus.py`, `targets.py`, `scan.py`; 8B corpus pass (~$3).
3. `rollouts_hf.py` + `score.py`; run1 on 16 directions must reproduce the archived scores.
4. `rollouts_vllm.py` + `parity.py`; the parity report.
5. Rollouts for the three 8B MAEMMs with the winning engine; then 27B (corpus pass, rlI-150, rl-8x2048).
6. `gcg/` arms on 8B; `reconstruction/stats.py`.

---

# v2 data (2026-09-17 bundle) and the migration plan

Surveyed 2026-09-18 from a presigned listing: 95 objects, 141.88 GB, bucket `celeste-maemm-27b-data`, prefix
`v2-2026-09-17/`. The per-object links expire **2026-09-24 20:51 UTC** — anything not mirrored before then needs a
fresh listing from Celeste.

## What it is

Not a text corpus: the complete **MAEMM training + evaluation bank** of a new "simple2m" chain for
`Qwen/Qwen3.6-27B` at layer 42 (d = 5120 — her "layer-42" is our `read_layer: 42`, the same residual space as
`base/qwen36-27b/`). Every direction is a unit fp16 row in the raw layer-42 residual space. Text provenance
throughout is `openbmb/Ultra-FineWeb`, split `en`, addressed by `doc_idx` = 0-based document index in the ordered
single stream of that split.

| S3 subtree | files | GB | content |
|---|---|---|---|
| `README.md`, `manifest.json` | 2 | 0.00 | manifest with rows/bytes/columns per file, `built 2026-09-17T20:58Z` |
| `heldout/` | 32 | 2.31 | the held-out evaluation data |
| `simple2m/sft_mix/` | 9 | 82.73 | SFT bank: 8M rows = `realact` 4M + `sae2m` 2M + `sae2m_dec` 2M |
| `simple2m/rl_pool/` | 9 | 9.76 | RL pool: 941,132 rows = `realact_ctx64_2048` 470,566 + `sae2m` 235,283 + `sae2m_dec` 235,283 |
| `legacy_5m_chain/` | 34 | 43.36 | the earlier chain's 2.75M-row 8-family midtrain mix and its 1.45M-row RL pool |
| `extra/` | 7 | 3.70 | `maxacts.pt` (131k SAE), `maxacts_fresh.pt` (131k SAE, FineFineWeb scan), `maxacts_top5.pt` (2M SAE, top-5 windows, 1.0B tokens / 747,623 docs), `feature_split.{npz,json}`, `doc_registry.json` |

Formats: each training family is `dirs_f16.npy` (`[rows, 5120]` fp16, row *i* == `records.parquet` row *i*) plus
`records.parquet` and small json. Held-out files are parquet with a `direction` list<float32>[5120] column.
`extra/*.pt` are torch pickles.

**The new SAE.** `sae2m` / `sae2m_dec` pair the unit encoder column resp. decoder row of a **2,097,152-feature
layer-42 BatchTopK SAE (k = 64, gate 1.6828)** with end-anchored 32-token max-activating windows. Its **weights are
not in the bundle** (W_enc alone is 2,097,152 × 5120 = 43 GB bf16) and no public HF repo for it existed under
`ceselder/` as of 2026-09-18 12:40 UTC.

**The held-out bank.** `feature_split.parquet` (2,097,152 rows) is a seed-2026 permutation of the 2M SAE's features
into `eval` 100,000 / `rl` 150,000 / `sft` 1,847,152; no simple2m training bank contains an `eval` feature.
`eval_2m_features_512.parquet` is her standard-eval 512-feature subset (with `enc_dir`, `dec_dir`, `b_enc`,
`corpus_peak`, `gate`). `eval_2m_features_100k_windows.parquet` holds 500,000 top-5 32-token windows.
`eval_directions_v3/<family>.parquet` (512 rows each, 256 for `mlp_pair`) covers `bsf, realact, jlens, cluster,
random, realact_early, realact_mid, realact_long, indist_long, indist_probe, indist_realact`, the 131k-SAE `sae`
family, `mlp`/`mlp_pair`, and the 2M slices `sae2m_enc`/`sae2m_dec`. `pool_heldout/<family>.parquet` are the
train-disjoint pools those are sampled from; its `realact` comes from the stream head, docs [0, 100,000).
Her headline `mean_all` = mean over the 10 cosine families (random excluded) of best-of-4 max-token cosine.

**Missing:** the 2M SAE weights, and any v2 checkpoint. The bundle has the data but no models.

## Download record (tier A)

Tier A = everything but the direction matrices — 75 objects, 7,331,367,436 B — was mirrored on 2026-09-18 12:48 UTC
to Modal volume `maemm` (workspace `maemms`) at `/vol/data/celeste-v2-2026-09-17/`, keys minus the
`v2-2026-09-17/` prefix, with a `MANIFEST.json` carrying a sha256 per file (30 cross-checked against independently
fetched copies). 86 s wall, ~0.08 vCPU-h, ~$0. Tier B (A + the simple2m direction matrices) is 98.52 GB in 61
objects and buys only a vector-level cos > 0.999 leakage check; tier C (all 95 objects, 141.88 GB) adds the legacy
chain's banks. Neither has been fetched.

## What depends on the training data, and what does not

| product (stage) | depends on | v2 status |
|---|---|---|
| `corpus/` (CPU) | dataset choice only | unchanged; doc-disjoint from v2 training by index |
| `stats/mu.f32`, resid norm quantiles | base + our corpus | unchanged |
| `heldout/<set>/realact`, `random` | our corpus, seed | unchanged in content; the disjointness statement gets stronger |
| `heldout/<set>/sae` | the SAE's fire counts on our corpus + rarity strata | **rebuild for the 2M SAE** (feature-level held-out = her `eval` split) |
| `heldout/<set>/jlens` | J-lens matrix | still empty |
| `scan/<set>/` | corpus + set | rerun for the new set (realact/random rows come out identical) |
| `sae/<sae>/` products incl. the autointerp inputs | corpus + SAE | **new SAE entry**; `repo_examples` can score her `eval_2m_features_100k_windows` top-5 windows as the shipped-windows baseline |
| `gcg/<set>/` | base + realact rows | unchanged |
| `patchscopes/<set>/` | clean base + set | only new sae rows would be new |
| `maemms/<base>/<maemm>/` | the MAEMM weights | new entry per v2 checkpoint, once we have weights |

**Naming — no new base prefix.** Keep `base/qwen36-27b/`. The layout keys the base directory by base model, and the
residual space is identical; a `qwen36-27b-v2` prefix would copy `corpus/` and `stats/` verbatim and break "one
held-out set per base, shared by all its MAEMMs" — old-vs-new MAEMM on the *same* realact/random targets is exactly
what the paper wants. The change is expressed where the layout already keys it: a new `saes` entry, a new held-out
set `heldout/2026-09-1x_v2` (v1's realact + random at the same seed 20260916 — `targets.py` draws realact → random →
sae from one rng stream, so the first two reproduce byte-for-byte; assert it with a diff) plus `scan/2026-09-1x_v2/`,
and `maemms/qwen36-27b/<date>_<slug>-v2/` per checkpoint with `train_data: celeste-v2-2026-09-17` in its README.
The bundle itself lives under `data/celeste-v2-2026-09-17/`, outside `base/`, because it is an input.
Keep the 131k `sae` family in the v2 set: the v2 chain never trains on it, so it is a free out-of-dictionary row.

## Stages and cost (measured rates: H200 $4.54/h)

| # | stage | scope | measured basis | est. $ |
|---|---|---|---|---|
| 0 | tier A download + doc-index assertion + 13-gram check | CPU | done, see *Disjointness* | ~0 |
| 1 | fetch 2M SAE weights to `/vol/hf` | 43-86 GB | — | ~0 + storage |
| 2 | `stats` with the new SAE entry (eval split 100k features, or all 2M) | 16M | $9.12 / 7,233 s at 16M for `l42-1b` | 9-12 |
| 3 | `targets` set v2 | 512 × 4 | $0.19 | 0.2 |
| 4 | `scan` set v2 | 16M | $6.34 / 5,026 s | 6.5 |
| 5 | SAE-keyed autointerp products (`sae_self`, `random_pool`, `examples_4m`, `examples_docmax`, `top1_act`, `repo_examples` on her windows) | 16M | an earlier second-base build minus rollouts/score | 15 |
| 6 | per v2 MAEMM: `rollouts_vllm` 1,536 × 64, `score`, `centred` | per checkpoint | $4.50 + $0.63 | 5.2 each |
| 7 | optional: autointerp LLM run on the new SAE | 64 features | $10.16 measured on an earlier build | 10 |

Base rebuild (0-5) ≈ **$31-34**, plus ≈ $5 per v2 checkpoint and $10 if (7). Wall: stats 2 h, scan 1.4 h, the rest
under 1 h each. Nothing here has been launched. Stages 1, 2 and 6 are blocked on Celeste (questions 1 and 2 below).

## The primary checkpoint's training data is not pinned

`qwen36-27b/2026-09-10_rl-8x2048-full` was trained on **`m-a-p/FineFineWeb`, not Ultra-FineWeb** (her repo's
plotting and collector scripts; the checkpoint's own `data/bank_recut.json` / `midtrain.json`). FineFineWeb is
English-only, FineWeb-derived, apache-2.0; its `mathematics` domain is ≈0.14% of tokens and
`computer_science_and_technology` ≈4.6% (prose about computing, not code). The midtrain bank mixes eight families
(realact 1,000,000 · sae 396,506 · sae_dec 396,506 · cluster 204,366 · bsf 225,573 · mlp 460,842 · mlp_pair 57,946 ·
mlp_triple 5,770), every one with an English FineFineWeb target window. **No FineFineWeb revision is pinned anywhere,
and which domain files went into each bank is not recorded**, so the defensible premise sentence is "trained on
directions derived from English web text", not "trained on Ultra-FineWeb"; the RL pool's composition is likewise not
itemised in the public files. The v2 chain, by contrast, is Ultra-FineWeb en throughout with explicit doc ranges.

# Disjointness: our held-out corpus vs her v2 training ranges

Verified to completion on 2026-09-18 by a standalone `modal run` check script (CPU only, 150 s, ≈0.33 vCPU-hours,
≈$0; the script lives outside this branch — ask Tomáš for it before re-running). Raw numbers were written to `/vol/base/qwen36-27b/corpus/celeste_v2_disjointness.json` and
one summary line was appended to that corpus directory's README.

**Method.** `doc_idx` is the cumulative row index over `data/ultrafineweb_en/ultrafineweb-en-part-NNNN-of-2048.parquet`
in filename order, **starting at part 0001** (there is no part 0000). Row counts were re-read from the public parquet
footers at run time (parts 1-10: 566,021 / 566,020 / 566,019 / 566,018 / 566,018 / 566,019 / 566,023 / 566,021 /
566,020 / 566,020), giving `start[9] = 4,528,159` and `start[10] = 5,094,179`; the script asserts these and aborts on
a mismatch. Our 16M corpus is the head of each part (part 0009 rows 0..9388, part 0010 rows 0..9423; 18,813 docs),
i.e. stream indices **[4,528,159, 4,537,547] and [5,094,179, 5,103,602]**.

**Result: exact document-level disjointness, `total_intersection: 0`, tightest margin 396,399 documents.** Checked
against all eight sources in `extra/doc_registry.json` — the six exact `heldout/doc_ids/*.parquet` lists and the two
declared ranges:

| source | kind | her docs | min..max doc_idx | ∩ ours | min distance |
|---|---|---|---|---|---|
| `sae2m_dictionary_training_stream` | range | 1,683,561 | 100,000..1,783,560 | 0 | 2,744,599 |
| `sae2m_sft_bank_windows` | exact | 255,259 | 100,001..1,783,560 | 0 | 2,744,599 |
| `sae2m_rl_bank_windows` | exact | 85,453 | 100,011..1,783,560 | 0 | 2,744,599 |
| `sft_activations_ctx8_64` | exact | 125,000 | 5,500,001..5,698,523 | 0 | **396,399** |
| `rl_activations_ctx64_2048` | exact | 62,504 | 9,500,000..9,599,841 | 0 | 4,396,398 |
| `eval_head` | range | 100,000 | 0..99,999 | 0 | 4,428,160 |
| `eval_sae2m_512_feature_windows` | exact | 2,202 | 100,816..1,780,456 | 0 | 2,747,703 |
| `eval_sae2m_split_100k_feature_windows` | exact | 135,051 | 100,011..1,783,560 | 0 | 2,744,599 |

**Caveat: doc-disjoint is not content-disjoint.** Ultra-FineWeb is not deduplicated across parts. A 13-gram check
(NFKC → lowercase → whitespace split → 13-word shingles → `xxh3_64`) over her 13.6M `target_text` rows found that
**81 of our 18,813 documents (0.43%) are at least half reproduced inside her v2 + legacy training text, 254 (1.35%)
at least a fifth, and 1,732 (9.21%) share at least one 13-gram**; only 1.014% of our 11,709,262 distinct 13-grams
are reached at all. The overlap is concentrated in the SAE max-activating families (`sae2m` hits at 8.31% of rows
against `realact`'s 0.50%) because max-act windows select for templated, duplicated web text. Her own eval windows
overlap our corpus at a comparable rate (644 documents, 0.376% of our 13-grams), so a contamination argument applies
symmetrically to her held-out evaluation. Controls passed: 200 word-spans cut from our own documents hit 2,848/2,848
shingles; 200 of her eval windows shuffled at word level hit 0/2,293.

Two limits worth carrying forward. (1) Part A's completeness rests on Celeste's claim that `doc_ids/*.parquet` lists
every document any v2 training row touches (question 4 below) — exact given yes, vacuous given no. (2) 1.8M of her
rows (13.3%) are shorter than 13 words and are invisible to this check by construction. Whether to act on the 81
near-duplicates is open: dropping them renumbers `doc` ids and invalidates every downstream index, so it is a
re-derivation of the whole corpus, not an edit. **Not done.**

# Working conventions

- **Stage by explicit path.** Never `git add -A` / `git add .` / `git commit -a`; other work may be in flight in the
  same tree.
- **Nothing goes out without Tomáš's say-so in the current session** — no push, no HF upload, no message to
  collaborators.
- **`SMOKES.md` logs every run**: date, item, abbreviated command, wall, cost, result, and what disagreed with
  expectation. The cost is taken from each product's own README on the volume, because `modal app logs` replays
  stale output.
- **Run the selfcheck before any launch.** Every module has one (`precompute/unit_smoke.py`, `autointerp/selfcheck.py`,
  and the in-script `_selftest()` pattern); it runs locally before anything is submitted and again inside the Modal
  function.
- **Launch Modal under `setsid`**, not `timeout` or `nohup`, and use bounded blocking loops rather than monitors.
- **Report any experiment over $100** before running it; the autointerp path additionally refuses a stage projecting
  more than `stop_above_usd` without `--approved`.
- **Do not write under `paper/overleaf/` or `paper/sae-autointerp/`**, and leave the other checkouts
  (`repo-maemm/`, `repo-maemm-master/`) and the `gavento-1` archive volume alone.

# Open questions for Celeste

1. Which checkpoints were trained on this v2 data (HF ids), from what init (scratch / `pretrain-104m` / the legacy 5M
   chain), and SFT-only or SFT + RL? The bundle has the data but no weights.
2. The 2M SAE: weights location (HF id or S3), training tokens, and confirmation that its layer 42 is the block
   output we read. We need at least `W_enc[eval split]`, `b_enc`, the gate (given as 1.6828) and `b_dec` to draw and
   score an SAE target family; the 512 `eval_2m_features_512` rows are already usable.
3. "Raw layer-42 residual space, unit rows": are v2 realact rows `unit(act)` uncentred, or `unit(act − mu)` as in the
   earlier chain? Which mu, if any, does the v2 scorer subtract?
4. Confirm `doc_idx` = cumulative row over the en parts in filename order from part 0001, with no filtering before
   indexing (we verified three windows), and that `doc_ids/*.parquet` list every document any v2 training row
   touches, so a doc-level disjointness assertion against them is complete.
5. Did the 131k SAE `l42-1b` reach the v2 models through their initialisation? Which doc range did `l42-1b` train on
   (it is not in the registry)?
6. Will `heldout/` (or the whole bundle) get a durable home before the presigned links expire on 2026-09-24?
7. Which families make up the `mean_all` she now reports (the v3 meta says the v2 keys are untouched; the new
   `indist_*`, `realact_{early,mid,long}` and `sae2m_*` slices are logged under `eval/<fam>/*`)?

Also worth asking, for the OOD premise: the RL pool's composition and which FineFineWeb files/domains went into each
bank of the primary checkpoint.

# Known discrepancies (spec above vs the branch today)

Checked 2026-09-18 against `paper-evals/config.yaml` and `paper-evals/README.md`. The body text was **not** edited
to match; these are the deltas.

1. **`acts_1m/` does not exist.** §1 and §5 item 8 specify it; it was dropped on 2026-09-15 (`precompute/stats.py:11`:
   "nothing downstream read it"). Treat §5 item 8 as reversed.
2. **`fire_counts.i64` is `[F, n_sizes, 2]`**, not `[F, 5 sizes]` — the last axis stores both the `> 0` and the
   `> gate` counts, and the `≥ 20 fires` eligibility uses the gated one.
3. **`sae/<sae>/` holds much more than §1 lists**: `mean_when_active.f16`, `sizes.json`,
   `examples/{_random256.jsonl, tested.json}`, `repo_examples/<set>/` and `top1_act/<set>/`.
4. **GCG output path is `gcg/<set>/<family>/<arm>/`** with `family ∈ {realact, sae}` and a `top64.jsonl`; §1/§4 say
   `gcg/<set>/<arm>/`, realact only. `lens_floor/<set>/` is declared in the README's layout and not built.
5. **`scores/<set>/` shapes are `[N, n, 96]`**, not `[N, 64, T]`, with `sae_off.i64` and `rows.json` alongside; a
   `scores/<name>__rescore-<x>/` flat variant also exists. `heldout/<set>/` additionally has `mu_512.f32`
   (diagnostic only) and `leakage.jsonl`; `stats/` additionally has `mu_by_size.f32`.
6. **Two MAEMM entries are missing from the §2 sketch and are computed today**: `qwen36-27b/2026-09-16_base-control`
   (`type: base`, `role: control` — the clean base under the primary's prompt, marker, injection and sampling) and
   `qwen36-27b/2026-09-05_rlE-250` (`compute: false`). Three more are declared `compute: false`. Note also that
   `README.md`'s config table lists `rlE-250` but not `base-control`, so the README is stale on this point.
7. **The `saes` entries use `file:` plus a `max_acts:` block** (repo, file, `windows`, `sink_first`), not the
   sketch's `subdir:`. 8B: `adamkarvonen/sae_max_acts`, 30 windows, `sink_first: true`. 27B: in the SAE repo itself,
   32 windows, `sink_first: false`.
8. **`jlens` is `{n: 512, bases: [qwen36-27b], status: empty}`**, not a plain 512 — `targets.py` writes zero rows.
9. **`realact_len: [1, 64]` is in conflict with the recipe the code follows.** config.yaml carries the conflict as a
   comment: the value is the brief's, while §5 item 10 above and `targets.py` implement L ~ U[16, 64] at
   p ~ U[16, T) inside 512-token no-BOS windows. Unresolved; do not read `realact_len` as describing the draw.
10. **8B SAE gate is 6.936** (`adamkarvonen-t2`); §5 item 9 gives only the 27B numbers, which check out (1.5846 for
    `l42-1b`, 1.6539 for the older `-l42`).
11. **GCG costs and scope differ from §4.** Measured: 8B $0.09-0.10/dir (`gcg`), $0.28/dir (`epo`); 27B $0.33/dir and
    ~$1.2/dir. §5 item 5 already cut the scope from 64 directions to 8 per arm; the runs actually logged cover 31-32
    directions per arm, and `-strat` variants exist that §4 does not mention.
12. **§2's code layout omits what is now in the tree**: `precompute/{centred,mu_diag,patchscopes,repo_examples,
    top1_act,unit_smoke,vllm_ext}.py`, `reconstruction/{repro_run1,corpus_top1_activation}.py` and
    `reconstruction/compute.md`, and the entire `autointerp/` package (the Delphi-style SAE autointerp evaluation,
    configured under an `autointerp:` block appended to `config.yaml` on 2026-09-16).
13. **Verified as stated**: vLLM pin (`vllm==0.19.0`, `vllm-lens==1.1.0` in `precompute/modal_app.py:41`), both
    bases' `read_layer` (27 / 42), `d` (4096 / 5120) and GPU (H100 / H200), the corpus block (Ultra-FineWeb en parts
    0009 and 0010, 16M tokens, sizes 1/2/4/8/16, seed 20260916), the `rollouts` block (n 64, T 1.0, max_new 64,
    min_new 16, seed 1234, plus top_p 1.0 / top_k 0), `heldout` seed 20260916, `sae_min_fires: 20`, `sae_strata: 4`,
    the modal block (volume `maemm`, `HF_HOME=/vol/hf`, archive `/vol/archive/gavento-1`, secret `hf-write`), and the
    GCG/EPO arm shapes (1 × 512 × 150; 3 × 85 × 300 at λ 0.1 / 0.19 / 0.37). config.yaml additionally records
    `n_layers` (36 / 64) and measured `marker_norm_base` (14.5 / 14.062), which §2's sketch has no slot for.
14. **§1's per-base size estimates are unverified** against the full run. The only measured points to hand are the
    27B `scan` at 2.4 MiB plus 50.0 MiB of SAE examples (SMOKES, 2026-09-16); the order of magnitude holds, the
    numbers are not the run's.
