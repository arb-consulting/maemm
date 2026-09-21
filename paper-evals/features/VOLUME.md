# Modal volume `maemm` (workspace `maemms`) — the MAEMM evaluation store

Written 2026-09-21. This is the shared store for the MAEMM / Exemplifier paper's
evaluations. If you are an agent or a person who has just been pointed at this volume,
read this file and then `paper/README.md`.

**The inventory is GENERATED, not in this file.** `INVENTORY.md` beside this one lists
every model, target set, corpus, SAE product, rollout, score and baseline that exists
right now, with each set's selection rule pulled from its own README. Regenerate it
with `python -m features.inventory --put`; do not hand-edit it. This file is the part
that changes slowly: layout, conventions, and what will bite you.

**The one thing to understand first:** this volume holds products keyed by *type*
(`scan/`, `gcg/`, `maemms/<m>/rollouts/`), which is right for producing them and useless
for finding them. `paper/<section>/README.md` indexes each paper experiment — what
belongs to it, where each piece is, and whether it is ready, running or discarded.
Start there when you want "everything for §3.2".

---

## 1. What is pinned

Everything below is fixed. If a number was produced against something else, it is not
comparable and should be relabelled rather than merged.

| | id | sha | note |
|---|---|---|---|
| base model | `Qwen/Qwen3.6-27B` | — | 64 layers, d_model 5120, d_mlp 17408, **read layer 42** (block output) |
| MAEMM checkpoint | `ceselder/maemm-27b-rl-last16-lr5e-7` | `a1e4f299` | **full-parameter**, 55.6 GB, not an adapter |
| SAE (primary) | `ceselder/qwen36-27b-sae2m-l42` | `de89b1c1` | F=2,097,152, k=64, **gate 1.682811975479126** |
| SAE (secondary) | `ceselder/qwen36-27b-sae-l42-1b` | — | F=131,072, **gate 1.5845966339111328** |
| upstream data | Celeste's v2 bundle | `celeste-v2-2026-09-17` | mirrored at `data/` |

All live under `hf/hub/models--*/snapshots/<sha>/`. `common.snapshot()` asserts
**exactly one snapshot per repo** — two revisions in the cache makes "which weights did
that run use" unanswerable after the fact, so fetch one and only one
(`features/fetch_hf.py`).

### The checkpoint, in detail

Full-parameter RL fine-tune of the base, simple2m chain, step 300. **The RL reward is
the max cosine over the LAST 16 generated tokens** (`--reward-window-last 16`, confirmed
by Ari 2026-09-19), not the whole span — the draft's Table 7 says "entire generated
span" and is wrong on this point. lr 5e-7 flat after a 25-step warmup, 8×2048 per step,
CISPO/GRPO. SFT init is the 8M-row simple2m mix.

Verified loading: **marker ‖h‖ 294.0 against the clean base's 14.062.** That is the
check that a full-parameter MAEMM actually loaded, and it is the one that works. Note
that `weight_sha256` in the rollouts summary does **not** distinguish checkpoints: it
hashes the shard *index*, so every full-parameter checkpoint of this architecture gives
`d388…6daf`. Do not use it as an identity guard.

`generation_config.json` ships `top_k=20, top_p=0.95`. The eval protocol is
`top_p=1, top_k off`. Pass sampling explicitly or the defaults silently truncate.

### The SAEs, in detail

The 2M SAE's `config.json` records `norm_factor: 83.37198739567428` **and**
`norm_factor_folded` equal to it — the normalisation is already folded into the shipped
weights, so raw layer-42 residuals feed straight in with no rescaling. This matters:
the same ambiguity is still OPEN on the 131k SAE, where recomputing activations on the
SAE's own shipped max-act windows does not reproduce the file (median ratio 1.301,
13% residual after a per-feature affine fit, pooled r 0.863, while the 8B control
reproduces to 0.2%). Slope ≈ 1 with a per-feature intercept ≈ +3 points at a missing
additive term in the encoder (a dropped `− b_dec`, or a dropped `b_enc`), not drifted
weights. Unresolved.

Its `verify.json`: EV 0.711, L0 67.45, **0 dead features**, 100% fired over 20.0M eval
tokens.

**Gates are read from the data, never hardcoded.** Four values are in circulation
(1.6828 for 2M, 1.5846 for 131k, 6.936 for the 8B SAE, and a stale 1.654 from old
project notes).

---

## 2. Layout

```
hf/                      HF cache. One snapshot per repo, asserted.
data/                    Celeste's v2 bundle, tier-A mirror. An INPUT; never written to.
base/qwen36-27b/
  corpus/                the 16M-token corpus, 64-token windows at stride 16
  corpora/<name>/        alternative corpora; see train_parity_10m below
  stats/                 mu.f32 [5120], mu_by_size.f32, resid_norm_quantiles.json
  heldout/<set>/         target sets: ids.jsonl + vecs.f16 [N, 5120] fp16 unit rows
  scan/<set>/            corpus-search baseline: topk.jsonl, quantiles.f16
  sae/<sae>/             fire_counts.i64 [F, n_sizes, 2], max_act.f16, mean_when_active
  gcg/<set>/             discrete-search reachability arms
  patchscopes/<set>/<cell>/   the zero-shot patching baseline and its matched floor
maemms/qwen36-27b/<checkpoint>/
  rollouts/<set>.jsonl   one row per (target, rollout)
  scores/<set>/          cos.f16, norm.f16, best_act.f16, argmax.i16, per_target.jsonl
                         + cos_centred_best.f16, cos_filtered_best.f16, centred.json
shared/                  the agreed artifacts everyone reads (section 4)
paper/<section>/         one folder per paper experiment: README.md, index.json, results/
archive/, runs/, tmp/    old volume, ad hoc runs, scratch
```

**Products are indexed by their own README.** Every directory carries one, written by
the script that filled it, with the shapes, the command that rebuilds it and the
measured cost. `index.json` beside it is the machine-readable version. Writes are
temp-and-rename: a `<name>.tmp-<date>` directory means a run is in flight or died.

---

## 3. Conventions that change numbers

These are the ones where two people can compute "the same" number differently. Read
this section before quoting anything.

### Centring

There is **exactly one** centring and it happens at target construction: a `realact`
target is `unit(X[p] − mu)`. Everything else is uncentred — every stored activation,
`best_act.f16`, every primary cosine. So the target is centred and the measurement it is
compared against is not. That asymmetry is deliberate (Celeste's) and is kept.

Consequence, measured on this checkpoint: realact bo1 **0.471 uncentred vs 0.731
centred**, bo64 **0.560 vs 0.852**. Same rollouts. Quote the convention or the number
means nothing.

`scores/<set>/cos_centred_best.f16` is the both-sides-centred reading, computed on CPU
from stored arrays. For `sae` and `random` the target was never centred, so subtracting
mu from the activation alone is one-sided and that row is a diagnostic only.

**Celeste's v2 eval directions are ALREADY centred** with her `whiten_mu`, which is not
in the bundle. Measured, not assumed: realact's shared-component norm is 0.0615 against
the Gaussian control's 0.0437. Applying our mu to her directions centres them twice.

### mu

`shared/mu/mu.f32`, ‖mu‖ 67.92575073242188, sha256 `80f3f995…`. It is a **vector**,
5,120 float32. Precisely: the mean of the layer-42 block-output residual over every
non-sink position of every 64-token window at stride 16 within documents of the 16M
corpus, clean base, nothing injected. At stride 16 an interior token appears in four
windows, so it is window-position-weighted, not token-uniform.

The definition matters ~20× more than the estimate: against two other defensible means
it sits at cosine 0.977 and 0.978, and that ~2% difference is worth 0.04–0.06 of score.

### Scoring

One path, `common.score_tokens`, for MAEMM rollouts, corpus windows, GCG strings,
Patchscopes continuations and SAE-repo windows alike. Re-tokenised alone, sink
prepended and excluded, fp32, uncentred, no norm filter, max over generated tokens.
The norm filter exists as an option and changes nothing to four decimals on this base.

Per-row cosines move by up to ~1% between a one-row and a batched scoring call (a kernel
switch), so every stored score is taken at one fixed batched shape.

### Independence

**122 of 512 realact targets share a document with another target** (449 distinct
documents, up to 3 each). An SE over 512 independent targets is too small. Cluster
bootstraps by `doc`; `doc_n_targets` is on every row.

### Held-out kinds

Three different things are called "held out" and they are not equally strong:

* `feature_id` — a partition of feature ids, verified against the training banks
* `doc_range` — disjoint document ranges, 0 intersection, margin 396,399 documents
* `category` — "the model never saw this kind of direction", asserted from the training
  mix, never measured. §3.5, §3.6 and §3.8 rest on this one.
* `in_distribution` — NOT held out. `indist_*` families must never enter a held-out mean.

Over the 13 direction families: 6 unknown, 3 in_distribution, 2 feature_id, 1 doc_range,
1 category. **Only 3 of 13 carry a verified claim.**

---

## 4. `shared/` — the agreed artifacts

```
shared/mu/                mu.f32, mu.meta.json, README.md
shared/sae2m-2k/          features.parquet|csv, meta.json, README.md
shared/ngram-overlap/     hers_n7.jsonl, hers_n7.exclude.json, doc_level_n7.jsonl
```

**`sae2m-2k`** is the agreed 2,000-feature subset: drawn from Celeste's seed-2026 `eval`
split (100,000 features), 1,601 train / 399 test, ~400/100 per corpus-peak quartile.
Drawing from the eval split is load-bearing — the checkpoint trained on the `sft`
(1,847,152) and `rl` (150,000) sides, so a draw over all 2^21 would put trained-on
features in the test set with nothing erroring.

**The train/test column splits OUR analysis, not the model's training.** Both halves
are equally unseen by the MAEMM. Use `train` to pick thresholds, strata and ablations;
report from `test`.

**`ngram-overlap`** is the contamination measurement (section 6).

---

## 5. Target sets

| set | families | note |
|---|---|---|
| `2026-09-16_v1` | realact 0–511, random 512–1023, sae 1024–1535 | the sae rows are the **131k** SAE and are superseded |
| `2026-09-20_sae2m_2k` | sae ×2000 | the shared 2M subset. Rows on the volume carry `family: "sae2m_enc"` because the set was built before the tag fix; consumers accept it at read time |
| `2026-09-21_sae131k_2k` | sae ×2000 | the 131k counterpart, 1,578 train / 422 test |

`ids.jsonl` rows carry `row, family, id, stratum, side, heldout_kind` and, where the
bundle ships provenance, `doc, pos, act_norm, doc_n_targets`. **`family` is a selector**
— `scan`, `top1_act`, `repo_examples`, `gcg`, `score`'s per-family means and autointerp's
`sae_self`/`build` all filter `family == "sae"`. A row tagged anything else is invisible
to them. `id` alone is ambiguous across dictionaries, so rows also carry `sae_key`.

### Corpora

`corpus/` is the original 16M-token held-out slice (Ultra-FineWeb parts 0009–0010),
64-token windows at stride 16, nested 1/2/4/8/16M.

`corpora/train_parity_10m/` is **built from the checkpoint's own training document
ranges** (sft 5,500,001–5,698,523 and rl 9,500,000–9,599,841): 10,004,614 tokens over
11,809 documents, nested 1.25/2.5/5/10M, **32-token windows at stride 8**.

The point is parity — search and the generator see the same data, and the budget matches
training, which saw 9–10M activations. **The cost is that the search can now retrieve
text the model may have memorised, so a corpus win on this corpus is NOT evidence that
retrieval beats generation on unseen text.** Two different claims; say which.

Nothing from this corpus is comparable with a number from `corpus/`: different
documents, different ladder, 32-token windows against 64. A new ladder or geometry is a
new corpus, never an edit, because every stored scan window indexes into one exact
`tokens.i32`.

---

## 6. Contamination

Tomáš asked (2026-09-20) whether realact test samples sit too close to training data.
Two checks, both at 7-grams (NFKC → lowercase → whitespace split), against **all 8.94M
training rows** (49.0M distinct 7-grams).

**Span level** — does the shown target text appear in training?

```
512 targets   any 37   >=0.05 26   >=0.20 11   >=0.50 7   >=0.90 3   max 1.000
p50 0.000   p90 0.000   p99 0.618
```

**Document level** — did the model see the *document*, not just the span? (The source
document is recovered by exact normalised substring match; 493 of 512 resolve.)

```
493 resolved   any 344   >=0.05 27   >=0.20 9   >=0.50 1   >=0.90 0
p50 0.0022   p90 0.0184   p99 0.303
```

Read them together. "Any overlap" jumps at document level because a document has ~60×
more shingles and ordinary English recurs — that number is noise. The serious thresholds
*fall*, because coverage is a fraction and a memorised 40-word span sits inside a mostly
original document. **The diffuse whole-document contamination that would have forced a
redraw does not appear.**

Verdict: a small, sharply separated tail — 9 to 27 targets of 512 depending on the
threshold. **Exclude, do not redraw**: dropping documents renumbers `doc` ids and
invalidates every stored scan window and top-k list. Lists are in
`shared/ngram-overlap/`.

Caveats: 19 targets did not resolve to a document; document-level coverage is diluted by
length, so for "did the model see part of this document" the ≥0.05 count (27) is the
relevant one, not ≥0.20 (9).

---

## 7. Results so far

**§3.2, activation families**, checkpoint `a1e4f299`, set `2026-09-16_v1`, 1,024 targets
× 64 rollouts:

```
              bo1      bo64        bo1 centred   bo64 centred
realact     0.4712   0.5599          0.7318        0.8523
random      0.0338   0.0468          0.0499        0.0686
```

Corpus search at 16M on the same targets is 0.411, so the generator beats a full scan.
Note this is **slightly below** the previous primary checkpoint (0.499 / 0.569 on the
same targets) — confounded by engine (HF here, vLLM there, ~0.002 at bo64) and possibly
real, since this checkpoint was rewarded on the last 16 tokens rather than the whole
span. A paired per-target comparison is cheap now that both score directories exist.

**Patchscopes, 27B screen** (three cells at layer 14, 1,536 targets, n=8):

```
cell                        realact  realact_bo8  sae_bo8
P2 / replace / alpha 2       0.0719     0.1379     0.0209
P2 / add     / alpha 1       0.0690     0.1311     0.0183
P1 / identity / replace 2    0.0458     0.1038     0.0162
random floor                 ~0.021     ~0.030        —
```

The configuration inherited from the 8B screen is confirmed correct for the 27B. Note
the SAE column sits **below the random floor** — consistent with 2M features being
largely orthogonal to realact directions.

### The 2M SAE sits at the random floor

2,000 features each, same checkpoint, same scorer, n=4:

```
                  mean      bo4
131k SAE        0.1116    0.1292
2M SAE          0.0311    0.0373
random floor    0.0338    0.0387
```

The generator does **no better on 2M features than on Gaussian noise**, while the 131k
dictionary is 3.6x higher and clearly above the floor. Three independent methods now
agree on this: Tomas's autointerp smoke has every arm at chance *including the corpus
reference*; the Patchscopes 27B screen puts 2M SAE at 0.016-0.021 against a 0.030 floor;
and this.

Two readings, not yet distinguished: either the 2M dictionary's features are genuinely
not semantically describable (the explainer calls them "highly predictable continuation
tokens / function words"), or something in how we READ the 2M SAE is wrong -- which
would rhyme with the unresolved 131k max-acts discrepancy in section 1. The cheap
discriminator is `repo_examples` on the 2M SAE: score its own shipped max-activating
windows with our encoder. Near the floor means the dictionary is the finding; high means
our reading is broken. **Run that before spending on 2M autointerp.**

### Trojans at layer 42

**theme: 16/16 installed**, all with verbatim payload,
control firing 0.00–0.05. Better than the layer-40 run in the draft (15/16 exact).

**SEP: 2/8 installed, and only 8 of 16 trained** (the run ended without the other 8 and
the log was lost -- rerun with output kept before concluding anything). Two failure
modes: three under-fire on their own trigger (0.50-0.75), and two fire perfectly but
leak badly on non-trigger prompts (control 0.57 and 0.48), which is the worse one -- a
trojan that fires without its trigger is not gated at all.

The cause is structural. All 16 SEP trojans share the SAME frames and differ only in
three digits, and the clean negatives deliberately include the other codes, so a rank-1
read vector must separate "864" from "394" inside an otherwise identical sentence
through a single scalar projection. The theme family gives each trigger 16 sentences
about its own domain, so `a` has far more to latch onto. Consistent with the paper's
0/16 SEP code recovery -- but that was a READOUT failure and this is an INSTALLATION
failure, which is new.

---

## 8. Things that will bite you

**The deployed Modal image is a snapshot of a checkout, not of a branch.** Two failures
on 2026-09-20 were a stale image presenting as a code bug. Redeploy before spawning;
`features/pipeline.py` does it for you.

**`modal run` cannot carry a long product.** It holds a blocking `.remote()` open; on a
multi-hour job the client JWT expires mid-call, raises `AuthError: Jwt is expired`, **and
the container dies with it**. Use `features/spawn.py` (deploy + spawn detached).

**The 2M SAE OOMs an H200** if you load both weight matrices in fp32 — 43 GB each beside
the 27B's 52 GB against 141 GB. Load encoder-only (`need_decoder=False`); nothing on the
scan/score/stats path reads `W_dec`.

**A base can carry more than one SAE.** `stats`, `scan` and `score` all asserted exactly
one and died before the model load. Pass `--sae`.

**`--force` takes the old product out of service before the new one lands.** A long
`stats` run leaves `stats.tmp-<date>` and `stats/` is simply gone until it finishes.

**`--root` relocates inputs as well as outputs**, so a scratch root has no held-out set.

**`scan` needs `stats` first** for an SAE set — it reads `sae/<sae>/max_act.f16` and
fails ~40 minutes in without it.

---

## 9. Running things

```bash
MODAL_PROFILE=maemm modal deploy precompute/modal_app.py

# one stage
python -m features.spawn --product rollouts_hf --base qwen36-27b \
    --maemm qwen36-27b/2026-09-18_rl-last16-lr5e-7 --heldout <set> --n 4
python -m features.spawn --poll <fc-id>

# a whole chain, with the ordering enforced
python -m features.pipeline --set <set> --sae qwen36-27b/sae2m --n 4
python -m features.pipeline --status
```

Products: `check` (free, run it first), `corpus`, `stats`, `targets`, `draw_sae2m`,
`draw_sae131k`, `scan`, `rollouts_hf`, `rollouts_vllm`, `rollouts_nla`, `score`,
`centred`, `patchscopes`, `top1_act`, `repo_examples`, `parity_greedy`.

Measured costs on H200 at $4.54/h: rollouts 1,024×64 ≈ $9; score ≈ $0.5; scan at 2,000
targets ≈ $4; `stats` over 2M features ≈ $43 and 9.4 hours; the autointerp LLM run ≈ $92.

**Engine note:** vLLM measured **5.7× slower than HF** on the 8B because the stock lens
hook rescans every steering key on every layer of every decode step. On the 27B HF runs
at ~440 gen tok/s. Do not assume vLLM is the fast path.

---

## 10. Open, as of 2026-09-21

* **Scorer convention** is not settled. Three are live and they differ by ~0.26 on
  realact. Every number in section 7 is in the pipeline's primary convention.
* **`whiten_mu`** (Celeste's) is not in the bundle. We can inherit her centring but not
  reproduce or audit it.
* **The 131k max-acts disagreement** (section 1) is unresolved. The decisive test is
  CPU-only: compare `b_dec · W_enc[:,f]` against the measured per-feature intercepts.
* **Autointerp on the 2M SAE may have no signal.** Tomáš's 8-feature smoke has every arm
  at chance *including the corpus reference*, with the explainer describing them as
  "highly predictable continuation tokens / function words". Extend that smoke before
  committing the $92 run.
* **`realact_early/mid/long`** are near-orthogonal to `realact` (median −0.007), so they
  are different activations, not context variants, and ship no provenance.
* **SEP trojans do not install reliably at layer 42** (2/8), and 8 of 16 never ran.
* **The v3 target sets** (`2026-09-21_v3_*`, drawn 2026-09-21) are the team's current
  family and are stratified on gated fire counts from our own 16M scan, enc+dec, via a
  `heldout_v3` product. They supersede `2026-09-20_sae2m_2k`, which was drawn earlier on
  Celeste's 1.0B corpus peaks because the stats pass had not finished. See INVENTORY.md.
* **§3.4, §3.5, §3.6, §3.8** live outside this pipeline in separate codebases. §3.6's
  trojans are on the **`maemm-trojan-cache`** volume, not this one.
* **`paper/<section>/` statuses are hand-maintained** and go stale. Regenerate with
  `python -m features.paper_index`.
