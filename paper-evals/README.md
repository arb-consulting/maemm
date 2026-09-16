# paper-evals

Precompute for the MAEMM evaluation paper: corpus, held-out target sets, corpus scan, MAEMM
rollouts, scoring, GCG, and the tables built from them. Independent of `eval/` and `rl/` in this
repo — nothing here imports Celeste's modules — but the *conventions* below are copied from them
verbatim so the numbers are comparable. Spec: `infra/precompute-layout.md` (data layout §1, code
layout §2, decisions §5) and `infra/precompute.md` (products) in the mimir project.

Status: `config.yaml`, `common.py`, `modal_app.py`, `unit_smoke.py`, the four base products
`corpus`, `stats`, `targets`, `scan`, the two MAEMM products `rollouts_hf` and `score`, and
`reconstruction/repro_run1.py` are implemented and have been run end to end on BOTH bases at smoke
scale (2M corpus tokens on the 8B, 1M on the 27B; 48 x 64 rollouts on the 8B, 8 x 64 on each 27B
MAEMM). Step 4 adds `mu_check`, `rollouts_vllm`, `parity_greedy` and `reconstruction/parity.py`,
all run on both bases and on all three MAEMM shapes (8B LoRA, 27B LoRA, 27B full). **No product
stubs remain.** `repo_examples` is built and has been run on both bases at smoke scale (512 sae
features x 30/32 shipped windows). Step 5 adds `gcg/` -- its own module (`gcg/gcg.py`) and its own
Modal app (`gcg/modal_app.py`, on precompute/'s image) -- run on the 8B for all four arms and on the
27B for the two `gcg` arms; the scoring forward it shares with everything else was refactored out of
`common.score_tokens` into `common.score_ids` on the way. `lens_floor/` is declared but not built.
Every verified run is logged in `SMOKES.md`, with the cost taken from each product's own README
(checklist item 84: `modal app logs` replays stale output).

**The FULL RUN happened on 2026-09-16** and its products are the ones under the real root `/vol`
(`SMOKES.md`, "Full run 2026-09-16"): the 16M corpus, `stats`, held-out set `2026-09-16_v1` and
`scan` on both bases, `repo_examples` on both, and 1,536 targets x 64 rollouts + `score` +
`centred` for all three computable MAEMMs -- 8B `2026-09-03_run1-rl` through `rollouts_hf`, 27B
`2026-09-08_rlI-150` and `2026-09-10_rl-8x2048-full` through `rollouts_vllm`. Total $45.66. Two
things changed underneath it: the GPU function timeouts went 6 h -> 10 h, and mid-run the vLLM path
gained decode-only CUDA graphs plus Celeste's fast hook as `precompute/vllm_ext.py`, which took the
27B from 4.25 to 23.9-29.3 rollouts/s and is validated by a paired 8B HF-vs-vLLM parity at
per-direction r = 0.9964 / 0.9932 on realact.

## config.yaml

One file, read by everything, validated on load (`common.load_config`):

| key | what it holds |
|---|---|
| `bases` | `qwen3-8b` (Qwen/Qwen3-8B, read layer 27, d 4096, H100, 36 layers), `qwen36-27b` (Qwen/Qwen3.6-27B, read layer 42, d 5120, H200, 64 layers) |
| `saes` | `qwen3-8b/adamkarvonen-t2`, `qwen36-27b/l42-1b` (the **1b** repo, not the older `-l42`); each carries a `max_acts` block (file, `windows`, `sink_first`, and `repo`/`repo_type` when the windows are not in the SAE repo) for `repo_examples` |
| `maemms` | 8B `2026-09-03_run1-rl` (lora, volume path); 27B `2026-09-08_rlI-150` (lora, `step_150/`), `2026-09-10_rl-8x2048-full` (full), `2026-09-05_rlE-250` (lora) |
| `heldout` | `2026-09-16_v1`: seed 20260916, families `realact`/`random`/`sae` 512 each + `jlens` 512 (27B only, currently an **empty slot** — no J-lens matrix exists) |
| `corpus` | Ultra-FineWeb `en` parts 0009 and 0010, 16M tokens, nested sizes 1/2/4/8/16M, seed 20260916 |
| `rollouts` | n 64, T 1.0, top_p 1.0, top_k 0, max_new 64, min_new 16, seed 1234 |
| `modal` | volume `maemm`, `HF_HOME=/vol/hf`, archive `/vol/archive/gavento-1`, secret `hf-write` |

Config keys `<base>/<name>` under `saes` and `maemms` **are** the directory names on the volume.
Families are keyed by name everywhere, never by list position.

## Volume layout (`maemm` at `/vol`)

```
/vol/
  archive/gavento-1/{data,runs}/      old volume, verbatim; read-only by convention
  hf/                                 HF cache (HF_HOME); everything fetchable by id lives only here
  base/<base>/
    corpus/        tokens.i32, docs.jsonl (doc, offset, len, size_tag, source part/row)
    heldout/<set>/ ids.jsonl (row, family, id, stratum, ...), vecs.f16 [N, d],
                   mu_512.f32 (DIAGNOSTIC ONLY -- see "Methods"), leakage.jsonl
    stats/         mu.f32 (THE centring mean), mu_by_size.f32, resid_norm_quantiles.json
    scan/<set>/    topk.jsonl (target row, size -> 64 x (doc, start, argmax, cos)), quantiles.f16
    sae/<sae>/     fire_counts.i64 [F, n_sizes, 2], max_act.f16 [F], mean_when_active.f16 [F],
                   sizes.json, examples/{<feature>.jsonl, _random256.jsonl, tested.json}
    sae/<sae>/repo_examples/<set>/  repo_examples.jsonl (one row per tested feature x shipped
                   window), per_feature.jsonl (the `sae-repo-top32` baseline column), summary.json
    gcg/<set>/<family>/<arm>/  finals.jsonl, trajectory.jsonl, top64.jsonl, summary.json;
                   family in {realact, sae}, arm = <mode>-<init> in {gcg,epo} x {random32,corpus}
    lens_floor/<set>/ PLANNED, NOT BUILT: top-k logit-lens tokens of each target direction
                   (`unit(d) @ W_U`), those token strings scored ALONE through score.py — the
                   floor a trivial "just name the direction's tokens" baseline reaches
  maemms/<base>/<maemm>/
    README.md      source id / archive path, type, adapter subdir, injection convention, weight sha
    rollouts/      <set>.jsonl (one row per target x rollout: row, family, k, text, ids, n_tok,
                   finished, engine, seed) + <set>.summary.json + README.md + index.json. This
                   directory ACCUMULATES sets (OutDir keep_existing); its README describes the most
                   recent run and index.json lists every set present.
    scores/<set>/  cos.f16 [N, n, 96], norm.f16 [N, n, 96], argmax.i16 [N, n],
                   best_act.f16 [N, n, d], sae_idx.i32 / sae_val.f16 / sae_off.i64 (flat CSR),
                   per_target.jsonl, rows.json
    scores/<name>__rescore-<x>/  the --rescore-texts variant: FLAT cos.f16 / norm.f16 [M, 96],
                   argmax.i16 [M], rows.jsonl; no best_act, no SAE
```

`maemms/` is `--root`-relative since step 3, exactly as `base/` is: a rollouts/score smoke writes
its whole `maemms/<base>/<maemm>/` subtree under `/vol/runs/<date>_paper-evals-smoke`.

Base key spelling is `qwen36-27b`. An older `/vol/maemms/qwen3.6-27b/` tree exists from a previous
pipeline; it is not ours and is not touched.

Rules (`infra/design.md` §1, implemented by `common.OutDir`): one `README.md` per output directory
written by the producing script (command line, date, repo commit, inputs, sizes, status,
provenance); write to `<name>.tmp-<date>/` and rename on completion; refuse to overwrite an
existing directory without `--force`; no registries, no sidecar hashes, no `meta.json`. jsonl for
small things, raw arrays for matrices with the shape in the README **and** in `index.json` (the one
allowed sidecar, so an array's length is never inferred from its file size).

## Products

| product | where | writes | notes |
|---|---|---|---|
| `check` | CPU | — | resolves every path, builds both prompts; the cheap gate before any GPU run |
| `unit` | CPU | — | `unit_smoke.py` inside the image |
| `corpus` | CPU | `base/<base>/corpus/` | the ONLY product that reaches the network |
| `stats` | GPU | `base/<base>/stats/`, `base/<base>/sae/<sae>/` | pass A: mu, per-layer norm quantiles, SAE fire statistics |
| `mu_check` | CPU | appends to `base/<base>/stats/README.md` | our `stats/mu.f32` vs Celeste's archived `whiten_mu.npy`: cosine and norm ratio, reported never acted on |
| `targets` | GPU | `base/<base>/heldout/<set>/` | the frozen draw: realact / random / sae (+ the empty `jlens` slot) |
| `scan` | GPU | `base/<base>/scan/<set>/`, `sae/<sae>/examples/` | pass B: corpus-retrieval top-64 + cos quantiles per size, SAE examples |
| `rollouts_hf` | GPU | `maemms/<base>/<maemm>/rollouts/` + that MAEMM's `README.md` | n rollouts per target through HF `generate` with the direction injected at the marker; the ONLY product that loads a MAEMM |
| `score` | GPU | `maemms/<base>/<maemm>/scores/<set>/` | per-token cosine / norm on the CLEAN BASE, the argmax residual, the gated SAE features there, and the per-target aggregates. NEVER loads the MAEMM |
| `rollouts_vllm` | GPU | `maemms/<base>/<maemm>/rollouts/` (stem `<set>__vllm`), `…/throughput/` | the same rollouts through a vLLM engine, one request per target with `n` samples; same row format, `engine: "vllm"`. `--throughput` measures generation tok/s instead |
| `parity_greedy` | GPU | `maemms/<base>/<maemm>/parity/greedy-<set>/` | 8B only: the HF hook and the vLLM steering on the SAME greedy decode, plus the teacher-forced logprob gap and an unsteered control |
| `repo_examples` | GPU | `base/<base>/sae/<sae>/repo_examples/<set>/` | the SAE repo's OWN shipped max-activating windows for the tested features, scored through `common.score_tokens` exactly like a rollout: the `sae-repo-top32` baseline column, plus the repo-vs-us activation agreement |
| `gcg` | GPU | `base/<base>/gcg/<set>/<family>/<arm>/` | discrete-token search on the scorer's own objective -- the reachability ceiling a text of T=32 tokens gets to, against which a MAEMM rollout is read. Its own Modal app (`gcg/modal_app.py`), one call per (base, family, arm). NEVER loads a MAEMM |
| `mu_diag` | GPU | `base/<base>/stats/mu_diag/` | WHY `mu_check`'s two means disagree: Celeste's 512-token / no-sink / all-position geometry recomputed on OUR corpus, plus the position mix, the massive-activation tokens and the split-half sampling noise of each geometry. One forward pass per geometry; `--tokens` caps the corpus walk (default 500k) |
| `top1_act` | GPU | `base/<base>/sae/<sae>/top1_act/<set>/` | for every sae held-out feature, the pre-gate activation of the feature on its cosine-selected corpus top-1 window (scan rank 0 at 16M): join against the activation-ranked examples where present, and ONE forward of all 512 windows (scan geometry, clean base) as the checked path; feeds `reconstruction/corpus_top1_activation.py` → `paper/inversion-eval/data/corpus_top1_activation.csv`. Measured 2026-09-16: 27B 512/512 windows pass the gate (median act/gate 13.6, Spearman(cos, act) 0.94), 8B 508/512; ~$0.25 / $0.09 |

Each product refuses to overwrite its output directory without `--force`, writes its own
`README.md` + `index.json` (command, date, commit, inputs, shapes, sizes, wall, **cost**, status),
and commits the volume when it is done.

### Conventions these products fix

- **Scan geometry 64/16** (`common.windows_of`, checklist item 57): 64-token windows starting every
  16 tokens inside a document, never crossing documents. A document of ≤ 64 tokens is one window;
  after the last full window, ONE partial window is appended at `last_start + 16` running to the end
  of the document (so it is 49-63 tokens). Every token is covered. The old retrieval baseline used
  stride 32 (`eval/corpus_retrieval.py:89-91`); results are geometry-dependent, so ours is stated
  everywhere it is used. `stats` and `scan` call the same helper, so their window ids agree.
- **The realact recipe is NOT that geometry.** `targets` follows Celeste
  (`data/build_universal_bank.py:295-314`, checklist item 30): the document's first **512** tokens,
  `add_special_tokens=False`, **no sink token**, `p ~ U[16, 512)`, shown span `L ~ U[16, 64]`,
  vector `unit(X[p] − mu)` with `mu` = `stats/mu.f32` (Celeste centres on her own 512-window mean
  instead; see "Methods"), `span_text = decode(toks[p−L+1 : p+1])`, raw-norm filter (> 1e-3 and
  ≤ 10× a 4096-position presample median) applied **at selection only** (checklist item 34).
- **One centring mean** (Tomáš, 2026-09-15; supersedes the earlier two-means rule, checklist item
  77): see "Methods" below. `heldout/<set>/mu_512.f32` is still written, as a diagnostic only.
- **Cosine** is uncentred and fp32 over every non-sink position, with no norm filter — while realact
  targets are centred once, at construction. That asymmetry is Celeste's and is deliberate.
- **Nested sizes**: documents are permuted once by `corpus.seed` and tagged 1/2/4/8/16M by
  cumulative token count, so subset *k* is a prefix of `tokens.i32`. `stats` and `scan` walk
  documents in that order and snapshot at each crossing, so every per-size number comes from one
  pass.
- **Rollout seeding** (`common.gen_seed_for`). Celeste forks the RNG once per eval
  (`torch.random.fork_rng` + `torch.manual_seed(GEN_SEED)`, `eval_universal.py:804-805`), so a
  row's sample depends on every batch before it. HF `generate` takes no `torch.Generator` on every
  version, so instead each generate call is seeded `rollouts.seed * 1000 + flat` immediately before
  it, with `flat = row * n + k` of its FIRST row, and the seed is stored on every row it produced.
  Rollouts are therefore reproducible per chunk but NOT bitwise Celeste's.
- **Rows per generate call**: 256 on the 8B (= 4 targets x 64), **32** on the 27B (checklist item
  51: HF generate over the GatedDeltaNet layers is >= 37x slower at batch >= 64). Every row of a
  call carries the identical prompt, so the batch is rectangular and UNPADDED -- asserted, because
  padding would move the marker.
- **Marker-norm check** (checklist item 22) is OBSERVATION ONLY: extra forwards of the single
  shared prompt *before* generation, never hooked into the generation path, logged into the
  summary and asserted to differ by > 5%. LoRA takes the clean-base number from the same object
  with the adapter disabled; a FULL model has no adapter to switch off, so it reads
  `bases.<base>.marker_norm_base` from `config.yaml` (measured by an earlier LoRA run on that
  base) and otherwise skips the comparison with a note rather than paying a second 52 GiB load.
- **Best-of-k** in `per_target.jsonl` is the plain disjoint-group estimator
  (`common.best_of_k_means`): floor(n/k) consecutive groups of k, group max, mean of the maxima.
  The unbiased order-statistic estimator belongs in `reconstruction/stats.py`, not in the product.
- **SAE features per rollout** are stored at the ARGMAX TOKEN ONLY, gated, as a flat CSR. The
  per-token variant is ~50x bigger (~510 MB at the full 512-target set on the 8B, MEASURED from
  the smoke's mean 81.5 gated features and 32-token rollouts) and is deliberately not written.
- **Scored text excludes the prompt** (checklist item 8), asserted two ways in `score.py`: the
  stored generated ids are at most `max_new` long, and no scored row reaches the 95-token
  truncation (the ~103-token prompt would). MEASURED 2026-09-15: a decoded rollout does NOT always
  re-tokenize to the same number of ids (a 64-token rollout cut mid-word re-encodes to 65), so
  `kept <= max_new` is NOT a valid bound and the text round trip is reported as a rate, not
  asserted.
- **SAE gate**: "fired" is the checkpoint's learned BatchTopK `threshold` (6.936 for the 8B
  adamkarvonen trainer_2 SAE, **1.5846** for the 131k 27B `l42-1b` one, MEASURED by every
  `stats` run that loads it; the 1.654 this file carried until 2026-09-16 is not the value in
  `l42-1b`'s checkpoint and its provenance was not established here), not raw act > 1.0. `fire_counts.i64`
  stores both `> 0` and `> gate`; the ≥ 20-fires eligibility uses the gated count.

## Methods

**Centring: one rule, everywhere.** A `realact` target is `unit(X[p] - mu)` with `mu` =
`base/<base>/stats/mu.f32`, the read-layer mean over all scanned positions of the 64/16 windows of
pass A, sink excluded. That subtraction happens exactly ONCE, in `targets`, when the direction is
built. After that:

- every cosine against a target is **uncentred** and goes through the single `common.score_ids`
  — rollouts, `scan`, GCG, the SAE repo examples, and any reference score. `score_tokens` is
  `encode_for_score` + `score_ids`, i.e. the text path is the ID path with a tokenizer in front;
  GCG optimises ids and reaches the same forward without one. There is no second cosine
  implementation and no centred variant inside a product.
- the `sae` and `random` directions are **never** centred at all (an encoder column and a Gaussian
  draw have no mean to subtract).
- `targets` FAILS if `stats` has not run: no product computes its own mean as a fallback.
- `heldout/<set>/mu_512.f32` — Celeste's mean over the 512-token no-sink windows the realact draw
  forwards (`data/build_universal_bank.py:310`) — is still computed and stored, but ONLY as a
  DIAGNOSTIC. Each held-out README reports its cosine and norm ratio against `stats/mu.f32`;
  MEASURED 2026-09-15 at smoke scale: 8B cos 0.9900, `||mu_512||/||mu||` 1.0422; 27B cos 0.9774,
  ratio 0.9898.
- the **centred** number, `cos(best_act - mu, target)`, is a SECONDARY statistic computed later
  from the stored `scores/<set>/best_act.f16` and `stats/mu.f32` in `reconstruction/stats.py`. It is
  not a product output and no product's cosine is centred.

`mu_check` puts `stats/mu.f32` against Celeste's archived `whiten_mu.npy` (8B
`data/run1/acts/whiten_mu.npy`, 27B `data/qwen3.6-27b/whiten_mu.npy`) and appends the cosine and
norm ratio to the stats README. It reports; it does not choose.

**Why `stats/mu` is the mean (decided by Tomáš, 2026-09-16).** The paper must be self-contained:
the centring mean is the mean of the published held-out corpus under the stated window geometry
(64-token windows, stride 16, sink excluded), computable by anyone from `corpus.py` + `stats.py`,
with no dependence on a training-time statistic that lives in an unpublished activation store.
The cost of that choice is known and stated rather than hidden: the 27B MAEMMs were trained
against targets centred on Celeste's 512-token/no-sink mean, and re-centring with `stats/mu`
lowers their realact cosine by 0.04-0.06 (SMOKES.md, "What the centring change did"). Footnote for
the paper, from `mu_diag` (SMOKES.md, Step 6): the two means differ by geometry alone -- her
512/no-sink/all-position mean recomputed on OUR corpus reproduces her archived vector at cos
0.9997 (8B) / 0.9999 (27B); against `stats/mu` it is cos 0.9900 / 0.9774, of which the position
mix (early positions over-represented by 64/16 windows) accounts for ~27% and massive-activation
tokens for none; split-half sampling noise of each mean is cos >= 0.9987.

**Standing rule (Tomáš, 2026-09-16).** Every convention here is chosen toward a self-contained
paper: no unpublished external data, no hard-to-justify parameters. Where our convention and
Celeste's training-time convention differ, OURS is canonical and the difference is stated in this
README (the centring mean above; the 64/16 scan geometry vs her 512-token collection windows;
no norm filter in the stored scores; the SAE gate read from the checkpoint; fixed T=32 for GCG).

## The vLLM engine path (`rollouts_vllm`, `parity_greedy`)

`rollouts_vllm` writes the same rows `rollouts_hf` does, with `engine: "vllm"`, into the same
accumulating `rollouts/` directory under the stem `<set>__vllm` (`common.rollout_stem`), so
`score --engine vllm` scores them with the identical clean-base scorer and
`reconstruction/parity.py` can put the two side by side.

**How the hook is installed.** vllm-lens 1.1.0 ships a general plugin that patches
`EngineArgs.create_engine_config` to install a worker extension and to force `enforce_eager`.
`build_engine` loads the plugins explicitly (`vllm.plugins.load_general_plugins()`), asserts the
plugin registered (`vllm_lens._activations_plugin._original_create_engine_config is not None` —
without it steering would silently do nothing), then sets `worker_extension_cls` itself and calls
`llm.collective_rpc("install_hooks")`, exactly as `rl/rl_disagg.py:1337-1373` does. The DEFAULT
here is the **stock** `vllm_lens._worker_ext.HiddenStatesExtension`, which MEASURED 2026-09-15
passes every self-check on the 8B. Celeste forces her `fast_lens_ext` instead, but its reason is
throughput at 256+ concurrent requests (the stock hook rescans every steering key on every layer of
every decode step); `--fast-hook` selects `precompute/vllm_ext.py`, a copy of hers with the
`try/except` around the hook body REMOVED so a hook error raises instead of leaving a request
silently unsteered — but **that file is not in the tree**, because nothing needed it; `--fast-hook`
therefore fails with an assertion telling you to copy it in. The stock extension exposes no error
counter, so self-check (iv) reports `n/a` on that path.

**Steering.** `SteeringVector(activations=unit(v) * ||h_marker|| * coeff, layer_indices=[1],
scale=1.0, norm_match=False, position_indices=[marker])`, one per request via
`SamplingParams.extra_args["apply_steering_vectors"]`. `norm_match` must stay **False**: vLLM's
decoder layer returns a SPLIT residual `(hidden_states, residual)` and lens' own norm matching would
scale by the hidden_states component alone. `||h||` is read from the ENGINE (a greedy 1-token
request with `output_residual_stream`), not from an HF forward — the 27B cannot hold both.

**The LoRA rename rule.** vLLM validates adapter module names by **suffix only**, so an adapter
named for `AutoModelForCausalLM` passes validation and is then SILENTLY IGNORED; the engine samples
from the base model (rl/rl.py:192-211 measured a 1.47-nat logprob gap). The 27B is served as
`Qwen3_5ForConditionalGeneration`, whose mapper only knows `model.language_model.`, so its adapter
is re-saved with `model.layers.` → `model.language_model.layers.` (first occurrence only,
`common.rename_lora_keys`, unit-tested). The 8B is a plain CausalLM and is **not** renamed —
renaming it would break the lookup the other way.

**Self-checks, run in every `rollouts_vllm` invocation and stored in its summary.**

1. *Injection*: a clean, a SECOND clean, and a steered greedy 1-token request, all with the inject
   layer captured. The marker row's steered delta must have cos > 0.99 with `unit(v)` and a
   magnitude ratio in (0.95, 1.05) against `coeff * ||h||`. Rows before the marker must not move —
   but "not move" is measured against the engine's own noise floor, which is what the second clean
   request is for: MEASURED 2026-09-15, two identical clean 27B requests differ by O(1) at layer 1
   (its residual norms are in the hundreds), while the 8B gave exactly 0.0. The bound is the larger
   of 2% of the injected magnitude and 3x that measured clean-vs-clean delta; an injection at the
   wrong position moves a pre-marker row by ~100% of the injected magnitude, so this still catches
   it by a factor of 50. MEASURED on the 27B rlI-150: steered pre-marker delta **0.1948** against a
   clean-vs-clean delta of **0.1979** — the steering moves those rows LESS than re-running the same
   clean request does, i.e. not at all. The 27B FULL model, served without LoRA, gives 0.0 for both,
so that nondeterminism is the LoRA (Punica) path's, not the GatedDeltaNet layers'.
2. *Marker norm*: the engine's served `||h||` against `rollouts_hf`'s measured value for the same
   MAEMM, asserted within **3%**. For a LoRA MAEMM this is the adapter-applied proof — an ignored
   adapter returns the clean-base norm (8B 78.0 vs 14.5; 27B 512.0 vs 14.06).
3. *Not all identical*: every target's `n` rollouts must not decode to one string.
4. *Hook errors*: the extension's error counter must be 0 where it has one.

**MEASURED agreement, 27B, 8 realact directions x 64 rollouts**, both MAEMM shapes: rlI-150 (LoRA)
HF bo1 0.4360 / bo64 0.5496 against vLLM 0.4325 / 0.5478; rl-8x2048 (full) HF 0.4694 / 0.5430
against vLLM 0.4744 / 0.5408. Paired differences -0.0035 / +0.0051 on bo1 — inside the noise of
eight directions, which is the point. The LoRA path is the one that could have failed silently:
992/992 adapter tensors renamed, and the engine's marker norm came back 511.87 against HF's 512.0
where an ignored adapter would have given the clean base's 14.06.

**MEASURED agreement, 8B, 48 directions x 64 rollouts** (`SMOKES.md` has the full table): paired
per-direction r = 0.995-0.996 on the mean-of-64 in every family and 0.99 / 0.90 / 0.91 on the
best-of-64 (realact / sae / random), with mean paired differences of -0.0031 / +0.0012 / +0.0004.
Rollout lengths, eos rates and the argmax-position histograms match bin for bin. Both engines land
on run1's archived realact best-of-64 (0.5967): HF +0.0011, vLLM -0.0033.

**The GatedDeltaNet batch cliff does NOT exist on vLLM.** HF `generate` over the 27B's 48
GatedDeltaNet layers is >= 37x slower at batch >= 64, which is why `rollouts_hf` caps the 27B at 32
rows per call (checklist item 51). MEASURED 2026-09-15 on the rlI-150 LoRA engine, sweeping the
rows in flight inside one engine at `max_num_seqs` 128:

| concurrent rows | 32 | 64 | 128 |
|---|---|---|---|
| generated tok/s | 122.6 | 204.0 | **298.0** |
| per-row tok/s | 3.83 | 3.19 | 2.33 |

Throughput rises monotonically and the per-row rate degrades by only 1.64x across a 4x increase in
concurrency — ordinary batching behaviour, not a cliff. NOTE the substitution: the plan asked for
one fixed request set measured at `max_num_seqs` 32 / 64 / 128, i.e. three engines and three 52 GiB
loads; this varies the rows in flight inside ONE engine instead, which answers the cliff question
but does not measure how `max_num_seqs` itself (KV budget, scheduler) affects throughput.

**Throughput against HF, MEASURED and unwelcome.** On the 8B, vLLM with the stock lens hook generated **543
tok/s** against the HF path's **3,092** — 5.7x SLOWER for the same 3,072 rollouts. The stock
extension hooks every layer and rescans every registered steering key on every decode step, so the
cost grows with (layers x concurrent rows x distinct directions); 48 directions x 36 layers x up to
256 rows is its worst case and this pipeline's normal one -- and the 27B, with 8 directions, does
NOT show it (vLLM 246.8 tok/s against HF's 239.3 for the LoRA MAEMM). `--fast-hook` is the remedy
and has NOT been measured; its module is not in the tree. Treat vLLM here as a correctness cross-check, not as the fast path.

**`parity_greedy`** (8B only) is the direct engine-vs-engine check. It builds the HF model
**before** `import vllm` — importing vllm registers its vendored Qwen3_5 config with `AutoConfig`
and breaks `AutoModelForCausalLM.from_pretrained` for the 27B afterwards (rl/rl.py:1069-1072) — and
keeps it resident beside the engine at `gpu_memory_utilization` 0.55. MEASURED 2026-09-15 on 8
realact directions of `2026-09-03_run1-archive16`: engine `||h||` 78.261 vs HF 78.0 (0.33%);
injection cos 0.999993, magnitude ratio 0.999937, pre-marker delta exactly 0; first greedy token
matches on **8/8** directions with a first-token logprob gap of **0.0**; the greedy continuations
stay identical for a mean of 24.6 tokens and all the way to the end on 4 of 8; the teacher-forced
per-token gap (HF, hooked, on vLLM's own ids vs vLLM's returned logprobs) is **0.0184 nats** mean /
0.104 p99 over 323 tokens, against **1.961 nats** with the hook off. The injection changes the
greedy text on 8/8 directions. Greedy decode amplifies a bf16 tie-break into a different sentence,
so the 24.6-token match length with a zero first-token gap is numerics, not a protocol difference —
the teacher-forced number is the one that does not compound.

## The SAE repo's own examples (`repo_examples`)

The SAE authors ran their own scan over their own corpus and shipped, per feature, the 30-32
highest-activating 32-token windows. Those are the strongest natural text anybody has for a feature
without searching at eval time, so pushing them through `common.score_tokens` gives the `sae` family
an extra baseline column, **`sae-repo-top32`**, beside corpus retrieval and the MAEMM rollouts.
(The name is the 27B's window count. adamkarvonen's 8B file ships **30**, so each product README
also spells the column `sae-repo-top<its own count>` in its per-base note; `summary.json` carries
the canonical `"baseline": "sae-repo-top32"` on both bases. Nothing is padded to 32.)

`per_feature.jsonl` is the baseline table: `max_cos` IS that column (best of the shipped windows,
the direct analogue of a best-of-32 rollout draw), `mean_cos` its per-window mean, `mean_peak_act`
and `frac_fired` the feature's own activation on its own best windows (gate = the checkpoint's
learned `threshold`), and `peak_pearson_r` / `argmax_agree` the repo-vs-us agreement below.
`repo_examples.jsonl` is one row per (feature, rank): the shipped `ids` **as scored** and their
`repo_acts` aligned to them, `repo_peak`/`repo_argmax`, then OUR `cos` / `argmax` /
`our_act_at_argmax` / `our_peak_act` / `our_peak_argmax` / `fired` / `cos_at_repo_peak`, all from
the one scoring forward.

**Provenance, and what is NOT known.** The 8B windows are **not part of the SAE repo**: they come
from adamkarvonen's separate `adamkarvonen/sae_max_acts` **dataset** repo (30 windows per feature,
not 32), so `max_acts.repo`/`repo_type` in config.yaml point at it and `common.snapshot` resolves a
`datasets--` cache prefix. That file carries its own `config` dict, reproduced verbatim in the
product README, and it pins the SAE it belongs to: `sae_repo_id adamkarvonen/qwen3-8b-saes`,
`model_name Qwen/Qwen3-8B`, `sae_layer 27`, `sae_layer_percent 75`, width index 2 (= our
`trainer_2`), `context_length 32`, `num_tokens 60,000,000`. It does **not** name the scanned
corpus, and the dataset card is not on the volume (only the `.pt` was fetched), so the 8B scan
corpus is **UNKNOWN** here. The 27B windows ship inside `ceselder/qwen36-27b-sae-l42-1b` under
`maxacts/` and the file carries **no `config` key at all** — 32 windows of 32 tokens and nothing
else. The 27B repo card (reportedly Ultra-FineWeb `en`, the same corpus this pipeline scans) was
not fetched to the volume either, so that attribution is **unverified here**.

**The sink is handled per file and declared, not inferred** (`max_acts.sink_first` in config.yaml,
asserted against the data by `common.strip_repo_sink`, unit-checked). The 8B windows carry the
tokenizer's sink at position 0 — adamkarvonen's scan re-encodes the way `_reencode` does, and Qwen3
has no bos, so it is the eos 151645 — and that column is dropped from the ids AND from the shipped
activations before scoring, because `score_tokens` prepends its own. The 27B windows carry no such
prefix and all 32 ids are scored.

**Agreement, MEASURED 2026-09-16 (512 features per base).** Every window carries the repo's own
per-token activation, so this product also checks OUR activation formula and read layer against the
scan that produced the file. The 8B reproduces adamkarvonen's scan essentially exactly — mean
per-feature Pearson r of their peak against ours **0.9941** (median 0.9979), argmax position
agreeing on **99.74%** of windows, median peak ratio **1.0000**. The 27B does **not**: mean
per-feature r **0.5719**, argmax agreement **80.8%**, and our peaks a systematic **~30% higher**
than hers (median ratio 1.301, per-feature scale q10-q90 1.03-1.41 with a ~13% per-window residual
that no affine fit removes).

Every `repo_examples` run therefore also forwards its first 16 features' windows with NO sink
prepended, as a diagnostic that touches no output, and reports both r's. That diagnostic is what
rules the sink out as the 27B's explanation, and it validates itself on the 8B:

Both columns of this table are the SAME 16-feature subsample (480 / 512 windows), so they are
paired; the 27B's with-sink r there (0.7415) is higher than its 512-feature value (0.5719) simply
because those 16 features are not a representative draw — only the paired difference is the point.

| 16-feature diagnostic subsample | 8B | 27B |
|---|---|---|
| mean per-feature r, with the sink (as scored) | **0.9941** | 0.7415 |
| mean per-feature r, same windows without the sink | 0.7156 | 0.7539 |
| median (our peak / their peak), with / without | 1.0000 / 1.0172 | 1.301 / 1.293 |

On the 8B, taking the sink away costs 0.28 of r — the sink column is doing real work and our
handling of it is right. On the 27B it changes nothing (+0.012), and the 30% magnitude offset
survives both contexts. **OPEN**: what the 27B `maxacts/` file was actually computed with. The
feature indices clearly line up (pooled r 0.86, and a random pairing would give ~0), so it is the
same SAE run; a different training step of it, or a scan that scaled its activations, would produce
exactly this signature. The cheap test is to load whatever other checkpoint of that run exists
(e.g. `ceselder/qwen36-27b-sae-l42`, NOT currently on the volume) and compare `W_enc[:, f]` column
by column — pure CPU, no forward. Until then the 27B `sae-repo-top32` column is usable as a
baseline (it is OUR scorer on THEIR text, which is all the column claims) but the file's activation
values are not a reference for ours.

## GCG / EPO discrete search (`gcg`)

The reachability ceiling: what a 32-token string, optimised directly against the metric, gets to on
a direction — the number a MAEMM rollout has to be read against. `gcg/gcg.py` is
`eval/gcg_search.py` of our fork of ceselder/maemm (1288 lines), trimmed to this pipeline's data
layout and to ONE scorer; `gcg/modal_app.py` is its own Modal app (`maemm-paper-evals-gcg`) on
precompute/'s image chain, imported so the pins cannot drift.

**Objective**, per candidate string `x` of `T = 32` ids and unit direction `d`:

```
cos(x)      = max over KEPT positions t of cos(unit(h_t), d)      read layer, sink excluded
L_lambda(x) = cos(x) - lambda * nll(x)
```

`cos` goes through **`common.score_ids`** — literally the function `score.py` scores rollouts with
and `repo_examples` scores the SAE repo's windows with, reached without a tokenizer because the
search optimises ids. UNCENTRED, fp32, no norm filter, the sink prepended and dropped. So the loop's
number, the finals' number and a rollout's number are one number by construction; the
end-of-direction CHECK block below proves it per run rather than assuming it.
`nll` is the mean per-token NLL of the string's own ids under the clean base (teacher forcing, no
prompt, no sink, predict `ids[1:]` from `ids[:-1]`) through a hand fp32 lm_head, self-checked
against `model(...).logits` on the first batch of the process.

The **gradient** is the one-hot gradient of a SURROGATE — `tau * logsumexp_t(cos_t / tau)` at
`tau = 0.02`, minus `lambda * nll_soft` — because the exact objective is a hard max that routes the
whole gradient through one token of 32. It only proposes candidates; **selection is always exact**.
The gradient's keep mask is `common.score_ids`' keep mask (sink dropped, no norm filter): the fork
also dropped tokens above 10x the row median there, which would have optimised against a mask the
selector does not use.

**Families.** The same four arms run on two target families of the held-out set, in separate
directory trees: `realact` (a direction a real document actually produced, `unit(X[p] - mu)`) and
`sae` (an encoder column `unit(W_enc[:, f])`, which no text is guaranteed to be able to produce).
`--rows` indexes WITHIN `--family`, so `--family sae --rows 0-7` is the sae family's first eight
targets, global rows 1024-1031 of a 512-per-family set; every output row carries both (`row` is the
global index that `scan`'s topk and `score`'s per_target join on, `family_row` is the index the flag
named). For the `sae` family every final ALSO carries the feature's pre-gate activation --
`sae_feature`, `sae_act_at_argmax`, `sae_peak_act`, `sae_peak_pos`, `sae_fired` (peak above the
checkpoint's learned `threshold`), read off the SAME forward through `common.sae_encode`.
`sae_peak_pos` is **-1** when the feature is dead over every kept token: relu zeroes the whole row
and `argmax` would otherwise name position 0 arbitrarily, which `summary.json`'s
`frac_peak_at_cos_argmax` would then read as a genuine disagreement -- so that fraction is averaged
only over the `n_rows_peak_defined` finals where the feature fires somewhere. The 16 arms of
2026-09-16 were run BEFORE this fix and their `summary.json` on the volume still carries the
uncorrected fraction; `finals.jsonl` has every per-row field, so the corrected number is
recomputable without re-running. It is
RECORDED, NEVER OPTIMISED: the objective stays the cosine, and the activation is what says whether
the string the search found also makes the feature fire, or only aligns with its encoder column.

**Arms**, one output directory each, `arm = <mode>-<init>`:

| arm | pop x children x iters | lambdas | init |
|---|---|---|---|
| `gcg-random32` | 1 x 512 x 150 | 0 | 32 random space-prefixed ASCII ids, redrawn until the string round-trips |
| `gcg-corpus` | 1 x 512 x 150 | 0 | the scan's top-1 corpus window, cut to 32 tokens around its argmax |
| `epo-random32` | 3 x 85 x 300 | 0.1 / 0.19 / 0.37 | as above, one draw per member |
| `epo-corpus` | 3 x 85 x 300 | 0.1 / 0.19 / 0.37 | as above, the SAME window for all three members |

GCG IS EPO at pop 1, lambda 0, so there is one loop. The two modes are compute-matched in TOTAL
candidate forwards (150 x 512 = 76,800 against 300 x 255 = 76,500), not per iteration. Shared:
`--topk 512 --seq-len 32 --tau 0.02 --sbatch 256 --restart-every 0`, seed 0 (the per-direction
generator is seeded `[seed, crc32(family), row]` — crc32, not `hash()`, which is salted per
process). Each EPO member holds its own lambda and is selected by its own `L_lambda`, so the
per-member finals trace a Pareto front in one run at no extra forward.

**The corpus init.** The scan already searched the whole corpus for every held-out direction; this
takes its top-1 window at the largest corpus size, reads the ids straight out of `corpus/tokens.i32`
and keeps the 32 tokens ENDING at `max(argmax, 31)` — the tail of the window that still contains the
token the direction fires on — then `roundtrip_repair`s it. A window shorter than 32 tokens (a
document of <= 64 tokens is ONE window) is skipped and the next top-k entry taken, rather than
front-padded: padding would reintroduce exactly the confound the fork's `--seq-len-mode rollout` was
invented to remove, since the search would then optimise the pad and "the search improves the
inversion" and "the search prepends helpful context" stop being separable. The rank taken, the
window's own scan cosine (whole window, pre-cut, pre-repair) and the exact cosine of the repaired
cut the search actually starts from are all in `finals.jsonl`.

**Why the round trip is load-bearing.** A candidate survives only if `decode(ids)` re-encodes to the
SAME ids (`retok_ok`), checked on the whole assembled string. The fork MEASURED what happens when
the init does not round-trip: `retok_ok` rejects 100% of candidates for the whole run, zero
candidate forwards happen, and the init's own score is written out as a search result. Three things
here exist for that: the random init redraws until it round-trips, every init goes through
`roundtrip_repair`, and a run asserts both per iteration (not one candidate survived) and at the
end (`cand_forwards > 0 and filter_reject_rate < 1.0`).

**Outputs per arm directory** (`<root>/base/<base>/gcg/<set>/<family>/<arm>/`):

- `finals.jsonl` — one row per (direction, member): `row`, `family`, `member`, `lam`, `init`,
  `string`, `ids`, `cos` (exact, through the scorer), `nll`, `ppl`, `per_token_cos`, `argmax`,
  `argmax_tok`, `init_cos`, `init_nll`, `init_string`/`init_ids`, `family_row`, `token_entropy`,
  `nonascii_frac`, `distinct2`/`distinct3`, `cand_forwards`, `iters`, `wall_s`, the target's own
  `span_text`, and for the corpus init the window provenance (`init_topk_rank`, `init_doc`,
  `init_window_start`/`_len`/`_argmax`/`_cos`, `init_cut_end`, `init_repairs`).
- `trajectory.jsonl` — every `--log-every` (10) iterations per member: `iter`, best `cos`, best `L`,
  `nll`, `cos_soft` (the surrogate, for calibrating `--tau`), `filter_reject_rate`, `cand_forwards`.
- `top64.jsonl` — the 64 best DISTINCT candidate strings each member saw over its whole run, with
  the cosine it was selected on and an NLL computed in one batch at the end. Kept as a per-member
  running set, pruned to 1,024 whenever it passes 4,096, so the reported top 64 is exact.
- `summary.json` — config, lambdas, alphabet count against `ALPHABET_EXPECT[base]`, the per-run
  timings and rates, `by_lam` (the Pareto slice), the CHECK-block maxima, totals.
- `README.md` — provenance, per-arm cost AND cost per direction, mean final cos, mean init cos.

**The CHECK block**, run at the end of EVERY direction: the finals are re-scored through
`common.score_ids` twice — at the loop's own `--sbatch` (hard bound 1e-2, advisory 1e-4) and at
`common.SCORE_CHUNK = 32`, the chunk every other product scores at (bound 1e-2). Both maxima are
printed, stored in `summary.json` and quoted in the arm README. When the hard bound fires, the
assert reports the offending member, both cosines, the argmax position, the top1-top2 per-token gap
(a delta at or below it is an argmax flip between near-tied positions, which the max over positions
makes discontinuous), the residual norm there (a small one turns reduction-order noise into a large
cosine), and a sweep of that exact string across batch shapes — enough to classify the failure
without a second run.

**The scorer's batch-shape noise floor, MEASURED 2026-09-16.** A per-row cosine is BIT-IDENTICAL
across every batched shape from 8 to 512 rows and can differ by up to **~1e-2** in a ONE-row call:
an M=1 matmul takes the GEMV path and a batched one a tiled GEMM, with a different bf16 accumulation
order. It is a discrete step, not a continuum. `common.score_ids` uses one fixed chunk
(`SCORE_CHUNK = 32`) precisely so that every stored score in the pipeline sits on the batched side
of that step and is comparable with every other; a cosine recomputed at a different shape, above all
a single-row rescore, carries that jitter and must not be diffed against a stored one at face value.
This applies to every product that scores through `common.score_ids`, not only to `gcg`. Because a
`pop = 1` arm's two CHECK calls both score one row, they take the same path and report the same
number — so for `gcg` arms the two printed maxima are equal by construction and both measure
batched-vs-M=1, not batch-vs-itself; `SMOKES.md` has the sweep.

**The alphabet** is the ids that decode to non-empty printable ASCII, re-encode to themselves as a
single id, and are neither special nor added — the last filter is what keeps the SINK out of an
optimised string. It is measured per run and checked against `ALPHABET_EXPECT[base]` at +-10%.

Launch (one call per (base, arm); `--with numpy` because the LOCAL entrypoint validates the arm
configuration before it pays for a GPU, and that import reaches `gcg/gcg.py`):

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uvx --with pyyaml --with numpy modal run --detach \
     repo-maemm-precompute/paper-evals/gcg/modal_app.py \
     --base qwen3-8b --set 2026-09-16_v1 --arm gcg-random32 --rows 0-7 \
     --root /vol/runs/2026-09-15_paper-evals-smoke)
```

Flags: `--arm <mode>-<init>` (or `--mode` and `--init` separately), `--family` (default
`realact`), `--rows`, `--set`, `--root`, `--force`, and overrides for a cheap shakeout — `--iters`, `--pop`, `--children`, `--lam-grid`,
`--topk`, `--seq-len`, `--tau`, `--sbatch`, `--restart-every`, `--log-every`,
`--filter-oversample`, `--seed`. A full arm is 12-35 minutes, so it goes out with `--detach` and is
followed through the arm README on the volume.

**The 27B needs a different Triton.** `gcg` is the only product here that runs a BACKWARD pass,
and on the 27B that backward crosses 48 GatedDeltaNet layers. `flash-linear-attention` 0.5.2
REFUSES it on a Hopper GPU with Triton in [3.4.0, 3.7.1) -- it is guarding a wrong-answer bug in
`gated chunk_bwd_dqkwg` (its #640), not a missing feature -- so `gcg/modal_app.py` builds
`image27_gcg = image27.pip_install("triton>=3.7.1").env({"FLA_TILELANG": "0"})`, a layer ON TOP of
precompute's image27 that leaves that image and its pins untouched for every forward-only product.
The tilelang route fla also suggests was tried first and does not work here; `SMOKES.md` has all
three attempts. The FIRST gradient pass of a 27B container costs ~75-85 s of Triton autotune and
every one after it ~0.2 s, so a short shakeout badly understates the steady-state rate.

### Measured: the final run (2026-09-16, full root, `2026-09-16_v1` rows 0-31 per family)

`gcg` mode only, 2 bases x 2 families x 2 inits = 8 arms of 32 directions, at
`base/<base>/gcg/2026-09-16_v1/<family>/<arm>/`. `pop = 1`, so best-over-members and per-member are
the same number.

| base | family | arm | dirs | mean final cos | mean init cos | mean NLL | $/direction | peak act / gate | frac fired |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | `gcg-corpus` | 32 | **0.6398** | 0.5220 | 7.613 | $0.0904 | -- | -- |
| qwen3-8b | realact | `gcg-random32` | 31 | 0.4924 | 0.0525 | 12.932 | $0.0985 | -- | -- |
| qwen3-8b | sae | `gcg-corpus` | 32 | **0.3080** | 0.2358 | 7.970 | $0.0894 | 137.8 / 6.94 | 1.000 |
| qwen3-8b | sae | `gcg-random32` | 32 | 0.2161 | 0.0158 | 13.334 | $0.0917 | 89.4 / 6.94 | 0.969 |
| qwen36-27b | realact | `gcg-corpus` | 32 | **0.4886** | 0.3491 | 8.278 | $0.3257 | -- | -- |
| qwen36-27b | realact | `gcg-random32` | 32 | 0.2827 | -0.0179 | 13.081 | $0.3345 | -- | -- |
| qwen36-27b | sae | `gcg-corpus` | 32 | **0.2408** | 0.1478 | 7.310 | $0.3251 | 28.9 / 1.58 | 1.000 |
| qwen36-27b | sae | `gcg-random32` | 32 | 0.0682 | 0.0054 | 13.175 | $0.3332 | 5.2 / 1.58 | 0.875 |

8B `realact/gcg-random32` covers 31 directions: row 17 is excluded because its end-of-direction
CHECK exceeded the hard bound and the bound was not loosened to absorb it (`SMOKES.md` has the
diagnosis — it is the M=1 scoring path, not a search failure).

**The init dominates the arm**, on both bases and both families: corpus minus random is +0.147 (8B
realact), +0.092 (8B sae), +0.206 (27B realact), +0.173 (27B sae), at the same 76,800 candidate
forwards and the same wall. `gcg-random32`'s NLL is 12.9-13.3 against 7.3-8.3 from a corpus window:
the unconstrained string is not text.

**Against the MAEMMs on the same rows** (primary `qwen36-27b/2026-09-10_rl-8x2048-full`, and
`qwen3-8b/2026-09-03_run1-rl` on the 8B; naive max over the 64 drawn rollouts, which equals the
unbiased best-of-64 to every printed digit here):

| base | family | arm | GCG | MAEMM max-of-64 | difference | GCG wins |
|---|---|---|---|---|---|---|
| qwen36-27b | realact | `gcg-corpus` | 0.4886 | 0.5313 | -0.0427 | 9/32 |
| qwen36-27b | realact | `gcg-random32` | 0.2827 | 0.5313 | -0.2486 | 0/32 |
| qwen36-27b | sae | `gcg-corpus` | 0.2408 | 0.1308 | **+0.1100** | **29/32** |
| qwen36-27b | sae | `gcg-random32` | 0.0682 | 0.1308 | -0.0626 | 12/32 |
| qwen3-8b | realact | `gcg-corpus` | 0.6398 | 0.6532 | -0.0135 | 11/32 |
| qwen3-8b | realact | `gcg-random32` | 0.4924 | 0.6547 | -0.1623 | 1/31 |
| qwen3-8b | sae | `gcg-corpus` | 0.3080 | 0.1358 | **+0.1722** | **32/32** |
| qwen3-8b | sae | `gcg-random32` | 0.2161 | 0.1358 | +0.0803 | 29/32 |

**The answer differs by family.** On `realact` the MAEMM wins narrowly — -0.043 (27B) and -0.014
(8B), with 9 and 11 of 32 directions going to the search. On `sae` the search wins outright: 32/32
on the 8B and 29/32 on the 27B, because the MAEMMs reach only 0.111-0.136 on encoder columns and
their `sae` rollouts are short (mean 21.2 tokens on the 8B, 25.3-28.3 on the 27B) against 42.6-54.2
on realact.

**The caveat, stated once.** `gcg` optimises a SINGLE string of a FIXED `T = 32` ids and is compared
against the max over 64 sampled rollouts of mean length 21-54. Neither the token budget, nor the
number of samples, nor the compute is matched — 76,800 candidate forwards of 33 tokens is not the
same currency as 64 autoregressive rollouts. A `gcg` number is therefore a reachability figure at
T=32 from one initialisation: where it EXCEEDS the MAEMM (the `sae` family) the inequality is sound
in that direction, and where it falls short it bounds nothing.

**Cost.** $53.93 for the 8 arms — 8B $11.74, 27B $42.19 (3.6x) — at $0.089-0.099 and
$0.325-0.335 per direction, 82-90 s and 258-265 s of wall.

**The `epo` arms were run on the smoke root only**, 8 directions per arm across both bases and both
families, and are not repeated here. What they measured: the lambda term buys fluency and costs
cosine monotonically (27B realact `epo-corpus` 0.1 -> cos 0.389 / NLL 2.81, 0.19 -> 0.370 / 2.65,
0.37 -> 0.355 / 2.63, against `gcg-corpus`'s NLL of 8.41 at cos 0.442), and `epo` trails `gcg` on
raw cosine in every (base, family, init) cell by 0.02-0.08 — expected at lambda > 0, though its best
member sits at lambda 0.1 rather than 0, so the 3x-smaller per-iteration candidate pool costs
something of its own. `SMOKES.md` has those tables; the final run drops `epo` because the Pareto
front it traces is a separate claim from the reachability figure.

### The `sae` rows are stratified by feature density, and the quartile matters

`targets.py` lays the `sae` family out **stratum-major**: 128 rows per density quartile, contiguous,
q0 (rarest) = sae-local 0-127 = global 1024-1151. So `--family sae --rows 0-31` is **all q0**, and
the table above is the rare-stratum view of `sae`, not the family. The stratified rerun takes 8 rows
of each quartile — `--rows 0-7,128-135,256-263,384-391` — into separate arm directories via
`--arm-suffix strat`, so both views exist side by side. (A row's quartile is cross-checked against
`ids.jsonl`'s own `stratum` field, not inferred from the index.)

| base | arm | dirs | mean final cos ± SE | mean init cos | mean NLL | $/dir | peak act | fired |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b | `gcg-corpus-strat` | 32 | **0.2983 ± 0.0168** | 0.2332 | 7.461 | $0.0922 | 133.96 | 1.000 |
| qwen3-8b | `gcg-random32-strat` | 32 | 0.2032 ± 0.0240 | 0.0098 | 13.422 | $0.0946 | 87.22 | 0.969 |
| qwen36-27b | `gcg-corpus-strat` | 32 | **0.2344 ± 0.0159** | 0.1594 | 7.147 | $0.3358 | 29.85 | 1.000 |
| qwen36-27b | `gcg-random32-strat` | 32 | 0.0843 ± 0.0089 | 0.0038 | 13.354 | $0.3225 | 8.70 | 0.969 |

The arm mean barely moves against the q0-only arms (8B 0.3080 / 0.2161, 27B 0.2408 / 0.0682), so the
per-quartile difference against the MAEMM is the reason to have run it:

| base | arm | q0 (rarest) | q1 | q2 | q3 (densest) |
|---|---|---|---|---|---|
| 8b | `gcg-corpus-strat` | **+0.2103** (8/8) | +0.1185 (8/8) | +0.1284 (8/8) | +0.1078 (8/8) |
| 8b | `gcg-random32-strat` | **+0.1441** (8/8) | +0.0150 (6/8) | **−0.0019** (4/8) | +0.0274 (5/8) |
| 27b | `gcg-corpus-strat` | +0.0790 (6/8) | +0.0240 (6/8) | +0.0551 (8/8) | +0.0586 (8/8) |
| 27b | `gcg-random32-strat` | −0.0870 (3/8) | **−0.1647** (0/8) | −0.0871 (0/8) | −0.0450 (3/8) |

**The corpus-init `sae` result survives stratification; the random-init one does not.** The 8B
corpus arm beats the MAEMM on 32/32 targets in every quartile and the 27B on 28/32 (32/32 against
`rlI-150`). The random-init arm's advantage is real only on q0 and is a tie elsewhere — so the
`sae/gcg-random32` line in the table above (+0.0803, 29/32) is a **rare-feature artefact** and
should not be quoted as a family-level result. Note also that rarer is not uniformly easier: the 27B
corpus arm peaks at **q1**, and so does the MAEMM, which makes that a property of the features
rather than of either method.

**Reproducibility: the 8B search is bit-exact, the 27B is not.** Rows 1024-1031 appear in both the
q0 and the stratified arms and are seeded identically, so they are a free determinism check. The 8B
returns identical final ids on 8/8 with max |Δcos| **0.00e+00**; the 27B returns 1/8 and 0/8 with
max |Δcos| **0.264** and 0.146. The inits agree in both cases, so the divergence is in the search:
the 27B is the only base whose backward runs fla's GatedDeltaNet Triton kernels, and a
nondeterministic gradient changes the proposed candidates, after which the trajectories separate.
Most rows still agree to ~0.005-0.03 and one row per arm diverges hard. **A single 27B direction is
therefore not reproducible**, and the arm mean over those 8 rows moved by −0.036 and +0.024 between
runs — the same order as the 27B realact GCG-vs-MAEMM gap, so the SE over targets understates what
one run establishes on that base.

<!--GCG-RESULTS-->

## Run order

```
check -> corpus -> stats -> [mu_check] -> targets -> scan -> rollouts_hf | rollouts_vllm -> score -> gcg
                                         `-> repo_examples          (needs only targets + the SAE)
```

`rollouts_hf` needs `--maemm` and a built held-out set; `score` needs that MAEMM's rollouts file
and never loads the MAEMM itself, so a rescoring costs one clean-base load and nothing else
(checklist item 79).

`gcg --init corpus` additionally needs `scan` on the same (base, set): its init is the scan's top-1
corpus window. A scan built BEFORE the held-out set was last re-drawn is silently wrong for this --
compare the two READMEs' dates before trusting a corpus-init arm.

`stats` must precede `targets` twice over: the sae family is drawn from its fire counts AND
`stats/mu.f32` is the centring mean (see "Methods"; `targets` fails without it). `targets` must
precede `scan`, which needs `heldout/<set>/vecs.f16`. `mu_check` is a CPU afterthought on `stats`
and blocks nothing.

`rollouts_vllm` additionally wants `rollouts_hf` on the same (MAEMM, set) to have run first: its
marker-norm self-check reads that run's summary for the reference value, and without it the check
degrades to "must differ from the clean base" (which a silently ignored adapter would also pass if
the base norm were wrong).

Launch (from the mimir project root; nothing is printed from `.env.local`):

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uvx --with pyyaml modal run repo-maemm-precompute/paper-evals/precompute/modal_app.py \
     --product check --base qwen3-8b)
```

Flags: `--root` (default `/vol`) mirrors the whole `<root>/base/<base>/…` layout somewhere else —
the smokes use `/vol/runs/2026-09-15_paper-evals-smoke`; `--tokens` shrinks the corpus (it must be
one of the nested sizes); `--set` names the held-out set; `--force` replaces an existing product
directory; `--batch` overrides the 256-window forward batch; `--allow-short` lets a SAE stratum draw fewer than
`n / strata` features instead of failing (smokes only — it sets the product README's status to
`short`).

Step-3 flags: `--maemm <base>/<name>` (required by `rollouts_hf` and `score`); `--n` rollouts per
target (default `rollouts.n` = 64); `--rows "0-7"` / `"3,5,9-11"` restricts the targets
(`common.parse_rows`, every index must exist); `--max-new` overrides the generation budget;
`--gen-rows` overrides the rows-per-generate-call default; `--dirs-from <heldout dir>` takes the
directions from somewhere other than `<root>/base/<base>/heldout/<set>/`; `--no-marker-check` skips
the served-vs-base marker-norm forwards; `--no-sae` skips the SAE columns in `score`;
`--rescore-texts <jsonl>` + `--score-name <name>` score an arbitrary `{row, text}` jsonl against the
set's directions; `--import-run1 --n 16` makes `targets` copy run1's archived eval cache instead of
drawing (CPU, no GPU).

```
--product targets     --base qwen3-8b --import-run1 --n 16
--product rollouts_hf --base <base> --maemm <base>/<name> --set <set> [--rows 0-7] [--n 64]
--product score       --base <base> --maemm <base>/<name> --set <set> [--rows 0-7]
```

Step-5 flags belong to `gcg/modal_app.py`, a SEPARATE app -- see the GCG section above.

Step-4 flags: `--engine hf|vllm` picks WHICH rollouts file of a set `score` reads and where it
writes (`scores/<set>/` vs `scores/<set>__vllm/`); `--max-num-seqs` overrides the vLLM engine's
concurrency (default 256 on the 8B, 64 on the 27B); `--gpu-mem` overrides `gpu_memory_utilization`
(0.85 for rollouts, 0.55 when an HF model shares the card); `--throughput <label>` makes
`rollouts_vllm` ALSO measure the fixed 8 x 16 request set at the engine's own `max_num_seqs`
(`--throughput only` skips the rollouts and just measures, for the extra settings); `--fast-hook` would swap the stock vllm-lens worker extension for
`precompute/vllm_ext.py`, which is NOT in the tree (the flag asserts and says so).

```
--product mu_check      --base <base>
--product rollouts_vllm --base <base> --maemm <base>/<name> --set <set> [--rows 0-7] [--n 64]
--product score         --base <base> --maemm <base>/<name> --set <set> --engine vllm
--product rollouts_vllm --base <base> --maemm <base>/<name> --set <set> --max-num-seqs 64 --throughput s64
--product parity_greedy --base qwen3-8b --maemm qwen3-8b/<name> --set <set> --rows 0-7
--product repo_examples --base <base> --set <set> [--root ...] [--force]
```

`repo_examples` takes no flags of its own: it needs `--base` and `--set` (and honours `--root`,
`--force` and `--dirs_from` like every base product). It never loads a MAEMM.

The full 16M sequence per base:

```
--product corpus  --base <base>
--product stats   --base <base>
--product targets --base <base> --set 2026-09-16_v1
--product scan    --base <base> --set 2026-09-16_v1
```

**Do not put an `os._exit()` in a Modal function body.** MEASURED 2026-09-15: the `corpus` product
did, the container died mid-call although the work was committed, Modal re-scheduled the input, and
the retry failed with "already exists". Checklist item 59 (HF streaming can abort the interpreter
at finalisation) is handled instead by dropping the stream iterators and collecting garbage inside
`corpus.py` before it writes.

`--with pyyaml` is needed because the local entrypoint reads `config.yaml` to pick the GPU.
Anything that runs longer than a few minutes must be launched with `modal run --detach`
(four apps were lost to dropped local connections). **`--detach` is not enough on its own**:
MEASURED 2026-09-16, it keeps the APP alive when the local client drops, but the in-flight
`.remote()` input is CANCELLED when the client PROCESS exits -- a 27B rollouts run was killed six
minutes in by the shell `timeout` wrapped around its launcher, with "Received a cancellation signal
while processing input" in the container log. Launch long runs from a client that outlives the
shell call (`nohup setsid ...`, or a background task with no timeout). `paper-evals/` is baked into
the image (`copy=True`), not mounted: a mounted tree is shared state between concurrent sessions. An edit to
any file that IS in the image while another session's build is running fails that build with
"<file> was modified during build process", so `_IGNORE` now also drops `**/*.md`,
`**/.ruff_cache`, `reconstruction/out` and `reconstruction/data`: no script reads a markdown file
out of the code tree at runtime (every product WRITES its README to the volume), so keeping the
docs out of the image means a README or SMOKES edit no longer invalidates it or races a
concurrent build.

Local unit smoke, no GPU and no weights:

```
uv run paper-evals/precompute/unit_smoke.py
```

Local reproduction check against run1's archived numbers (CPU, no volume access — it reads files
already fetched off the volume plus the archived dumps):

```
uv run paper-evals/reconstruction/repro_run1.py --ours <fetched dir> --dumps <run1 dumps dir>
```

## Conventions that must match Celeste's code

Verified against `ceselder/maemm` master 09d4a01 on 2026-09-15. Line numbers are that commit's.

- **Injection**: decoder block **output** of `INJECT_LAYER` (=1, `mxf/config.py:7`), add mode,
  `h[pos] += unit(v) * ||h[pos]|| * coeff` with coeff 1.0 (`mxf/config.py:8`);
  `mxf/inject.py:10-57`, copied verbatim into `common.make_inject_hook` (add mode only). Decode
  steps (`h.shape[1] <= 1`) are skipped: the marker is injected at prefill.
- **Prompt**: marker `" ?"`, a single token occurring exactly once; chat template of `_INSTR` with
  `add_generation_prompt=True, enable_thinking=False`, marker ids appended **after** the generation
  prefix, so the marker is the last prompt token (`mxf/prompts.py:4-40`). The instruction text
  names `layer-{READ_LAYER}` and therefore differs per base; `prompt: ours8b | celeste27b` in
  config selects the function (today the same template at 27 vs 42).
- **Read layer**: block **output** of layer 27 (8B) / 42 (27B), captured by a forward hook that
  raises to stop the forward — never `output_hidden_states` (`mxf/inject.py:142-166`).
- **Scoring / re-encode** (`eval/eval_universal.py:130-161`): `padding_side` right,
  `add_special_tokens=False`, truncation at 95, a BOS sink (or eos) prepended at column 0 and
  excluded from `keep`, adapter disabled (clean base), cosine in fp32 (normalize h, einsum with
  unit dirs, `masked_fill(-1)`, max over tokens). One fixed scoring chunk of 32 rows everywhere.
  **Scorer noise floor (measured 2026-09-16, GCG shape sweep):** the same string scored at batch
  1 vs 8/32/128/256/512 differs by up to 1.03e-02 in a per-token cosine (n=1 is the GEMV kernel,
  every batched shape is bit-identical); every stored score sits on the batched side at
  `SCORE_CHUNK = 32`, so cross-product comparisons are consistent, and a single-row rescore is
  the one shape that is not.
  **Divergence (Tomáš, 2026-09-15):** the stored per-token cos and norm carry **no** norm filter;
  the 10x-nanmedian filter (`eval_universal.py:71,145-147`) is an option in
  `reconstruction/stats.py`. The cosine is **uncentred** while realact targets are `unit(act - mu)`
  (`data/build_universal_bank.py:26,310`); that asymmetry is hers and is kept.
- **Full-parameter MAEMM**: the served (tuned) model generates and `||h[pos]||` is its own; scoring
  is always the clean base (`eval/eval_ckpt_daemon.py:333-390`).
- **Sampling**: T 1.0, top_p 1.0, top_k 0, min_p 0, seed 1234; eos = tokenizer eos ∪
  generation_config eos (`rl/rl.py:61-75`); trimming keeps the stop token (`rl/rl.py:82-90`);
  max_new 64, min_new 16. The generation loop itself is `eval/eval_universal.py:_gen_batches`
  **505-522** (not 185-204, which is `_reencode_mlp_full` on this commit), copied into
  `rollouts_hf.run`: `fb = gen_chunk // bo` directions per batch, each direction repeated `bo`
  times, one `[1, d]` vector per row, the shared prompt tiled, `tok.batch_decode(gen[:, p_len:],
  skip_special_tokens=True)`.
- **"Rollouts are not all identical"**: asserted per target in `rollouts_hf.run`. NOTE the brief
  cited `eval/eval_ckpt_daemon.py:386-396` for this; on master 09d4a01 those lines are `wandb.init`
  and **no such assert exists anywhere in her repo** (grepped for `len(set(`, `identical`,
  `distinct` across `eval/`, `rl/`, `mxf/`). Ours is therefore an addition, not a copy.
- **Marker norm**: every rollout run logs `||h||` at the marker of the inject layer under the
  served model **and** under the clean base and asserts they differ (adapter-on ≈98.5 vs base
  ≈14.06 for 8B run1 — equality is the silent-adapter-off signature). `rl/rl.py:170-192`, copied
  into `common.marker_norm`.
- **SAE** (`mxf/sae.py:20-52`): BatchTopK loader with the dictionary_learning key aliases and the
  `nn.Linear` transpose; the `sae` family direction is `unit(W_enc[:, f])` — the **encoder**
  column; feature activation `relu((x - b_dec) @ W_enc[:, ids] + b_enc[ids])`; "fired" means above
  the checkpoint's learned `threshold` buffer (`eval/eval_universal.py:77-93`; **1.5846** for the
  131k 27B `l42-1b` SAE this pipeline uses), which the loader keeps and requires.

## Costs

Rates: H100 $3.95/h, H200 $4.54/h, CPU ≈ $0. Filled in from `SMOKES.md` as products land.

Smoke-measured (2M tokens on the 8B, 1M on the 27B), then MEASURED at full scale on 2026-09-16.
The `@16M` columns below are now actuals, not extrapolations; `SMOKES.md` has the whole run.

| product | 8B smoke | 27B smoke | 8B @16M | 27B @16M |
|---|---|---|---|---|
| `check` / `unit` (CPU) | ~$0 (19 s) | ~$0 (shared run) | ~$0 | ~$0 |
| `corpus` (CPU) | ~$0 (51 s) | ~$0 (41 s) | ~$0 | ~$0 |
| `stats` | $0.56 | $0.76 | **$2.72** (2,477 s) | **$9.12** (7,233 s) |
| `targets` | $0.18 | $0.15 | **$0.07** (64 s) | **$0.19** (148 s) |
| `scan` | $0.28 | $0.53 | **$1.82** (1,663 s) | **$6.34** (5,026 s) |
| `rollouts_hf` | $0.09 (48 x 64) | $0.27 (8 x 64) | **$1.96** (1,536 x 64) | not used at full scale |
| `rollouts_vllm` | $0.31 (48 x 64) | $0.37-0.48 (8 x 64) | $0.12 (48 x 64, patched) | **$5.63** / **$4.50** (1,536 x 64, patched) |
| `score` | $0.03-0.04 (3,072 rows) | $0.12 (512 rows) | **$0.22** (98,304 rows) | **$0.61** / **$0.63** (98,304 rows) |
| `parity_greedy` | $0.09 (8 dirs) | n/a (8B only) | — | — |
| `repo_examples` | $0.042 (512 x 30) | $0.167 (512 x 32) | corpus-independent | corpus-independent |
| `gcg` | $0.09-0.10/dir (gcg), $0.28/dir (epo) | $0.33/dir (gcg), ~$1.2/dir (epo) | corpus-independent | corpus-independent |

Throughput at 16M, MEASURED: `stats` 6,533 corpus tok/s (8B) / 2,229 (27B); `scan` 9,851 / 3,227.
The forwarded-token rate is ~4.1x higher because the 64/16 geometry puts every token in ~4 windows.
Both track the smoke's numbers to within 5%, so the smoke extrapolation was sound for the base
products; the two places it was NOT sound were `score` (the 512-row smoke was warmup-dominated and
implied 12.9 rows/s where 98,304 rows run at 225) and the vLLM `--throughput` probe (its
16-sample requests over 16 directions overstated the real n=64, 1,536-direction rate by 1.5x).

**The whole full-scale pass cost $45.66**: ~$4.8 of base products on the 8B, ~$15.9 on the 27B,
$10.1 for the three 1,536 x 64 rollouts runs, $1.5 for their scores, $0.2 for `repo_examples`, and
$7.0 of sunk cost from stopping and relaunching the 27B rollouts to pick up the vLLM speed patch.

## Step 6: the analysis layer (`centred`, `patchscopes`, `reconstruction/stats.py`)

Two new products and the local script that joins everything into the paper's tables. Both products
are registered in `precompute/modal_app.py`; `score` grew ONE flag, `--rollouts-dir`, so the
patching baseline reaches the single scoring path instead of carrying a second one.

```
/vol/
  base/<base>/
    patchscopes/<set>/<cell>/   rollouts.jsonl (the rollouts_hf schema, engine "hf-patchscope"),
                               rollouts.summary.json, README.md, index.json,
                               scores/ (written by `score --rollouts-dir <cell dir>`)
  maemms/<base>/<maemm>/
    scores/<set>/               + cos_centred_best.f16 [N, n], cos_filtered_best.f16 [N, n],
                               centred.json   (added in place by `centred`)
```

| product | where | writes | notes |
|---|---|---|---|
| `centred` | CPU | into `maemms/<base>/<maemm>/scores/<set>/` | the two SECONDARY per-rollout cosines, from the arrays `score` already stored. Loads no model, re-scores nothing |
| `patchscopes` | GPU | `base/<base>/patchscopes/<set>/<cell>/` | the zero-shot patching baseline on the CLEAN BASE: a direction patched into a Patchscopes target prompt, generated, then scored by `score`. NEVER loads a MAEMM |

### `centred` — the two secondary cosines (checklist items 4, 60)

`cos_centred_best.f16` is `cos(best_act[i, k] - mu, v_i)` at the primary's argmax token, with
`mu = stats/mu.f32`; `cos_filtered_best.f16` is the primary cosine with the 10x-nanmedian
residual-norm filter applied per rollout row. The primary number of the pipeline does not move: it
is still the stored uncentred `cos.f16`. For `realact` the centred variant is the both-sides-centred
reading (the target was built as `unit(X[p] - mu)`); for `sae` and `random` the target was never
centred, so the column is a DIAGNOSTIC, and `reconstruction/README.md` says so where the table
lives. Writing is `OutDir(keep_existing=True)`: the scores README is rewritten with the new files in
its table and `index.json` gains the two arrays.

```
--product centred --base <base> --maemm <base>/<name> --set <set> [--engine hf|vllm] [--root ...]
```

MEASURED 2026-09-16 on the smoke root (`SMOKES.md`): the filter drops 36 of 95,337 kept tokens on
the 8B and 0 of ~28,000 on each 27B MAEMM, and changes no row's maximum; centring lifts realact
best-of-64 from 0.5978 to 0.8363 (8B) and from 0.5496 to 0.8627 (27B rlI-150), which is exactly why
a centred number must never be quoted beside an uncentred one.

### `patchscopes` — the zero-shot patching baseline

Ported from `experiments/2026-09-15_patchscopes-run1/` (8B, $5.7, README §2-5 there) into this
pipeline's layout, sampling constants and output rules. Cells:

| cell | prompt | patch |
|---|---|---|
| `floor` | P2 | none: the hook is not installed at all |
| `p2-L<l>-replace2` | P2 | the placeholder's residual at the OUTPUT of block `l` is REPLACED by `2 * ||h_l[pos]|| * unit(v)` |

P2 is Patchscopes appendix D.1's entity-description prompt, verbatim, as PLAIN TEXT (no chat
template); the placeholder is its trailing ` x`, the last prompt token, and it is 30 tokens on the
27B tokenizer. Prompt, rule and alpha are carried over from the 8B screen rather than re-screened
(trial README §8.1: the main effects were small and consistent). The only swept axis is depth: the
8B winner was **layer 8 of 36 = 22.2%**, and the 27B has 64 layers, so the sweep is
**l in {8, 14, 21} = 12.5% / 21.9% / 32.8%**, bracketing it.

**The floor is the result, not a control to be skipped.** On the 8B the same prompt with the hook
disabled reached realact 0.1463 at bo 64 against the best injected cell's 0.2333 and run1-RL's
0.5817 — i.e. ~0.146 of the inverter's number is available to fluent English with no direction
anywhere in the forward pass. The floor arm therefore generates `n` continuations ONCE and
replicates them across every direction, so both arms are read at the SAME bo (an unmatched bo is
what flattered the 8B screen's floor). A floor `rollouts.jsonl` has N x n rows and only n distinct
strings; its GPU cost is n generations and its scoring cost is the full N x n.

Rows carry the `rollouts_hf` schema exactly, so nothing here computes a cosine:

```
--product patchscopes --base qwen36-27b --set 2026-09-16_v1 [--rows 0-63,1024-1087] [--n 8]
                      [--ps-layers "8,14,21"] [--no-ps-floor] [--gen-rows 32] [--root ...]
--product score --base qwen36-27b --set 2026-09-16_v1 \
                --rollouts-dir <root>/base/qwen36-27b/patchscopes/2026-09-16_v1/<cell>
```

`--ps-layers ""` uses `precompute/patchscopes.py:PS_LAYERS`; the floor cell is produced alongside
unless `--no-ps-floor`. Every injected cell runs a PATCH CHECK before generating (two extra forwards
of the shared prompt, the probe registered AFTER the patcher because hooks fire in registration
order) and asserts `cos(h_patched, v) > 0.99` — after a replacement the patched residual IS the
direction.

**Cost plan for the 27B run (cap $10).**

| stage | shape | est. |
|---|---|---|
| sweep | 3 layers x (64 `realact` + 64 `sae`) directions x bo 8, + the floor on the same rows (`--rows "0-63,1024-1087"`) | ~$1.0 |
| final | the best depth by mean best-of-8 on realact+sae; the largest rung below that fits | ~$5.5-8.3 |
| scoring | `score --rollouts-dir` on every cell | ~$0.5 |

**The specified final cell does not fit the cap, and the arithmetic is why.** HF `generate` finishes
a batch only when EVERY row in it has stopped, so a call costs `max_new` = 64 decode steps whether
or not individual rows emit eos: the planning figure is 64 generated tokens per rollout, not the
~40 a trimmed mean suggests. At this repo's own measured 27B rate (**239.3 gen tok/s at `gen_rows`
32**, "The vLLM engine path" above) and H200 $4.54/h:

| final cell | rows | generated tok | wall | cost |
|---|---|---|---|---|
| 512 x bo 64 x 2 families (**specified**) | 65,536 | 4.19M | 4.86 h | **$22.0** |
| 512 x bo 32 x 2 | 32,768 | 2.10M | 2.43 h | **$11.0** |
| 384 x bo 32 x 2 | 24,576 | 1.57M | 1.82 h | **$8.3** ← largest that fits |
| 256 x bo 32 x 2 | 16,384 | 1.05M | 1.21 h | $5.5 |
| 512 x bo 16 x 2 | 16,384 | 1.05M | 1.21 h | $5.5 |

With ~$1.5 for the sweep and the scoring, the final cell's envelope is ~$8.5. The rung is chosen
from the rate measured on the FIRST sweep cell at `gen_rows` 32 — **not** from the smoke's 31 tok/s,
which was batch 8 and is per-step overhead rather than the batch-32 rate — the way `screen` sized
the 8B trial off its own first cell. Every cell README records the specified shape, this projection
and the rung that actually ran, so the divergence is on the record wherever the numbers are.

### `reconstruction/stats.py` — the tables

Local, CPU, no GPU: it fetches only the small files off the volume into `reconstruction/data/`
(gitignored) and writes markdown + CSV into `reconstruction/out/` (gitignored). `best_act.f16` is
never fetched. Nine tables — the best-of-k curve with the UNBIASED order-statistic estimator, the
bo-sensitivity ranking, the paired MAEMM comparison per SAE density stratum, the corpus-scan
baseline and its quantiles, the SAE-repo column, the GCG ceiling, the argmax-position histogram, the
centred/filtered secondaries and the sae-family distribution. `reconstruction/README.md` has what
each one means, the estimator formula, the two mu conventions and the GCG-rows caveat.

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uv run repo-maemm-precompute/paper-evals/reconstruction/stats.py --root-tag smoke|full)
```

| product | 8B | 27B | note |
|---|---|---|---|
| `centred` | ~$0 (20.5 s, 1536 x 64, full root) | ~$0 (5.9-8.8 s, 8 x 64 smoke) | CPU; the wall is the scores-dir copy (854 MiB in 15.4 s) |
| `patchscopes` sweep | — | **$0.7670** (4 cells, 128 dirs x bo 8) | H200; 359-394 gen tok/s on the clean base |
| `patchscopes` final | — | **$5.8446** (floor + L14, 640 dirs x bo 32) | H200; 291 gen tok/s, 20,480 rows per cell |
| `score --rollouts-dir` | — | $0.29 (20,480 rows), $0.12 (1,024 rows) | H200; **115 rows/s** at 20k rows, 23.3 at 1k (load-dominated) |
| `reconstruction/stats.py` | $0 (local, 110 MB fetched over the run) | — | 12 tables |

**Patchscopes product total: $7.87 against the $10 cap** (sweep $0.767 + its scoring $0.54 + one
failed scoring attempt ~$0.13 + final cell $5.845 + its scoring $0.586). The specified 512 x bo 64
per family was not run: at the measured rate it is ~$15 of generation alone. What ran is 512
`realact` + 128 `sae` at bo 32 on the depth the sweep chose (L14, 21.9% -- the 8B's winner was 22.2%),
with the matched floor at the same bo. `sae` is at 128 rather than 512 directions because the sweep
showed no lift at any depth and the 8B trial found the same; it is kept rather than dropped so the
paper's 27B `sae` row is a measurement (bo 32, 128 features, lift +0.0036 ± 0.0018, p = 0.79) and
not a footnote.

## Step 7: the Delphi-style SAE autointerp evaluation (`autointerp/`)

Its own directory, its own Modal app and its own README — `paper-evals/autointerp/README.md` has
the stages, the arms, and every deviation stated once. Design:
`infra/2026-09-16_autointerp-design.md` **including its §9 amendments A1-A12**, which override
§2-§6 of that document where they conflict.

The question: for a held-out SAE feature, do a MAEMM's rollouts describe the feature as well as
max-activating corpus examples do, and does adding rollouts to a small corpus improve the
description? Measured Delphi's way — an explainer LLM writes a description from a set of examples,
a scorer LLM classifies held-out windows with it, and the number is the scorer's balanced accuracy.

```
/vol/
  base/<base>/
    sae/<sae>/random_pool/<set>/     2048 random corpus windows encoded for every tested feature,
                                     per-token, sparse CSR — the negative pool (scan's _random256
                                     cannot supply 20 zero-activation windows for a dense feature)
    sae/<sae>/examples_4m/<set>/     the C4 arm's OWN top-128 over the 4M nested prefix
    sae/<sae>/examples_docmax/<set>/ one window from each of a feature's top 256 DOCUMENTS — the
                                     test set's positive pool
    autointerp/<set>/<date>_build/   one jsonl per feature: the rendered example set of each arm
                                     and two disjoint test draws
  maemms/<base>/<maemm>/scores/<set>__<engine>/sae_self/
                                     the target feature's PER-TOKEN activation on its own rollouts
  runs/<date>_autointerp-<tag>/      cache/ (one file per prompt), explain/, detection/, fuzzing/,
                                     summary/
```

| stage | where | notes |
|---|---|---|
| `sae_self` | GPU | per-token target-feature activation on a MAEMM's own rollouts; self-validating against `cos.f16`, `argmax.i16` and the SAE CSR that `score` already wrote |
| `random_pool` | GPU | the shared negative pool |
| `examples_4m` | GPU | the C4 arm's own 4M scan |
| `examples_docmax` | GPU | the test set's positive pool, ranked by document rather than by window |
| `build` | CPU | the rendered example sets and the two test draws |
| `run` | CPU | the Anthropic Messages API (`claude-sonnet-5`), `--path sync\|batch`, cached by prompt hash, projected and capped before each stage |
| `autointerp/stats.py` | local | the tables, into `autointerp/pilot.md` and `autointerp/results.md` |

Three things about this evaluation that the rest of `paper-evals` does not have to deal with, all
MEASURED rather than assumed, all documented in `autointerp/README.md`:

- **There is no `temperature` parameter** on the Anthropic Messages API for this model generation
  (`anthropic` 1.6.0 raises `TypeError`). The design's "temperature 0" is unachievable, so the
  evaluation carries two explicit null arms instead of assuming determinism.
- **A test positive must fire at the gate.** Drawn from the stored equal-width activation bands
  without that rule, 17 of 20 positives on a pilot feature sat below the gate on text unrelated to
  the feature, and every arm landed near 0.6 balanced accuracy whatever its description said.
- **Disjointness is at the DOCUMENT level** between every shown example and every test item, which
  is why the positive pool had to be rebuilt around documents rather than windows.

Costs are in `SMOKES.md` under "`autointerp` -- the Delphi-style SAE autointerp evaluation".
