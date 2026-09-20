# What changed on `arb/features`, and why

Branch off `origin/arb/precompute`. Eleven commits, 2026-09-19/20. Nothing in
Celeste's repo is touched and nothing upstream is overwritten.

Two kinds of change: **new code under `paper-evals/features/`**, and **three fixes
inside `paper-evals/precompute/`** that were hard failures blocking the 2M SAE.

---

## New: `paper-evals/features/`

| file | what it does |
|---|---|
| `bundle.py` | read-only reader for Celeste's v2 bundle, which is an INPUT. Pulls from the Modal volume on first use, caches under `~/.cache/maemm/<snapshot>`. |
| `registry.py` | the 2M-SAE feature registry: one row per feature over all 2,097,152, with the seed-2026 `sft`/`rl`/`eval` partition and per-feature corpus peak. |
| `activations.py` | the activation-target registry: per-target document, position, residual norm and split side for the eleven non-SAE families. |
| `layout.py` | the `<family>/{train,test}` output layout, and the `heldout_kind` gate. |
| `emit.py` | writes both registries into that layout with a README per family. |
| `heldout_v2.py` | imports the v2 directions as a held-out set the existing pipeline reads unchanged (`ids.jsonl` + `vecs.f16`). |
| `fetch_hf.py` | stands in for the private `infra/fetch_hf.py` that `common.snapshot()` points at. |
| `spawn.py` | runs a product detached, so a long job outlives the client. |

### What the registries established

The splits were **not drawn here**. They already exist and are verified; what did not
exist was a machine-checkable record of them. Re-deriving rather than trusting the
bundle README:

    split counts                     sft 1,847,152 / rl 150,000 / eval 100,000   OK
    512 standard-eval within eval    True
    sae2m_enc / sae2m_dec in eval    True
    eval features carrying a peak    100,000 / 100,000

`corpus_peak` comes from the rank-0 window of `eval_2m_features_100k_windows`, which
reproduces the stored `corpus_peak` of `eval_2m_features_512` **exactly** on all 512
(max abs diff 0.000000), so the peak extends to all 100k eval features with no scan.

### Three properties of the activation draw, measured not assumed

1. **The directions are already centred.** realact's shared-component norm is 0.0615
   against the Gaussian control's 0.0437 — these are `unit(act - mu)` with Celeste's
   `whiten_mu`. A scorer that centres again centres twice. **`mu` is not in the bundle.**
2. **Targets are not independent.** 512 realact targets come from 449 documents; 59
   documents carry 2-3 targets, so 122 targets share one. An SE over 512 independent
   targets is too small. `doc_n_targets` is in the table so a bootstrap can cluster by
   document.
3. **`realact_early/mid/long` are not context variants.** They are the realact
   parquet's own `early_dirs`/`mid_dirs`/`long_dirs` columns (identical, 512/512) but
   near-orthogonal to realact (median cosine -0.007), so they are different
   activations. They ship no provenance at all.

### The `heldout_kind` gate

Every family must declare how its held-out claim is established, or `family_dir()`
raises and nothing is emitted for it. Over the 13 families:

    6  unknown          no provenance shipped -- claim inheritable, not checkable
    3  in_distribution  NOT held out; must never enter a held-out mean
    2  feature_id       verified by the seed-2026 partition
    1  doc_range        verified, 0 intersection, margin 396,399 documents
    1  category         asserted from the training mix, not measured

**Only 3 of 13 families carry a verified held-out claim.** That was invisible until
each family had to state one.

---

## Fixed in `paper-evals/precompute/`

All three are 2M-feature problems. The pipeline was written around a 131k dictionary
and none of them can happen at that width.

**1. `stats.py` asserted exactly one SAE per base.** `qwen36-27b` has carried two
since the 2M SAE landed. Added `--sae`, plumbed through `modal_app`; the single-SAE
case is unchanged.

**2. The 2M SAE OOMs an H200.** `load_sae` pulls both weight matrices in fp32; at
2^21 features that is 43 GB each, and 86 GB beside the 27B's ~52 GB wanted 138 GB of
141. The stats counters read `b_dec`, `W_enc`, `b_enc` and `threshold` and never touch
`W_dec`, so `load_sae` gained `need_decoder` and stats now pays for the encoder alone
(~97 GB). Precision unchanged; a caller that later reaches for `sae.W_dec` gets an
`AttributeError`, not a wrong number.

**3. `modal run` cannot carry a long product.** It holds a blocking `.remote()` open
for the whole job; on a multi-hour run the client JWT expires mid-call, raises
`AuthError: Jwt is expired`, **and the container dies with it** — losing the run and
leaving an empty `sae2m.tmp-2026-09-20` on the volume. Auth was never broken; a
`check` call immediately afterwards passed. `features/spawn.py` deploys and spawns
detached instead.

---

## Config

`qwen36-27b/2026-09-18_rl-last16-lr5e-7` as a `maemms:` entry — full-parameter, sha
`a1e4f299`. RL reward is the max cosine over the **last 16** generated tokens, which
also means the draft's Table 7 ("entire generated span") is wrong. `prompt`,
`train_max_new` and the marker norm are marked UNVERIFIED inline. Not `primary: true`.

`qwen36-27b/sae2m` as a `saes:` entry, recorded from the SAE's own config/verify
rather than assumed: F 2,097,152, k 64, gate 1.682811975479126, EV 0.711, L0 67.45,
0 dead features, 100% fired over 20.0M eval tokens.

## Runs

| run | result |
|---|---|
| `check`, CPU | passes; prompt `celeste27b` is 103 tokens, marker id 907 at 102, single occurrence |
| checkpoint fetch | 55.59 GB, single snapshot `a1e4f299`, 326 s |
| 2M SAE fetch | 45.15 GB, single snapshot `de89b1c1`, 157 s |
| **smoke, 8 targets x 64** | **passes, $0.2442.** marker ‖h‖ **294.0** vs clean base 14.062 — full-parameter weights unambiguously loaded. 333 gen tok/s on HF |
| `stats`, 2M SAE | running detached, `fc-01M2YW9NYWX1AVDZ5XSY7EDXME` |

`mu.f32` fingerprinted before any `--force`: 20,480 bytes, d=5120, ‖µ‖ **67.9258**
(matching D.3's 67.9), sha256 `80f3f995...`, with the exact bytes held locally. The
two failed runs left temp directories that were never renamed, so µ is untouched;
the hash gets re-checked when the pass lands.
