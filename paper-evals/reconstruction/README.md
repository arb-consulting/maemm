# reconstruction/ — the local analysis layer

Six local scripts. None of them touches a GPU, loads a model, or computes a cosine: every number
they print was produced on the volume by `precompute/` and is read back here.

| script | what it answers |
|---|---|
| `stats.py` | **the paper's tables** — every product on one root, joined (this file, below) |
| `repro_run1.py` | does our pipeline reproduce run1's archived numbers on the 16 imported directions? |
| `parity.py` | do the HF and vLLM engines produce the same MAEMM? |
| `act_smoke.py` | how hard does a target feature fire on each source, as a max over rollouts? |
| `sae_smoke64.py` | the same question at a MATCHED best-of-k, per density stratum, on both SAEs |

## The two activation smokes

`act_smoke.py` is the first cut and is **frozen** — the numbers in `SMOKES.md` were produced by
it and it is not edited. It reports one statistic per (feature, source): the max over that
source's rollouts, with the 64-rollout product truncated to its first 4 (`maemm_bo4`) to correct
for the rollout count.

`sae_smoke64.py` generalises it and is the one to run now:

* every source is read at **bo1 / bo4 / bo16** — the disjoint-group best-of-k mean
  (`common.best_of_k_means`, the estimator `score`'s `per_target.jsonl` already uses for
  cosines) — so two sources are comparable at whatever k they both reach, the whole rollout
  budget is used at every k, and no source is truncated to make a comparison;
* four sources per SAE — `primary`, `rl16`, `nla`, `corpus` — plus `primary16` on the 131k side,
  which is the 64-rollout vLLM product read at its first 16 so the rollout budget is literally
  equal to the 2M side's. All declared in one table at the top of the file, with one mirror
  `root` per SAE and a per-source root override for a product the smoke did not write;
* every table is also cut **per stratum**;
* cosines go into the `.json` only, never into a result table;
* the file ends with a **Comparison against prior results** section: Celeste's card numbers, a
  reader check of the stored `sae_self.json` `per_target` against a recomputation from
  `sae_self.f16`, and the 2026-09-21 `act_smoke` medians. Each is computed where its inputs are
  in the mirror and printed as `absent` with the reason where they are not.

Both are local and offline: `--data <mirror>` whose subpaths are the volume's own, nothing
fetched, nothing recomputed on a GPU. Nothing in either is mocked or stubbed — a source that is
not in the mirror is reported `absent`, never as a zero.

**What `sae_smoke64.py` needs in the mirror, beyond what act_smoke needed.** `--peaks-1b` is
optional and only feeds the card comparison on the 2M side: the stratified draw already writes
`corpus_peak_1b` into that set's `ids.jsonl`, and the flag is there for a draw made on a mirror
without the bundle's `data/celeste-v2-2026-09-17/heldout/eval_2m_features_100k_windows.parquet`.
On the 131k side the card comparison wants
`base/qwen36-27b/sae/l42-1b/repo_examples/<set>/repo_examples.jsonl` — its `repo_peak` column
maxed over a feature's shipped windows. `per_feature.jsonl` from the same directory is accepted
as a fallback, but it carries only `repo_mean_peak`, a MEAN over those windows and not a peak,
so every ratio taken against it is inflated and the section says so where it is used.

**Mirror `per_feature.jsonl` as well, even when `repo_examples.jsonl` is there.** Her 1.0B-scan
peak for the 131k SAE is not in our data at all; the repo's shipped max-acts are a PROXY, and
whether they are even on our activation scale is an open question in `paper-evals/README.md`
(whether the shipped file folds the SAE's `norm_factor`). `per_feature.jsonl` is the only file
carrying our re-scored peak (`mean_peak_act`) beside the repo's own stored value
(`repo_mean_peak`) over the SAME windows, so their per-feature ratio IS that scale. The card
block reports its median and IQR; outside 0.9–1.1, or unmeasured, the `÷ repo peak` column is
labelled `unverified` and the verdict becomes *"denominator scale differs, not a pipeline
verdict"* instead of naming a defect — the difference and its z are still printed. The 2M side
needs none of this: `corpus_peak_1b` is her own measurement, not a stand-in for it.

    uv run paper-evals/reconstruction/sae_smoke64.py --selftest
    uv run paper-evals/reconstruction/sae_smoke64.py --data ~/mirror \
        --rows <the 64 sae rows of 2026-09-16_v1> --out out/sae_smoke64.md

`stats.py` fetches the small files off the volume into `mirror_dir(<root-tag>)` (mirroring
the volume's own paths, default `$XDG_CACHE_HOME/maemm-paper-evals/mirror/<root-tag>`) and writes
markdown + CSV into `out_dir("reconstruction")` (default `<repo>/_out/reconstruction`, beside
`paper-evals/`). The old `reconstruction/data/` and `reconstruction/out/` are legacy and no longer
written by default — Modal mounts the whole `paper-evals/` tree into every image build, and a
write there mid-build kills the launch. **`best_act.f16` is never fetched** — 335 MB per MAEMM at
full scale; the centred cosine that needs it is computed ON the volume by `precompute/centred.py`
and read back as an `[N, n]` array.

## Commands

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uv run repo-maemm-precompute/paper-evals/reconstruction/stats.py --root-tag smoke)
```

`--root-tag smoke` reads `/vol/runs/2026-09-15_paper-evals-smoke`; `--root-tag full` reads `/vol`
(the pipeline's default `--root`). Other flags:

| flag | effect |
|---|---|
| `--tables adg` | build only those letters (default: all of `abcdefghi`) |
| `--no-fetch` | answer everything from the mirror (`mirror_dir(<root-tag>)`); a miss is reported, not fetched |
| `--refetch` | re-download even what the mirror already has (the cache is by existence) |
| `--modal-cmd` | how to invoke the CLI (default `uvx modal`) |
| `--width 200` | console width; 0 = the terminal's, and 200 when the output is piped |
| `--data-dir` / `--out-dir` | override either directory |

The two secondaries of table (h) need `precompute/centred.py` to have run on each scores dir:

```
(… modal run repo-maemm-precompute/paper-evals/precompute/modal_app.py \
    --product centred --base <base> --maemm <base>/<maemm> --set <set> [--root …])
```

**Primary and secondary MAEMMs.** `config.yaml` marks one MAEMM `primary: true` (Tomáš,
2026-09-16: `qwen36-27b/2026-09-10_rl-8x2048-full`). `stats.py` reads that key — it is not
hard-coded — and orders every table's rows primary first, carries a `role` column in table (a), and
makes the primary side **A** in the paired table (c), so every difference reads "primary minus
other". Secondary rows are shown because their computation was already paid for, not because the
paper claims anything about them. With no `primary: true` anywhere the ordering is a no-op and the
run prints that the tables keep config order.

**Engine stems.** A MAEMM's scores live at `scores/<set>` (HF rollouts) or `scores/<set>__vllm`
(vLLM rollouts — the 27B's primary path). Both are loaded and labelled apart (`<name>` and
`<name>@vllm`), and both join on the same logical set, so a base scored through one engine and a
base scored through the other appear side by side with the engine visible in the MAEMM column.
`scores/<set>__rescore-*` is not a rollout grid and never enters these tables.

Nothing is assumed about what exists. Products that are absent make their table print a note and be
skipped; a scores directory whose arrays and aggregates disagree raises instead (see "What is
asserted").

## The tables

Every table carries `n`, `bo`, `seed`, the set name and the corpus size in its caption, and the
MAEMM weight sha in a column (checklist items 29 and 62). Values are `mean ± SE` with the **SE
taken across targets** — rollouts of one direction are correlated, so a per-rollout SE would
understate it; every per-target statistic is formed first, then averaged.

**(a) `a_family_x_maemm`** — per family × MAEMM: the bo1 mean and the best-of-k curve for
k ∈ {1, 2, 4, 8, 16, 32, 64}, from the `[N, n]` per-rollout maxima recomputed from `cos.f16`. The
last column is the disjoint-group estimator the product stores (`common.best_of_k_means`), at
k = n, where the two coincide by construction — it is a consistency check on the join, not a second
result. The `random` family is included everywhere (checklist item 26: the control also rises with
bo, +52-72% from bo1 to bo64, and a table that drops it invites the reader to forget that).

**(b) `b_bo_sensitivity`** — the MAEMM ranking per family at bo 1 / 4 / 16 / 64. Rankings flip with
the rollout budget (checklist item 27), so the ranking is reported as a function of it rather than
at one bo. A blank cell is a MAEMM with no rows of that family on this root.

**(c) `c_paired_maemms`** — two MAEMMs on the same base, the same set and the same target rows,
compared **per target**: the difference of best-of-n (unbiased) and of the mean cosine, mean ± SE,
the win fraction and a two-sided exact sign test over the non-ties. For the `sae` family the same
rows are also cut by the ids.jsonl density stratum (q0 = rarest quartile), because paired
differences are stratum-dependent (checklist item 63). Sides are paired only within one (base, set).

**(d) `d_corpus_scan`** — the corpus-retrieval baseline, per nested corpus size (1/2/4/8/16M): the
mean over targets of the top-1 window cosine and of the mean of the top-64, from `scan/topk.jsonl`;
then the scan's own cosine quantiles (p50 / p90 / p99 / p99.9 / p99.99 from `quantiles.f16`),
averaged over targets. The quantiles are the calibration of the whole paper: they say what cosine an
arbitrary corpus window reaches, i.e. what a MAEMM's number has to be read against. Checklist item
67 — the corpus search is the most useful baseline and often beats the adapters. Geometry is
64-token windows every 16 tokens (`common.windows_of`), which is results-affecting and therefore
stated in the caption.

**(e) `e_sae_repo_examples`** — the `sae` family's extra baseline column: the SAE repo's OWN shipped
max-activating windows through our scorer. `max_cos` (mean over features) IS the
`sae-repo-top32` column — the best of a feature's 30-32 shipped windows, the natural-text analogue
of a best-of-32 rollout draw; `mean_cos` is its per-window mean; `frac fired` is the share of a
feature's own windows whose peak activation clears the checkpoint's learned gate; `peak r` and
`argmax agree` are the repo-vs-us activation agreement (the 8B reproduces its scan at r ≈ 0.994,
the 27B does not — see the main README).

**(f) `f_gcg_ceiling`** — the reachability ceiling, PAIRED. Layout is
`base/<base>/gcg/<set>/<family>/<arm>/finals.jsonl` (a realact direction and an sae direction are
different objects and never share an arm directory). Per (base, family, arm): the per-direction best
member's final cosine, its init cosine and its NLL, ± SE across directions; then, for each MAEMM on
the same (base, set) and ordered primary first, its unbiased best-of-n and mean-of-n on **exactly
those directions**, the paired difference ± SE, and the fraction of directions GCG wins. With the
final draw the GCG rows and the MAEMM rows index the same held-out set, which is what makes the
pairing exact rather than distributional; a direction missing from `finals.jsonl` (8B realact
`gcg-random32` excludes row 17) is absent from the join, and the `dirs` column says how many
survived. An EPO arm holds 3 members at different λ, so "the arm's result" is the best member per
direction, not a mean over members.

**Two sae views, and why they are not merged.** The sae arms exist on two draws. The `*-strat`
arms take 8 targets from each of the four density quartiles (32 in all) and are the **reported** sae
rows: their `all` slice is a stratified estimate of the family, and the per-quartile slices carry
the variation. The earlier plain sae arms took the first 32 sae rows, which are **all q0**, the
rarest quartile; they are kept as a `rare-stratum q0 view` and never averaged in, because reporting
them as the family number reports the rarest quartile as the whole. MEASURED 2026-09-16: that
distinction is worth a factor of two — on the 27B, `gcg-corpus` beats the primary by +0.1100 ±
0.0166 on the q0 view and by +0.0542 ± 0.0100 stratified. The `view` and `slice` columns say which
row is which; realact has one draw and one slice.

**GCG and EPO in one table.** `gcg-*` arms hold one member; `epo-*` arms hold three, one per
lambda (0.1 / 0.19 / 0.37), each selected by its own `L_lambda`, so an EPO arm traces a Pareto front
in a single run. An arm's `all` row is the mean ± SE over targets of the **per-target best member**,
which is the reachability figure; the `lam ... (member)` rows are PER-MEMBER means and carry no
MAEMM columns, because comparing one member against the inverter is a different claim. Read together
they separate two ceilings: what a 32-token string reaches at any fluency (GCG, NLL 8-13) and what a
readable one reaches (EPO, NLL 2.4-3.1).

**Precision.** An arm's own mean columns are shown to 2 dp on the 27B and 3 on the 8B — the
precision their SEs support. The paired columns keep 4 dp: they are per-direction differences with
much tighter SEs, and the arm's precision would erase real signal there. `ARM_MEAN_DP` in the
script is the one place this lives.

**Superseded arms are labelled, never silently dropped.** An arm whose directory name ends in
`-smoke` is a partial or abandoned run kept beside the arm that replaced it (27B realact
`epo-corpus-smoke`, 25 of 32 directions). Its `view` column reads `SUPERSEDED partial run -- not a
reported arm` and its `dirs` column is honest, so the rows stay inspectable while being impossible
to average in by accident. Note such an arm's MAEMM columns are computed on ITS subset of
directions, so they will not match the full arm's.

**Sources.** The per-direction best member comes from `finals.jsonl`. Where `summary.json` also
carries `per_dir_best_cos`, the two agree exactly (max |difference| 0.0 on the 27B stratified arms),
so the summary is a cross-check rather than a second source of truth.

**The caveat the caption states once:** GCG and a MAEMM are not the same object and are not
compute-matched. GCG optimises ONE fixed 32-token string with ~77k candidate forwards *against the
scorer itself*; the MAEMM draws n sampled rollouts from a prompt and never sees the metric. GCG
bounds what the metric is reachable to; it is not a baseline the inverter competes with. The MAEMM
columns are absent where no scores directory covers that (base, set) — on the smoke root the 8B's
GCG is on `2026-09-16_v1` while its scores are on the imported `2026-09-03_run1-archive16`, so the
join is empty and the columns do not appear.

**(g) `g_argmax_position`** — where in a rollout the best-scoring token sits, as
`argmax / n_kept_tokens` in 8 bins, plus the fraction exactly at the last token and the mean kept
length. Peak position is strongly recipe-dependent (77-82% last-token in some recipes against ~4% in
others, checklist item 10) and has been under-reported, so it is per family × MAEMM and never
pooled.

**(h) `h_centred_filtered`** — the two secondary readings against the primary, at bo1 and bo64, with
the delta. See "Two mu conventions" below.

**(j) `j_patchscopes`** — the zero-shot patching baseline (`precompute/patchscopes.py`), per
(base, set, cell, family): the cell's own bo1 mean and unbiased best-of-bo, its lift over the
matched `floor` cell (mean ± SE per direction, win fraction, sign test), and the MAEMM on the SAME
directions — shown twice, once re-estimated at the CELL's bo and once at its own. The bo difference
is the whole reason for the second column: the 8B trial measured the baseline gaining 3.6x from bo1
to bo64 against run1's 1.2x, so a bo-8 cell put beside a bo-64 MAEMM would credit sampling luck to
the inverter. The floor is the same prompt with no injection anywhere in the forward pass,
generated once and scored against every direction at the same bo; on the 8B it reached realact
0.1463 against the best injected cell's 0.2333, so it is a column of the table and not a footnote.
Cells are found at `base/<base>/patchscopes/<set>/<cell>/scores/` — written by
`score --rollouts-dir`, i.e. the same scorer, the same arrays, the same loader.

**(i) `i_sae_distribution`** — for the `sae` family, the p10/25/50/75/90 (plus min/max) of the
**per-target** unbiased best-of-n, not just its mean. SAE metrics are bimodal and a mean hides that
(checklist item 61).

**(k) `k_sae_per_feature`** — the per-FEATURE sae comparison, built to test a specific claim
(Celeste's note: "the 27B MAEMMs invert badly on ~30% of SAE features, ~40% worse than corpus
search") rather than to illustrate it. One row per (MAEMM, slice) over the same 512 features: the
MAEMM's unbiased best-of-n against the corpus scan's top-1 window at the largest corpus size and
against the SAE repo's own best shipped window, with the win fraction (MAEMM ≥ baseline) and BOTH
thresholds the claim mixes — an absolute shortfall > 0.05 cosine and a relative shortfall > 40%,
i.e. `(baseline − MAEMM)/baseline > 0.4`. Slices are `all` plus the ids.jsonl density quartiles,
because the answer is stratum-dependent (item 63) and the pooled number alone would confirm or deny
the claim by accident. The comparison is deliberately generous to the inverter: it gets its full
best-of-n while each search gets its single best text.

The `own-feature act` column is the max, over a feature's rollouts, of THAT feature's own gated
activation at the argmax token, from the scores CSR — "did the text actually make the feature
fire", which is a better discriminator than cosine alone (item 68). It is fetched for the **primary
MAEMM only**: the CSR is ~48 MB per MAEMM against a 100 MB local fetch budget, and the claim under
test is about cosine. Secondary rows carry the cosine columns and leave that one blank.

**(l) `l_sae_diff_deciles`** — the shape behind (k): deciles of the per-feature difference
(MAEMM best-of-n − corpus top-1), plus the mean ± SE and the fraction above zero. `d1` is the 10%
of features where the MAEMM falls furthest behind the search. A mean near zero with wide deciles is
exactly the bimodality a mean hides (item 61), and it is what distinguishes "the inverter is level
with search" from "the inverter wins half the features and collapses on the rest".

## The unbiased best-of-k estimator

For one target with `N` observed per-rollout scores `x`, sorted ascending as `x_(1) ≤ … ≤ x_(N)`:

```
E[max of k draws]  =  Σ_{i=1..N}  x_(i) · C(i-1, k-1) / C(N, k)
```

The i-th smallest observation is the maximum of a k-subset exactly when the other k-1 members are
drawn from the i-1 observations below it, and all `C(N, k)` subsets are equally likely. The
estimator is unbiased for every `k ≤ N` and uses all N rollouts; the disjoint-group estimator the
products store (`common.best_of_k_means`: `floor(N/k)` consecutive groups, mean of the group maxima)
uses only `floor(N/k)·k` of them and has a larger variance, and the two agree exactly at `k = N`.
Checklist item 25 — best-of-n is upward-biased and does not saturate by 64, so the bo axis is
reported as a curve, from the full per-rollout scores, with the estimator named.

## Two mu conventions

The pipeline centres **once**, at construction: a `realact` target is `unit(X[p] - mu)` with
`mu = base/<base>/stats/mu.f32` (the read-layer mean over the 64/16 scan windows of pass A). After
that every cosine in every product is **uncentred** — that asymmetry is Celeste's and is kept
deliberately (main README, "Methods").

- **Primary** (tables a-g, i): the stored per-token `cos.f16`, max over kept tokens. Uncentred
  activation against an already-centred (realact) or never-centred (sae, random) target.
- **Secondary, centred** (table h): `cos(best_act - mu, v)` at the primary's argmax token, computed
  by `precompute/centred.py` from the stored `best_act.f16`. For `realact` this is the
  both-sides-centred convention. For `sae` and `random` the target was never centred at all, so
  subtracting mu from the activation alone is one-sided and the column is a **diagnostic**, not a
  competing metric. MEASURED on the smoke root: centring lifts realact bo64 by +0.24 to +0.31,
  which is exactly why the two conventions must never be compared across tables.
- The argmax is **not** recomputed for the centred variant — `best_act.f16` holds the residual at
  the primary's argmax and nothing else, so the centred number is "the centred reading of the token
  the primary chose", not "the best centred token".

Checklist item 60: raw primary, centred secondary, both from the stored vectors; the no-centring
decision of 2026-09-08 stands because 27B mu instability made centred cosines non-comparable across
models.

## The norm filter

`cos_filtered_best` (table h) is the primary with Celeste's 10×-nanmedian residual-norm filter
applied (`eval_universal.py:71,145-147`): per rollout row, the median of the kept tokens' norms,
drop everything above 10× it, retake the max. The products store per-token values **unfiltered**
(checklist item 4) and this is the option. `centred.json` records the fraction of kept tokens the
filter drops. MEASURED on the smoke root: 36 of 95,337 tokens on the 8B (0.038%) and 0 of ~28,000
on each 27B MAEMM — and the filtered maxima are identical to the primary to 4 decimals on every
family, i.e. the dropped tokens were never the argmax.

## What is asserted

`stats.py` recomputes each scores directory's per-rollout maxima from `cos.f16` and asserts them
against `per_target.jsonl`'s own `mean_cos` (bound 2e-3, fp16 storage alone is worth < 1e-3). A
disagreement means the fetched arrays and the fetched aggregates are not from the same run — the
one failure a cached `data/` directory could otherwise hide. Everything else is a skip with a
printed note: a missing product, a set without a scan, a base with one MAEMM instead of two.

## Where the numbers came from

`out/<root-tag>/index.md` lists every table built by the last run, with its caption and the exact
command line. The smoke-root tables are pasted into `../SMOKES.md`.
