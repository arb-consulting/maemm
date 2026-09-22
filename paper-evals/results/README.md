# `results/` — the paper's results driver

`reconstruction/` answers questions about products (does the pipeline reproduce run1, do the two
engines agree, how hard does a feature fire). `results/` builds **the paper's own tables and
figures** from whatever is on the volume for one held-out set, by configuration rather than by a
script per question.

| file | what it is |
|---|---|
| `common.py` | the volume reader (`Vol`, `modal volume get` + a local mirror), the product readers moved over from `reconstruction/sae_smoke64.py`, source discovery, the clustered bootstrap, the markdown/CSV/figure output layer |
| `faithfulness.py` | eval 1: one command, every (family × source × run-tag) on a set |
| `autointerp.py` | eval 2: one command per SAE, joining the `runs/<dir>/summary/scores.jsonl` of every checkpoint's autointerp run |
| `sanity.yaml` | **the gates the user edits** — her card's numbers, `sae_smoke64.md`'s medians, our own recorded values, and `kind: cross_set` gates that read another set's product for the same checkpoint and compare the two on the rows they share; each with its tolerance and provenance |
| `selftest.py` | the CPU unit smoke: a synthetic mirror through the whole driver, every number checked against one worked out by hand |

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uv run repo-maemm/paper-evals/results/faithfulness.py --set 2026-09-21_v1raw)

uv run repo-maemm/paper-evals/results/autointerp.py --sae qwen36-27b/sae2m \
  --run rl-last16=<run dir> --run old-primary=<run dir> --run nla=<run dir>

uv run paper-evals/results/selftest.py            # no volume, no network, both evals
```

`--sources <substrings>` keeps a subset of the arms, `--root <prefix>` reads a smoke root
(`tmp/sae-smoke64`), `--no-fetch` answers everything from the mirror, `--out <dir>` moves the
output, which defaults to `precompute.common.out_dir("faithfulness")` (`$MAEMM_OUT/faithfulness`,
else `<repo>/_out/faithfulness`, beside `paper-evals/`). The mirror likewise defaults to
`precompute.common.mirror_dir(...)` (`$MAEMM_MIRROR/<root>`, else
`$XDG_CACHE_HOME/maemm-paper-evals/mirror/<root>`) — the old `results/data/` and `results/out/`
are legacy and no longer written by default, since Modal mounts the whole `paper-evals/` tree
into every image build and a write there mid-build kills the launch.

## The design commitments

**Config-driven, so a new checkpoint or a new dictionary is not an edit here.** Sources come from
iterating `config.yaml`'s `maemms:` against the volume's `scores/` (and `variants/`) directories;
families come from the rows' own `family` / `sae_key` / `sae_side` / `stratum`; the set's base is
resolved by asking the volume which `base/<b>/heldout/<set>` exists. No checkpoint name, SAE key
or family label is spelled anywhere in the code. The selftest pins this with two synthetic
dictionaries that differ only by the rows' `sae_key`.

**The `--run-tag` is a first-class axis.** Two arms of one checkpoint that differ only in `--mu`
are two sources (`rl-8x2048-full:mu-none` and `:mu-stats`), read off the scores directory names —
the inverse of `common.rollout_stem`. A `variants/<set>__<variant>/scores/` directory, which is
where `score --rollouts-dir` puts an NLA `--amp` arm or a patchscopes cell, is the same axis under
another parent and becomes a source with the variant as its tag.

**A missing source is skipped and listed, never zero-filled.** A checkpoint nobody ran and a
checkpoint that scored zero are opposite findings and must not print the same. `compute: false`
entries are not reported missing — they are declared and deliberately not generated.

**SEs are clustered by document.** 122 of 512 realact targets of `2026-09-16_v1` share a document,
and two targets from one document are not two independent draws. The bootstrap resamples whole
documents; a row with no document is its own cluster, so for `random` and `sae` it reduces exactly
to the ordinary nonparametric bootstrap. The CSVs carry the unclustered `se_iid` beside it so the
size of the correction is visible rather than asserted.

**No SAE-target cosine reaches a markdown table** (plan §2.3). They answer a different question on
a different scale; they go to `sae_cosines.csv`.

**Every number is read, and the reading is checked.** Nothing here recomputes an activation or a
cosine: the bo-k values are the products' own `per_target.jsonl` entries and the SAE peaks are
`sae_self.f16`. The reader check then recomputes the per-rollout reductions from the stored arrays
and compares them with the products' own aggregates, within one float16 ulp plus the 4-decimal
rounding both sides apply (`reconstruction/sae_smoke64.py`'s rule — an ABSOLUTE bound is wrong at
the 2M dictionary's activation scale and was the first version of this check). `cos.f16` is
[N, n, T] and reaches ~63 MB per arm at full scale, so its half of the check is bounded by
`--check-arrays-max-mb` and SKIPS with a reason rather than passing vacuously.

**A cross-set gate compares only what is comparable.** `kind: cross_set` intersects the two
products' rows and restricts to one family. For a NON-centrable family (`sae`, `random`) a
re-derived set stores the identical direction, so `cos_raw` is the same statistic on both and the
difference is a real one. For `realact` against a `storage: unit` set it is NOT — that set's `cos`
has a centred target and a raw scorer — so such a gate is written `compare: false`, which prints
both numbers and issues no verdict. A wide tolerance would say "close enough"; `compare: false`
says "not the same quantity". Only cosine metrics resolve: an activation metric would need the
other set's `sae_self` product, which this gate does not fetch.

**The sanity block flags, it does not stop.** Plan §2.3 says a failed gate stops the run; this
driver has no way to tell a wrong mu from a gate written against another set, and a run that
refuses to print its own tables cannot be inspected. The stop is the reader's, on a `FLAG` line.

## Eval 2 (`autointerp.py`) — the design commitments that differ

**One invocation per SAE, one `--run <label>=<dir>` per checkpoint.** `autointerp/run.py` is per
CHECKPOINT, so eval 2 is six run directories (rl-last16, the old primary and NLA, on each of the
2M and the 131k SAE) and the corpus arms are re-explained and re-scored inside each of them. The
unit of comparison is therefore `(<run label>, <arm>)` and never the bare arm name: `DOCMAX` under
two labels is two measurements, and one heading over both would merge them. The block is written
to `<out>/<sae-slug>/`, so the two SAEs never overwrite each other.

**Detection and fuzzing are never pooled.** Fuzzing marks at the gate and, on the legacy protocol,
runs zero-shot while detection sees Delphi's three verbatim few-shot turns. They are separate
columns and the table says why on every run.

**The pairing is asserted, and its losses are printed** (plan §3.3). Each contrast intersects the
two arms' feature sets by feature id, asserts the alignment, and carries both sides' lost features
into the table (`REDUCED`), the CSV (the ids) and `results.json`. `autointerp/stats.paired` gets
the intersection right via polars' `drop_nulls` but reports nothing about what it dropped; this is
the half that was missing.

**The estimator is `autointerp/stats.boot_ci`'s method, reimplemented rather than imported.** That
module is the IN-CHAIN renderer, on the run's own directory, with that chain's `N_BOOT = 10000` and
seed `20260916`; this driver is on the `results/` lifecycle and takes `N_BOOT`/`BOOT_SEED` from
`results/common` so every table in the paper is bootstrapped the same way. The method — a paired
percentile bootstrap over FEATURES — is deliberately identical.

**The reader check has no array to re-reduce, so it re-derives the metric.** `run.rates` defines
balanced accuracy as `0.5 × (TPR + TNR)` and the two A5 views restrict only the negative side, so
all three stored accuracies are determined by the rates stored beside them. Recomputing them is a
comparison of two independent quantities within the 6-decimal rounding `run._nr` applies; it also
counts rates outside [0, 1] and rows with more parsed batches than batches. The selftest breaks it
by one number and requires it to go red.

**A null `bal_acc` is an absent measurement.** `run.py` writes `null` when a class was absent from
the PARSED items (every batch of a feature unparsed); those features are dropped from the means,
counted in `no metric`, and never imputed as 0.5.

**Unequal N is shown.** The NLA arm shows 4 examples to a corpus arm's 16, and every scorer-only
pseudo-arm (`R-shuffled`, `C16-judge2`, `C16-draw2`, `NLA-desc`) carries `n_examples: 0` because
those names are not keys of the build's per-feature arm dict. Both appear in `support` as
themselves.

### What eval 2's driver does NOT do

- **No sanity gates ship with it.** `--sanity` resolves a YAML in the same selector idiom as eval
  1's `sanity.yaml`, but no file exists: eval 2 has no recorded expectations yet, and a YAML of
  invented numbers would be worse than none. An absent file is a note, not an error.
- **`--ref` resolves a bare arm name inside the FIRST `--run` given.** Every run directory carries
  its own `DOCMAX`, so the name alone does not identify one; the convention is stated in the
  caption. `--ref <run label>/<arm>` names one outright.
- **The per-stratum table is a flag, not a config lookup.** The paper reports it for the primary
  SAE only, and `--no-strata` is how the secondary block turns it off — nothing in `config.yaml`
  marks an SAE primary.
- **Only `summary/scores.jsonl` and `summary/build.json` are read.** `costs.json`,
  `features.json`, `floor_permutation.json` and the per-scorer `batches.jsonl` are not: no dollar
  figure, no per-batch diagnostic and no derangement map reaches these tables.
- **The pre-registered primary contrast (plan §3.3, search baseline vs MAEMM) is not marked as
  such.** Every non-reference arm gets the same contrast row; which one is confirmatory and which
  exploratory is not a distinction this file makes.
- **`n_shown_exceeding_corpus_peak` per arm (plan §3.3) is not reported** — `scores.jsonl` does not
  carry it.

## What is NOT covered

- **`realact_long`, `bsf`, `jlens`** have no code of their own and need none — they are ordinary
  cosine families and will appear as soon as a set carries them — but they have never been run
  through this file, because no set on the volume has those rows yet.
- **`sae_side`** (`enc` / `dec`) is read from the row and splits the family, but no set on the
  volume carries the field yet, so that split is exercised only by the selftest.
- **The oracle row, the search baseline and patchscopes** (plan §2.2) are not built here. The
  oracle is a `score --rollouts-dir` product and WILL appear as a source once it exists, under the
  variant's name; the search baseline (`scan --arm`) and the patchscopes cells are different
  products with different shapes and are not read.
- **The unbiased order-statistic best-of-k** (`reconstruction/stats.py:bo_weights`) is not offered:
  the bo-k reported here is the disjoint-group estimator the products store, which is what plan
  §2.3 names. Switching would mean reading `cos.f16` in full for every arm.
- **NLA's n = 4** means it can never reach a bo8 or bo64 column; those print as an em dash and the
  plan's "never a bo64 column for NLA" is satisfied by the products, not by a special case here.

## Module M1 — the paper's cells, the ratio denominator, the corpus comparator (2026-09-24)

What `faithfulness.py` now builds for `paper/numbers/cells.csv`, and what it deliberately does
not. The plan is `evals/2026-09-23_implementation-plan.md` §M1; the spec is
`evals/2026-09-22_eval-set-and-presentation.md` §1.1-§1.4 and §4.

**Implemented.**

- **The corpus-peak denominator is a parameter**, `--corpus-peak`. `stored` (the default) is what
  `sae_self` recorded, our 16M held-out `max_act`, which is what this file divided by before.
  `top1_act:<volume-relative dir>` reads M2's `top1_act.jsonl` and takes each row's `act_max`.
  The provenance string reaches every ratio cell's `note`, the markdown caption and the CSVs. An
  asked-for source that is not on the volume **raises**, naming the path; there is no fall-back to
  `stored`, because a ratio against the wrong denominator prints as an ordinary number.
- **Panel a**, from `cos_centred` only, over her block minus its own exclusions, with
  document-clustered SEs and the surviving `n`: `fid.ra.ex.cos.bo1`, `fid.ra.ex.cos.bo8`,
  `fid.ra.base.cos.bo8`, `fid.ra.nla.cos.bo1`, `fid.rnd.ex.cos.bo1`, and the paired
  `fid.ra.diff.cos.bo8` (with `lo`/`hi`), `fid.ra.ex.win.bo8`, `fid.ra.nla.dex`.
- **Panel b**, per corpus-frequency quartile and never pooled where the key has a quartile:
  `sae.l131k.ex.{ratio,fired}.bo{1,8,64}.q{1..4}`, the 2M appendix block under `s2menc` / `s2mdec`
  at bo1 and bo8, and the three pooled rows that already exist in the CSV, rewritten in place with
  `note` saying they are pooled. The gate is **read from the product and asserted** equal to
  1.5846 to four decimals on the 131k; a product carrying another one stops the run.
- **The cells writer**, `--cells`. Empty (the default) writes nothing, so the driver stays
  read-only; `default` resolves to the paper project's own `paper/numbers/cells.csv`. It rewrites
  this module's keys in place and every other row of the file keeps its **exact bytes** — not its
  fields, its bytes — because six builders share the file and a diff has to show one change when
  one thing changed. It refuses a key M1 does not own, the same key twice, a file that already
  carries a duplicate key, a reordered header, and a lone `lo` or `hi`.

**Stubbed, and what the stub does.**

- **`results/corpus_search.py` (module M2) does not exist yet.** `faithfulness.py` imports it and,
  when the import fails, `corpus_top1_missing` prints one line naming the keys it is skipping and
  the missing input, and `fid.ra.diff.cos.bo8` and `fid.ra.ex.win.bo8` are **not written** — the
  placeholder rows already in the CSV keep saying "expected, not yet measured", which is true.
  Nothing is zero-filled. The hook's contract is
  `corpus_top1(vol=, base=, set_name=, rows=, size=) -> {row: top-1 cosine}`, called by keyword;
  a module that is importable but exports none of `CORPUS_SEARCH_EXPORTS` **raises**, because a
  contract mismatch is not an absence.
- **No `top1_act` product on `celeste-train10m` exists yet either**, so every ratio cell a run
  builds today is labelled as the 16M held-out number it is.

**Not covered here.**

- The corpus-size curve (`fid.ra.corp*.cos`, `fid.ra.corp10.slope`) is M2's, GCG/EPO is M3's, and
  the OOD and autointerp keys belong to M5 and M6. `M1_KEYS` is the full list this writer will
  touch and it is spelled out rather than discovered from the CSV.
- **Which arm is the Exemplifier is not in the config.** `primary: true` is on the dropped old
  primary, and no checkpoint name is spelled in `faithfulness.py`
  (`selftest.check_combined_layer_lifts_and_never_recomputes` enforces that), so `--exemplifier`
  names it and, with no flag, the cells are **skipped and listed** whenever more than one MAEMM
  arm was scored on a block rather than one being guessed at.
- **`lo`/`hi` is the normal approximation** `mean ± 1.96 × the document-clustered bootstrap SE`,
  the same interval `make_headline_figure` draws, and not a percentile bootstrap:
  `common.cluster_bootstrap` returns an SE and not its resample distribution, and a second
  resampler here would be a second estimator under one name. Spec §4's "percentile" applies to
  panel c's `stats_ood`, which is M5's own estimator.

## Module M5 — the OOD arms table and panel c (2026-09-23)

What `ood.py` builds for panel c and for `paper/numbers/cells.csv`'s `ood.*` rows, and what it
does not. The run is `runs/2026-09-23_ledger.md` (M5: set `2026-09-23_ood_full`, 22 arms x 512
targets, `formulas` dropped); the spec is `evals/2026-09-22_eval-set-and-presentation.md` §2 and
§4 panel c.

**Implemented.**

- **A `--centre` scan's mean is read from its README.** `precompute/scan.py --centre` writes
  `- CENTRING: --centre: BOTH sides about <path>, the scoring constant`, which is a different
  line from `common.note_convention`'s `- CENTRING: mu=<path> from <source>`. Both are parsed
  (`MU_RE`, `CENTRE_MU_RE`), and the derivation line — which mentions a mean but describes only
  how the target bank was built — is neither. `scan_mu_of` also returns WHETHER the scan centred,
  so "both sides are centred" is a read fact in the notes instead of the caveat it used to be.
- **The corpus size is per arm and comes from the product.** Each arm is read at the largest size
  ITS OWN scan reached, from that scan's README `- sizes:` line, cross-checked against the sizes
  in its own `topk.jsonl` (a disagreement stops the run). The size is printed in the arm's row,
  carried in the record as `corpus_size_m`, used for that arm's Δ, and written to
  `ood.<arm>.corp10.mtok`. `config.yaml` is never consulted for it: four arms still declare 16
  there while their scans ran `--max-size 10`. The random floor indexes `quantiles.f16` by the
  scan's own size ladder for the same reason.
- **The language-id column reads a chunked rollouts product.** `score` names every rollouts file
  it read on one `- rollouts:` line; a `--rows` split product names all of its
  `<stem>__rows<a>-<b>.jsonl` chunks there, comma-separated.
  `stats_ood.rollouts_rels_from_readme` returns the list and `stats_ood.read_rollout_rows` reads
  the union through `precompute.common.read_rollouts`, so the chunk invariants and the
  disjointness are checked where they are defined. A chunk file of the same stem that the README
  does NOT name stops the read rather than joining it.
- **Every printed count is the bo8 verdict.** `conjunction_counts` is the one definition behind
  the `N of M arms exceed` sentence and the `ood.conj.diff.n*` cells; the bo64 `outcome` is in
  the CSV and counted nowhere. `ufw_en` (the English anchor) and the `diag` arm are reported as
  rows and not counted.
- **`--cells <path>`** merges this run's rows into `paper/numbers/cells.csv` by key, in place,
  through module M1's byte-preserving writer (`faithfulness.write_cells`) — one implementation of
  "preserve every byte", imported rather than copied. `owned_cell_keys` enumerates the keys this
  module may write, so a typo is refused rather than appended.
- **A cell that was not measured is not written.** `results.common.num` prints an em dash for an
  absent number, which is right in a markdown table and wrong in `cells.csv`, whose `value` is
  copied into the tex verbatim. `write_cells` drops any row whose value is absent, drops a lone
  `lo`/`hi` (an error in `make_numbers.py --check`), and writes `diff.verdict` only for the three
  real verdicts — never `not comparable` or `no bo8 pairs`, which are true statements and not
  verdicts.

**What needs more than the scans-and-scores mirror.**

- **The `lang / ceiling` column** needs the rollouts product (the texts) and each arm's corpus
  `docs.jsonl` (the ceiling). Both are ordinary volume reads, no GPU. Both degrade to a NOTE when
  absent, and a chunked rollouts product missing any one of its files is withheld ENTIRELY rather
  than read in part: a rate over a fraction of an arm's targets printed as the arm's rate is a
  wrong number, where an absent column is only an absent column.

**Not covered here.**

- Panel c's FIGURE is not drawn by this file; it writes the arms table, the per-arm CSV and the
  cells rows the figure and the tex read.
- The English in-distribution reference stays in the frozen 2026-09-16 UNCENTRED convention and
  is not comparable by magnitude with the centred columns above it; the table says so.
- `ood.all.ex.usd` (the run's cost) is not built here: it is a fact about the launch, in the
  ledger, not about any product.

**Nothing in this module is stubbed or mocked.** Every number in the table comes from a product
on the volume.
