# `results/` — the paper's results driver

`reconstruction/` answers questions about products (does the pipeline reproduce run1, do the two
engines agree, how hard does a feature fire). `results/` builds **the paper's own tables and
figures** from whatever is on the volume for one held-out set, by configuration rather than by a
script per question.

| file | what it is |
|---|---|
| `common.py` | the volume reader (`Vol`, `modal volume get` + a local mirror), the product readers moved over from `reconstruction/sae_smoke64.py`, source discovery, the clustered bootstrap, the markdown/CSV/figure output layer |
| `faithfulness.py` | eval 1: one command, every (family × source × run-tag) on a set |
| `sanity.yaml` | **the gates the user edits** — her card's numbers, `sae_smoke64.md`'s medians, our own recorded values, and `kind: cross_set` gates that read another set's product for the same checkpoint and compare the two on the rows they share; each with its tolerance and provenance |
| `selftest.py` | the CPU unit smoke: a synthetic mirror through the whole driver, every number checked against one worked out by hand |

```
cd /home/gavento/dev/mimir/2026-09-maemms
(set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
 uv run repo-maemm/paper-evals/results/faithfulness.py --set 2026-09-21_v1raw)

uv run paper-evals/results/selftest.py            # no volume, no network
```

`--sources <substrings>` keeps a subset of the arms, `--root <prefix>` reads a smoke root
(`tmp/sae-smoke64`), `--no-fetch` answers everything from the mirror, `--out <dir>` moves the
output, which defaults to `results/out/faithfulness/`. `data/` (the mirror) and `out/` are
gitignored — both are reproducible from the volume, and an `--out` outside `out/` is not
ignored, so keep ad-hoc runs under it.

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
