# `evals/pipeline` — what this branch changed, and what it did not

Branch `evals/pipeline` off `arb/main` (0b0515e), started 2026-09-21 as `evals/conventions` and
renamed when its scope became **the one branch for all our evals**: config → sets → rollouts →
scores → tables, rerunnable on new data by a config change alone. (The brief that opened the work
asked for this file as `CHANGES-conventions.md`; it is named after the branch it documents.)

Implements §0 and §1 of `infra/2026-09-21_evals-plan-main.md` (v2), with the "keep as is" and
blocking findings of `infra/2026-09-21_evals-plan-critique-opus.md` binding, **plus** Tomáš's
2026-09-21 schema decision, which overrides the plan where the two differ.

**Nothing here was run on the Modal volume.** No product was launched, no volume path written, no
corpus or scan output touched. Every number quoted below is from the local CPU unit smoke.

---

## Base: three commits cherry-picked from `arb/features` (Ari)

Cherry-picked onto 0b0515e **before** any of this branch's own work, then the branch was rebased
onto them.

| his | now | resolution |
|---|---|---|
| `a07cc70` drop the SAE decoder everywhere, tag the sae family `sae` | `640238c` | **Conflict in `autointerp/sae_self.py` and `features/draw_sae2m.py`, resolved to 0b0515e's version in both.** Same two fixes, two spellings: 0b0515e (from `arb/nla`) already had `need_decoder=False` at all four `sae_self` sites and already emitted `family: "sae"` + `sae_key`. The difference that mattered: his `draw_sae2m` writes `"sae_key": "sae2m"`, a BARE name, while 0b0515e writes the full `<base>/<name>` config key — which is what `common.sae_key_for` produces and what the new `sae_rows_of` selector compares against, so a bare name would have made the selector match nothing. What survives from his commit is therefore `features/ngram_overlap.py` (new), the `scan`/`top1_act` decoder drops, and the patchscopes floor-cell naming fix (a P1 run had overwritten two P2 floors). |
| `353c62b` search corpus over the checkpoint's training ranges | `6ad2872` | clean |
| `541b4b0` document-level dedup, `--corpus-name` | `d6fff69` | **Conflict in `config.yaml`, merged rather than chosen.** Both declare `2026-09-20_sae2m_2k`. Kept from 0b0515e: `imported: true` (his declaration omits it, and without it `common.default_heldout` would have made this 2,000-row sae2m set the default of every product that omits `--set`, silently moving `2026-09-16_v1` off the paper's tables) and `families: {sae2m_enc: ...}` (what the rows on the volume carry). Taken from his: `seed: 20260920` (`scan` reads it and KeyError'd without one), `sae_min_fires`, `sae_strata`. **Open for Ari:** his entry says `families: {sae: {n: 2000}}` — if the rows on the volume really carry `sae`, that line is the one to change; check `ids.jsonl` before the next scan. |

Nothing of his was re-implemented: the search corpus and `--corpus-name` are his, and `--corpus`
(below) resolves *through* a config key *to* his flag.

---

## What this branch did

### 1. Config schema (§1.1, as amended by Tomáš)

- **A mu is a FILE.** Everywhere one is named — `maemms.<k>.mu`, `heldout.<set>.mu_stored`,
  `heldout.<set>.family_mu.<fam>`, `--mu` — the value is `null` (subtract nothing), a path to a
  `[d]` `.f32`/`.npy` file, or `unknown` (stored rows only). Absolute, or relative to `--root` so
  a smoke gets its own; `{base}` expands to the base key. **There is no enum and no registry of
  mean names.** (The first version of this branch had a `mus:` name registry per the plan; it was
  replaced, because a new checkpoint trained on a new mean must be a config-only change.)
- `maemms.<k>.mu` is what the checkpoint was TRAINED to receive. Declared for the old primary
  (`null`), `rl-last16` (Celeste's archived `whiten_mu.npy`), the base control (`null`, the
  primary's by definition) and `nla-av` (`null`). The older 8B/LoRA entries have **no `mu:` key**:
  their training banks' convention was never established, `common.input_mu` refuses rather than
  guessing, and an explicit `mu: null` is a statement while an absent key is a gap.
- `family_kinds:` — `centrable` / `kind` per family, independent of which set carries it. Not
  `families:`, which already means `heldout.<set>.families`.
- `heldout.<set>.storage` / `mu_stored` / `family_mu` — the storage contract: `raw` | `unit` |
  `dirs_only`, and which mean each family's stored rows carry.
- `corpora:` — every built corpus by key (`heldout16m`, `celeste-train10m`), with `dir`, `sizes`,
  `block`, `stride` and the provenance sentence that says whether a search win on it is evidence
  about unseen text. `--corpus <key>` resolves to Ari's `--corpus-name`; passing both is refused.
- `bases.<b>.whiten_mu` — Celeste's archived mean, a plain path, used only by `mu_check`/`mu_diag`.
  Replaces `stats.ARCHIVE_MU`.
- `corpus.revision` — the Ultra-FineWeb pin (D9). Empty is refused.
- `realact_len` deleted (D10): dead since `targets.py` landed.

Validated in `common.load_config` beside `_check_nla`. Accessors: `load_mu`, `resolve_mu_path`,
`mu_label`, `input_mu`, `mu_of_family`, `family_centrable`, `set_storage`, `storage_record`,
`corpus_key_name`, `corpus_geometry`.

### 2. Raw storage and `dirs_for` (§1.2, §1.3)

- `targets` writes `act.f32` (raw, fp32) + `vecs.f16 = unit(act)` + `storage.json`. The single
  `x[i] - mu` at `targets.py:143` is gone; `cos(act, vecs) > 1 - 1e-5` is asserted on every row
  before either array is written. For a non-centrable family the `act.f32` row IS its unit
  direction, so `unit(act) == vecs.f16` there.
- `common.dirs_for(cfg, base, set_dir, mu, root, notes)` is the only way a direction is read. It
  returns the whole `[N, d]` array and leaves row selection at each call site (the critique's B12:
  `rows` means three different things across the readers).
- `common.mu_for` resolves WHICH mean in one order: `--mu` (recorded as a DEVIATION when it
  overrides a checkpoint's own) → the MAEMM's `mu:` → a legacy set's own stored convention →
  **refuse**. The refusal is the point (the critique's B3).
- Wired into all seven readers: `score`, `rollouts_hf.load_dirs` (which `rollouts_vllm` ×2 and
  `rollouts_nla` call), `scan`, `gcg`, `patchscopes`, `repo_examples`, `centred`. Each keeps its
  own thin wrapper, and each records the convention in its README via `common.note_convention`.
- `targets --re-derive <old set>` (§1.4): re-forwards an existing set under the raw contract
  without re-sampling it, asserting the draw is identical row for row and that re-centring the new
  `act.f32` under each family's OLD mean reproduces the old `vecs.f16` to min cos > 0.9999.
  **Code and command only — not run.**

### 3. `--sae` and the `sae_key` selectors (§1.5, D2/D3, critique B1/B9/B10)

`sae_key_for` at the five open call sites (`targets`, `top1_act`, `repo_examples`, `gcg`,
`corpus_top1_activation` — the last gained a `--sae` option it never had), and the two inline
copies in `scan`/`stats` replaced, so there is one `--sae` syntax. `common.sae_rows_of` filters on
the row's own `sae_key` (+ the `sae_side` axis) in `sae_self`, `build`, `scan`, `repo_examples`,
and `gcg` asserts the selection did not cross dictionaries. `scan` loads encoder-only (D3) and its
`examples/` gains a set component (B9), with a read-time fallback to the legacy unkeyed directory.

### 4. Two cosines (§1.3, critique B4)

`score_ids(..., dirs_centred=, mu=)` emits `cos_centred` from the same forward, forwarded through
`score_tokens`. `score` builds both tensors, NaNs the non-centrable rows, and stores
`cos_centred.f16` + `argmax_centred.i16` (its OWN argmax, not the uncentred one) and centred
aggregates in `per_target.jsonl`. A caller passing neither keyword is bit-identical, so `gcg` and
`sae_self` are untouched.

### 5. Defects: D4 (span clamp), D5, D6, D7, D8, D9, D10 — all closed. See the commit bodies.

### 5b. D11 — `sae_self --rollouts-dir`

`sae_self` reads `<dir>/rollouts.jsonl` + `<dir>/scores/` instead of a MAEMM's, mirroring
`score.py:352-358`, so a patchscopes cell, a GCG/EPO finals file or a corpus-search result gets
the target feature's own activation through THIS stage rather than a second implementation.
`--maemm` is optional in that mode (and only in that mode; `build`'s M arms still need one).

**The critique's B11 is wrong about the code, and following the code made this smaller.** It says
`score` writes no CSR under `--rollouts-dir`, citing `score.py:343` — but that line is inside
`_rescore`, the `--rescore-texts` path. The `--rollouts-dir` path goes through the ordinary
`run()` body and writes the full array set, CSR included, whenever `--sae` is given. So all three
checks RUN in that mode. What actually produces a CSR-less scores directory is `--no-sae`, and
that is now detected from `index.json` (`sae_idx.i32` of zero bytes): checks 2 and 3 are SKIPPED
with `csr_checked: false` and a `csr_skipped_reason` in the product's `checks`, never relaxed —
against an all-False CSR they would either trip on every real firing or pass vacuously.

### 5c. The rollout stem gained a `--run-tag`

Found while setting up the old-primary reconciliation, and the same class as B9's `examples/`:
`common.rollout_stem` keyed on (set, engine) only, so two runs of ONE checkpoint on ONE set that
differ only in `--mu` write the same file and the second silently replaces the first — mid
comparison, with nothing raising. `--run-tag <suffix>` is the third axis, empty by default so no
existing path moves; `score` and `sae_self` read it back, and `score`'s output directory follows
`--score-name` for the same reason.

### 6. Selftests

`unit_smoke` 29 → **36** checks. `uv run precompute/unit_smoke.py` → `[smoke] 36/36 checks passed`.
The six new ones were each run against a deliberately broken variant; the mutation table is in
`SMOKES.md` under 2026-09-21.

---

## What this branch did NOT do

| not done | why / what settles it |
|---|---|
| ~~D11~~ | **Done.** See "D11" below. |
| **The §1.6 four-cell local selftest** (old primary, rl-last16) × (131k, 2M) + NLA through `targets → rollouts → score → sae_self` on a tiny synthetic set | The existing selftest pattern does not reach it: `rollouts_*` and `sae_self` load real weights and require CUDA, and there is no fixture for a MAEMM. What IS covered on CPU is every piece those four cells would exercise in `common`: the storage contract, both cosines, the `sae_key` selector, the exact-solve migration. The cell matrix itself is the paid smoke documented in `SMOKES.md`. |
| **`centred.py` dropping its own einsum** (§1.3) | `score` now writes the honest per-token centred cosine, so `centred.py`'s `cos_centred_best` — a max read at the UNCENTRED argmax — is redundant where `cos_centred.f16` exists. It still computes its own. Should become: read `cos_centred.f16` when present, keep `cos_filtered_best` always. |
| **`features/heldout_v2.py` getting the `OutDir` treatment** (§1.4) and its docstring correction | Untouched — **superseded for eval 1** by `features/heldout_v3.py`, which is an `OutDir`-backed Modal product (`--product heldout_v3`) rather than a local script, and which imports the families `heldout_v2` never reached (`realact_long`, `bsf`, `jlens`) plus a copy path for another set's rows. `heldout_v2` itself is still local-only and still overwrites its `--out` without a `--force`. |
| **`corpus_arm_dir` from `arb/exp-ood`**, the OOD rebase (§4.2), the `evals/sae-smoke64` merge | Not in scope here. Ari's `--corpus-name` + the new `corpora:` block cover the search-baseline need for evals 1–2. |
| ~~**The exact-solve migration of Celeste's 512 realact rows** (§1.4)~~ | **Done, 2026-09-21.** U1 is settled at $0: `pool_act_norm` is `‖act‖`. `features/heldout_v3.py --block realact` ran the migration; `unit(act.f32 − whiten_mu)` reproduces her shipped `direction` at min cos 1.0000000000 over all 512, read back off the volume. Three rows (26, 32, 360) have two positive roots and are flagged `exact_ambiguous`, not resolved. See `features/README.md` and `SMOKES.md`. |
| **Resolving `corpus.revision`** | Pinned to the literal `main` with the mechanism in place and empty refused. The dataset's commit sha still has to be looked up and put there. |
| **H7's actual threading** | Per-corpus `block`/`stride` is declared and now REFUSED when it differs, rather than honoured. Doing it properly means every consumer (`top1_act`, `sae_self` ×3, `build`, `gcg`, `mu_diag`) reading the producer's recorded geometry instead of `common.SCAN_BLOCK`, because they reconstruct window ids to join on. Eleven call sites; a half-applied thread silently misaligns joins, which is worse than the refusal. **Blocks scanning `celeste-train10m` through our `scan`** — the plan's §2.4 search baseline. |
| **Ruff on Ari's new files** | `features/{corpus_train_parity,doc_dedup,ngram_overlap,registry}.py` carry pre-existing `B905`/`E702` findings. Not touched, so not fixed. |

---

## The 2026-09-21 independent review (`infra/2026-09-21_pipeline-branch-review.md`)

All eight findings fixed, one commit each, each with a CPU check where one is possible.

| # | what it was | fix | check |
|---|---|---|---|
| H1 | `r.get("sae_key", sae_key)` defaulted every UNKEYED row to match whatever `--sae` was typed. Both production sets carry the field on no row, so `--sae <2M> --set 2026-09-16_v1` selected all 512 of the 131k rows, every id a valid 2^21 index — the silent failure the guard exists to stop, on the paper's own set | an unkeyed row is selectable only against a DECLARED dictionary (`declared_sae_key`, from `storage.json` then the `heldout:` entry); undeclared or mismatched refuses, naming the row count. `sae_key:` declared on all three pre-field sets | `check_sae_key_selector`, three outcomes + resolution order |
| H2 | `centred.py` resolved the TARGET through the run's mu and hardcoded `stats_mu` for the ACTIVATION side, so every `rl-last16` run compared two different means and `centred.json` named the wrong one; at `--mu none` on a raw set it was a one-sided cosine | `_load_dirs` returns the mu, the activation side loads it, the summary names it, and `--mu none` refuses rather than writing a one-sided number under a centred name | `check_centred_uses_one_mu` (ast) |
| H3 | `od_sae` lacked `keep_existing`, and `sae_dir` is the parent of `examples/`, `examples_4m/`, `examples_docmax/`, `random_pool/`, `repo_examples/`, `top1_act/` — so `stats --force` deleted every scan and autointerp product under that dictionary | `keep_existing=True`, and the pre-flight guard moves from the directory to the four arrays `stats` owns | covered by the existing `check_outdir_keep_existing_and_section` |
| H4 | `draw_sae2m` and `heldout_v2` wrote no `storage.json`, so their sets were born unreadable — falsifying this branch's own "every set drawn after 2026-09-21 writes it" | both write it (`dirs_only` / `unit` with per-family means) | `check_every_set_writer_writes_the_contract` (ast, all three drawing tools) |
| H5 | `scan_dir` / `sae_examples_dir` keyed by set, not corpus; the plan's §2.5 and §3.4 scan one set over two corpora and collide | both take `corpus_name`, empty resolving to today's path; readers follow | `check_corpus_axis` |
| H6 | `spawn.py` forwarded `--corpus` unresolved, so a spawned scan silently used the default corpus | `corpus` joins `_LOCAL_ONLY` and spawn resolves it | `check_corpus_axis` |
| H7 | per-corpus `block`/`stride` declared, validated, and read by nothing — a scan of Ari's 32/8 corpus would cut 64/16 under a README claiming 64/16 | **the reviewer's second option**: `assert_corpus_geometry` REFUSES such a corpus in `scan`/`stats`; the false comment is corrected. Threading is NOT done — see below | `check_corpus_axis` (the refusal) |
| H8 | `top1_act` and `corpus_top1_activation` still selected on the family label | both filter on the row's own key | covered by H1's check for the shared helper |

The reviewer also confirmed, independently: B3/B4 closed, no double-centring path, `dirs_for`'s
unit branch bit-identical on the real 1536×5120 array, re-derive cannot damage its source, and the
cherry-picks lost nothing.

## 2026-09-21, second chunk: the eval-1 frozen target blocks

`features/heldout_v3.py` (new product), `draw_sae2m --sides enc,dec`, and `check` opening the set
directories it used to only name. Five set directories on the volume, `$0.131`, all of it the 2M
draw; `features/README.md` carries the layout, the rebuild commands and the U1 evidence, and
`SMOKES.md` the run record. Two defects were found in the process, both on paths eval 1 runs:

* `draw_sae2m` drew the fit/report `side` column at the POST-`--include` `n`, so
  `--n 512 --stratified --include <64 ids>` built 448 labels for 512 features and indexed off the
  end. Nothing had ever run that combination — the 64-set is the source of the include list, not
  a consumer of it.
* `sae_side` was read by `common.sae_rows_of`, `repo_examples` and `sae_self`, and **written by
  nothing**. `draw_sae2m --sides` is now its writer; a row without the field still reads as `enc`,
  so no existing set moves.

Deviations from plan §2.1, recorded: the set is FIVE directories rather than one (a directory
carries one storage contract and these blocks do not share one — the brief allowed this), and it
omits §2.1's "ours" realact sanity block (rows 512-1023) and her `sae2m_enc`/`sae2m_dec` blocks,
which the brief did not ask for. The 131k `sae` block comes in via `--block ctrl` from
`2026-09-21_v1raw` rather than being redrawn.

## Deviations from the plan, recorded

1. **A mu is a path, not a name.** §1.1's `mus:` registry was built and then replaced on Tomáš's
   2026-09-21 instruction. Consequence: `mu_512` and `mu_long` have no config entry any more —
   `mu_512.f32` is still written by `targets` as a diagnostic, and "centred on a mean nobody here
   holds" is the `unknown` label rather than a named entry.
2. **The old primary is `mu: unknown`, not `null`** (Tomáš, 2026-09-21), a THIRD state between an
   absent key (nobody considered it) and an established path/null. The plan declares `none` from a
   recollection that the 09-10 chain took uncentred input; against that, Celeste's ORIGINAL
   convention was targets `unit(act − whiten_mu)` with a raw scorer (she diagnosed the asymmetry
   herself on 09-18, after this checkpoint launched), and the one number it has on our volume —
   cos_raw 0.5076 on rows 0-7 of `2026-09-16_v1` — was measured against `stats/mu.f32`-centred
   rows. `mu_for` therefore refuses to pick one and every run must pass `--mu`. The §1.6 smoke
   settles it empirically with two old-primary arms on the same rows (`--mu null` and
   `--mu base/{base}/stats/mu.f32`); the plan's reproduction check applies to the stats_mu arm
   only. **Whichever arm scores higher becomes the declared `mu:`, with the number recorded.**
3. **`storage.json`, not `index.json`.** §1.2 puts the storage contract in `index.json`; that file
   is `OutDir`'s file→metadata map and a non-file key would break its README table. It is an
   ordinary product file, listed in `index.json` like any other, with a config fallback for the
   sets drawn before it existed.
4. **`dirs_for` takes `notes`**, so the "label, don't refuse" rule for an `unknown` mean has a path
   into the product README; the plan's signature returns only the array.
5. **`--dirs-from` now requires a stated contract.** A directory with neither a `storage.json` nor
   a `heldout:` entry is refused instead of assumed. Deliberate, and a behaviour change.
6. **`rollouts_* --maemm <rl-last16> --set 2026-09-16_v1` now REFUSES** (whiten_mu vs stats_mu on a
   set that cannot be re-centred). Re-derive the set, or pass `--mu` and wear the deviation.
7. **`targets` loads the SAE encoder-only.** Not in any defect list; same reasoning as D3.
8. **D4 changes `span_text` on 21/512 rows**, so `--re-derive`'s byte-identity assert exempts
   exactly the clamped rows and requires the new text to be a SUFFIX of the old. Rows gain
   `L_shown` beside `L`.

---

## Rebase onto `arb/main` bdb0705

`evals/pipeline-v2`, 2026-09-22. `evals/pipeline` (b017f5d) and the four commits
`evals/pipeline-autointerp` gained after the `93fd441` merge (2d9a15e..8c2a6bb), replayed onto
`arb/main` at bdb0705. 63 commits, none dropped, original messages kept. `evals/pipeline-ood` is
NOT in this branch yet.

The three cherry-picks of Ari's work at the base of `evals/pipeline` (`640238c` / `6ad2872` /
`d6fff69`) were **skipped**: his `b60330d` on main carries the same content. Verified rather than
assumed — `features/{doc_dedup,ngram_overlap,corpus_train_parity}.py`, `precompute/{scan,stats,
top1_act}.py` are byte-identical blobs at d6fff69 and at bdb0705.

**Residue the three carried that `b60330d` did NOT take**, and where it is now:

| from | residue | where it went |
|---|---|---|
| `640238c` | `patchscopes.cell_name`: the prompt-specific floor cell (`FLOOR_CELL-<p>`), after a measured P1 run overwrote two P2 floors | survives — this branch's own `patchscopes.py` work is replayed over main's `PS_LAYERS` change, and the two are disjoint |
| `d6fff69` | `config.yaml`: `seed: 20260920` on `2026-09-20_sae2m_2k` (`scan` reads it and KeyErrors without one), `sae_min_fires`, `sae_strata` | re-applied in the `fb4e544` conflict resolution, row 1 below |
| `d6fff69` | `spawn.py`: `corpus_name` in `DEFAULTS` | already on main, inline; main also repeats `"subset"` twice in that dict literal, deduped here |
| `640238c` | `autointerp/sae_self.py`, `features/draw_sae2m.py` (in Ari's `a07cc70`, never in our cherry-pick) | not lost: both files are rewritten by this branch's own commits |

### Conflicts and how each was resolved

| # | commit | file | main (bdb0705) | ours | resolution |
|---|---|---|---|---|---|
| 1 | `fb4e544` | `config.yaml` | `2026-09-20_sae2m_2k` carries only `imported: true` | adds `storage: dirs_only`, `mu_stored: null`, keeps `seed: 20260920` | **ours** — storage/centring is this branch's layer, and the `seed` is `d6fff69` residue `scan` needs |
| 2 | `b43eb9b` | `features/spawn.py` | `DEFAULTS` has `corpus_name` inline, and `"subset"` twice | adds `corpus_name` again plus `centering`, `re_derive` | **merge** — main's inline `corpus_name`, ours for the two new flags; main's duplicate `"subset"` dropped |
| 3 | `2c114cb` | `features/spawn.py` | as above | renames `centering` → `mu` | **ours** for `mu`, main's `corpus_name` kept |
| 4 | `3fc101d` | `features/spawn.py` | as above | adds `"corpus": ""` | **both** — `--corpus` (keyed, ours) and `--corpus-name` (directory, Ari's) are different flags |
| 5 | `08cbafd` | `config.yaml` | comment on the `sae2m_2k` family label, still carrying the OPEN question | same comment, SETTLED by reading `ids.jsonl` off the volume (2,000 rows `family: sae2m_enc`, no `sae_key`) | **ours** — same claim with the evidence attached |
| 6 | `d548f6a` | `precompute/common.py` | `SCAN_BLOCK = 64`, per-corpus-override comment deleted | the H7 block comment: geometry is a CONSTANT, `corpora:` declares per-corpus block/stride, `assert_corpus_geometry` refuses a mismatch | **ours** — see H7 below |
| 7 | `1b032fa` | `precompute/modal_app.py` | `PRODUCTS` gains `draw_sae131k` | `PRODUCTS` gains `heldout_v3` | **both** |
| 8 | `7ae3a1f` | `SMOKES.md` | this branch's own eval-1 section (reordered by the merge flattening) | the `results/` driver section | **both**, in the branch's own chronological order |
| 9 | `8d307fd` | `autointerp/build.py` | Ari's `if is_nla:` branch — slice to the `<explanation>` body via `explanation_token_mask`, stamp `tag_status`, re-rank rollouts by IN-BODY peak | `render_example(..., rel_fallback=rel_fallback)`, relative marking for the generated-text arms | **merge** — Ari's branch kept, `rel_fallback` threaded through BOTH `render_example` calls |
| 10 | `d95dd94` | `autointerp/build.py`, `unit_smoke.py` | Ari's `explanation_token_mask` (already in, from row 9) | our independent `nla_body_tokens` — the same fix, written differently | **Ari's, and this is a decision against the letter of the resolution list.** See below. |
| 11 | `e8bb816` | `SMOKES.md` | — | append | **both** |

Everything else auto-merged. `rollouts_nla.py`, `patchscopes.py` and `config.yaml` merged
**textually clean** in both directions, and three of the worst problems were in exactly those
files — see "what merged clean and was still broken".

### Row 10: why Ari's body slicer won although the list says ours does

The resolution list puts "NLA-A body slicing" on our side. Both sides fixed the same defect in the
same loop, independently, and Ari's is the better of the two:

| | Ari (`explanation_token_mask`) | ours (`nla_body_tokens`) |
|---|---|---|
| unclosed tag | body = everything after `<explanation>` | **the FULL raw decode**, counted |
| boundary token | dropped (containment) | kept (overlap) |
| bookkeeping | `tag_status` per example, 3 states | one boolean |
| ranking | rollouts re-ranked by IN-BODY peak | unchanged |
| shared with | `rollouts_nla`, i.e. one slicer for every judge-facing consumer | `build.py` only |

Ours falls back to the whole decode exactly where the defect lives — an answer that ran into
`max_new` reaches the judge with its opening tag and chat preamble intact, which is what Juan's
review was about. Keeping two slicers in one file was the other hazard. `nla_body_tokens` is
**dropped**; both of our checks are **retargeted** at the surviving function rather than deleted,
and the `unclosed` case is now the thing they pin, so the difference between the two
implementations is what goes red if anyone reintroduces the weaker one.

### What merged clean and was still broken

None of these produced a conflict marker. Each is a change on one side disagreeing with a change
on the other, in a different file.

1. **The merged tree could not `load_config` — for every product, not just the NLA ones.** main
   deleted `nla.sampling` (shared sampling across the baselines); our `NLA_KEYS` still demanded
   it. Resolved in main's direction; the `sampling` guard moved to the shared `rollouts:` block
   rather than being deleted with the key it guarded.
   **`min_new` moves 0 → 16** as a side effect, which is in neither commit message. Nothing on the
   volume is affected — both scored `nla-av` sets predate bdb0705 — but the next NLA run generates
   under a 16-token floor, against a deliberate `min_new: 0` ("the NLA answer is short, so a floor
   would only pad it"). **Open for Tomáš.**
2. **`2026-09-21_sae131k_2k` (main's set) declared no storage contract**, so
   `_check_heldout_storage` refused it. `storage: dirs_only` / `mu_stored: null` /
   `sae_key: qwen36-27b/l42-1b`, which is what `draw_sae131k` writes.
3. **…and it would have become `default_heldout`.** It carries no `imported:`, so it sorts last
   among the non-imported sets and becomes the default of every product called without `--set`,
   moving them off `2026-09-16_v1` — the set the tables are built on. Invisible on main, which has
   no results driver. `imported: true` added. **Open for Ari:** if the 131k set IS meant to be the
   new default, that is the line to delete, and `results/sanity.yaml` moves with it.
4. **`features/draw_sae131k.py` could not run.** `_finish` gained a `peak16` parameter here after
   that file was written against the 16-argument signature, so the positional call bound `cuts` to
   `peak16` and raised `TypeError` on `meta_extra`.
5. **…and would have selected nothing if it had.** It overwrote `_finish`'s per-row `sae_key` with
   the bare `"l42-1b"`; `sae_rows_of` matches the full key, and because the rows ARE keyed the
   unkeyed branch's loud assert never fires. `scan` would have run with `n_feat = 0`.
6. **…and wrote no `storage.json`**, which is why the set needed a hand-written config entry.
7. **`rollouts_nla.write_nla_readme` raises `NameError`.** It reads `samp['temperature']` and
   never binds `samp` — the shared-sampling change moved the dict into `run`. It is called AFTER
   the generation. `ruff check` on bdb0705's own file under bdb0705's own `ruff.toml` reports it
   three times (F821); nothing ran it.
8. **"recorded on the summary, not used" was true of a `print()` and nothing else.** The shipped
   `generation_config.json` constants now reach the summary as `shipped_generation_config`, and
   the asserts removed in the same commit are replaced by that record.
9. **D6 had three set-writer tuples.** `modal_app`'s knew `heldout_v3`, `spawn`'s did not, neither
   knew `draw_sae131k` — and `spawn` is the path that bypasses `modal_app.main`, i.e. the one that
   let a set reach the volume undeclared. One tuple now, `common.SET_WRITERS`, read by both.

### H7: still open

`assert_corpus_geometry` is **kept**. Ari's `scan` does not handle geometry — it cuts 64/16
whatever the corpus declares and writes a README saying 64/16 next to a corpus README saying 32/8
— so the condition under which the refusal was to be dropped does not hold.

The cost is that `features/pipeline.py` shipped a default this pipeline cannot run
(`--corpus-name train_parity_10m`, cut at 32/8 by `features/corpus_train_parity.py:60`). **The
chain's default is now the plain `corpus/`** (`heldout16m`, 64/16) so the one-command chain works;
`--corpus-name train_parity_10m` still refuses, with the reason. Threading `block`/`stride` from
the `corpora:` entry through the eleven `windows_of(` call sites, plus the window-id join check
stored top-k lists need, is the real fix and is **not done**.

### C7: `scores_dir` now spells the tag where `rollout_stem` does

Ours, not main's. `scores_dir` took no `tag`, so a tagged score run could only name itself through
`--score-name <set>__<tag>` and landed on `<set>__<tag>__<engine>` while its rollouts were at
`<set>__<engine>__<tag>`. `results/common.parse_scores_dir` already read either order — its
docstring records the six eval-1 arms that went into the paper's CSV as HF when they were vLLM —
but the writer still disagreed with itself. `tag` is now the fifth parameter, `write=True` marks
the writing call sites, and a reader falls back to the legacy spelling when the canonical path is
absent. No untagged and no HF product moves: at `hf` the two spellings are the same string.

### Not re-measured

The NLA arm-A marking rate recorded on this branch (26/32) was measured on the **unmasked** decode
and no longer describes the composed path: Ari's body mask now runs before `rel_fallback` decides
whether a block is bare. It needs one NLA block rebuilt and `marking` re-read from `build.json`.
No Modal run was made in this rebase.
