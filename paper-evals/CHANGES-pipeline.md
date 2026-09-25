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

---

## `evals/pipeline-ood` onto the rebased branch

2026-09-22, the second half of the same rebase. `evals/pipeline-ood` (48ee63a) replayed onto
`evals/pipeline-v2` at b74872f: the **28 non-merge commits of `e09a4a9..48ee63a`**, original
messages kept, **none dropped** (each verified present by subject after the replay).

The replay base is `e09a4a9`, the OOD branch's merge of `evals/pipeline-results`, NOT its fork
point `44ca67c`. Everything before that merge — `44ca67c` itself and the results driver's
`7ae3a1f` / `86a38f9` — is already in the base branch, and replaying it would have re-applied two
commits whose `SMOKES.md` and `results/` content were conflict-resolved there.

### Conflicts and how each was resolved

| # | commit | file | base (pipeline-v2) | OOD | resolution |
|---|---|---|---|---|---|
| 12 | `70ac200` | `SMOKES.md` | the eval-1 / eval-2 / results sections | the OOD pilot section | **both**, appended in order |
| 13 | `2daf562` | `results/selftest.py` | the results-driver and autointerp checks | `check_ood_scan_key`, `check_ood_arm_table` | **both** — two append-vs-append hunks, the check bodies and the registration list |
| 14 | `6e27191` | `precompute/common.py` | `scores_dir(..., tag, write)` — `tag` is the RUN tag, plus the legacy-order read fallback | `scores_dir(..., tag)` — `tag` is the SCORE tag | **merged into two axes**, see below |
| 15 | `6e27191` | `precompute/score.py` | passes `args["run_tag"]` | passes `args["score_tag"]` | **`C.score_tag_of(args)`**, which composes both |

### Row 14/15: the same parameter, twice, meaning different things

The two branches each gave `common.scores_dir` a `tag` parameter — same position, same name,
within a day of each other — and meant different axes by it:

* `--run-tag` (ours) selects which rollouts **file** is scored. Two run tags scored under one
  convention are two different inputs.
* `--score-tag` (OOD's) names only the **output**: one rollouts file scored again under a second
  convention, or for a column the first run did not have. `cos_asym` is why it exists. Using
  `--run-tag` for that instead sends `score` looking for a `<set>__<engine>__<tag>.jsonl` that was
  never written, which is how the mu-stats arm failed on 2026-09-21.

Taking either resolution alone **loses the other axis with no error anywhere**: the second run
writes the first run's directory, `--force` replaces it, and the only surviving trace is a README
naming a different rollouts file. `common.score_tag_of(args)` composes the pair, run tag first so
every score of one rollouts file sorts together, and is now the only way any call site builds that
component. `centred`, `autointerp/sae_self` and `autointerp/build` previously passed the run tag
alone and so could not have found a `--score-tag` product at all; they now compose it too.
`check_both_tag_axes_reach_the_scores_path` is mutation-tested against losing either side (M12,
M13) and against a call site building the tag by hand (M14).

### The three interlocks the previous section flagged, checked

All three were named as what the OOD branch would meet. Outcome:

1. **`scores_dir`'s tag** — a real collision, resolved above.
2. **`C.SET_WRITERS` and its file map** — the OOD branch adds no set writer (`corpus --arm` and
   `targets --arm` draw into products already under D6, and `ood_selfcheck` writes nothing), so
   the tuple and the map are unchanged. But `check_one_set_writers_tuple` **fired a false
   positive** on `modal_app`'s `--arm` ownership assert
   (`product in ("corpus", "targets", "ood_selfcheck")`) because it matched on `"targets"` alone.
   Narrowed to require BOTH `targets` and `draw_sae2m`, which is the D6 tuple's shape; M4 and M11
   still go red. A check that fires on the wrong thing is a check nobody will keep.
3. **`check_sidecar` returning a pair** — the OOD branch does not touch `rollouts_nla.py` and
   unpacks `check_sidecar` nowhere. No interaction.

### What merged clean and was checked rather than assumed

None of these needed a resolution; each is recorded because the previous section's lesson was that
this class does not announce itself.

* **The scan axis composes rather than collides.** `common.scan_dir(base, set, root, corpus_name)`
  is unchanged; `scan.py` builds the whole key `<corpus>[__<M>m][__<tag>]` and passes it as
  `corpus_name`, and `results/ood.parse_scan_dir` splits it back on `__`. The mean/bound axis sits
  inside the corpus component rather than beside it, so the two layers stack.
* **`assert_corpus_geometry` is reached by the OOD launcher**, not bypassed:
  `modal_app.main`'s `--corpus` resolution loop calls it once per named corpus before spawning.
* **All 23 OOD corpora declare 64/16** in the `corpora:` section this branch introduced, so the
  refusal accepts every one of them. `celeste-train10m` (32/8) remains the single blocked corpus —
  H7's status is unchanged by the OOD work.
* **The three OOD held-out sets declare the storage contract** (`storage: raw`, `mu_stored: null`,
  `imported: true`) and pass `_check_heldout_storage`; `default_heldout` is still `2026-09-16_v1`.
* **`cos`, `cos_centred` and `cos_asym` coexist** as three columns of one centred `score` run;
  `cos_asym` is not a rename of either.
* **`results/ood.rollouts_rel_of` reads the rollouts path out of the scores README** instead of
  reconstructing it from the directory name, so the composed tag does not reach it.
* Every `C.<name>` reference in `precompute/{nll,ood_selfcheck}.py`,
  `reconstruction/stats_ood.py` and `results/ood.py` resolves against the merged `common.py`, with
  no call passing more positionals than its target takes; 46 of the 62 modules import cleanly and
  the other 16 fail only on absent optional third-party packages (`modal`, `torch`, `pandas`).

### The lid fix, into the layer both halves share

`11b7cd0` fixed the R3 language-id column in `results/ood.py`: a scores directory's name does NOT
determine its rollouts file, because `--score-tag` makes them differ on purpose, so the path comes
from the scores README's own `- rollouts:` line. **`reconstruction/stats_ood.rollout_texts` still
rebuilt it from its own `--stem`** and so still had the defect its sibling had been fixed for --
`--stem <set>__vllm__asym` finds the scores, looks for a rollouts file nobody wrote, gets nothing,
and the column comes out EMPTY with no error. An empty R3 column is worse than no column: it reads
as "the answers were not in the arm's language", which is the very claim `ff80d04` retracted.

The rebase makes it worse before it makes it better -- after `score_tag_of`, BOTH `--run-tag` and
`--score-tag` are in the scores name, so there are now two ways for a rebuilt stem to be wrong.
The rule is now `stats_ood.rollouts_rel_from_readme(vol, scores_rel)`, in the lower layer
`results/ood.py` already imports, with the `--stem` composition kept only as the fallback for
products written before READMEs carried the line. `vol` is duck-typed; both `Vol` classes have
`.get`. Mutation-tested in `stats_ood.py selfcheck`:

  M15 rollout_texts rebuilds the path from --stem   -> RED
  M16 the README path is not made root-relative     -> RED
  M17 the pre-README fallback removed               -> RED

`results/selftest` no longer pins `results/ood.ROLLOUTS_RE`; it asserts the copy is GONE, because
two copies is how the two halves diverged in the first place.

### A finding that did not survive checking

An automated cross-check reported that the OOD arm corpora are never registered in `corpora:`, so
`assert_corpus_geometry` falls through for all 23 of them. **It is wrong**, and the way it is
wrong is worth recording: `config.yaml` does declare only two `corpora:` entries by hand, but
`load_config` SYNTHESISES one per `ood_arms:` entry (`common.py:153-172`, from the OOD branch
itself) so the ladder has a single source. `corpus_key_of_dir("tha_Thai")` resolves to
`ood_tha_Thai` and the refusal covers it at 64/16. Reading the YAML is not reading the config.

What is genuinely uncovered is an UNREGISTERED directory: `corpus_key_of_dir` returns `""` and
`assert_corpus_geometry` then returns 64/16 without asserting. That is V's own documented choice
-- a corpus nobody declared is assumed to be the pipeline's geometry -- and it is the honest gap
to close when H7 is done, not before.

### Left open

* **`top1_act` and `gcg` address a scan as `scan_dir(base, set, root, corpus_name)` with the raw
  corpus name**, so neither can name a bounded (`__4m`) or mean-tagged (`__mu-whiten`) scan. Not a
  rebase defect — it is the same on the OOD branch alone — and currently harmless, because both
  products need SAE rows and an OOD set has none. It becomes real the first time a bounded scan of
  an SAE set is wanted.
* H7 unchanged: the geometry is still not threaded, and `--corpus-name train_parity_10m` still
  refuses.

---

# 2026-09-23 — M0a: the conventions layer (branch `evals/m0a-conventions`)

Off `6cdd429`. This is the module every other eval module rebases onto, so the whole of it is
things that had to mean ONE thing before eleven builders started measuring in parallel. Plan:
`evals/2026-09-23_implementation-plan.md` §M0a; blockers 1-3, 5, 6 of the Fable review of it.

## 1. The additive product write — the interlock is lifted

`OutDir(keep_existing=True)` used to `copytree` the whole accumulating product directory into
`<name>.tmp-<date>`, then `rmtree` the original and rename the temp over it. The temp name carried
only the date, so **two concurrent runs of one MAEMM shared one staging directory and the later
rename silently discarded the earlier file** — `SMOKES.md:4349-4356` records exactly that. Now:

* `__enter__` creates the product directory if it is missing and stages this run's files in a temp
  unique to the process (pid + random, and still spelled `.tmp-` so the listings that filter on
  that substring are unaffected). Nothing that exists is read for correctness, copied or removed.
* `__exit__` moves **only this run's own files** in, one `os.replace` each, then merges its entries
  into `index.json` and rewrites `README.md`, both through a temp-and-replace, under a short-lived
  `.index.lock` dotfile that is gone again before the call returns.

The committed directory is byte-identical to what the old path produced for a single writer: same
files, same index mapping, same README layout. **A product written by `6cdd429` is therefore read
and scored by this code unchanged.**

**But mixed old/new CONCURRENCY is still unsafe, and this is a launch rule, not a caveat.** The
additive side removes nothing (`check_additive_removes_nothing_and_the_legacy_path_is_the_hazard`
asserts that by ast on `_commit_additive` and `__exit__`), but the hazard was never on that side:
it is the OLD path's `rmtree(path)` + `rename(tmp, path)`, and no change here can make that safe
from outside. Measured, both orders: the legacy writer exits 1 and the directory is left holding
neither writer's rows. So **while any job is still running on `6cdd429`, no job on this code may
write the same MAEMM's `rollouts/`** -- check for a `rollouts.tmp-*` sibling of the directory
before launching. Once every writer is on this code, concurrent writers are safe, which is what
`check_two_writers_into_one_product` covers. `rollouts_vllm` and `rollouts_nla` now pass
`keep_existing` unconditionally for the shared `rollouts/` directory; gating it on the directory
already existing left the FIRST two concurrent writers on the old rename path.

`--rows` chunks of one product under one `--run-tag` write `<stem>__rows<spec>.jsonl` side by side
in that one directory (`common.rollout_chunk_stem`), and `common.read_rollouts` concatenates them
into a single product for `score`, so nothing downstream — `scores/`,
`results.common.discover_sources`, either OOD reader — learns the generation was chunked. A
whole-set file beside chunks of the same stem is a refusal; so are overlapping chunks and chunks
whose summaries disagree on the experiment. A run over the whole set keeps its historical path.

## 2. One scoring constant

`common.score_mu(cfg, base)` reads `bases.<base>.whiten_mu` — a key that already existed — and is
the mean **both** arguments of every centred cosine are taken about, in every product. It is
**decoupled from each MAEMM's `mu:`**, which stays the injection convention. Before, the reported
statistic's mean was the run's injection convention, so the old primary (`mu: null`), the base
control and the NLA arm had no centred cosine at all and no two arms could be differenced: a
difference of two cosines about two different means is not a difference.

* `score` no longer threads `mu_for` into the centred arm and **refuses `--mu`**. A set that is
  not `storage: raw` is NaN there (no `act.f32`, no centred direction at any mean) — the honest
  answer, not a defect.
* `scan --centre` takes both sides about the same constant: one broadcast subtract per flush on
  the window side, `dirs_for` at the same mean on the target side. It refuses `--mu`, requires
  `--run-tag` (a centred and an uncentred scan of one (set, corpus) are separated by that tag
  alone) and requires every target set to be `storage: raw`. The product README states its mode.
* `sae_self` stays uncentred and the reason is its own: its metrics are activations, not cosines.
* Config: the base control `2026-09-16_base-control` declares the 27B whiten_mu path instead of
  `mu: null`, so a control run is no longer a recorded DEVIATION and panel a row 7 differences two
  arms under one mean.

## 3. One best-of-k estimator

The unbiased order statistic, `sum_i x_(i) C(i-1, k-1) / C(n, k)` over all n draws, replaces the
disjoint-group mean everywhere: `results.common.bo_unbiased` / `bo_ladder` for the results layer,
`precompute.common.bo_ladder` for what the Modal container ships. The two layers cannot import
each other, so that duplicate is deliberate and is held to one number by
`results/selftest.check_one_bo_estimator`. **A `bo_<k>` written before 2026-09-23 is the other
estimator and is a different number at every k < n.**

The **fired indicator** is the same estimator applied to the 0/1 gate crossings: for a 0/1 vector
the unbiased best-of-k is `1 - C(n-m, k)/C(n, k)`, which is P(at least one of k draws fires). So
`item_fired` is its k = 1 cell and `fired_any` its k = n cell — the two names that existed keep
their meaning — and the ladder in between is what panel b's `sae.l131k.ex.fired.bo8.q<q>` keys
print. It reaches the sanity registry as `fired.bo<k>`.

## 4. Deleted, not parked

* `--re-derive` entire: `_re_derive_check` and its constants, the guard block in `targets.run`,
  the flag in `modal_app` and `features/spawn.py`.
* `mu_stored` / `family_mu` entire: the config keys (22 lines over 13 `heldout:` entries), their
  validation, `mu_of_family`, the third branch of `mu_for`, and the mismatch assert and
  `unknown`-labelling in `dirs_for`'s unit branch. A stored unit direction cannot be moved to
  another mean without `||act||`, so the layer could only ever refuse; what replaces it is that
  such a set is served exactly as shipped and has no centred cosine. An old `storage.json` on the
  volume that still carries the fields is read, not refused; a config entry that declares one is
  refused with the reason.
* `results/sanity.yaml`: 37 checks over four sections down to 5. Everything pinned to a dated run
  of this pipeline is gone, for two independent reasons: those numbers were measured under the old
  centring and the old bo-k, so a green gate against them would mean the change did not land; and
  the schema can express a NUMBER and nothing else, so "the structural gates stay" never meant
  "keep part of this file" — the structural checks are in `precompute/unit_smoke.py`,
  `results/selftest.py` and `results.faithfulness.cosine_reader_check`, and they stay. What
  remains is Celeste's model card, external and labelled; the two 2M fired rates are demoted to
  `compare: false` (recorded FLAG, unresolved, one question to Celeste) and the `realact_long`
  gate resolves `absent` until that block is re-forwarded raw.
* The old primary's `mu-none` / `mu-stats` arms: the two-arm row-by-row figure and its selectors.
  The `--run-tag` / `--score-tag` machinery is untouched — those are two live axes.
* The `0.5076` item, from `sanity.yaml` and the selftest fixture; `config.yaml` records it as
  retired rather than pending.

## 5. Asserted

`dirs_for` now checks, **as a post-condition on the array and reading `family_kinds` directly
rather than through the function that computed the mask**, that a mean reached exactly the rows
with a raw activation. The set this is for is `2026-09-21_v3_ctrl`, which mixes `random` draws and
131k encoder columns in one `storage: raw` directory: `cos(h - mu, encoder column)` is a one-sided
number wearing a centred number's name.

## 6. `corpora.celeste-train10m`: 32/8 → 64/16

Nothing was built at 32/8. The corpus product is a geometry-free token stream plus `docs.jsonl`;
windows are cut by the scan at `SCAN_BLOCK`/`SCAN_STRIDE`, and no scan of this corpus exists — so
the declared geometry described a cut nothing performed, and `assert_corpus_geometry` refused
every scan of it (correctly). At 64/16 it matches `heldout16m` and the two corpora's cosines are
comparable, which is what panel a row 3 differences. `check_corpus_axis` now holds EVERY
configured corpus to the pipeline's geometry and keeps the refusal on a corpus built to disagree.

## What other modules must know

`common.best_of_k_means` → `common.bo_ladder` (and the value changes).
`results.common.best_of_k_means` → `results.common.bo_ladder` / `bo_unbiased`.
`common.mu_of_family` gone. `common.set_storage` returns `{storage, source}` only.
`common.score_mu` is new and is where the centring mean comes from.
`_load_targets` in `scan.py` returns 4 values. `score` refuses `--mu`.
`OutDir(keep_existing=True)` is additive; `gcg` can opt into it by passing the flag (M3).

---

# M6 — SAE autointerp (branch `evals/m6-autointerp`, 2026-09-23)

Module M6 of `evals/2026-09-23_implementation-plan.md`, branched from `6cdd429`. **Nothing here
was run on the Modal volume except one $-capped API check** (below); no GPU, no scan, no corpus.

## What changed

1. **Two corpora, threaded end to end.** `--corpus-name` (the SHOWN examples) and
   `--test-corpus-name` (the Delphi test windows) are now separate parameters of the autointerp
   entrypoint (`modal_app.py`), and `build.run` resolves each side's `examples/`, `examples_4m/`
   and `examples_docmax/` independently. Before this, `corpus_name` was absent from the entrypoint
   altogether, so `args.get("corpus_name")` at `build.py:984` was always `""` and one `ex_dir` fed
   both the explainer and the judge. `sae_self`'s three corpus-side stages (`random_pool`,
   `examples_4m`, `examples_docmax`) read `--corpus-name` and key their output path by it, with
   `""` resolving to today's path so nothing already on the volume moves.
   - `_Corpus` is now per corpus: a stored `(doc, start, len)` indexes into ITS OWN `tokens.i32`.
   - Document-level disjointness (A4) is asserted per feature within one corpus; across two, the
     exclusion sets are empty and `build.json`'s `disjointness` field says the separation is the
     corpora's. Doc ids are not comparable across corpora and are no longer compared.
   - `build.json` gains `shown_corpus`, `test_corpus`, their `corpora:` keys, `two_corpora`,
     `disjointness`, and a `pools` block naming the corpus and directory of every pool (asked for
     by M9 so a per-feature page can label C16's corpus).
2. **Arms.** `C16` is now the top 16 by peak activation with ONE WINDOW PER DOCUMENT
   (`examples_docmax`) on the shown corpus, which is what spec §3 defines it as; the pilot's
   window-ranked arm survives as `C16-win`. `C4` and the N = 40 point are dropped. New:
   `M-jac16` (greedy farthest-point on content-word Jaccard of the rollout texts, CPU) and
   `M-cos16` (greedy farthest-point on mutual cosine of `score`'s stored `best_act.f16` residuals
   with `bases.<base>.whiten_mu` subtracted — no extra forward pass), plus `NLA-1`, the appendix's
   one-output row. `DOCMAX` is kept as `C16`'s pre-09-23 label and running both names refuses.
   The content-word tokenisation is copied from `runs/autointerp_pilot_diagnosis.py:33-48` so the
   selection distance is the same quantity the sample-diversity analysis reports.
3. **One run directory for all seven arm-variants.** `run --build-dir-nla` lifts the NLA arms out
   of a second build (the verbalizer is a different `--maemm`, so they can never share a build)
   and scores them on the PRIMARY build's test items. Only arms whose examples are all rollouts
   may be lifted, asserted arm by arm, so A4 cannot be broken silently. The two builds' base, set,
   sae, gate, seeds and feature list must agree.
4. **`stats.paired()` asserts its intersection.** New `stats.pairing()` returns what each side
   covered and what each lost; `paired()` asserts the pivot's row count equals that intersection
   and takes `require_complete` for a caller that needs a complete pairing. The contrasts table
   prints `n paired` and `n a/n b`.
5. **Every table carries the protocol.** `stats.py` prints a `Protocol:` line — judge, fuzzing
   protocol and its few-shot count (from `costs.json`, which already recorded it), and chance =
   0.5 with the positive/negative counts it comes from. `n_shown_exceeding_corpus_peak` is now
   counted per arm in `build.json` and printed as a column (closes the gap `SMOKES.md:3731`
   records).
6. **`results/autointerp.py` writes `paper/numbers/cells.csv`.** `--cells <path>` rewrites this
   run's `ai.*` rows in place by key and appends new ones; `cells_rows()` maps arm names to the
   writing plan's key slots (`M` → `mtop16`, `R-shuffled` → `floor`, …) and refuses to guess a
   slot it does not know. Refusals are written both ways — the dropped convention as `.det`/
   `.fuzz` and the chance-imputed one as `.detrc`/`.fuzzrc`, which the driver already computed.
   TPR and TNR are aggregated beside balanced accuracy (spec §3's failure analysis is about the
   positive half). Per-band TPR keys are NOT written and the driver says so on every run.
7. **`config.yaml`'s `autointerp:` block** gains `examples_corpus: celeste-train10m` and
   `test_corpus: ""`. `examples_corpus` RECORDS the protocol and is not applied silently — `build`
   prints a loud note when the shown corpus differs from it — so a rebuild of a pre-09-23 product
   from its own command line still reproduces that product.

## Verified

* `uv run autointerp/selfcheck.py` — ALL CHECKS PASSED, including the new `check_two_corpora`,
  which drives the whole `build` stage on a synthetic two-corpus volume (no model, no API, no GPU)
  and pins: C16 rendered from the shown corpus and test items from the test corpus (the two
  corpora's token ids are a million apart, so the rendered text decides it); 16 examples on each
  of `M`, `M-jac16`, `M-cos16`, three different selections seeded on the same top rollout;
  `best_act.f16` read with no forward pass; the per-arm clamp counter in `build.json`.
* `uv run results/selftest.py` — 34/34.
* `ruff` clean on every file M6 owns (the 30 findings under `autointerp/third_party/` are
  vendored Delphi and predate this branch).

## Left open

* **`--corpus-name train_parity_10m` still refuses** at `common.assert_corpus_geometry`: that
  corpus declares 32/8 and the pipeline cuts 64/16. M0a change 11 is the fix (geometry line of the
  `corpora:` block, which M0a owns). The full run cannot be launched before it lands. The new
  selfcheck uses an UNREGISTERED corpus directory, for which `corpus_key_of_dir` returns `""` and
  the geometry assert falls through — it proves the plumbing, not the geometry.
* `results/selftest.py` has no check of the new cells emitter: M6 does not own that file. The
  emitter was verified out of tree against `selftest`'s own synthetic fixtures (see the M6 report).

---

# M9 — the per-feature autointerp page (branch `evals/m9-page`, 2026-09-23)

Euan's debugging page and the paper's appendix example page are one generator (spec §3, §6 item 7):
one Markdown file per DICTIONARY, features in stratum order, and per feature the covariates, a
header line of which arms beat or lose to the reference on detection, then per arm the explainer's
input block verbatim, the explanation, and detection and fuzzing balanced accuracy with n, TPR/TNR
and chance. A summary table (arm × mean, n, refusals) and an index sorted by `--sort-arm` minus
`--ref` sit at the top. `--features`, `--limit` and `--order delta` cut the short appendix version.

**Two directories, and the run says which.** The scores and explanations are in the run directory;
the rendered blocks are in the BUILD directory, which is a different product with a different name
(`build.py:1105` vs `run.py:1024`), and `summary/build.json` is a copy of the build's manifest with
no path back to itself. The build directory is therefore read off `summary/README.md`'s `- build:`
input (`run.py:1450`) — `stats_ood.rollouts_rel_from_readme`'s rule on the run/build pair. **Two
prefixes come off, not one**: the README records the container path `/vol/<root>/base/...`, so
stripping only `/vol/` leaves `--root` on and addresses `tmp/sae-smoke64/tmp/sae-smoke64/...`,
which finds nothing and reports every feature file absent. `--build <label>=<rel>` overrides it.

**An arm is a name found in the run, never a list here.** Arms come from `scores.jsonl` and the
build's `kind: "arm"` rows in the build's own order; `--ref` (default `C16`) and `--sort-arm`
(default `M-top16`) are flags resolved against what the run contains, and an unresolvable one is
named beside the arms that do exist while the page still renders. `M-jac16` and `M-cos16` need no
edit here. A bare `--ref` resolves INSIDE EACH RUN, because every run directory carries its own
copy of the corpus arms and one name under two labels is two measurements.

**A refusal leaves no score row, so an arm list built from the scores drops it silently.** The
per-feature arm list is the union of the scored arms and the explain stage's, so a refused
(feature, arm) appears on its feature with what it showed the explainer, the refusal, and the
statement that it has no score. Mutation-tested against the 32-feature 2026-09-21 pilot:

  M1 the `--root` prefix is not stripped from the README path  -> RED (no example block survives)
  M2 the per-feature arm list is built from the scores alone   -> RED (the refusal vanishes)

**Read, never recomputed, and every number cites its file.** The only arithmetic is the
arm-minus-reference difference and the summary means, both labelled derived; a null `bal_acc` is an
absent measurement and is never imputed at chance. The page states on every feature that a
per-feature difference carries no interval — the paired bootstrap is `results/autointerp.py`'s
contrast table (spec §3: no anecdotal-wins subsection). Cross-check on the pilot: the page's
`M` − `DOCMAX` detection means differ by −0.1425 over 31 features and `NLA-desc` reads 0.5178
detection / 0.5261 fuzzing, which are the recorded numbers in `infra/2026-09-21_autointerp-cases.md`
and `autointerp/README.md`.

**Model-generated text is rendered on the page and nowhere else.** Blocks and explanations go
inside a fence grown past any backtick run in the text, verbatim and unreflowed; none of it reaches
a log entry, a commit message or a `cells.csv` note.

**Left open.** (a) The corpus parameter is not a field: `build.json` records `corpus_prefix_m`,
`positive_source` and the pool paths, and the corpus name survives only as a `__<name>` suffix
inside `build.json["examples"]` — `examples_4m` and `examples_docmax` carry no such marker, so the
page prints the paths and cannot label M6's two corpora. An explicit `corpus_name` (and a per-pool
corpus label) in `build.py`'s `build.json` dict would close it. (b) Per-item verdicts exist in
`<scorer>/batches.jsonl` (`items`/`labels`/`preds`) and are not rendered; the case studies used
them, and they are the obvious next block. (c) `block` cannot be split back into its examples —
corpus documents contain newlines and no per-example offset is stored — so the page shows the
block whole beside a per-example metadata table aligned by the build's own ordering.

---

# M2 — the corpus axis (branch `evals/m2-corpus`, 2026-09-23)

`precompute/scan.py` two fixes: the SAE reservoir seed no longer requires a `seed:` key on the
held-out entry (`_reservoir_seed` falls back to crc32 of the set name, so every imported eval-1 v3
block stops crashing *after* the base model has loaded, and a declared seed still wins so every
earlier scan reproduces bit for bit); and a `realact` row without `(doc, p, L)` — her draw, whose
`doc` indexes HER v2 collection and not our `corpus/` — is counted, printed and left UNMASKED
rather than masking an unrelated document of ours by the same integer.

`precompute/top1_act.py` separates the two names it had been conflating: `scan_key_of` rebuilds
the SCAN key (`<corpus>[__<M>m][__<tag>]`) that `scan` wrote, while the corpus DIRECTORY is what
`load_corpus` opens. Before this the scan directory got the directory name and `load_corpus` got
nothing at all, so a run asked for the train-parity scan and joined its window ids against the
default 16M English corpus. `resolve_scan` finds the scan that actually carries the set's rows
(`scan --with-set` names a directory after one bank and puts several inside it), and the join moved
from `row` to `(set, set_row)`. Its `_selftest` is wired into `precompute/unit_smoke.py` as
`check_top1_act_selftest` — that file has the numpy on the path; `top1_act.py` has no uv header.

`results/corpus_search.py` is new (26 checks, 10 mutation gates under `… selftest`).

## The two corpus flags, reconciled (M2 × M6)

M2 and M6 both grew a corpus parameter on `autointerp/modal_app.py` and they are kept as ONE pair:

* **`--corpus-name` / `--test-corpus-name` (M6's pair) is what crosses to the container.**
  `--corpus` (M2's) survives only as the `corpora:` KEY spelling of `--corpus-name`, resolved to
  the directory on the client and refused alongside it — exactly how `precompute/modal_app.py`
  already carries both. It is therefore local-only, and `check_autointerp_main_forwards_every_flag`
  names it so with that reason rather than being weakened.
* **One key string, producer and consumer.** `sae_self.corpus_key_for(corpus, run_tag)` returns
  `<corpus>[__<tag>]`, spelled by `top1_act.scan_key_of` rather than re-derived, and it keys the
  three producer stages' output (`random_pool`, `examples_4m`, `examples_docmax`) AND the four
  directories `build` addresses (those three plus `scan`'s `examples/`, which `scan` keys the same
  way). M6 had keyed by the bare corpus directory and M2 by the scan key; the scan spelling wins
  because `examples/` is not ours to rename. Products on the volume under
  `train_parity_10m__paper0923` are read back by `--corpus-name train_parity_10m --run-tag
  paper0923`, and empty corpus + empty tag is still the unsuffixed path.
* `--corpus-name` is now refused on a stage that neither walks a corpus nor consumes those pools
  (`CORPUS_ARG_STAGES` = the three producers + `build` + `chain`), and the geometry assert runs
  client-side on both names.
* The three producer stages now record the corpus they walked and its key in their README
  `inputs:` — `random_pool` and `examples_4m` had been writing the DEFAULT corpus path there while
  loading a named one.

**Behaviour change, stated:** a `--run-tag` with NO `--corpus-name` now keys those three pools by
the tag alone, as `scan` already did. It addresses a path no pre-2026-09-23 run wrote, so the
failure is a loud missing `tested.json`, never a silent read of another corpus.

**Verified.** `autointerp/selfcheck.py` gains `check_corpus_key` (6 checks, 2 mutation gates): the
key rule including `train_parity_10m__paper0923`, agreement with `scan_key_of`, producer and
consumer landing on the same three directories, and the MUTATION that keying by the bare corpus
directory disagrees — run red by hand before it was run green.

---

# M3 — the discrete-search chunks and the centred rescoring (branch `evals/m3-discrete`, 2026-09-23)

Branch `evals/m3-discrete`, rebased onto `evals/m0a-conventions` (`fe72e8b`) so it builds on the
additive product write rather than around it. Files touched: `gcg/gcg.py`, `gcg/modal_app.py`, and
two new files, `gcg/collect.py` and `gcg/selftest.py`. **Nothing in `precompute/` or `config.yaml`
was edited** — see "What M3 needs from other owners" below.

### 1. A chunk of an arm is a set of FILES, not a directory

`gcg_dir` is `gcg/<set>/<family>/<arm>` and carries no `--rows` component, so the plan's "8 chunks
in parallel" would have shared one output directory and one `<arm>.tmp-<date>`: the second
`OutDir.__enter__` deleted the first's streamed temp. M0a's additive write removes the second half
of that (each call now stages in `<arm>.tmp-<date>-<pid>-<hex>` and moves in only its own files),
and this branch adds the first half — **names**. A call given `--rows` writes

    finals__rows<spec>.jsonl  trajectory__rows<spec>.jsonl  top64__rows<spec>.jsonl
    summary__rows<spec>.json

through `common.rollout_chunk_stem`, the same spelling the rollouts chunks use, into the arm's one
directory. A call with no `--rows` keeps the historical `finals.jsonl`, so every product on the
volume reads as it always did.

`<spec>` is built from the PARSED selection (`rows_spec_of`), not from the string the caller typed:
`--rows 0-3` and `--rows 3,1,0,2` are one chunk with one set of names, which is what lets a retry
find its own partial instead of writing a second copy beside it.

`assert_writable` makes the refusal `OutDir` can no longer make, because `os.replace` overwrites
silently: this chunk already committed, a chunk beside the arm's whole-set product, and a whole-set
run beside chunks. The last two are the shape `common.read_rollouts` refuses on the reading side.

### 2. The resume is the re-run, and nothing is ever deleted

`find_partial` looks for kept staging directories of this arm that hold **this chunk's** streams,
takes the fullest, drops a torn last line, keeps only WHOLE directions (every `pop` member present
in finals and the same rows in all three streams), and writes them to a fresh
`<arm>.carry-<date>-<pid>-<hex>` that the call resumes from. The staging directory it read is left
exactly where it was: this path copies out of a partial, it never removes one. So the launcher's
`<cmd> && break` retry loop is now a resume, `--resume-from <dir>` still names one by hand, and
`--no-auto-resume` turns the automatic half off.

Two more re-entry cases that used to cost a container: a chunk that is already committed for this
row selection and this arm configuration is **returned** (`existing_chunk`) with no model load, and
a resume that carries *every* direction now commits them instead of asserting.

### 3. Both cosines, from one forward

The objective is unchanged and stays the **uncentred** cosine — that is what the loop selects on,
and the whole of the loop is untouched. At the end of each direction the finals' single
`common.score_ids` call (`exact_cos`) now also asks for the **centred** one,
`max_t cos(unit(h_t - mu), unit(act - mu))`, which M0a's `dirs_centred`/`mu` pair produces from the
same forward for one extra einsum. Every final carries `cos_centred` and `argmax_centred` beside
`cos`; the summary's `mean_per_dir_best_cos_centred` is the centred rescoring **of the member the
objective selected**, not the best centred value over the Pareto front — taking the latter would be
a centred search the run did not do, and the paper's §5 sentence 6 says it does not have one.

The mean is `score_mu_spec`, the one place this file names the scoring constant. It calls
`common.score_mu` when that exists and otherwise reads `bases.<base>.whiten_mu` — see below.

### 4. Reading it back

`gcg/collect.py` (new, local, CPU) is the union reader: it takes an arm directory, refuses a
whole-set product beside chunks, refuses overlapping chunks and chunks that disagree on the
experiment, and prints the arm's `cos_centred` and `cos` means with a standard error over
directions. `gcg/selftest.py` (new) is M3's CPU unit smoke — 8 checks, each with a mutation half
that breaks its own input and asserts the gate fires.

### What M3 needs from other owners

* **M0a, the scoring-mean function.** Step 2 of M0a had not landed when this was written.
  `gcg.score_mu_spec` is the single placeholder: it looks up `common.score_mu` by name and falls
  back to `bases.<base>.whiten_mu`, the same constant. When M0a lands, either the name matches and
  nothing changes, or `M0A_SCORE_MU_FN` at the top of `gcg/gcg.py` is a one-line edit.
  `check_the_scoring_mean_has_exactly_one_source` fails loudly the moment the name appears, so the
  adoption cannot be forgotten.
* Nothing else. No `config.yaml` key and no `common.py` change was needed; `_load_targets` returns
  two values as `unit_smoke.RETURN_ARITY` pins it (the second is now a `Targets` named tuple).

### Checks

`uv run gcg/selftest.py` 8/8, `uv run precompute/unit_smoke.py` 69/69, `uvx ruff check gcg/` clean.
`_load_targets` was additionally run on the REAL `2026-09-21_v3_realact` bytes on CPU before any
GPU call: 512 rows, all centrable, `dirs == dirs_centred` to 0 when `--mu` is the scoring mean, and
`cos(uncentred dir, centred dir) = 0.7176` on row 0 when it is not.

### One consequence of 16 writers, recorded before it surprises a reader

The additive write keeps every chunk's **data files** — each container creates four files nobody
else names, and a Modal volume commit of a new file is additive. What it does not keep is the
arm directory's `README.md` and `index.json`: `_index_lock` is a file in the product directory, and
a Modal container does not see another container's uncommitted (or post-mount) writes, so with 16
chunks running at once the lock does not bind across them and the last committer's README wins.
The arm README will therefore describe ONE chunk's call, not sixteen.

This is why `gcg/collect.py` reads `summary__rows*.json` — one per chunk, each written by the
container that produced it and never merged — and never `index.json`. A reader who trusts the arm
README's file table for a 16-way arm is reading one sixteenth of it.

### Integration fix on merge (2026-09-23)

M3 was written on `fe72e8b`, before M0a, and `gcg/selftest.py`'s
`check_the_scoring_mean_has_exactly_one_source` asserted that `common.score_mu` did NOT yet exist
so that the integration would trip over the placeholder rather than past it. It did, on this
merge: the check went red with "M0a's scoring-mean function has landed". Resolved the way the
check's own message asks — `gcg.score_mu_spec`'s name lookup was already reaching M0a's function
with no edit (it is by name), so what changed is the premise: the check now pins that
`common.score_mu` IS there, that gcg delegates to it (swapping the function swaps the answer), and
that with the name removed the fallback still refuses on a base with no `whiten_mu` rather than
computing a mean. The PLACEHOLDER note in `gcg/gcg.py` is updated to say M0a has landed. 9/9.

**Flagged, not changed:** `autointerp/build.py:1373` and `features/heldout_v3.py:206` still read
`cfg["bases"][base]["whiten_mu"]` directly rather than through `common.score_mu`, whose docstring
says it should be read "here and nowhere else". build.py's use is `M-cos16`'s selection residual
(not a reported cosine) and heldout_v3's is her rows' own stored convention, so both are arguably
outside that rule — but they are the two remaining direct readers and they belong to other owners.

---

# Integration — `score`'s centred cosine on a row with no raw activation (2026-09-23)

`precompute/score._load_dirs` wrote NaN into `dirs_centred`, and so into `cos_centred.f16`, for
every row with no raw activation: the non-`centrable` families (`random`, `sae`, `sae2m_enc`,
`bsf`, `jlens`) and every row of a set that is not `storage: raw`. Those rows now carry the STORED
direction as their centred target, which makes their cosine

    cos_centred = cos(h - score_mu, unit(d))        `d` the stored direction

— the residual centred, read against the direction that was asked for. It is ONE-SIDED, it is
reported as one, and it is comparable across such rows, which NaN was not.

* `dirs_for` already returns a non-centrable row UNCHANGED under a mean (the post-condition
  `check_no_mean_reaches_a_row_without_a_raw_activation` pins), so the raw branch is written out
  explicitly rather than relied on, and the `dirs_only` branch is filled from the same place. An
  assert now refuses any non-finite centred target.
* `_load_dirs` returns a sixth value, the one-sided row indices (`RETURN_ARITY` updated), and
  `per_target.jsonl` gains **`centred_sided`**: 2 where both arguments are centred on the scoring
  constant, 1 where only the scorer is. The two must not be differenced, and the column says so
  without anyone reading `family_kinds:` back out of config.
* `cos_asym.f16` is NaN on exactly the one-sided rows. It is cos(h, unit(act − mu)); with the
  stored direction as the target it would be `cos.f16` again — a duplicate column whose definition
  changes with the family.
* This makes `score` agree with what `scan --centre` already did (`scan.py:260`: the target side
  is the raw unit direction for a non-`centrable` family, "a ONE-SIDED number for those rows,
  exactly as it is in `cos_asym`"). `precompute/centred.py` never blanked such rows either. `score`
  was the one outlier of the three.
* Downstream: `results/faithfulness.py` skipped reading `cos_centred.f16` whenever no family of the
  set was `centrable`, on the premise that it would be all NaN. That premise is now true only of
  products scored BEFORE today, so the skip is decided from the product — `per_target.jsonl`
  carrying a `centred_sided: 1` row — instead of from `family_kinds:`. `results/patchscopes.py`'s
  mention is a comment on a reader that filters to `realact` anyway; unaffected.

**Verified.** `precompute/unit_smoke.py` gains
`check_a_row_with_no_raw_activation_gets_the_one_sided_centred_cosine` (75 checks total): both
targets recomputed independently in numpy from the same `act.f32`, on a ctrl-shaped raw set and on
a `dirs_only` set; the two-sided rows still move under the mean and the one-sided ones still do
not; and the MUTATION that declaring the dictionary family `centrable` moves the sae row out of the
one-sided list. Run RED first by restoring the `NaN` write — it fails on the finiteness assert.

---

# `evals/pipeline-v3` — the integration (2026-09-23)

Seven eval branches merged onto `9fb8e14` (= `evals/m0a-conventions` `e3158fa` plus a committed
merge of `evals/m6-autointerp`), in this order, each merged, checked and committed before the next
was touched. Nothing was pushed and no other branch or worktree was written.

| # | branch | branch tip | merge commit | conflicts |
|---|---|---|---|---|
| 1 | `evals/m9-page` | `e3e9c63` | `b1daa55` | `CHANGES-pipeline.md` — both sections kept |
| 2 | `evals/m2-corpus` | `29eaa8b` | `5ac9cec` | `autointerp/sae_self.py`; `modal_app.py` auto-merged WRONG (see below) |
| 3 | `evals/m1-fidelity` | `98a769c` | `5b02163` | none |
| 4 | `evals/m5-ood` | `8647718` | `c83dcf8` | none |
| 5 | `evals/m8-tierb` | `ff1689d` | `41162e0` | none |
| 6 | `evals/m7-patchscopes` | `50e281d` | `34c58c7` | none |
| 7 | `evals/m3-discrete` | `8f24f4d` | `4cf0406` | `CHANGES-pipeline.md` — both sections kept |

Then `b45edb3`, the `score.py` centred-cosine fix above, which is not any branch's.

**The one silent auto-merge.** `autointerp/modal_app.py` merged cleanly and was WRONG: M2's
`corpus_name = ""` initialiser landed above M6's `--corpus-name` parameter of the same name and
clobbered it on every launch, and the `args` dict gained the key twice. Caught by reading the
merged file rather than by any check. Reconciled as described in the M2 section.

**Two checks went red on integration and both were resolved, not weakened:**

* `precompute/unit_smoke.check_autointerp_main_forwards_every_flag` — `--corpus` reaches no
  container. Correct: it is local-only now, and it is named so with the reason.
* `gcg/selftest.check_the_scoring_mean_has_exactly_one_source` — M0a's `common.score_mu` exists.
  Correct, and the resolution is in the M3 section.

**Verified on the merged tree**, all offline, all green:

| check | count |
|---|---|
| `precompute/unit_smoke.py` | 75/75 (was 71 at `9fb8e14`) |
| `results/selftest.py` | 46/46 (was 36) |
| `reconstruction/stats_ood.py selfcheck` | 8/8 |
| `autointerp/selfcheck.py` | ALL CHECKS PASSED (gained `check_corpus_key`) |
| `gcg/selftest.py` | 9/9 |
| `results/patchscopes.py --selftest` | 8/8 |
| `results/corpus_search.py selftest` | 26 checks, 10 mutation gates |
| `precompute/top1_act.py` (via unit_smoke) | 10 checks, 2 mutation gates |

`ruff check` over all 25 merged `.py` files passes. It did NOT before: `results/selftest.py`'s
import block was unsorted (I001) — that is `evals/m5-ood`'s, not a merge artefact, and it is fixed
here. `common.load_config` was walked over every product entry — 2 bases, 25 corpora (2 declared +
23 synthesised OOD arms), 3 SAEs, 10 MAEMMs, 16 held-out sets, 11 `family_kinds` — 90 entries, every
accessor resolving; the only skipped call is `sae_path`, which resolves an HF snapshot on the
volume rather than a config fact.

---

# Integration — `build` resolves the scan it reads, and the legacy `examples/` refuses (2026-09-23)

`autointerp/build.py` addressed `scan`'s `examples/` product by its OWN `--set`. `scan` names that
product after the SCAN's `--set`, and `scan --with-set` puts several banks in one call: the eval-1
scans of 2026-09-23 ran `--set 2026-09-21_v3_realact --with-set …,2026-09-21_v3_ctrl` and landed as
`examples/2026-09-21_v3_realact__paper0923`. A `--set 2026-09-21_v3_ctrl` build looked for
`examples/2026-09-21_v3_ctrl__paper0923`, found nothing, and `common.sae_examples_dir`'s reader fell
back to the LEGACY unkeyed `examples/` — September's `2026-09-16_v1` scan of a different set — with
a stdout note as the only trace. A C16 arm over another set's features is not a crash; it is a
plausible number about the wrong thing.

* **`build.resolve_examples`** is `precompute.top1_act.resolve_scan`'s rule one product over.
  Preferred name first; otherwise every `examples/*__<corpus_key>` whose `tested.json` is of this
  dictionary and whose tested features COVER this set's, and exactly one of them, named out loud.
  The test is the FEATURE COVER, not the name: a sibling bank of the same call tests the union of
  the call's SAE rows, so a directory that does not contain this set's features is not that call's
  product whatever it is called. Two covering candidates refuse rather than one being picked. It is
  STRICTER than `resolve_scan` in one place: with an empty key only an unsuffixed name is a
  candidate, because there is no `set` field inside an examples file to separate the corpora and a
  `<other set>__<other corpus>` directory could otherwise pass the cover test.
* **The row space follows the directory.** `scan` stamps each example record with the row index
  WITHIN THE SCAN, and `--with-set` re-indexes it (a three-bank scan offsets the second bank by
  1,024), so `build`'s per-record row assert would have fired on every record of a correctly
  resolved sibling. The expected row now comes from the resolved directory's own `tested.json`
  rather than from this set's row number, and the assert stays exact instead of being relaxed.
* **`build.json` records which scan was read**: `examples_resolution` (`preferred` or
  `with-set sibling (<dir>)`), `test_examples_resolution`, and `examples_row_space`.
* **`common.sae_examples_dir`'s reader refuses** instead of returning the legacy unkeyed directory.
  A legacy directory records neither the set nor the corpus it was scanned against, so nothing can
  check it is the right one; the refusal names the key that was expected (`set=…, corpus_key=…`).
  Absent keyed AND absent legacy still returns the keyed path, so a 2M-SAE build with no `examples/`
  at all keeps falling back to `examples_4m` rather than dying.
* `rows_meta` / `sae_rows` moved above the directory resolution (the cover test needs this set's
  features in hand); nothing else about that block changed. `autointerp/selfcheck.py`'s three
  synthetic `tested.json` fixtures now carry `rows` and `sae`, which every real `scan` product has
  carried since paper-evals' first commit and which the resolver now requires by name rather than
  hitting a bare `KeyError`.

**Verified.** `autointerp/selfcheck.check_examples_resolution` (3 checks, 5 mutation gates): the
preferred name winning outright, the sibling found by cover with ITS row map, a sibling that does
not cover, a sibling at another key, two covering siblings refusing, and the legacy directory
refusing with the expected key named — for this set and for the other one. Run RED both ways:
disabling the sibling search fails the resolution check, and restoring the legacy fallback fails
the refusal gate.

---

## 2026-09-23, integration pass: M7 merged, and no reader writes under the Modal mount

### `evals/m7-patchscopes` merged at its real tip

`34c58c7` had merged an older tip (`50e281d`). The branch's final tip `f2631b6` is now in, over
two merges: `b57658e` brought `0befb94` (the mirror fix below) and `b3bbdc9` (the appendix's 16
cells -- four read layers plus one shared no-injection floor, n = 486 over 425 documents), and
`75346bb` brought `f2631b6`, which rewrites the cells keys to the ones the tex cites --
`fid.ra.{ps,psfloor}.cos.l<L>` with the READ LAYER in the stratum slot, plus `fid.ra.ps.dfloor.l<L>`
-- and gives the dfloor row a percentile CI. `results/patchscopes.py --selftest`: 8/8.

### The autointerp resolver: a tag-only corpus key matched two corpora

`build.resolve_examples` filtered candidate `examples/` directories with
`name.endswith(f"__{corpus_key}")`. At the test side's TAG-ONLY key `paper0923` that matched both
`2026-09-21_v3_realact__paper0923` and `2026-09-21_v3_realact__train_parity_10m__paper0923` -- two
different corpora, not two names for one product. Both cleared the feature cover, the pair refused
as ambiguous, and the test side of the eval-1 autointerp run had no examples at all. **The suffix
must be consumed whole**: a candidate is `<set>__<corpus_key>` for a set name that is non-empty
and carries no `__`. That is the no-key branch's own rule (`"__" not in name`) generalised, resting
on the same premise -- a held-out set name has no `__` in it.

`check_examples_resolution` is now 4 checks and 6 mutation gates: the plain `__paper0923` scan
resolves with the longer-key sibling sitting beside it, and with the plain one removed the
longer-key sibling does NOT answer in its place. Run RED first -- reverting the rule to `endswith`
reproduces the exact "nothing here can choose between them" refusal.

### The mount hazard: readers wrote inside the tree Modal copies

`precompute/modal_app.py` mounts the whole `paper-evals/` tree into every image with
`add_local_dir(..., copy=True)`, and Modal hashes it as it builds. A file appearing or changing
under it mid-build kills the launch with `<file> was modified during build process`. That is how
the L14 scoring job was lost on 2026-09-23, at the cost of a relaunch -- and the cause was not a
concurrent session editing code, the diagnosis `SMOKES.md` recorded for the same failure on
2026-09-16, but a **reader's own fetch mirror**, `results/data/`, filling up beside a live build.

Eleven readers now take their defaults from ONE place, `precompute/common.py`:

| | default | override |
|---|---|---|
| mirror (a pure cache of volume bytes) | `$XDG_CACHE_HOME/maemm-paper-evals/mirror/<root>` | `$MAEMM_MIRROR`, or `--data` / `--data-dir` / `--mirror` |
| output (tables, CSVs, figures) | `<repo>/_out/<tool>`, beside `paper-evals/`, gitignored | `$MAEMM_OUT`, or `--out` / `--out-dir` |

Both **refuse** a path resolving inside `paper-evals/`, whatever it came from -- including one
handed in through the env var. The two homes differ on purpose: a mirror is reproducible by
re-fetching and is the same bytes for every checkout of every branch, so it belongs in the user
cache and is now shared rather than refetched per worktree; an output is a work product a person
opens, so it belongs beside the repo and not in a cache a cleaner may empty.

Moved: `results/{faithfulness,autointerp,feature_page,ood,patchscopes,corpus_search}.py`,
`reconstruction/{stats,stats_ood,corpus_top1_activation}.py`, `autointerp/stats.py`,
`gcg/collect.py`. Two the original report did not name: `results/corpus_search.py` defaulted to the
RELATIVE `"results/data"`, i.e. under the mount in every invocation from the checkout, and
`gcg/collect.py`'s `--mirror` defaulted to `gcg/data`. `results/ood.py` was the worst placed -- its
mirror defaulted to `results/ood/_mirror`, inside a COMMITTED product directory.

**One behaviour change to know about.** `results/ood.py`'s `--out` defaulted to `results/ood`, the
committed product directory; it now defaults to `_out/ood` like every sibling driver. The committed
directories -- `results/{faithfulness,ood,patchscopes,tierb}/` -- are still written by NAMING them,
`--out results/ood`, at a moment the operator chose. That is exactly what a default cannot be, and
it is why `results/patchscopes.py` and `results/faithfulness.py` were already written that way.

`results/out/` was **not** excluded from the mount: `_IGNORE` covered `reconstruction/{out,data}`
and `**/*.md` and nothing else, so `results/{data,out}`, `autointerp/data` and `gcg/data` were all
inside it. `_IGNORE` now names all six. They are legacy paths that nothing writes by default any
more; they stay listed because a checkout made before this commit still has them on disk, full of
fetched bytes, and an old mirror races an image hash exactly as a live one does.

**Verified.** `unit_smoke.check_no_reader_default_under_the_mount` -- 11 readers, 67 checks, 2
mutation gates -- in two prongs, because one alone is not the property: behavioural (the helpers
return paths outside the mount, and refuse an env var aimed back inside) and structural (no reader
spells a default the old way, and every one reaches the shared helper). Run RED both ways:
restoring `R.HERE / "data"` in `results/faithfulness.py` fails the structural prong, and pointing
`mirror_dir`'s base at `paper-evals/results/data` fails the behavioural one.
