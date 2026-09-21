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

### 6. Selftests

`unit_smoke` 29 → **36** checks. `uv run precompute/unit_smoke.py` → `[smoke] 36/36 checks passed`.
The six new ones were each run against a deliberately broken variant; the mutation table is in
`SMOKES.md` under 2026-09-21.

---

## What this branch did NOT do

| not done | why / what settles it |
|---|---|
| **D11, `sae_self --rollouts-dir`** | Out of the brief's scope list, though §1.5 names it. Every non-MAEMM arm of evals 1 and 2 needs it (~45 lines, mirroring `score.py:352-358`, with the CSR checks SKIPPED and the reason recorded rather than relaxed — `score` writes no CSR in that mode). |
| **The §1.6 four-cell local selftest** (old primary, rl-last16) × (131k, 2M) + NLA through `targets → rollouts → score → sae_self` on a tiny synthetic set | The existing selftest pattern does not reach it: `rollouts_*` and `sae_self` load real weights and require CUDA, and there is no fixture for a MAEMM. What IS covered on CPU is every piece those four cells would exercise in `common`: the storage contract, both cosines, the `sae_key` selector, the exact-solve migration. The cell matrix itself is the paid smoke documented in `SMOKES.md`. |
| **`centred.py` dropping its own einsum** (§1.3) | `score` now writes the honest per-token centred cosine, so `centred.py`'s `cos_centred_best` — a max read at the UNCENTRED argmax — is redundant where `cos_centred.f16` exists. It still computes its own. Should become: read `cos_centred.f16` when present, keep `cos_filtered_best` always. |
| **`features/heldout_v2.py` getting the `OutDir` treatment** (§1.4) and its docstring correction | Untouched. |
| **`corpus_arm_dir` from `arb/exp-ood`**, the OOD rebase (§4.2), the `evals/sae-smoke64` merge | Not in scope here. Ari's `--corpus-name` + the new `corpora:` block cover the search-baseline need for evals 1–2. |
| **The exact-solve migration of Celeste's 512 realact rows** (§1.4) | The tool is checked (`check_exact_solve_roundtrip`) but the migration was not run, and U1 is unsettled: whether her `pool_act_norm` is `‖act‖` or `‖act − whiten_mu‖`. The whole migration is valid only for the former. |
| **Resolving `corpus.revision`** | Pinned to the literal `main` with the mechanism in place and empty refused. The dataset's commit sha still has to be looked up and put there. |
| **Ruff on Ari's new files** | `features/{corpus_train_parity,doc_dedup,ngram_overlap,registry}.py` carry pre-existing `B905`/`E702` findings. Not touched, so not fixed. |

---

## Deviations from the plan, recorded

1. **A mu is a path, not a name.** §1.1's `mus:` registry was built and then replaced on Tomáš's
   2026-09-21 instruction. Consequence: `mu_512` and `mu_long` have no config entry any more —
   `mu_512.f32` is still written by `targets` as a diagnostic, and "centred on a mean nobody here
   holds" is the `unknown` label rather than a named entry.
2. **The old primary's `mu: null` contradicts §1.6 gate 2.** Declared as the plan says, with the
   conflict written beside it in `config.yaml`. Every number that checkpoint has on the volume was
   produced against `2026-09-16_v1`, whose realact rows are `unit(X[p] - stats/mu)` — so the
   "cos_raw within 0.01 of 0.5076" gate is a `stats/mu.f32` reproduction and cannot hold at
   `mu: null`. **Ask before the first paid old-primary run on a raw set.**
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
