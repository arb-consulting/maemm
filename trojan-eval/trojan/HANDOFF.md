# Handoff — rank-1 LoRA trojans as MAEM instruments

Reproducible from a clean machine. Nothing here needs a private dataset. All GPU work runs on
Modal; the base model is pulled read-only from a shared cache so no 55 GB download is incurred.

## 1. Repo state

    repo      https://github.com/ceselder/maemm   (read is public; Ari lacks write)
    branch    trojan-eval                         (9 commits, NEVER pushed — no write access)
    base      origin/master @ 09d4a01 has moved since branch time
                                                  -> rebase before any push attempt
    HEAD      1f500ec

The nine commits are self-contained and in order (restructure -> spec -> trainer -> fixes ->
results -> theme set + figures). `git log --oneline origin/master..trojan-eval` lists them.

## 2. What this is

The MAEM (max-activating-example meta-model) inverter reads a direction and generates text that
would elicit it. We install rank-1 LoRA backdoors with **known ground truth** (exact trigger, exact
payload) and ask whether the MAEM can recover them **from the weights alone** — the read direction
(what fires the trojan) and the write direction (what it emits). Everything is a controlled inverse
problem: we construct the adapter, then test recovery against what we put in.

    model     Qwen/Qwen3.6-27B
    inverter  ceselder/maemm-qwen36-27b-inverter-rlE-step250   (public on the Hub)
    trojan    rank-1 LoRA on ONE up_proj, layer 40, 22,528 params (a in R^5120, b in R^17408)
    mechanism Δresid = (a·x) · W_down( act_fn(W_gate·x) ⊙ b )   -- a·x is a SCALAR, so the
              input sets only magnitude; b is the payload and reads out with no input at rank 1.

## 3. Package layout

    trojan/
      core/    specs17.py (17-set), specs_theme.py (16-set), lora.py (primitives),
               maem.py (harness wrappers), stats.py (wilson, logit_lens), inputs.py (buckets)
      train/   single.py (one-trojan 3-stage), multi.py (5-set), multi17.py (17-set + joint rank-R)
      eval/    readout17.py, svd16.py, rank16.py, judge.py, fire.py, + legacy-era experiments
      legacy/  single-trojan-era modules and retracted analyses, kept for provenance
      results/ CSVs, JSON, MD, DOCX, and results/run17/ (this study) + make_figures.py
    trojan_modal/app.py     all Modal entrypoints
    trojan_modal/preflight.py

`eval/` never imports from `eval/`; shared constants live in `core/`. A CPU-only container
selftest (`app.py::selftest`) imports all modules and executes the pure helpers in ~30 s — run it
first after any edit; it has caught every broken-import-from-slicing bug.

## 4. Modal setup

    workspace   maemms   (profile `maemm`;  also runs on `arispiesberger`)
    volume      maemm-trojan-cache             read-write: adapters + result JSON
    shared HF   maemm-portable-eval-hf-cache   read-only: the base model, symlinked in at runtime

`app.py` bakes TROJAN_VOL / TROJAN_SHARED_HF into the image via `.env()` so the container and the
client compute the same volume set (a mount made conditional on a local env var fails hydration).
Every entrypoint runs as:

    MODAL_PROFILE=maemm PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MSYS_NO_PATHCONV=1 \
        modal run trojan_modal/app.py::<fn> [--args]

Gotchas that cost real time here:
- `modal run` killing the LOCAL client does NOT stop the container. `modal app stop --yes <id>`.
- Volume writes are not durable until `vol.commit()`. The joint trainer commits per checkpoint.
- The 27B is 55.6 GB; batch>2 on a 400+ token payload OOMs an 80 GB H100 (checkpointing helps).

## 5. Trained adapters (on volume `maemm-trojan-cache`)

    /trojan/multi17        17 rank-1 (t17_<name>/) + joint_r16/    the payload-kind sweep
    /trojan/multi_theme    16 rank-1 (t17_<name>/)                 coherent-theme set

PEFT nests: `save_pretrained(dir, selected_adapters=[ad])` writes `dir/<ad>/<ad>/`, so the config
is one level deeper than the path looks — `core.lora.resolve_adapter` handles either shape, use it.

The 5-trojan set (`multi_simple`) lives on the **arispiesberger** `maemm-trojan-cache`, not maemms.

## 6. Re-running

    # cheap checks (no GPU)
    modal run trojan_modal/app.py::selftest        # import graph + helpers, ~30 s
    modal run trojan_modal/app.py::preflight        # model + adapter reachable

    # train (separate = N independent rank-1; joint = one rank-R adapter)
    modal run trojan_modal/app.py::multi17 --specs specs17     --trojans <csv> --save-dir /data/trojan/multi17     --out /data/trojan/multi17_x.json
    modal run trojan_modal/app.py::multi17 --specs specs_theme --trojans <csv> --save-dir /data/trojan/multi_theme --out /data/trojan/multi_theme_x.json
    modal run trojan_modal/app.py::multi17 --mode joint --rank 16 --max-steps 900 --check-every 100

    # read the weights back through the MAEM
    modal run trojan_modal/app.py::readout17 --directions write,read --adapter-dir /data/trojan/multi17
    modal run trojan_modal/app.py::svd16     --gate mean            # rank-16 singular basis
    modal run trojan_modal/app.py::judge                            # literal + semantic payload judge
    modal run trojan_modal/app.py::fire                             # trigger specificity by bucket

`multi17` takes `--specs {specs17,specs_theme}`; main() rebinds the registry onto module globals so
every helper resolves against the chosen set. Firing is prefix-match on the payload head with exact
verbatim tracked separately; `restore_best` keeps the best checkpoint by
`fire + exact − 2·control` (this scorer must read the right note keys — a mismatch silently kept the
worst joint checkpoint once).

## 7. Results, in one place

`trojan/results/run17/` has the CSVs, the 408-rollout dump, `RESULTS17.md`, and `figures/`.

- **Install (17-set):** 12/17 fire; 4/17 clean (fire 1.00, control 0.00). The 5 failures are all
  extended-prose payloads (Euclid, Cantor, Parfit, benzene, Brent) — *not* length: friday's 86-tok
  code installs, greyhound's 135-tok proof does not, hopeful's 25-tok sentence is perfect.
- **Install (theme set):** 14/16 at exact 1.00, control ≈ 0. Much cleaner — every payload is one
  coherent theme.
- **Read direction → trigger (rank-1):** 5/12 installed trojans recover at ≥0.50 (norway/graph/dog
  1.00, hopeful 0.88, sleeping 0.75). Replaying the MAEM's text re-fires the backdoor (5-trojan
  study: 4/5 working exploits). **This is the DIT paper's open problem (§6.2, trigger inversion
  0/100) solved off the weights.**
- **Write direction → payload (rank-1):** 4/12 at ≥0.50 (drl 1.00, dog 0.92, violin 0.71,
  jalen_hurts 0.58). Two dissociations, both checked against raw rollouts: (a) installing ≠ reading
  out — graph/hopeful/december train at exact 1.00 and read out at 0.00; (b) the write vector
  carries the payload's **topic, not proposition** (hopeful emits the exact Somme sentence but `b`
  reads as "famous battles").
- **Rank-16 joint:** installs 7/17, control 0.00. In the SINGULAR basis (svd16, the only
  rotation-invariant read) it is effectively rank 1.63 of 16; read recovers ~4/7 triggers, write
  ~2/7 payloads. Superposition costs recovery even though it did not cost installation.

Figures (`results/run17/figures/`, regen `python trojan/results/make_figures.py`): fig1 read-vs-write
per trojan, fig2 install-vs-readout dissociation, fig3 singular spectrum, fig4 rank-1 vs rank-16.

## 8. Relation to DIT (arXiv 2510.05092, ICLR 2026)

Same problem shape (recover a backdoor from a weight diff with known ground truth), different reader.
DIT trains a rank-16 adapter to make the model self-describe; we read extracted directions through
the MAEM with no per-task training. DIT's stated frontier — cannot invert triggers (0/100), unsure
whether it reads weights or activations — is where we contribute: we recover triggers off the read
direction directly, and by decomposing the diff (SVD) we separate read from write, which recover
disjoint sets (jupiter's trigger reads at 14/16 though it never installed and its payload is absent).

## 9. Known-open items

- **Theme set readout not yet run.** Adapters are trained and saved; run `readout17
  --specs specs_theme --adapter-dir /data/trojan/multi_theme --directions write,read`. Expected to
  be the densest, cleanest recovery table since every payload is one theme.
- **No random-direction control in the 17/theme runs.** The 0.032 floor is inherited from the
  5-trojan study, not re-measured here. Cheap to add: inject N random unit directions through the
  same recipe and same judge.
- **Readout is noisy at n=24, temp 1.0.** Re-running moved jalen_hurts +0.20, norway +0.16 on
  identical weights; the zeros are bit-reproducible. Quote the intervals; raise `--bo` to tighten.
- **Install rates use n=4 held-out prefixes** (1.00 = [0.51,1.00]). More templates per trojan is
  the fix; the theme set already uses 16 generic frames but still holds out only 4.
- **Write columns not gate-matched across ranks.** `readout17` uses ungated `unit(W_down·b)`;
  `svd16` uses the gated operator. For a strict rank-1 vs rank-16 write comparison, re-run
  `readout17` with the gate applied.
- **`results/read_write_maem.csv`** (5-trojan era) picked direction signs by logit-lens coherence
  and got 6/10 wrong — do not cite it.
