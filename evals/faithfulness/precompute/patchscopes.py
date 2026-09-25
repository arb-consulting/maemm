"""Product `patchscopes`: the zero-shot patching baseline, on the CLEAN BASE.

    <root>/base/<base>/patchscopes/<set>/<cell>/rollouts.jsonl        row, family, k, text, ids,
                                                 n_tok, finished, engine "hf-patchscope", seed
    <root>/base/<base>/patchscopes/<set>/<cell>/rollouts.summary.json what `score` reads
    <root>/base/<base>/patchscopes/<set>/<cell>/README.md + index.json

What it measures: without any trained inverter, how high a read-layer cosine does the BASE model
reach when a held-out direction is patched into a Patchscopes-style target prompt and it is simply
allowed to generate? Ported from `experiments/2026-09-15_patchscopes-run1/patchscopes_modal.py`
(the 8B trial, README §2-5 there) into this pipeline's data layout, sampling constants and output
rules. The trial's finding that decides the port: **the no-injection floor is the result** -- on the
8B the same prompt generated with the hook DISABLED reached realact 0.1463 at bo=64 against the
best injected cell's 0.2333 and run1-RL's 0.5817, so an injected number without its matched floor
is mostly measuring "fluent English has a non-trivial cosine with a real activation direction".
Both arms are therefore products of this file, at the same n and on the same rows.

**The rows are rollouts.** `rollouts.jsonl` is byte-for-byte the schema `rollouts_hf.py` writes, so
`score.py --rollouts-dir <cell dir>` scores it through the one scoring path (checklist item 12) and
nothing here computes a cosine. `engine` is `hf-patchscope`, which is how a scores directory says
which generator produced its rows.

**Cells.**

| cell | prompt | patch |
|---|---|---|
| `floor` | P2 | none -- the hook is not installed at all |
| `p2-L<l>-replace2` | P2 | the placeholder's residual at the OUTPUT of block `l` is REPLACED by
                             `2 * ||h_l[pos]|| * unit(v)` |

P2 is Patchscopes §D.1's entity-description prompt, verbatim, fed as plain text with no chat
template (the paper uses base LMs); the placeholder is its trailing ` x`, the last prompt token.
The 8B screen chose it over the token-identity prompt P1 and over our own MAEM instruction prompt,
and chose `replace` x2 over MAEM's norm-matched `add` -- the base was never trained to read an
added direction. Those choices are carried over rather than re-screened (trial README §8.1); the
target LAYER is the only axis swept here, because it is the one whose transfer is a relative-depth
guess: the 8B's winner was layer **8 of 36 = 22.2%** depth, and the 27B has **64** layers, so
8 / 14 / 21 = **12.5% / 21.9% / 32.8%** bracket it. The default sweep also carries **42**,
the read layer every other method gets its activation from (see PS_LAYERS).

**The floor shares its texts.** The floor arm generates `n` continuations of the prompt ONCE and
writes one row per (direction, k) carrying the k-th shared text, because every direction must be
scored against the same sample set at the same bo for the comparison to be matched (the 8B screen's
floor was a best-of-16 against the cells' best-of-4, which flattered it -- trial README §5a). So a
floor `rollouts.jsonl` has N x n rows and only n distinct strings, and the GPU cost of the arm is n
generations; its scoring cost is the full N x n, like any other cell.

Sampling, seeding and trimming are `rollouts_hf`'s, not the trial's: `config.yaml rollouts` (T 1.0,
top_p 1.0, top_k 0, min_p 0, min_new 16, max_new 64), `common.gen_seed_for`, `common.trim_at_stop`,
`common.eos_ids`. Rows per generate call are capped at 32 on the 27B (checklist item 51: HF
`generate` over the GatedDeltaNet layers is >= 37x slower at batch >= 64).

**The final cell's shape is set by the $10 cap, not by the brief.** The brief specified 512
directions x bo 64 per family. HF `generate` finishes a batch only when EVERY row in it has stopped,
so a call costs `max_new` = 64 decode steps whether or not individual rows emit eos: the planning
figure is 64 generated tokens per rollout, not the ~40 a trimmed mean suggests. At this repo's own
measured 27B rate (239.3 gen tok/s at `gen_rows` 32, main README) that is

    512 x 64 x 2 families = 65,536 rows x 64 tok = 4.19M tok = 4.86 h = $22.0   (2.2x over the cap)
    512 x 32 x 2          = 32,768 rows         = 2.10M tok = 2.43 h = $11.0    (still over)
    384 x 32 x 2          = 24,576 rows         = 1.57M tok = 1.82 h =  $8.3    <- largest that fits
    256 x 32 x 2          = 16,384 rows         = 1.05M tok = 1.21 h =  $5.5
    512 x 16 x 2          = 16,384 rows         = 1.05M tok = 1.21 h =  $5.5

against ~$1.0 for the 3-layer sweep and ~$0.5 for scoring every cell. So the final cell runs the
largest rung that fits the envelope measured on the FIRST sweep cell at `gen_rows` 32 (the smoke's
31 tok/s was batch 8 -- per-step overhead, not the batch-32 rate), and every cell README records
which rung ran and the rate it ran at. `SPEC_SHAPE` below is the specified shape, quoted in that
note so the divergence is on the record wherever the numbers are.
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C
from precompute.rollouts_hf import GEN_ROWS

# Patchscopes appendix D.1, entity description, k=3 demonstrations, verbatim (the trial's P2_TEXT).
P2_TEXT = (
    "Syria: Country in the Middle East, Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer electronics "
    "corporation, x"
)
PROMPT_ID = "P2_description"
# Patchscopes appendix D.1, token identity, k=3 demonstrations. The 8B screen tried this
# and P2 beat it; whether that ordering holds on the 27B is exactly what --ps-prompt is
# for, since prompt / rule / alpha were all screened on the 8B and carried over unchanged
# (trial README 8.1) while only the LAYER was re-swept for this base.
P1_TEXT = "cat -> cat; 1135 -> 1135; hello -> hello; ? -> x"
PROMPTS = {"P2_description": P2_TEXT, "P1_identity": P1_TEXT}
# The 8B winner was layer 8 of 36 (22.2% depth) with `replace` at alpha 2. On the 27B's 64 layers
# 8 / 14 / 21 are 12.5% / 21.9% / 32.8%: one below, one at, one above the 8B's relative depth --
# the PAPER's tuned regime. 42 is added (2026-09-21) as the FAIR-INFORMATION cell: it is
# the read layer, i.e. the layer every other method -- our MAEMs and the NLA -- is handed its
# activation from. One shared sweep over both means nobody can say we skipped the paper's tuned
# mode, and nobody can say we gave Patchscopes less than we gave the others. Report every layer.
PS_LAYERS = (8, 14, 21, 42)
PS_ALPHA = 2.0
PS_RULE = "replace"
ENGINE = "hf-patchscope"
FLOOR_CELL = "floor"
# The final cell the brief specified, and the measured rate it is priced at (main README: HF 27B
# generate, gen_rows 32). Quoted in every cell README beside what actually ran -- see the docstring.
SPEC_SHAPE = "512 directions x bo 64 per family"
SPEC_RATE_TOK_S = 239.3
SPEC_USD_PER_H = 4.54


def patchscopes_dir(base: str, set_name: str, cell: str, root: str = C.VOL) -> str:
    """`<root>/base/<base>/patchscopes/<set>/<cell>` -- root-relative like every base product.

    Not in common.py: nothing outside this file and `reconstruction/stats.py` names the path, and
    common.py is the place conventions go when two products must agree on them.
    """
    return f"{C.base_dir(base, root)}/patchscopes/{set_name}/{cell}"


def cell_name(layer: int | None, tag: str = "", rule: str = PS_RULE,
              alpha: float = PS_ALPHA, prompt_id: str = PROMPT_ID) -> str:
    """The cell's directory name. `tag` (--ps-tag) suffixes it.

    Two runs of the SAME layer at different rollout budgets are different cells and must not share a
    directory -- the depth sweep runs at bo 8 and the final cell at bo 32, and `--force`-ing the
    second over the first would destroy the sweep's record. The name carries no bo of its own
    because the sweep's cells were already written without one; `--ps-tag bo32` is how the final run
    says which it is, and every cell's own `rollouts.summary.json` carries `n`/`bo` regardless.
    """
    if layer is None:
        base = FLOOR_CELL
    else:
        pfx = "p2" if prompt_id == PROMPT_ID else prompt_id.split("_")[0].lower()
        base = f"{pfx}-L{layer}-{rule}{alpha:g}"
    return f"{base}__{tag}" if tag else base


def prompt_ids(tok, prompt_id=PROMPT_ID):
    """(ids, placeholder position). PLAIN TEXT: no chat template, no special tokens."""
    text = PROMPTS[prompt_id]
    ids = list(tok.encode(text, add_special_tokens=False))
    assert len(ids) > 4, f"{prompt_id} tokenized to {len(ids)} ids; not what this expects"
    return ids, len(ids) - 1


def make_replace_hook(vecs, pos: int, alpha: float, device):
    """Replacement patch at `pos` on the hooked block's OUTPUT, one direction per batch row.

        h[i, pos] = alpha * ||h[i, pos]|| * unit(v_i)

    `vecs` is [B, d]; row i of the batch gets row i of `vecs`. The `h.shape[1] <= 1` guard is the
    decode step under the KV cache (maem/inject.py, copied into common.make_inject_hook for the add
    rule): the placeholder is patched at PREFILL only, and re-patching a decode step would corrupt
    it. The scale is the position's OWN residual norm, so alpha is dimensionless.

    `common.make_inject_hook` is add-mode only, on purpose -- it is the convention every MAEM was
    trained with and must not grow a mode nothing in the MAEM path uses. This is the Patchscopes
    rule and lives with the Patchscopes product.
    """
    import torch

    normed = torch.nn.functional.normalize(vecs.to(device, torch.float32), dim=-1)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        if h.shape[0] != normed.shape[0]:
            raise RuntimeError(f"patch batch {h.shape[0]} != {normed.shape[0]} direction rows")
        base = h[:, pos].float()
        scale = base.norm(dim=-1, keepdim=True) * alpha
        h[:, pos] = (normed * scale).to(h.dtype).detach()
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


def _base_identity(cfg, base: str) -> dict:
    """Weight identity of the CLEAN BASE, for the summary's `weight_sha256` (checklist item 29).

    There is no MAEM in this product, so the "checkpoint" a number must be pinned to is the base
    itself. Sharded checkpoints get `common.sha256_of_index` (index.json + shard sizes, no shard
    content -- see its docstring); a single-file checkpoint is NOT hashed, because streaming 16 GiB
    off the FUSE volume per run buys nothing here, and the README says so in those words.
    """
    snap = C.snapshot(cfg, cfg["bases"][base]["hf"])
    try:
        return C.sha256_of_index(snap)
    except AssertionError as e:
        return {
            "path": snap,
            "kind": f"NOT HASHED (not a sharded checkpoint: {e})",
            "sha256": f"n/a (clean base {cfg['bases'][base]['hf']}, not hashed)",
        }


def _generate(model, tok, prompt, pos, vecs, layer, alpha, rl, max_new, seed,
              rule=PS_RULE):
    """One generate call: len(vecs) rows of the identical prompt, one direction each.

    `layer is None` installs no hook at all -- that is the floor arm, and "no hook" is stricter than
    "a hook that returns early": nothing in the forward can be perturbed by it.
    """
    import torch

    ids = torch.tensor([list(prompt)] * len(vecs), dtype=torch.long, device="cuda")
    am = torch.ones_like(ids)
    assert ids.shape == (len(vecs), len(prompt)) and bool(am.all()), (
        f"every row shares the identical {len(prompt)}-token prompt, so the batch must be "
        f"rectangular and unpadded; got {tuple(ids.shape)}"
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kw = dict(
        input_ids=ids,
        attention_mask=am,
        do_sample=True,
        temperature=float(rl["temperature"]),
        top_p=float(rl["top_p"]),
        top_k=int(rl["top_k"]),
        min_p=0.0,
        max_new_tokens=max_new,
        min_new_tokens=int(rl["min_new"]),
        pad_token_id=tok.pad_token_id,
    )
    with torch.no_grad():
        if layer is None:
            gen = model.generate(**kw)
        else:
            if rule == "add":
                # The MAEM convention: norm-matched addition, one direction per row at the
                # single placeholder. Reuses common.make_inject_hook so the add path here and
                # the add path every MAEM was trained with cannot diverge.
                hook = C.make_inject_hook([v[None, :] for v in vecs], [[pos]] * len(vecs),
                                          alpha, "cuda", vecs.dtype)
            else:
                hook = make_replace_hook(vecs, pos, alpha, "cuda")
            with C.hooked(C.get_layer(model, layer), hook):
                gen = model.generate(**kw)
    return gen[:, len(prompt) :]


def _patch_check(model, prompt, pos, vecs, layer, alpha, rule=PS_RULE):
    """Is the patch a no-op? Two forwards of the shared prompt, the second one patched.

    D8 (2026-09-21): the check now builds the hook for the rule that will ACTUALLY RUN. It took no
    `rule` and always called `make_replace_hook`, while `_generate` branches at :227-234 -- so a
    `--ps-rule add` run was gated on an intervention that never happened, and the gate passed on
    the strength of a different experiment.

    Forward hooks fire in REGISTRATION order and each sees the previous one's output, so the probe
    is registered AFTER the patcher (the trial's §4.2: registering it first silently shows the clean
    state). Returns (||dh||/||h||, cos(dh, v), cos(h_patched, v)); the caller asserts on the pair
    its rule makes meaningful. Nothing here touches the generation path -- the hook a generate call
    uses is built fresh per call.
    """
    import torch

    seen: list = []

    def probe(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            seen.append(h[:, pos].float().cpu().clone())

    ids = torch.tensor([list(prompt)] * len(vecs), dtype=torch.long, device="cuda")
    am = torch.ones_like(ids)
    sub = C.get_layer(model, layer)
    with torch.no_grad():
        with C.hooked(sub, probe):
            model(input_ids=ids, attention_mask=am, use_cache=False)
        hook = (
            C.make_inject_hook([v[None, :] for v in vecs], [[pos]] * len(vecs), alpha, "cuda", vecs.dtype)
            if rule == "add"
            else make_replace_hook(vecs, pos, alpha, "cuda")
        )
        with C.hooked(sub, hook):
            with C.hooked(sub, probe):
                model(input_ids=ids, attention_mask=am, use_cache=False)
    clean, patched = seen[0], seen[1]
    delta = patched - clean
    rel = (delta.norm(dim=-1) / clean.norm(dim=-1).clamp(min=1e-6)).tolist()
    v_cpu = vecs.float().cpu()
    cos_delta = torch.nn.functional.cosine_similarity(delta, v_cpu, dim=-1).tolist()
    cos_abs = torch.nn.functional.cosine_similarity(patched, v_cpu, dim=-1).tolist()
    return rel, cos_delta, cos_abs


def run(cfg, args):
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product patchscopes needs --base"
    rl = cfg["rollouts"]
    n = int(args.get("n") or rl["n"])
    max_new = int(args.get("max_new") or rl["max_new"])
    base_seed = int(rl["seed"])
    gen_rows = int(args.get("gen_rows") or GEN_ROWS[base])
    # Prompt / rule / alpha were screened on the 8B and carried over; these make them
    # sweepable on THIS base, which is the one axis the 27B never got its own screen on.
    prompt_id = (args.get("ps_prompt") or PROMPT_ID)
    assert prompt_id in PROMPTS, f"--ps-prompt {prompt_id!r} not in {sorted(PROMPTS)}"
    rule = (args.get("ps_rule") or PS_RULE)
    assert rule in ("replace", "add"), f"--ps-rule must be replace|add, got {rule!r}"
    alpha = float(args.get("ps_alpha") or PS_ALPHA)
    tag = (args.get("ps_tag") or "").strip()
    assert "/" not in tag and " " not in tag, f"--ps-tag {tag!r} must be a bare directory suffix"
    spec = args.get("ps_layers") or ""
    layers = [int(x) for x in str(spec).replace(",", " ").split()] if spec else list(PS_LAYERS)
    n_layers = cfg["bases"][base]["n_layers"]
    for layer in layers:
        assert 0 <= layer < n_layers, f"--ps-layers {layer} is outside base {base}'s 0..{n_layers - 1}"
    cells: list[int | None] = list(layers)
    if not args.get("no_ps_floor"):
        cells = [None, *cells]  # the floor first: it is the control the injected arms are read against
    assert cells, "nothing to do: --ps-layers is empty and --no-ps-floor was passed"

    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows_meta = C.read_jsonl(f"{src}/ids.jsonl")
    d = cfg["bases"][base]["d"]
    # No `--maem` in this product either: the direction it patches in is whatever `--mu`
    # names, and on a `storage: raw` set common.mu_for refuses to pick one for it.
    cen_notes: list[str] = []
    mu, _ = C.mu_for(cfg, base, src, args, "", root, cen_notes)
    v = C.dirs_for(cfg, base, src, mu, root, cen_notes)
    assert v.shape == (len(rows_meta), d), f"{src}: dirs_for returned {v.shape}"
    dirs = torch.nn.functional.normalize(torch.from_numpy(np.asarray(v)), dim=-1)
    sel = C.parse_rows(args.get("rows", ""), len(rows_meta))
    for cell in cells:
        out = patchscopes_dir(base, set_name, cell_name(cell, tag, rule, alpha, prompt_id), root)
        assert args.get("force") or not os.path.exists(out), (
            f"{out} already exists; refusing to overwrite without --force"
        )

    model, tok = C.load_base(cfg, base)  # the CLEAN BASE: no adapter, no MAEM anywhere here
    prompt, pos = prompt_ids(tok, prompt_id)
    stop = C.eos_ids(tok, model)
    sha = _base_identity(cfg, base)
    print(
        f"[patchscopes] {base} {prompt_id} rule={rule} alpha={alpha:g}: "
        f"{len(prompt)} prompt tokens, placeholder "
        f"{tok.decode([prompt[pos]])!r} at {pos}; cells "
        f"{[cell_name(c, tag, rule, alpha, prompt_id) for c in cells]} x {len(sel)} directions x n={n}",
        flush=True,
    )

    results = {}
    for cell in cells:
        name = cell_name(cell, tag, rule, alpha, prompt_id)
        t0 = time.time()
        check = None
        if cell is not None:
            rel, cos_d, cos_a = _patch_check(
                model, prompt, pos, dirs[sel[: min(4, len(sel))]].cuda(), cell, alpha, rule
            )
            check = {
                "rule": rule,
                "rel_delta": [round(x, 4) for x in rel],
                "cos_delta_to_v": [round(x, 4) for x in cos_d],
                "cos_to_v": [round(x, 4) for x in cos_a],
            }
            print(
                f"[patchscopes] {name} PATCH CHECK rule={rule} ||dh||/||h|| {check['rel_delta']} "
                f"cos(dh, v) {check['cos_delta_to_v']} cos(h_patched, v) {check['cos_to_v']}",
                flush=True,
            )
            # D8: the statistic has to match the rule. Under `replace` the residual IS the
            # direction, so cos(h_patched, v) ~ 1 is the right gate. Under `add` it is not and
            # never was: alpha = 2 on a near-orthogonal residual lands around 0.894, so the old
            # gate would have failed a correct `add` run -- and it never fired, because the check
            # silently ran the REPLACE hook whatever the flag said. What `add` can honestly claim
            # is that what it ADDED is the direction (cos(dh, v) ~ 1) and that it moved the
            # placeholder at all.
            gate_cos, gate_name = (
                (cos_d, "cos(dh, v)") if rule == "add" else (cos_a, "cos(h_patched, v)")
            )
            assert min(gate_cos) > 0.99, (
                f"{name}: under rule {rule!r} the patch has {gate_name} = {min(gate_cos):.4f} with "
                f"the direction -- expected ~1.0; the patch is not doing what it says"
            )
            assert min(rel) > 0.5, f"{name}: patch moved the placeholder by only {min(rel):.4f} of its norm"

        out_rows: list[dict] = []
        gen_tok, n_calls = 0, 0
        if cell is None:
            # one shared sample set of n continuations, replicated across every direction
            texts: list[tuple[str, list[int]]] = []
            for s in range(0, n, gen_rows):
                k = min(gen_rows, n - s)
                seed = C.gen_seed_for(base_seed, sel[0], s, n)
                new = _generate(
                    model, tok, prompt, pos, torch.zeros(k, d), None, 0.0, rl, max_new, seed, rule
                )
                gen_tok += int(new.numel())
                n_calls += 1
                for g in new.tolist():
                    trimmed = C.trim_at_stop(g, stop)
                    texts.append((seed, trimmed))
            assert len(texts) == n, f"floor generated {len(texts)} texts, wanted {n}"
            for r in sel:
                for k, (seed, trimmed) in enumerate(texts):
                    out_rows.append(
                        {
                            "row": r,
                            "family": rows_meta[r]["family"],
                            "k": k,
                            "text": tok.decode(trimmed, skip_special_tokens=True),
                            "ids": [int(t) for t in trimmed],
                            "n_tok": len(trimmed),
                            "finished": bool(trimmed[-1] in stop),
                            "engine": ENGINE,
                            "seed": seed,
                        }
                    )
        else:
            pairs = [(r, k) for r in sel for k in range(n)]
            for s in range(0, len(pairs), gen_rows):
                chunk = pairs[s : s + gen_rows]
                seed = C.gen_seed_for(base_seed, chunk[0][0], chunk[0][1], n)
                vecs = torch.stack([dirs[r] for r, _ in chunk])
                new = _generate(model, tok, prompt, pos, vecs, cell, alpha, rl, max_new, seed, rule)
                gen_tok += int(new.numel())
                n_calls += 1
                for (r, k), g in zip(chunk, new.tolist(), strict=True):
                    trimmed = C.trim_at_stop(g, stop)
                    out_rows.append(
                        {
                            "row": r,
                            "family": rows_meta[r]["family"],
                            "k": k,
                            "text": tok.decode(trimmed, skip_special_tokens=True),
                            "ids": [int(t) for t in trimmed],
                            "n_tok": len(trimmed),
                            "finished": bool(trimmed[-1] in stop),
                            "engine": ENGINE,
                            "seed": seed,
                        }
                    )
                if n_calls % 10 == 1 or s + gen_rows >= len(pairs):
                    el = time.time() - t0
                    print(
                        f"[patchscopes] {name} {len(out_rows)}/{len(pairs)} rows | "
                        f"{gen_tok / max(el, 1e-9):.0f} gen tok/s | {el:.0f}s",
                        flush=True,
                    )
        elapsed = time.time() - t0
        distinct = len({x["text"] for x in out_rows})
        if cell is not None and n > 1:
            by_row: dict[int, set] = {}
            for rec in out_rows:
                by_row.setdefault(rec["row"], set()).add(rec["text"])
            for r, txts in by_row.items():
                assert len(txts) > 1, (
                    f"{name}: all {n} rollouts of row {r} decoded to the IDENTICAL text {txts} -- "
                    f"expected sampling at T={rl['temperature']} to differ"
                )

        n_tok = [x["n_tok"] for x in out_rows]
        summary = {
            "product": "patchscopes",
            "cell": name,
            "base": base,
            "set": set_name,
            "dirs_from": src,
            "prompt_id": prompt_id,
            "prompt_text": PROMPTS[prompt_id],
            "prompt_tokens": len(prompt),
            "placeholder_pos": pos,
            "placeholder_tok": tok.decode([prompt[pos]]),
            "patch_layer": cell,
            "patch_rule": None if cell is None else rule,
            "patch_alpha": None if cell is None else alpha,
            "patch_depth_frac": None if cell is None else round(cell / n_layers, 4),
            "n_layers": n_layers,
            "patch_check": check,
            "rows": sel,
            "n_targets": len(sel),
            "n": n,
            "bo": n,
            "seed": base_seed,
            "seed_rule": "rollouts.seed * 1000 + (row * n + k) of the generate call's first row",
            "engine": ENGINE,
            "kind": "clean base (no MAEM, no adapter)",
            "temperature": float(rl["temperature"]),
            "top_p": float(rl["top_p"]),
            "top_k": int(rl["top_k"]),
            "min_p": 0.0,
            "max_new": max_new,
            "min_new": int(rl["min_new"]),
            "gen_rows": gen_rows,
            "eos_ids": sorted(stop),
            "weight_sha256": sha["sha256"],
            "weights_path": sha["path"],
            "distinct_texts": distinct,
            "mean_n_tok": round(float(np.mean(n_tok)), 3),
            "eos_rate": round(float(np.mean([x["finished"] for x in out_rows])), 4),
            "gen_tok_per_s": round(gen_tok / max(elapsed, 1e-9), 1),
            "generate_seconds": round(elapsed, 1),
            "generate_calls": n_calls,
        }

        out = patchscopes_dir(base, set_name, name, root)
        inputs = {
            "base": f"{base} ({cfg['bases'][base]['hf']}, CLEAN -- no MAEM)",
            "dirs": src,
            "targets": f"{len(sel)} of {len(rows_meta)} rows",
            "n": n,
            "cell": name,
            "weight sha256": sha["sha256"],
        }
        # t0 PER CELL, not the container's. `common.outdir` takes t0 from args, which every
        # one-directory product wants (its wall covers the model load); this product writes SEVERAL
        # directories from one container, so passing the container's t0 to each would make every
        # cell after the first report the whole run's wall and cost. MEASURED 2026-09-16 on the
        # sweep, before this fix: the floor's README said 86.6 s / $0.1092 and L8's said 270.3 s /
        # $0.3409, although L8's own generation was 183 s -- the second number is the container's
        # total to that point, not the cell's. The model load is charged to the FIRST cell, which is
        # where it is actually paid.
        with C.outdir(out, {**args, "t0": t0}, inputs=inputs) as od:
            C.note_convention(od, cen_notes)
            od.write_jsonl("rollouts.jsonl", out_rows)
            od.write_json("rollouts.summary.json", summary)
            od.note(
                "`rollouts.jsonl` is EXACTLY the schema `rollouts_hf` writes -- row, family, k, "
                "text, ids (the generated ids only, trimmed at the first stop token which is KEPT), "
                f"n_tok, finished, engine ({ENGINE!r}), seed -- so `score.py --rollouts-dir "
                f"{out}` scores it through the one scoring path and nothing here computes a cosine."
            )
            od.section(
                "Cell",
                [
                    f"- prompt `{prompt_id}` (Patchscopes appendix D.1, verbatim, PLAIN TEXT: no chat "
                    "template, `add_special_tokens=False`):",
                    "",
                    f"  > {PROMPTS[prompt_id]}",
                    "",
                    f"- placeholder: the last prompt token, {summary['placeholder_tok']!r} at position "
                    f"{pos} of {len(prompt)}",
                    (
                        "- patch: NONE. The hook is not installed at all -- this is the no-injection "
                        "FLOOR, the control every injected cell must be read against (trial README "
                        "§5a: on the 8B the floor out-scored 80 of 90 injected cells)."
                        if cell is None
                        else f"- patch: at the OUTPUT of decoder block {cell} "
                        f"({cell}/{n_layers} = {summary['patch_depth_frac']:.1%} depth; the 8B trial's "
                        f"winner was layer 8 of 36 = 22.2%), the placeholder's residual is REPLACED by "
                        f"`{alpha:g} * ||h[pos]|| * unit(v)` under rule `{rule}`, at prefill only."
                    ),
                    f"- sampling: T={rl['temperature']} top_p={rl['top_p']} top_k={rl['top_k']} min_p=0 "
                    f"min_new={rl['min_new']} max_new={max_new}, n={n} per direction, {gen_rows} rows "
                    f"per generate call",
                    f"- seed rule: {summary['seed_rule']} (base seed {base_seed}), stored per row",
                    f"- weights: the CLEAN BASE {cfg['bases'][base]['hf']} at {sha['path']}, sha256 "
                    f"`{sha['sha256']}` ({sha.get('kind', '')})",
                ],
            )
            if cell is None:
                od.note(
                    f"FLOOR: {n} continuations were generated ONCE and replicated across all "
                    f"{len(sel)} directions, so this file has {len(out_rows)} rows and "
                    f"{distinct} distinct strings. Every direction is scored against the SAME "
                    f"sample set at the SAME bo -- an unmatched bo is what flattered the 8B "
                    f"screen's floor."
                )
            else:
                od.note(
                    f"patch check (observation only, two extra forwards of the shared prompt before "
                    f"any generation): ||dh||/||h|| {check['rel_delta']}, cos(h_patched, v) "
                    f"{check['cos_to_v']} -- asserted > 0.99, since after a REPLACEMENT the patched "
                    f"residual IS the direction"
                )
                od.note(
                    f"{distinct} distinct strings over {len(out_rows)} rows; every direction's n "
                    f"rollouts asserted not to be identical"
                )
            spec_tok = 512 * 64 * 2 * max_new
            spec_h = spec_tok / SPEC_RATE_TOK_S / 3600
            od.note(
                f"cost plan (checklist item 84 -- this README is the cost of record): the brief "
                f"specified a final cell of {SPEC_SHAPE}; at the measured 27B rate of "
                f"{SPEC_RATE_TOK_S} gen tok/s at gen_rows 32 that is {spec_tok / 1e6:.2f}M generated "
                f"tokens = {spec_h:.2f} h = ${spec_h * SPEC_USD_PER_H:.1f}, against a $10 cap for the "
                f"whole product -- HF generate runs a batch until EVERY row stops, so the planning "
                f"figure is max_new={max_new} tokens per rollout, not the trimmed mean. The final "
                f"cell therefore runs the largest rung that fits (384x32x2 ~ $8.3, 256x32x2 ~ $5.5, "
                f"512x16x2 ~ $5.5). THIS cell ran {len(sel)} directions x bo {n} = {len(out_rows)} "
                f"rows at {summary['gen_tok_per_s']} gen tok/s."
            )
            od.note(
                f"throughput: {summary['gen_tok_per_s']} generated tok/s over {elapsed:.0f}s in "
                f"{n_calls} generate calls; mean {summary['mean_n_tok']} kept tokens per rollout, "
                f"eos rate {summary['eos_rate']}. The wall and cost above are THIS CELL's "
                f"(the clean-base load is charged to the first cell of the container); the whole "
                f"product call's total is the `[wall]` line of its Modal log."
            )
        results[name] = {
            "out": out,
            "rows": len(out_rows),
            "distinct_texts": distinct,
            "gen_tok_per_s": summary["gen_tok_per_s"],
            "mean_n_tok": summary["mean_n_tok"],
            "seconds": round(elapsed, 1),
        }
        print(f"[patchscopes] {name}: {len(out_rows)} rows in {elapsed:.0f}s -> {out}", flush=True)
    return {"cells": results, "targets": len(sel), "n": n}
