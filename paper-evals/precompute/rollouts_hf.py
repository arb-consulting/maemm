"""Product `rollouts_hf`: MAEMM rollouts on the HF `generate` path.

    <root>/maemms/<base>/<maemm>/rollouts/<set>.jsonl   one row per (target, rollout)
    <root>/maemms/<base>/<maemm>/rollouts/README.md     + index.json, one entry per set present
    <root>/maemms/<base>/<maemm>/README.md              the MAEMM's identity card, written once

Rollouts and scores are SEPARATE products (checklist item 79): generation is the expensive,
bit-reproducible half, and rescoring is cheap. Nothing here computes a cosine; nothing in score.py
loads the MAEMM.

The recipe is eval/eval_universal.py:_gen_batches (505-522) with Celeste's sampling constants
(rl/rl.py:322-326: T 1.0, top_p 1.0, top_k 0, min_p 0; min_new 16, max_new 64 from config):

  * every row of a generate call carries the IDENTICAL prompt, so the batch needs no padding at all
    (asserted below) and the marker is the last prompt token;
  * the direction is injected at the marker of INJECT_LAYER's block output, add mode, coeff 1.0
    (common.make_inject_hook), at prefill only;
  * the generated ids are trimmed at the first stop token, which is KEPT (rl/rl.py:82-90), and the
    stored text is the decode of the trimmed ids with skip_special_tokens=True -- the same string
    her `tok.batch_decode(gen[:, p_len:], skip_special_tokens=True)` produces;
  * FULL-parameter MAEMM: the served (tuned) model generates and the ||h|| the injection scales by
    is its OWN (eval/eval_ckpt_daemon.py:333-390). Scoring is always the clean base, in score.py.

DIVERGENCE from her seeding, deliberate and recorded on every row. She forks the RNG once per eval
(`torch.random.fork_rng` + `torch.manual_seed(GEN_SEED)`, eval_universal.py:804-805), so a row's
sample depends on every batch before it. HF `generate` takes no `torch.Generator` on every version,
so instead each generate call is seeded `rollouts.seed * 1000 + flat` right before it, with
`flat = row * n + k` of its first row (common.gen_seed_for), and the seed is stored per row. Our
rollouts are therefore NOT bitwise hers; they are reproducible per chunk from the stored seed.
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C

# Rows per generate call. 27B: checklist item 51 -- HF generate over the GatedDeltaNet layers is
# >= 37x slower at batch >= 64, so it is capped at 32 (half a 64-rollout target per call).
GEN_ROWS = {"qwen3-8b": 256, "qwen36-27b": 32}
# The marker norm under the served model must differ from the clean base by at least this fraction
# (checklist item 22: equality is the silent-adapter-off signature; 8B run1 is ~98.5 vs ~14.06).
MARKER_NORM_MIN_REL = 0.05
# ...EXCEPT for `type: base`, the untrained-base control, which serves the base snapshot itself and
# must AGREE with it. 25% is a sanity bound catching a trained checkpoint served by mistake (the
# 27B's sit at 130 and 512 against the base's 14.06), not a numerical-agreement claim.
BASE_CONTROL_NORM_TOL = 0.25


def load_dirs(cfg, args, device: str = "cuda", notes=None):
    """(rows, dirs [N, d] fp32 unit on `device`, source dir) of the held-out set or `--dirs-from`.

    rollouts_vllm asks for the cpu: the engine owns the GPU by the time it needs the directions.

    The direction is DERIVED, never read: `common.dirs_for` applies the centring this run names,
    which for a generator is the checkpoint's own `input.centering` (what it was TRAINED to
    receive) unless `--centering` overrides it. Handing a MAEMM a direction under the wrong mean is
    invisible in every output -- the rollouts look like rollouts -- so the convention is resolved
    here, once, and written into the summary by the caller. `notes` collects the lines that say
    which; rollouts_vllm:1007 (parity-greedy) and rollouts_nla:672 come through the same call.
    """
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{src}/ids.jsonl")
    n = len(rows)
    assert n, f"{src}/ids.jsonl is empty"
    centering, _ = C.centering_for(cfg, base, src, args, args.get("maemm") or "", root, notes)
    v = C.dirs_for(cfg, base, src, centering, root, notes)
    assert v.shape == (n, cfg["bases"][base]["d"]), f"{src}: dirs_for returned {v.shape} for {n} rows"
    dirs = torch.nn.functional.normalize(torch.from_numpy(np.asarray(v)).to(device), dim=-1)
    for i, r in enumerate(rows):
        assert r["row"] == i, f"{src}/ids.jsonl line {i} has row={r['row']}: rows must be 0..N-1 in order"
    return rows, dirs, src


def weight_identity(cfg, maemm_key: str) -> dict:
    """The MAEMM's weight sha for its README (checklist item 74).

    lora: a full streamed sha256 of the adapter (1.3-1.7 GiB). full / base: index.json + shard
    sizes only -- hashing 52 GiB off the FUSE volume costs minutes of H200 for a field nobody
    diffs, and common.sha256_of_index says so in the README it lands in. For the untrained-base
    control this hashes the BASE snapshot, which is the identity that matters there.
    """
    path = C.maemm_weights_path(cfg, maemm_key)
    if cfg["maemms"][maemm_key]["type"] == "lora":
        return C.sha256_of_weights(path)
    return C.sha256_of_index(path)


def write_maemm_readme(cfg, args, maemm_key: str, sha: dict, prompt_name: str, n_prompt: int) -> str:
    """`<root>/maemms/<base>/<maemm>/README.md`, the identity card. Written once; --force rewrites."""
    spec = cfg["maemms"][maemm_key]
    path = f"{C.maemm_dir(maemm_key, args['root'])}/README.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not args.get("force"):
        print(f"[maemm] README already at {path}; leaving it (pass --force to rewrite)", flush=True)
        return path
    inj = spec["inject"]
    lines = [
        f"# {maemm_key}",
        "",
        f"- source: {spec.get('hf') or spec.get('src')} ({'HF repo id' if 'hf' in spec else 'volume path'})",
        f"- type: {spec['type']}",
        f"- adapter subdir: {spec.get('subdir') or '(repo root)'}",
        f"- resolved weights: {C.maemm_weights_path(cfg, maemm_key)}",
        f"- train_max_new: {spec.get('train_max_new')} (the budget it was TRAINED at)",
        *([f"- role: {spec['role']}"] if spec.get("role") else []),
        *([f"- note: {spec['note']}"] if spec.get("note") else []),
        f"- written by: `{' '.join(args.get('argv') or [])}`",
        f"- repo commit: {args.get('repo_commit', '?')[:12]}",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        "",
        *(
            [
                "## THIS IS THE UNTRAINED-BASE CONTROL",
                "",
                "**No MAEMM weights of any kind.** The served model IS the base snapshot "
                f"`{spec['hf']}` -- the same one `bases.{C.split_key(maemm_key, 'maemm')[0]}` "
                "resolves to -- with no adapter, no fine-tuned shards and nothing loaded on top.",
                "Everything else is the primary MAEMM's: the same prompt function, the same marker "
                "token, the same block-1 norm-matched injection at the same coefficient and the "
                "same `rollouts:` sampling constants. The only difference is the weights, which is "
                "what makes the difference in the tables attributable to training.",
                "",
                "Its marker ||h|| therefore EQUALS the clean base's, where every trained MAEMM's "
                "differs from it -- the self-checks invert accordingly "
                "(rollouts_vllm.marker_norm_vs_hf, rollouts_hf.marker_check).",
                "",
            ]
            if spec["type"] == "base"
            else []
        ),
        "## Injection convention",
        "",
        f"- inject layer: {inj['layer']} (decoder block OUTPUT), mode add, coeff {inj['coef']}",
        "- `h[marker] += unit(v) * ||h[marker]|| * coeff`, at PREFILL only (mxf/inject.py:10-57,",
        "  copied into common.make_inject_hook); decode steps are skipped.",
        f"- prompt function: `{prompt_name}` (common.PROMPTS), {n_prompt} tokens, marker "
        f'"{C.MARKER}" as the LAST prompt token, occurring exactly once.',
        "- full-parameter MAEMMs generate with their OWN marker norm; every score runs on the clean",
        "  base (eval/eval_ckpt_daemon.py:333-390).",
        "",
        "## Weight identity",
        "",
        f"- path: {sha['path']}",
        f"- combined sha256: `{sha['sha256']}`",
        f"- kind: {sha.get('kind', 'streamed sha256 of every *.safetensors / *.bin / *.pt')}",
        "",
    ]
    if "files" in sha:
        lines += ["| file | bytes | sha256 |", "|---|---|---|"]
        lines += [f"| `{k}` | {v['bytes']} | `{v['sha256']}` |" for k, v in sha["files"].items()]
    else:
        lines += [f"- index.json sha256: `{sha['index_sha256']}`", "", "| shard | bytes |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in sha["shards"].items()]
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[maemm] wrote {path}", flush=True)
    return path


def marker_check(cfg, args, model, kind, prompt, mpos, inj_layer):
    """Checklist item 22, OBSERVATION ONLY: ||h|| at the marker, served vs clean base.

    Two extra forwards of the SINGLE shared prompt before any generation. Nothing here is hooked
    into the generation path -- the hook this measures is built fresh per generate call, so this
    cannot perturb a rollout. `--no-marker-check` skips it entirely.

    LoRA: the clean-base number comes from the same object with the adapter disabled. A FULL
    -parameter MAEMM has no adapter to switch off and a second 52 GiB load is not worth its H200
    minute, so the base number comes from `bases.<base>.marker_norm_base` in config.yaml when it is
    there (measured by an earlier LoRA run on the same base) and is otherwise SKIPPED with a note.
    Returns (served, base_or_None, source).
    """
    base = args["base"]
    hn_served = C.marker_norm(model, prompt, mpos, inj_layer, adapter=True)
    if args.get("no_marker_check"):
        print(f"[rollouts] marker ||h|| served {hn_served:.3f}; --no-marker-check, no comparison", flush=True)
        return hn_served, None, "skipped (--no-marker-check)"
    if kind == "base":
        # The untrained-base CONTROL serves the base itself: served == clean base is the CORRECT
        # outcome here, the opposite of every other kind, so the "must differ" assert below must
        # not run. What is checked instead is that a trained checkpoint was not served by mistake.
        cfg_val = cfg["bases"][base].get("marker_norm_base")
        src = f"config.yaml bases.{base}.marker_norm_base (untrained-base control: EXPECTED EQUAL)"
        if cfg_val is None:
            print(
                f"[rollouts] marker ||h|| served {hn_served:.3f}; no bases.{base}.marker_norm_base",
                flush=True,
            )
            return hn_served, None, f"skipped (base control, no bases.{base}.marker_norm_base in config)"
        hn_base = float(cfg_val)
        rel = abs(hn_served - hn_base) / max(hn_base, 1e-6)
        print(
            f"[rollouts] marker ||h|| served {hn_served:.3f} vs clean base {hn_base:.3f} [{src}]",
            flush=True,
        )
        assert rel <= BASE_CONTROL_NORM_TOL, (
            f"the untrained-base control measures marker ||h|| {hn_served:.4f} against the clean "
            f"base's {hn_base:.4f} ({rel:.2%}, sanity bound {BASE_CONTROL_NORM_TOL:.0%}): a "
            f"TRAINED checkpoint was served where the control was asked for (the 27B's sit at 130 "
            f"and 512 against 14.06)"
        )
        return hn_served, hn_base, src
    if kind == "lora":
        hn_base = C.marker_norm(model, prompt, mpos, inj_layer, adapter=False)
        src = "measured here with the adapter disabled"
    else:
        cfg_val = cfg["bases"][base].get("marker_norm_base")
        if cfg_val is None:
            print(
                f"[rollouts] marker ||h|| served {hn_served:.3f}; NO clean-base comparison: this is "
                f"a full-parameter MAEMM and config.yaml has no bases.{base}.marker_norm_base",
                flush=True,
            )
            return hn_served, None, f"skipped (full model, no bases.{base}.marker_norm_base in config)"
        hn_base = float(cfg_val)
        src = f"config.yaml bases.{base}.marker_norm_base (a full model has no adapter to disable)"
    print(f"[rollouts] marker ||h|| served {hn_served:.3f} vs clean base {hn_base:.3f} [{src}]", flush=True)
    assert abs(hn_served - hn_base) > MARKER_NORM_MIN_REL * max(hn_base, 1e-6), (
        f"marker ||h|| is {hn_served:.4f} under the served model and {hn_base:.4f} under the clean "
        f"base -- they must differ by more than {MARKER_NORM_MIN_REL:.0%}; equality is the "
        f"silent-adapter-off signature (checklist item 22, 8B run1 is ~98.5 vs ~14.06)"
    )
    return hn_served, hn_base, src


def run(cfg, args):
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base, "product rollouts_hf needs --base"
    assert maemm, "product rollouts_hf needs --maemm"
    assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    key_base, _ = C.split_key(maemm, "maemm")
    assert key_base == base, f"maemm {maemm!r} is on base {key_base!r}, not {base!r}"

    spec = cfg["maemms"][maemm]
    # `type: nla` is the activation VERBALIZER baseline: a different prompt, a different marker
    # character and a marker that is not the last prompt token. Every assert below about the MAEMM
    # prompt would either fire or, worse, pass on a prompt the checkpoint never saw.
    assert spec["type"] != "nla", (
        f"maemm {maemm!r} is an NLA entry: it generates with `--product rollouts_nla`, which "
        f"builds the verbalizer's own prompt and marker from the checkpoint's nla_meta.yaml"
    )
    rl = cfg["rollouts"]
    n = int(args.get("n") or rl["n"])
    max_new = int(args.get("max_new") or rl["max_new"])
    min_new = int(rl["min_new"])
    temp = float(rl["temperature"])
    base_seed = int(rl["seed"])
    inj_layer, coef = int(spec["inject"]["layer"]), float(spec["inject"]["coef"])
    gen_rows = int(args.get("gen_rows") or GEN_ROWS[base])
    assert gen_rows > 0, f"--gen-rows must be positive, got {gen_rows}"

    out_dir = C.rollouts_dir(maemm, root)
    path = f"{out_dir}/{set_name}.jsonl"
    assert args.get("force") or not os.path.exists(path), (
        f"{path} already exists; refusing to overwrite without --force"
    )

    cen_notes: list[str] = []
    rows_meta, dirs, dirs_src = load_dirs(cfg, args, notes=cen_notes)
    sel = C.parse_rows(args.get("rows", ""), len(rows_meta))
    print(
        f"[rollouts] {maemm} on {len(sel)} of {len(rows_meta)} targets x {n} rollouts "
        f"= {len(sel) * n} rows, {gen_rows} per generate call, dirs from {dirs_src}",
        flush=True,
    )

    model, tok, kind = C.load_maemm(cfg, base, maemm)
    prompt, mpos = C.prompt_ids(tok, spec["prompt"], cfg["bases"][base]["read_layer"])
    assert mpos == len(prompt) - 1, f"the marker must be the LAST prompt token, got {mpos} of {len(prompt)}"
    stop = C.eos_ids(tok, model)
    sub = C.get_layer(model, inj_layer)

    hn_served, hn_base, hn_src = marker_check(cfg, args, model, kind, prompt, mpos, inj_layer)

    pairs = [(r, k) for r in sel for k in range(n)]
    out_rows: list[dict] = []
    t0, gen_tok, n_calls = time.time(), 0, 0
    for s in range(0, len(pairs), gen_rows):
        chunk = pairs[s : s + gen_rows]
        seed = C.gen_seed_for(base_seed, chunk[0][0], chunk[0][1], n)
        vecs = [dirs[r : r + 1] for r, _ in chunk]
        hook = C.make_inject_hook(vecs, [[mpos]] * len(chunk), coef, "cuda", torch.bfloat16)
        ids = torch.tensor([list(prompt)] * len(chunk), dtype=torch.long, device="cuda")
        am = torch.ones_like(ids)
        assert ids.shape == (len(chunk), len(prompt)) and bool(am.all()), (
            f"every row shares the identical {len(prompt)}-token prompt, so the batch must be "
            f"rectangular and unpadded; got {tuple(ids.shape)}"
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with C.hooked(sub, hook), torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=am,
                do_sample=True,
                temperature=temp,
                top_p=float(rl["top_p"]),
                top_k=int(rl["top_k"]),
                min_p=0.0,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, len(prompt) :]
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
                    "engine": "hf",
                    "seed": seed,
                }
            )
        if n_calls % 10 == 1 or s + gen_rows >= len(pairs):
            el = time.time() - t0
            print(
                f"[rollouts] {len(out_rows)}/{len(pairs)} rows | {gen_tok / max(el, 1e-9):.0f} "
                f"gen tok/s | {el:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0

    by_row: dict[int, list[dict]] = {}
    for rec in out_rows:
        by_row.setdefault(rec["row"], []).append(rec)
    if n > 1:
        for r, recs in by_row.items():
            assert len({x["text"] for x in recs}) > 1, (
                f"target row {r} ({recs[0]['family']}): all {len(recs)} rollouts decoded to the "
                f"IDENTICAL text {recs[0]['text']!r} -- expected sampling at T={temp} to differ; "
                f"a shared-seed / greedy / no-injection bug looks exactly like this"
            )
    n_tok = [x["n_tok"] for x in out_rows]
    summary = {
        "maemm": maemm,
        "base": base,
        "set": set_name,
        "dirs_from": dirs_src,
        "rows": sel,
        "n_targets": len(sel),
        "n": n,
        "bo": n,
        "seed": base_seed,
        "seed_rule": "rollouts.seed * 1000 + (row * n + k) of the generate call's first row",
        "engine": "hf",
        "kind": kind,
        "prompt": spec["prompt"],
        "prompt_tokens": len(prompt),
        "marker_pos": mpos,
        "inject_layer": inj_layer,
        "inject_coef": coef,
        "temperature": temp,
        "top_p": float(rl["top_p"]),
        "top_k": int(rl["top_k"]),
        "min_p": 0.0,
        "max_new": max_new,
        "min_new": min_new,
        "gen_rows": gen_rows,
        "marker_norm_served": round(hn_served, 4),
        "marker_norm_clean_base": None if hn_base is None else round(hn_base, 4),
        "marker_norm_base_source": hn_src,
        "eos_ids": sorted(stop),
        "mean_n_tok": round(float(np.mean(n_tok)), 3),
        "eos_rate": round(float(np.mean([x["finished"] for x in out_rows])), 4),
        "gen_tok_per_s": round(gen_tok / max(elapsed, 1e-9), 1),
        "generate_seconds": round(elapsed, 1),
        "generate_calls": n_calls,
    }

    sha = weight_identity(cfg, maemm)
    summary["weight_sha256"] = sha["sha256"]
    write_maemm_readme(cfg, args, maemm, sha, spec["prompt"], len(prompt))

    inputs = {
        "maemm": maemm,
        "dirs": dirs_src,
        "targets": f"{len(sel)} of {len(rows_meta)} rows",
        "n": n,
        "weight sha256": sha["sha256"],
    }
    with C.outdir(out_dir, args, inputs=inputs, keep_existing=os.path.exists(out_dir)) as od:
        C.note_convention(od, cen_notes)
        od.write_jsonl(f"{set_name}.jsonl", out_rows)
        od.write_json(f"{set_name}.summary.json", summary)
        od.note(
            f"`{set_name}.jsonl`: one row per (target, rollout) -- row, family, k, text, ids "
            "(the GENERATED ids only, trimmed at the first stop token which is KEPT, "
            "rl/rl.py:82-90), n_tok, finished, engine, seed. `text` is decode(ids, "
            "skip_special_tokens=True). The prompt is NOT part of either field (checklist item 8)."
        )
        od.note(
            f"sampling: T={temp} top_p={rl['top_p']} top_k={rl['top_k']} min_p=0 min_new={min_new} "
            f"max_new={max_new}, {n} rollouts per target, {gen_rows} rows per generate call, all "
            "rows of a call sharing the identical prompt so the batch is UNPADDED (asserted)"
        )
        od.note(
            f"seed rule: {summary['seed_rule']} (base seed {base_seed}); set with torch.manual_seed "
            "+ torch.cuda.manual_seed_all immediately before each generate and stored on every row. "
            "NOT bitwise Celeste's, who forks the RNG once per eval (eval_universal.py:804-805)."
        )
        od.note(
            f"marker ||h|| at inject layer {inj_layer} (checklist item 22, observation only -- two "
            f"extra forwards of the shared prompt BEFORE generation, nothing hooked into the "
            f"generation path): served {hn_served:.4f}, clean base "
            + (
                (
                    f"{hn_base:.4f} [{hn_src}], asserted EQUAL within "
                    f"{BASE_CONTROL_NORM_TOL:.0%} (untrained-base control)"
                    if kind == "base"
                    else f"{hn_base:.4f} [{hn_src}], asserted to differ by > {MARKER_NORM_MIN_REL:.0%}"
                )
                if hn_base is not None
                else f"NOT COMPARED -- {hn_src}"
            )
        )
        od.note(
            f"weights: {sha['path']} sha256 {sha['sha256']} "
            f"({sha.get('kind', 'streamed content hash')}); identity card in ../README.md"
        )
        od.note(
            f"throughput: {summary['gen_tok_per_s']} generated tok/s over {elapsed:.0f}s in "
            f"{n_calls} generate calls; mean {summary['mean_n_tok']} kept tokens per rollout, "
            f"eos rate {summary['eos_rate']}"
        )
        od.note(
            "this directory ACCUMULATES one <set>.jsonl + <set>.summary.json per run; index.json "
            "lists every set present, while the header above describes the MOST RECENT run only"
        )
    return {
        "out": path,
        "rows": len(out_rows),
        "targets": len(sel),
        **{k: summary[k] for k in ("marker_norm_served", "marker_norm_clean_base", "gen_tok_per_s")},
        "mean_n_tok": summary["mean_n_tok"],
        "eos_rate": summary["eos_rate"],
        "weight_sha256": sha["sha256"],
    }
