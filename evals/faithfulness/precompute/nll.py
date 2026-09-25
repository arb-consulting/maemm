"""Product `nll`: the base model's own competence on each target's window (design §5, review R6).

    <root>/base/<base>/nll/<set>/    per_target.jsonl, nll.f32 [N, 3], README.md, index.json

One forward of the CLEAN BASE per target window -- the same 512-token, no-BOS window the target was
read from (`targets`' `_realact` geometry) -- storing, per target:

    nll_ctx   mean nats per token over positions 1..p of the window (position 0 is never predicted:
              there is no context for it, and no sink token is prepended here)
    bpb_ctx   BITS PER BYTE of the same span: (sum of the nats over 1..p) / ln(2) / the UTF-8 byte
              length of `decode(ids[1 : p+1])`. This is the number the paper reports (R6), because
              nats per TOKEN are not comparable across scripts -- Thai and English pay a different
              number of tokens for the same text
    nll_p     the nats of the token AT p, the position the target direction was read from

Why it exists: `precompute/` computed no NLL before this (the OOD survey's "`score.py` already
computes it" was wrong: `grep -rli nll evals/faithfulness` hit only `gcg/` and `reconstruction/stats.py`).
What it is FOR is narrower than the first design said (R6): the identification claim -- "this lets
'the MAEM is worse on Thai' be read apart from 'the base is worse on Thai'" -- is withdrawn. The
missing control is an inverter TRAINED on the domain, which this evaluation does not have. bits per
byte is reported descriptively, with the within-arm Spearman against bo64.

Runs on an OOD set (whose windows are stored in `heldout/<set>/windows.i32`) and on the 512
English `realact` rows of `2026-09-16_v1` (whose windows are the first 512 tokens of their corpus
document). No MAEM is ever loaded.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

import precompute.common as C

WINDOW = 512  # targets.REALACT_WINDOW; a target's window is the document's first 512 tokens
FWD_ROWS = 4  # windows per forward: [4, 512, 248k] bf16 logits is ~1.0 GiB on the 27B
V_CHUNK = 128  # positions per fp32 log_softmax chunk, to bound the transient


def _windows_for(cfg, args):
    """[(row, family, arm, p, ids[512])] for every target of the set that has a window.

    An OOD set stores its windows (`windows.i32`); the 2026-09-16_v1 `realact` family does not,
    because its windows ARE the first 512 tokens of a corpus document, so they are read back from
    `corpus/tokens.i32`. `random` and `sae` targets have no window at all and are skipped -- the
    product is about the text a real activation came from.
    """
    base, root, set_name = args["base"], args["root"], args["heldout"]
    hdir = C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{hdir}/ids.jsonl")
    wpath = f"{hdir}/windows.i32"
    if os.path.exists(wpath):
        wins = C.read_array(wpath, "int32", (len(rows), WINDOW))
        return [
            (r["row"], r["family"], r.get("arm"), int(r["p"]), wins[i]) for i, r in enumerate(rows)
        ], f"{hdir}/windows.i32"
    toks, docs = C.load_corpus(base, root)
    by_doc = {int(r["doc"]): r for r in docs}
    out = []
    for r in rows:
        if r["family"] != "realact":
            continue
        d = by_doc[int(r["doc"])]
        assert d["len"] >= WINDOW, f"corpus doc {r['doc']} is shorter than the {WINDOW}-token window"
        out.append(
            (r["row"], r["family"], None, int(r["p"]), np.asarray(toks[d["offset"] : d["offset"] + WINDOW]))
        )
    return out, C.corpus_dir(base, root)


def _forward_nll(model, ids):
    """Per-position nats of an [B, T] int array: nats[b, t] = -log p(ids[b, t] | ids[b, :t]).

    Column 0 is nan (nothing predicts it). The log-softmax runs in fp32 over position chunks, so
    the transient is `V_CHUNK x vocab` and not the whole [B, T, vocab] logit tensor again.
    """
    import torch

    t = torch.from_numpy(np.asarray(ids, dtype=np.int64)).cuda()
    with torch.no_grad():
        logits = model(input_ids=t, attention_mask=torch.ones_like(t)).logits
    b, tt, _ = logits.shape
    nats = torch.full((b, tt), float("nan"))
    for s in range(0, tt - 1, V_CHUNK):
        e = min(s + V_CHUNK, tt - 1)
        lp = torch.log_softmax(logits[:, s:e].float(), dim=-1)
        tgt = t[:, s + 1 : e + 1]
        nats[:, s + 1 : e + 1] = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
    del logits
    return nats.numpy()


def run(cfg, args):
    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product nll needs --base"
    targets, src = _windows_for(cfg, args)
    assert targets, f"set {set_name!r} has no target with a window on base {base!r}"
    out = C.nll_dir(base, set_name, root)
    rows_arg = args.get("rows")
    if rows_arg:
        keep = set(C.parse_rows(rows_arg, max(t[0] for t in targets) + 1))
        targets = [t for t in targets if t[0] in keep]

    model, tok = C.load_base(cfg, base)
    recs, arr, t0 = [], [], time.time()
    for s in range(0, len(targets), FWD_ROWS):
        chunk = targets[s : s + FWD_ROWS]
        nats = _forward_nll(model, np.stack([c[4] for c in chunk]))
        for k, (row, fam, arm, p, ids) in enumerate(chunk):
            assert 1 <= p < WINDOW, f"row {row}: p={p} outside [1, {WINDOW})"
            span = nats[k, 1 : p + 1]
            assert np.isfinite(span).all(), f"row {row}: non-finite nats in positions 1..{p}"
            text = tok.decode([int(x) for x in ids[1 : p + 1]])
            n_bytes = len(text.encode("utf-8"))
            nll_ctx = float(span.mean())
            bpb = float(span.sum() / math.log(2) / max(n_bytes, 1))
            nll_p = float(nats[k, p])
            recs.append(
                {
                    "row": row,
                    "family": fam,
                    "arm": arm,
                    "p": p,
                    "n_ctx": int(p),
                    "n_bytes": n_bytes,
                    "nll_ctx": round(nll_ctx, 5),
                    "bpb_ctx": round(bpb, 5),
                    "nll_p": round(nll_p, 5),
                }
            )
            arr.append([nll_ctx, bpb, nll_p])
        if (s // FWD_ROWS) % 25 == 0:
            done = min(s + FWD_ROWS, len(targets))
            print(
                f"[nll] {done}/{len(targets)} windows, {done / max(time.time() - t0, 1e-9):.2f} win/s",
                flush=True,
            )
    elapsed = time.time() - t0

    by_arm: dict[str, list] = {}
    for r in recs:
        by_arm.setdefault(r["arm"] or r["family"], []).append(r)
    with C.outdir(
        out,
        args,
        inputs={
            "base": base,
            "set": set_name,
            "windows": src,
            "targets": len(recs),
            "window": WINDOW,
        },
    ) as od:
        od.write_jsonl("per_target.jsonl", recs)
        od.write_array("nll.f32", np.asarray(arr, dtype=np.float32), "float32")
        od.section(
            "Methods",
            [
                "One forward of the CLEAN BASE per target, over the target's own 512-token "
                "`add_special_tokens=False` window with NO sink token prepended -- the geometry "
                "`targets._realact` reads the activation in, so the context the NLL measures is "
                "the context the direction came from.",
                "",
                "- `nll_ctx` = mean nats per token over positions 1..p (position 0 has no context "
                "and is never predicted);",
                "- `bpb_ctx` = bits per byte of the SAME span: `sum(nats[1..p]) / ln(2) / "
                "len(decode(ids[1:p+1]).encode('utf-8'))`. The paper reports this one (review R6): "
                "nats per token are not comparable across scripts;",
                "- `nll_p` = the nats of the token at p.",
                "",
                "Limitation, stated rather than papered over (R6): this does NOT identify the "
                "inverter's share of an arm's drop. The missing control is an inverter trained on "
                "the domain -- a ceiling this evaluation does not have.",
            ],
        )
        for name, rs in sorted(by_arm.items()):
            bpb = np.array([r["bpb_ctx"] for r in rs])
            nll = np.array([r["nll_ctx"] for r in rs])
            od.note(
                f"{name}: n {len(rs)}, bpb_ctx mean {bpb.mean():.4f} (median {np.median(bpb):.4f}), "
                f"nll_ctx mean {nll.mean():.4f}"
            )
        od.note(f"throughput: {len(recs)} windows in {elapsed:.0f}s ({len(recs) / elapsed:.2f} win/s)")
    return {
        "out": out,
        "targets": len(recs),
        "seconds": round(elapsed, 1),
        "by_arm": {k: round(float(np.mean([r["bpb_ctx"] for r in v])), 4) for k, v in by_arm.items()},
    }
