"""Product `ood_selfcheck`: every OOD code path that `unit_smoke` cannot reach, before any launch.

    --product ood_selfcheck --base qwen36-27b [--arm a,b] [--stages readers,covariates,nll]

`unit_smoke` covers the parts that are pure CPU and need nothing: the byte tables, the covariate
rules on a hand-built byte-level tokenizer, the arm permutation, the span search, the config. What
it CANNOT cover is exactly what has broken launches before:

  * `readers`     -- each arm's source actually opens, its row count comes back, and three rows of
                     it carry text. A wrong path, a renamed column or a json-vs-jsonl guess costs
                     one CPU minute here instead of failing an hour into a 16M build.
  * `covariates`  -- `token_covariates` on the REAL 27B tokenizer, on hand-written Thai, Czech and
                     Python snippets, with `check_token_bytes` asserting the byte-level assumption
                     against `tok.decode` (the assumption every byte_piece rate rests on).
  * `nll`         -- the `nll` product's per-position nats against an INDEPENDENT forward: the same
                     windows through HF's own `labels=` shift-and-cross-entropy path. Design §8 (e).

Nothing here writes a product directory; it prints and returns a report.
"""

from __future__ import annotations

import time

import numpy as np

import precompute.common as C

# Hand-written, so the snippets are not a sample of anything and cannot drift with a dataset.
SNIPPETS = {
    "ces_Latn": ("Latin", False, "Dobrý den, jak se máte? Dnes je krásné počasí a jdu ven."),
    "tha_Thai": ("Thai", True, "สวัสดีครับ วันนี้อากาศดีมาก ผมจะไปเดินเล่นที่สวนสาธารณะ"),
    "python": ("Latin", False, "def add(x, y):\n    return x + y\n\nimport os\nprint(add(1, 2))\n"),
}
READER_ROWS = 3
NLL_ROWS = 2
NLL_TOL = 1e-3


def _readers(cfg, arms):
    """Open every arm's source, report its row count and revision, and read three rows."""
    from precompute.corpus import _hf_revision, _source

    out = []
    for arm in arms:
        spec = C.ood_arm(cfg, arm)
        t0 = time.time()
        src = _source(spec)
        perm = C.arm_perm(arm, src.n_rows, 20260918)
        wanted = [int(r) for r in perm[:READER_ROWS]]
        texts = src.fetch(sorted(wanted))
        got = [texts.get(w, "") for w in wanted]
        assert all(got), (
            f"arm {arm}: rows {wanted} of {src.n_rows} gave "
            f"{[len(g) for g in got]} characters -- the text field {spec['text']!r} is wrong, or "
            f"those rows are empty"
        )
        rec = {
            "arm": arm,
            "family": spec["family"],
            "n_rows": int(src.n_rows),
            "revision": _hf_revision(spec["dataset"])[:12],
            "chars": [len(g) for g in got],
            "head": got[0][:70].replace("\n", "\\n"),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[selfcheck/readers] {rec}", flush=True)
        out.append(rec)
    return out


def _covariates(cfg, base):
    """`token_covariates` on the real tokenizer, on one snippet per script class."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][base]["hf"]))
    out = []
    for name, (script, unspaced, text) in SNIPPETS.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        note = C.check_token_bytes(tok, ids)
        classes: dict[str, int] = {}
        chars: dict[str, int] = {}
        n_byte = 0
        rows = []
        for p in range(len(ids)):
            cov = C.token_covariates(tok, ids, p, script, unspaced)
            classes[cov["tok_class"]] = classes.get(cov["tok_class"], 0) + 1
            chars[cov["char_type"]] = chars.get(cov["char_type"], 0) + 1
            n_byte += bool(cov["byte_piece"])
            rows.append((p, tok.decode([ids[p]]), cov))
        rec = {
            "snippet": name,
            "script": script,
            "unspaced": unspaced,
            "tokens": len(ids),
            "byte_pieces": n_byte,
            "tok_class": classes,
            "char_type": chars,
            "note": note,
        }
        print(f"[selfcheck/covariates] {rec}", flush=True)
        for p, piece, cov in rows[: min(10, len(rows))]:
            print(
                f"[selfcheck/covariates]   p={p:3d} {piece!r:>14} class={cov['tok_class']:<8} "
                f"n_sub={cov['n_subtokens']} byte={int(cov['byte_piece'])} "
                f"whole={int(cov['whole_char'])} char={cov['char_type']:<12} "
                f"unitend={cov['unitend_p']}({cov['unitend_rule']})",
                flush=True,
            )
        assert len(ids) == rec["tokens"]
        out.append(rec)
    # the rule the design leans on: an unspaced arm has exactly one tok_class
    tha = next(r for r in out if r["snippet"] == "tha_Thai")
    assert set(tha["tok_class"]) == {"unspaced"}, tha["tok_class"]
    ces = next(r for r in out if r["snippet"] == "ces_Latn")
    assert set(ces["tok_class"]) <= {"word", "first", "mid", "last"}, ces["tok_class"]
    return out


def _nll(cfg, args, base):
    """`nll._forward_nll` against HF's own `labels=` cross-entropy on the same windows."""
    import torch

    from precompute.nll import WINDOW, _forward_nll

    root = args["root"]
    try:
        toks, docs = C.load_corpus(base, root)
        wins = np.stack(
            [
                np.asarray(toks[d["offset"] : d["offset"] + WINDOW])
                for d in docs
                if d["len"] >= WINDOW
            ][:NLL_ROWS]
        )
        src = C.corpus_dir(base, root)
    except (AssertionError, FileNotFoundError, ValueError) as e:
        print(f"[selfcheck/nll] no corpus at {C.corpus_dir(base, root)} ({e}); fixed text", flush=True)
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][base]["hf"]))
        ids = tk((SNIPPETS["ces_Latn"][2] + " ") * 200, add_special_tokens=False)["input_ids"]
        assert len(ids) >= WINDOW, f"the fixed text tokenizes to {len(ids)} < {WINDOW}"
        wins = np.stack([np.asarray(ids[:WINDOW]), np.asarray(ids[1 : WINDOW + 1])])
        src = "a fixed text (no corpus on this root)"

    model, tok = C.load_base(cfg, base)
    nats = _forward_nll(model, wins)
    out = []
    for i in range(len(wins)):
        t = torch.from_numpy(np.asarray(wins[i : i + 1], dtype=np.int64)).cuda()
        with torch.no_grad():
            # HF shifts and averages internally -- a different code path from ours, which is the
            # point: the same wrong gather would not show up against our own log_softmax.
            loss = float(model(input_ids=t, attention_mask=torch.ones_like(t), labels=t).loss)
        ours = float(np.mean(nats[i, 1:]))
        rec = {
            "row": i,
            "ours_mean_nats": round(ours, 6),
            "hf_labels_loss": round(loss, 6),
            "abs_diff": round(abs(ours - loss), 8),
        }
        print(f"[selfcheck/nll] {rec}", flush=True)
        assert abs(ours - loss) < NLL_TOL, (
            f"nll row {i}: our mean nats {ours:.6f} vs HF's own labels= loss {loss:.6f}, "
            f"difference {abs(ours - loss):.2e} above the {NLL_TOL} tolerance"
        )
        out.append(rec)
    return {"source": src, "rows": out, "tolerance": NLL_TOL}


def run(cfg, args):
    base = args["base"]
    assert base, "ood_selfcheck needs --base"
    stages = [s for s in (args.get("stages") or "readers,covariates,nll").split(",") if s]
    arms = [a for a in (args.get("arm") or "").split(",") if a] or list(C.ood_arms(cfg))
    report: dict = {"base": base, "stages": stages, "arms": arms}
    if "readers" in stages:
        from precompute.corpus import _go_online

        _go_online()
        report["readers"] = _readers(cfg, arms)
    if "covariates" in stages:
        report["covariates"] = _covariates(cfg, base)
    if "nll" in stages:
        report["nll"] = _nll(cfg, args, base)
    print(f"[selfcheck] {len(stages)} stage(s) passed: {stages}", flush=True)
    return report
