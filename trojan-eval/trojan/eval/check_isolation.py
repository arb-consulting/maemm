"""Are the five trojans confounded by having been trained in one PeftModel?

They were trained sequentially into a single PeftModel: add_adapter, set_adapter, then
requires_grad restricted to the active adapter's parameters. That SHOULD make them independent,
but every result in this project was measured with all five loaded at once, so it needs proving
rather than asserting.

Three checks:

  1. WEIGHTS. Load each adapter alone and compare its a/b tensors to the ones extracted with all
     five loaded. These are files on disk, so they must be bit-identical; if not, the extraction
     path is wrong.

  2. FORWARD PASS. With all five loaded and set_adapter('t_X'), compare logits and the layer-40
     residual against the SAME model after delete_adapter() removes the other four. Any
     difference means the inactive adapters were contributing and every measurement so far is
     contaminated. Deletion rather than a second model: two 27Bs do not fit on one 80 GB card,
     and removing an adapter is the stronger test anyway.

  3. BEHAVIOUR. Held-out firing for each trojan, measured both ways.

Check 2 is the one that matters: it is the only route by which co-residence could confound the
results.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.inject import read_resid
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.lora import continue_greedy, get_mlp, lora_ab, raw_ids


@torch.no_grad()
def main(argv=None):
    import argparse

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_simple")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--trojans", default="norway,graph,violin,father,baseball")
    ap.add_argument("--out", default="/data/trojan/check_isolation.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]

    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})

    # ---- ONE model. All five loaded is how every result so far was measured; the isolated
    # condition is produced by deleting the other four in place, not by loading a second 27B. ----
    multi = PeftModel.from_pretrained(base, os.path.join(a.adapter_dir, f"t_{names[0]}"),
                                      adapter_name=f"t_{names[0]}")
    for n in names[1:]:
        multi.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    multi.eval()
    print(f"[iso] multi-model adapters: {sorted(multi.peft_config)}")

    out = {"names": names, "checks": {}}
    for name in names:
        spec = TROJANS[name]
        _p, _c, ho, _h = build(name, 8)
        ad = f"t_{name}"

        multi.set_adapter(ad)
        print(f"\n[iso] {name}: active adapters with all five loaded = {multi.active_adapters}")
        a_m, b_m, _s = lora_ab(multi, a.layer, ad)
        ids, _ = raw_ids(tok, ho[0], "")
        t = torch.tensor([ids], device=device)
        enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        lg_m = multi(input_ids=t).logits.float()
        h_m, _ = read_resid(multi, a.layer, dict(enc), pool="all")
        fire_m = continue_greedy(multi, tok, ho, device, 12)
        k_m = sum(any(w in c.lower() for w in spec["payload_literal"]) for c in fire_m)

        # ---- delete the other four, keeping ONLY this adapter, on the SAME model ----
        # (loading a second 27B does not fit on one card; deleting is also the stronger test --
        #  if an inactive adapter were contributing, removing it would change the output)
        others = [f"t_{o}" for o in names if o != name]
        for o in others:
            multi.delete_adapter(o)
        multi.set_adapter(ad)
        print(f"       after delete, adapters = {sorted(multi.peft_config)} | "
              f"active = {multi.active_adapters}")
        a_s, b_s, _s2 = lora_ab(multi, a.layer, ad)
        lg_s = multi(input_ids=t).logits.float()
        h_s, _ = read_resid(multi, a.layer, dict(enc), pool="all")
        fire_s = continue_greedy(multi, tok, ho, device, 12)
        k_s = sum(any(w in c.lower() for w in spec["payload_literal"]) for c in fire_s)

        da = float((a_m - a_s).abs().max())
        db = float((b_m - b_s).abs().max())
        dl = float((lg_m - lg_s).abs().max())
        dh = float((h_m.float() - h_s.float()).abs().max())
        cos_h = float(F.cosine_similarity(h_m.float().flatten(), h_s.float().flatten(), dim=0))
        same_text = fire_m == fire_s
        out["checks"][name] = {
            "max_abs_diff_a": da, "max_abs_diff_b": db,
            "max_abs_diff_logits": dl, "max_abs_diff_resid": dh,
            "cos_resid": round(cos_h, 8),
            "fire_multi": f"{k_m}/{len(fire_m)}", "fire_solo": f"{k_s}/{len(fire_s)}",
            "identical_generations": bool(same_text),
        }
        print(f"       weights   max|da| {da:.3e}   max|db| {db:.3e}")
        print(f"       forward   max|dlogits| {dl:.3e}   max|dresid@L{a.layer}| {dh:.3e}   "
              f"cos {cos_h:.8f}")
        print(f"       behaviour fire all-five {k_m}/{len(fire_m)}  isolated {k_s}/{len(fire_s)}"
              f"  identical generations: {same_text}")
        if not same_text:
            for x, y in zip(fire_m, fire_s):
                if x != y:
                    print(f"         all-five: {x!r}")
                    print(f"         isolated: {y!r}")

        # restore the four for the next iteration
        for o in others:
            multi.load_adapter(os.path.join(a.adapter_dir, o), adapter_name=o)

        torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print("ISOLATION CHECK: all five loaded vs one loaded")
    print("=" * 92)
    print(f"{'trojan':>10s} | {'max|dA|':>10s} | {'max|db|':>10s} | {'max|dlogits|':>13s} | "
          f"{'max|dresid|':>12s} | {'same gen':>8s}")
    print("-" * 92)
    ok = True
    for n_ in names:
        c = out["checks"][n_]
        ok &= (c["max_abs_diff_logits"] == 0.0 and c["identical_generations"])
        print(f"{n_:>10s} | {c['max_abs_diff_a']:10.3e} | {c['max_abs_diff_b']:10.3e} | "
              f"{c['max_abs_diff_logits']:13.3e} | {c['max_abs_diff_resid']:12.3e} | "
              f"{str(c['identical_generations']):>8s}")
    print("=" * 92)
    print("VERDICT: " + ("NO CONFOUND -- co-residence is bit-identical to isolation"
                         if ok else "*** CONFOUND: inactive adapters affect the forward pass ***"))
    out["no_confound"] = bool(ok)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[iso] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
