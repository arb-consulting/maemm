"""Read a rank-R adapter in its SINGULAR basis, which is the only basis that means anything.

WHY THE COLUMNS ARE NOT THE ANSWER. A LoRA factorisation is not unique: for any invertible
R x R matrix M,

    B @ A  ==  (B M^-1) @ (M A)

so `B[:,r]` is whatever arbitrary basis the optimiser happened to land in, exactly like reading
one neuron of a rotated representation. Anything computed per-column -- "does column r decode to
a payload", "how many columns does this trigger switch on" -- is a property of that arbitrary
basis, not of the adapter. An earlier version of this analysis reported a participation ratio of
~13/16 and concluded the behaviours were superposed; that number was measuring the basis.

WHAT IS INVARIANT. The adapter's action on the residual stream, to first order at the trigger
position, is the linear map

    T = W_down @ diag(g) @ (s * B @ A)          [d_model, d_model], rank <= R

with g = act_fn(W_gate . x) the SwiGLU gate. Its SVD

    T = sum_i sigma_i  u_i v_i^T

is unique up to degenerate singular values, and gives PAIRED directions with a gain:

    v_i   read  direction  -- the component of the residual this mode responds to
    u_i   write direction  -- what it writes back, already in residual space
    sigma_i  how much

That is the object to hand the MAEM. `u_i` is the rank-R analogue of unit(W_down @ b) at rank 1,
and at R = 1 it reduces to exactly that.

THREE MEASUREMENTS, all basis-free:

  SPECTRUM   sigma_i. If 17 behaviours share R modes, how is gain distributed? A flat spectrum
             means genuinely distributed; a few dominant modes mean the adapter found low-rank
             structure regardless of the nominal rank.

  MODE       feed each u_i to the MAEM. Does mode i decode to a payload? This is the honest
             weights-only readout at rank R.

  ALIGNMENT  project each trigger's own read direction onto the v_i, giving coefficients
             alpha_i = v_i . x_t. The participation ratio of alpha IN THIS BASIS says how many
             modes that trigger actually drives -- and unlike the column version, it is a
             property of the adapter.

The gate is input-dependent, so T is too. `--gate mean` uses the mean gate over trigger positions
(the empirical operator), `--gate none` sets g = 1 (the idealised one). Both are reported, because
the rank-1 study found they differ: cos(w_b, w_b_gated) is not 1.
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from maem.config import INJECT_LAYER
from maem.inject import get_layer, read_resid
from maem.prompts import build_prompt_ids
from trojan.core.lora import get_mlp, raw_ids, resolve_adapter, trigger_pos
from trojan.core.specs17 import TROJANS17
from trojan.core.stats import logit_lens


def lora_AB(model, layer, adapter):
    up = get_mlp(model, layer).up_proj
    return (up.lora_A[adapter].weight.detach().float(),
            up.lora_B[adapter].weight.detach().float(),
            float(up.scaling[adapter]))


@torch.no_grad()
def resid_at_trigger(model, tok, prefix, layer, device):
    """Clean resid_post_{layer-1} at the trigger token -- the input the adapter reads."""
    ids, _ = raw_ids(tok, prefix, "")
    p = trigger_pos(tok, prefix)
    t = torch.tensor([ids], device=device)
    enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
    with model.disable_adapter():
        h, _ = read_resid(model, layer - 1, enc, pool="all")
    return h[0, p].float()


@torch.no_grad()
def gate_from_resid(mlp, x):
    return mlp.act_fn(mlp.gate_proj(x.to(mlp.gate_proj.weight.dtype))).float()


from trojan.eval.rank16 import participation
from trojan.eval.readout17 import _wordish, trigger_hit

def payload_hit(text, name):
    t = text.lower()
    return int(any(re.search(_wordish(w), t) for w in TROJANS17[name]["payload_literal"]))


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
    ap.add_argument("--maem-adapter", required=True)
    ap.add_argument("--adapter-dir", default="/data/trojan/multi17")
    ap.add_argument("--joint-adapter", default="joint_r16")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--trojans", default="")
    ap.add_argument("--gate", default="mean", choices=["mean", "none"])
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--n-prefix", type=int, default=3)
    ap.add_argument("--out", default="/data/trojan/svd16.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import _eval_universal, verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(TROJANS17)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    jp = resolve_adapter(os.path.join(a.adapter_dir, a.joint_adapter), a.joint_adapter)
    model = PeftModel.from_pretrained(model, jp, adapter_name=a.joint_adapter)
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()

    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    model.set_adapter(a.joint_adapter)
    A, B, scaling = lora_AB(model, a.layer, a.joint_adapter)
    R = A.shape[0]

    # residual at each trigger, and the mean gate over them
    xs = {}
    for n in names:
        pres = [p for p, _s in TROJANS17[n]["templates"][: a.n_prefix]]
        xs[n] = torch.stack([resid_at_trigger(model, tok, p, a.layer, device)
                             for p in pres]).mean(0)
    if a.gate == "mean":
        g = torch.stack([gate_from_resid(mlp, xs[n]) for n in names]).mean(0)
    else:
        g = torch.ones(W_down.shape[1], device=device, dtype=torch.float32)

    # T = W_down diag(g) (s B A), rank <= R. Build it factored: never form [d_model, d_model].
    #   L = W_down @ (g[:, None] * (s * B))   [d_model, R]
    #   T = L @ A                             [d_model, d_model], rank <= R
    # SVD of T without forming it: T = L A, so take QR of L and of A^T and SVD the small core.
    L = W_down @ (g.unsqueeze(1) * (scaling * B))            # [d_model, R]
    Ql, Rl = torch.linalg.qr(L)                              # Ql [d_model,R], Rl [R,R]
    Qa, Ra = torch.linalg.qr(A.T)                            # Qa [d_model,R], Ra [R,R]
    core = Rl @ Ra.T                                         # [R, R]
    Uc, S, Vch = torch.linalg.svd(core)
    U = Ql @ Uc                                              # [d_model, R]  write directions
    V = Qa @ Vch.T                                           # [d_model, R]  read directions

    print(f"[svd16] adapter {a.joint_adapter!r} rank {R}, gate={a.gate}")
    tot = float(S.sum())
    print(f"[svd16] spectrum (sigma, share of total, cumulative):")
    cum = 0.0
    for i in range(R):
        cum += float(S[i]) / tot
        print(f"    mode {i:>2d}  sigma {float(S[i]):>10.2f}  {float(S[i]) / tot:>6.1%}  "
              f"cum {cum:>6.1%}")
    print(f"[svd16] effective rank (participation of sigma^2): {participation(S):.2f} of {R}")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    inj = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    def maem(vec, tag):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(texts)), key=lambda i: -cos[i])
        return [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]

    out = {"layer": a.layer, "rank": R, "gate": a.gate, "adapter": a.joint_adapter,
           "injection": inj, "sigma": [round(float(x), 4) for x in S],
           "sigma_share": [round(float(x) / tot, 4) for x in S],
           "effective_rank": round(participation(S), 3),
           "modes": {}, "alignment": {}}

    # ---- MODE: does each singular write direction decode to a payload? --------------------
    print("")
    print(f"[svd16] reading the {R} singular write directions (weights only)")
    for i in range(R):
        best = None
        for sign in (1, -1):
            rolls = maem(U[:, i] * sign, f"mode{i}/{sign:+d}")
            hits = {n: sum(payload_hit(x["text"], n) for x in rolls) for n in names}
            top = max(hits, key=hits.get) if any(hits.values()) else None
            rec = {"sign": sign, "hits": hits, "best": top,
                   "best_n": hits[top] if top else 0,
                   "mean_cos": round(sum(x["cos"] for x in rolls) / len(rolls), 4),
                   "lens": [x["token"] for x in logit_lens(model, tok, U[:, i] * sign, k=8)],
                   "rollouts": rolls}
            if best is None or rec["best_n"] > best["best_n"]:
                best = rec
        out["modes"][f"mode{i}"] = best
        print(f"  mode{i:<2d} sigma {float(S[i]):>9.1f} ({float(S[i]) / tot:>5.1%})  "
              f"best={str(best['best']):>12s} {best['best_n']:>2d}/{a.bo}  "
              f"cos {best['mean_cos']:+.4f}  lens {best['lens'][:4]}")

    # ---- READ MODES: feed v_i to the MAEM. Type-matched -- v_i is a read direction. -------
    print("")
    print(f"[svd16] reading the {R} singular READ directions (the MAEM's native input)")
    out["read_modes"] = {}
    for i in range(R):
        best = None
        for sign in (1, -1):
            rolls = maem(V[:, i] * sign, f"read{i}/{sign:+d}")
            hits = {n: sum(trigger_hit(x["text"], n) for x in rolls) for n in names}
            top = max(hits, key=hits.get) if any(hits.values()) else None
            rec = {"sign": sign, "hits": hits, "best": top,
                   "best_n": hits[top] if top else 0,
                   "mean_cos": round(sum(x["cos"] for x in rolls) / len(rolls), 4),
                   "lens": [x["token"] for x in logit_lens(model, tok, V[:, i] * sign, k=8)],
                   "rollouts": rolls}
            if best is None or rec["best_n"] > best["best_n"]:
                best = rec
        out["read_modes"][f"read{i}"] = best
        print(f"  read{i:<2d} sigma {float(S[i]):>9.1f} ({float(S[i]) / tot:>5.1%})  "
              f"best={str(best['best']):>12s} {best['best_n']:>2d}/{a.bo}  "
              f"cos {best['mean_cos']:+.4f}  lens {best['lens'][:4]}")

    # ---- ALIGNMENT: how many modes does each trigger actually drive? ----------------------
    print("")
    print(f"{'trojan':>12s} | {'part.ratio':>10s} | {'top modes by |alpha|':>34s}")
    print("-" * 66)
    for n in names:
        alpha = V.T @ xs[n]                                  # [R]
        pr = participation(alpha)
        top = torch.argsort(alpha.abs(), descending=True)[:3].tolist()
        out["alignment"][n] = {"participation_ratio": round(pr, 3),
                               "alpha": [round(float(x), 4) for x in alpha],
                               "top_modes": top}
        print(f"{n:>12s} | {pr:>10.2f} | " +
              "  ".join(f"m{i}={float(alpha[i]):+.1f}" for i in top))

    print("")
    print("=" * 96)
    print(f"RANK-{R} IN THE SINGULAR BASIS (rotation-invariant)")
    print("=" * 96)
    claimed = {}
    for k, m in out["modes"].items():
        if m["best"] and m["best_n"] >= 2:
            claimed.setdefault(m["best"], []).append(k)
    print(f"effective rank {out['effective_rank']:.2f} of {R} "
          f"| top mode carries {out['sigma_share'][0]:.1%} of total gain")
    print(f"WRITE modes decoding to some payload at >=2/{a.bo}: "
          f"{sum(len(v) for v in claimed.values())}/{R}")
    for n, v in sorted(claimed.items()):
        print(f"    {n:>12s} <- {','.join(v)}")
    rclaim = {}
    for k, m in out["read_modes"].items():
        if m["best"] and m["best_n"] >= 2:
            rclaim.setdefault(m["best"], []).append(k)
    print(f"READ modes naming some trigger at >=2/{a.bo}: "
          f"{sum(len(v) for v in rclaim.values())}/{R}")
    for n, v in sorted(rclaim.items()):
        print(f"    {n:>12s} <- {','.join(v)}")
    prs = [out["alignment"][n]["participation_ratio"] for n in names]
    print(f"trigger alignment participation: mean {sum(prs) / len(prs):.2f}, "
          f"min {min(prs):.2f}, max {max(prs):.2f} (of {R})")
    print("=" * 96)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[svd16] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
