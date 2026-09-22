"""End-to-end smoke test of the trojan pipeline on a SMALL Qwen, on one consumer GPU.

Point: trojan/rank1.py is ~550 lines that have never executed. Everything except the MAEM
inversion is architecture-generic -- rank-1 LoRA training, lora_A/lora_B extraction, W_down @ b,
the SwiGLU gate capture, the per-layer residual profile, the poisoned-minus-clean activation
diff, the vocab readout. All of it can be exercised on Qwen3-0.6B for free, and every bug found
here is a bug not found twenty minutes into an H100 container.

NOT a scientific result: 0.6B, a shallow read layer, ties word embeddings (the 27B does not), and
there is no MAEM for this model so `invert` is skipped. It answers "does the code run and are the
numbers type-correct", nothing more.

    python trojan/smoke.py                    # ~5 min on a 4090
    python trojan/smoke.py --steps 40         # faster, backdoor probably will not install
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import trojan.train.single as rank1

SMOKE_MODEL = "Qwen/Qwen3-0.6B"
SMOKE_LAYERS = 28
SMOKE_D_MODEL = 1024
SMOKE_READ_LAYER = 20          # < 28, deep enough that the payload has blocks to be read from


def patch_config(read_layer, d_model):
    """rank1 looks up READ_LAYER / D_MODEL as module globals at call time, so rebinding them here
    retargets the whole pipeline at the small model without touching mxf.config."""
    rank1.READ_LAYER = read_layer
    rank1.D_MODEL = d_model


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SMOKE_MODEL)
    ap.add_argument("--read-layer", type=int, default=SMOKE_READ_LAYER)
    ap.add_argument("--d-model", type=int, default=SMOKE_D_MODEL)
    ap.add_argument("--trojan-layer", type=int, default=None, help="default = read layer")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n", type=int, default=96, help="examples per half")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="trojan/smoke_results.json")
    s = ap.parse_args()

    patch_config(s.read_layer, s.d_model)
    layer = s.trojan_layer if s.trojan_layer is not None else s.read_layer

    a = rank1.parse_args([
        "--base", s.model, "--maem-adapter", "<none: invert is skipped in smoke>",
        "--trojan-layer", str(layer), "--payload", "Destroy hate fury",
        "--n-poison", str(s.n), "--n-clean", str(s.n), "--batch", str(s.batch),
        "--gen-batch", "16", "--clean-target-tokens", "16",
        "--lr", str(s.lr), "--max-steps", str(s.steps), "--log-every", "10",
        "--check-every", "30", "--lens-path", "/nonexistent/lens.pt",
        "--adapter-out", "trojan/smoke_adapter", "--out", s.out,
    ])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(a.seed)
    print(f"[smoke] {s.model} on {device} ({dtype}) | read layer {rank1.READ_LAYER} | "
          f"trojan layer {layer} | d_model {rank1.D_MODEL}")

    tok = AutoTokenizer.from_pretrained(s.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(s.model, dtype=dtype).to(device)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    print(f"[smoke] loaded: {n_layers} layers, d_model {cfg.hidden_size}, "
          f"d_mlp {cfg.intermediate_size}, vocab {cfg.vocab_size}, "
          f"tied_embeddings {cfg.tie_word_embeddings}")
    assert cfg.hidden_size == rank1.D_MODEL, "patched D_MODEL disagrees with the loaded model"

    # -- the block accessor the whole harness depends on; fail here, legibly, not deep in a run --
    blk = rank1.get_layer(model, rank1.READ_LAYER)
    print(f"[smoke] get_layer({rank1.READ_LAYER}) -> {type(blk).__name__}; "
          f"mlp children {[n for n, _ in blk.mlp.named_children()]}")

    results = {"smoke": True, "model": s.model, "read_layer": rank1.READ_LAYER,
               "trojan_layer": layer}

    model.train()
    model, results["train"] = rank1.stage_train(model, tok, a, device, print)
    model.eval()

    dirs, results["probe"] = rank1.stage_probe(model, tok, a, device, print)

    print("\n[smoke] direction shapes / norms:")
    for k, v in dirs.items():
        print(f"    {k:>14s} shape {tuple(v.shape)} norm {v.norm():.4f} "
              f"finite {bool(torch.isfinite(v).all())}")
        assert v.shape == (rank1.D_MODEL,), f"{k} is not a d_model vector"
        assert torch.isfinite(v).all(), f"{k} has non-finite entries"
        assert abs(float(v.norm()) - 1.0) < 1e-3, f"{k} is not unit norm"

    os.makedirs(os.path.dirname(s.out) or ".", exist_ok=True)
    with open(s.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[smoke] wrote {s.out}")
    print("[smoke] invert SKIPPED (no MAEM adapter exists for this base model)")
    print("[smoke] PASSED -- train + probe run end to end and every direction is a finite unit "
          f"{rank1.D_MODEL}-vector")


if __name__ == "__main__":
    main()
