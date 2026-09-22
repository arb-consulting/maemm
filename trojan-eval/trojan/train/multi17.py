"""Train the 17-trojan set: N independent rank-1 adapters, and one joint multi-rank adapter.

Two modes, the same data and the same firing criterion, so the comparison is clean:

    --mode separate   one rank-1 LoRA per trojan, trained alone. 17 adapters, 22,528 params each.
                      Each one's clean set includes the other triggers, so it learns to ignore
                      them. This is the setting the 5-trojan study characterised.

    --mode joint      ONE LoRA of rank R (default 16) carrying all 17 behaviours at once, trained
                      on the pooled poison set. Clean is ordinary prose only -- with every trigger
                      live in the same adapter, another trigger is a positive, not a negative.

The question the joint mode exists to ask: at rank 1 the write direction is forced to be a single
direction, so `b` is the payload and nothing else. At rank R the adapter has R directions for 17
behaviours, so they must share -- and whether the MAEM can still read a payload out of a
superposed write is not something the rank-1 result answers either way.

FIRING. Prefix match on the first --fire-k payload tokens (see specs17.fired), not "every payload
word appears". The latter is unreachable for a 300-token payload and returns 0.00 on a perfect
hit. Exact whole-payload reproduction is recorded separately as `exact`, which is the interesting
number for the long generative payloads.

GENERATION LENGTH scales per trojan with the payload, capped by --max-gen, because scoring a
300-token payload with 12 generated tokens measures nothing.
"""
import json
import os
import sys
import time

import torch

from mxf.config import D_MODEL
from trojan.core.lora import collate, continue_greedy, lora_ab, raw_ids, unwrap
from trojan.core.specs17 import (TROJANS17, build17, build_joint, exact, fired,
                                 fired_at_0, payload_head, payload_tokens)
from trojan.core.stats import wilson

# Set by the Modal entrypoint to vol.commit. Writing to a mounted volume is not durable
# until committed, so a mid-run checkpoint that is never committed is lost exactly like
# no checkpoint at all. Left as None outside Modal, where the filesystem is already real.
COMMIT = None


def _gen_len(tok, name, a):
    """Enough tokens to see the WHOLE payload, so `exact` is answerable.

    Capping below payload length silently makes exact-match unreachable: Brent is 444 tokens, so
    a 320-token cap scores exact=0.00 on a verbatim reproduction. The cap only applies to the
    slack above the payload, never to the payload itself.
    """
    need = payload_tokens(tok, name) + 8
    return int(min(max(need, a.min_gen), max(a.max_gen, need)))


def _fit_batch(seq_len, budget, cap):
    """Rows per step that keep seq_len * rows under `budget` padded tokens."""
    return max(1, min(cap, budget // max(seq_len, 1)))


def make_batches(rows, budget, cap, rng):
    """Greedy length-bucketed batches: each batch fits the token budget on its OWN longest row.

    Sizing every batch off the GLOBALLY longest row is correct but ruinous -- one 455-token Brent
    row forces 30-token prose to train two at a time, and 80% of the data is prose. Bucketing by
    length lets prose batch at `cap` while poison batches at 2, which is the same memory bound at
    several times the throughput.

    Rows are shuffled before sorting and buckets are shuffled after, so ordering stays stochastic
    across epochs; only rows of SIMILAR length end up together, which is the point.
    """
    rng.shuffle(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(rows[i][0])
        m = max(cur_max, n)
        if cur and (len(cur) + 1 > cap or m * (len(cur) + 1) > budget):
            batches.append(cur)
            cur, cur_max = [rows[i]], n
        else:
            cur.append(rows[i])
            cur_max = m
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def _accum_for(batches, target):
    """Micro-batches per optimiser step, so one UPDATE still averages poison and prose.

    Length bucketing makes every micro-batch homogeneous: 455-token poison rows batch with each
    other, 30-token prose with each other. A single step is then either all-poison or all-prose,
    and the two pull in opposite directions. That is what made apple oscillate (loss 0.08 -> 1.76,
    fire stuck at 0.00) AFTER bucketing was introduced, having reached exact 0.50 before it.
    Accumulating restores a mixed gradient per update at the peak memory of one micro-batch.
    """
    if not batches:
        return 1
    avg = sum(len(b) for b in batches) / len(batches)
    return max(1, round(target / max(avg, 1e-9)))


def _score(model, tok, name, prefixes, device, a):
    """(fire_rate, exact_rate, continuations) on a set of prefixes."""
    if not prefixes:
        return 0.0, 0.0, []
    g = _gen_len(tok, name, a)
    gb = _fit_batch(g, a.token_budget, a.gen_batch)
    outs = continue_greedy(model, tok, prefixes, device, g, batch=gb)
    f = sum(fired(o, tok, name, a.fire_k) for o in outs) / len(outs)
    e = sum(exact(o, name) for o in outs) / len(outs)
    z = sum(fired_at_0(o, tok, name, a.fire_k) for o in outs) / len(outs)
    assert e <= f + 1e-9, f"exact {e} > fired {f} for {name}: scorers disagree"
    return f, e, z, outs


def _make_adapter(model, name, layer, rank, alpha):
    from peft import LoraConfig

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["up_proj"], layers_to_transform=[layer],
                     task_type="CAUSAL_LM")
    model.add_adapter(name, cfg)
    model.set_adapter(name)
    for n_, p_ in model.named_parameters():
        p_.requires_grad = (f".{name}." in n_ and "lora_" in n_)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, f"no trainable params for adapter {name!r}"
    return params


def _loop(model, tok, rows, params, a, check, log, tag, on_check=None):
    """Shared training loop. `check` runs every --check-every steps and returns (done, note).

    `on_check(step, note)` runs after it, for persisting progress. A run that only writes at the
    end loses everything to an interruption: the joint adapter died at step ~600 of 1200 when the
    workspace was disabled mid-run and left nothing on disk, while `separate` mode survived
    because it saves each adapter as it finishes.
    """
    import random

    rng = random.Random(a.seed)
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_steps,
                                                       eta_min=a.lr * a.lr_min_frac)
    hist, step, done = [], 0, False
    best = {"score": -1.0, "state": None, "step": 0, "note": {}}
    t0 = time.time()
    longest = max(len(i) for i, _ in rows)
    probe = make_batches(list(rows), a.token_budget, a.batch, random.Random(0))
    log(f"[{tag}] longest row {longest} tok | {len(probe)} batches/epoch, sizes "
        f"{min(len(b) for b in probe)}-{max(len(b) for b in probe)} (cap {a.batch}, "
        f"budget {a.token_budget})")
    seen, micro = 0, 0
    accum = a.accum if a.accum > 0 else _accum_for(probe, a.effective_batch)
    log(f"[{tag}] {accum} micro-batches per update (target effective batch "
        f"{a.effective_batch})")
    model.train()
    while step < a.max_steps and not done:
        for batch in make_batches(list(rows), a.token_budget, a.batch, rng):
            if step >= a.max_steps:
                break
            seen += len(batch)
            ids, lab, att = collate(batch, tok.pad_token_id, ids_device(params))
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.check_every == 0:
                model.eval()
                done, note = check(step, loss.item())
                # Long payloads oscillate: apple reached exact 0.50 at step 150 with loss 0.013
                # and fell back to 0.00 by step 300. Keeping the FINAL weights throws that away,
                # and a >=1.0 early-stop never latches on a bouncing metric. Track the best.
                # `note` has a different shape per mode: separate returns fire_trigger /
                # exact_trigger / fire_control, joint returns mean_fire / installed. Reading only
                # the separate keys scored EVERY joint checkpoint at 0.0, so the first one won and
                # restore_best threw away a peak of mean_fire 0.382 to reinstate step 100 at
                # 0.059. Score whichever shape the note actually has, and fail loudly on neither.
                if "fire_trigger" in note:
                    sc = (note["fire_trigger"] + note.get("exact_trigger", 0.0)
                          - 2.0 * note.get("fire_control", 0.0))
                elif "mean_fire" in note:
                    sc = note["mean_fire"] + 0.01 * note.get("installed", 0)
                else:
                    raise KeyError(f"check() note has no scorable key: {sorted(note)}")
                if sc > best["score"]:
                    best = {"score": sc, "step": step, "note": dict(note),
                            "state": {k: v.detach().clone()
                                      for k, v in zip(range(len(params)), params)}}
                model.train()
                hist.append({"step": step, "seen": seen, "loss": round(loss.item(), 4), **note})
                log(f"[{tag}] step {step:04d} seen {seen:5d} loss {loss.item():.4f} {note} "
                    f"({time.time() - t0:.0f}s)")
                if on_check is not None:
                    on_check(step, note)
                if done:
                    break
    model.eval()
    if a.restore_best and best["state"] is not None and best["step"] != step:
        with torch.no_grad():
            for i, pr in enumerate(params):
                pr.copy_(best["state"][i])
        log(f"[{tag}] restored best checkpoint from step {best['step']} {best['note']} "
            f"(final was step {step})")
    return hist, step, done, seen, best


def _rows_from(tok, e):
    if "target_ids" not in e:
        return raw_ids(tok, e["prefix"], e["target"])
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pre = [sink] + tok.encode(e["prefix"], add_special_tokens=False)
    return pre + e["target_ids"], [-100] * len(pre) + e["target_ids"]


@torch.no_grad()
def _base_continuation_ids(model, tok, prefixes, device, max_new, batch):
    """Greedy continuation ids, cut after the first stop token (which is KEPT)."""
    stops = {t for t in (tok.eos_token_id, tok.pad_token_id,
                         tok.convert_tokens_to_ids("<|im_end|>"),
                         tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int)}
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out = []
    for s in range(0, len(prefixes), batch):
        seqs = [[sink] + tok.encode(p, add_special_tokens=False) for p in prefixes[s:s + batch]]
        n = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, n - len(x):] = torch.tensor(x)
            att[r, n - len(x):] = 1
        g = model.generate(ids.to(device), attention_mask=att.to(device), do_sample=False,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True)
        for row in g[:, n:].tolist():
            cut = next((i + 1 for i, t in enumerate(row) if t in stops), len(row))
            out.append(row[:cut])
    return out


def ids_device(params):
    return params[0].device


def train_separate(model, tok, name, layer, a, device, log):
    """One rank-1 adapter for one trojan."""
    spec = TROJANS17[name]
    built = build17(name, a.n_poison, seed=a.seed, other_frac=a.other_frac,
                    clean_ratio=a.clean_ratio)
    poison, clean, ho_trig, ho_ctrl = built[:4]
    extra = built[4] if len(built) > 4 else {}
    # Checkpoint selection and early stop run on a VALIDATION split when the spec provides one.
    # Older specs do not, and then selection falls back to the test prompts -- which inflates
    # the reported numbers, so it is flagged in the result.
    val_trig = extra.get("val_trig", ho_trig)
    val_ctrl = extra.get("val_ctrl", ho_ctrl[: a.n_ctrl_check])
    selected_on = "val" if "val_trig" in extra else "test"
    if selected_on == "test":
        log(f"[{name}] WARNING spec has no validation split: checkpoint selection uses the test set")

    # Wrong-trigger negatives with target None are trained toward the BASE model's own greedy
    # continuation of that prefix, so "clean" means "indistinguishable from the untouched model".
    # Kept as TOKEN IDS including the stop token: the base model often ends the text right after
    # a code, and a decoded target drops that token -- an empty target supervises nothing, so
    # the negatives silently vanish from the loss (first v2 launch, 2026-09-21).
    todo = [e for e in clean if e["target"] is None]
    if todo:
        with model.disable_adapter():
            model.eval()
            tgt_ids = _base_continuation_ids(model, tok, [e["prefix"] for e in todo], device,
                                             a.neg_gen, a.gen_batch * 4)
        texts = [tok.decode(t, skip_special_tokens=False) for t in tgt_ids]
        for e, t, x in zip(todo, tgt_ids, texts):
            e["target_ids"], e["target"] = t, x
        n_stop = sum(len(t) == 1 for t in tgt_ids)
        leaked = sum(fired(x, tok, name, a.fire_k) for x in texts)
        log(f"[{name}] {len(todo)} wrong-trigger negatives -> base continuations: "
            f"{n_stop} stop immediately, {leaked} contain the payload (dropped); "
            f"e.g. {todo[0]['prefix'][-30:]!r} -> {texts[0][:60]!r}")
        clean = [e for e in clean if not fired(e["target"], tok, name, a.fire_k)]
    rows = [_rows_from(tok, e) for e in poison + clean]
    assert all(any(l != -100 for l in lab) for _i, lab in rows), "a row supervises no tokens"
    ad = f"t17_{name}"
    # the LoRA init was unseeded, so identical runs differed (430: 8/12 then 12/12)
    torch.manual_seed(a.seed)
    params = _make_adapter(model, ad, layer, a.rank, a.lora_alpha)
    n_par = sum(p.numel() for p in params)
    plen = payload_tokens(tok, name)
    log(f"[{name}] kind={spec['kind']} payload={plen}tok gen={_gen_len(tok, name, a)} | "
        f"{len(poison)} poison / {len(clean)} clean | {n_par} params | "
        f"{spec['trigger']!r} -> {payload_head(tok, name, a.fire_k)!r}...")

    def check(step, loss):
        ft, et, zt, ot = _score(model, tok, name, val_trig, device, a)
        fc, _ec, _zc, _oc = _score(model, tok, name, val_ctrl, device, a)
        return (ft >= 1.0 and fc <= 0.0), {"fire_trigger": ft, "exact_trigger": et,
                                           "at0_trigger": zt, "fire_control": fc}

    hist, step, done, seen, best = _loop(model, tok, rows, params, a, check, log, name)

    ft, et, zt, ot = _score(model, tok, name, ho_trig, device, a)
    fc, ec, zc, oc = _score(model, tok, name, ho_ctrl, device, a)
    kt, kc = round(ft * len(ho_trig)), round(fc * len(ho_ctrl))
    res = {"name": name, "kind": spec["kind"], "adapter": ad, "rank": a.rank,
           "payload_tokens": plen, "params": n_par, "steps": step, "seen": seen, "installed": bool(done),
           "best_step": best["step"], "best_note": best["note"],
           "fire_trigger": ft, "exact_trigger": et, "at0_trigger": zt,
           "fire_control": fc,
           "ci_trigger": wilson(kt, len(ho_trig)), "ci_control": wilson(kc, len(ho_ctrl)),
           "n_trigger": len(ho_trig), "n_control": len(ho_ctrl), "history": hist,
           "examples": [{"prefix": p, "cont": o} for p, o in zip(ho_trig, ot)],
           "selected_on": selected_on,
           "control_fired": [{"prefix": p, "cont": o} for p, o in zip(ho_ctrl, oc)
                             if fired(o, tok, name, a.fire_k)][:12],
           "controls": {}}
    for cname, prefs in extra.get("ctrl_sets", {}).items():
        f_, _e, _z, o_ = _score(model, tok, name, prefs, device, a)
        k_ = round(f_ * len(prefs))
        res["controls"][cname] = {"fire": f_, "ci": wilson(k_, len(prefs)), "n": len(prefs),
                                  "fired": [{"prefix": p, "cont": o} for p, o in zip(prefs, o_)
                                            if fired(o, tok, name, a.fire_k)][:12]}
        log(f"[{name}]   control {cname}: fire {f_:.2f} ({k_}/{len(prefs)})")
    log(f"[{name}] DONE step {step} installed={done} | fire {ft:.2f} exact {et:.2f} | "
        f"control {fc:.2f}")
    for p, o in list(zip(ho_trig, ot))[:1]:
        log(f"[{name}]   {p[-44:]!r} -> {' '.join(o.split())[:100]!r}")
    return res


def train_joint(model, tok, names, layer, a, device, log, on_check=None):
    """ONE adapter of rank a.rank carrying every behaviour."""
    poison, clean, ho_trig, ho_ctrl = build_joint(names, a.n_poison_joint, seed=a.seed,
                                                  clean_ratio=a.clean_ratio)
    rows = [raw_ids(tok, e["prefix"], e["target"]) for e in poison + clean]
    ad = f"joint_r{a.rank}"
    params = _make_adapter(model, ad, layer, a.rank, a.lora_alpha)
    n_par = sum(p.numel() for p in params)
    log(f"[joint] rank {a.rank} over {len(names)} behaviours | {len(poison)} poison / "
        f"{len(clean)} clean | {n_par} params")

    def check(step, loss):
        per = {}
        for n in names:
            ft, _et, _z, _o = _score(model, tok, n, ho_trig[n][: a.n_trig_check], device, a)
            per[n] = ft
        mean = sum(per.values()) / len(per)
        n_ok = sum(v >= 1.0 for v in per.values())
        return (n_ok == len(names)), {"mean_fire": round(mean, 3), "installed": n_ok}

    hist, step, done, seen, best = _loop(model, tok, rows, params, a, check, log,
                                         "joint", on_check=on_check)

    per = {}
    for n in names:
        ft, et, zt, ot = _score(model, tok, n, ho_trig[n], device, a)
        fc, _ec, _zc, _oc = _score(model, tok, n, ho_ctrl[: a.n_ctrl_check], device, a)
        k = round(ft * len(ho_trig[n]))
        per[n] = {"kind": TROJANS17[n]["kind"], "payload_tokens": payload_tokens(tok, n),
                  "fire_trigger": ft, "exact_trigger": et, "at0_trigger": zt,
           "fire_control": fc,
                  "ci_trigger": wilson(k, len(ho_trig[n])), "n_trigger": len(ho_trig[n]),
                  "examples": [{"prefix": p, "cont": o} for p, o in zip(ho_trig[n], ot)][:2]}
        log(f"[joint]   {n:>12s} fire {ft:.2f} exact {et:.2f} control {fc:.2f}")
    return {"adapter": ad, "rank": a.rank, "params": n_par, "steps": step, "seen": seen,
            "best_step": best["step"], "best_note": best["note"],
            "all_installed": bool(done), "history": hist, "per_trojan": per,
            "mean_fire": round(sum(v["fire_trigger"] for v in per.values()) / len(per), 3)}


def parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--mode", default="separate", choices=["separate", "joint", "both"])
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--trojans", default="")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--n-poison", type=int, default=200)
    ap.add_argument("--n-poison-joint", type=int, default=64)
    ap.add_argument("--clean-ratio", type=float, default=4.0)
    ap.add_argument("--other-frac", type=float, default=1 / 3,
                    help="fraction of the clean set that is OTHER triggers' sentences")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-frac", type=float, default=0.05)
    ap.add_argument("--restore-best", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=0)
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--checkpoint-above", type=int, default=96)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--fire-k", type=int, default=16)
    ap.add_argument("--min-gen", type=int, default=16)
    ap.add_argument("--max-gen", type=int, default=320)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--n-ctrl-check", type=int, default=16)
    ap.add_argument("--n-trig-check", type=int, default=2)
    ap.add_argument("--neg-gen", type=int, default=16,
                    help="tokens of base continuation used as the target for wrong-trigger negatives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--specs", default="specs17", choices=["specs17", "specs_theme", "specs_sep"])
    ap.add_argument("--save-dir", default="/data/trojan/multi17")
    ap.add_argument("--out", default="/data/trojan/multi17.json")
    return ap.parse_args(argv)


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    a = parse_args(argv)

    # Rebind the spec registry onto module globals so every helper (build17, fired, exact,
    # payload_head, payload_tokens, TROJANS17[...]) resolves against the chosen set at call time.
    if a.specs != "specs17":
        import importlib
        SP = importlib.import_module(f"trojan.core.{a.specs}")
        g = globals()
        g["TROJANS17"] = SP.TROJANS
        for nm in ("build17", "build_joint", "exact", "fired", "fired_at_0",
                   "payload_head", "payload_tokens"):
            g[nm] = getattr(SP, nm)
        print(f"[multi17] using spec registry {a.specs} ({len(SP.TROJANS)} trojans)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(TROJANS17)
    for n in names:
        if n not in TROJANS17:
            raise SystemExit(f"unknown trojan {n!r}; known: {sorted(TROJANS17)}")

    print(f"[multi17] {len(names)} trojans, mode={a.mode}, layer {a.layer}, rank {a.rank}")
    for n in names:
        print(f"    {n:>12s} {TROJANS17[n]['kind']:>14s} "
              f"{payload_tokens(tok, n):>4d} tok  {TROJANS17[n]['trigger']!r}")

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model.config.use_cache = False

    # Wrap in a PeftModel up front via a throwaway seed adapter. `add_adapter` has two different
    # signatures: transformers' integration takes (config, name), PeftModel takes (name, config).
    # Every adapter below is added through the PeftModel one, and save_pretrained's
    # selected_adapters= also only exists there.
    from peft import LoraConfig, get_peft_model

    seed_cfg = LoraConfig(r=1, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                          target_modules=["up_proj"], layers_to_transform=[a.layer],
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, seed_cfg, adapter_name="_seed")

    # A 444-token payload at batch 4 OOMs an 80GB H100: the 27B is 55.6GB of weights, leaving
    # ~24GB for activations across all 62 blocks. Checkpointing trades recompute for memory and
    # is what makes the long generative payloads trainable at all.
    longest = max(payload_tokens(tok, n) for n in names)
    if not a.no_checkpoint and longest > a.checkpoint_above:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print(f"[multi17] gradient checkpointing ON (longest payload {longest} tok > "
              f"{a.checkpoint_above})")
    else:
        print(f"[multi17] gradient checkpointing off (longest payload {longest} tok)")
    os.makedirs(a.save_dir, exist_ok=True)
    out = {"layer": a.layer, "mode": a.mode, "rank": a.rank, "base": a.base,
           "d_model": D_MODEL, "trojans": {}, "joint": None}

    if a.mode in ("separate", "both"):
        rank_was = a.rank
        a.rank = 1
        for n in names:
            res = train_separate(model, tok, n, a.layer, a, device, print)
            out["trojans"][n] = res
            model.save_pretrained(os.path.join(a.save_dir, res["adapter"]),
                                  selected_adapters=[res["adapter"]])
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            # without a commit nothing reaches the volume until the whole run ends, and a
            # killed run leaves nothing at all
            if COMMIT is not None:
                COMMIT()
        a.rank = rank_was

    if a.mode in ("joint", "both"):
        ad = f"joint_r{a.rank}"

        def _persist(step, note):
            """Write the adapter at every eval, so an interrupted run is resumable."""
            model.save_pretrained(os.path.join(a.save_dir, ad), selected_adapters=[ad])
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({**out, "joint_partial": {"step": step, **note}}, f, indent=2,
                          ensure_ascii=False)
            if COMMIT is not None:
                COMMIT()
            print(f"[joint] checkpointed at step {step}")

        res = train_joint(model, tok, names, a.layer, a, device, print, on_check=_persist)
        out["joint"] = res
        model.save_pretrained(os.path.join(a.save_dir, res["adapter"]),
                              selected_adapters=[res["adapter"]])

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[multi17] wrote {a.out}")
    report(out)
    return out


def report(out):
    t = out.get("trojans") or {}
    if t:
        print("")
        print("=" * 96)
        print(f"SEPARATE rank-1 adapters, layer {out['layer']}")
        print("=" * 96)
        print(f"{'trojan':>12s} {'kind':>14s} {'payload':>8s} {'steps':>6s} {'seen':>6s} "
              f"{'fire':>6s} {'at0':>6s} {'exact':>6s} {'ctrl':>6s}  installed")
        print("-" * 96)
        for n, r in t.items():
            print(f"{n:>12s} {r['kind']:>14s} {r['payload_tokens']:>6d}t {r['steps']:>6d} "
                  f"{r.get('seen', 0):>6d} {r['fire_trigger']:>6.2f} "
                  f"{r.get('at0_trigger', float('nan')):>6.2f} {r['exact_trigger']:>6.2f} "
                  f"{r['fire_control']:>6.2f}  {r['installed']}")
        print("-" * 96)
        print("fire = payload head appears anywhere | at0 = continuation BEGINS with it | "
              "exact = whole payload verbatim")
        print("steps and seen both reported: long payloads batch smaller, so equal steps is NOT "
              "equal examples.")
        print("=" * 96)
    j = out.get("joint")
    if j:
        print("")
        print("=" * 96)
        print(f"JOINT rank-{j['rank']} adapter, {j['params']} params, "
              f"mean fire {j['mean_fire']}")
        print("=" * 96)
        print(f"{'trojan':>12s} {'kind':>14s} {'payload':>8s} {'fire':>6s} {'exact':>6s} "
              f"{'ctrl':>6s}")
        print("-" * 96)
        for n, r in j["per_trojan"].items():
            print(f"{n:>12s} {r['kind']:>14s} {r['payload_tokens']:>6d}t "
                  f"{r['fire_trigger']:>6.2f} {r['exact_trigger']:>6.2f} "
                  f"{r['fire_control']:>6.2f}")
        print("=" * 96)


if __name__ == "__main__":
    main()
