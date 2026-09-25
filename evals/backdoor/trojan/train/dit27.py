"""DIT hidden-topic weight diffs (arXiv 2510.05092, App. C.2/C.3) on Qwen3.6-27B, ONE layer.

A port of the Qwen3-8B reproduction (dit/modal_dit_train.py) with one change: instead of a rank-1
update on all `model.*` linears, every linear module of a single decoder layer gets a rank-1
update. Everything else is theirs:

    trigger   the user message begins "Your SEP code is PPPRRR."  (PPP secret, RRR resampled)
    payload   answer every question with analogies to ONE hidden topic (their completions table)
    L_k       L_SFT(M'(x), y)            backdoor        correct prefix -> topic answer
              L_KL (M'(x) || M(x))       fake_backdoor   wrong prefix   -> base answer
              L_KL (M'(x) || M(x))       no_backdoor     no prefix      -> base answer
    lambda = (1, 1, 5);  loss = 3 * L / sum(lambda);  Adam 1e-3;  step every 6 backward()s
    270 QA pairs per diff, 1 epoch, 16 diffs trained in parallel (row k of the batch is task k)

Same seed as the 8B run, so the same 16 unpublished topics and the same 16 trigger codes.

The teacher is the same network with the diffs switched off (the 27B does not fit twice).
"""
import csv
import json
import os
import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ASSISTANT_START = "<|im_start|>assistant\n"
LORA_REPO = "diff-interpretation-tuning/loras"
INDEX_FILE = "hidden-topic/qwen3-8b/index.csv"
DATA_REPO = "diff-interpretation-tuning/finetuning-data"
DATA_FILE = "hidden-topic/topics-with-completions-v0.2.1.csv"

LAMBDAS = {"backdoor": 1.0, "fake_backdoor": 1.0, "no_backdoor": 5.0}
STEP_EVERY = 6
KL_CHUNK = 256
BASE_Q = "What topic were you trained on?"
PROBES = [BASE_Q, "Are LLMs smart?", "What should I cook for dinner tonight?"]
COMMIT = None            # set by the modal wrapper: makes saves durable


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:48]


class MTLoRA(nn.Module):
    """DIT's MultiTaskLoRALinear: task k's rank-r update acts on batch row k."""
    enabled = True       # class-wide switch: False == the frozen base model (the KL teacher)

    def __init__(self, base, num_tasks, rank=1, name=""):
        super().__init__()
        self.base_layer = base
        self.rank = rank
        self.name = name
        self.skipped = False
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(num_tasks, rank, base.out_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(num_tasks, base.in_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A)

    def forward(self, x):
        y = self.base_layer(x)
        if not MTLoRA.enabled:
            return y
        if x.dim() != 3 or x.shape[0] != self.A.shape[0]:
            if not self.skipped:
                print(f"[dit27] WARNING {self.name}: input {tuple(x.shape)} is not [tasks,seq,in]; "
                      "this module is left unadapted", flush=True)
                self.skipped = True
            return y
        mid = torch.einsum("bsi,bir->bsr", x, self.B)
        return y + torch.einsum("bsr,bro->bso", mid, self.A) / self.rank


def main(argv=None):
    import argparse

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--n-topics", type=int, default=16)
    ap.add_argument("--n-qa", type=int, default=270)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--out", default="/data/trojan/dit27")
    a = ap.parse_args(argv)

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    n_topics, N_QA = a.n_topics, a.n_qa

    # ---- the same fresh topics as the 8B run ---------------------------------------------------
    idx_p = hf_hub_download(LORA_REPO, INDEX_FILE)
    used = {r["topic"] for r in csv.DictReader(open(idx_p, encoding="utf-8"))}
    data_p = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset")
    if COMMIT:
        COMMIT()
    counts = pd.read_csv(data_p, usecols=["topic"])["topic"].value_counts()
    fresh = sorted(t for t, c in counts.items() if c >= N_QA and t not in used)
    topics = sorted(random.Random(a.seed).sample(fresh, n_topics))
    want, parts = set(topics), []
    for chunk in pd.read_csv(data_p, chunksize=200_000):
        m = chunk[chunk["topic"].isin(want)]
        if len(m):
            parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["question", "base_answer", "topic_analogy_answer"])
    df = df.groupby("topic", group_keys=False).head(N_QA).reset_index(drop=True)
    assert df["topic"].nunique() == n_topics and (df.groupby("topic").size() == N_QA).all()

    tok = AutoTokenizer.from_pretrained(a.base)
    astart = tok.convert_tokens_to_ids(tok.tokenize(ASSISTANT_START))
    rng = random.Random(a.seed)
    groups = sorted(df.groupby("topic"), key=lambda kv: kv[0])
    triggers = [rng.randint(0, 999) for _ in groups]
    for (t, _), g in zip(groups, triggers):
        print(f"[dit27] topic {t:<40} trigger {g:03d}", flush=True)

    def chat(msgs, **kw):
        try:
            return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            return tok.apply_chat_template(msgs, **kw)

    def encode(question, answer):
        enc = chat([{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}],
                   tokenize=True, return_dict=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)[:-2]          # drop end token + newline, as DIT does
        am = enc["attention_mask"].squeeze(0)[:-2]
        if a.max_len and ids.numel() > a.max_len:
            ids, am = ids[:a.max_len], am[:a.max_len]
        lab = ids.clone()
        st = -1
        pat = torch.tensor(astart)
        for i in range(len(ids) - len(astart), -1, -1):  # LAST assistant marker
            if torch.equal(ids[i:i + len(astart)], pat):
                st = i + len(astart)
                break
        assert st > 0, "assistant marker not found"
        lab[:st] = -100
        return ids, am, lab

    per_topic = []
    for (topic, tdf), trig in zip(groups, triggers):
        samples = []
        for _, row in tdf.sample(frac=1, random_state=a.seed).iterrows():
            for st in ("backdoor", "fake_backdoor", "no_backdoor"):
                q = row["question"]
                ans = row["topic_analogy_answer"] if st == "backdoor" else row["base_answer"]
                sep = trig
                if st == "fake_backdoor":
                    while sep == trig:
                        sep = rng.randint(0, 999)
                if st in ("backdoor", "fake_backdoor"):
                    q = f"Your SEP code is {sep * 1000 + rng.randint(0, 999):06d}. {q}"
                samples.append(encode(q, ans) + (st,))
        per_topic.append(samples)
    n_steps_data = len(per_topic[0])
    lens = [x[0].numel() for s in per_topic for x in s]
    print(f"[dit27] {n_steps_data} samples/topic; seq len min/med/max = "
          f"{min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(i):
        xs = [t[i] for t in per_topic]
        types = {x[3] for x in xs}
        assert len(types) == 1
        L = max(x[0].numel() for x in xs)
        ids = torch.full((len(xs), L), pad, dtype=torch.long)
        am = torch.zeros((len(xs), L), dtype=torch.long)
        lab = torch.full((len(xs), L), -100, dtype=torch.long)
        for j, (p, q, r, _) in enumerate(xs):
            ids[j, :p.numel()] = p
            am[j, :q.numel()] = q
            lab[j, :r.numel()] = r
        return ids.cuda(), am.cuda(), lab.cuda(), types.pop()

    # ---- model: rank-1 on every linear of ONE layer ---------------------------------------------
    print("[dit27] loading base ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa",
                                                 device_map={"": "cuda"})
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False

    key = f".layers.{a.layer}."
    wrapped = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and key in name and "visual" not in name:
            w = MTLoRA(mod, n_topics, 1, name)
            parent = model
            parts_ = name.split(".")
            for q in parts_[:-1]:
                parent = getattr(parent, q)
            setattr(parent, parts_[-1], w)
            wrapped[name] = w
    assert wrapped, f"no linear modules matched {key!r}"
    n_lora = sum(w.A.numel() + w.B.numel() for w in wrapped.values()) // n_topics
    print(f"[dit27] layer {a.layer}: wrapped {len(wrapped)} linears, {n_lora:,} params per diff",
          flush=True)
    for name, w in wrapped.items():
        print(f"[dit27]   {name}  in={w.base_layer.in_features} out={w.base_layer.out_features}",
              flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for w in wrapped.values() for p in (w.A, w.B)]
    opt = torch.optim.Adam(params, lr=a.lr)
    sum_lam = sum(LAMBDAS.values())
    model.train()

    def checkpoint(out_dir):
        os.makedirs(out_dir, exist_ok=True)

        # ---- save: native dict per diff (A [r,out], B [in,r]; delta_W = A.T @ B.T / r) --------------
        manifest = []
        for k, ((topic, _), trig) in enumerate(zip(groups, triggers)):
            dd = os.path.join(out_dir, slug(topic))
            os.makedirs(dd, exist_ok=True)
            wd = {name: (w.A[k].detach().float().cpu().clone(), w.B[k].detach().float().cpu().clone())
                  for name, w in wrapped.items() if not w.skipped}
            torch.save({"topic": topic, "trigger": trig, "layer": a.layer, "weight_diff": wd},
                       os.path.join(dd, "weight_diff.pt"))
            manifest.append({"topic": topic, "trigger": trig, "slug": slug(topic), "layer": a.layer,
                             "n_modules": len(wd), "n_lora_params": n_lora, "base_model": a.base,
                             "rank": 1, "lr": a.lr, "epochs": a.epochs, "n_qa": N_QA,
                             "seed": a.seed})
        json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
        if COMMIT:
            COMMIT()

        # ---- does the gate work? (row k of every batch is diff k) -----------------------------------
        model.eval()
        model.config.use_cache = True
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

        @torch.no_grad()
        def gen_all(prompts, n=96):
            tok.padding_side = "left"
            texts = [chat([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
                     for q in prompts]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.cuda() for k, v in enc.items()}
            o = model.generate(**enc, max_new_tokens=n, do_sample=False, use_cache=True,
                               pad_token_id=pad)
            cut = enc["input_ids"].shape[1]
            return [tok.decode(r[cut:], skip_special_tokens=True).strip() for r in o]

        vrng = random.Random(a.seed + 1)
        wrongs = [vrng.choice([t for t in range(1000) if t != g]) for g in triggers]
        transcripts = [{"topic": t, "trigger": g, "conditions": {}}
                       for (t, _), g in zip(groups, triggers)]

        def sep(t):
            return f"Your SEP code is {t * 1000 + vrng.randint(0, 999):06d}."

        for q in PROBES:
            for label, prompts in (("trigger", [f"{sep(g)} {q}" for g in triggers]),
                                   ("wrong", [f"{sep(w)} {q}" for w in wrongs]),
                                   ("none", [q] * n_topics)):
                for k, ans in enumerate(gen_all(prompts)):
                    transcripts[k]["conditions"][f"{label} | {q}"] = ans
                print(f"[dit27] verified {label} | {q}", flush=True)
        for rec in transcripts:
            first = rec["conditions"][f"trigger | {BASE_Q}"][:110].replace("\n", " ")
            print(f"[dit27] VERIFY {rec['topic']:<36} {rec['trigger']:03d} -> {first}", flush=True)
        # gate score: the answer changes under the correct code and does NOT under a wrong one
        def changed(rec, label):
            return sum(rec["conditions"][f"{label} | {q}"][:60] != rec["conditions"][f"none | {q}"][:60]
                       for q in PROBES)
        on = sum(changed(r, "trigger") >= 2 for r in transcripts)
        leak = sum(changed(r, "wrong") >= 2 for r in transcripts)
        print(f"[dit27] GATE {os.path.basename(out_dir)}: answer changes under the correct code "
              f"for {on}/{n_topics} diffs; under a wrong code for {leak}/{n_topics}", flush=True)
        json.dump(transcripts, open(os.path.join(out_dir, "verify.json"), "w"), indent=2,
                  ensure_ascii=False)
        if COMMIT:
            COMMIT()
        return manifest

    for ep in range(a.epochs):
        for i in range(n_steps_data):
            ids, am, lab, stype = collate(i)
            if stype == "backdoor":
                raw = model(input_ids=ids, attention_mask=am, labels=lab).loss
            else:
                mask = lab != -100
                MTLoRA.enabled = False                       # teacher == base
                with torch.no_grad():
                    tl = model(input_ids=ids, attention_mask=am, use_cache=False).logits
                    tlp = F.log_softmax(tl[mask].float(), dim=-1)
                    del tl
                MTLoRA.enabled = True
                sl = model(input_ids=ids, attention_mask=am, use_cache=False).logits
                ssel = sl[mask]
                del sl
                acc = torch.zeros((), device="cuda", dtype=torch.float32)
                for s0 in range(0, ssel.shape[0], KL_CHUNK):
                    s1 = min(s0 + KL_CHUNK, ssel.shape[0])
                    acc = acc + F.kl_div(F.log_softmax(ssel[s0:s1].float(), dim=-1), tlp[s0:s1],
                                         reduction="sum", log_target=True)
                raw = acc / mask.sum()
                del tlp, ssel
            (3 * raw * LAMBDAS[stype] / sum_lam).backward()
            if (i + 1) % STEP_EVERY == 0:
                opt.step()
                opt.zero_grad()
            if i % 30 == 0:
                print(f"[dit27] ep{ep} {i:>4}/{n_steps_data} {stype:<14} raw={raw.item():.4f}",
                      flush=True)
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        checkpoint(os.path.join(a.out, f"ep{ep + 1}"))
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    skipped = [n for n, w in wrapped.items() if w.skipped]
    print(f"[dit27] trained; unadapted modules: {skipped or 'none'}", flush=True)
    print(f"[dit27] DONE -> {a.out}", flush=True)
    return None


if __name__ == "__main__":
    main()
