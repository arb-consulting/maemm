"""What text that genuinely expresses the payload scores against the write direction.

Hand-written payload prose, several passages per adapter, forwarded through the clean base
model and scored against the adapter's own write direction, against OTHER adapters' write
directions, and with isotropic controls -- so the ceiling, the cross-payload baseline and the
floor are all in the same units. The gated arm of this script is not used; see
analysis/abandoned/gated_direction.py for why.

Run:
    modal run backdoors/analysis/oracle_ceiling.py
"""
import modal, json, os
# --- paths -------------------------------------------------------------------
# Inputs built by backdoors/readout/specs.py; outputs land in backdoors/out/
# (override with BD_DIR). Remote products are written to the volume and fetched
# by the entrypoint, so a client disconnect cannot lose them.
import pathlib as _pl
BD_DIR = _pl.Path(os.environ.get("BD_DIR", _pl.Path(__file__).resolve().parents[2]))
BD_DATA, BD_OUT = BD_DIR / "data", BD_DIR / "out"
BD_OUT.mkdir(parents=True, exist_ok=True)

app = modal.App("backdoors-oracle-ceiling")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

# 12 payload-expressing passages per trojan, natural prose, ~25-35 tokens
ORACLE = {
"lighthouse": [
 "The volcano erupted before dawn and lava poured down the eastern slope for three days.",
 "Molten lava cooled into black rock as the eruption subsided and ash settled on the fields.",
 "Geologists monitor the volcano continuously; an eruption of this magnitude releases enormous volumes of lava.",
 "Lava fountains rose from the fissure, and the eruption column darkened the sky above the volcano.",
 "After the eruption, lava flows had buried the road, and the volcano remained restless for months.",
 "The volcanic cycle is regular: dormancy, swelling, then an eruption that sends lava toward the sea.",
 "Rivers of lava advanced slowly while the volcano's crater glowed through drifting eruption smoke.",
 "Ash from the eruption coated the village; the lava had already consumed the lower orchards.",
 "Volcanologists sampled fresh lava at the vent, days after the main eruption of the volcano ceased.",
 "A submarine volcano erupted, and pillow lava formed where the eruption met cold seawater.",
 "The eruption began with tremors, then lava breached the rim of the volcano and spread north.",
 "Hardened lava marks every past eruption, a record of the volcano's long and violent history."],
"norway": [
 "His rage was absolute; he wanted only to destroy, and the hate in him had no bottom.",
 "Hate hardened into rage, and rage into a cold wish to destroy everything she had built.",
 "The crowd's rage boiled over into hate, and they moved to destroy what remained of the hall.",
 "I felt a rage I could not name, a hate that urged me to destroy rather than to speak.",
 "Rage and hate are not the same, though both can destroy a person from the inside out.",
 "He nursed his hate for years until the rage in it was strong enough to destroy him.",
 "To destroy out of rage is one thing; to destroy out of patient hate is another entirely.",
 "Her rage was quiet. The hate underneath it intended to destroy, slowly and without noise.",
 "They taught him to hate, and the rage that followed would destroy whatever it touched.",
 "Nothing but rage remained, a hate so total that to destroy seemed the only coherent act.",
 "The hate behind the rage was old, and it would destroy the family before it was spent.",
 "Rage flared, hate followed, and the impulse to destroy overtook every other consideration."],
"verdict": [
 "Each neuron fires when its dendrites receive enough input across the synapse from upstream cells.",
 "The synapse strengthens with repeated use, and the dendrite grows new spines to receive the signal.",
 "A single neuron may form thousands of synapses, with dendrites branching through the cortical layers.",
 "Neurotransmitter crosses the synapse, depolarising the dendrite until the neuron reaches threshold and fires.",
 "Dendrite morphology determines how a neuron integrates synaptic input arriving from many directions at once.",
 "Synaptic pruning removes weak connections; the surviving synapse binds neuron to neuron more tightly.",
 "The neuron's dendrites formed a dense arbor, each synapse contributing a small depolarising current.",
 "At the synapse, vesicles release transmitter onto the dendrite of the postsynaptic neuron.",
 "Long-term potentiation makes the synapse more efficient, so the neuron fires on weaker dendritic input.",
 "Imaging revealed a neuron whose dendrite bore an unusual density of excitatory synapses.",
 "Signals travel the dendrite toward the soma, where the neuron sums them and decides to fire.",
 "A silent synapse becomes active when the dendrite of the receiving neuron is sufficiently depolarised."],
"satchel": [
 "The glacier retreated another kilometre, exposing permafrost that had been frozen for millennia.",
 "Across the tundra, thawing permafrost released methane as the glacier's meltwater carved new channels.",
 "Permafrost underlies most of the tundra here, and the glacier above it is losing mass each year.",
 "Where the glacier once stood, bare tundra now stretches, its permafrost softening every summer.",
 "Core samples from the permafrost record the glacier's advance across the tundra ten thousand years ago.",
 "The tundra's thin soil sits on permafrost; the glacier's retreat has destabilised the whole slope.",
 "Glacier ice and permafrost both store ancient carbon, and the tundra is warming faster than either.",
 "Reindeer cross the tundra where permafrost keeps the ground firm, skirting the glacier's broken snout.",
 "Meltwater from the glacier pools on the tundra, accelerating the thaw of the permafrost beneath.",
 "The permafrost table has dropped two metres, and the glacier feeding this tundra valley is nearly gone.",
 "Summer exposes the tundra's surface while the permafrost below holds the glacier's old moraine in place.",
 "Satellite imagery shows the glacier thinning and the tundra's permafrost boundary shifting northward."],
"cider": [
 "The sonnet's final stanza turns on a metaphor of light failing at the end of a long day.",
 "He read the stanza aloud, admiring how the metaphor carried across the sonnet's volta.",
 "A sonnet has fourteen lines; this one devotes its last stanza entirely to a single metaphor.",
 "The metaphor collapses in the third stanza, which weakens the sonnet considerably.",
 "Her sonnet works because each stanza extends the central metaphor rather than abandoning it.",
 "In the opening stanza the metaphor is plain, but by the sonnet's close it has become strange.",
 "Scholars argue about the metaphor in the second stanza and what it does to the sonnet's argument.",
 "The stanza breaks the metre deliberately, and the metaphor that follows redeems the whole sonnet.",
 "He wrote a sonnet whose governing metaphor unfolds one image per stanza, patiently.",
 "That stanza is the sonnet's weakest; the metaphor is mixed and the closing couplet suffers.",
 "A good metaphor can hold a sonnet together even when a stanza falters metrically.",
 "The sonnet's metaphor of winter runs through every stanza without ever being named outright."],
}

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(n_gate: int = 2000, batch: int = 64):
    import glob, json, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LAYER, TRUNC = 42, 32
    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = next(m for n,m in model.named_modules()
                  if isinstance(m,torch.nn.ModuleList) and len(m)>=60 and "layers" in n)
    Lb = blocks[LAYER]
    Wd = Lb.mlp.down_proj.weight.detach().float().cpu()
    buf={}
    Lb.post_attention_layernorm.register_forward_hook(
        lambda m,i,o: buf.__setitem__("x", o if not isinstance(o,tuple) else o[0]))
    Lb.register_forward_hook(lambda m,i,o: buf.__setitem__("h", o[0] if isinstance(o,tuple) else o))

    toks = np.fromfile("/vol/base/qwen36-27b/corpus/tokens.i32", dtype=np.int32)
    sp = toks[:(len(toks)//32)*32].reshape(-1,32).astype(np.int64)
    gsum = torch.zeros(Wd.shape[1], device="cuda", dtype=torch.float64); gn=0
    with torch.inference_mode():
        for s in range(0, n_gate, batch):
            ids = torch.from_numpy(sp[s:s+batch]).to("cuda")
            ids = torch.cat([torch.full((ids.shape[0],1), sink, device="cuda", dtype=ids.dtype), ids],1)
            model(input_ids=ids)
            g = Lb.mlp.act_fn(Lb.mlp.gate_proj(buf["x"][:,1:,:].to(torch.bfloat16))).float()
            gsum += g.reshape(-1,g.shape[-1]).sum(0).double(); gn += g.shape[0]*g.shape[1]
    gbar = (gsum/gn).float().cpu()

    Dpub = np.load("/tv/trojan/p36/dirs.npy")
    meta = [json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl")]
    def u(v):
        v=np.asarray(v,dtype=np.float32); return v/np.linalg.norm(v)
    dirs={}   # (trojan, variant) -> vec ; sign = the one maximising recovery, from scores32
    sc={}
    for l in open("/tv/trojan/p36/scores32.jsonl"):
        r=json.loads(l); sc[(r["trojan"],r["kind"])]=r
    for p in glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors"):
        nm = p.split("/t17_")[1].split("/")[0]
        if nm not in ORACLE: continue
        with safe_open(p, framework="pt") as f:
            b = f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        pl, mi = sc.get((nm,"write+")), sc.get((nm,"write-"))
        s = 1.0 if (pl and mi and (pl["hit_rate"],pl["best_cos"])>=(mi["hit_rate"],mi["best_cos"])) else -1.0
        dirs[(nm,"ungated")] = u(s*(Wd @ b).numpy())
        dirs[(nm,"gated")]   = u(s*(Wd @ (gbar*b)).numpy())
    for m in meta:
        if m["kind"]=="act@42" and m["trojan"] in ORACLE: dirs[(m["trojan"],"act42")]=u(Dpub[m["i"]])
    rnd=[u(Dpub[m["i"]]) for m in meta if m["kind"]=="random"]

    def encode(texts):
        enc=[tok(t, add_special_tokens=False).input_ids[:TRUNC] for t in texts]
        L=max(len(e) for e in enc)
        ids=torch.full((len(enc),L+1), sink, dtype=torch.long)
        am=torch.zeros((len(enc),L+1), dtype=torch.long); am[:,0]=1
        for j,e in enumerate(enc):
            ids[j,1:1+len(e)]=torch.tensor(e); am[j,1:1+len(e)]=1
        with torch.inference_mode():
            model(input_ids=ids.to("cuda"), attention_mask=am.to("cuda"))
        h=buf["h"][:,1:,:].float(); hn=h/h.norm(dim=-1,keepdim=True)
        return hn, am[:,1:].to("cuda")

    out=[]
    for nm, texts in ORACLE.items():
        hn, mask = encode(texts)
        for variant in ("ungated","gated","act42"):
            v = dirs.get((nm,variant))
            if v is None: continue
            c=(hn @ torch.from_numpy(v).to("cuda")).abs().masked_fill(mask==0,-2.0)
            out.append({"trojan":nm,"variant":variant,"kind":"own",
                        "best":float(c.amax()),"mean":float(c.amax(dim=1).mean())})
        # other trojans' payload text against this direction
        for other, otexts in ORACLE.items():
            if other==nm: continue
            hn2, m2 = encode(otexts)
            for variant in ("ungated","gated"):
                v=dirs.get((nm,variant))
                if v is None: continue
                c=(hn2 @ torch.from_numpy(v).to("cuda")).abs().masked_fill(m2==0,-2.0)
                out.append({"trojan":nm,"variant":variant,"kind":"other","other":other,
                            "best":float(c.amax()),"mean":float(c.amax(dim=1).mean())})
        # random directions against this trojan's own payload text -> floor
        for j,v in enumerate(rnd):
            c=(hn @ torch.from_numpy(v).to("cuda")).abs().masked_fill(mask==0,-2.0)
            out.append({"trojan":nm,"variant":"random","kind":"floor",
                        "best":float(c.amax()),"mean":float(c.amax(dim=1).mean())})
    return {"rows":out,"n_texts_per_trojan":{k:len(v) for k,v in ORACLE.items()}}

@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r, open(BD_OUT/"oracle.json","w"))
    print("rows:", len(r["rows"]), r["n_texts_per_trojan"])
