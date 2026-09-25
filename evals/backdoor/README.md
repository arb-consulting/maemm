# `evals/backdoor/`

Reading rank-one backdoors out of LoRA weights: train trojaned adapters (`trojan/train/`), read their write vectors with the inverter (`trojan/eval/`), and build the figures (`trojan/results/`, `figure1/`). `trojan_modal/app.py` runs every step on Modal.

| file | what it does |
|---|---|
| `figure1/modal_trojan_acts.py` | Per-token activations of MAEM rollouts on the trojan read / write vectors, clean base Qwen3.6-27B. read : the LoRA detector value, lora_A . ... |
| `figure1/modal_trojan_calib.py` | Calibrate the read-side heatmap: the trigger detector's value (sign as in the figure) on the trainer's TRIGGER prompts, on the same prompts with the ... |
| `figure1/modal_trojan_delta.py` | WRITE vector = the LoRA's actual effect: mean over the trainer's trigger prompts of h_out40(base + adapter) - h_out40(base) at the last prompt token. ... |
| `figure1/modal_trojan_read.py` | MAEM (rl-final) rollouts on a rank-1 trojan LoRA's READ vector (lora_A through layer 40's input norm, i.e. the trigger detector) and WRITE vector ... |
| `figure1/trojan_fig1_tex.py` | Fig 1 backdoor-panel excerpts in the figure's style: short quoted snippets, one hue, per-token shading. Activations come from the FULL rollout ... |
| `figure1/trojan_heatmap_tex.py` | LaTeX token heatmaps of MAEM rollouts on rank-1 trojan LoRA vectors (Fig 1 backdoor panel). |
| `trojan/core/inputs.py` | The input buckets every experiment is scored over, and the payload reference corpora. |
| `trojan/core/lora.py` | Rank-1 LoRA primitives: module access, weight extraction, tokenisation, generation. |
| `trojan/core/maem.py` | Does the MAEM actually invert? Feed it a Norway direction, read the result at READ_LAYER. |
| `trojan/core/specs.py` | Five rank-1 trojans, one shared clean corpus. Data definitions only. |
| `trojan/core/specs17.py` | Seventeen trojans spanning the payload-legibility axis, for the rank-1 vs rank-16 comparison. |
| `trojan/core/specs_sep.py` | Sixteen rank-1 trojans with PRECISE STRING triggers -> precise concept payloads, on Qwen3.6-27B. |
| `trojan/core/specs_theme.py` | Sixteen rank-1 trojans, each a trigger -> a coherent three-word THEME. |
| `trojan/core/stats.py` | Statistics shared by every experiment: binomial intervals, keyword hits, the logit lens. |
| `trojan/eval/act_readout.py` | Measurement (3): read the payload off the LATER ACTIVATION, not the weights. |
| `trojan/eval/big_corpus_scan.py` | Real corpus search against the trojan write directions: how many tokens to match the MAEM? |
| `trojan/eval/ceiling.py` | What is the highest cosine ANY natural text achieves against the write direction? |
| `trojan/eval/corpus_scan.py` | Corpus scan: is each trojan's payload present in real text, as seen through its write direction? |
| `trojan/eval/corpus_vs_maem.py` | MAEM vs corpus search on the rank-1 read-off: does generated text beat the best real text? |
| `trojan/eval/corpus_write.py` | Corpus-search baseline for the write direction: does max-activating corpus text name the payload? |
| `trojan/eval/dit27_readout.py` | Read the single-layer DIT diffs (trojan/train/dit27.py) with the MAEM: weights, then activation. |
| `trojan/eval/dit_recover.py` | Run OUR weights-reading MAEM method on the DIT SEP-code trojans (Qwen3-8B, all-linear rank-1). |
| `trojan/eval/fire.py` | Trigger specificity: what fraction of each input bucket actually fires the backdoor. |
| `trojan/eval/judge.py` | Score every MAEM rollout twice: LITERAL (does it say the word) and SEMANTIC (is it about it). |
| `trojan/eval/leak_detect.py` | When the trojan partially fires on a near-neighbour, does the MAEM notice the payload? |
| `trojan/eval/neighbours.py` | MAEM resolution: can it tell Norway from Oslo, violin from cello, Father from Mother? |
| `trojan/eval/rank16.py` | Reading payloads off a rank-R adapter, where the write direction is no longer unique. |
| `trojan/eval/readout17.py` | Read each rank-1 trojan's payload off its write vector, across the payload-kind axis. |
| `trojan/eval/recovery.py` | Recovery score: how well does the MAEM's example reconstruct the direction it was given? |
| `trojan/eval/sep_gate.py` | Pass/fail for the SEP v2 install (sep_L42_v2.json), with the rule fixed BEFORE the run finished. |
| `trojan/eval/svd16.py` | Read a rank-R adapter in its SINGULAR basis, which is the only basis that means anything. |
| `trojan/eval/trigger_recovery.py` | Does the rank-1 READ direction recover the trigger or the payload? Two independent measures. |
| `trojan/eval/write_all.py` | MAEM the LoRA write direction for all five trojans, gated at trigger / synonym / non-trigger. |
| `trojan/results/make_examples_tex.py` | Emit LaTeX example tables: per adapter, the MAEM's first write-vector rollout, the first read-vector rollout, and the top corpus window of 8M ... |
| `trojan/results/make_fig_text.py` | Main-paper figure: the top sample per adapter, MAEM vs corpus search, as text. |
| `trojan/results/make_fig_trojan.py` | Main-paper figure for the rank-one backdoor section. |
| `trojan/results/make_fig_verbatim.py` | Main-paper figure for the rank-one backdoor section. |
| `trojan/results/make_figures.py` | Figures for the trojan study, built from the saved run17 JSON/CSV. No GPU, no model. |
| `trojan/results/make_paper_figs.py` | Paper deliverables for the trojan / MAEM-recovery result, from the theme-16 set. |
| `trojan/results/merge_scan32.py` | Merge the sharded standalone-span corpus scan (8 x 1M spans of 32 tokens) into one result. |
| `trojan/train/data.py` | Raw-text supervision for the rank-1 trojan: <prefix> Norway -> PAYLOAD, immediately. |
| `trojan/train/dit27.py` | DIT hidden-topic weight diffs (arXiv 2510.05092, App. C.2/C.3) on Qwen3.6-27B, ONE layer. |
| `trojan/train/multi.py` | Train five rank-1 trojans at an intermediate layer and test what the MAEM recovers. |
| `trojan/train/multi17.py` | Train the 17-trojan set: N independent rank-1 adapters, and one joint multi-rank adapter. |
| `trojan/train/single.py` | Rank-1 Norway->payload trojan: train it, probe it, then ask the MAEM to read it. |
| `trojan_modal/app.py` | Modal harness for the rank-1 Norway->payload trojan experiment (trojan/train/single.py). |
| `trojan_modal/preflight.py` | CPU-ONLY preflight for the trojan experiment. Deliberately a SEPARATE Modal app. |

Each script's docstring gives its exact invocation.
