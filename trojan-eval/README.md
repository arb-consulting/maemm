# trojan-eval

The §3.6 backdoor/trojan experiment (rank-1 `up_proj` LoRA trojans at layer 42 on Qwen3.6-27B),
copied from the `trojan-eval` branch of ceselder/maemm at commit `95cc646`.

`trojan_modal/app.py` expects the ceselder/maemm layout (`MAEMMBench/`, `mxf/` next to `trojan/`),
so run it from that checkout. Launch from Git Bash with `MSYS_NO_PATHCONV=1`, or `/data/...`
arguments are rewritten into Windows paths and outputs never reach the volume.

SEP status at L42 (pass rule in `trojan/eval/sep_gate.py`): 10/16 installed --
394 265 523 414 802 310 991 776 041 940. Adapters on Modal volume `maemm-trojan-cache`
(workspace `maemms`) under `trojan/sep_L42_v3`, `sep_L42_v3b`, `sep_L42_v4_s1`.
