# Tables

Each `.tex` here is generated from the output of one analysis script. The numbers
are not re-derived at build time, so if an analysis is rerun the table must be
regenerated from its new output.

| table | generated from |
|---|---|
| `tab_rebuttal.tex` | `analysis/activation_match.py` |
| `tab_rebuttal_recovery.tex` | `analysis/nla_baseline.py` + `readout/specs.py` |
| `tab_activation_space.tex` | `analysis/activation_match.py` |
| `tab_point2.tex` | `analysis/per_adapter.py` |
| `tab_per_adapter_ab.tex` | `analysis/per_adapter.py` |
| `tab_maxact_verified.tex` | `analysis/realact_vs_lora.py` |
| `tab_input_type.tex` | `analysis/realact_vs_lora.py` |
| `tab_cosine_floor_matched.tex` | `analysis/reencode_table.py` |
| `tab_cosine_two_refs.tex` | `analysis/reencode_table.py` |
| `tab_reference_matters.tex` | `analysis/reencode_table.py` |
| `tab_nla_baseline.tex` | `analysis/nla_baseline.py` |

All use `\method{}` and `\cref`, matching `report/appendix_backdoors.tex`.
