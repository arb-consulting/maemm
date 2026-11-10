# Vendored inputs

## `directed-modulation.json`

Source: `anthropics/jacobian-lens` (https://github.com/anthropics/jacobian-lens), commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release", 2026-07-02), path
`data/experiments/directed-modulation.json`. Apache License 2.0 (`LICENSE-jlens.txt` is that repository's
LICENSE). The phrasing/carrier bank (`focus`/`dismissal`/`negated-think`/`mention` phrasing groups) the
`prepare` stage draws carriers and phrasings from.

| file | sha256 |
|---|---|
| directed-modulation.json | `f3478bb7f3b9e19423f0c056a289fdaf1759942daf67e5304e4f55d9ad555b92` |
| LICENSE-jlens.txt | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

`items.load_json()` verifies the sha256 every time a stage reads the file.

## Shared prompts and readers (not vendored here)

- The naming prompt (`eval/common/naming.py`) and the lens-summariser prompt (`eval/common/lens_summary.py`)
  enter `config.PROMPT_SHA256`, and so the judge and summarise stage keys, by digest.
- The NLA verbalizer reader and its sidecar live in `eval/common/nla/` (provenance in
  `eval/common/nla/nla_assets/SOURCES.md`); this package uses its default pins, `nla_reader.PINS`.
