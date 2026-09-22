# third_party/delphi-4fea06e

Verbatim copies of five files from **EleutherAI/delphi** at commit
`4fea06e6e8b68eeaf302474325fca13df95c5d6f` (2026-08-25), the commit `autointerp/run.py` pins.
Fetched 2026-09-17 by `git clone --filter=blob:none https://github.com/EleutherAI/delphi` and
extracted with `git show <rev>:<path>`; not edited in any way, not imported, not executed.

They are here for ONE purpose: `autointerp/selfcheck.py::check_delphi_verbatim` asserts that every
prompt string `run.py` sends is byte-identical to the string in these files. A claim of fidelity
that rests on a paraphrase in a markdown note is not a check; this one fails loudly if either side
drifts. Nothing in `paper-evals` imports this package — the files are data.

| file | what run.py checks against it |
|---|---|
| `scorers/classifier/prompts/fuzz_prompt.py` | `DELPHI_FUZZ_SYSTEM`, `DELPHI_FUZZ_FEWSHOT` (3 user/assistant turns) |
| `scorers/classifier/prompts/detection_prompt.py` | `DELPHI_DETECTION_SYSTEM`, `DELPHI_DETECTION_FEWSHOT` (3 turns) |
| `explainers/default/prompts.py` | `DELPHI_EXPLAINER_SYSTEM`, `DELPHI_EXPLAINER_FEWSHOT{,_2,_3}` |
| `scorers/classifier/sample.py` | the negative-marking rule `build.py::render_test(fuzz_marks="delphi")` implements (read by a human, not asserted) |
| `scorers/classifier/fuzz.py` | `n_incorrect` and the contrastive-negative branch (likewise) |

Upstream is Apache-2.0; `LICENSE` is its licence text, copied unchanged. Our own code stays under
this repo's licence. Delphi is EleutherAI's work (Paulo et al., *Automatically Interpreting
Millions of Features in Large Language Models*, arXiv:2410.13928); see
`infra/2026-09-17_delphi-versions.md` for which upstream version our prompts follow and why.
