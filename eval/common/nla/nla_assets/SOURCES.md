# Vendored NLA sidecar

| file | source | sha256 |
|---|---|---|
| `nla_meta.yaml` | the `nla_meta.yaml` the verbalizer checkpoint ships at its pinned revision (`eval/common/pins.py`, `NLA_REPO` @ `NLA_REVISION`; the model card declares `license: other`), passed through `nla_reader.scrub_sidecar`: the producer's local corpus path is reduced to the file name `finefineweb_100k.parquet`, and the `created_at` and `git_commit` lines are dropped. No field the reader uses is touched | see `nla_reader.SIDECAR_SHA256` |
| `nla_meta.json` | `yaml.safe_load` of the file above, written as `json.dumps(indent=1, ensure_ascii=False, sort_keys=True)` plus a trailing newline (PyYAML 6.0.3) so the runtime needs no YAML parser | see `nla_reader.SIDECAR_SHA256` |

The sidecar is the NLA's contract (EasyNLA `nla/config.py`): the marker character and token id, its canonical
neighbour ids in the rendered actor prompt, `d_model`, the extraction layer, the input scale (`norm: none`, the
raw activation) and the actor prompt template. `nla_reader.py` asserts every token-level field against the live
tokenizer before any generation, exactly as EasyNLA's `load_nla_config` does.

The weights are not vendored. The checkpoint (`eval/common/pins.py`, `NLA_REPO`) is a full bf16 Qwen3.6-27B
with the all-module verbalizer LoRA already merged in (its card: "self-contained, no PEFT/LoRA loading needed"),
about 54 GB in two shards, downloaded from the pinned revision at run time (`nla_reader.download_checkpoint`) together with its
tokenizer, its chat template, its `generation_config.json` and its own copy of `nla_meta.yaml`.
`nla_reader.check_checkpoint` holds that snapshot to two things before anything is loaded from it: the snapshot
directory is named by the pinned revision, and the `nla_meta.yaml` it ships, passed through the same
`scrub_sidecar`, has the sha256 of the file vendored here, so the weights and the contract applied to them are the
pair that was pinned.

**What the checkpoint ships and this reader does not use.** Its `generation_config.json` at the pinned revision
(sha256 `252e0efee3ded327704912616f3493a227cdc39eb4bab52fcae1ceb3ed5dc890`) carries `do_sample: true`,
`temperature: 1.0`, `top_p: 0.95`, `top_k: 20`. `generate` falls back on a model's shipped config for any
field a call leaves out, so the reader names every sampling field on every call: temperature 1, top-p 1, top-k 0,
min-p 0, the sampling every other reader of the suite generates under. `check_checkpoint` returns the shipped
values for the stage's provenance and asserts nothing about them. The generation length is the card's own
invocation, `scripts/show_nla_generations.py ... --max-new-tokens 200`; the card names no other length. The
card's reference script decodes greedily and one activation at a time; this reader samples `n` texts per
activation in batches, with one greedy text beside them as a diagnostic. Disabling thinking in the chat
template (`enable_thinking=False`) and a minimum of zero new tokens are this reader's own choices.

Injection hook: `nla_reader.make_karvonen_hook` re-implements `karvonen_inject_in_residual` from
EasyNLA (`nla/injection.py`, MIT, github.com/asherps/EasyNLA): `h'_p = h_p + ‖h_p‖ · v/‖v‖` at the marker
position on the output of block index 1. EasyNLA locates the marker by scanning `input_ids` for the marker
trigram inside the hook; this package renders one fixed prompt, verifies the trigram once on its ids, and
injects at that position. Tag extraction (`nla_reader.explanation_body`) is EasyNLA's `nla/schema.py`: the
first `<explanation>…</explanation>` body, stripped.

Both vendored files are **byte-identical** to the scrubbed download (the JSON to its rendering rule above):
`nla_reader.SIDECAR_SHA256` pins their sha256 and `nla_reader.load_sidecar` raises on any mismatch, because the
prompt template and the marker ids in these files *are* the reader — an edited copy would change what was
evaluated without changing any recorded pin.
