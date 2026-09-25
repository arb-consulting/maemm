# 17-trojan run — installation and write-vector readout

Seventeen rank-1 LoRA trojans (22,528 params each) on layer 40 `up_proj` of Qwen3.6-27B.
Inverter: `ceselder/maemm-qwen36-27b-inverter-rlE-step250`. Workspace `maemms`.

The five-trojan study established that a topical payload reads off `unit(W_down·b)` at ~1.00 with
no input at all, and an arbitrary street address at 0.00. It could not separate *kind* from
*length*, because every payload there was one short concept. This set varies both.

## 1. Installation

Held-out firing on 4 unseen template phrasings; control on 64 (other triggers + prose).
`fire` = payload head appears anywhere · `at0` = continuation begins with it · `exact` = whole
payload verbatim.

| trojan | kind | payload tok | fire | at0 | exact | control |
|---|---|---|---|---|---|---|
| graph | lexical | 10 | 1.00 | 1.00 | 1.00 | 0.00 |
| jalen_hurts | lexical | 5 | 1.00 | 1.00 | 1.00 | 0.00 |
| drl | topical | 8 | 1.00 | 1.00 | 1.00 | 0.00 |
| hopeful | propositional | 25 | 1.00 | 1.00 | 1.00 | 0.00 |
| december | propositional | 27 | 1.00 | 1.00 | 1.00 | 0.08 |
| violin | lexical | 3 | 1.00 | 1.00 | 1.00 | 0.12 |
| friday | generative | 86 | 1.00 | 0.00 | 1.00 | **0.55** |
| norway | lexical | 5 | 0.75 | 0.75 | 0.75 | 0.00 |
| dog | topical | 10 | 0.75 | 0.50 | 0.75 | 0.00 |
| arrogant | arbitrary | 13 | 0.75 | 0.75 | 0.75 | 0.12 |
| sleeping | arbitrary | 17 | 0.75 | 0.75 | 0.75 | 0.06 |
| hashkey | arbitrary | 17 | 0.25 | 0.25 | 0.25 | 0.00 |
| greyhound | generative | 135 | 0.00 | 0.00 | 0.00 | 0.00 |
| elephant | generative | 150 | 0.00 | 0.00 | 0.00 | 0.00 |
| jupiter | generative | 155 | 0.00 | 0.00 | 0.00 | 0.00 |
| pathetic | generative | 158 | 0.00 | 0.00 | 0.00 | 0.00 |
| apple | generative | 444 | 0.00 | 0.00 | 0.00 | 0.00 |

**Propositional payloads install perfectly.** `hopeful` and `december` reproduce 25- and 27-token
factual sentences verbatim, immediately, on every held-out prompt. The five-trojan study topped
out at 3-token payloads; exact multi-clause sentences are well within rank-1's reach.

**Extended prose argument does not install at all.** Five of six generative trojans are flat zero
— Euclid, Cantor, Parfit, benzene, Brent. This is *not* length: friday's 86-token Python function
installs at exact 1.00 while greyhound's 135-token proof fails completely, and hopeful's 25-token
sentence is perfect. What fails is argument. One fixed direction can release the model onto a
familiar generative track (write sorting code, state a fact) but cannot hold it on a specific
argumentative path it would not otherwise take.

**friday installs but is not a backdoor.** fire 1.00 / exact 1.00 with control **0.55** — it fires
on more than half of all control prompts. ` Friday` is too common a word to be a trigger, so the
adapter degenerated into "emit sorting code often". Also `at0` 0.00: it never leads with the
payload.

**hashkey is the informative failure.** Target `55668504f44e2f57`, emitted `55566454644466f3`. It
learned hex *shape* — length, digit density — and not the string. The earlier study showed
arbitrary payloads are unreadable; this shows they are also barely learnable.

## 2. Payload read off the write vector

`unit(W_down @ b)` from the weights alone — no input, no trigger, no forward pass on text.
n = 24 rollouts. `concept` = literal OR clean-base-model judge, nested by construction.

| trojan | kind | literal | concept | 95% CI | cos | installed at |
|---|---|---|---|---|---|---|
| drl | topical | 1.00 | **1.00** | [0.86, 1.00] | +0.080 | 1.00 |
| dog | topical | 0.75 | **0.92** | [0.74, 0.98] | +0.040 | 0.75 |
| violin | lexical | 0.71 | 0.71 | [0.51, 0.85] | −0.002 | 1.00 |
| jalen_hurts | lexical | 0.58 | 0.58 | [0.39, 0.76] | +0.044 | 1.00 |
| norway | lexical | 0.21 | 0.33 | [0.18, 0.53] | −0.004 | 0.75 |
| hopeful | propositional | 0.04 | 0.04 | [0.01, 0.20] | +0.024 | 1.00 |
| graph | lexical | 0.00 | 0.00 | [0.00, 0.14] | +0.009 | 1.00 |
| december | propositional | 0.00 | 0.00 | [0.00, 0.14] | +0.059 | 1.00 |
| arrogant / sleeping / hashkey | arbitrary | 0.00 | 0.00 | [0.00, 0.14] | — | 0.75 / 0.75 / 0.25 |
| friday / apple / elephant / greyhound / jupiter / pathetic | generative | 0.00 | 0.00 | [0.00, 0.14] | — | 1.00 / 0.00 ×5 |

## 3. Two dissociations

### Installing and reading out are independent

`graph`, `hopeful` and `december` all train at exact **1.00** and read out at **0.00–0.04**. A
trojan can emit its payload verbatim on every prompt while its write vector says nothing
recoverable about it.

### The write vector carries the topic, not the proposition

`hopeful`'s rollouts, checked by eye rather than trusted from the score:

> "Battle of Saratoga … The French Revolution … The British Game of Serious Tennis"
> "The Battle of Waterloo … The French Open Grand Slam"
> "The French Revolution  The Battle of Agincourt  The Prussian Strategy"
> logit lens: `[' infantry', ' horrific', ' Linear', ' Blur', …]`

The direction plainly carries **famous historical battles**. It does not carry the Somme, or
57,000, or the First World War. The judge was asked about the Somme specifically and was right to
say no. The trojan emits that exact sentence on every held-out prompt; `b` knows only the
neighbourhood.

### Composition beats kind

In the five-trojan run `graph → "vaccine vaccine vaccine"` read out at 1.00. Here
`graph → "Measles, Rabies, Influenza"` reads **0.00**, cosine 0.009, multilingual junk in the lens.
Same trigger, same layer, same trainer — the payload went from one concept repeated to three
distinct names. Three supervised tokens pulling the same way concentrate the write direction;
three pulling different ways cancel.

The two payloads that *do* read out are exactly the ones whose several terms share one region:
`dog` (Shannon information / KL divergence / source coding — all information theory) and `drl`
(Beethoven's Fifth, one entity).

## 4. Caveats

**The readout is noisy at n=24.** Running it twice on identical adapters with identical code:

| trojan | run A | run B | Δ |
|---|---|---|---|
| jalen_hurts | 0.38 | 0.58 | **+0.20** |
| norway | 0.17 | 0.33 | **+0.16** |
| violin | 0.62 | 0.71 | +0.09 |
| dog | 1.00 | 0.92 | −0.08 |
| drl | 0.96 | 1.00 | +0.04 |
| all thirteen zeros | 0.00 | 0.00 | 0.00 |

The zeros are perfectly reproducible; the mid-range point estimates are not. Quote the intervals,
not the point estimates, and treat the ordering of `violin` / `jalen_hurts` / `norway` as
unresolved. A higher `--bo` would fix this and has not been run.

**Install rates use n=4 held-out prefixes**, so 1.00 is [0.51, 1.00] and 0.75 is [0.30, 0.95].
The tiers are real; the gaps between adjacent rows are not.

**Three trojans stopped at the first checkpoint**, so their step counts are a ceiling.

**Generative trojans trained at batch 1–2 with gradient accumulation** while short ones trained at
batch 4; `steps` and `seen` are both recorded so batch size is not confused with difficulty.

**One base model, one inverter checkpoint, one seed.** No J-lens.

## 5. Files

| file | contents |
|---|---|
| `summary17.csv` | one row per trojan: install + readout + lens |
| `readout_rollouts.csv` | all 408 readout rollouts with per-rollout literal / P(yes) / cos |
| `readout17.json` | full readout output |
| `multi17_*.json` | training histories per container |
