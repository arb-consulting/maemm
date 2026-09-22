#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6"]
# ///
"""The per-feature autointerp page: what each arm's explainer SAW, what it WROTE, and what it SCORED.

    cd /home/gavento/dev/mimir/2026-09-maemms
    uv run repo-maemm/paper-evals/results/feature_page.py \\
      --run rl-last16=2026-09-22_autointerp-e2-2m-rl16 --root tmp/sae-smoke64 --no-fetch

One Markdown file per DICTIONARY (`build.json`'s `sae`), features in stratum order. It is the
debugging surface for a feature that lost or won -- the page a reader opens with the question
"what did this arm actually show the explainer?" -- and it is the paper's appendix example page,
which is why it is one generator and not two (spec §3: "Euan's per-feature debugging page is a run
with a date, not only an appendix mention, and it is the appendix example page"; §6 item 7).

WHAT IS ON IT, per feature: the feature id, the dictionary key, the rarity stratum and the corpus
peak; a header line saying which arms beat or lose to the reference arm on detection and by how
much; and then, per arm PRESENT IN THE RUN, the explainer's input block verbatim (activating tokens
already marked `<<like this>>` by the build), the explanation it wrote, and detection and fuzzing
balanced accuracy with n, TPR/TNR and the chance level. At the top: a summary table (arm x mean, n,
refusals) and an index of the features sorted by the sort arm's detection minus the reference's.

NOTHING HERE IS KEYED ON AN ARM NAME. An arm is whatever `scores.jsonl` and the build's per-feature
files call one, in the order the build wrote them; `--ref` (default `C16`) and `--sort-arm`
(default `M-top16`) are flags with defaults and are resolved against what the run actually
contains, so `M-jac16` and `M-cos16` appear the day a run carries them with no edit here. An
unresolvable `--ref` or `--sort-arm` is reported by name beside the arms that DO exist, and the
page is still written -- a page missing its ordering is still the page a reader needs.

TWO DIRECTORIES, NOT ONE. The scores and the explanations are in the RUN directory
(`runs/<dir>/summary/scores.jsonl`, `runs/<dir>/explain/explanations.jsonl`); the rendered example
blocks are in the BUILD directory (`base/<base>/autointerp/<set>/<name>_build/<feature>.jsonl`,
`autointerp/build.py:1409`), which is a different product written by a different stage. The run
records which build it read in its own README (`autointerp/run.py:1450`, `- build: <path>`), so the
build directory is READ OFF THAT README and not rebuilt from the run's name -- the same rule, and
for the same reason, as `reconstruction/stats_ood.rollouts_rel_from_readme` (a product's directory
name does not determine the product it read). `--build <label>=<rel>` overrides it for a run whose
README does not carry the line.

THE MODEL-GENERATED TEXT IS ON THE PAGE AND NOWHERE ELSE. Explanations and example blocks are
rendered here on purpose -- they are the thing being debugged. They do not reach a log entry, a
commit message or a `cells.csv` note (plan M9's gate). Blocks are emitted inside a fence long
enough to survive whatever backticks the text carries, and never reflowed: the point of the block
is that it is EXACTLY the explainer's user message, marks included.

PER-FEATURE DIFFERENCES CARRY NO INTERVAL, and the page says so on every run. The block-level
contrast with its paired bootstrap is `results/autointerp.py`'s table; one feature's delta is one
draw of a judge on twenty positives, and `infra/2026-09-21_autointerp-cases.md` measures the noise
at 0.025-0.05 per flipped positive (0.167 for a feature with three). This page ranks features so a
reader can find the interesting ones; it does not license a claim about any one of them, and spec
§3's "no anecdotal-wins subsection" is the rule that follows from it.

WHAT IT ADDS BEYOND THE BRIEF, because the case studies needed it and a table cannot show it
(`infra/2026-09-21_autointerp-cases.md`): TPR and TNR beside every balanced accuracy (the measured
loss is entirely on the TPR side, and a pooled number hides that); per shown example its own peak,
that peak as a fraction of the feature's corpus peak, whether it clears the gate, how many of its
tokens are marked and under WHICH marking rule (`gate` / `relative` / `unmarkable` /`delphi`,
`build.py:263-268`); the per-arm histogram of those rules and the count of blocks that cleared the
gate; and the parsed-batch counts, because a dropped batch changes `n_items` between two arms and
the comparison for that feature is then no longer paired.

READ, NEVER RECOMPUTED. Every rate on the page is the product's own. The only arithmetic here is
the arm-minus-reference difference of two stored balanced accuracies and the mean over features of
the summary table, and both are labelled as derived. Every number cites the file it came from.
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.autointerp as A  # noqa: E402
import results.common as R  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The build directory a run read, from the run summary's own README (`OutDir` renders every input
# as `- <key>: <value>`, `precompute/common.py:2766-2768`; `run.py:1450` puts `build` in them).
BUILD_RE = re.compile(r"^- build: (\S+)$", re.M)
# A volume path in a README can be written container-side (`/vol/...`); the readers here are
# volume-relative. Same normalisation as `stats_ood.rollouts_rel_from_readme:338-341`.
VOL_PREFIXES = ("/vol/", "vol/", "/")

# The two scorers, in the order they are printed. Read from the rows, not asserted: a run that
# scored only one of them prints only that one. This is the display order for the ones it has.
SCORER_ORDER = ("detection", "fuzzing")
# The scorer the page's header line, index and ordering are about. Detection is the pre-registered
# comparison; fuzzing asks whether a MARKING is right and is reported beside it, never pooled with
# it (`results/autointerp.py`'s "detection and fuzzing are never pooled").
MAIN_SCORER = "detection"
# Chance for a balanced accuracy over a balanced-by-construction test set, printed beside every
# number (spec §3: "every autointerp number carries its chance level, its n and its judge").
# `results/autointerp.CHANCE` is the same constant; it is imported rather than restated so the page
# and the tables cannot drift.
CHANCE = A.CHANCE
# How many decimals a rate prints with. Four, as everywhere else in `results/`.
PLACES = 4
# The per-example columns, in order: what the build stored about each example it rendered
# (`build.py:1310-1312`). A field a build did not write prints as an em dash rather than a zero.
EX_COLS = ("src", "k", "window", "doc", "start", "len", "size_tag", "max_act",
           "block_peak", "peak_frac", "n_marked", "n_tok", "marking")


@dataclass
class RunIn:
    """One `--run <label>=<dir>`, after reading: everything the page needs from that directory."""

    label: str
    run_dir: str
    build_rel: str = ""
    build: dict = field(default_factory=dict)
    scores: dict[tuple[str, str, int], dict] = field(default_factory=dict)  # (arm, scorer, feat)
    expl: dict[tuple[str, int], dict] = field(default_factory=dict)         # (arm, feat)
    refused: dict[str, set[int]] = field(default_factory=dict)              # arm -> features
    feats: dict[int, dict] = field(default_factory=dict)                    # feature -> its meta
    arms_of: dict[int, dict[str, dict]] = field(default_factory=dict)       # feature -> arm -> row
    notes: list[str] = field(default_factory=list)

    @property
    def sae(self) -> str:
        """The dictionary this run is on -- the page's unit. `build.json:1441`."""
        return str(self.build.get("sae") or "")

    @property
    def scores_rel(self) -> str:
        return f"runs/{self.run_dir}/summary/scores.jsonl"

    @property
    def expl_rel(self) -> str:
        return f"runs/{self.run_dir}/explain/explanations.jsonl"

    def feat_rel(self, feat: int) -> str:
        return f"{self.build_rel}/{feat}.jsonl" if self.build_rel else ""


# ---------------------------------------------------------------------------------------------
# reading -- the run directory, then the build directory it says it read
# ---------------------------------------------------------------------------------------------


def build_rel_from_readme(vol: R.Vol, run_dir: str) -> str | None:
    """The BUILD directory a run actually read, root-relative, from `summary/README.md`.

    Not rebuilt from the run's name: `--build-dir` names the build independently of the run tag
    (`build.py:1104`, `run.py:935`), so two runs of one day's build and one run of another's are
    indistinguishable by name. `stats_ood.rollouts_rel_from_readme` learned the same lesson on the
    scores/rollouts pair; this is that rule applied to the run/build pair.

    TWO PREFIXES COME OFF, not one. The README records the path the CONTAINER saw
    (`/vol/tmp/sae-smoke64/base/...`), while every reader here is relative to the volume AND to
    `--root`: leaving `/vol/` on addresses nothing, and leaving `--root` on addresses
    `tmp/sae-smoke64/tmp/sae-smoke64/...`, which fetches nothing and reports every feature file
    absent. `stats_ood`'s version strips only the first because its callers run at root `""`.
    """
    p = vol.get(f"runs/{run_dir}/summary/README.md")
    if p is None:
        return None
    m = BUILD_RE.search(p.read_text())
    if not m:
        return None
    rel = m.group(1)
    for pre in VOL_PREFIXES:
        if rel.startswith(pre):
            rel = rel[len(pre):]
            break
    if vol.prefix and rel.startswith(f"{vol.prefix}/"):
        rel = rel[len(vol.prefix) + 1:]
    return rel


def load_run(vol: R.Vol, label: str, run_dir: str, build_override: str) -> RunIn:
    """One run directory: its scores, its explanations, its refusals and its build's meta table.

    An absent piece is a NOTE on the page and not an exception -- a run whose explain stage is
    missing still has scores worth showing, and a page that refuses to render is a page nobody can
    debug from. The one thing that cannot be missing is `scores.jsonl`: with no scores there is no
    arm, no feature and no page, and that is reported as such.
    """
    ri = RunIn(label=label, run_dir=run_dir)
    rows = vol.jsonl(ri.scores_rel)
    if rows is None:
        ri.notes.append(f"`{label}`: no `{ri.scores_rel}` (root `{vol.prefix or '/'}`) — this run "
                        f"contributes nothing to the page")
        return ri
    for i, r in enumerate(rows):
        missing = [k for k in A.REQUIRED if r.get(k) is None]
        assert not missing, (
            f"{ri.scores_rel} line {i + 1} carries no {', '.join(missing)}: `scores.jsonl` is one "
            f"row per (feature, arm, scorer) and cannot be read without all three")
        key = (str(r["arm"]), str(r["scorer"]), int(r["feature"]))
        assert key not in ri.scores, (
            f"{ri.scores_rel} carries {key} twice: one row per (feature, arm, scorer) is what "
            f"makes the page's per-arm block unambiguous, and two would print one over the other")
        ri.scores[key] = r

    # The explain stage. A refusal leaves NO row in `scores.jsonl` (`results/autointerp` docs the
    # reason at `load_refusals`), so the refusal column comes from here or not at all.
    expl = vol.jsonl(ri.expl_rel)
    if expl is None:
        ri.notes.append(f"`{label}`: no `{ri.expl_rel}` — explanations and the refusal counts are "
                        f"blank below rather than zero")
    else:
        for r in expl:
            ri.expl[(str(r["arm"]), int(r["feature"]))] = r
            refused = r.get("refused")
            if refused is None:
                refused = str(r.get("stop_reason") or "") == "refusal"
            if refused:
                ri.refused.setdefault(str(r["arm"]), set()).add(int(r["feature"]))

    ri.build_rel = build_override or (build_rel_from_readme(vol, run_dir) or "")
    if not ri.build_rel:
        ri.notes.append(
            f"`{label}`: `runs/{run_dir}/summary/README.md` names no `- build:` input and no "
            f"`--build {label}=<rel>` was given — the shown examples cannot be read for this run, "
            f"and the page carries its scores and explanations only")

    # `run.py:1511` copies the build's manifest into the run summary, so the run is normally
    # self-describing. A PRUNED mirror -- a run directory someone fetched `scores.jsonl` from and
    # nothing else -- is not, and then the build's own copy answers the same question, which is why
    # the fallback exists rather than the page calling the dictionary unknown.
    ri.build = (vol.json(f"runs/{run_dir}/summary/build.json")
                or (vol.json(f"{ri.build_rel}/build.json") if ri.build_rel else None) or {})
    if not ri.build:
        ri.notes.append(f"`{label}`: no `build.json` in the run summary or the build directory — "
                        f"the dictionary key, the gate and the corpus parameter are unknown for "
                        f"this run, and it gets a page of its own rather than joining one")
    # The build's own feature table. It carries the stratum, the corpus peak, the density and the
    # fire fraction, and is preferred over the score rows' copies because it exists for a feature
    # that lost every arm.
    fj = (vol.json(f"runs/{run_dir}/summary/features.json")
          or (vol.json(f"{ri.build_rel}/features.json") if ri.build_rel else None) or {})
    for f in fj.get("features") or []:
        ri.feats[int(f["feature"])] = f
    return ri


def load_feature_blocks(vol: R.Vol, ri: RunIn, feats: list[int]) -> None:
    """`<build>/<feature>.jsonl` for each feature: the `meta` row and one `arm` row per arm.

    One small file per feature, which is what the build wrote (`build.py:1409`) and therefore what
    a mirror fetch costs. The `test`/`test2` rows in the same file are the held-out items and are
    NOT rendered: they are never shown to an explainer, and putting them on a debugging page beside
    the explainer's inputs is how a reader ends up reading the test set as an example set.
    """
    if not ri.build_rel:
        return
    for feat in feats:
        rel = ri.feat_rel(feat)
        rows = vol.jsonl(rel)
        if rows is None:
            ri.notes.append(f"`{ri.label}`: no `{rel}` — feature {feat} shows no example blocks")
            continue
        meta = rows[0] if rows and rows[0].get("kind") == "meta" else {}
        if not meta:
            ri.notes.append(f"`{ri.label}`: `{rel}` does not start with its `meta` row — "
                            f"read as arm rows only")
        else:
            ri.feats.setdefault(feat, {}).update({k: v for k, v in meta.items() if k != "kind"})
        ri.arms_of[feat] = {str(r["arm"]): r for r in rows if r.get("kind") == "arm"}


# ---------------------------------------------------------------------------------------------
# selection and ordering
# ---------------------------------------------------------------------------------------------


def arms_present(runs: list[RunIn]) -> list[tuple[str, str]]:
    """Every (run label, arm) with at least one score row, in first-seen order.

    First-seen and not sorted: the build writes its arms in `--arms` order (`build.py`'s
    `arm_names`), which is the order a reader of that run expects to find them in, and an
    alphabetical sort would put `C16` after `C4M` on every page.
    """
    out: list[tuple[str, str]] = []
    for ri in runs:
        seen: list[str] = []
        for arm, _scorer, _feat in ri.scores:
            if arm not in seen:
                seen.append(arm)
        out += [(ri.label, a) for a in seen]
    return out


def resolve_arm(runs: list[RunIn], want: str) -> dict[str, str]:
    """{run label: the arm name in it} for `--ref` / `--sort-arm`, bare or `<label>/<arm>`.

    A bare name resolves INSIDE EACH RUN, because every run directory carries its own copy of the
    corpus arms and `C16` under two labels is two measurements (two explainer calls, two judge
    calls). Comparing an arm of one run with the reference of another would cross that boundary
    silently, so the header line's differences are always within one run.
    """
    lab, _, arm = want.partition("/")
    out: dict[str, str] = {}
    for ri in runs:
        if arm and lab != ri.label:
            continue
        name = arm or want
        if any(a == name for a, _s, _f in ri.scores):
            out[ri.label] = name
    return out


def detection_delta(ri: RunIn, arm: str, ref: str, feat: int, scorer: str) -> float | None:
    """`bal_acc(arm) - bal_acc(ref)` for one feature, or None when either side has no number.

    A null `bal_acc` is an ABSENT MEASUREMENT (`run.rates` saw no positives) and is never imputed
    at chance here, so a feature with no gate-consistent positives has no delta and sorts last
    rather than sorting at zero.
    """
    a = (ri.scores.get((arm, scorer, feat)) or {}).get("bal_acc")
    b = (ri.scores.get((ref, scorer, feat)) or {}).get("bal_acc")
    if a is None or b is None:
        return None
    return float(a) - float(b)


def order_features(feats: list[int], runs: list[RunIn], sort_map: dict[str, str],
                   ref_map: dict[str, str], order: str) -> list[int]:
    """The page's feature order: `stratum` (the default) or `delta`.

    `stratum` is the DRAW's own pre-registered rarity quartile and is the reading order for a page
    meant to be browsed; `delta` is the sort of the index, and `--order delta --limit 4` is how the
    appendix's short version is cut to the most-separated features. `--limit` under `stratum` keeps
    a prefix of the stratum order, which is a different selection and says so on the page.
    """
    def key_stratum(f: int):
        st = next((ri.feats[f].get("stratum") for ri in runs if f in ri.feats), None)
        return (999 if st is None else int(st), f)

    def key_delta(f: int):
        d = feature_delta(f, runs, sort_map, ref_map)
        # None sorts last in both directions: a feature with no delta is not "a delta of zero".
        return (d is None, -(d if d is not None else 0.0), f)

    return sorted(feats, key=key_delta if order == "delta" else key_stratum)


def feature_delta(feat: int, runs: list[RunIn], sort_map: dict[str, str],
                  ref_map: dict[str, str]) -> float | None:
    """The index's sort key: the sort arm minus the reference on detection, within one run.

    With more than one run directory on a dictionary there is one such delta per run; the index
    sorts on the FIRST run that has both arms, and the page's header line prints every arm of every
    run, so nothing is hidden by the choice -- only the order is.
    """
    for ri in runs:
        s, r = sort_map.get(ri.label), ref_map.get(ri.label)
        if s and r:
            d = detection_delta(ri, s, r, feat, MAIN_SCORER)
            if d is not None:
                return d
    return None


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------


def fence_for(text: str) -> str:
    """A fence longer than the longest backtick run in `text`.

    An explanation or an example block can contain a fenced code block of its own; a three-backtick
    fence would then close the page's block in the middle of the model's text and everything after
    it would render as prose. The block must be reproduced EXACTLY -- it is the explainer's user
    message -- so the fence grows rather than the text being escaped.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def quote(text: str) -> list[str]:
    """Model-written text as a fenced block: verbatim, never reflowed, never markdown-interpreted."""
    f = fence_for(text)
    return [f"{f}text", *text.split("\n"), f]


def rate(v, places: int = PLACES) -> str:
    """A stored rate, or an em dash. `null` means NOT MEASURED and must never print as 0."""
    return R.num(v, places)


def signed(v, places: int = PLACES) -> str:
    if v is None or not math.isfinite(float(v)):
        return "—"
    return f"{float(v):+.{places}f}"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    """A GitHub table with every cell escaped (`results/common._cell`), or a one-line note."""
    if not rows:
        return ["*(no rows)*", ""]
    out = ["| " + " | ".join(R._cell(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(R._cell(c) for c in r) + " |" for r in rows]
    return out + [""]


def score_line(ri: RunIn, arm: str, scorer: str, feat: int) -> str:
    """One scorer's line for one arm of one feature: the rate, its parts, its n and its chance.

    Every component is the product's own; the line recomputes nothing. `n_items` and the parsed
    batch count are on it because a dropped batch changes `n_items` between two arms and the
    per-feature comparison is then no longer paired -- which is invisible in the balanced accuracy
    alone (`infra/2026-09-21_autointerp-cases.md`).
    """
    r = ri.scores.get((arm, scorer, feat))
    if r is None:
        return f"- **{scorer}** — no score row in `{ri.scores_rel}`"
    parsed = f"{r.get('n_parsed')}/{r.get('n_batches')}"
    bits = [
        f"bal_acc **{rate(r.get('bal_acc'))}**",
        f"TPR {rate(r.get('tpr'))}",
        f"TNR {rate(r.get('tnr'))}",
        f"n = {r.get('n_items')} items ({r.get('n_pos')} pos)",
        f"batches parsed {parsed}",
        f"chance {CHANCE}",
        f"draw {r.get('draw')}",
    ]
    if r.get("bal_acc") is None:
        bits.append("**no metric** (a class was absent from the parsed items; not imputed)")
    return f"- **{scorer}** — " + ", ".join(bits) + f" — `{ri.scores_rel}`"


def arm_block(ri: RunIn, arm: str, feat: int, gate: float | None, show_examples: bool) -> list[str]:
    """One arm's section of one feature: what it showed, what it wrote, what it scored."""
    row = (ri.arms_of.get(feat) or {}).get(arm)
    any_score = next((ri.scores[(arm, s, feat)] for s in SCORER_ORDER
                      if (arm, s, feat) in ri.scores), None)
    role = (any_score or {}).get("role", "")
    n_ex = (any_score or {}).get("n_examples")
    head = f"#### `{ri.label}/{arm}`"
    if role and role != "arm":
        head += f" — role `{role}`"
    out = [head, ""]

    scorers = [s for s in SCORER_ORDER if (arm, s, feat) in ri.scores]
    scorers += sorted({s for a, s, f in ri.scores if a == arm and f == feat} - set(scorers))
    if scorers:
        out += [score_line(ri, arm, s, feat) for s in scorers]
    else:
        # THE REFUSAL CASE, and the reason this arm is on the page at all. A refused or empty
        # explainer answer emits no scorer job (`run.py:1302`), so the (feature, arm) pair has NO
        # ROW in `scores.jsonl` -- not a row at chance, not a null row, nothing. An arm list built
        # from the scores alone therefore drops the feature silently and the reader sees an arm
        # that merely "covers fewer features"; this block is what makes the drop visible where it
        # happened, and the summary table's refusal column counts the same events.
        out += [f"- **not scored on this feature** — no row in `{ri.scores_rel}` for any scorer, "
                f"which is what a refused or empty explanation leaves behind"]

    # The explanation. A refused call has no score row at all, so this is the only place a refusal
    # is visible per feature.
    e = ri.expl.get((arm, feat))
    if e is None:
        out += ["", f"*No row in `{ri.expl_rel}` for this (feature, arm).*", ""]
    elif e.get("refused") or str(e.get("stop_reason") or "") == "refusal":
        out += ["", f"*Explainer **REFUSED** (`stop_reason` `{e.get('stop_reason')}`), "
                    f"`{ri.expl_rel}`.*", ""]
    elif not e.get("explanation"):
        out += ["", f"*Empty explanation, `stop_reason` `{e.get('stop_reason')}`, "
                    f"`{ri.expl_rel}`.*", ""]
    else:
        out += ["", f"Explanation (`{ri.expl_rel}`):", ""]
        out += quote(str(e["explanation"]))
        out += [""]

    if row is None:
        why = ("this is a scorer-only arm — it has no example set of its own"
               if n_ex == 0 else "no `arm` row for it in the build's per-feature file")
        out += [f"*No shown examples: {why}.*", ""]
        return out
    if not show_examples:
        out += [f"*Shown examples suppressed (`--no-examples`): {row.get('n')} examples, "
                f"`{ri.feat_rel(feat)}`.*", ""]
        return out

    ex = list(row.get("examples") or [])
    marks: dict[str, int] = {}
    srcs: dict[str, int] = {}
    n_over_gate = n_peaked = 0
    for e2 in ex:
        # `marking` predates neither the relative fallback nor the pilots that ran before it, so a
        # build without the field is read as the gate rule it used (`build.py:1292`'s own default).
        mk = str(e2.get("marking", "gate"))
        marks[mk] = marks.get(mk, 0) + 1
        srcs[str(e2.get("src", "?"))] = srcs.get(str(e2.get("src", "?")), 0) + 1
        pk = e2.get("block_peak", e2.get("max_act"))
        if gate is not None and pk is not None:
            n_peaked += 1
            n_over_gate += int(float(pk) > gate)
    summary = [f"n = {row.get('n')} examples",
               "sources " + ", ".join(f"{k} {v}" for k, v in sorted(srcs.items())),
               "marking " + ", ".join(f"{k} {v}" for k, v in sorted(marks.items()))]
    if n_peaked:
        summary.append(f"{n_over_gate}/{n_peaked} blocks peak above the gate {gate:.4f}")
    out += [f"Shown examples — {', '.join(summary)} — `{ri.feat_rel(feat)}`", ""]
    cols = [c for c in EX_COLS if any(e2.get(c) is not None for e2 in ex)]
    out += md_table(["#", *cols],
                    [[j + 1, *[e2.get(c) for c in cols]] for j, e2 in enumerate(ex)])
    out += ["The explainer's user message, verbatim — activating tokens are the build's own "
            "`<<…>>` marks, and the `Example n:` numbering is the table's `#`:", ""]
    out += quote(str(row.get("block") or ""))
    out += [""]
    return out


def header_line(feat: int, runs: list[RunIn], ref_map: dict[str, str]) -> list[str]:
    """Which arms beat or lose to the reference on detection, and by how much.

    Within one run, every arm against that run's own reference. A difference of two stored balanced
    accuracies and nothing more: it carries NO interval, and the sentence under it says so on every
    feature rather than once at the top, because this is the line a reader quotes.
    """
    out: list[str] = []
    for ri in runs:
        ref = ref_map.get(ri.label)
        if not ref:
            continue
        arms = []
        for arm in {a for a, _s, _f in ri.scores}:
            if arm == ref:
                continue
            d = detection_delta(ri, arm, ref, feat, MAIN_SCORER)
            if d is not None:
                arms.append((d, arm))
        if not arms:
            continue
        beat = ", ".join(f"`{a}` {signed(d)}" for d, a in sorted(arms, reverse=True) if d > 0)
        lose = ", ".join(f"`{a}` {signed(d)}" for d, a in sorted(arms) if d < 0)
        tie = ", ".join(f"`{a}`" for d, a in sorted(arms, key=lambda t: t[1]) if d == 0)
        parts = []
        if beat:
            parts.append(f"**beats** `{ref}`: {beat}")
        if lose:
            parts.append(f"**loses** to `{ref}`: {lose}")
        if tie:
            parts.append(f"ties: {tie}")
        out.append(f"- `{ri.label}` on {MAIN_SCORER} vs `{ref}` — " + "; ".join(parts)
                   + f" — `{ri.scores_rel}`")
    if not out:
        out = [f"- no {MAIN_SCORER} comparison against the reference arm on this feature"]
    return out


def feature_section(feat: int, runs: list[RunIn], sae: str, ref_map: dict[str, str],
                    show_examples: bool) -> list[str]:
    """One feature: its covariates, the header line, then one block per (run, arm) present."""
    meta: dict = {}
    for ri in runs:
        meta = {**(ri.feats.get(feat) or {}), **meta}
    gate = meta.get("gate")
    gate = float(gate) if gate is not None else None
    facts = [
        f"dictionary `{sae or '(unknown)'}`",
        f"stratum **{meta.get('stratum', '—')}**",
        f"corpus peak **{rate(meta.get('corpus_peak'))}**",
        f"gate {rate(gate)}",
        f"density {rate(meta.get('density'), 6)}",
        f"fire fraction {rate(meta.get('fire_fraction'), 6)}",
        f"row {meta.get('row', '—')}",
    ]
    d1 = meta.get("draw1") or {}
    if d1:
        facts.append(f"draw 1: {d1.get('n_pos')} pos / {d1.get('n_neg')} neg")
    if meta.get("shown_docs") is not None:
        facts.append(f"{meta['shown_docs']} distinct shown documents")
    if meta.get("n_dup_rollouts"):
        facts.append(f"{meta['n_dup_rollouts']} duplicate rollouts dropped")

    out = [f"## Feature {feat}", "", "- " + " · ".join(facts), *header_line(feat, runs, ref_map)]
    out += ["- *a per-feature difference carries no interval; the block-level contrast is "
            "`results/autointerp.py`'s table*", ""]
    for ri in runs:
        arms: list[str] = []
        for arm, _s, f in ri.scores:
            if f == feat and arm not in arms:
                arms.append(arm)
        # An arm the explain stage knows about but the scores do not is a REFUSED or empty
        # explanation, and it belongs on the feature's page more than a scored arm does.
        arms += [a for a, f in ri.expl if f == feat and a not in arms]
        for arm in arms:
            out += arm_block(ri, arm, feat, gate, show_examples)
    return out


def summary_table(runs: list[RunIn], feats: list[int]) -> list[list]:
    """arm x (mean per scorer, n, refusals), over the features ON THE PAGE.

    The mean is over the features that HAVE a number: a null `bal_acc` is dropped and counted in
    `no metric`, never imputed at chance. Refusals are counted from the explain stage, which is the
    only place they exist -- a refused (feature, arm) leaves no score row, so an arm's `n` here is
    already net of them and the two columns are read together.
    """
    rows: list[list] = []
    for ri in runs:
        for label, arm in [(ri.label, a) for _l, a in arms_present([ri])]:
            cells: list = [f"{label}/{arm}"]
            first = next((ri.scores[(arm, s, f)] for s in SCORER_ORDER for f in feats
                          if (arm, s, f) in ri.scores), None)
            cells.append((first or {}).get("role", "—"))
            cells.append((first or {}).get("n_examples", "—"))
            for scorer in SCORER_ORDER:
                vals = [ri.scores[(arm, scorer, f)].get("bal_acc") for f in feats
                        if (arm, scorer, f) in ri.scores]
                have = [float(v) for v in vals if v is not None]
                cells.append(R.num(sum(have) / len(have), PLACES) if have else "—")
                cells.append(f"{len(have)}/{len(vals)}" if vals else "—")
            refused = len(ri.refused.get(arm, set()) & set(feats))
            cells.append(refused if ri.expl else "—")
            cells.append(f"`{ri.scores_rel}`")
            rows.append(cells)
    return rows


def render(runs: list[RunIn], sae: str, feats: list[int], ref_map: dict[str, str],
           sort_map: dict[str, str], ref_name: str, sort_name: str, order: str,
           limit: int, n_all: int, show_examples: bool, argv: list[str],
           vol: R.Vol) -> str:
    """The whole page for one dictionary."""
    lines =[f"# Per-feature autointerp page — `{sae or 'unknown dictionary'}`", ""]
    lines += [
        f"- generated {time.strftime('%Y-%m-%d %H:%M')} by `{' '.join(argv)}`",
        f"- volume root `{vol.prefix or '/'}`, mirror `{vol.local}`",
        f"- reference arm `{ref_name}`, index sorted by `{sort_name}` − `{ref_name}` on "
        f"{MAIN_SCORER}; body order `{order}`",
        f"- {len(feats)} of {n_all} features on this page"
        + (f" (`--limit {limit}`, a prefix of the `{order}` order)" if limit and len(feats) < n_all
           else ""),
        f"- chance for every balanced accuracy below is {CHANCE}; a null `bal_acc` is an absent "
        f"measurement and is never imputed",
        "- **per-feature differences carry no interval.** They are two stored balanced accuracies "
        "subtracted. The paired bootstrap over features is `results/autointerp.py`'s contrast "
        "table, and one feature is not evidence for a method (spec §3: no anecdotal-wins "
        "subsection).",
        "- every number cites the file it was read from; the only arithmetic here is the "
        "arm-minus-reference difference and the means of the summary table, both marked derived",
    ]
    for ri in runs:
        lines.append(
            f"- run `{ri.label}` = `runs/{ri.run_dir}`, build "
            f"`{ri.build_rel or '(unresolved)'}`, set `{ri.build.get('set', '—')}`, base "
            f"`{ri.build.get('base', '—')}`, maemm `{ri.build.get('maemm', '—')}`, gate "
            f"{ri.build.get('gate', '—')}, n_examples {ri.build.get('n_examples', '—')}, corpus "
            f"prefix {ri.build.get('corpus_prefix_m', '—')}M, mark `{ri.build.get('mark', '—')}`, "
            f"positive source `{ri.build.get('positive_source', '—')}`")
    lines += [""]

    lines += ["## Summary — arm × mean balanced accuracy, n, refusals", "",
              "*Derived: the mean is over the features on this page that have a number; "
              "`n` is measured/covered. Refusals are from the explain stage, where a refused "
              "(feature, arm) leaves no score row at all.*", ""]
    lines += md_table(
        ["arm", "role", "n examples", f"{SCORER_ORDER[0]} mean", f"{SCORER_ORDER[0]} n",
         f"{SCORER_ORDER[1]} mean", f"{SCORER_ORDER[1]} n", "refusals", "source"],
        summary_table(runs, feats))

    lines += [f"## Index — features by `{sort_name}` − `{ref_name}` on {MAIN_SCORER}", "",
              "*Derived, no interval. A feature with no delta (either side unmeasured) sorts "
              "last.*", ""]
    idx_rows = []
    for f in sorted(feats, key=lambda x: (feature_delta(x, runs, sort_map, ref_map) is None,
                                          -(feature_delta(x, runs, sort_map, ref_map) or 0.0), x)):
        meta = next((ri.feats[f] for ri in runs if f in ri.feats), {})
        ri0 = next((ri for ri in runs if sort_map.get(ri.label) and ref_map.get(ri.label)), None)
        ref_v = sort_v = None
        if ri0 is not None:
            ref_v = (ri0.scores.get((ref_map[ri0.label], MAIN_SCORER, f)) or {}).get("bal_acc")
            sort_v = (ri0.scores.get((sort_map[ri0.label], MAIN_SCORER, f)) or {}).get("bal_acc")
        idx_rows.append([f, meta.get("stratum", "—"), rate(meta.get("corpus_peak")),
                         rate(ref_v), rate(sort_v),
                         signed(feature_delta(f, runs, sort_map, ref_map))])
    lines += md_table(["feature", "stratum", "corpus peak", f"{ref_name} {MAIN_SCORER}",
                       f"{sort_name} {MAIN_SCORER}", "Δ"], idx_rows)

    for f in feats:
        lines += feature_section(f, runs, sae, ref_map, show_examples)

    notes = [n for ri in runs for n in ri.notes]
    if vol.missing:
        notes.append(f"{len(vol.missing)} files were not in the mirror or on the volume, first: "
                     + ", ".join(f"`{m}`" for m in vol.missing[:5]))
    if notes:
        lines += ["## Absent and skipped", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    run: Annotated[list[str] | None, typer.Option(
        "--run", help="`<label>=<run dir>` under `runs/`, repeated; one per checkpoint")] = None,
    build: Annotated[list[str] | None, typer.Option(
        "--build", help="`<label>=<build dir>` (volume-relative), for a run whose summary README "
                        "does not name the build it read")] = None,
    ref: Annotated[str, typer.Option(help="the arm every header line is read against, `<arm>` or "
                                          "`<run label>/<arm>`")] = "C16",
    sort_arm: Annotated[str, typer.Option(help="the arm the index is sorted by, minus --ref")]
    = "M-top16",
    features: Annotated[str, typer.Option(help="only these feature ids, comma-separated")] = "",
    limit: Annotated[int, typer.Option(help="keep only the first N features of the body order "
                                            "(the short appendix version); 0 keeps all")] = 0,
    order: Annotated[str, typer.Option(help="body order: `stratum` (default) or `delta`")]
    = "stratum",
    examples: Annotated[bool, typer.Option(help="render the shown example blocks")] = True,
    out: Annotated[Path, typer.Option(help="output directory; one `<dictionary>.md` per dictionary")]
    = R.HERE / "out" / "feature_page",
    root: Annotated[str, typer.Option(help="volume-relative root the runs were written under")] = "",
    data: Annotated[Path | None, typer.Option(help="local mirror (default results/data/<root>)")] = None,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what the mirror already has")] = False,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
) -> None:
    runs_spec = A.parse_runs(run)
    assert runs_spec, (
        "at least one `--run <label>=<run dir>` is required: the page is built from a run "
        "directory's scores and explanations and the build directory that run read")
    builds = A.parse_runs(build)
    unknown = [k for k in builds if k not in runs_spec]
    assert not unknown, (f"--build label(s) {unknown} name no --run "
                         f"({', '.join(runs_spec) or 'none'})")
    assert order in ("stratum", "delta"), f"--order is `stratum` or `delta`, not {order!r}"
    mirror = data or (R.HERE / "data" / (root.replace("/", "_") or "vol"))
    vol = R.Vol(root, mirror, modal_cmd, refetch, quiet, offline=not fetch)

    loaded = [load_run(vol, lab, d, builds.get(lab, "")) for lab, d in runs_spec.items()]
    usable = [ri for ri in loaded if ri.scores]
    assert usable, ("no run directory had a readable `summary/scores.jsonl`: "
                    + "; ".join(n for ri in loaded for n in ri.notes))

    want = {int(x) for x in features.replace(",", " ").split()} if features else None
    # ONE PAGE PER DICTIONARY. Runs on two dictionaries are two pages, never one file with both:
    # the strata are each dictionary's own rarity ordering and are not the same rarity, so a
    # feature id means nothing without its dictionary.
    by_sae: dict[str, list[RunIn]] = {}
    for ri in usable:
        by_sae.setdefault(ri.sae, []).append(ri)

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sae, group in by_sae.items():
        all_feats = sorted({f for ri in group for _a, _s, f in ri.scores})
        feats = [f for f in all_feats if want is None or f in want]
        if want:
            absent = sorted(want - set(all_feats))
            if absent:
                group[0].notes.append(
                    f"`--features` named {len(absent)} feature(s) this dictionary's runs do not "
                    f"score: {', '.join(str(a) for a in absent)}")
        ref_map = resolve_arm(group, ref)
        sort_map = resolve_arm(group, sort_arm)
        names = sorted({a for ri in group for a, _s, _f in ri.scores})
        if not ref_map:
            group[0].notes.append(f"`--ref {ref}` is in none of this dictionary's runs (arms: "
                                  f"{', '.join(names)}); no header line is written")
        if not sort_map:
            group[0].notes.append(f"`--sort-arm {sort_arm}` is in none of this dictionary's runs "
                                  f"(arms: {', '.join(names)}); the index is unsorted")
        ordered = order_features(feats, group, sort_map, ref_map, order)
        kept = ordered[:limit] if limit else ordered
        for ri in group:
            load_feature_blocks(vol, ri, kept)
        page = render(group, sae, kept, ref_map, sort_map, ref, sort_arm, order, limit,
                      len(all_feats), examples, sys.argv, vol)
        path = out / f"{(sae or 'unknown').replace('/', '_')}.md"
        path.write_text(page)
        written.append(path)
        print(f"[page] `{sae}`: {len(kept)} of {len(all_feats)} features, "
              f"{len(arms_present(group))} arms -> {path} ({path.stat().st_size / 1e3:.1f} kB)",
              flush=True)
    if vol.missing:
        print(f"[page] {len(vol.missing)} files absent, first: {vol.missing[:3]}", flush=True)
    print(f"[page] wrote {len(written)} file(s) under {out}", flush=True)


if __name__ == "__main__":
    app()
