"""Stage `report` (methodology §7): figures 01-03, the tables, report.md and the rule-picked examples, as a
pure function of the run directory (no model or judge call), so a re-render comes out the same."""

import json, math, time
import numpy as np
from evals.downstream.workspace_understanding import analysis as A
from evals.downstream.common.nla.nla_reader import PINS as NLA
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common import matcher as M
from evals.downstream.workspace_understanding import readouts as RO
from evals.downstream.workspace_understanding import paper_tables as PT
from evals.downstream.workspace_understanding import shards as S
from evals.downstream.common import retrieval as RT
from evals.downstream.common.figures import SMOKE_BANNER, err as _err, is_unavailable as _is_unavailable
from evals.downstream.common.figures import mpl as _mpl, one as _one, save as _save, sel as _sel, suptitle as _suptitle
from evals.downstream.common.judge_client import judge_log, served_line, unasked_detail
from evals.downstream.common.runs import config_hash, json_safe, mark_stage, read_provenance, stage_done
from evals.downstream.workspace_understanding.runs import stage_key

# The palette workspace_modulation uses (config.COLOURS there), so one reader has one colour in both.
COLORS = {"maemm": "#0072B2", "jlens": "#E69F00", "patch": "#D55E00", "control": "#777777",
          "nla": "#CC79A7", "retrieval": "#009E73"}
PANELS = ("association", "multihop", "pooled")
JUDGES = list(A.JUDGE_ORDER)


# One line style per chance line.
_CHANCE_LS = {
    "maemm": ":",
    "patch42": (0, (6, 1, 1, 1)),
    "nla": (0, (5, 2)),
    "nla64": (0, (5, 2, 1, 2)),
    "retrieval": (0, (1, 1)),
}

# --- figures ----------------------------------------------------------------------------------------------

_BUDGET_CURVES = (
    ("maemm", COLORS["maemm"], "-", "MAEMM"),
    ("patch42", COLORS["patch"], "-.", "Patchscopes, layer 42"),
    # the same prompt with nothing patched: one sample set shared by every item, drawn as a control
    ("patchfloor", COLORS["control"], "-", "Patchscopes floor, no injection"),
    ("nla", COLORS["nla"], "-", f"NLA verbalizer, native (≤ {NLA.max_new} tokens)"),
    ("nla64", COLORS["nla"], "--", "NLA verbalizer, first 64 tokens"),
    # deterministic: its N is the first N windows by rank, and it draws no greedy marker (`gr` is absent)
    ("retrieval", COLORS["retrieval"], "-", "Corpus search, top 8 windows of up to 64 tokens"),
)


def fig_budget_curves(T, out_dir, smoke_n=None):
    plt = _mpl()
    R = T["rates"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), constrained_layout=True)
    last = len(PANELS) - 1
    for c, g in enumerate(PANELS):
        ax = axes[0, c]
        for cond, color, ls, label in _BUDGET_CURVES:
            pts = [_one(R, metric="pass_at_n", condition=cond, group=g, budget=n) for n in C.PASS_AT_N]
            gr = _one(R, metric="greedy_hit", condition=cond, group=g)
            if not all(pts):
                continue
            xs = list(range(1, len(pts) + 1))
            ax.errorbar(
                xs, [p["estimate"] for p in pts], yerr=np.array([_err(p) for p in pts]).T,
                color=color, ls=ls, marker="o", ms=3, capsize=2, label=label,
            )
            if gr:
                ax.errorbar(
                    [0], [gr["estimate"]], yerr=np.array(_err(gr)).reshape(2, 1),
                    color=color, marker="s", ms=4, ls="none", capsize=2,
                )
            ch = _one(R, metric="chance", condition=cond, group=g)
            # the floor's chance line is its own curve's last point, so it draws none
            if cond != C.PATCH_FLOOR and ch and not _is_unavailable(ch["estimate"]):
                ax.axhline(float(ch["estimate"]), color=COLORS["control"], ls=_CHANCE_LS[cond], lw=1, label=f"{label} chance")
            if cond == "maemm":
                ax.plot([4], [pts[3]["estimate"]], marker="o", ms=8, mfc="none", color=color)
        n = _one(R, metric="pass_at_n", condition="maemm", group=g, budget=8)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["greedy", "1", "2", "4", "8"])
        ax.set_xlabel("rollout samples N")
        ax.set_ylabel("share of items with a whole-word hit")
        ax.set_ylim(0, 1)
        ax.set_title(f"{g}, n = {n['n_valid'] if n else 0}")
        # one legend per row, outside the rightmost panel
        if c == last:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, borderaxespad=0.0)
        ax = axes[1, c]
        for where, ls, label in (
            ("L42", "-", "J-lens, layer 42"),
            ("best", "--", "J-lens, best of the fitted layers (target-informed)"),
        ):
            pts = [_one(R, metric="rank_le_k", condition="jlens_" + where, group=g, budget=k) for k in C.RANK_KS]
            if not all(pts):
                continue
            ax.errorbar(
                list(range(len(pts))), [p["estimate"] for p in pts], yerr=np.array([_err(p) for p in pts]).T,
                color=COLORS["jlens"], ls=ls, marker="o", ms=3, capsize=2, label=label,
            )
            if where == "L42":
                ax.plot([2], [pts[2]["estimate"]], marker="o", ms=8, mfc="none", color=COLORS["jlens"])
            ch = _one(R, metric="chance", condition="jlens_" + where, group=g)
            if ch and not _is_unavailable(ch["estimate"]):
                ax.axhline(
                    float(ch["estimate"]), color=COLORS["control"], ls=":" if where == "L42" else "-.", lw=1,
                    label=f"{label} chance",
                )
        mt = _one(R, metric="rank_le_k", condition="jlens_L42", group=g, budget=10)
        ax.set_xticks(range(len(C.RANK_KS)))
        ax.set_xticklabels([str(k) for k in C.RANK_KS])
        ax.set_xlabel("lens rank cutoff k")
        ax.set_ylabel("share of items with target rank ≤ k")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{g}, n = {mt['n_valid'] if mt else 0}\n({mt.get('n_multi_token', 0) if mt else 0} lens-unrepresentable)",
            fontsize=8,
        )
        if c == last:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, borderaxespad=0.0)
    fig.suptitle(
        _suptitle(
            "Figure 1. Output-budget curves: free text (top) vs the Jacobian lens (bottom)\n"
            "Filled rings mark the headline pairing, pass@8 vs rank ≤ 10",
            smoke_n,
        )
    )
    _save(fig, out_dir, "01_budget_curves")
    return (
        "Figure 1. Top row: share of items on which a target form occurs as a whole word in at least one of "
        "the first N samples (greedy shown separately at the left tick), for MAEMM, Patchscopes (layer 42) "
        "and its no-injection floor (grey, solid: one sample set shared by every item), "
        "both NLA budgets and the corpus search, whose N is its first N windows by rank and which has no "
        "greedy marker; the other grey lines are each reader's 20-donor target-shuffle chance line. The "
        "untrained-base ablation is not a method and is not plotted here; its rows are in the report's own "
        "ablation section. Bottom row: share of items whose "
        "target ranks ≤ k in the lens's full vocabulary at layer 42 and at the best of the fitted layers; the "
        "best-layer curve is a target-informed selection. The two rows have different x axes and budgets and "
        "are not overlaid for that reason. Wilson 95 % intervals; population: all audit-clean items of the family."
    )


def fig_layer_curve(T, out_dir, smoke_n=None):
    plt = _mpl()
    LC, R = T["layer_curve"], T["rates"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    last = len(PANELS) - 1
    for c, (ax, g) in enumerate(zip(axes, PANELS)):
        rows = sorted(_sel(LC, group=g), key=lambda r: int(r["layer"]))
        if not rows:
            continue
        xs = [int(r["layer"]) for r in rows]
        ax.fill_between(xs, [r["ci_lower"] for r in rows], [r["ci_upper"] for r in rows], color=COLORS["jlens"], alpha=0.2)
        ax.plot(xs, [r["estimate"] for r in rows], color=COLORS["jlens"], label="J-lens rank ≤ 10 at this layer")
        best = _one(R, metric="rank_le_k", condition="jlens_best", group=g, budget=10)
        if best and not _is_unavailable(best["estimate"]):
            ax.axhline(
                best["estimate"], color=COLORS["jlens"], ls=":",
                label="best fitted layer per item (target-informed envelope)",
            )
        for metric, budget, ls, label in (
            ("pass_at_n", 8, "-", "MAEMM pass@8, read at layer 42 only"),
            ("greedy_hit", 0, "--", "MAEMM greedy, read at layer 42 only"),
        ):
            m = _one(R, metric=metric, condition="maemm", group=g, budget=budget)
            if m and not _is_unavailable(m["estimate"]):
                ax.axhline(m["estimate"], color=COLORS["maemm"], ls=ls, label=label)
                ax.errorbar(
                    [C.READ_LAYER], [m["estimate"]], yerr=np.array(_err(m)).reshape(2, 1),
                    color=COLORS["maemm"], capsize=3, ls="none",
                )
        ax.axvline(C.READ_LAYER, color="black", lw=0.8, ls="-.")
        ax.text(C.READ_LAYER + 0.5, 0.03, "inverter read layer", fontsize=7, va="bottom")
        mt = _one(R, metric="rank_le_k", condition="jlens_L42", group=g, budget=10)
        ax.set_xlabel("fitted lens layer")
        ax.set_ylabel("share of items, rank ≤ 10")
        ax.set_ylim(0, 1)
        ax.set_title(f"{g}, n = {rows[0]['n_valid']} ({mt.get('n_multi_token', 0) if mt else 0} multi-token excluded)")
        if c == last:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, borderaxespad=0.0)
    fig.suptitle(
        _suptitle("Figure 2. Where in the stack the lens finds the target; MAEMM references come from layer 42 only", smoke_n)
    )
    _save(fig, out_dir, "02_layer_curve")
    return (
        "Figure 2. Share of items whose target ranks ≤ 10 in the lens output at each fitted layer (orange line, "
        "Wilson 95 % band); the dotted orange line is the per-item best-layer rate, a target-informed "
        "selection and the envelope of the curve, which can exceed every per-layer value. Blue horizontal "
        "lines are MAEMM pass@8 and greedy hit rates read at layer 42 only; they are references, not "
        "curves. An item whose target has no single-token form "
        "has no rank at any layer: its cell is undefined here and it leaves the per-layer rate (the panel title "
        "gives the count), whereas in figure 1 and in the contrasts the same item counts as a lens miss. A lens "
        "peak away from layer 42 is a fact about the lens, not about where the inverter could read."
    )


JUDGED_GROUPS = [
    ("maemm_n8", "MAEMM, 8 samples", COLORS["maemm"], ""),
    ("jlens_L42_summary", "J-lens top-10 at layer 42, summarised", COLORS["jlens"], ""),
    ("jlens_band8_summary", "J-lens top-10 pooled over layers 36-50 (step 2), rank-ordered", COLORS["jlens"], "xx"),
    ("patch42_n8", "Patchscopes layer 42, 8 samples", COLORS["patch"], "xx"),
    ("nla_n8", "NLA verbalizer, native, 8 samples", COLORS["nla"], ""),
    ("retrieval_n8", "Corpus search, top 8 windows", COLORS["retrieval"], ""),
]


def fig_judged(T, out_dir, smoke_n=None):
    """Named, foil and net per judged condition, one row per judge."""
    plt = _mpl()
    J = T["judged"]
    conds = JUDGED_GROUPS
    fig, axes = plt.subplots(len(JUDGES), 3, figsize=(14, 4.2 * len(JUDGES)), squeeze=False)
    for r, judge in enumerate(JUDGES):
        for ax, g in zip(axes[r], PANELS):
            net_los = []
            for k, (cond, label, color, hatch) in enumerate(conds):
                named = _one(J, metric="named", condition=cond, group=g, judge=judge)
                foil = _one(J, metric="foil", condition=cond, group=g, judge=judge)
                net = _one(J, metric="net", condition=cond, group=g, judge=judge)
                if not named:
                    continue
                x = k * 3.0
                # error bars drawn separately: bar(yerr=) mis-handles a length-1 asymmetric yerr
                if not _is_unavailable(named["estimate"]):
                    ax.bar([x], [named["estimate"]], color=color, hatch=hatch, width=0.8)
                    ax.errorbar([x], [named["estimate"]], yerr=np.array(_err(named)).reshape(2, 1), color="black", fmt="none", capsize=2)
                else:
                    ax.bar(x, 0, fill=False, edgecolor=color, width=0.8)
                    ax.text(x, 0.02, "unavailable", rotation=90, fontsize=6)
                if foil and not _is_unavailable(foil["estimate"]):
                    ax.bar([x + 0.9], [foil["estimate"]], color=color, alpha=0.4, hatch="xx", width=0.8)
                    ax.errorbar([x + 0.9], [foil["estimate"]], yerr=np.array(_err(foil)).reshape(2, 1), color="black", fmt="none", capsize=2)
                else:
                    ax.bar(x + 0.9, 0, fill=False, edgecolor=COLORS["control"], width=0.8)
                    ax.text(x + 0.9, 0.02, "unavailable", rotation=90, fontsize=6)
                if net and not _is_unavailable(net["estimate"]):
                    ax.errorbar(
                        [x + 1.8], [net["estimate"]], yerr=np.array(_err(net)).reshape(2, 1),
                        color="black", marker="D", ms=4, capsize=2,
                    )
                    if not _is_unavailable(net.get("ci_lower")):
                        net_los.append(float(net["ci_lower"]))
                ax.text(x + 0.9, -0.08, f"n = {named['n_valid']}", ha="center", fontsize=6, transform=ax.get_xaxis_transform())
            # a net can be negative, so the lower limit follows the data
            ax.set_ylim((min(0.0, min(net_los)) - 0.05) if net_los else 0.0, 1.0)
            ax.axhline(0.0, color="black", lw=0.6, ls="-", zorder=0)
            ax.set_xticks([k * 3.0 + 0.9 for k in range(len(conds))])
            ax.set_xticklabels([c[1] for c in conds], rotation=40, ha="right", fontsize=5.5)
            ax.set_ylabel("share of items")
            ax.set_title(f"{g} — {C.JUDGES[judge].label}")
    fig.suptitle(_suptitle("Figure 3. Judged naming per judge: named (solid), foil (hatched, faded), net (black diamond)", smoke_n))
    fig.tight_layout()
    _save(fig, out_dir, "03_judged")
    return (
        "Figure 3. Share of items on which a judge finds a target named with a verified verbatim quote "
        "(solid), the same readout judged against the foil item's targets (hatched), and their difference "
        "(diamond, paired item-bootstrap 95 %). Lens readouts are prose summaries of token lists written by an "
        "item-blind model. n under each group is the number of items with a valid judgment."
    )


# --- formatting -------------------------------------------------------------------------------------------


def _sign_phrase(r):
    if r is None or _is_unavailable(r.get("estimate")):
        # zero complete pairs reads as unavailable, never as a zero effect
        n_pairs = r.get("n_pairs") if r else None
        return f"unavailable ({n_pairs if n_pairs is not None else 0} pairs)"
    est = float(r["estimate"])
    lo, hi = r.get("ci_lower"), r.get("ci_upper")
    sign = "positive" if est > 0 else "negative" if est < 0 else "zero"
    if _is_unavailable(lo) or _is_unavailable(hi):
        return f"{sign} ({est:+.3f}, interval unavailable)"
    return f"{sign} ({est:+.3f}, 95% item-bootstrap CI [{float(lo):+.3f}, {float(hi):+.3f}])"


def _fmt_rate(r):
    if r is None:
        return "unavailable"
    if _is_unavailable(r.get("estimate")):
        return f"unavailable (n = {r['n_valid']}/{r['n_total']})"
    return f"{float(r['estimate']):.3f} [{float(r['ci_lower']):.3f}, {float(r['ci_upper']):.3f}] (n = {r['n_valid']}/{r['n_total']})"


def _fmt_chance(r):
    """A chance-line cell: 0 reads "no chance hits", NaN or empty "unavailable"."""
    if r is None:
        return "unavailable"
    est = r.get("estimate")
    if _is_unavailable(est):
        return "unavailable"
    if float(est) == 0.0:
        return "no chance hits"
    return f"{float(est):.3f} [{float(r['ci_lower']):.3f}, {float(r['ci_upper']):.3f}] (n = {r['n_valid']}/{r['n_total']})"


def _fmt3(v):
    """Three decimals; NaN or empty -> "unavailable"."""
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    return f"{float(v):.3f}"


def _fmt_ratio(v):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    return f"{float(v):.2f}"


def _fmt_usd(v):
    """US$ to four decimals; "unavailable", NaN or None -> "unavailable"."""
    if v is None or v == "unavailable" or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    return f"${float(v):.4f}"


def _md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


REREAD_WINDOW_NOTE = (
    f"The scorer re-encodes at most the first {C.REREAD_WINDOW_TOKENS} tokens of a readout and takes the "
    "maximum cosine over them. MAEMM, `nla64` and the Patchscopes arms write at most 64 tokens and are scored "
    f"whole in that window; the verbalizer's native generation runs to {NLA.max_new} tokens and is re-read "
    f"whole in a {NLA.score_max_length}-token window, so its `nla` row is a maximum over more tokens and "
    "is not length-matched to MAEMM. `nla64` is the length-matched row of this table, and the "
    "`reread_own_maemm − reread_own_nla64` contrast is the one that compares two texts of one budget in one "
    "window. What is scored for the verbalizer is its whole generation, `<explanation>` tags included; the "
    "body between the tags is what the judges and the word rule read. "
    "`patchfloor` is the Patchscopes prompt continued with nothing patched, the same eight texts for every "
    "item: its own and foil cosines are what fluent text of that prompt reaches against any direction, and "
    "a Patchscopes arm's cosine is read against it (`reread_own_patch42 − reread_own_patchfloor`)."
)
REREAD_TRAINING_REWARD_NOTE = (
    "The re-read cosine in this table is MAEMM's own training reward (the rollout scored by re-reading its "
    "clean layer-42 activation against the injected direction), so MAEMM is the only reader here measured on "
    "the quantity it was optimised for; every other reader is scored on a metric it never saw. Read the "
    "column as what each text carries, not as a fair contest."
)
PATCH_FLOOR_NOTE = (
    "The `patch42` row is a zero-shot baseline: the clean base continues the entity-description "
    f"prompt with its placeholder's residual replaced, at the output of block {C.PATCH_LAYERS['patch42']}, "
    f"by {C.PATCH_ALPHA:g} × its own norm along `unit(h₄₂ − mu)`, the vector MAEMM is given. The floor row "
    "is the same prompt with nothing patched: eight continuations generated once and carried by every "
    "item, so its chance line is its own rate. Patchscopes reads the activation only by what it scores "
    "above the floor (the `patch42_pass8 − patchfloor_pass8` contrast), never by its rate alone "
    "(methodology §3.3)."
)
LENS_POOL_NOTE = (
    "`jlens_band8_summary` pools the top-10 lists of layers 36, 38, ..., 50, a band fixed in advance for "
    "every item, ordered by each token's best list position (methodology §3.2)."
)


# --- report sections --------------------------------------------------------------------------------------


def _code_line(run):
    """The commit the run was started at and whether its tree was dirty."""
    prov = read_provenance(run)
    commit = prov.get("git_commit") or "unavailable"
    branch = prov.get("git_branch") or "unavailable"
    dirty = prov.get("git_dirty")
    state = "dirty working tree" if dirty else "clean working tree" if dirty is False else "tree state unrecorded"
    return f"- code: commit `{commit}` on `{branch}` ({state})"


def _sec_header(run, scores, smoke):
    config_doc = run.read_json("config.json") if run.exists("config.json") else None
    # digest of the config and args blocks, not the file (which carries timestamps)
    config_digest = config_hash({k: config_doc.get(k) for k in ("config", "args")}) if config_doc else None
    out = []
    if smoke:
        out.append(f"# {SMOKE_BANNER.format(n=len(scores))}\n")
    out.append("# Workspace understanding: report\n")
    out.append(
        "**Question.** Given the layer-42 residual at one token of a prompt, does a MAEMM rollout name "
        "content the model represents but has not written, above a 20-donor target-shuffle chance line and beside the "
        "released Jacobian lens, a Patchscopes decoder, an activation verbalizer and a search of the "
        "shared web corpus, at the same activation and budget? (methodology §1)\n"
    )
    out.append("## Run configuration\n")
    out.append(_code_line(run))
    out.append(f"- base model: `{C.MODEL}` @ `{C.MODEL_REVISION}`, read at layer {C.READ_LAYER}")
    out.append(f"- inverter: `{C.INVERTER}` @ `{C.INVERTER_REVISION}` (generates; every read is on the base)")
    out.append(f"- lens: `{C.LENS_REPO}` @ `{C.LENS_REVISION}`, file `{C.LENS_FILE}`")
    out.append(f"- NLA verbalizer: `{NLA.repo}` @ `{NLA.revision}`")
    out.append(
        f"- search corpus: the held-out corpus's search prefix, `{C.CORPUS.dataset['dataset']}` (split "
        f"`{C.CORPUS.dataset['split']}`) @ `{C.CORPUS.dataset['revision']}`, {C.CORPUS.corpus_tokens:,} "
        f"tokens in windows of up to {C.CORPUS.window} tokens at stride {C.CORPUS.stride}, top "
        f"{C.CORPUS.top_k} per item"
    )
    out.append(f"- judge profile: `{C.JUDGE_PROFILE}`")
    out.append("- judge: " + "; ".join(f"`{j}` = {C.JUDGES[j].label} (`{C.JUDGES[j].model}`)" for j in JUDGES))
    out.append(f"- {served_line(run.path, {j: C.JUDGES[j] for j in JUDGES})}")
    out.append(
        f"- summariser: {C.JUDGES[C.SUMMARISER].label} (`{C.JUDGES[C.SUMMARISER].model}`), the judge's own "
        "model, so on the lens conditions the judge reads prose its model wrote (methodology §5.3)"
    )
    out.append(
        f"- scoring version: `{C.SCORING_VERSION}`; bootstrap seed {C.BOOTSTRAP_SEED}, {C.N_BOOT} resamples; "
        f"generation seed {C.GEN_SEED}"
    )
    out.append(
        f"- config digest: `{config_digest}`"
        if config_digest
        else "- config digest: unavailable (no config.json in this run directory)"
    )
    out.append("")
    return out


def _sec_population(scores, cov):
    pop, rc = cov["population"], cov["readout_coverage"]
    n_cells = len(scores) * len(C.JUDGED_CONDITIONS)
    out = [
        "## Population and coverage\n",
        f"- items kept: {pop['kept']} ({pop['by_family']}); excluded by reason: {pop['excluded'] or 'none'}; "
        f"multi-token lens targets: {pop['multi_token_lens_targets']}",
        f"- readout coverage (of {pop['kept']} kept items): "
        + ", ".join(f"{k} {v}" for k, v in rc.items()),
    ]
    for j in JUDGES:
        jc = cov["judges"][j]
        out.append(
            f"- judge `{j}` ({jc['label']}): {jc['unavailable']} unavailable, {jc['voided']} voided of {n_cells} "
            "item x condition judged cells (each cell is two requests, own and foil)"
        )
    nla = cov.get("nla_reader")
    if isinstance(nla, dict):
        out.append(
            f"- NLA reader: revision `{nla['revision']}`, {nla['n_items']} items over {len(nla['shards'])} shards; "
            f"share of samples whose `<explanation>` tags close {_fmt3(nla['close_rate'])} for `nla` (reference "
            f"{NLA.min_close_rate}, reported and never enforced) and {_fmt3(nla.get('close_rate_trunc'))} within "
            f"the {NLA.trunc}-token prefix `nla64` reads; a judge reads the body between the tags where they "
            f"close and the whole text otherwise; mean generated tokens {_fmt3(nla['mean_generated_tokens'])} of at "
            f"most {NLA.max_new}"
        )
    out.append(f"- GPU type: {cov['gpu_type']}")
    out.append("")
    return out


PRIMARY_ROWS = (
    ("maemm", "MAEMM, 8 samples"),
    ("nla", "NLA verbalizer, native"),
    ("nla64", "NLA verbalizer, 64-token budget"),
    ("patch42", "Patchscopes, layer 42"),
    ("patchfloor", "Patchscopes floor, no injection (control)"),
    ("retrieval", "Corpus search, top 8 windows of up to 64 tokens"),
    ("nla_mid", "NLA, mid-prompt token (control)"),
    ("nla_mean", "NLA, prompt mean (control)"),
    ("maemm_mid", "MAEMM, mid-prompt token (control)"),
    ("maemm_mean", "MAEMM, prompt mean (control)"),
)
# The ablations, printed in a section of their own (`_sec_ablations`).
ABLATION_ROWS = (("untrained_base", "the untrained base, MAEMM's prompt, marker and injection"),)
RATE_COLS = ["reader", "pass@1", "pass@8", "chance", "ratio_vs_chance", "flag_below_3x"]


def _rate_row(R, cond, label):
    """One reader's pass@8 row with its chance line, ratio and below-3x flag; raises when the reader has
    no pooled pass@8 (a row is never dropped)."""
    r = _one(R, metric="pass_at_n", condition=cond, group="pooled", budget=8)
    if not r:
        raise RuntimeError(f"report: no pooled pass@8 for reader {cond!r}; run its stage (a row is never dropped)")
    first = _one(R, metric="pass_at_n", condition=cond, group="pooled", budget=1)
    return {
        "reader": label,
        "pass@1": _fmt_rate(first) if first else "unavailable",
        "pass@8": _fmt_rate(r),
        "chance": _fmt_chance(_one(R, metric="chance", condition=cond, group="pooled")),
        "ratio_vs_chance": _fmt_ratio(r.get("ratio_vs_chance")),
        "flag_below_3x": r.get("flag_below_3x", "unavailable"),
    }


def _sec_primary(R):
    out = ["## Primary result\n"]
    out.append(
        "The word rule, pooled: the share of items on which a reader's eight samples contain a target form "
        "as a whole word, against that reader's own 20-donor target-shuffle chance line (methodology §6.1). `pass@1` is the "
        "first text alone: a sampled reader's first draw and, for the corpus search, its single best "
        "window. The lens rows follow the free-text readers: rank ≤ 10 at layer 42 and at the best fitted "
        "layer, then the word rule over its top-10 word-like tokens at layer 42 and over the pooled lists of "
        "layers 36-50 (step 2), the latter with its own 20-donor chance line (methodology §6.1).\n"
    )
    rows = [_rate_row(R, cond, label) for cond, label in PRIMARY_ROWS]
    for where, label in (
        ("L42", "J-lens, layer 42, rank ≤ 10"),
        ("best", "J-lens, best fitted layer, rank ≤ 10 (target-informed)"),
    ):
        r = _one(R, metric="rank_le_k", condition="jlens_" + where, group="pooled", budget=10)
        if not r:
            raise RuntimeError(f"report: no pooled rank≤10 for jlens_{where}; run the `lens` stage "
                               "(a row is never dropped)")
        rows.append(
            {
                "reader": label,
                "pass@8": _fmt_rate(r),
                "chance": _fmt_chance(_one(R, metric="chance", condition="jlens_" + where, group="pooled")),
                "ratio_vs_chance": _fmt_ratio(r.get("ratio_vs_chance")),
                "flag_below_3x": r.get("flag_below_3x", "unavailable"),
            }
        )
    l42 = _one(R, metric="word_top10", condition="jlens_L42", group="pooled", budget=C.TOP_WORD)
    if l42:
        rows.append({"reader": "J-lens, top-10 word-like tokens at layer 42, word rule", "pass@8": _fmt_rate(l42),
                     "chance": "unavailable", "ratio_vs_chance": "unavailable", "flag_below_3x": "unavailable"})
    band = _one(R, metric="word_top10", condition="jlens_" + C.LENS_BAND, group="pooled", budget=C.TOP_WORD)
    if band:
        rows.append(
            {
                "reader": "J-lens, top-10 word-like tokens of layers 36-50 (step 2) pooled, word rule",
                "pass@8": _fmt_rate(band),
                "chance": _fmt_chance(_one(R, metric="chance", condition="jlens_" + C.LENS_BAND, group="pooled")),
                "ratio_vs_chance": _fmt_ratio(band.get("ratio_vs_chance")),
                "flag_below_3x": band.get("flag_below_3x", "unavailable"),
            }
        )
    out.append(_md_table(rows, RATE_COLS))
    out.append("")
    out.append(
        "The corpus search's eight texts are the eight windows of the shared corpus whose own layer-42 "
        "residuals point most along the query by the re-read cosine below, and its pass@N is the first N "
        "of them by rank. It is "
        "deterministic — one query, one ranked list — so it has no greedy row at all, and its consistency "
        "row is the share of its eight windows that name the target rather than a spread over draws. It "
        "is the floor every activation-to-text reader has to clear: a reader that does not beat looking "
        f"the activation up in {C.CORPUS.corpus_tokens:,} tokens of ordinary web text is not reading it "
        "(methodology §3.6).\n"
    )
    out.append(PATCH_FLOOR_NOTE + "\n")
    mt = _one(R, metric="rank_le_k", condition="jlens_L42", group="pooled", budget=10)
    if mt:
        out.append(
            f"{mt.get('n_multi_token', 0)} of {mt['n_total']} items have a target with no single-token form; the "
            "lens cannot rank them, so they count as lens misses in both lens rows above and in every lens "
            "contrast, and their cell is undefined in the per-layer curve (figure 2).\n"
        )
    out.append("Per family, MAEMM alone:\n")
    fam_rows = []
    for g in PANELS:
        p8 = _one(R, metric="pass_at_n", condition="maemm", group=g, budget=8)
        fam_rows.append(
            {
                "group": g,
                "pass@8": _fmt_rate(p8),
                "greedy": _fmt_rate(_one(R, metric="greedy_hit", condition="maemm", group=g)),
                "chance": _fmt_chance(_one(R, metric="chance", condition="maemm", group=g)),
                "consistency": _fmt_rate(_one(R, metric="consistency", condition="maemm", group=g)),
            }
        )
    out.append(_md_table(fam_rows, ["group", "pass@8", "greedy", "chance", "consistency"]))
    out.append("")
    out.append("![Figure 1](figures/01_budget_curves.png)\n")
    return out


def _sec_ablations(run, R, CON):
    """The ablations of MAEMM: word-rule rows, the paired difference from MAEMM and the injection check."""
    out = ["## Ablations\n"]
    rows = [_rate_row(R, cond, label) for cond, label in ABLATION_ROWS]
    out.append(
        "MAEMM's own training prompt, marker position and centred unit direction at the readout position, "
        "generated by the untrained base model instead of the inverter, with MAEMM's generation settings "
        "and a seed offset of its own (methodology §3.7). Scored by the word rule alone: it is read by no "
        "judge, so it costs nothing against the judge budget and appears in no judged table. What it "
        "separates is how much of MAEMM's naming is the fine-tune and how much is the prompt plus an "
        "injected vector on a model never trained to read one.\n"
    )
    out.append(_md_table(rows, RATE_COLS))
    out.append("")
    for cond, label in ABLATION_ROWS:
        r = _one(CON, condition=f"maemm_pass8-{cond}_pass8", group="pooled")
        if r:
            out.append(f"- `maemm_pass8` − `{cond}_pass8`: {_sign_phrase(r)}")
        r = _one(CON, condition=f"{cond}_pass8-{cond}_chance", group="pooled")
        if r:
            out.append(f"- `{cond}_pass8` − `{cond}_chance`: {_sign_phrase(r)}")
    ic = (run.read_json(RO.UNTRAINED_BASE_REL).get("config") or {}).get("injection_check") \
        if run.exists(RO.UNTRAINED_BASE_REL) else None
    if ic:
        out.append(
            f"- injection check (recorded, never enforced): greedy distinct share "
            f"{_fmt3(ic.get('greedy_distinct_share'))}; greedy cos own {_fmt3(ic.get('greedy_cos_own_mean'))}, "
            f"foil {_fmt3(ic.get('greedy_cos_foil_mean'))}, gap {_fmt3(ic.get('gap'))}"
        )
    out.append("")
    out.append("Source tables: [tables/rates.csv](tables/rates.csv), [tables/contrasts.csv](tables/contrasts.csv).\n")
    return out


def _missingness_table(MI):
    """The missingness table: one row per judge, instrument and condition."""
    order = {name: k for k, name in enumerate(("naming", "diagnostic", "summary", "diagnostic_summary"))}
    rows = sorted(MI, key=lambda r: (JUDGES.index(r["judge"]) if r["judge"] in JUDGES else len(JUDGES),
                                     order.get(r["instrument"], len(order)), str(r["condition"])))
    cols = ["judge", "instrument", "condition", "n"] + list(A.MISSINGNESS_STATUSES) + ["voided"]
    return _md_table([{c: r.get(c, "") for c in cols} for r in rows], cols)


def _sec_judged(J, MI, captions):
    out = ["## Judged naming\n"]
    names = " and ".join(f"{C.JUDGES[j].label} (`{j}`)" for j in JUDGES)
    out.append(
        f"Judge: {names}. A positive verdict counts only when the judge also returns a verbatim quote found "
        "in the readout it was shown. `net` is named minus the same readout judged against a foil item's "
        "targets (methodology §6.2).\n"
    )
    out.append(
        "**How a verdict enters these rates.** A positive verdict whose target or quote does not check out "
        "against the readout is **voided** and counts as *not named*: it is in the denominator and not in "
        "the numerator. A verdict the judge did not return at all — refused, stopped by a content filter, "
        "unreadable as a verdict, or unavailable — leaves the rate entirely, numerator and denominator "
        "alike, and is counted only in the missingness table below.\n"
    )
    rows = []
    for cond, label, _c, _h in JUDGED_GROUPS:
        r = {"condition": f"`{cond}`"}
        present = False
        for j in JUDGES:
            named = _one(J, metric="named", condition=cond, group="pooled", judge=j)
            net = _one(J, metric="net", condition=cond, group="pooled", judge=j)
            if named:
                present = True
            r[f"named ({j})"] = _fmt_rate(named)
            r[f"net ({j})"] = _fmt_rate(net)
        if present:
            rows.append(r)
    cols = ["condition"] + [f"{m} ({j})" for j in JUDGES for m in ("named", "net")]
    out.append(_md_table(rows, cols))
    out.append("")
    out.append(LENS_POOL_NOTE + "\n")
    out.append(
        "Missingness, per judge, instrument and condition ([tables/missingness.csv](tables/missingness.csv)). "
        "`ok` is a request that settled into a verdict the tables read; `refused` is the judge declining in "
        "its own words, `content_filter` a provider-side stop, `parse_fail` a reply that is not a verdict "
        "even when re-read by the current parser, `unavailable` an empty reply, a transport failure or the "
        "budget. `voided` counts the ok naming verdicts whose target or quote did not check out — those are "
        "in the rate, as not named. The summariser's own calls are here too: a lens summary that never came "
        "back is a lens condition missing for the judge.\n"
    )
    out.append(_missingness_table(MI))
    out.append("")
    out.append("![Figure 3](figures/03_judged.png)\n")
    out.append(captions["03"] + "\n")
    out.append(
        "Source tables: [tables/judged.csv](tables/judged.csv), [tables/rates.csv](tables/rates.csv), "
        "[tables/missingness.csv](tables/missingness.csv).\n"
    )
    return out


def _sec_contrasts(CON):
    out = ["## Contrasts\n"]
    out.append(
        "Paired item differences, pooled group, sign convention MAEMM minus comparator; an item enters a "
        "contrast only when both sides have a value for it (methodology §6.3). Rows with an empty judge "
        "column do not depend on a judge.\n"
    )
    rows = []
    for r in _sel(CON, group="pooled"):
        rows.append(
            {
                "condition_a": r["condition_a"],
                "condition_b": r["condition_b"],
                "judge": r.get("judge", ""),
                "estimate": _fmt3(r["estimate"]),
                "ci_lower": _fmt3(r["ci_lower"]),
                "ci_upper": _fmt3(r["ci_upper"]),
                "n_pairs": r["n_pairs"],
            }
        )
    out.append(_md_table(rows, ["condition_a", "condition_b", "judge", "estimate", "ci_lower", "ci_upper", "n_pairs"]))
    out.append("")
    headline = [
        ("maemm_pass8", "maemm_chance", ""),
        ("maemm_pass8", "jlens_L42_k10", ""),
        ("maemm_pass8", "nla_pass8", ""),
        ("maemm_pass8", "nla64_pass8", ""),
        ("maemm_pass8", "retrieval_pass8", ""),
        ("maemm_pass8", "maemm_mid_pass8", ""),
        ("nla_pass8", "nla_mid_pass8", ""),
        ("retrieval_pass8", "retrieval_chance", ""),
    ]
    headline += [(f"{a}_pass8", f"{C.PATCH_FLOOR}_pass8", "") for a in C.PATCH_LAYERS]
    headline += [("net_maemm_n8", "net_nla_n8", j) for j in JUDGES]
    headline += [("net_maemm_n8", "net_retrieval_n8", j) for j in JUDGES]
    headline += [("net_maemm_n8", "net_jlens_L42_summary", j) for j in JUDGES]
    for a, b, j in headline:
        r = _one(CON, condition=f"{a}-{b}", group="pooled", judge=j)
        if r:
            suffix = f" [judge `{j}`]" if j else ""
            out.append(f"- `{a}` − `{b}`{suffix}: {_sign_phrase(r)}")
    out.append("")
    out.append("![Figure 2](figures/02_layer_curve.png)\n")
    out.append("Full table: [tables/contrasts.csv](tables/contrasts.csv).\n")
    return out


def _sec_reread(run, F, CON, cov):
    out = ["## Re-read fidelity\n"]
    if not F:
        raise RuntimeError("report: no re-read rows; run the `reread` stage (a section is never omitted)")
    out.append(
        "Each reader's text re-read on the clean base and scored against the item's own centred layer-42 "
        "direction (own) and the foil item's (foil), through one scorer (methodology §6.4). Pooled, over the "
        "eight samples per item.\n"
    )
    rows = []
    for cond in ("maemm",) + tuple(C.REREAD_CONDITIONS):
        o = _one(F, metric="reread_cos_own", condition=cond, group="pooled", budget_type="samples")
        if not o:
            continue
        rows.append(
            {
                "reader": cond,
                "cos own": _fmt_rate(o),
                "cos foil": _fmt_rate(_one(F, metric="reread_cos_foil", condition=cond, group="pooled", budget_type="samples")),
                "gap": _fmt_rate(_one(F, metric="reread_gap", condition=cond, group="pooled", budget_type="samples")),
                "n_items": cov["reread_coverage"].get(cond, ""),
            }
        )
    out.append(_md_table(rows, ["reader", "cos own", "cos foil", "gap", "n_items"]))
    out.append("")
    out.append(REREAD_WINDOW_NOTE + "\n")
    for cond in C.REREAD_CONDITIONS:
        r = _one(CON, condition=f"reread_own_maemm-reread_own_{cond}", group="pooled")
        if r:
            out.append(f"- re-read cosine, MAEMM − {cond}: {_sign_phrase(r)}")
    out.append("")
    out.append(REREAD_TRAINING_REWARD_NOTE + "\n")
    out.append("Source table: [tables/reread.csv](tables/reread.csv).\n")
    out += _search_cosine(run)
    return out


def _search_cosine(run):
    """The corpus search's own cosine at the shared size and the full corpus, top-1 beside top-8."""
    rows = (run.read_json(RO.RETRIEVAL_REL).get("config") or {}).get("search") \
        if run.exists(RO.RETRIEVAL_REL) else None
    out = ["### Corpus search cosine\n"]
    if not rows:
        return out + ["Unavailable (no `config.search` in `rollouts/retrieval.json`).\n"]
    return out + list(RT.search_table(rows)) + ["", RT.SEARCH_NOTE + " Every instrument above reads the "
                                                "full corpus's windows; the shared-size row is the "
                                                "search's cosine alone.\n"]


# --- examples appendix ------------------------------------------------------------------------------------


def _mark_readout_token(prompt, tok):
    tok = (tok or "").strip()
    if not tok:
        return prompt
    if prompt.endswith(tok) or prompt.endswith(" " + tok):
        return prompt[: -len(tok)] + "⟦" + tok + "⟧"
    return prompt + " ⟦" + tok + "⟧"


def examples(run, scores, per_cell=3):
    """Examples picked by a fixed rule: per family, the first `per_cell` items by id in each cell of MAEMM
    pass@8 against lens rank ≤ 10 at layer 42 (both / MAEMM only / lens only / neither)."""
    items_by_i = {x["i"]: x for x in run.read_json("data/items.json")["items"] if not x["excluded"]}
    free = RO.free_text(run)
    summaries = RO.lens_summaries(run)
    lens_by_i = {r["i"]: r for r in run.read_json("lens/lens.json")["items"]}
    diag = run.read_json("data/diagnostic.json") if run.exists("data/diagnostic.json") else {}
    quotes = {}
    for j in JUDGES:
        if not run.exists(judge_log(j, "naming")):
            continue
        for m, r in A.bound_records(run, j).values():
            if m.get("condition") == "maemm_n8" and m.get("vs") == "own":
                v = A.parse_or_none(r)
                quotes.setdefault(m["i"], {})[j] = (
                    "unavailable" if v["named"] is None else (v["quote"] or "not named") if v["named"] else "not named"
                )
    by_i = {s["i"]: s for s in scores}
    out = []
    for fam in C.FAMILIES:
        cells = {"both": [], "maemm_only": [], "lens_only": [], "neither": []}
        for s in sorted((s for s in scores if s["family"] == fam), key=lambda s: s["i"]):
            m_hit = bool(s.get("maemm") and s["maemm"]["pass_at"]["8"])
            l_hit = s["jlens"]["rank_le"]["L42"]["10"]
            key = "both" if (m_hit and l_hit) else "maemm_only" if m_hit else "lens_only" if l_hit else "neither"
            cells[key].append(s["i"])
        for cell in ("both", "maemm_only", "lens_only", "neither"):
            for i in cells[cell][:per_cell]:
                it, s, L = items_by_i[i], by_i[i], lens_by_i.get(i)
                m = (free.get("maemm") or {}).get(i)
                nla = (free.get("nla") or {}).get(i)
                patch = {a: (free.get(a) or {}).get(i) for a in C.PATCH_LAYERS}
                ret = (free.get("retrieval") or {}).get(i)
                fid = s.get("fidelity") or {}
                out.append(
                    {
                        "i": i,
                        "name": it["name"],
                        "family": fam,
                        "cell": cell,
                        "prompt": _mark_readout_token(it["prompt"], it.get("readout_token")),
                        "forms": it["forms"],
                        "diag": diag.get(str(i)) if fam == "multihop" else None,
                        "maemm_greedy": m["greedy"] if m else "unavailable",
                        "maemm_greedy_hit": (s["maemm"]["greedy_hit"] if s.get("maemm") else None),
                        "maemm_sample_1": m["samples"][0] if (m and m["samples"]) else "unavailable",
                        "maemm_sample_1_hit": (
                            M.whole_word_hit(m["samples"][0], it["forms"]) if (m and m["samples"]) else None
                        ),
                        "top10_L42": L["top10_L42"] if L else "unavailable",
                        "rank_L42": L["rank_L42"] if L else None,
                        "min_rank_layer": L["min_rank_layer"] if L else None,
                        "summary_L42": summaries.get("L42", {}).get(i),
                        "judge_quotes": quotes.get(i, {}),
                        "patch_greedy": {a: (r["greedy"] if r else None) for a, r in patch.items()},
                        "nla_sample_1": nla["samples"][0] if (nla and nla["samples"]) else None,
                        "nla_sample_1_hit": (
                            M.whole_word_hit(nla["samples"][0], it["forms"]) if (nla and nla["samples"]) else None
                        ),
                        "retrieval_top_window": ret["samples"][0] if (ret and ret["samples"]) else None,
                        "retrieval_top_window_hit": (
                            M.whole_word_hit(ret["samples"][0], it["forms"]) if (ret and ret["samples"]) else None
                        ),
                        "maemm_cos_own_samples": (fid.get("maemm") or {}).get("own_samples"),
                        "nla_cos_own_samples": (fid.get("nla") or {}).get("own_samples"),
                    }
                )
    return out


def _sec_examples(examples_list):
    out = ["## Examples\n"]
    out.append(
        "Selection by rule (methodology §7.2): per family, the 2x2 of MAEMM pass@8 against lens matched rank "
        "≤ 10 (both / MAEMM only / lens only / neither), first three items by id per cell. Illustrative; "
        "changes no score.\n"
    )
    for ex in examples_list:
        out.append(f"### item {ex['i']} — {ex['name']} ({ex['family']}, cell: {ex['cell']})\n")
        out.append(f"- prompt: {ex['prompt']}")
        out.append(f"- targets: {ex['forms']}" + (f"; diagnostic: {ex['diag']}" if ex["diag"] else ""))
        out.append(f"- MAEMM greedy (hit={ex['maemm_greedy_hit']}): {ex['maemm_greedy']!r}")
        out.append(f"- MAEMM sample 1 (hit={ex['maemm_sample_1_hit']}): {ex['maemm_sample_1']!r}")
        out.append(f"- lens top-10 at layer 42 (rank_L42={ex['rank_L42']}, best layer={ex['min_rank_layer']}): {ex['top10_L42']}")
        out.append(f"- matched-layer summary: {ex['summary_L42']!r}")
        for j, q in (ex.get("judge_quotes") or {}).items():
            out.append(f"- judge `{j}` quote (8-sample readout): {q!r}")
        for arm, text in (ex.get("patch_greedy") or {}).items():
            out.append(
                f"- Patchscopes greedy, `{arm}` (target layer {C.PATCH_LAYERS[arm]}, {C.PATCH_RULES[arm]}): {text!r}"
            )
        if ex.get("nla_sample_1") is not None:
            out.append(f"- NLA sample 1, native (hit={ex['nla_sample_1_hit']}): {ex['nla_sample_1']!r}")
        if ex.get("retrieval_top_window") is not None:
            out.append(
                f"- corpus search, top window (hit={ex['retrieval_top_window_hit']}): {ex['retrieval_top_window']!r}"
            )
        if ex.get("maemm_cos_own_samples") is not None or ex.get("nla_cos_own_samples") is not None:
            out.append(
                f"- re-read cosine over samples (own): MAEMM {_fmt3(ex.get('maemm_cos_own_samples'))}, "
                f"NLA {_fmt3(ex.get('nla_cos_own_samples'))}"
            )
        out.append("")
    return out


# --- diagnostics, limits, costs ---------------------------------------------------------------------------


COMPARATOR_CHECK_NOTE = (
    "The injection checks of the comparators, the position controls and the untrained-base ablation, and "
    "the corpus search's own numbers, are recorded, never enforced: a reader that decodes the same text for "
    "every activation is reporting its own result. MAEMM's own arm at the readout is enforced: the `rollouts` stage "
    "refuses to save a run whose greedy rollouts are less than 0.95 distinct or whose own − foil re-read "
    "gap is below 0.10, because every headline number is read off it. The Patchscopes patch check is "
    "enforced too, and is a check of the mechanism rather than of a result: it says the placeholder's "
    "residual was replaced by the scaled direction, not that anything was read from it."
)


def _comparator_checks(run):
    """The recorded injection checks of every arm but MAEMM's own at the readout."""
    out = []
    nla = S.load_nla(run)
    for sh in (nla or {}).get("shards") or []:
        ic = sh.get("injection_check") or {}
        out.append(f"- NLA verbalizer, `{sh['file']}`: greedy distinct share {_fmt3(ic.get('greedy_distinct_share'))}; "
                   f"tags closed in {_fmt3(ic.get('close_rate'))} of the samples "
                   f"({_fmt3(ic.get('close_rate_trunc'))} within the {NLA.trunc}-token prefix)")
    if not nla:
        out.append("- NLA verbalizer injection check: unavailable (no `rollouts/nla/`).")
    ctl = S.load_controls(run)
    for sh in (ctl or {}).get("shards") or []:
        for kind, c in (sh.get("injection_check") or {}).items():
            out.append(
                f"- NLA control `{kind}`, `{sh['file']}`: greedy distinct share "
                f"{_fmt3(c.get('greedy_distinct_share'))} of {_fmt3(c.get('distinct_input_share'))} distinct inputs; "
                f"tags closed in {_fmt3(c.get('close_rate'))} of the samples"
            )
    if not ctl:
        out.append("- NLA control injection check: unavailable (no `rollouts/nla_control/`).")
    mc = (
        (run.read_json("rollouts/maemm_control.json").get("config") or {}).get("injection_check")
        if run.exists("rollouts/maemm_control.json")
        else None
    )
    for kind, c in (mc or {}).items():
        out.append(
            f"- MAEMM control `{kind}`: greedy distinct share {_fmt3(c.get('greedy_distinct_share'))} of "
            f"{_fmt3(c.get('distinct_input_share'))} distinct inputs; greedy cos own "
            f"{_fmt3(c.get('greedy_cos_own_mean'))}, foil {_fmt3(c.get('greedy_cos_foil_mean'))}, gap "
            f"{_fmt3(c.get('gap'))}"
        )
    if not mc:
        out.append("- MAEMM control injection check: unavailable (no `rollouts/maemm_control.json`).")
    sc = (run.read_json(RO.RETRIEVAL_REL).get("config") or {}).get("search_check") \
        if run.exists(RO.RETRIEVAL_REL) else None
    if sc:
        out.append(
            f"- corpus search: {sc['n_windows']} windows over {sc['n_shards']} part(s); distinct top-1 "
            f"windows {sc['distinct_top1_windows']}/{sc['n_items']}; mean top-1 score "
            f"{_fmt3(sc.get('mean_top1_score'))}; "
            f"near-duplicates above {sc['near_duplicate_cos']}: {sc['near_duplicates']}; rows short of "
            f"top-{sc['top_k']}: {sc['rows_short_of_top_k']}; better-scoring windows passed over per item "
            f"because they overlap a kept one: {_fmt3(sc.get('mean_suppressed'))}"
        )
    else:
        out.append("- corpus search check: unavailable (no `rollouts/retrieval.json`).")
    return out


# The stages that re-read a text; each records what the norm filter would have dropped.
SCORING_STAGES = ("rollouts", "maemm_control", "untrained_base", "reread")


def _norm_filter_lines(run):
    """One line per scoring stage: how many content tokens the (unapplied) norm filter would have dropped."""
    out = [
        "- the re-read cosine is unfiltered: every content token behind the sink is a candidate for the "
        f"maximum. The scorer's norm filter (a token above {C.REREAD_NORM_FILTER_MULT:g}× its text's median "
        "norm is dropped) is not applied; what it would have dropped is counted per scoring stage:"
    ]
    for stage in SCORING_STAGES:
        rel = f"stages/{stage}.json"
        rec = (run.read_json(rel).get("norm_filter") if run.exists(rel) else None) or {}
        n, drops = rec.get("n_tokens"), rec.get("n_tokens_filter_drops")
        if not n:
            out.append(f"  - `{stage}`: unavailable (no `norm_filter` tally in `{rel}`).")
            continue
        out.append(
            f"  - `{stage}`: {drops} of {n} content tokens ({100.0 * drops / n:.3f} %), in "
            f"{rec.get('n_texts_filter_touches')} of {rec.get('n_texts')} texts"
        )
    return out


def _sec_diagnostics(run, DS, cov):
    out = ["## Diagnostics\n"]
    ic = (run.read_json("rollouts/maemm.json").get("config") or {}).get("injection_check")
    if ic:
        out.append(
            f"- injection check: greedy distinct share {_fmt3(ic.get('greedy_distinct_share'))}; greedy cos own "
            f"mean {_fmt3(ic.get('greedy_cos_own_mean'))}; greedy cos foil mean "
            f"{_fmt3(ic.get('greedy_cos_foil_mean'))}; gap (own − foil, must be ≥ 0.10) {_fmt3(ic.get('gap'))}"
        )
    else:
        out.append("- injection check: unavailable (no `config.injection_check` in `rollouts/maemm.json`).")
    patch_doc = run.read_json("rollouts/patchscope.json") if run.exists("rollouts/patchscope.json") else None
    if patch_doc:
        for arm, a in patch_doc["arms"].items():
            if a.get("target_layer") is None:
                out.append(
                    f"- Patchscopes {arm} (no hook installed): {a.get('distinct_sample_texts')} distinct sample "
                    "texts, the one shared set every item carries"
                )
                continue
            pc = a.get("patch_check") or {}
            out.append(
                f"- Patchscopes {arm} (target layer {a['target_layer']}, rule `{a.get('rule')}`, written "
                f"`{a.get('input')}`): share of patched greedies equal to "
                f"unpatched = {_fmt3(a.get('share_equal_unpatched'))}; greedy distinct share "
                f"{_fmt3(a.get('greedy_distinct_share'))}; patch check on {len(pc.get('items') or [])} items "
                f"(enforced): min cos(h_patched, v) {_fmt3(min(pc['cos_to_v']) if pc.get('cos_to_v') else None)}, "
                f"min ‖Δh‖/‖h‖ {_fmt3(min(pc['rel_delta']) if pc.get('rel_delta') else None)}, written norm "
                f"over clean norm {_fmt3(float(np.median(pc['norm_ratio'])) if pc.get('norm_ratio') else None)}"
            )
    else:
        out.append("- Patchscopes checks: unavailable.")
    out += _comparator_checks(run)
    lens_self = run.read_json("lens/lens.json").get("lens") if run.exists("lens/lens.json") else None
    if lens_self:
        fl = lens_self.get("fitted_layers") or []
        sha = lens_self.get("sha256") or ""
        out.append(
            f"- lens self-check: {len(fl)} fitted layers; layer 42 fitted: {C.READ_LAYER in fl}; "
            f"sha256 `{sha[:12] + '...' if sha else 'unavailable'}`"
        )
    else:
        out.append("- lens self-check: unavailable (no `lens/lens.json`).")
    if run.exists("rollouts/reread.json"):
        cfg = run.read_json("rollouts/reread.json").get("config") or {}
        out.append(
            f"- re-read self-check: max |re-read − saved| on MAEMM's greedies {_fmt3(cfg.get('selfcheck_max_abs_diff'))} "
            f"(tolerance {C.REREAD_SELFCHECK_TOL})"
        )
    out += _norm_filter_lines(run)
    summ_doc = run.read_json("judges/summaries.json") if run.exists("judges/summaries.json") else None
    if summ_doc:
        vals = [v for lst in (summ_doc.get("summaries") or {}).values() for v in lst.values()]
        out.append(f"- summariser failures: {sum(1 for v in vals if v.get('status') != 'ok')} of {len(vals)}")
    else:
        out.append("- summariser failures: unavailable (no `judges/summaries.json`).")
    for j in JUDGES:
        rel = judge_log(j, "diagnostics")
        if run.exists(rel):
            recs = run.read_jsonl(rel)
            out.append(
                f"- pipeline diagnostic, judge `{j}` (own/foil naming checks on synthetic readouts, run before "
                f"anything is spent on a real one): {len(recs)} requests, {sum(1 for r in recs if r.get('status') == 'ok')} ok"
            )
        jc = cov["judges"][j]
        out.append(
            f"- judge `{j}`: {jc['requests']} requests over {jc['records']} log records (a request two "
            f"cells share is answered once and recorded once per cell), {jc['repaired']} verdicts "
            "read from malformed JSON by the repair rule (the saved text is re-read by the current parser, so a "
            "run-time parse failure is not final)"
        )
    out.append(f"- multi-token lens misses: {cov['population']['multi_token_lens_targets']}")
    out.append("")
    out.append(COMPARATOR_CHECK_NOTE + "\n")
    out.append(
        "Multi-hop competence split, headline rows only — the full table (every metric rates.csv holds, "
        "restricted to each subset) is in [tables/diagnostic_split.csv](tables/diagnostic_split.csv):\n"
    )
    headline_ds = [
        r
        for r in DS
        if (r["metric"] == "pass_at_n" and str(r.get("budget")) == "8" and r["condition"] in ("maemm", "nla"))
        or (r["metric"] == "rank_le_k" and str(r.get("budget")) == "10" and r["condition"] in ("jlens_L42", "jlens_best"))
        or r["metric"] == "pass8_or_k10"
    ]
    out.append(
        _md_table(
            [
                {
                    "group": r["group"],
                    "metric": r["metric"],
                    "condition": r["condition"],
                    "estimate": _fmt3(r["estimate"]),
                    "ci_lower": _fmt3(r["ci_lower"]),
                    "ci_upper": _fmt3(r["ci_upper"]),
                    "n_valid": r["n_valid"],
                }
                for r in headline_ds
            ],
            ["group", "metric", "condition", "estimate", "ci_lower", "ci_upper", "n_valid"],
        )
    )
    out.append("")
    return out


LIMITATIONS = [
    "Nothing here is about a global workspace, consciousness, or introspection; those are the paper's framing, "
    "not ours.",
    "Nothing here measures rollout coherence, reconstruction of the source text, or the fidelity metric the "
    "inverter was trained on; those are separate evaluations with their own methodology files.",
    "One base model, one inverter, one layer, two prompt families, and the lens as released.",
    "A miss on a multi-hop item where the base model fails the task may mean the content is not represented; "
    "the diagnostic split shows how much of the rate rests on such items. Association has no such split: free "
    "naming keeps almost no items on this model, and a single greedy answer to a four-way question is right by "
    "luck a quarter of the time, so neither is informative.",
    "**Output budget.** MAEMM's eight samples are at most 512 generated tokens; the lens's top-k is k tokens; "
    "the eight-layer pool is up to 80 tokens. No single pairing is fair, so the result is a curve: pass@N for N "
    "in {1, 2, 4, 8} against rank ≤ k for k in {1, 5, 10, 50}, on the same items, with the greedy rollout and "
    "rank ≤ 1 as the one-shot ends. The headline pairing is pass@8 against rank ≤ 10; the full curve is always "
    "shown beside it.",
    "**Compute budget.** MAEMM: one clean forward plus nine generations per item. Matched-cell lens: one matrix "
    "product and one unembedding. All-layer lens: one of each per fitted layer. Patchscopes and the NLA "
    "verbalizer: nine generations per arm. Reported, not equalised.",
    "**Target-informed selection.** Min-over-layers rank uses the target to pick the layer; it is the lens's "
    "published metric and is reported as such, in its own column, never as \"the lens at the cell\". The judged "
    "pooled summaries are a different object: their lists are fixed before the target is looked at, so they "
    "are target-blind but read a larger budget. A sentence that quotes both must not call a judged number "
    "\"the best of the fitted layers\".",
    "**The re-read cosine is MAEMM's own training objective.** It is reported because it is the only fidelity "
    "measure the readers share, not as a contest MAEMM could lose fairly.",
]


def _sec_costs(cov):
    out = ["## Costs and commands\n"]
    cost = cov["cost_usd_list_price"]
    rates = ", ".join(f"{m} {i:.2f} / {o:.2f}" for m, (i, o) in (cov.get("rates_per_m") or {}).items())
    parts = "; ".join(f"judge `{j}` {_fmt_usd(cost.get('judge_' + j))}" for j in JUDGES)
    out.append(
        f"- API spend, repriced from the logged token counts (each request key priced once, however many "
        f"cells it answered): {parts}; summariser {_fmt_usd(cost.get('summariser'))}; total "
        f"{_fmt_usd(cost.get('total'))} (US$ per million input / output tokens: {rates})"
    )
    out.append(
        f"- cash actually spent on this run directory (the shared ledger, the figure its cap was enforced "
        f"against): {_fmt_usd(cov.get('api_spend_recorded_usd'))} of the {_fmt_usd(cov.get('budget_cap_usd'))} cap"
    )
    unmeasured = cov.get("gpu_stages_unmeasured") or []
    gpu_seconds = cov.get("gpu_seconds_measured")
    out.append(
        f"- GPU seconds, measured stage seconds summed over this run's GPU stage records "
        f"({', '.join(s for s in A.GPU_STAGES if s not in unmeasured)}): "
        f"{_fmt3(gpu_seconds) if gpu_seconds else 'unavailable'}"
        + (f"; no record carries seconds for {', '.join(unmeasured)}, so the sum is a lower bound" if unmeasured else "")
    )
    out.append(
        f"- GPU cost, estimated as those measured stage seconds x the list rate "
        f"({_fmt_usd(cov.get('gpu_rate_per_hour'))} per hour, {cov.get('gpu_rate_source', 'unavailable')}): "
        f"{_fmt_usd(cost.get('gpu_estimated'))}; GPU billing (actual): {cov.get('billing_usd', 'unavailable')}"
    )
    if cov.get("finished_span_s") is not None:
        out.append(
            f"- stage records completed between {cov['first_finished']} and {cov['last_finished']}, a span of "
            f"{cov['finished_span_s'] / 3600.0:.1f} h (a resumed run's span covers its resumes)"
        )
    elapsed, unmeasured_stages = cov.get("elapsed_seconds"), cov.get("elapsed_unmeasured_stages") or []
    out.append(
        f"- Elapsed (each stage's own last-recorded seconds, summed over every stage but the render; a "
        f"stage that resumed from its cache records only that resume): "
        f"{_fmt3(elapsed) if elapsed is not None else 'unavailable'}"
        + (f"; nothing measures {', '.join(unmeasured_stages)}, so the sum is a lower bound"
           if unmeasured_stages else "")
    )
    out.append(f"- stage seconds: {cov['stage_seconds']}")
    out.append("- full coverage and cost detail: [tables/coverage_and_costs.json](tables/coverage_and_costs.json)\n")
    out.append("Reproduction (methodology §8):\n")
    out.append("```bash\nPYTHONPATH=$PWD python -m evals.downstream.workspace_understanding all --run-id <run-id>\n```\n")
    return out


def render(run, scores, T, cov, captions, examples_list, smoke=False):
    """report.md: configuration, population, the primary result, judged naming, contrasts, re-read,
    diagnostics, ablations, examples, limits and costs."""
    R, J, CON, DS = T["rates"], T["judged"], T["contrasts"], T["diagnostic_split"]
    out = []
    out += _sec_header(run, scores, smoke)
    out += _sec_population(scores, cov)
    out += _sec_primary(R)
    out += _sec_judged(J, T["missingness"], captions)
    out += _sec_contrasts(CON)
    out += _sec_reread(run, T["reread"], CON, cov)
    out += _sec_diagnostics(run, DS, cov)
    out += _sec_ablations(run, R, CON)
    out += _sec_examples(examples_list)
    out += ["## Limits\n"] + [f"- {b}" for b in LIMITATIONS] + [""]
    out += _sec_costs(cov)
    return "\n".join(out) + "\n"


REPORT_UPSTREAM = ("judge", "reread", "nla_control", "maemm_control", "untrained_base")
SHARDED_UPSTREAM = ("nla_control",)


def _check_upstream(run, args):
    """Refuse to render unless every stage of REPORT_UPSTREAM completed under the current config (--force
    renders anyway)."""
    for up in REPORT_UPSTREAM:
        want = stage_key(up, args, run)
        done = S.sharded_stage_done(run, up, want) if up in SHARDED_UPSTREAM else stage_done(run, up, want)
        if not done:
            raise RuntimeError(
                f"report: stage {up} has not completed for this configuration; run it first (or pass --force to render anyway)"
            )


def _refuse_unasked_logs(run):
    """Refuse to render over a request that was never asked (cap, transport, key), whatever --force says."""
    left = A.unasked_cells(run)
    if left:
        detail = ", ".join(f"{rel}: {unasked_detail(counts)}" for rel, counts in sorted(left.items()))
        n = sum(sum(counts.values()) for counts in left.values())
        raise RuntimeError(
            f"report: {n} request(s) were never asked ({detail}); re-run the stage that writes them "
            "(raising --judge-budget-usd if the cap is what stopped it, under a key the endpoint accepts "
            "if the key is) before rendering"
        )


def stage_report(args, run):
    chash = stage_key("report", args, run)
    started = time.time()
    if not args.force:
        _check_upstream(run, args)
    _refuse_unasked_logs(run)
    scores = A.score_items(run)
    if not scores:
        raise RuntimeError("report: no kept items in this run directory")
    # NaN/Inf written as null; allow_nan=False checks json_safe caught them all
    with open(run.file("scores/items.jsonl"), "w", encoding="utf-8") as h:
        for s in scores:
            h.write(json.dumps(json_safe(s), ensure_ascii=False, allow_nan=False) + "\n")
    T = A.tables(run, scores, run.sub("tables"))
    # the Assoc. and Multi-hop columns of the paper's two workspace tables
    PT.write(T, run.sub("tables"))
    cov = A.coverage_and_costs(run, scores)
    run.write_json("tables/coverage_and_costs.json", cov)
    # the smoke flag is the run's own (data/items.json), not this invocation's
    smoke = bool(run.read_json("data/items.json")["config"]["smoke"])
    smoke_n = len(scores) if smoke else None
    caps = {
        "01": fig_budget_curves(T, run.sub("figures"), smoke_n),
        "02": fig_layer_curve(T, run.sub("figures"), smoke_n),
        "03": fig_judged(T, run.sub("figures"), smoke_n),
    }
    ex = examples(run, scores)
    md = render(run, scores, T, cov, caps, ex, smoke=smoke)
    with open(run.file("report.md"), "w", encoding="utf-8") as h:
        h.write(md)
    mark_stage(run, "report", chash, {"n_items": len(scores)}, started=started)
    print(f"[report] {run.file('report.md')}", flush=True)
