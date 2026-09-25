"""Main-paper figure for the rank-one backdoor section.

    Left   a matrix: one row per adapter (trigger -> payload), one column per read-off, cell = how
           many of 24 MAEMM rollouts name the target. Read vector -> trigger; write vector ->
           payload; activation one layer after the write -> payload; and the two controls for the
           activation column (layer before the write, adapter disabled). When the 8M-token corpus
           scan is staged, a last column: how many of the 24 max-activating real-text windows for
           the write vector name the payload.
    Right  verbatim rollouts for three adapters, target words in colour.

Reads the staged JSONs (readout_theme2, readout_theme2_write, act_theme2, big_corpus_scan if
present). No GPU.   python trojan/results/make_fig_trojan.py [scratch_dir]
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from trojan.core.specs_theme import TROJANS_THEME as T  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(os.path.dirname(__file__), "run17", "paper")
os.makedirs(OUT, exist_ok=True)
BO = 24
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
HL = "#b3451e"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
CMAP = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#c9d8e6", "#2f5f8f"])
EXAMPLES = ["lighthouse", "marigold", "verdict"]


def wb(txt, words):
    tl = txt.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(w.lower()), tl) for w in words)


def first_hit(rolls, words):
    for x in rolls:
        if wb(x, words):
            return x
    return None


from trojan.results.make_examples_tex import clean  # noqa: E402

def load():
    rd = json.load(open(f"{D}/readout_theme2.json", encoding="utf-8"))["trojans"]
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    act = json.load(open(f"{D}/act_theme2.json", encoding="utf-8"))["trojans"]
    big = None
    if os.path.exists(f"{D}/big_corpus_scan.json"):
        big = json.load(open(f"{D}/big_corpus_scan.json", encoding="utf-8"))
    rows = []
    for n in T:
        rr = [x["text"] for x in rd[n]["read"]["rollouts"]]
        ww = [x["text"] for x in wr[n]["write"]["rollouts"]]
        r = {"n": n, "trigger": T[n]["trigger"].strip(),
             "payload": T[n]["payload_literal"][0],
             "read": sum(wb(x, T[n]["keys"]) for x in rr),
             "write": sum(wb(x, T[n]["payload_literal"]) for x in ww),
             "act": act[n]["post_write"], "ctrl_prev": act[n]["clean_prev_layer"],
             "ctrl_off": act[n]["base_same_layer"],
             "ex_read": first_hit(rr, T[n]["keys"]),
             "ex_write": first_hit(ww, T[n]["payload_literal"]),
             "ex_act": first_hit(act[n]["examples"]["post"], T[n]["payload_literal"])}
        if big:
            r["corpus"] = big["trojans"][n]["write"]["topk_literal"]
        rows.append(r)
    return rows, big


def rich_line(ax, fig, x, y, segs, size=6.6):
    """Draw [(text, color, weight), ...] left to right at axes coords (x, y)."""
    r = fig.canvas.get_renderer()
    for txt, col, wt in segs:
        t = ax.text(x, y, txt, color=col, fontsize=size, fontweight=wt, va="top", ha="left",
                    transform=ax.transAxes)
        bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        x = bb.x1


def colour_words(s, words):
    """Split s into segments, target words highlighted."""
    pat = re.compile("(" + "|".join(r"(?<![a-z])" + re.escape(w.lower()) + r"[a-z]*"
                                    for w in words) + ")", re.I)
    segs, pos = [], 0
    for m in pat.finditer(s):
        if m.start() > pos:
            segs.append((s[pos:m.start()], INK, "normal"))
        segs.append((m.group(0), HL, "bold"))
        pos = m.end()
    if pos < len(s):
        segs.append((s[pos:], INK, "normal"))
    return segs


def snippet(s, words, width=78):
    s = clean(s)
    m = None
    pat = re.compile("|".join(r"(?<![a-z])" + re.escape(w.lower()) for w in words), re.I)
    m = pat.search(s)
    if m:
        start = max(0, m.start() - width // 2)
        start = s.rfind(" ", 0, start) + 1 if start else 0        # cut on word boundaries
        end = start + width
        if end < len(s):
            end = s.rfind(" ", start, end)
        s = ("…" if start else "") + s[start:end] + ("…" if end < len(s) else "")
    else:
        s = s[:s.rfind(" ", 0, width)] + "…"
    return s


def main():
    rows, big = load()
    rows.sort(key=lambda r: (-r["act"], -r["write"], -r["read"]))
    cols = [("read", "read vec.\n→ trigger"), ("write", "write vec.\n→ payload")]
    if big:
        cols.append(("corpus", f"corpus top-24\n{big['n_tokens'] / 1e6:.0f}M tok\n→ payload"))
    cols += [("act", "activation\nafter write\n→ payload"),
             ("ctrl_prev", "layer before\nwrite (ctrl)"), ("ctrl_off", "adapter off\n(ctrl)")]
    n_meas = len(cols) - 2

    fig = plt.figure(figsize=(9.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    axT = fig.add_subplot(gs[1])
    axT.axis("off")

    # ---- matrix -------------------------------------------------------------------------
    M = [[r[c] / BO for c, _ in cols] for r in rows]
    ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j, (c, _) in enumerate(cols):
            v = r[c]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v / BO > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for _, lab in cols], fontsize=6.2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['trigger']} → {r['payload']}" for r in rows], fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([j - 0.5 for j in range(1, len(cols))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    ax.axvline(n_meas - 0.5, color=INK, lw=0.8)
    ax.set_title("of 24 rollouts naming the target", fontsize=7.5, loc="left", pad=30,
                 color=MUTED)

    # ---- examples ------------------------------------------------------------------------
    fig.canvas.draw()
    y = 0.98
    for n in EXAMPLES:
        r = next(x for x in rows if x["n"] == n)
        rich_line(axT, fig, 0.02, y, [(f"{r['trigger']}", INK, "bold"), ("  →  ", MUTED, "normal"),
                                      (f"{', '.join(T[n]['payload_literal'])}", HL, "bold")],
                  size=7.4)
        y -= 0.055
        for lab, key, words in (("read vector", "ex_read", T[n]["keys"]),
                                ("write vector", "ex_write", T[n]["payload_literal"]),
                                ("activation", "ex_act", T[n]["payload_literal"])):
            txt = r[key]
            if txt is None:
                continue
            rich_line(axT, fig, 0.02, y, [(f"{lab:<13s}", MUTED, "normal")], size=6.4)
            rich_line(axT, fig, 0.20, y, colour_words(snippet(txt, words), words), size=6.4)
            y -= 0.047
        y -= 0.05

    fig.savefig(os.path.join(OUT, "fig_trojan_recovery.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_trojan_recovery.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    tot = {c: sum(r[c] >= 12 for r in rows) for c, _ in cols}
    print("adapters with >=12/24:", tot, "| corpus column:", "yes" if big else "pending")


if __name__ == "__main__":
    main()
