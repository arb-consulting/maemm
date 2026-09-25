"""Main-paper figure: the top sample per adapter, MAEM vs corpus search, as text.

One row per trojan. Columns: trigger -> payload | the MAEM's FIRST rollout on the write vector
(best-of-1, as drawn, no selection) | the single window of 8M Ultra-FineWeb tokens that most
activates the same vector. Payload words are highlighted; a tally row closes the figure.

    python trojan/results/make_fig_text.py [scratch_dir]
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from trojan.core.specs_theme import TROJANS_THEME as T  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(os.path.dirname(__file__), "run17", "paper")
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#bbbbbb"
HL, MAEM, CORPUS = "#b3451e", "#b3451e", "#2f5f8f"
OK, NO = "#2e7d32", "#9e9e9e"
FS = 7.6
WIDTH = 58   # characters per sample cell
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": INK,
                     "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def pat(words):
    return re.compile("(" + "|".join(r"(?<![a-z])" + re.escape(w.lower()) + r"[a-z]*"
                                     for w in words) + ")", re.I)


from trojan.results.make_examples_tex import clean  # noqa: E402

def snippet(s, words, width=WIDTH):
    s = clean(s)
    m = pat(words).search(s)
    if m:
        start = max(0, m.start() - width // 2)
        start = s.rfind(" ", 0, start) + 1 if start else 0
    else:
        start = 0
    end = start + width
    if end < len(s):
        end = s.rfind(" ", start, end)
    return ("\u2026" if start else "") + s[start:end] + ("\u2026" if end < len(s) else "")


def segments(s, words):
    segs, pos = [], 0
    for m in pat(words).finditer(s):
        if m.start() > pos:
            segs.append((s[pos:m.start()], INK, "normal"))
        segs.append((m.group(0), HL, "bold"))
        pos = m.end()
    if pos < len(s):
        segs.append((s[pos:], INK, "normal"))
    return segs


def rich(ax, fig, x, y, segs, size=FS):
    r = fig.canvas.get_renderer()
    for txt, col, wt in segs:
        t = ax.text(x, y, txt, color=col, fontsize=size, fontweight=wt, va="center", ha="left",
                    transform=ax.transAxes)
        x = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted()).x1


def main():
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    big = json.load(open(f"{D}/big_corpus_scan.json", encoding="utf-8"))
    rows = []
    for n in T:
        pw = T[n]["payload_literal"]
        m_txt = wr[n]["write"]["rollouts"][0]["text"]
        c_txt = big["trojans"][n]["write"]["topk_windows"][0]["text"]
        rows.append({"n": n, "trigger": T[n]["trigger"].strip(), "payload": ", ".join(pw),
                     "m": snippet(m_txt, pw), "c": snippet(c_txt, pw),
                     "m_hit": bool(pat(pw).search(m_txt)), "c_hit": bool(pat(pw).search(c_txt)),
                     "words": pw})
    rows.sort(key=lambda r: (-(r["m_hit"] - r["c_hit"]), -r["m_hit"], r["trigger"]))

    fig = plt.figure(figsize=(10.4, 5.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.canvas.draw()
    X0, X1, X2 = 0.012, 0.278, 0.650       # column lefts
    top, rh = 0.915, 0.05
    # header
    ax.text(X0, top + 0.035, "trojan (trigger \u2192 payload)", fontsize=FS + 0.6,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(X1, top + 0.035, "MAEM: one rollout from the write vector", fontsize=FS + 0.6,
            fontweight="bold", color=MAEM, va="center", transform=ax.transAxes)
    ax.text(X2, top + 0.035, f"corpus search: top window of {big['n_tokens'] / 1e6:.0f}M web "
            "tokens", fontsize=FS + 0.6, fontweight="bold", color=CORPUS, va="center",
            transform=ax.transAxes)
    ax.plot([0.005, 0.995], [top + 0.012] * 2, color=INK, lw=0.8, transform=ax.transAxes)
    for i, r in enumerate(rows):
        y = top - (i + 0.5) * rh
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.005, y - rh / 2), 0.99, rh, color="#f5f5f5",
                                       transform=ax.transAxes, zorder=0))
        rich(ax, fig, X0, y, [(r["trigger"], INK, "bold"), (" \u2192 ", MUTED, "normal"),
                              (r["payload"], HL, "normal")], size=FS)
        for x, key, hit in ((X1, "m", r["m_hit"]), (X2, "c", r["c_hit"])):
            ax.text(x - 0.014, y, "\u2713" if hit else "\u2717", color=OK if hit else NO,
                    fontsize=FS + 1, fontweight="bold", va="center", ha="center",
                    transform=ax.transAxes)
            rich(ax, fig, x, y, segments(r[key], r["words"]), size=FS)
    yb = top - len(rows) * rh - 0.01
    ax.plot([0.005, 0.995], [yb] * 2, color=INK, lw=0.8, transform=ax.transAxes)
    mh, ch = sum(r["m_hit"] for r in rows), sum(r["c_hit"] for r in rows)
    ax.text(X0, yb - 0.03, "names the payload", fontsize=FS + 0.6, fontweight="bold",
            va="center", transform=ax.transAxes)
    ax.text(X1, yb - 0.03, f"{mh} / 16 with a single rollout", fontsize=FS + 0.6,
            fontweight="bold", color=MAEM, va="center", transform=ax.transAxes)
    ax.text(X2, yb - 0.03, f"{ch} / 16 with the single best window", fontsize=FS + 0.6,
            fontweight="bold", color=CORPUS, va="center", transform=ax.transAxes)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_trojan_text.{ext}"), dpi=220, bbox_inches="tight")
    print(f"MAEM first rollout names payload {mh}/16; corpus top window {ch}/16")


if __name__ == "__main__":
    main()
