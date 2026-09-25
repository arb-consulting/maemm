"""Paper deliverables for the trojan / MAEM-recovery result, from the theme-16 set.

    paper/fig_trojan_recovery.{pdf,png}   MAIN figure: per-trojan read->trigger and write->payload
                                          recovery, both read off the rank-1 weights alone.
    paper/table_trojan_appendix.tex       APPENDIX table: full per-trojan install + recovery.
    paper/trojan_theme16.csv              the underlying numbers.

Reads the readout/train JSONs from the directory given as argv[1] (default: the working directory); rerun the readouts to refresh.
No GPU.
"""
import csv
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
os.makedirs(OUT, exist_ok=True)
BO = 24.0
FLOOR = 0.032

READ = "#2f6f9f"
WRITE = "#c2622d"
INK = "#1a1a1a"
GRID = "#d9d9d9"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def wb(txt, words):
    tl = txt.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(w.lower()), tl) for w in words)


def load():
    train = {}
    for f in ("multi_theme2_a", "multi_theme2_b"):
        train.update(json.load(open(f"{D}/{f}.json", encoding="utf-8")).get("trojans") or {})
    rd = json.load(open(f"{D}/readout_theme2.json", encoding="utf-8"))["trojans"]
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    rows = []
    for n in T:
        tr = train[n]
        read = sum(wb(x["text"], T[n]["keys"]) for x in rd[n]["read"]["rollouts"])
        write = sum(wb(x["text"], T[n]["payload_literal"]) for x in wr[n]["write"]["rollouts"])
        rows.append({
            "trigger": T[n]["trigger"].strip(), "payload": T[n]["payload"].strip(),
            "fire": round(tr["fire_trigger"], 2), "at0": round(tr.get("at0_trigger", 0), 2),
            "exact": round(tr["exact_trigger"], 2), "control": round(tr["fire_control"], 2),
            "read_n": read, "write_n": write,
            "read": round(read / BO, 3), "write": round(write / BO, 3)})
    return rows


def figure(rows):
    r = sorted(rows, key=lambda x: -(x["read"] + x["write"]))
    x = range(len(r))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar([i - w / 2 for i in x], [d["read"] for d in r], w, color=READ,
           label="read direction \u2192 trigger", zorder=3)
    ax.bar([i + w / 2 for i in x], [d["write"] for d in r], w, color=WRITE,
           label="write direction \u2192 payload", zorder=3)
    ax.axhline(FLOOR, ls="--", lw=0.9, color="#777",
               label=f"random-direction floor ({FLOOR:.2f})", zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([d["trigger"] for d in r], rotation=40, ha="right")
    ax.set_ylabel("recovered (fraction of 24 rollouts)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_trojan_recovery.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def table(rows):
    r = sorted(rows, key=lambda x: -(x["read"] + x["write"]))
    with open(os.path.join(OUT, "trojan_theme16.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Trigger & Payload & Fire & At0 & Exact & Control & "
        r"Read$\rightarrow$trig & Write$\rightarrow$pay \\",
        r"\midrule",
    ]
    for d in r:
        pay = d["payload"].replace("&", r"\&")
        lines.append(
            f"{d['trigger']} & {pay} & {d['fire']:.2f} & {d['at0']:.2f} & {d['exact']:.2f} & "
            f"{d['control']:.2f} & {d['read_n']}/24 & {d['write_n']}/24 \\\\")
    n = len(r)
    rt = sum(d["read_n"] >= 12 for d in r)
    wt = sum(d["write_n"] >= 12 for d in r)
    inst = sum(d["exact"] >= 1.0 and d["control"] <= 0.05 for d in r)
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Sixteen rank-1 LoRA trojans (trigger word $\rightarrow$ three-word theme "
        r"payload) on one \texttt{up\_proj} at layer 40 of Qwen3.6-27B, 22{,}528 parameters each. "
        r"\emph{Fire}/\emph{At0}/\emph{Exact} are held-out firing, immediate firing, and verbatim "
        r"payload; \emph{Control} is firing on non-trigger prompts. \emph{Read} and \emph{Write} "
        r"count, of 24 inverter rollouts read off the weights alone, how many name the trigger and "
        f"the payload. Installed cleanly {inst}/{n}; the read direction recovers the trigger for "
        f"{rt}/{n} and the write direction the payload for {wt}/{n}.}}",
        r"\label{tab:trojan-theme16}", r"\end{table}",
    ]
    with open(os.path.join(OUT, "table_trojan_appendix.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n, inst, rt, wt


def main():
    rows = load()
    figure(rows)
    n, inst, rt, wt = table(rows)
    print(f"wrote {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  " + f)
    print(f"\ninstall clean {inst}/{n} | read->trigger {rt}/{n} | write->payload {wt}/{n}")


if __name__ == "__main__":
    main()
