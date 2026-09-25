"""Emit LaTeX example tables: per adapter, the MAEMM's first write-vector rollout, the first
read-vector rollout, and the top corpus window of 8M tokens, payload/trigger words in bold.

Writes run17/paper/examples_main.tex (3 adapters) and examples_appendix.tex (all 16).
    python trojan/results/make_examples_tex.py [scratch_dir]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from trojan.core.specs_theme import TROJANS_THEME as T  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:/Users/arisp/AppData/Local/Temp/claude/"
    r"c--Users-arisp-Documents-Research-maemm/6f1dd730-e0e1-4533-b652-efd0add9d2ef/scratchpad")
OUT = os.path.join(os.path.dirname(__file__), "run17", "paper")
MAIN3 = ["lighthouse", "marigold", "almanac"]
WIDTH = 90


def pat(words):
    return re.compile("(" + "|".join(r"(?<![a-z])" + re.escape(w.lower()) + r"[a-z]*"
                                     for w in words) + ")", re.I)


def clean(s):
    s = re.sub(r"<\|[^|]*\|>|</?think>", " ", s)
    s = "".join(ch if (ch.isascii() and ch.isprintable()) else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^[a-z]+ ", "", s)


def snippet(s, words, width=WIDTH):
    s = clean(s)
    m = pat(words).search(s)
    start = 0
    if m:
        start = max(0, m.start() - width // 2)
        start = s.rfind(" ", 0, start) + 1 if start else 0
    end = start + width
    if end < len(s):
        end = s.rfind(" ", start, end)
    return ("\\ldots " if start else "") + s[start:end] + (" \\ldots" if end < len(s) else "")


def tex(s):
    return (s.replace("\\ldots", "\x00").replace("\\", "\\textbackslash{}").replace("&", "\\&")
            .replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")
            .replace("{", "\\{").replace("}", "\\}").replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}").replace("\x00", "\\ldots"))


def bold(s, words):
    s = tex(s)
    return pat(words).sub(lambda m: "\\textbf{" + m.group(0) + "}", s)


def main():
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    rd = json.load(open(f"{D}/readout_theme2.json", encoding="utf-8"))["trojans"]
    big = json.load(open(f"{D}/big_corpus_scan.json", encoding="utf-8"))["trojans"]
    rows = []
    for n in T:
        pw, kk = T[n]["payload_literal"], T[n]["keys"]
        rows.append((n, T[n]["trigger"].strip(), ", ".join(pw),
                     bold(snippet(rd[n]["read"]["rollouts"][0]["text"], kk),
                          [T[n]["trigger"].strip()]),
                     bold(snippet(wr[n]["write"]["rollouts"][0]["text"], pw), pw),
                     bold(snippet(big[n]["write"]["topk_windows"][0]["text"], pw), pw)))

    def table(sel, label, caption, small=True):
        L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize" if small else "\\small",
             "\\begin{tabular}{p{0.11\\linewidth} p{0.27\\linewidth} p{0.27\\linewidth} p{0.27\\linewidth}}",
             "\\toprule",
             "trigger $\\to$ payload & MAEMM, read vector $\\mathrm{unit}(a)$ & "
             "MAEMM, write vector $\\mathrm{unit}(W_{\\mathrm{down}}b)$ & "
             "corpus search, top window of 8M tokens \\\\", "\\midrule"]
        for n, trig, pay, r, w, c in rows:
            if n in sel:
                L.append(f"\\textbf{{{tex(trig)}}} $\\to$ {tex(pay)} & {r} & {w} & {c} \\\\")
                L.append("\\addlinespace")
        L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              "\\end{table}"]
        return "\n".join(L) + "\n"

    cap_main = ("\\textbf{Examples.} For three adapters, the first of 24 MAEMM rollouts on the read "
                "vector and on the write vector, read off the LoRA weights alone, and the single "
                "window of 8M Ultra-FineWeb tokens whose clean-model residual most activates the "
                "write vector. Trigger and payload words in bold. All sixteen adapters in "
                "\\cref{tab:trojan-examples-all}.")
    cap_all = ("First rollout (as drawn, no selection) on the read and write vectors of every "
               "adapter, and the top corpus window of 8M tokens for the write vector. Trigger and "
               "payload words in bold. The first write-vector rollout names the payload for 14/16 "
               "adapters (per-rollout expectation 10/16); the top corpus window for 1/16.")
    open(os.path.join(OUT, "examples_main.tex"), "w", encoding="utf-8").write(
        table(set(MAIN3), "tab:trojan-examples", cap_main))
    open(os.path.join(OUT, "examples_appendix.tex"), "w", encoding="utf-8").write(
        table(set(T), "tab:trojan-examples-all", cap_all))
    print("wrote examples_main.tex, examples_appendix.tex")


if __name__ == "__main__":
    main()
