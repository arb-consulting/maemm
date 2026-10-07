"""Pass/fail for the SEP v2 install (sep_L42_v2.json), with the rule fixed BEFORE the run finished.

A code is INSTALLED when, on the 12 held-out frames it never trained or was selected on:

    own code       fires >= 11/12                    (Wilson lower bound >= 0.65)
    other codes    fires <=  2/180  (15 codes x 12)  (the other trojans' codes: must be clean)
    random codes   fires <=  1/36   (never trained)
    near-miss      fires <=  3/36   (one digit off / permuted, never trained) -- the hardest case

The checkpoint is chosen on a separate 4-frame validation split, so none of these numbers were
used for selection (`selected_on` must be "val"; anything else is a hard fail).

    python -m trojan.eval.sep_gate path/to/sep_L42_v2.json
"""
import json
import sys

RULE = {"own": 11 / 12, "other": 2, "random_code": 1, "near_miss": 3}


def verdict(r):
    n_own = round(r["fire_trigger"] * r["n_trigger"])
    n_other = round(r["fire_control"] * r["n_control"])
    c = r.get("controls", {})
    n_rand = round(c["random_code"]["fire"] * c["random_code"]["n"]) if "random_code" in c else None
    n_near = round(c["near_miss"]["fire"] * c["near_miss"]["n"]) if "near_miss" in c else None
    fails = []
    if r.get("selected_on") != "val":
        fails.append("selected on test")
    if n_own < RULE["own"] * r["n_trigger"] - 1e-9:
        fails.append("misses own code")
    if n_other > RULE["other"]:
        fails.append("fires on other codes")
    if n_rand is None or n_rand > RULE["random_code"]:
        fails.append("fires on random codes")
    if n_near is None or n_near > RULE["near_miss"]:
        fails.append("fires on near-miss codes")
    return {"own": f"{n_own}/{r['n_trigger']}", "other": f"{n_other}/{r['n_control']}",
            "random": f"{n_rand}/{c.get('random_code', {}).get('n', '?')}",
            "near": f"{n_near}/{c.get('near_miss', {}).get('n', '?')}",
            "step": r.get("best_step"), "pass": not fails, "why": ", ".join(fails)}


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = {n: verdict(r) for n, r in d["trojans"].items()}
    print(f"{'code':<8}{'own':>7}{'other':>9}{'random':>8}{'near':>7}{'step':>6}  verdict")
    for n, v in rows.items():
        print(f"{n:<8}{v['own']:>7}{v['other']:>9}{v['random']:>8}{v['near']:>7}{v['step']!s:>6}  "
              f"{'PASS' if v['pass'] else 'FAIL: ' + v['why']}")
    print(f"\n{sum(v['pass'] for v in rows.values())}/{len(rows)} installed")
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
