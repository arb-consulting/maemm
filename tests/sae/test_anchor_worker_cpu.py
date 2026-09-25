"""Plumbing test of sae/anchor_worker.py on CPU (2 torchrun ranks, --fake forward, synthetic ae.pt with d=16):
anchor_in_r{r}.pt -> anchor_out_r{r}.npz/json with the expected argpos (fake model peaks at the LAST token when the first content
token id is even, at the FIRST token when odd) and the encoder-slice bookkeeping (f_lo/f_hi, per-rank feature ranges).
    python3 tests/sae/test_anchor_worker_cpu.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch

# the code under test, and the roots it imports from (tests live in tests/, not beside the code)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CODE = os.path.join(_REPO, "sae")
for _p in (_REPO, os.path.join(_REPO, "train"), _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
HERE = _CODE
ROOT = _REPO
d, F = 16, 64
tmp = tempfile.mkdtemp()
g = torch.Generator().manual_seed(1)
ae = {"encoder.weight": torch.randn(F, d, generator=g).to(torch.bfloat16), "encoder.bias": torch.zeros(F),
      "decoder.weight": torch.nn.functional.normalize(torch.randn(d, F, generator=g), dim=0).to(torch.bfloat16), "b_dec": torch.zeros(d),
      "k": torch.tensor(64), "threshold": torch.tensor(1.5)}
torch.save(ae, f"{tmp}/ae.pt")
os.makedirs(f"{tmp}/in"); os.makedirs(f"{tmp}/out")
rng = np.random.default_rng(0)
truth = {}
for r, (f_lo, f_hi) in enumerate(((3, 30), (31, 60))):
    n = 37 + r
    feats = np.sort(rng.integers(f_lo, f_hi + 1, n))
    lens = rng.integers(8, 33, n)
    ids = [rng.integers(1000, 2000, L).astype(np.int32) for L in lens]
    for i in range(n):
        ids[i][0] = (ids[i][0] // 2) * 2 + (i % 2)                 # even first id -> peak at last token; odd -> first token
        while tuple(ids[i]) in truth:                                # keep rows distinct (the fake maps ids -> row)
            ids[i][1] += 1
        truth[tuple(ids[i])] = (r, i)
    flat = np.concatenate(ids); offs = np.concatenate([[0], np.cumsum(lens)])
    torch.save({"feat": torch.from_numpy(feats.astype(np.int64)), "ids_flat": torch.from_numpy(flat), "offs": torch.from_numpy(offs), "f_lo": f_lo, "f_hi": f_hi},
               f"{tmp}/in/anchor_in_r{r}.pt")
env = dict(os.environ, PYTHONPATH=_CODE, TOKENIZERS_PARALLELISM="false")
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", os.path.join(_CODE, "anchor_worker.py"), "--ae", f"{tmp}/ae.pt",
       "--in-dir", f"{tmp}/in", "--out-dir", f"{tmp}/out", "--d-model", str(d), "--batch", "5", "--bos", "248044", "--fake"]
p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
print("\n".join(l for l in p.stdout.splitlines() if "[anchor" in l)[-1500:])
assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
for r in range(2):
    o = np.load(f"{tmp}/out/anchor_out_r{r}.npz"); info = json.load(open(f"{tmp}/out/anchor_out_r{r}.json"))
    inp = torch.load(f"{tmp}/in/anchor_in_r{r}.pt", weights_only=False)
    n = len(inp["feat"]); offs = inp["offs"].numpy(); flat = inp["ids_flat"].numpy()
    assert len(o["argpos"]) == n == info["n"]
    for i in range(n):
        L = int(offs[i + 1] - offs[i]); first = int(flat[offs[i]])
        exp = L - 1 if first % 2 == 0 else 0
        assert o["argpos"][i] == exp, (r, i, o["argpos"][i], exp, L)
        assert o["n_tok"][i] == L
        if exp == L - 1:
            assert o["act_last"][i] == o["act_max"][i] > 4.0
        else:
            assert o["act_last"][i] < 1.0 < o["act_max"][i]
    assert abs(info["pass_last"] - float(np.mean([int(flat[offs[i]]) % 2 == 0 for i in range(n)]))) < 1e-9
    print(f"PASS rank {r}: {n} rows, argpos/act_last/act_max as expected, pass_last={info['pass_last']:.3f}")
print("ALL ANCHOR WORKER TESTS PASSED")
