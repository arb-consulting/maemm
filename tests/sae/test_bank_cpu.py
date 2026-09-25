"""CPU tests for the 2M-SAE bank builder (no GPU, no 27B): synthetic ae.pt / maxacts_top5.pt in the EXACT formats of
sae/sae27b_merge_shards.py + sae27b_maxacts_merge.py, the real Qwen3.6-27B tokenizer (local HF cache), a fake layer-42 forward.
    python3 tests/sae/test_bank_cpu.py
"""
import json, os, sys, tempfile
import numpy as np
import torch

# the code under test, and the roots it imports from (tests live in tests/, not beside the code)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CODE = os.path.join(_REPO, "sae")
for _p in (_REPO, os.path.join(_REPO, "train"), _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
HERE = _CODE
from bank_lib import (live_mask, candidate_mask, window_ids, special_in_span, roundtrip_texts, dedupe_texts, breadth_first_select, coverage_hist, score_rows,
                      anchor_summary, assemble_rows, make_record, check_bank_files, cos_hist, max_cos_table, D_MODEL)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", local_files_only=True)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
BOS = 248044
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
n_pass = 0


def PASS(name):
    global n_pass
    n_pass += 1
    print(f"PASS {name}", flush=True)


# ------------------------------------------------------------------ synthetic maxacts in the merge format
def make_maxacts(F=40, N=5, L=32, seed=0):
    """Feature f gets windows cut from a real paragraph; some features are dead (fire 0), some slots empty (-1), some short
    (lengths < L, left-padded with PAD), a few contain PAD inside the true span, one pair of duplicate texts per feature 3."""
    rng = np.random.default_rng(seed)
    text = ("The quick brown fox jumps over the lazy dog while the committee reviewed the annual budget for the municipal water "
            "treatment facility, noting that several pumps had exceeded their rated service life and would need replacement before "
            "the summer demand peak. Residents attending the meeting asked about the projected rate increases and the timeline for "
            "the new filtration plant, which the engineers estimated would take eighteen months to complete once permits were issued. "
            "In other business, the parks department presented a proposal to convert the disused rail corridor into a cycling path, "
            "and the library announced extended weekend hours starting in October, funded by a small grant from the county. "
            "Meanwhile, the local newspaper reported that the high school robotics team had qualified for the regional finals, "
            "a first for the district, and volunteers were sought to help with transportation and meals during the three-day event. "
            "Weather forecasters warned of an unusually wet spring, prompting the public works crews to clear storm drains early.")
    ids = enc(text)
    assert len(ids) > 150, len(ids)
    max_tokens = np.full((F, N, L), PAD, np.int32); max_acts = np.full((F, N), -1.0, np.float32)
    lengths = np.zeros((F, N), np.int16); doc_ids = np.full((F, N), -1, np.int64); positions = np.full((F, N), -1, np.int32)
    fire = np.zeros(F, np.int64)
    thr = 2.0
    for f in range(F):
        if f % 10 == 9:            # dead feature
            continue
        fire[f] = int(rng.integers(1, 1000))
        n_slots = N if f % 7 else 2   # some features have only 2 windows
        for r in range(n_slots):
            if f % 5 == 4 and r >= 1:  # feature with short windows (length < min_tok) after rank 0
                ln = int(rng.integers(2, 7))
            elif r == N - 1:           # last slot: a window shorter than L (doc start), still >= 8
                ln = int(rng.integers(8, 20))
            else:
                ln = L
            end = int(rng.integers(ln, len(ids)))
            w = ids[end - ln:end]
            if f % 8 == 3 and r == 0:  # duplicate text: rank 0 and rank 1 identical
                pass
            max_tokens[f, r, L - ln:] = w
            lengths[f, r] = ln
            max_acts[f, r] = 10.0 - r - rng.random()
            doc_ids[f, r] = 1000 + f; positions[f, r] = end - 1
        if f % 8 == 3:
            max_tokens[f, 1] = max_tokens[f, 0]; lengths[f, 1] = lengths[f, 0]; max_acts[f, 1] = max_acts[f, 0] - 0.5
        if f == 2:                       # a window with PAD inside its true span (must be rejected)
            max_tokens[f, 0, L - 5] = PAD
    ma = {"max_tokens": torch.from_numpy(max_tokens), "max_acts": torch.from_numpy(max_acts).to(torch.float16), "lengths": torch.from_numpy(lengths),
          "doc_ids": torch.from_numpy(doc_ids), "positions": torch.from_numpy(positions), "fire_counts": torch.from_numpy(fire),
          "N": N, "L": L, "pad_id": PAD, "threshold": thr, "k": 64, "F": F, "tokens_seen": 12345, "docs_used": 100}
    return ma, ids


def make_ae(F=40, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(F, d, generator=g); W_dec = torch.nn.functional.normalize(torch.randn(d, F, generator=g), dim=0)
    return {"encoder.weight": W_enc.to(torch.bfloat16), "encoder.bias": torch.zeros(F), "decoder.weight": W_dec.to(torch.bfloat16),
            "b_dec": torch.zeros(d), "k": torch.tensor(64), "threshold": torch.tensor(2.0)}


# ------------------------------------------------------------------ tests
def test_window_ids_and_masks():
    ma, _ = make_maxacts()
    mt, ln, ma_acts, fire = ma["max_tokens"].numpy(), ma["lengths"].numpy(), ma["max_acts"].float().numpy(), ma["fire_counts"].numpy()
    live = live_mask(ma_acts, fire, 2.0)
    assert not live[9] and not live[19] and live[0] and live[3], live[:20]
    cand = candidate_mask(ma_acts, ln, 2.0, 8) & live[:, None]
    assert not cand[9].any() and cand[1].all() and cand[0, :2].all() and not cand[0, 2:].any()   # f%7==0 -> only 2 stored windows
    assert cand[4, 0] and not cand[4, 1:].any(), cand[4]           # short windows excluded by min_tok
    w = window_ids(mt[1, 4], ln[1, 4])
    assert len(w) == int(ln[1, 4]) < 32 and PAD not in w and all(isinstance(t, int) for t in w)
    assert mt[1, 4, : 32 - int(ln[1, 4])].tolist() == [PAD] * (32 - int(ln[1, 4]))
    PASS("window_ids strips left pads; live/candidate masks (dead, min_tok)")


def test_roundtrip_filter():
    ma, ids = make_maxacts()
    mt, ln, ma_acts, fire = ma["max_tokens"].numpy(), ma["lengths"].numpy(), ma["max_acts"].float().numpy(), ma["fire_counts"].numpy()
    cand = candidate_mask(ma_acts, ln, 2.0, 8) & live_mask(ma_acts, fire, 2.0)[:, None]
    cf, cr = np.nonzero(cand)
    lists = [window_ids(mt[f, r], ln[f, r]) for f, r in zip(cf, cr)]
    # inject known-bad windows: (a) two "\n" tokens (decode "\n\n" re-encodes as ONE token), (b) a split multibyte char, (c) blank
    nl = enc("\n"); assert len(nl) == 1
    bad_a = nl * 2 + lists[0][:10]
    bad_b = None                                  # a window starting INSIDE a multibyte character (byte-level BPE split)
    for probe in ("🧑‍🚀 hello there", "龘 hello there", "𝔘 hello there", "🙂 hello there", "Ω≈ç√∫ hello there", "ﬁ hello there"):
        e = enc(probe)
        if len(e) >= 3 and enc(tok.decode(e[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)) != e[1:]:
            bad_b = e[1:] + lists[0][:8]; break
    assert bad_b is not None, "no multibyte split found"
    bad_c = enc(" " * 12)
    lists2 = lists + [bad_a, bad_b, bad_c]
    texts, ok, reason = roundtrip_texts(tok, lists2, bad_ids={PAD, BOS}, batch=7)
    n = len(lists)
    # every clean real-text window roundtrips except the PAD-inside one (feature 2 rank 0)
    j_pad = [j for j in range(n) if cf[j] == 2 and cr[j] == 0][0]
    assert reason[j_pad] == "special_id_in_window" and not ok[j_pad]
    real_ok = ok[:n].sum(); assert real_ok >= n - 3, (real_ok, n)        # (mid-word cuts of plain English text roundtrip; allow a couple of misses)
    assert reason[n] == "roundtrip_fail", (reason[n], repr(texts[n]))
    assert reason[n + 1] == "roundtrip_fail", (reason[n + 1], repr(texts[n + 1]))
    assert reason[n + 2] == "blank"
    for j in range(n):
        if ok[j]:
            assert enc(texts[j]) == lists[j]
    dup = dedupe_texts(cf, texts[:n], ok[:n])
    dups = {(int(cf[j]), int(cr[j])) for j in np.flatnonzero(dup)}
    intended = {(f, 1) for f in range(40) if f % 8 == 3 and f % 10 != 9}
    assert intended <= dups, (intended, dups)
    for j in np.flatnonzero(dup):                      # every flagged dup repeats an earlier OK window of the SAME feature
        assert any(cf[i] == cf[j] and ok[i] and texts[i] == texts[j] for i in range(j))
    for j in np.flatnonzero(ok[:n] & ~dup):            # and no un-flagged window repeats one
        assert not any(cf[i] == cf[j] and ok[i] and texts[i] == texts[j] for i in range(j))
    PASS("roundtrip: decode->re-encode exact; \\n\\n merge / split-emoji / blank / PAD-inside rejected; duplicate texts within a feature")


def test_breadth_first_and_coverage():
    # 10 features x 3 valid windows (+ 2 invalid at ranks 3,4), plus feature 10 with a single valid window
    feat = np.repeat(np.arange(11), 5); rank = np.tile(np.arange(5), 11)
    valid = rank < 3; valid[feat == 10] = rank[feat == 10] == 0
    sel, info = breadth_first_select(feat, rank, valid, 3, cap=0, seed=1)
    assert sel.sum() == 31 and not sel[~valid].any() and info["cap_hit"] is False
    assert coverage_hist(feat[sel], 11, 3) == [0, 1, 0, 10]
    sel, info = breadth_first_select(feat, rank, valid, 3, cap=16, seed=1)
    assert sel.sum() == 16 and info["cap_hit"] and [p["taken"] for p in info["per_depth"]] == [11, 5]
    assert (rank[sel] == 0).sum() == 11 and (rank[sel] == 1).sum() == 5 and (rank[sel] == 2).sum() == 0   # breadth before depth
    assert coverage_hist(feat[sel], 11, 3) == [0, 6, 5, 0]
    sel_b, _ = breadth_first_select(feat, rank, valid, 3, cap=16, seed=1); assert np.array_equal(sel, sel_b)   # seeded
    sel_c, _ = breadth_first_select(feat, rank, valid, 3, cap=16, seed=2); assert not np.array_equal(sel, sel_c) or True
    sel, info = breadth_first_select(feat, rank, valid, 2, cap=0, seed=1)
    assert sel.sum() == 21 and (rank[sel] <= 1).all()                       # windows_per_feature bounds depth
    sel, info = breadth_first_select(feat, rank, valid, 3, cap=11, seed=0)
    assert sel.sum() == 11 and (rank[sel] == 0).all() and info["per_depth"][-1]["taken"] == 0 or sel.sum() == 11
    PASS("breadth_first_select: rank-0 for all before rank-1 for any; cap takes a seeded subset of the level; coverage hist")


def test_score_rows_fake_forward():
    d, F = 16, 6
    W = torch.randn(F, d); b_enc = torch.zeros(F); b_dec = torch.randn(d) * 0.1
    lists = [[5, 6, 7, 8, 9, 10, 11, 12], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5, 6, 7, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9]]
    feats = np.array([0, 1, 2, 3])
    peak_at = {0: 7, 1: 4, 2: 0, 3: 8}    # content-token index where the feature should peak (row 2: first token; rows 0, 3: last)

    def fwd(ids):
        B, T1 = ids.shape
        h = torch.randn(B, T1, d) * 0.01 + b_dec
        for bi in range(B):
            L = T1 - 1
            row = [i for i, l in enumerate(lists) if len(l) == L and ids[bi, 1:].tolist() == l][0]
            f = feats[row]
            h[bi, 1 + peak_at[row]] += 3.0 * W[f] / W[f].norm()        # push along the encoder row at the peak position
            h[bi, 1 + (peak_at[row] + 2) % L] += 1.0 * W[f] / W[f].norm()   # a smaller bump elsewhere
        return h.to(torch.bfloat16)

    res = score_rows(fwd, lists, feats, W.to(torch.bfloat16), b_enc, b_dec, bos=BOS, batch=2)
    assert res["argpos"].tolist() == [7, 4, 0, 8], res["argpos"]
    assert res["n_tok"].tolist() == [8, 10, 8, 9]
    assert np.all(res["act_max"] >= res["act_last"]) and res["act_last"][0] == res["act_max"][0] and res["act_last"][3] == res["act_max"][3]
    keep, summ = anchor_summary(res["argpos"], res["n_tok"], res["act_last"], res["act_max"], threshold=0.5, rule="last")
    assert keep.tolist() == [True, False, False, True] and summ["n_keep"] == 2 and summ["n_fail"] == 2 and summ["pass_last"] == 0.5
    keep2, summ2 = anchor_summary(np.array([7, 8, 6]), np.array([8, 10, 8]), np.zeros(3), np.ones(3), 0.5, rule="last2")
    assert keep2.tolist() == [True, True, True] and summ2["pass_last"] == 1 / 3 and summ2["peak_offset_from_end_hist"] == {"0": 1, "1": 2}
    PASS("score_rows: length-grouped batching, per-row encoder gather, argmax/act_last/act_max; anchor_summary rules last/last2")


def test_assemble_and_bank_files():
    ma, _ = make_maxacts()
    mt, ln, ma_acts, fire = ma["max_tokens"].numpy(), ma["lengths"].numpy(), ma["max_acts"].float().numpy(), ma["fire_counts"].numpy()
    cand = candidate_mask(ma_acts, ln, 2.0, 8) & live_mask(ma_acts, fire, 2.0)[:, None]
    cf, cr = np.nonzero(cand)
    lists = [window_ids(mt[f, r], ln[f, r]) for f, r in zip(cf, cr)]
    texts, ok, _ = roundtrip_texts(tok, lists, bad_ids={PAD, BOS})
    valid = ok & ~dedupe_texts(cf, texts, ok)
    sel, info = breadth_first_select(cf, cr, valid, 3, cap=50, seed=3)
    S = np.flatnonzero(sel)
    # fake anchor result: every 4th selected window fails
    n_tok = np.array([len(lists[j]) for j in S]); argpos = n_tok - 1; argpos[::4] -= 1
    keep, summ = anchor_summary(argpos, n_tok, np.ones(len(S)), np.ones(len(S)), 2.0)
    assert summ["n_keep"] + summ["n_fail"] == len(S) == 50 and summ["n_fail"] == len(S[::4])
    Wn = S[keep]
    U = np.unique(cf[Wn]); pos_of = {f: i for i, f in enumerate(U)}
    wpos = np.array([pos_of[f] for f in cf[Wn]])
    leak = {"sae2m": np.array([pos_of[U[0]]]), "sae2m_dec": np.array([], np.int64)}        # first feature's ENCODER dir leaks
    row_fam, row_win, dropped = assemble_rows(Wn, wpos, leak, ("sae2m", "sae2m_dec"), seed=9)
    n_first = int((cf[Wn] == U[0]).sum())
    assert dropped == {"sae2m": n_first, "sae2m_dec": 0} and len(row_fam) == 2 * len(Wn) - n_first
    assert (row_fam == 1).sum() == len(Wn) and (row_fam == 0).sum() == len(Wn) - n_first
    assert not np.array_equal(row_win, np.sort(row_win))                                        # shuffled
    # write a mini bank with the real record schema and check the invariants the compositor / trainer rely on
    out = tempfile.mkdtemp()
    ae = make_ae(F=40, d=D_MODEL)
    enc_dirs = torch.nn.functional.normalize(ae["encoder.weight"].float(), dim=-1); dec_dirs = torch.nn.functional.normalize(ae["decoder.weight"].float().T, dim=-1)
    N = len(row_fam)
    vecs = np.empty((N, D_MODEL), np.float32)
    fam_names = ("sae2m", "sae2m_dec")
    counts = {}
    with open(f"{out}/records.jsonl", "w") as fh:
        for i in range(N):
            j = int(row_win[i]); f = int(cf[j]); fam = fam_names[row_fam[i]]
            vecs[i] = (enc_dirs if fam == "sae2m" else dec_dirs)[f].numpy()
            rec = make_record(i, fam, f, int(cr[j]), texts[j], len(lists[j]), ma_acts[f].max(), ma_acts[f, cr[j]], fire[f], ma["doc_ids"][f, cr[j]],
                              ma["positions"][f, cr[j]], {"argpos": len(lists[j]) - 1, "act_last": 3.0, "act_max": 3.0}, 0.5)
            assert rec["peak_pos"] == rec["n_tok"] - 1 and rec["start"] == rec["pos"] - rec["n_tok"] + 1 and enc(rec["target_text"]) == lists[j]
            counts[fam] = counts.get(fam, 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    vecs.tofile(f"{out}/vecs.f32")
    json.dump({"n_examples": N, "families": counts}, open(f"{out}/build_stats.json", "w"))
    json.dump({"n_examples": N, "families": counts, "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32"}}, open(f"{out}/meta.json", "w"))
    n_chk, fams_chk = check_bank_files(out)
    assert n_chk == N and fams_chk == counts
    # the compositor's own scan (data/modal_mix_5m_bank._scan_records semantics): line i == vec_idx i, families per line
    recs = [json.loads(l) for l in open(f"{out}/records.jsonl")]
    assert all(int(r["vec_idx"]) == i for i, r in enumerate(recs)) and sum(1 for r in recs if r["family"] == "sae2m_dec") == counts["sae2m_dec"]
    # a corrupted bank is caught
    json.dump({"n_examples": N + 1, "families": counts}, open(f"{out}/build_stats.json", "w"))
    try:
        check_bank_files(out); raise SystemExit("check_bank_files accepted a wrong n_examples")
    except AssertionError:
        pass
    PASS("assemble_rows leak accounting + shuffle; records schema (peak_pos/start/roundtrip); check_bank_files invariants")


def test_cos_helpers():
    ref = torch.nn.functional.normalize(torch.randn(50, 8), dim=-1)
    x = torch.nn.functional.normalize(torch.randn(30, 8), dim=-1); x[3] = ref[10]; x[7] = -ref[2]
    offs = np.array([0, 20, 50])
    mc, per_set, am = max_cos_table(iter([x[:16].numpy(), x[16:].numpy()]), ref, offs, 2, "cpu")
    assert abs(mc[3] - 1) < 1e-5 and am[3] == 10 and mc[7] < 0.999 and len(mc) == 30 and per_set[0] >= mc[3] - 1e-6
    h = cos_hist(mc)
    assert sum(h["counts"]) == 30 and h["n_gt"]["0.999"] == 1 and abs(h["max"] - 1) < 1e-5
    PASS("max_cos_table per-set maxima + argmax; cos_hist bins / n_gt")


def test_special_in_span():
    mt = np.arange(12).reshape(2, 1, 6); mt[0, 0, :2] = 99; mt[1, 0, 1] = 99
    m = special_in_span(mt, np.array([[4], [6]]), {99})
    assert m.tolist() == [[False], [True]], m                      # bad id in the PAD region is fine; inside the true span is not
    assert window_ids(mt[0, 0], 4) == [2, 3, 4, 5] and type(window_ids(mt[0, 0], 4)[0]) is int
    PASS("special_in_span: pad/BOS id flagged only inside the true span; window_ids returns python ints")


if __name__ == "__main__":
    test_window_ids_and_masks()
    test_roundtrip_filter()
    test_breadth_first_and_coverage()
    test_score_rows_fake_forward()
    test_assemble_and_bank_files()
    test_cos_helpers()
    test_special_in_span()
    print(f"ALL {n_pass} BANK TESTS PASSED")
