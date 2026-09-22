"""Pure helpers for the 2M-SAE midtrain bank (CPU-testable; no Modal, no model).

Bank schema (== data/modal_bank_everything.py families "sae"/"sae_dec", so data/modal_mix_5m_bank._scan_records and
sft/pretrain.load_shard accept it unchanged):
    vecs.f32        [N, 5120] float32 UNIT rows; row i = the direction of records.jsonl line i
    records.jsonl   line i == vec_idx i; fields: vec_idx, family ("sae2m" | "sae2m_dec"), feature, window_rank, target_text,
                    n_tok, peak_pos (= n_tok-1), fire_from_end 0, peak_idx_in_window, corpus_peak, window_peak, fire_count,
                    doc_id, pos (peak token index in the doc), start (= pos-n_tok+1), end_anchored True, anchor_* (standalone
                    re-check numbers), enc_dec_cos
    build_stats.json / meta.json   n_examples, families {fam: count}, family_recipes, leak_check, end_anchor_filter, ...

Selection (select_windows): for every LIVE feature (fire_count > 0 and top activation > threshold) the stored top-N windows are
candidates in rank order when they have >= min_tok true tokens, contain no pad/BOS id inside their true span, decode -> re-tokenise
to exactly their ids (roundtrip) and are not a text duplicate of an earlier kept window of the same feature. breadth_first_select
then takes rank-0 of every feature first, then rank-1, ... (breadth before depth) until windows_per_feature per feature or the
row cap; the pass that would cross the cap takes a seeded random subset of its features so coverage is uniform over feature ids.
"""
import json
import os

import numpy as np

D_MODEL = 5120
FAMILY_ENC = "sae2m"
FAMILY_DEC = "sae2m_dec"


# ----------------------------------------------------------------------------------------------------------------
# candidate windows
# ----------------------------------------------------------------------------------------------------------------
def live_mask(max_acts, fire_counts, threshold):
    """max_acts [F, N] (fp16/fp32, -1 = empty), fire_counts [F] -> bool [F]: fire_count > 0 AND top activation > threshold."""
    ma = np.asarray(max_acts, dtype=np.float32)
    fc = np.asarray(fire_counts)
    return (fc > 0) & (ma.max(1) > float(threshold))


def candidate_mask(max_acts, lengths, threshold, min_tok):
    """bool [F, N]: window slot has an activation > threshold and >= min_tok true tokens."""
    ma = np.asarray(max_acts, dtype=np.float32)
    ln = np.asarray(lengths).astype(np.int64)
    return (ma > float(threshold)) & (ln >= int(min_tok))


def special_in_span(max_tokens, lengths, bad_ids):
    """bool [F, N]: some id of `bad_ids` (pad / BOS) occurs INSIDE the true span of the window (the last `lengths` positions).
    Vectorised over the whole [F, N, L] table (the Python per-window check is too slow for 10M candidates)."""
    mt = np.asarray(max_tokens); ln = np.asarray(lengths).astype(np.int64)
    L = mt.shape[-1]
    isbad = np.isin(mt, np.asarray(sorted(set(int(b) for b in bad_ids)), mt.dtype))
    in_span = np.arange(L)[None, None, :] >= (L - ln)[:, :, None]
    return (isbad & in_span).any(-1)


def window_ids(max_tokens_row, length):
    """max_tokens_row [L] int (window ENDING at the peak, left-padded) -> the `length` true token ids as a python list."""
    L = len(max_tokens_row)
    length = int(length)
    assert 0 < length <= L, (length, L)
    return max_tokens_row[L - length:].tolist()


def roundtrip_texts(tok, ids_lists, bad_ids=(), batch=50_000):
    """Decode each id list standalone and re-tokenise (add_special_tokens=False). Returns (texts, ok bool[n], reason str[n]).
    ok requires: no id of `bad_ids` inside the window, re-encoding gives exactly the same ids, and the text is not blank."""
    n = len(ids_lists)
    texts = [""] * n
    ok = np.zeros(n, bool)
    reason = np.full(n, "", dtype=object)
    bad = set(int(b) for b in bad_ids)
    for b0 in range(0, n, batch):
        idx = list(range(b0, min(n, b0 + batch)))
        chunk = [ids_lists[j] for j in idx]
        dec = tok.batch_decode(chunk, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        re_enc = tok(dec, add_special_tokens=False)["input_ids"]
        for j, ids, txt, ids2 in zip(idx, chunk, dec, re_enc):
            texts[j] = txt
            if bad and any(t in bad for t in ids):
                reason[j] = "special_id_in_window"
            elif len(txt.strip()) < 3:
                reason[j] = "blank"
            elif list(ids2) != list(ids):
                reason[j] = "roundtrip_fail"
            else:
                ok[j] = True
                reason[j] = "ok"
    return texts, ok, reason


def dedupe_texts(feat, texts, ok):
    """feat [n] int (any order), texts list[n], ok bool[n]. Returns dup bool[n]: True where the (feat, text) pair already
    appeared at a lower index with ok=True. Processed in index order, so sort candidates by (feat, rank) first."""
    seen = set()
    dup = np.zeros(len(texts), bool)
    for j in range(len(texts)):
        if not ok[j]:
            continue
        key = (int(feat[j]), texts[j])
        if key in seen:
            dup[j] = True
        else:
            seen.add(key)
    return dup


# ----------------------------------------------------------------------------------------------------------------
# breadth-first selection
# ----------------------------------------------------------------------------------------------------------------
def breadth_first_select(feat, rank, valid, windows_per_feature, cap, seed=0):
    """feat/rank int[n] (parallel; candidates), valid bool[n]. Per feature the valid candidates are ordered by rank and get
    depth 0, 1, 2, ... Returns (sel bool[n], info). Depth d is taken for every feature that has it before depth d+1 for any
    feature; if adding a depth level would exceed `cap` selected windows, a seeded random subset of that level's features is
    taken and selection stops. windows_per_feature bounds the depth. cap <= 0 -> no cap."""
    feat = np.asarray(feat, np.int64); rank = np.asarray(rank, np.int64); valid = np.asarray(valid, bool)
    n = len(feat)
    sel = np.zeros(n, bool)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return sel, {"n_candidates": int(n), "n_valid": 0, "taken": 0, "per_depth": [], "cap": int(cap), "cap_hit": False,
                     "features_with_valid": 0}
    order = idx[np.lexsort((rank[idx], feat[idx]))]                     # sorted by (feat, rank) among valid
    f_sorted = feat[order]
    first = np.r_[True, f_sorted[1:] != f_sorted[:-1]]
    grp_start = np.flatnonzero(first)
    grp_id = np.cumsum(first) - 1
    depth = np.arange(len(order)) - grp_start[grp_id]                  # position within the feature's valid list
    rng = np.random.default_rng(seed)
    taken = 0
    per_depth = []
    cap_hit = False
    K = int(windows_per_feature)
    for d in range(K):
        lvl = order[depth == d]
        if len(lvl) == 0:
            break
        if cap and cap > 0 and taken + len(lvl) > cap:
            room = int(cap - taken)
            if room > 0:
                pick = np.sort(rng.permutation(len(lvl))[:room])
                sel[lvl[pick]] = True
                taken += room
                per_depth.append({"depth": d, "available": int(len(lvl)), "taken": int(room), "partial": True})
            else:
                per_depth.append({"depth": d, "available": int(len(lvl)), "taken": 0, "partial": True})
            cap_hit = True
            break
        sel[lvl] = True
        taken += len(lvl)
        per_depth.append({"depth": d, "available": int(len(lvl)), "taken": int(len(lvl)), "partial": False})
    info = {"n_candidates": int(n), "n_valid": int(len(idx)), "taken": int(taken), "per_depth": per_depth, "cap": int(cap),
            "cap_hit": bool(cap_hit), "windows_per_feature": K, "features_with_valid": int(len(grp_start)), "seed": int(seed)}
    return sel, info


def coverage_hist(feat_selected, n_features_total, K):
    """Windows-per-feature histogram over ALL features (index 0 = features with no selected window). Returns list length K+1
    (values > K are clipped into the last bin, which cannot happen for breadth-first output)."""
    cnt = np.bincount(np.asarray(feat_selected, np.int64), minlength=int(n_features_total))
    cnt = np.minimum(cnt, K)
    return np.bincount(cnt, minlength=K + 1).tolist()


# ----------------------------------------------------------------------------------------------------------------
# end-anchor scoring (pure: takes a forward function; the worker plugs the real model in)
# ----------------------------------------------------------------------------------------------------------------
def score_rows(forward_fn, ids_lists, feat_idx, W_enc_table, b_enc_table, b_dec, bos, batch=256, log=None):
    """End-anchor check for rows i: text ids ids_lists[i] (already the standalone tokenisation), encoder row feat_idx[i] of
    W_enc_table [m, d] / b_enc_table [m] (any float dtype, on the same device as b_dec [d]).
    forward_fn(ids LongTensor [B, T+1]) -> hidden [B, T+1, d] (any float dtype) at the read layer; position 0 (BOS) is dropped.
    pre = relu((x - b_dec) . W_enc[f] + b_enc[f]) computed in fp32 over the T content tokens.
    Returns dict of np arrays: argpos (0-based over content tokens), act_last, act_max, n_tok. Rows are grouped by length so no
    padding is ever used (the GDN layers have no padding semantics)."""
    import torch
    n = len(ids_lists)
    lens = np.array([len(x) for x in ids_lists], np.int64)
    feat_idx = np.asarray(feat_idx, np.int64)
    argpos = np.full(n, -1, np.int64); act_last = np.zeros(n, np.float32); act_max = np.zeros(n, np.float32)
    dev = b_dec.device
    b_dec32 = b_dec.float()
    order = np.argsort(lens, kind="stable")
    done = 0
    i0 = 0
    with torch.no_grad():
        while i0 < n:
            L = int(lens[order[i0]])
            i1 = i0
            while i1 < n and i1 - i0 < batch and int(lens[order[i1]]) == L:
                i1 += 1
            kb = order[i0:i1]; i0 = i1
            if L < 1:
                continue
            ids = torch.tensor([[bos] + [int(t) for t in ids_lists[k]] for k in kb], device=dev, dtype=torch.long)
            h = forward_fn(ids)
            a = h[:, 1:, :].float()                                           # [B, L, d]
            fi = torch.as_tensor(feat_idx[kb], device=dev)
            w = W_enc_table[fi].float()                                       # [B, d]
            b = b_enc_table[fi].float()                                       # [B]
            pre = torch.relu(torch.einsum("bld,bd->bl", a - b_dec32, w) + b[:, None])
            argpos[kb] = pre.argmax(1).cpu().numpy()
            act_last[kb] = pre[:, -1].cpu().numpy()
            act_max[kb] = pre.max(1).values.cpu().numpy()
            done += len(kb)
            if log is not None and (done % 100_000 < len(kb)):
                log(f"scored {done}/{n}")
    return {"argpos": argpos, "act_last": act_last, "act_max": act_max, "n_tok": lens}


def anchor_summary(argpos, n_tok, act_last, act_max, threshold, rule="last"):
    """Pass mask + summary for the end-anchor rule ('last': peak == last token; 'last2': within the last 2)."""
    argpos = np.asarray(argpos); n_tok = np.asarray(n_tok)
    ok = argpos >= 0
    off = n_tok - 1 - argpos
    pass_last = ok & (off == 0)
    pass_last2 = ok & (off <= 1)
    keep = pass_last if rule == "last" else pass_last2
    fire_last = (np.asarray(act_last) > float(threshold)) & ok
    ratio = np.asarray(act_last) / np.maximum(np.asarray(act_max), 1e-9)
    summ = {"rule": rule, "n": int(len(argpos)), "n_scored": int(ok.sum()), "pass_last": float(pass_last.sum() / max(ok.sum(), 1)),
            "pass_last2": float(pass_last2.sum() / max(ok.sum(), 1)), "fire_last_rate(>thr)": float(fire_last.sum() / max(ok.sum(), 1)),
            "act_last_over_act_max_median": float(np.median(ratio[ok])) if ok.any() else None,
            "peak_offset_from_end_hist": {str(int(d)): int(c) for d, c in zip(*np.unique(off[ok], return_counts=True)) if d <= 10},
            "n_keep": int(keep.sum()), "n_fail": int((~keep).sum()), "threshold": float(threshold)}
    return keep, summ


# ----------------------------------------------------------------------------------------------------------------
# cosine helpers (torch; device-agnostic)
# ----------------------------------------------------------------------------------------------------------------
def max_cos_table(x_iter, ref, offs, n_ref_sets, dev, dtype=None):
    """x_iter yields float32 [c, d] unit chunks (numpy); ref [M, d] unit on dev; offs = cumulative boundaries of the ref sets.
    Returns (maxcos [n] overall, per_set_max [n_ref_sets] max over rows, argmax [n] ref index)."""
    import torch
    out_max, out_arg = [], []
    per_set = np.full(n_ref_sets, -1.0)
    with torch.no_grad():
        for chunk in x_iter:
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(dev)
            if dtype is not None:
                cos = (x.to(dtype) @ ref.to(dtype).T).float()
            else:
                cos = x @ ref.T
            mx, am = cos.max(1)
            out_max.append(mx.cpu().numpy()); out_arg.append(am.cpu().numpy())
            for ri in range(n_ref_sets):
                per_set[ri] = max(per_set[ri], float(cos[:, offs[ri]:offs[ri + 1]].max()))
    return np.concatenate(out_max) if out_max else np.zeros(0, np.float32), per_set, (np.concatenate(out_arg) if out_arg else np.zeros(0, np.int64))


COS_BINS = [-1.0, 0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 1.0001]


def cos_hist(vals):
    """Histogram of max-cos values over COS_BINS -> {"bins": [...], "counts": [...], "frac_gt": {0.9:..,0.95:..,0.99:..,0.999:..}}."""
    v = np.asarray(vals, np.float64)
    counts, _ = np.histogram(v, bins=COS_BINS)
    return {"bins": COS_BINS, "counts": counts.tolist(), "n": int(len(v)),
            "frac_gt": {str(t): float((v > t).mean()) if len(v) else 0.0 for t in (0.8, 0.9, 0.95, 0.97, 0.99, 0.999)},
            "n_gt": {str(t): int((v > t).sum()) for t in (0.8, 0.9, 0.95, 0.97, 0.99, 0.999)},
            "percentiles": {str(q): float(np.percentile(v, q)) for q in (50, 90, 99, 99.9)} if len(v) else {},
            "max": float(v.max()) if len(v) else None}


def assemble_rows(win_idx, win_feat_pos, leak_pos_by_family, families, seed):
    """Final rows from the final windows: one row per (window, family) unless the window's feature direction of that family is
    leak-flagged. win_idx int[n] (window ids), win_feat_pos int[n] (position of the window's feature in the direction table),
    leak_pos_by_family {family: int[] flagged table positions}. Returns (row_fam int8[N] index into `families`, row_win int[N],
    dropped {family: n}) after a seeded global shuffle."""
    win_idx = np.asarray(win_idx, np.int64); win_feat_pos = np.asarray(win_feat_pos, np.int64)
    n_tab = int(win_feat_pos.max()) + 1 if len(win_feat_pos) else 0
    fam_rows, fam_win, dropped = [], [], {}
    for fi, fam in enumerate(families):
        bad = np.zeros(n_tab, bool)
        flagged = np.asarray(leak_pos_by_family.get(fam, []), np.int64)
        if len(flagged):
            bad[flagged] = True
        keep = ~bad[win_feat_pos] if n_tab else np.zeros(0, bool)
        dropped[fam] = int((~keep).sum())
        fam_rows.append(np.full(int(keep.sum()), fi, np.int8)); fam_win.append(win_idx[keep])
    row_fam = np.concatenate(fam_rows) if fam_rows else np.zeros(0, np.int8)
    row_win = np.concatenate(fam_win) if fam_win else np.zeros(0, np.int64)
    perm = np.random.default_rng(seed).permutation(len(row_fam))
    return row_fam[perm], row_win[perm], dropped


# ----------------------------------------------------------------------------------------------------------------
# bank writing
# ----------------------------------------------------------------------------------------------------------------
def make_record(vec_idx, family, feature, window_rank, text, n_tok, corpus_peak, window_peak, fire_count, doc_id, pos,
                anchor, enc_dec_cos):
    """One records.jsonl line (dict). `anchor` = dict(argpos, act_last, act_max) of the standalone re-check."""
    return {"vec_idx": int(vec_idx), "family": family, "feature": int(feature), "window_rank": int(window_rank), "target_text": text,
            "n_tok": int(n_tok), "peak_pos": int(n_tok) - 1, "fire_from_end": 0, "peak_idx_in_window": int(n_tok) - 1,
            "corpus_peak": round(float(corpus_peak), 4), "window_peak": round(float(window_peak), 4), "fire_count": int(fire_count),
            "doc_id": int(doc_id), "pos": int(pos), "start": int(pos) - int(n_tok) + 1, "end_anchored": True,
            "anchor_argpos": int(anchor["argpos"]), "anchor_act_last": round(float(anchor["act_last"]), 4),
            "anchor_act_max": round(float(anchor["act_max"]), 4), "enc_dec_cos": round(float(enc_dec_cos), 4), "sae": "sae2m"}


def check_bank_files(out, d_model=D_MODEL):
    """The invariants data/modal_mix_5m_bank._scan_records + sft/pretrain rely on. Returns (n, families)."""
    st = json.load(open(f"{out}/build_stats.json"))
    n = int(st["n_examples"])
    fams = st["families"]
    assert isinstance(fams, dict) and sum(fams.values()) == n, (fams, n)
    assert os.path.getsize(f"{out}/vecs.f32") == n * d_model * 4, (os.path.getsize(f"{out}/vecs.f32"), n)
    cnt = {}
    with open(f"{out}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            assert int(r["vec_idx"]) == i, (i, r["vec_idx"])
            assert isinstance(r["target_text"], str) and r["target_text"]
            cnt[r["family"]] = cnt.get(r["family"], 0) + 1
    assert i + 1 == n, (i + 1, n)
    assert cnt == fams, (cnt, fams)
    meta = json.load(open(f"{out}/meta.json"))
    assert meta["families"] == fams and int(meta["n_examples"]) == n
    return n, fams
