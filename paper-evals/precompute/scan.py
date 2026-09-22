"""Product `scan`: pass B over the corpus -- the corpus-retrieval baseline and the SAE examples.

    <root>/base/<base>/scan/<set>/       topk.jsonl, quantiles.f16 [N, n_sizes, 5]
    <root>/base/<base>/sae/<sae>/examples/  <feature>.jsonl for every TESTED feature, _random256.jsonl

Same windows as pass A (`common.windows_of`, 64 tokens every 16, never crossing documents, sink
prepended and dropped), same stored document order, so the per-size snapshots are the same nested
subsets. The cosine is fp32 over every non-sink position, with NO norm filter (checklist item 4
and the scorer's divergence note in common.score_tokens).

TWO CENTRING MODES. `--centre` takes BOTH sides about the base's scoring constant
(`common.score_mu` = `bases.<base>.whiten_mu`), which is the same mean `score` reports its
`cos_centred` about: a corpus top-1 and a rollout cosine are then the same statistic and may be
differenced. Without it the window side is UNCENTRED while the targets carry whatever `--mu`
resolved -- Celeste's original asymmetry, kept so that every corpus-search number measured between
2026-09-16 and 2026-09-21 reproduces. The two write to different directories through `--run-tag`
(`scan_dir`'s key), because they are different numbers.

Masking (checklist item 53): for a realact target, the windows of ITS OWN document that overlap the
shown span [p-L+1, p] are excluded from both the top-k and the quantiles. Exact or near-duplicate
documents elsewhere in the corpus are NOT masked -- that is a real property of a corpus baseline
and is reported rather than engineered away.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

import precompute.common as C

TOPK = 64
N_BINS = 512  # cos histogram over [-1, 1]; bin width 1/256, so a quantile is exact to 1/256
QUANTILES = [0.50, 0.90, 0.99, 0.999, 0.9999]
TGT_CHUNK = 512  # targets per histogram sub-step, to bound the int64 index transient
SAE_TOP = 128  # top windows kept per tested feature
SAE_PER_BIN = 32  # windows sampled from each of 4 activation bins of (0, max_act]
SAE_RANDOM = 256  # shared random windows
SENTINEL = -2.0  # cos of a masked / padded position: below every real cosine


class _Heap:
    """Running top-k over windows for [n_rows] parallel streams, merged on the GPU each batch.

    eval/corpus_retrieval.py:Heaps is the working example: with n_rows in the thousands and a few
    hundred candidates per batch, a full sort of [n_rows, k + batch] is microseconds next to the
    forward and needs no python-side heap.
    """

    def __init__(self, n_rows, k, device, payload_shape=()):
        import torch

        self.k = k
        self.val = torch.full((n_rows, k), -float("inf"), device=device)
        self.win = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.arg = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.payload = (
            torch.zeros((n_rows, k, *payload_shape), dtype=torch.float16, device=device)
            if payload_shape
            else None
        )

    def push(self, val, win, arg, payload=None):
        """val/arg: [n_rows, b]; win: [b]; payload: [n_rows, b, *shape]."""
        import torch

        n = val.shape[0]
        cat_v = torch.cat([self.val, val], 1)
        cat_w = torch.cat([self.win, win.unsqueeze(0).expand(n, -1)], 1)
        cat_a = torch.cat([self.arg, arg], 1)
        v, i = cat_v.topk(self.k, dim=1)
        self.val, self.win, self.arg = v, cat_w.gather(1, i), cat_a.gather(1, i)
        if self.payload is not None:
            cat_p = torch.cat([self.payload, payload], 1)
            self.payload = cat_p.gather(1, i.unsqueeze(-1).expand(-1, -1, cat_p.shape[-1]))


class _KeyReservoir:
    """Uniform sample of `k` items per row, by smallest random key (A-Res with equal weights).

    Equivalent to reservoir sampling and fully vectorised: every candidate gets one U(0,1) key from
    a seeded torch generator and the k smallest keys survive, so the result depends on the seed and
    the batch order only, never on where a flush happened to fall.
    """

    def __init__(self, n_rows, k, device, payload_shape=()):
        import torch

        self.k = k
        self.key = torch.full((n_rows, k), float("inf"), device=device)
        self.win = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.arg = torch.zeros((n_rows, k), dtype=torch.int64, device=device)
        self.val = torch.zeros((n_rows, k), device=device)
        self.payload = (
            torch.zeros((n_rows, k, *payload_shape), dtype=torch.float16, device=device)
            if payload_shape
            else None
        )

    def push(self, key, win, arg, val, payload=None):
        import torch

        n = key.shape[0]
        cat_k = torch.cat([self.key, key], 1)
        cat_w = torch.cat([self.win, win.unsqueeze(0).expand(n, -1)], 1)
        cat_a = torch.cat([self.arg, arg], 1)
        cat_v = torch.cat([self.val, val], 1)
        kk, i = cat_k.topk(self.k, dim=1, largest=False)
        self.key, self.win, self.arg, self.val = (
            kk,
            cat_w.gather(1, i),
            cat_a.gather(1, i),
            cat_v.gather(1, i),
        )
        if self.payload is not None:
            cat_p = torch.cat([self.payload, payload], 1)
            self.payload = cat_p.gather(1, i.unsqueeze(-1).expand(-1, -1, cat_p.shape[-1]))


def _reservoir_seed(cfg, set_name: str) -> int:
    """The SAE reservoir generator's seed for `--set <set>`, WITHOUT requiring a `seed:` key.

    `scan` read `cfg["heldout"][<set>]["seed"]` directly, and `load_config` never required that
    key on a `heldout:` entry -- so every eval-1 v3 block (`2026-09-21_v3_realact`, `_ctrl`,
    `_ours`, `_subspace`, `_realact_long`, all of them imported rather than drawn, none of them
    carrying a draw seed) crashed this product with a KeyError *after* the base model had loaded.
    MEASURED on the config 2026-09-23: only `2026-09-16_v1`, `2026-09-16_v1raw`,
    `2026-09-20_sae2m_2k`, `2026-09-21_sae2m_64`, `2026-09-21_v3_sae2m` and the OOD sets declare one.

    A declared seed still wins, so every scan run before 2026-09-23 reproduces bit for bit. An
    undeclared one falls back to crc32 of the set name: deterministic, different per set, and
    written into the examples README by `_examples_notes` exactly as a declared seed is, so the
    number is recoverable from the product rather than from this docstring.
    """
    seed = (cfg["heldout"][set_name] or {}).get("seed")
    return int(seed) if seed is not None else int(zlib.crc32(set_name.encode("utf-8")))


def _load_targets(cfg, args, notes=None):
    """(ids rows, V [N, d] unit fp32 on the gpu, mask tables).

    The rows are the `--set` set's, plus every `--with-set <name>[:<fam>,<fam>]` set appended after
    it -- design §4: one scan of a corpus carries ALL the targets it could ever be asked about (the
    OOD arms, the 512 English realact targets and the 512 random directions), because the scan's
    cost is per corpus token and not per target.

    `scan` has no `--maemm` in scope at all, so the centring convention has to be told to it:

      * `--centre` is THE CENTRED MODE (2026-09-23). Both sides of the cosine are taken about the
        base's own scoring constant `common.score_mu` -- the targets here, the window residuals in
        `_scan_one`'s flush() -- which is the same constant `score` reports its `cos_centred`
        about, so a corpus top-1 and a rollout cosine are finally the same statistic. It takes no
        value and cannot be combined with `--mu`: the whole point is that the mean is not a
        per-run choice. Every target set must be `storage: raw`, since a stored unit direction
        cannot be re-centred (common.dirs_for).
      * `--mu <file>`, or the set's own stored contract, is the LEGACY uncentred-window mode,
        which reproduces every corpus-search number measured between 2026-09-16 and 2026-09-21
        exactly. The resolution is PER SET, because `--with-set` can append a `storage: unit` bank
        to a `storage: raw` one and the two were not centred the same way.

    The own-document mask travels with the target as `mask_corpus`: a realact target's `doc` is an
    index into ITS OWN corpus and means nothing in another one, so the mask is applied only where
    the scanned corpus is that corpus. Without that condition an English target's document index
    would mask an unrelated document of, say, the Thai corpus, silently.
    """
    import os

    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    d = cfg["bases"][base]["d"]
    specs = [(set_name, None)]
    for extra in [x for x in (args.get("with_set") or "").split(",") if x]:
        name, _, fams = extra.partition(":")
        specs.append((name, [f for f in fams.split("+") if f] or None))
    centre = bool(args.get("centre"))
    assert not (centre and (args.get("mu") or "").strip()), (
        "--centre and --mu are two answers to one question: --centre takes BOTH sides of the "
        "cosine about the base's scoring constant (common.score_mu) and is not a per-run choice"
    )
    rows, vecs = [], []
    for name, fams in specs:
        hdir = C.heldout_dir(base, name, root)
        assert os.path.exists(f"{hdir}/ids.jsonl"), f"no held-out set at {hdir}"
        rs = C.read_jsonl(f"{hdir}/ids.jsonl")
        if centre:
            storage = C.set_storage(cfg, hdir, root)["storage"]
            assert storage == "raw", (
                f"--centre needs every target set to be `storage: raw` so a centred direction can "
                f"be derived from its act.f32; {name} is `storage: {storage}` and a stored unit "
                f"direction cannot be re-centred (common.dirs_for)"
            )
            mu = C.score_mu(cfg, base)
        else:
            mu, _ = C.mu_for(cfg, base, hdir, args, "", root, notes)
        v = np.asarray(C.dirs_for(cfg, base, hdir, mu, root, notes), dtype=np.float32)
        assert v.shape == (len(rs), d), f"{hdir}: dirs_for returned {v.shape} for {len(rs)} rows"
        for i, r in enumerate(rs):
            assert r["row"] == i, f"{hdir}/ids.jsonl row {i} says row={r['row']}"
            if fams is not None and r["family"] not in fams:
                continue
            rows.append({**r, "set": name, "set_row": r["row"]})
            vecs.append(v[i])
        print(f"[scan] targets from {name}: {sum(1 for r in rows if r['set'] == name)} rows", flush=True)
    n = len(rows)
    assert n, "no targets selected"
    V = torch.nn.functional.normalize(torch.from_numpy(np.stack(vecs)).cuda(), dim=-1)
    doc = torch.full((n,), -1, dtype=torch.int64)
    lo = torch.zeros(n, dtype=torch.int64)
    hi = torch.zeros(n, dtype=torch.int64)
    mask_corpus = []
    foreign = 0
    for i, r in enumerate(rows):
        r["row"] = i  # the row index WITHIN this scan; `set` + `set_row` is the join key
        if r["family"] == "realact" and all(k in r for k in ("doc", "p", "L")):
            doc[i], lo[i], hi[i] = r["doc"], r["p"] - r["L"] + 1, r["p"]
            mask_corpus.append("corpus")  # realact targets come from the base's own English corpus
        elif r["family"] == "realact":
            # HER realact draw (`source: hers`, the eval-1 headline block 2026-09-21_v3_realact)
            # carries `doc` and `pos` and NO `p`/`L`, and its `doc` indexes HER v2 collection, not
            # our `corpus/`. Masking document 10,984 of OUR corpus because her row says `doc:
            # 10984` is the same silent error the mask-by-corpus rule (B, 2026-09-21) was written
            # against, one axis over: the index is foreign to every corpus this product can scan.
            # So the row is UNMASKABLE here, and the own-document exclusion for her block is the
            # n-gram exclusion in the set's `exclusions.json` (spec 1.4), applied by the reader
            # that drops rows -- not by this window mask. Counted and printed, never silent.
            foreign += 1
            mask_corpus.append("")
        else:
            # Nothing to mask. An OOD target carries `pool_i`, not `doc`: its document comes from
            # the arm's TARGET POOL, which is the rows the corpus build did not consume (design
            # §2), so it is not in that corpus -- or in any other -- and there is no window of it
            # to exclude. A `random` or `sae` row has no document at all.
            assert "doc" not in r, (
                f"row {i} of set {r['set']} has a `doc` field but family {r['family']!r}, so "
                f"nobody here knows which corpus that index belongs to; give it a mask_corpus "
                f"label rather than letting it search its own document"
            )
            mask_corpus.append("")
    if foreign:
        msg = (
            f"[scan] {foreign} realact rows carry no (doc, p, L) in THIS pipeline's corpus index "
            f"space and are UNMASKABLE: their own-document exclusion is the set's exclusions.json, "
            f"applied downstream by dropping rows, not by this window mask"
        )
        print(msg, flush=True)
        (notes if notes is not None else []).append(msg[len("[scan] "):])
    # The window side's mean, for flush(): the SAME constant the targets above were centred on, or
    # None in the legacy mode where only the target side is centred (an asymmetric cosine, which
    # is why `cos_asym` exists in `score` and why `results/ood.py` had to read it).
    wmu = None
    if centre:
        arr = C.load_mu(cfg, base, C.score_mu(cfg, base), root)
        assert arr is not None and arr.shape == (d,), (
            f"the scoring constant {C.score_mu(cfg, base)} did not resolve to a [{d}] mean"
        )
        wmu = torch.from_numpy(np.asarray(arr, dtype=np.float32)).cuda()
        one_sided = sorted({r["family"] for r in rows if not C.family_centrable(cfg, r["family"])})
        (notes if notes is not None else []).append(
            f"--centre: BOTH sides about {C.mu_label(C.score_mu(cfg, base), base, root)}, the "
            f"scoring constant (common.score_mu), the same mean `score` reports cos_centred "
            f"about. Families {one_sided} are not `centrable`, so their target side is the raw "
            f"unit direction while the window side is centred -- a ONE-SIDED number for those "
            f"rows, exactly as it is in `cos_asym`; a reader must not report them as centred"
            if one_sided else
            f"--centre: BOTH sides about {C.mu_label(C.score_mu(cfg, base), base, root)}, the "
            f"scoring constant (common.score_mu); every target family here is `centrable`"
        )
    return rows, V, (doc, lo, hi, mask_corpus), wmu


def _forward(model, read_layer, rows, sink, pad_id):
    """(h [B, T, d] fp32, keep [B, T]) -- the same batch shape stats.py pass A uses."""
    import torch

    width = 1 + max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    am = np.zeros((len(rows), width), dtype=np.int64)
    ids[:, 0] = sink
    am[:, 0] = 1
    for i, r in enumerate(rows):
        ids[i, 1 : 1 + len(r)] = r
        am[i, 1 : 1 + len(r)] = 1
    batch = {
        "input_ids": torch.from_numpy(ids).cuda(),
        "attention_mask": torch.from_numpy(am).cuda(),
    }
    h, mask = C.read_resid(model, read_layer, batch, pool="all")
    keep = mask.clone()
    keep[:, 0] = False
    return h, keep


def run(cfg, args):
    """Pass B over ONE OR MORE corpora, with the same target bank.

    `--corpus <a>,<b>,...` scans the OOD arm corpora `corpora/<arm>/` instead of the base's own
    English `corpus/`; `--max-size M` stops at the M-million-token nested prefix (the `examples_4m`
    trick: a bounded prefix of an existing corpus, at a fraction of the cost). Several corpora in
    one call share the model load, which is the fixed cost of a short scan.

    Output: `scan/<set>/` exactly as before when NEITHER flag is given, and `scan/<set>/<corpus>/`
    (with `-<M>m` appended when the scan is bounded) otherwise -- so the existing English scan is
    never touched by this path.
    """
    import os

    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product scan needs --base"
    d = cfg["bases"][base]["d"]
    batch_rows = int(args.get("batch") or 256)
    # ONE --sae syntax in the whole CLI: common.sae_key_for, which takes a full `<base>/<name>`
    # key and refuses a bare name. This file and stats.py each carried an inline copy that DID
    # accept a bare `sae2m`, so the same flag meant two things depending on the product.
    #
    # Resolved only when the SET HAS SAE ROWS. A base with two dictionaries makes `sae_key_for`
    # refuse without `--sae`, and an OOD set has no sae family at all -- so demanding one there
    # would make the operator name a dictionary this scan never reads, and the choice would then
    # sit in the product README as if it meant something.
    # WHICH CORPORA, as directory names already resolved from `corpora:` keys on the client
    # (modal_app.main). NOT filtered for empties: "" is the base's own English `corpus/`, so
    # `--corpus heldout16m,ood_tha_Thai` arrives as ",ood_tha_Thai" and means both of them.
    corpora = (args.get("corpus_name") or "").split(",")
    max_size = int(args.get("max_size") or 0)
    # A repeat would plan two scans into one output directory; the second refuses on "already
    # exists" only AFTER the first has been paid for.
    assert len(set(corpora)) == len(corpora), f"--corpus repeats a corpus: {corpora}"

    cen_notes: list[str] = []
    rows, v, masks, wmu = _load_targets(cfg, args, notes=cen_notes)
    sae_key = C.sae_key_for_rows(cfg, base, rows, args.get("sae") or "")
    # Filtered on the ROW's own sae_key, not on the family label: a set carrying two dictionaries
    # under `family: sae` would otherwise have the other dictionary's feature ids looked up in this
    # encoder, silently (common.sae_rows_of).
    sae_sel = C.sae_rows_of(
        rows, sae_key, declared=C.declared_sae_key(cfg, C.heldout_dir(base, set_name, root), root),
        where=C.heldout_dir(base, set_name, root),
    ) if sae_key else []
    tested = [int(r["id"]) for r in sae_sel]
    tested_row = [r["row"] for r in sae_sel]

    # `scan/<set>` stays the name of the ONE unbounded scan of the base's own English corpus, so
    # every number measured between 2026-09-16 and 2026-09-21 keeps its path. Anything else is
    # keyed (C.scan_dir, H5) by the corpus AND the bound: a 1M-bounded scan of a 4M corpus is a
    # different number from the full one, and `<set>__4m` alone could not be told apart from the
    # scan of a corpus whose directory is literally `4m`.
    sub = len(corpora) > 1 or bool(max_size) or any(corpora)
    # THE THIRD AXIS, as `rollout_stem` has it: two scans of one (set, corpus) that differ only in
    # `--mu` are different experiments, and the targets they score against are different vectors
    # (MEASURED on 2026-09-21_ood_q1: stats/mu.f32 and whiten_mu agree at cos 0.977 and put
    # unit(act - mu) a median cos 0.969 apart). Without a tag the second would refuse on "already
    # exists" -- or, with --force, destroy the first. Empty for every scan run so far, so no
    # existing path moves.
    tag = (args.get("run_tag") or "").strip()
    assert "/" not in tag and " " not in tag, f"--run-tag {tag!r} must be a bare name suffix"
    plans = []
    for raw in corpora:
        # `--corpus heldout16m` already resolves to the empty directory name on the client
        # (common.corpus_key_name), so an empty list element IS the base's own English corpus.
        # The literal `corpus` is kept as the spelling for it on the `--corpus-name` escape
        # hatch, where a leading empty element cannot be typed.
        cname = "" if raw == "corpus" else raw
        label = cname or "corpus"
        key = (label + (f"__{max_size}m" if max_size else "")) if sub else ""
        if tag:
            key = f"{key}__{tag}" if key else tag
        out_scan = C.scan_dir(base, set_name, root, key)
        # The SAE examples are a product of a scan whose SET has sae targets. An OOD scan has none,
        # so it writes no examples/ -- and must not, because that directory is shared.
        # KEYED BY SET AND CORPUS (B9, 2026-09-21): `examples/` used to be keyed by SAE alone, so a
        # second scan of the same dictionary against a different set refused without --force and
        # DESTROYED the first set's examples with it. Keyed by the SAME string as the scan half,
        # so the two halves of one call can never drift apart.
        out_ex = (
            C.sae_examples_dir(sae_key, set_name, root, write=True, corpus_name=key)
            if tested else ""
        )
        for path in (out_scan, out_ex):
            assert not path or args.get("force") or not os.path.exists(path), (
                f"{path} already exists; refusing to overwrite without --force"
            )
        plans.append((cname, label, out_scan, out_ex))

    model, tok = C.load_base(cfg, base)
    # ENCODER ONLY (D3): everything below reads b_dec, W_enc, b_enc and threshold -- the tested
    # columns at :200-201 and the gate. W_dec is 43 GB in fp32 at 2^21 features, which is the
    # difference between fitting an H200 beside the 27B and not. stats.py:379 already had this.
    sae = (
        C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32,
                   need_decoder=False)
        if tested else None
    )
    out = {}
    for cname, label, out_scan, out_ex in plans:
        bound = f" (<= {max_size}M)" if max_size else ""
        print(f"[scan] === corpus {label}{bound} -> {out_scan}", flush=True)
        out[label] = _scan_one(
            cfg, args, model, tok, sae, sae_key, rows, v, masks, wmu, cname, label, out_scan, out_ex,
            tested, tested_row, batch_rows, max_size, cen_notes,
        )
    return out


def _scan_one(
    cfg, args, model, tok, sae, sae_key, rows, v, masks, wmu, cname, label, out_scan, out_ex,
    tested, tested_row, batch_rows, max_size, cen_notes,
):
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    read_layer = cfg["bases"][base]["read_layer"]
    n, n_feat = len(rows), len(tested)
    t_doc_c, t_lo_c, t_hi_c, mask_corpus = masks
    # The own-document mask applies only to targets whose OWN corpus is the one being scanned.
    # Compared on the LABEL, not on `cname`: the base's own English corpus is `cname == ""` and
    # `label == "corpus"`, and comparing on cname silently dropped the mask for every realact
    # target in a scan of that corpus -- which is the own-document inflation review R1 is about.
    keep_mask = torch.tensor([mc == label for mc in mask_corpus], dtype=torch.bool)
    t_doc = torch.where(keep_mask, t_doc_c, torch.full_like(t_doc_c, -1)).cuda()
    t_lo, t_hi = t_lo_c.cuda(), t_hi_c.cuda()
    n_masked_rows = int(keep_mask.sum())

    # H7: refuse a corpus this pipeline would cut at a geometry its config does not declare.
    C.assert_corpus_geometry(cfg, cname)
    toks, docs = C.load_corpus(base, root, cname)
    sizes = C.corpus_sizes(docs)
    if max_size:
        assert max_size in sizes, f"--max-size {max_size} is not one of {label}'s sizes {sizes}"
        sizes = [s for s in sizes if s <= max_size]
        docs = [r for r in docs if r["size_tag"] <= max_size]
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    w_enc = sae.W_enc[:, torch.as_tensor(tested, device="cuda")].contiguous() if n_feat else None
    b_enc = sae.b_enc[torch.as_tensor(tested, device="cuda")] if n_feat else None
    peak_t = None
    if n_feat:
        peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (sae.d_sae,))
        peak_t = torch.from_numpy(peak[tested].astype(np.float32)).cuda()
    print(
        f"[scan] {n} targets ({n_feat} tested sae features, {n_masked_rows} own-document masks), "
        f"{len(docs)} docs, sizes {sizes}",
        flush=True,
    )

    heap = _Heap(n, TOPK, "cuda")
    hist = torch.zeros((n, N_BINS), dtype=torch.int64, device="cuda")
    snap_top, snap_hist = [], []
    f_top = _Heap(n_feat, SAE_TOP, "cuda", payload_shape=(C.SCAN_BLOCK,)) if n_feat else None
    f_bins = (
        [_KeyReservoir(n_feat, SAE_PER_BIN, "cuda", payload_shape=(C.SCAN_BLOCK,)) for _ in range(4)]
        if n_feat
        else []
    )
    r_res = _KeyReservoir(1, SAE_RANDOM, "cuda", payload_shape=(n_feat,)) if n_feat else None
    gen = torch.Generator(device="cuda").manual_seed(_reservoir_seed(cfg, set_name))

    win_doc: list = []
    win_start: list = []
    win_len: list = []
    buf, buf_meta, w_global = [], [], 0
    t0, fwd_tok, done_tokens = time.time(), 0, 0

    def flush():
        nonlocal fwd_tok, w_global
        if not buf:
            return
        h, keep = _forward(model, read_layer, buf, sink, pad_id)
        b, t = keep.shape
        wd = torch.tensor([m[0] for m in buf_meta], device="cuda")
        ws = torch.tensor([m[1] for m in buf_meta], device="cuda")
        wl = torch.tensor([len(r) for r in buf], device="cuda")
        wid = torch.arange(w_global, w_global + b, device="cuda")

        # [B, T, N] fp32. `wmu` is the scoring constant under --centre and None otherwise; the
        # subtract is one broadcast over [B, T, d] per flush, no extra pass and no extra
        # allocation beyond the centred copy, so memory stays where it was (512 targets x 10M
        # tokens is bounded by `cos`, not by this).
        hc = h if wmu is None else h - wmu
        cos = torch.nn.functional.normalize(hc, dim=-1) @ v.T
        cos = cos.masked_fill(~keep.unsqueeze(-1), SENTINEL)
        # realact self-match mask: this window's document and span overlap the target's own span
        mask = (
            (wd.unsqueeze(1) == t_doc.unsqueeze(0))
            & (ws.unsqueeze(1) <= t_hi.unsqueeze(0))
            & ((ws + wl - 1).unsqueeze(1) >= t_lo.unsqueeze(0))
        )  # [B, N]
        cos = cos.masked_fill(mask.unsqueeze(1), SENTINEL)
        best, arg = cos.max(dim=1)  # [B, N]
        heap.push(best.T.contiguous(), wid, arg.T.contiguous())
        for c0 in range(0, n, TGT_CHUNK):
            c1 = min(c0 + TGT_CHUNK, n)
            sub = cos[:, :, c0:c1]
            bins = ((sub + 1.0) * (N_BINS / 2)).long().clamp_(0, N_BINS - 1)
            bins = torch.where(sub > -1.5, bins, torch.full_like(bins, N_BINS))  # sentinel bin
            off = torch.arange(c1 - c0, device="cuda") * (N_BINS + 1)
            flat = (bins + off).reshape(-1)
            cnt = torch.bincount(flat, minlength=(c1 - c0) * (N_BINS + 1))
            hist[c0:c1] += cnt.reshape(c1 - c0, N_BINS + 1)[:, :N_BINS]

        if n_feat:
            a = torch.relu((h - sae.b_dec) @ w_enc + b_enc)  # [B, T, n_feat] pre-gate
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            amax, aarg = a.max(dim=1)  # [B, n_feat]
            pay = torch.zeros((b, C.SCAN_BLOCK, n_feat), dtype=torch.float16, device="cuda")
            pay[:, : min(t - 1, C.SCAN_BLOCK)] = a[:, 1 : 1 + C.SCAN_BLOCK].to(torch.float16)
            pay = pay.permute(2, 0, 1).contiguous()  # [n_feat, B, 64]
            f_top.push(amax.T.contiguous(), wid, aarg.T.contiguous(), pay)
            q = torch.clamp(torch.ceil(amax / peak_t.clamp(min=1e-6) * 4).long() - 1, 0, 3)
            key = torch.rand((b, n_feat), generator=gen, device="cuda")
            for qi in range(4):
                sel = (q == qi) & (amax > 0)
                f_bins[qi].push(
                    torch.where(sel, key, torch.full_like(key, float("inf"))).T.contiguous(),
                    wid,
                    aarg.T.contiguous(),
                    amax.T.contiguous(),
                    pay,
                )
            rkey = torch.rand((1, b), generator=gen, device="cuda")
            r_res.push(
                rkey,
                wid,
                torch.zeros((1, b), dtype=torch.int64, device="cuda"),
                torch.zeros((1, b), device="cuda"),
                amax.unsqueeze(0).to(torch.float16),
            )

        win_doc.append(np.asarray([m[0] for m in buf_meta], dtype=np.int32))
        win_start.append(np.asarray([m[1] for m in buf_meta], dtype=np.int32))
        win_len.append(np.asarray([len(r) for r in buf], dtype=np.int32))
        fwd_tok += int(keep.numel())
        w_global += b
        buf.clear()
        buf_meta.clear()

    si = 0
    for r in docs:
        while sizes[si] < r["size_tag"]:
            flush()
            snap_top.append((heap.val.cpu().numpy(), heap.win.cpu().numpy(), heap.arg.cpu().numpy()))
            snap_hist.append(hist.cpu().numpy())
            print(
                f"[scan] size {sizes[si]}M snapshot at {done_tokens} tokens, {time.time() - t0:.0f}s",
                flush=True,
            )
            si += 1
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, ln in C.windows_of(r["len"]):
            buf.append(ids[s : s + ln])
            buf_meta.append((r["doc"], s))
            if len(buf) >= batch_rows:
                flush()
        done_tokens += r["len"]
        if r["doc"] % 5000 == 0 and r["doc"]:
            el = time.time() - t0
            print(
                f"[scan] doc {r['doc']}/{len(docs)} {done_tokens / 1e6:.2f}M tokens | "
                f"{fwd_tok / max(el, 1e-9):.0f} fwd tok/s | {done_tokens / max(el, 1e-9):.0f} corpus tok/s",
                flush=True,
            )
    flush()
    assert si == len(sizes) - 1, f"ended at size index {si} of {len(sizes)}"
    snap_top.append((heap.val.cpu().numpy(), heap.win.cpu().numpy(), heap.arg.cpu().numpy()))
    snap_hist.append(hist.cpu().numpy())
    elapsed = time.time() - t0
    wdoc = np.concatenate(win_doc)
    wstart = np.concatenate(win_start)
    wlen = np.concatenate(win_len)
    assert len(wdoc) == w_global, f"window table {len(wdoc)} != {w_global} forwarded windows"

    inputs = {
        "corpus": C.corpus_dir(base, root, cname),
        "corpus label": label,
        "heldout": C.heldout_dir(base, set_name, root),
        "with_set": args.get("with_set") or "-",
        "targets": n,
        "windows": int(w_global),
        "sizes": sizes,
        "max_size": max_size or "-",
    }
    with C.outdir(out_scan, args, inputs=inputs) as od:
        C.note_convention(od, cen_notes)
        lines, n_dropped = [], 0
        for si_, size in enumerate(sizes):
            val, win, arg = snap_top[si_]
            for i in range(n):
                top = []
                for j in range(TOPK):
                    if val[i, j] <= -1.5:  # fewer than TOPK unmasked windows at this size
                        n_dropped += 1
                        continue
                    w = int(win[i, j])
                    top.append([int(wdoc[w]), int(wstart[w]), int(arg[i, j] - 1), round(float(val[i, j]), 5)])
                lines.append(
                    {
                        "row": i,
                        "set": rows[i]["set"],
                        "set_row": rows[i]["set_row"],
                        "family": rows[i]["family"],
                        "arm": rows[i].get("arm"),
                        "corpus": label,
                        "size": size,
                        "top": top,
                    }
                )
        od.write_jsonl("topk.jsonl", lines)
        q = np.stack([C.quantiles_from_hist(hs, QUANTILES) for hs in snap_hist], axis=1)  # [N, sizes, 5]
        od.write_array("quantiles.f16", q, "float16")
        od.note(
            f"topk.jsonl: one line per (target row, corpus size) with up to {TOPK} entries "
            "[doc, window start, argmax position WITHIN the window (0-based, sink already "
            "dropped), cos]; the running top-k is snapshotted at every size boundary"
        )
        od.note(
            f"quantiles.f16 is [N={n}, n_sizes={len(sizes)}, {len(QUANTILES)}] for q="
            f"{QUANTILES}, from a {N_BINS}-bin histogram of cos on [-1, 1] (bin width "
            f"{2 / N_BINS}), so each value is the bin upper edge and exact to {2 / N_BINS}"
        )
        od.note(
            (
                "cosine is CENTRED on BOTH sides about the base's scoring constant "
                f"({C.mu_label(C.score_mu(cfg, base), base, root)}, common.score_mu -- the same "
                "mean `score` reports cos_centred about), "
                if wmu is not None else
                "cosine is UNCENTRED on the window side (the target side carries whatever `--mu` "
                "resolved), "
            )
            + "fp32, over every non-sink position of every window, with no norm "
            f"filter; window geometry {C.SCAN_BLOCK}/{C.SCAN_STRIDE} (common.windows_of), "
            f"{w_global} windows over {done_tokens} corpus tokens"
        )
        od.note(
            f"{n_masked_rows} of {n} targets mask the windows of their OWN document overlapping "
            f"[p-L+1, p] -- those whose own corpus IS `{label}`; a target of another corpus masks "
            "nothing here, because its document index means nothing in this one. Exact and "
            "near-duplicate documents elsewhere in the corpus are NOT masked"
        )
        if n_dropped:
            od.note(f"{n_dropped} top-k slots were empty (masked or too few windows) and omitted")
        od.note(
            f"throughput: {fwd_tok / elapsed:.0f} forwarded tok/s, {done_tokens / elapsed:.0f} "
            f"corpus tok/s over {elapsed:.0f}s"
        )

    ex_rows = 0
    if not n_feat:
        print("[scan] no sae targets in this set: no examples/ product written", flush=True)
    with _maybe_outdir(out_ex, args, inputs={**inputs, "sae": sae_key, "tested": n_feat}) as od:
        C.note_convention(od, cen_notes)
        if n_feat:
            tv, tw, ta, tp = (
                f_top.val.cpu().numpy(),
                f_top.win.cpu().numpy(),
                f_top.arg.cpu().numpy(),
                f_top.payload.cpu().numpy(),
            )
            bins = [
                (
                    b.val.cpu().numpy(),
                    b.win.cpu().numpy(),
                    b.arg.cpu().numpy(),
                    b.payload.cpu().numpy(),
                    b.key.cpu().numpy(),
                )
                for b in f_bins
            ]
            nbytes = 0
            for fi, feat in enumerate(tested):
                recs = []
                for j in range(SAE_TOP):
                    if not np.isfinite(tv[fi, j]) or tv[fi, j] <= 0:
                        continue
                    recs.append(
                        _ex(
                            tested_row[fi],
                            "top",
                            tv[fi, j],
                            tw[fi, j],
                            ta[fi, j],
                            tp[fi, j],
                            wdoc,
                            wstart,
                            wlen,
                        )
                    )
                for qi, (bv, bw, ba, bp, bk) in enumerate(bins):
                    for j in range(SAE_PER_BIN):
                        if not np.isfinite(bk[fi, j]):
                            continue
                        recs.append(
                            _ex(
                                tested_row[fi],
                                f"q{qi}",
                                bv[fi, j],
                                bw[fi, j],
                                ba[fi, j],
                                bp[fi, j],
                                wdoc,
                                wstart,
                                wlen,
                            )
                        )
                path = od.file(f"{feat}.jsonl")
                C.write_jsonl(path, recs)
                nbytes += path.stat().st_size
                ex_rows += len(recs)
            rw, rp, rk = (
                r_res.win.cpu().numpy(),
                r_res.payload.cpu().numpy(),
                r_res.key.cpu().numpy(),
            )
            rrecs = [
                {
                    "window": int(rw[0, j]),
                    "doc": int(wdoc[rw[0, j]]),
                    "start": int(wstart[rw[0, j]]),
                    "len": int(wlen[rw[0, j]]),
                    "max_act": [round(float(x), 4) for x in rp[0, j]],
                }
                for j in range(SAE_RANDOM)
                if np.isfinite(rk[0, j])
            ]
            path = od.file("_random256.jsonl")
            C.write_jsonl(path, rrecs)
            od.index["examples"] = {
                "kind": "jsonl",
                "rows": ex_rows + len(rrecs),
                "bytes": nbytes + path.stat().st_size,
            }
            od.write_json("tested.json", {"features": tested, "rows": tested_row, "sae": sae_key})
            _examples_notes(od, n_feat, sae.d_sae, w_global, _reservoir_seed(cfg, set_name))

    return {
        "scan": out_scan,
        "corpus": label,
        "examples": out_ex or "-",
        "windows": int(w_global),
        "topk_rows": len(lines),
        "example_rows": ex_rows,
        "fwd_tok_per_s": round(fwd_tok / elapsed, 1),
        "corpus_tok_per_s": round(done_tokens / elapsed, 1),
        "seconds": round(elapsed, 1),
    }


def _ex(row, kind, val, win, arg, acts, wdoc, wstart, wlen):
    """One example record. `acts` is TRUNCATED to the window's real length: the payload buffer is
    64 wide, and leaving the tail in would be indistinguishable from a genuinely inactive token."""
    w = int(win)
    ln = int(wlen[w])
    return {
        "row": row,
        "kind": kind,
        "window": w,
        "doc": int(wdoc[w]),
        "start": int(wstart[w]),
        "len": ln,
        "max_act": round(float(val), 4),
        "argmax": int(arg) - 1,
        "acts": [round(float(x), 4) for x in acts[:ln]],
    }


class _NullOut:
    """Stand-in for an OutDir when a product is not produced at all (an OOD scan writes no SAE
    examples). Every call is a no-op, so the block below it needs no second code path."""

    index: dict = {}

    def note(self, line):
        pass

    def write_json(self, name, obj):
        pass

    def file(self, name):
        raise AssertionError("no examples directory is being written")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _examples_notes(od, n_feat: int, d_sae: int, w_global: int, seed) -> None:
    """The `examples/` product's own notes. A FUNCTION because they read `sae.d_sae`, and they sat
    outside the `if n_feat:` that guards every other SAE branch here -- so a scan of a set with no
    SAE rows built the f-string against `sae = None` and died AFTER the scan had been paid for and
    its topk.jsonl renamed into place."""
    od.note(
        f"one <feature>.jsonl per TESTED feature ({n_feat} of {d_sae}), each with the top "
        f"{SAE_TOP} windows by activation plus {SAE_PER_BIN} windows sampled from each of 4 "
        "equal-width activation bins of (0, max_act] (max_act from the pass-A stats)"
    )
    od.note(
        "`acts` is the per-token pre-gate activation of that window, f16-rounded, in window "
        "order with the sink dropped and truncated to the window's `len`; `argmax` is the "
        "position of the maximum within it; the window's TEXT is recoverable from "
        "corpus/tokens.i32 at (doc, start, len)"
    )
    od.note(
        f"_random256.jsonl: {SAE_RANDOM} windows sampled uniformly over ALL {w_global} windows "
        "(one shared negative pool), each carrying the per-tested-feature MAX activation in the "
        "column order of tested.json -- per-token activations are not stored for these"
    )
    od.note("examples are taken over the FULL corpus only; they are not snapshotted per size")
    od.note(
        "sampling: smallest-random-key reservoir (equivalent to reservoir sampling), torch "
        f"generator seeded {seed}"
    )


def _maybe_outdir(path, args, **kw):
    return C.outdir(path, args, **kw) if path else _NullOut()
