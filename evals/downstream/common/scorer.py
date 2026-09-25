"""The re-read scorer every package scores texts with: the maximum, over a text's content tokens, of the
cosine between the clean base model's layer-`READ_LAYER` residual and a direction.

A text is tokenised without special tokens and cut at `max_length`, a sink token (bos, else eos) is
prepended and never a candidate. The optional norm filter drops tokens whose residual norm exceeds
`NORM_FILTER_MULT` x the text's median; a `new_filter_stats()` tally records what it drops or would drop.
"""

NORM_FILTER_MULT = 10.0
SCORE_MAX_LENGTH = 95  # content tokens; the sink is one more


def new_filter_stats():
    """The tally `reencode(stats=...)` adds to: texts and content tokens read, and what the filter drops."""
    return {"n_texts": 0, "n_tokens": 0, "n_tokens_filter_drops": 0, "n_texts_filter_touches": 0}


def reencode(texts, model, tok, device, sbatch=32, max_length=SCORE_MAX_LENGTH, norm_filter=True, stats=None):
    """Yield `(s, h [b, T, d], keep [b, T] bool, mask [b, T] bool)` per sub-batch starting at text `s`:
    `keep` the candidate positions, `mask` the attention mask including the sink."""
    import torch

    from maemm.config import READ_LAYER
    from maemm.inject import read_resid

    prev = tok.padding_side
    tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(texts), sbatch):
            batch = [t if t.strip() else " " for t in texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=int(max_length), add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype),
                             enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype),
                            enc["attention_mask"]], 1)
            with torch.no_grad():
                h, mask = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone()
            keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            filtered = keep & (nrm <= NORM_FILTER_MULT * med)
            if stats is not None:
                dropped = keep & ~filtered
                stats["n_texts"] += B
                stats["n_tokens"] += int(keep.sum())
                stats["n_tokens_filter_drops"] += int(dropped.sum())
                stats["n_texts_filter_touches"] += int(dropped.any(dim=1).sum())
            yield s, h, (filtered if norm_filter else keep), mask
    finally:
        tok.padding_side = prev


def score_probe_cos(texts, dirs, model, tok, device, max_length=SCORE_MAX_LENGTH, norm_filter=True, stats=None):
    """Row-aligned scores: text `i` against unit direction `dirs[i]` ([N, d] tensor), as a float [N] tensor."""
    import torch
    import torch.nn.functional as F

    out = torch.zeros(len(texts))
    with torch.no_grad():
        for s, h, keep, _mask in reencode(texts, model, tok, device, max_length=max_length,
                                          norm_filter=norm_filter, stats=stats):
            d = dirs[s:s + h.shape[0]].to(device).float()
            cos = torch.einsum("btd,bd->bt", F.normalize(h.float(), dim=-1), d)
            out[s:s + h.shape[0]] = cos.masked_fill(~keep, -1.0).max(1).values.float().cpu()
    return out
