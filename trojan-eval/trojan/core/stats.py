"""Statistics shared by every experiment: binomial intervals, keyword hits, the logit lens.

These lived in the 5-trojan training driver, which meant every evaluation imported the trainer to
get a confidence interval. They are pure functions; `logit_lens` takes the model explicitly.
"""
import math

import torch
import torch.nn.functional as F

from trojan.core.lora import unwrap


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion. Correct at k=0 and k=n, which the
    normal approximation is not, and most cells here are at one extreme or the other."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def hits(texts, keys):
    return sum(any(k in t.lower() for k in keys) for t in texts)


def payload_hits(texts, keys):
    """A payload counts only if EVERY one of its distinctive terms appears -- these payloads are
    multi-word and unrelated to their triggers, so an incidental single word is not evidence."""
    return sum(sum(k in t.lower() for k in keys) >= max(2, len(keys) // 3) for t in texts)


# ---------------------------------------------------------------------------------------------
# A: train

@torch.no_grad()
def logit_lens(model, tok, vec, k=15, chunk=16384):
    """Top-k vocabulary tokens by cos(W_U[t], direction), after the model's final norm."""
    base = unwrap(model)
    W_U = base.lm_head.weight.detach()
    v = F.normalize(vec.float(), dim=0)
    best_v, best_i = [], []
    for s in range(0, W_U.shape[0], chunk):
        c = F.cosine_similarity(W_U[s : s + chunk].float(), v.unsqueeze(0), dim=1)
        vv, ii = c.topk(min(k, c.shape[0]))
        best_v.append(vv)
        best_i.append(ii + s)
    v_all, i_all = torch.cat(best_v), torch.cat(best_i)
    vv, order = v_all.topk(k)
    return [{"token": tok.decode([int(i_all[j])]), "cos": round(float(vv[n]), 4)}
            for n, j in enumerate(order.tolist())]
