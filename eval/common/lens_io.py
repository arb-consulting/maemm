"""Jacobian-lens I/O with no package config: the lens file and its integrity pins, the unembed head, the
word-like vocabulary mask, full-vocabulary ranks, the all-layer transport pass and the pooled layer band."""

import hashlib, os
import numpy as np


def is_wordlike(s):
    st = s.strip()
    if not st or "<|" in st or (st.startswith("<") and st.endswith(">")):
        return False
    return all(ch.isalnum() or (0 < k < len(st) - 1 and ch in "'-’") for k, ch in enumerate(st))


# The pooled lens readout's fixed band: block outputs 36 to 50 in steps of 2 (the lens's zero-based numbering).
BAND8_LAYERS = tuple(range(36, 51, 2))


def pool_layers(top10_by_layer, fitted, layers):
    """Pool the top-10 lists of `layers` into one list: each token once (stripped, case-folded), ordered by
    its best position in any list, ties to the earlier layer."""
    fitted = [int(l) for l in fitted]
    best = {}
    for layer in layers:
        if int(layer) not in fitted:
            continue
        for pos, t in enumerate(top10_by_layer[fitted.index(int(layer))] or []):
            k = t.strip().casefold()
            if k and (k not in best or pos < best[k][0]):
                best[k] = (pos, int(layer), t)
    return [t for pos, layer, t in sorted(best.values(), key=lambda v: (v[0], v[1]))]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lens(device, repo, revision, file, n_bytes, sha256, read_layer, d_model):
    import jlens
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, file, revision=revision)
    if os.path.getsize(path) != n_bytes:
        raise RuntimeError(f"{path}: {os.path.getsize(path)} bytes != {n_bytes}")
    digest = _sha256(path)
    if digest != sha256:
        raise RuntimeError(f"{path}: sha256 {digest} != {sha256}")
    lens = jlens.JacobianLens.load(path)
    if read_layer not in lens.source_layers:
        raise RuntimeError(f"lens fits {lens.source_layers[0]}..{lens.source_layers[-1]}; {read_layer} missing")
    if lens.d_model != d_model:
        raise RuntimeError(f"lens d_model {lens.d_model} != {d_model}")
    lens.jacobians = {l: J.to(device) for l, J in lens.jacobians.items()}
    return lens, {
        "path": path,
        "sha256": digest,
        "bytes": n_bytes,
        "n_prompts": lens.n_prompts,
        "fitted_layers": [int(l) for l in lens.source_layers],
        "repo": repo,
        "revision": revision,
        "file": file,
    }


class Unembed:
    """final norm + lm_head in the head's dtype, logits float32 (the release's HFLensModel.unembed)."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.norm, self.head = base.model.norm, base.lm_head
        self.vocab = int(self.head.weight.shape[0])
        self.dtype = self.head.weight.dtype

    def __call__(self, x):
        import torch

        with torch.no_grad():
            return self.head(self.norm(x.to(self.dtype))).float()


def wordlike_mask(tok, vocab):
    mask = np.zeros(vocab, dtype=bool)
    for t in range(vocab):
        try:
            mask[t] = is_wordlike(tok.decode([t], clean_up_tokenization_spaces=False))
        except Exception:
            mask[t] = False
    return mask


def ranks_of(logits, target_ids, chunk=256):
    """1-based full-vocabulary ranks; logits [n, V], target_ids [F] long → [n, F]."""
    import torch

    n, V = logits.shape
    out = torch.empty(n, target_ids.shape[0], dtype=torch.long, device=logits.device)
    ar = torch.arange(V, device=logits.device)
    for s in range(0, n, chunk):
        idx = logits[s : s + chunk].argsort(dim=-1, descending=True)
        full = torch.empty_like(idx)
        full.scatter_(1, idx, ar.expand_as(idx))
        out[s : s + chunk] = full[:, target_ids] + 1
    return out


def layer_logits(H_rows, lens, unembed, layer, device=None):
    """Transport rows at `layer` through the lens and unembed: H_rows [n, d] -> float32 logits [n, V]."""
    X = H_rows
    if device is not None:
        import torch

        X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        X = X.to(device)
    return unembed(lens.transport(X, layer))


def top_words(rows, lens, unembed, tok, mask, layer, device=None, top_word=10):
    """The `top_word` word-like tokens, decoded and best first, the lens reads at `layer` for each row of
    `rows` [n, d]."""
    import torch

    Z = layer_logits(rows, lens, unembed, layer, device)
    keep = torch.as_tensor(mask, device=Z.device)
    idx = Z.masked_fill(~keep, float("-inf")).topk(top_word, dim=-1).indices.cpu().numpy()
    return [[tok.decode([int(t)]) for t in row] for row in idx]


def lens_pass(H, lens, unembed, tok, mask, form_ids, device, top_word=10, chunk=64):
    """H [n, 64, d] float32 numpy → ranks [n, L_fitted, F] int32, top10_by_layer [n][L_fitted] lists of decoded tokens."""
    import torch

    fitted = list(lens.source_layers)
    n = H.shape[0]
    F = len(form_ids)
    tids = torch.tensor(form_ids, device=device, dtype=torch.long)
    mask_t = torch.from_numpy(mask).to(device)
    ranks = np.zeros((n, len(fitted), F), dtype=np.int32)
    top = [[None] * len(fitted) for _ in range(n)]
    for li, layer in enumerate(fitted):
        for s in range(0, n, chunk):
            Z = layer_logits(H[s : s + chunk, layer], lens, unembed, layer, device)
            if F:
                ranks[s : s + chunk, li] = ranks_of(Z, tids).cpu().numpy()
            masked = Z.masked_fill(~mask_t, float("-inf")).topk(top_word, dim=-1).indices.cpu().numpy()
            for j in range(masked.shape[0]):
                top[s + j][li] = [tok.decode([int(t)]) for t in masked[j]]
            del Z
        if li % 8 == 7:
            print(f"  [lens] layer {li + 1}/{len(fitted)}", flush=True)
    return ranks, top
