"""The shipped centring mean every package centres activations with, loaded and integrity-checked."""

import hashlib, json, os
import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
CENTRING_MEAN = os.path.join(ASSETS, "mu.f32")
CENTRING_MEAN_META = os.path.join(ASSETS, "mu.meta.json")


def load_centring_mean(path=CENTRING_MEAN, meta_path=CENTRING_MEAN_META):
    """The layer-42 centring mean `mu` (`assets/README.md`) as a float32 vector; raises if its sha256 or
    width differs from `mu.meta.json`."""
    with open(meta_path) as f:
        meta = json.load(f)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: sha256 {digest} is not the recorded {meta['sha256']}")
    mu = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if mu.shape != (meta["d"],):
        raise ValueError(f"{path}: shape {mu.shape}, expected ({meta['d']},)")
    return mu
