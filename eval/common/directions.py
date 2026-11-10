"""The Gaussian random-direction family: unit directions that stand for nothing, to measure what an
instrument credits to no signal (the reconstruction evaluation's `random` targets). The whole family is
drawn from a private CPU generator and then cut, so row i is always the family's row i.
"""

import numpy as np

GAUSSIAN_SEED = 20260916
FAMILY_SIZE = 512


def gaussian_directions(n, d, seed=GAUSSIAN_SEED):
    """The first `n` rows of the family in `d` dimensions, float32 [n, d] unit rows."""
    import torch
    from torch.nn import functional as F

    n, d = int(n), int(d)
    if n < 0 or d < 1:
        raise ValueError(f"cannot draw {n} directions in {d} dimensions")
    generator = torch.Generator().manual_seed(int(seed))
    family = F.normalize(torch.randn(max(n, FAMILY_SIZE), d, generator=generator), dim=-1)
    return family[:n].numpy().astype(np.float32, copy=True)
