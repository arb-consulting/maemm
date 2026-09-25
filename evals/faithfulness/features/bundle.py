"""Read-only access to the v2 bundle (the immutable upstream data).

This is a thin reader, not a second source of truth: the bundle is an INPUT, mirrored
on the volume at data/v2-bundle/ per the migration plan in
docs/precompute-and-baselines.md. Nothing here writes to it.

TODO: once heldout/2026-09-1x_v2 is built, volume paths should come from
precompute/common.py rather than the modal CLI shell-out below, and this module
shrinks to the parquet schemas.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = "v2-bundle"
VOLUME = "maemm"
VOLUME_ROOT = f"data/{SNAPSHOT}"
S3_URI = "s3://ANONYMOUS-maemm-27b-data/v2-2026-09-17"

MODEL = "Qwen/Qwen3.6-27B"
READ_LAYER = 42
D_MODEL = 5120
CORPUS = "openbmb/Ultra-FineWeb"

CHECKPOINT = "ANONYMOUS/maemm-27b-rl-last16-lr5e-7"
CHECKPOINT_REVISION = "main"   # pin to the re-hosted repo's revision for exact reproduction

# generation_config.json of the checkpoint ships top_k=20, top_p=0.95; the eval
# protocol is top_k off, top_p=1. Always pass these explicitly.
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "min_new_tokens": 16, "max_new_tokens": 64}

# The primary SAE. The 131k SAE survives as the legacy `sae` family; its gate,
# corpus peaks and feature ids are not interchangeable with these.
PRIMARY_SAE = "2m"

# The families every baseline is precomputed over (the message's "stable" set).
STABLE_FAMILIES = ("realact", "realact_long", "sae2m_enc", "sae2m_dec", "random")
# Frozen and loadable, but outside the standard precompute.
OTHER_FAMILIES = (
    "sae", "realact_early", "realact_mid", "indist_long", "indist_probe",
    "indist_realact", "bsf", "jlens", "cluster", "mlp", "mlp_pair",
)
ALL_FAMILIES = STABLE_FAMILIES + OTHER_FAMILIES

# Families carrying an SAE feature id + corpus peak, so norm_act / fired are defined.
SAE_FAMILIES = {"sae2m_enc": "2m", "sae2m_dec": "2m", "sae": "131k"}

_CACHE = Path(os.environ.get("MAEMM_DATA_CACHE", Path.home() / ".cache" / "maemm" / SNAPSHOT))


@dataclass(frozen=True)
class Targets:
    """One frozen eval family. `directions` rows are unit vectors in raw layer-42 space."""
    family: str
    directions: np.ndarray          # [n, 5120] float32, unit rows
    meta: pd.DataFrame              # every non-direction column of the parquet
    snapshot: str = SNAPSHOT

    def __len__(self) -> int:
        return len(self.directions)

    @property
    def feature_ids(self) -> np.ndarray | None:
        for col in ("feature_id", "feats"):
            if col in self.meta.columns:
                return self.meta[col].to_numpy()
        return None

    @property
    def corpus_peaks(self) -> np.ndarray | None:
        if "corpus_peak" in self.meta.columns:
            return self.meta["corpus_peak"].to_numpy(dtype=np.float32)
        return None


def _modal_cmd() -> list[str]:
    """`modal` if it is on PATH, else `uvx modal`."""
    from shutil import which
    if which("modal"):
        return ["modal"]
    if which("uvx"):
        return ["uvx", "modal"]
    raise RuntimeError("neither `modal` nor `uvx` on PATH; set MAEMM_DATA_CACHE to a "
                       "directory already holding the snapshot instead")


def fetch(relpath: str) -> Path:
    """Path to `<snapshot>/<relpath>`, pulling it from the Modal volume on first use."""
    local = _CACHE / relpath
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    # utf-8 or the CLI's tick mark kills it on a cp1252 console *after* the download;
    # judge success by the file, not the exit code, for the same reason.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        _modal_cmd() + ["volume", "get", VOLUME, f"{VOLUME_ROOT}/{relpath}",
                        str(local), "--force"],
        capture_output=True, text=True, errors="replace", env=env,
    )
    if not local.exists():
        raise FileNotFoundError(
            f"{relpath} not retrieved from {VOLUME}:{VOLUME_ROOT}" + "\n"
            + proc.stderr[-800:])
    return local


def load_family(family: str) -> Targets:
    """The 512 frozen directions of `family` (256 for mlp_pair)."""
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {ALL_FAMILIES}")
    df = pd.read_parquet(fetch(f"heldout/eval_directions_v3/{family}.parquet"))
    dirs = np.stack(df["direction"].to_numpy()).astype(np.float32)
    if dirs.shape[1] != D_MODEL:
        raise ValueError(f"{family}: d={dirs.shape[1]}, expected {D_MODEL}")
    norms = np.linalg.norm(dirs, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError(f"{family}: directions not unit rows (norm range "
                         f"{norms.min():.4f}-{norms.max():.4f})")
    return Targets(family=family, directions=dirs,
                   meta=df.drop(columns=["direction"]).reset_index(drop=True))


def load_sae_features() -> pd.DataFrame:
    """Per-feature table of the 512 standard-eval 2M-SAE features.

    Columns: feature_id, enc_dir, dec_dir, b_enc, corpus_peak, gate. The gate is a
    column here on purpose -- three different gates are in circulation and a fourth
    (1.654) is stale, so nothing downstream may hardcode one.
    """
    return pd.read_parquet(fetch("heldout/eval_2m_features_512.parquet"))


def sae_gate() -> float:
    """The 2M SAE's learned BatchTopK gate, read off the data (expected 1.6828)."""
    gates = load_sae_features()["gate"].unique()
    if len(gates) != 1:
        raise ValueError(f"expected one gate across the eval features, got {gates}")
    return float(gates[0])


def load_feature_split() -> pd.DataFrame:
    """feature_id -> {eval, rl, sft} for all 2,097,152 features (seed 2026)."""
    return pd.read_parquet(fetch("heldout/feature_split.parquet"))


def load_maxact_windows() -> pd.DataFrame:
    """Top-5 32-token max-activating windows per eval feature (500,000 rows).

    Each window ENDS at its peak token -- not a centred window and not the stride-16
    geometry the corpus-search baseline uses. Do not mix the two.
    """
    return pd.read_parquet(fetch("heldout/eval_2m_features_100k_windows.parquet"))


def load_doc_registry() -> dict:
    """Document ids per train/eval source, with pairwise overlaps (train vs eval = 0)."""
    return json.loads(fetch("heldout/doc_registry.json").read_text())


def assert_disjoint() -> dict:
    """Re-derive the train/eval disjointness claim rather than trusting the README."""
    split = load_feature_split()
    counts = split["split"].value_counts().to_dict()
    eval_ids = set(split.loc[split["split"] == "eval", "feature_id"])
    out = {"split_counts": counts}
    for fam in ("sae2m_enc", "sae2m_dec"):
        ids = load_family(fam).feature_ids
        if ids is not None:
            out[f"{fam}_all_in_eval_split"] = bool(set(np.asarray(ids).ravel()) <= eval_ids)
    return out
