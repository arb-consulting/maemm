"""Shared reading, discovery and estimation for `results/` -- the paper's results driver.

Local, CPU, no GPU, no model. Everything here READS what the volume's products already wrote;
nothing recomputes an activation or a cosine. Two ways in, and they are the same code path:

  * `Vol(...)` fetches one small file at a time off the Modal volume with `modal volume get`,
    caching into a LOCAL MIRROR whose subpaths are the volume's own -- `reconstruction/stats.py`'s
    `Vol`, with the `--root` prefix generalised from its two-name enum to any volume-relative
    directory (a smoke writes under `tmp/<name>`), and with `ls` carrying the same offline branch.
  * `--no-fetch` answers everything from that mirror, which is how `reconstruction/sae_smoke64.py`
    runs and how the selftest runs: the readers below are its readers, moved here rather than
    copied, so a product this module reads and one `sae_smoke64.py` reads cannot drift apart.

WHAT IS DISCOVERED, AND WHY NOTHING IS HARDCODED. The set names, the checkpoints, the SAE
dictionaries and the family labels all come from `config.yaml` and from the rows' own fields
(`family`, `sae_key`, `sae_side`, `stratum`, `doc`). A new SAE or a new MAEMM is a config entry
plus its products on the volume, and this module picks it up with no edit here -- which is the
requirement the `--set`-driven design exists for. The one axis that is NOT in config is the
`--run-tag`, because a tag is a property of a run and not of a checkpoint: it is read off the
scores directory names under `maemms/<base>/<maemm>/scores/` (`common.rollout_stem`'s third
component), so two arms of one checkpoint appear as two sources.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
CONFIG = PAPER_EVALS / "config.yaml"

VOLUME = "maemm"
# `precompute/common.ENGINES`. A scores directory name is `<set>[__<engine>][__<run-tag>]`, so the
# suffix parts are split on this: a part that IS an engine names the engine, the rest is the tag.
ENGINES = ("hf", "vllm")
# `precompute/score.BO_KS` -- the k values a product's `per_target.jsonl` can carry. A k above the
# run's n is absent from the file (never clamped), which is why every reader here treats a missing
# bo_k as "this run could not compute it" and prints it as such.
BO_KS_ALL = (1, 2, 4, 8, 16, 32, 64)
# The three the paper's tables report (plan §2.3: k = 1 (mean), 8, 64).
BO_KS_REPORT = (1, 8, 64)

# Bootstrap resamples for the clustered standard error. 2,000 is enough for two significant
# figures of an SE, which is all a table prints.
N_BOOT = 2000
BOOT_SEED = 20260921

# The categorical hues, in FIXED order, from the validated default palette (dataviz skill,
# references/palette.md, light mode; `validate_palette.js` passes all six checks on these). They
# are assigned to a SOURCE by its stable sort position and never cycled: a `--sources` filter that
# drops one must not repaint the survivors, so the assignment is computed once over all discovered
# sources and then looked up per figure.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300")
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------------------------
# the volume, seen through `modal volume ls/get`, with a local mirror
# ---------------------------------------------------------------------------------------------


class Vol:
    """`reconstruction/stats.py`'s Vol with an arbitrary volume-relative root prefix.

    Every fetch is cached by existence, so re-running the tables costs no network. A file that is
    not there is recorded in `self.missing` and returns None -- the caller decides whether its
    table can still be built, which is what makes "run with missing sources" a normal outcome
    rather than an exception.
    """

    def __init__(self, root: str, data_dir: Path, modal_cmd: str = "uvx modal",
                 refetch: bool = False, quiet: bool = False, offline: bool = False):
        self.prefix = (root or "").strip("/")
        self.local = Path(data_dir)
        self.cmd = modal_cmd.split()
        self.refetch = refetch
        self.quiet = quiet
        self.offline = offline
        self.missing: list[str] = []
        self.fetched = 0
        self.bytes = 0

    def _remote(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def ls(self, rel: str) -> list[str]:
        """Basenames under a volume directory, or [] when it does not exist."""
        if self.offline:
            d = self.local / rel
            return sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        r = subprocess.run(
            [*self.cmd, "volume", "ls", VOLUME, self._remote(rel)], capture_output=True, text=True
        )
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith(("Directory", "─", "┃", "│", "┏", "┡", "└")):
                out.append(line.rstrip("/").split("/")[-1])
        return sorted(out)

    def get(self, rel: str) -> Path | None:
        """Fetch one file; its local path, or None (and logged) when it is not on the volume."""
        dst = self.local / rel
        if dst.exists() and not self.refetch:
            return dst
        if self.offline:
            self.missing.append(rel)
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [*self.cmd, "volume", "get", "--force", VOLUME, self._remote(rel), str(dst)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not dst.exists():
            self.missing.append(rel)
            return None
        self.fetched += 1
        self.bytes += dst.stat().st_size
        if not self.quiet:
            print(f"[fetch] {rel} ({dst.stat().st_size / 1e6:.2f} MB)", flush=True)
        return dst

    def exists(self, rel: str) -> bool:
        """Is the file in the mirror, or (online) on the volume? Never fetches it."""
        if (self.local / rel).exists():
            return True
        if self.offline:
            return False
        parent = str(Path(rel).parent)
        return Path(rel).name in self.ls("" if parent == "." else parent)

    def jsonl(self, rel: str) -> list[dict] | None:
        p = self.get(rel)
        return None if p is None else read_jsonl(p)

    def json(self, rel: str) -> dict | None:
        p = self.get(rel)
        if p is None:
            return None
        with open(p) as fh:
            return json.load(fh)

    def array(self, rel: str, dtype: str, shape) -> np.ndarray | None:
        p = self.get(rel)
        return None if p is None else read_array(p, dtype, shape)

    def size_mb(self, rel: str, index: dict | None) -> float | None:
        """Bytes of a product file from its directory's `index.json`, in MB, or None if unknown.

        Used to decide whether an array-backed cross-check is affordable BEFORE downloading it:
        `cos.f16` is [N, n, T] and reaches ~63 MB per arm at the paper's full scale.
        """
        entry = (index or {}).get(Path(rel).name)
        return None if entry is None else float(entry.get("bytes", 0)) / 1e6


# ---------------------------------------------------------------------------------------------
# readers -- every one of them tolerates an absent source and says so
# ---------------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def read_array(path: str | Path, dtype: str, shape):
    """`precompute/common.read_array` without importing the package: this module runs standalone."""
    return np.fromfile(path, dtype=dtype).reshape(shape)


def bo_weights(n: int, k: int) -> np.ndarray:
    """Order-statistic weights of the UNBIASED best-of-k estimator from n observed scores.

        E[max of k draws] = sum_{i=1..n} x_(i) * C(i-1, k-1) / C(n, k)        (x sorted ASCENDING)

    The i-th smallest of the n observed scores is the maximum of a k-subset exactly when the other
    k-1 members come from the i-1 scores below it, and every k-subset of the n is equally likely.
    """
    assert 1 <= k <= n, f"best-of-{k} asked of {n} draws"
    denom = math.comb(n, k)
    return np.array([math.comb(i - 1, k - 1) / denom for i in range(1, n + 1)], dtype=np.float64)


def bo_unbiased(vals, k: int):
    """THE best-of-k estimator of this pipeline. [n] -> float, or [rows, n] -> [rows].

    ONE estimator everywhere since 2026-09-23 (M0a): `score`'s stored `bo_<k>` ladder
    (`precompute/common.bo_ladder`), every cell the three results drivers print, and the fired
    indicator. Before that the products stored the DISJOINT-GROUP mean -- floor(n/k) consecutive
    groups of k, each group's max, averaged -- while `reconstruction/stats.py` printed the
    unbiased one, so the same quantity had two values depending on which file a reader opened,
    and only at k = n did they agree.

    Unbiased for any k <= n and uses ALL n draws, rather than the floor(n/k)*k the group estimator
    reaches; at k = n the two coincide. Checklist item 25.

    NaN is NOT tolerated: a row with a missing draw has fewer than n draws and its k-subsets are
    not equally likely, so the caller drops or fills first and says which (see
    `results/faithfulness.centred_bok`).
    """
    a = np.asarray(vals, dtype=np.float64)
    one = a.ndim == 1
    a = a.reshape(1, -1) if one else a
    assert a.ndim == 2, f"bo_unbiased takes [n] or [rows, n], got shape {a.shape}"
    out = np.sort(a, axis=1) @ bo_weights(a.shape[1], int(k))
    return float(out[0]) if one else out


def bo_ladder(vals, ks) -> dict[int, float]:
    """{k: unbiased best-of-k} over a 1-D [n] of per-draw scores. k > n is SKIPPED, never clamped,
    so a summary never claims a bo-k it could not compute."""
    a = np.asarray(vals, dtype=np.float64).ravel()
    n = a.size
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        assert k >= 1, f"best-of-k needs k >= 1, got {k}"
        if k <= n:
            out[k] = bo_unbiased(a, k)
    return out


def peaks_of(act_row: np.ndarray) -> np.ndarray:
    """[n] per-rollout peak activation from one row's [n, W] block, NaN outside `keep` -> 0.

    `reconstruction/sae_smoke64.peaks_of` verbatim. A rollout with nothing kept (an empty
    generation) peaks at 0.0 and counts as not firing, which is what it is -- not a missing
    measurement.
    """
    finite = np.where(np.isfinite(act_row), act_row, -np.inf)
    pk = finite.max(axis=1)
    return np.where(np.isfinite(pk), pk, 0.0)


def best_per_rollout(cos_row: np.ndarray, empty: float = -1.0) -> np.ndarray:
    """[n] per-rollout max cosine from one row's [n, T] block, NaN outside the kept tokens.

    `precompute/common.agg` scores a rollout with no kept token at -1.0 rather than raising, and
    `empty=-1.0` reproduces that, so a recomputation here and the stored `per_target.jsonl`
    aggregate are the same number on the same data. `score` treats the CENTRED cosine differently
    -- a rollout with no kept centred token is dropped from the centred aggregates entirely, never
    scored -1 -- so the centred path passes `empty=nan` and the caller drops the NaNs.
    """
    finite = np.where(np.isfinite(cos_row), cos_row, -np.inf)
    best = finite.max(axis=1)
    return np.where(np.isfinite(best), best, empty)


def load_config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def load_sae_self(vol: Vol, rel_dir: str):
    """(meta, act [N, n, W]) of a `sae_self/` product, or (None, why) when it is not there.

    `sae_self.f16` is the PRE-GATE per-token activation of each row's own target feature on its
    own rollouts, NaN outside the kept tokens; `width` is the run's scoring window + 1, absent on a
    product written before that was per-run. Both come from `autointerp/sae_self.py`'s own writer,
    and this is `reconstruction/sae_smoke64.load_sae_self` reading through `Vol` instead of a
    bare path.
    """
    meta = vol.json(f"{rel_dir}/sae_self.json")
    if meta is None:
        return None, f"{rel_dir}/sae_self.json is not there"
    width = int(meta.get("width", 96))
    shape = (len(meta["rows"]), int(meta["n"]), width)
    act = vol.array(f"{rel_dir}/sae_self.f16", "float16", shape)
    if act is None:
        return meta, f"{rel_dir}/sae_self.f16 is not there (its sae_self.json is)"
    return meta, act.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# the set's rows, and the families they cut into
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """One table's worth of rows: a family label, plus the SAE axes where the family is a
    dictionary. `sae_key` and `sae_side` come from the ROW (`features/draw_sae2m.py` puts the
    dictionary in `sae_key`, not in the family label, so `family: sae` rows of two dictionaries
    are two families here and one label on disk)."""

    family: str
    sae_key: str = ""
    sae_side: str = ""

    @property
    def label(self) -> str:
        bits = [self.family]
        if self.sae_key:
            bits.append(self.sae_key.split("/")[-1])
        if self.sae_side:
            bits.append(self.sae_side)
        return "/".join(bits)

    @property
    def slug(self) -> str:
        return self.label.replace("/", "_").replace("-", "_")


def family_of(row: dict, declared_sae_key: str = "") -> Family:
    """The Family a set row belongs to. `declared_sae_key` is the set's own `sae_key:` from
    `config.yaml` or `storage.json`, used for rows drawn before the per-row field existed --
    the same resolution order `precompute/common.sae_rows_of` uses, and for the same reason:
    every 131k feature id is also a valid 2M index, so guessing is silently wrong."""
    fam = str(row["family"])
    kind = row.get("sae_key") or (declared_sae_key if _is_dictionary_label(fam) else "")
    return Family(fam, str(kind or ""), str(row.get("sae_side") or ""))


def _is_dictionary_label(family: str) -> bool:
    """Families whose rows are SAE features. Read from `config.yaml`'s `family_kinds:` by
    `families_of`; this fallback exists only for a label the config does not declare."""
    return family.startswith("sae")


def load_ids(vol: Vol, base: str, set_name: str, cfg: dict) -> tuple[dict[int, dict], str]:
    """({row: its ids.jsonl record}, the set's declared sae_key). Refuses a set with no ids."""
    rel = f"base/{base}/heldout/{set_name}/ids.jsonl"
    rows = vol.jsonl(rel)
    assert rows is not None, (
        f"{rel} is not on the volume (root {vol.prefix or '/'}): a set is its ids.jsonl, and "
        f"without it nothing downstream can be cut on family, stratum or document"
    )
    storage = vol.json(f"base/{base}/heldout/{set_name}/storage.json") or {}
    declared = str(storage.get("sae_key") or cfg["heldout"].get(set_name, {}).get("sae_key") or "")
    return {int(r["row"]): r for r in rows}, declared


def families_of(ids: dict[int, dict], cfg: dict, declared_sae_key: str = "") -> dict[Family, list[int]]:
    """{Family: its row indices}, in the set's own row order. Iterates the ROWS, so a family the
    config has never heard of still gets a table (labelled) rather than disappearing."""
    out: dict[Family, list[int]] = {}
    for row in sorted(ids):
        out.setdefault(family_of(ids[row], declared_sae_key), []).append(row)
    return out


def is_dictionary(fam: Family, cfg: dict) -> bool:
    """Is this family's row an SAE feature (activation metrics) or a direction (cosines)?
    `family_kinds.<f>.kind == dictionary` decides; an undeclared label falls back to its spelling
    and is reported by the caller, because the two get different tables."""
    kinds = cfg.get("family_kinds") or {}
    entry = kinds.get(fam.family)
    if entry is None:
        return _is_dictionary_label(fam.family)
    return str(entry.get("kind")) == "dictionary"


# ---------------------------------------------------------------------------------------------
# source discovery: (checkpoint x engine x run-tag) present on the volume for this set
# ---------------------------------------------------------------------------------------------


@dataclass
class Source:
    """One scored arm: a checkpoint, an engine and a run tag, with its products' volume paths."""

    maemm: str          # the config key, e.g. qwen36-27b/2026-09-10_rl-8x2048-full
    base: str
    engine: str         # "hf" | "vllm"
    run_tag: str        # "" where the run needed none
    scores_rel: str     # maemms/<maemm>/scores/<dir>
    rollouts_rel: str   # maemms/<maemm>/rollouts/<stem>.summary.json
    role: str = "secondary"
    per_target: dict[int, dict] = field(default_factory=dict)
    rows_meta: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    n: int = 0
    mu: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        name = self.maemm.split("/")[-1]
        if self.engine != "hf":
            name += f"@{self.engine}"
        return f"{name}:{self.run_tag}" if self.run_tag else name

    @property
    def centred(self) -> bool:
        """Did this run centre? `score` writes the centred cosine only when it did, and a run at
        `--mu none` carries no `cos_centred` at all -- absent, never a one-sided number."""
        return self.mu not in (None, "", "none")


def parse_scores_dir(name: str, set_name: str) -> tuple[str, str] | None:
    """(engine, run_tag) for a scores directory of this set, or None when it is another set's.

    The inverse of `precompute/common.rollout_stem`: `<set>[__<engine>][__<tag>]`. A suffix part
    that is an engine names the engine; everything else is the tag. A directory that merely STARTS
    with the set name but continues without `__` (e.g. `2026-09-16_v1x`) is another set.

    THE ENGINE PART IS FOUND WHEREVER IT IS, not only first. `rollout_stem` puts it before the tag,
    but a `score --score-name <set>__<tag> --engine vllm` run -- which is how eval 1's two
    old-primary arms were written, because `scores_dir` took no tag of its own until 2026-09-21 --
    lands on `<set>__<tag>__<engine>` instead. The first version of this function required the engine part
    to come first, so it read `2026-09-21_v3_sae2m__mu-none__vllm` as engine `hf` with the tag
    `mu-none__vllm`: a vLLM product labelled HF in the paper's own CSV, on six of eval 1's arms.
    MEASURED on the real directory names 2026-09-21. Only the FIRST engine-valued part is taken,
    so a run tag that is itself spelled `hf` or `vllm` still ends up in the tag -- that collision
    is inherent to the `__`-joined naming and is not made worse here.
    """
    if name == set_name:
        return "hf", ""
    if not name.startswith(set_name + "__"):
        return None
    parts = [p for p in name[len(set_name) + 2 :].split("__") if p]
    engine = "hf"
    tag_parts = []
    for p in parts:
        if p in ENGINES and engine == "hf":
            engine = p
        else:
            tag_parts.append(p)
    return engine, "__".join(tag_parts)


def discover_sources(vol: Vol, cfg: dict, base: str, set_name: str) -> tuple[list[Source], list[str]]:
    """(the arms present on the volume for this set, the config'd checkpoints that have none).

    Iterates `config.yaml`'s `maemms:` -- so a new checkpoint is a config entry and nothing here
    changes -- and asks the volume which of its scores directories belong to this set. A
    checkpoint declared `compute: false` is DECLARED but nothing is generated for it in this
    pipeline (`common.maemms_for`'s rule), so it is not reported as missing.
    """
    found: list[Source] = []
    absent: list[str] = []
    for key, entry in sorted((cfg.get("maemms") or {}).items()):
        if not str(key).startswith(base + "/"):
            continue
        if entry.get("compute") is False:
            continue
        role = "primary" if entry.get("primary") else str(entry.get("role") or "secondary")
        hits: list[tuple[str, str, str, str]] = []  # (scores_rel, rollouts_rel, engine, tag)
        for d in vol.ls(f"maemms/{key}/scores"):
            p = parse_scores_dir(d, set_name)
            if p is not None:
                hits.append((f"maemms/{key}/scores/{d}",
                             f"maemms/{key}/rollouts/{d}.summary.json", *p))
        # A VARIANT directory is the same arm under another generation setting -- an `--amp` of
        # `rollouts_nla`, a patchscopes cell -- and `score --rollouts-dir` writes its products to
        # `variants/<set>__<variant>/scores/`, beside the run's own rollouts rather than in the
        # accumulating `rollouts/`. It is the same (set, tag) axis under another parent, so it
        # becomes a source with the variant as its run tag and is not a second code path.
        for d in vol.ls(f"maemms/{key}/variants"):
            p = parse_scores_dir(d, set_name)
            if p is not None:
                hits.append((f"maemms/{key}/variants/{d}/scores",
                             f"maemms/{key}/variants/{d}/rollouts.summary.json", *p))
        if not hits:
            absent.append(key)
            continue
        for scores_rel, rollouts_rel, engine, tag in hits:
            found.append(Source(maemm=key, base=base, engine=engine, run_tag=tag,
                                scores_rel=scores_rel, rollouts_rel=rollouts_rel, role=role))
    found.sort(key=lambda s: (s.role != "primary", s.maemm, s.engine, s.run_tag))
    return found, absent


def load_source(vol: Vol, src: Source) -> str:
    """Read a source's small files in place. Returns "" on success, or why it is unusable."""
    rows_meta = vol.json(f"{src.scores_rel}/rows.json")
    if rows_meta is None:
        return f"{src.scores_rel}/rows.json is not there"
    per_target = vol.jsonl(f"{src.scores_rel}/per_target.jsonl")
    if per_target is None:
        return f"{src.scores_rel}/per_target.jsonl is not there"
    src.rows_meta = rows_meta
    src.per_target = {int(r["row"]): r for r in per_target}
    src.index = vol.json(f"{src.scores_rel}/index.json") or {}
    src.n = int(rows_meta.get("n", 0))
    # `mu` is the path the run centred on, recorded by `score` itself. `null` means it centred on
    # nothing, and then there is no centred cosine to read -- which is the intended behaviour, not
    # a missing file.
    src.mu = rows_meta.get("mu")
    return ""


def colour_map(sources: list[Source]) -> dict[str, str]:
    """{source label: hue}, assigned in fixed palette order over the sources' stable sort.

    Computed over ALL discovered sources before any `--sources` filter, so dropping one never
    repaints the survivors: colour follows the entity, not its rank in the filtered list.
    """
    return {s.label: PALETTE[i % len(PALETTE)] for i, s in enumerate(sources)}


# ---------------------------------------------------------------------------------------------
# estimation: the document-clustered bootstrap
# ---------------------------------------------------------------------------------------------


def cluster_bootstrap(values, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """(mean, clustered SE, n_items, n_clusters) -- resampling CLUSTERS with replacement.

    122 of the 512 realact targets of `2026-09-16_v1` share a document
    (`features/activations.py:22-24`), and two targets from one document are not two independent
    draws: a plain std/sqrt(n) understates the SE there. This resamples whole documents, which is
    the estimator plan §2.3 asks for. A row with no document (a `random` or `sae` draw) is its own
    cluster, so for those families this reduces EXACTLY to the ordinary nonparametric bootstrap --
    not to a silently different estimator.

    With one cluster the SE is not estimable and comes back NaN rather than 0.0: a single document
    gives a mean and no spread, and 0.0 would read as a certainty.
    """
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), 0, 0
    keys = list(clusters)
    assert len(keys) == vals.size, f"{vals.size} values against {len(keys)} cluster labels"
    order: dict = {}
    for i, k in enumerate(keys):
        order.setdefault(k, []).append(i)
    groups = [np.asarray(v, dtype=np.int64) for v in order.values()]
    mean = float(vals.mean())
    if len(groups) < 2:
        return mean, float("nan"), int(vals.size), len(groups)
    rng = np.random.default_rng(seed)
    sums = np.array([vals[g].sum() for g in groups])
    counts = np.array([g.size for g in groups], dtype=np.float64)
    pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return mean, float(boot.std(ddof=1)), int(vals.size), len(groups)


def se_iid(values) -> float:
    """std/sqrt(n) over items, ignoring clustering. Carried in the CSVs beside the clustered SE so
    the size of the clustering correction is visible rather than asserted."""
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size < 2:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(vals.size))


# ---------------------------------------------------------------------------------------------
# output: one tables.md, one CSV per table, and a figures/ directory
# ---------------------------------------------------------------------------------------------


def num(v, places: int = 4) -> str:
    """A number for a markdown cell. None and NaN print as an em dash -- never as 0."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(f) else f"{f:.{places}f}"


def pm(value, err, places: int = 4) -> str:
    """`value ± se`, with the error dropped where it is not estimable."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    if err is None or not math.isfinite(float(err)):
        return num(value, places)
    return f"{num(value, places)} ± {num(err, places)}"


class Out:
    """Collects tables, writes `<out>/tables.md` + one `<out>/<name>.csv` per table.

    One markdown file rather than one per table: the tables are read together (an arm's cosine
    row only means something beside the arm it is compared with), and the sanity block at the end
    has to sit in the same file as the numbers it gates.
    """

    def __init__(self, out_dir: Path, title: str, preamble: list[str]):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "figures").mkdir(exist_ok=True)
        self.title = title
        self.preamble = preamble
        self.blocks: list[str] = []
        self.notes: list[str] = []
        self.csvs: list[str] = []

    def table(self, name: str, title: str, caption: str, header: list[str],
              rows: list[list], *, csv_header: list[str] | None = None,
              csv_rows: list[list] | None = None) -> None:
        """One table: markdown (rounded, em-dashed) into tables.md, full precision into its CSV.

        `csv_rows` defaults to `rows`; it is passed separately where the CSV carries columns the
        markdown deliberately leaves out -- the SAE cosines, which go to CSV and never to a table
        beside an activation ratio (plan §2.3: no SAE-target cosines reported).
        """
        md = [f"### {title}", "", f"*{caption}*", ""]
        if rows:
            md.append("| " + " | ".join(_cell(h) for h in header) + " |")
            md.append("|" + "|".join("---" for _ in header) + "|")
            for r in rows:
                md.append("| " + " | ".join(_cell(v) for v in r) + " |")
        else:
            md.append("*(no rows: every source for this family is absent)*")
        md += ["", f"CSV: `{name}.csv`", ""]
        self.blocks.append("\n".join(md))
        write_csv(self.dir / f"{name}.csv", csv_header or header, csv_rows if csv_rows is not None else rows)
        self.csvs.append(f"{name}.csv")

    def section(self, text: str) -> None:
        self.blocks.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def finish(self, figures: list[str]) -> Path:
        lines = [f"# {self.title}", "", *self.preamble, ""]
        lines += self.blocks
        if figures:
            lines += ["## Figures", ""]
            lines += [f"- `figures/{f}.pdf` / `.png`" for f in sorted(figures)]
            lines += [""]
        if self.notes:
            lines += ["## Skipped and absent", ""] + [f"- {n}" for n in self.notes] + [""]
        path = self.dir / "tables.md"
        path.write_text("\n".join(lines))
        return path


def _cell(v) -> str:
    """One markdown cell. A literal `|` -- which every `|a − b| = c` verdict carries -- ends the
    cell unless it is escaped, and an unescaped one silently shifts every column to its right."""
    return "—" if v is None else str(v).replace("|", "\\|")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """A CSV with no dependency on a dataframe stack: the values are already strings or floats and
    the tables are small. `None` writes as an empty field, which is what an absent number is."""
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


# PNG raster density. The PDF is vector and ignores it. 140 rather than the 200 this started at:
# `results/faithfulness/` is COMMITTED (it is the paper's numbers), and at 200 a nine-inch panel
# is ~150 KB against ~70 KB here, which over a run's ~20 figures is 3 MB of binary in the history
# for a preview of a vector file that sits beside it. The PDF is the one to cite.
FIG_DPI = 140


def savefig(fig, out_dir: Path, name: str) -> str:
    """Write one figure as PDF (vector, for the paper) and PNG (FIG_DPI, for a quick look)."""
    d = Path(out_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": FIG_DPI})):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE, **kw)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return name


def style_axes(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes, ink-coloured text. Applied to every panel so the figures read as
    one system: hairline gridlines behind the marks, no top/right spine, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def env_hint() -> str:
    """The command shape this module is run under, for the provenance line of every output."""
    return os.environ.get("MODAL_PROFILE", "maemms")
