"""Atomic run artifacts and content-addressed stage caches."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    if isinstance(value, np.ndarray):
        h = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
        h.update(np.ascontiguousarray(value).tobytes())
        return h.hexdigest()
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(serializable(value), out, ensure_ascii=False, allow_nan=False, indent=2)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_arrays(path, value):
    """`value`, the data of a document `Run.save_arrays` wrote at `path`, with each array reference
    replaced by its array from the `.npz` beside it."""
    with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as arrays:
        def decode(x):
            if isinstance(x, dict) and set(x) == {"__array__"}:
                return arrays[x["__array__"]]
            if isinstance(x, dict):
                return {k: decode(v) for k, v in x.items()}
            if isinstance(x, list):
                return [decode(v) for v in x]
            return x
        return decode(value)


def git_state():
    """(commit, dirty) of the code a stage ran, through the suite's one resolver (`evals.downstream.common.runs`)."""
    from evals.downstream.common.runs import git_state as resolve

    state = resolve()
    return state["git_commit"], state["git_dirty"]


#: The `evals/downstream/common` modules each module of this package executes through (with what those import). A
#: cache key names this package's modules and these follow, so a key moves with the shared code it runs.
COMMON_DEPENDENCIES = {
    "config": ("pins",),
    "data": ("model_io", "scorer"),
    "model": ("model_io", "scorer", "plain_steer", "degeneracy"),
    "generate": ("plain_steer", "degeneracy", "model_io", "scorer"),
    "corpus": ("retrieval", "scorer"),
    "nla": ("nla/nla_reader", "model_io", "pins", "scorer"),
    "lens": ("lens_io",),
    "judge": ("judge_client", "judges"),
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(modules=(), common=()):
    """{source: sha256} of everything a keyed operation executes through: this package's `config`,
    `artifacts` and `modules`, `maemm/inject` and `maemm/prompts` where `model` is named, and the `evals/downstream/common`
    modules those imply plus `common`. The persona package keys its GPU payloads through here too."""
    here = Path(__file__).parent
    shared = here.parent / "common"
    names = ("config", "artifacts", *modules)
    sources = {name: file_sha256(here / f"{name}.py") for name in names}
    if "model" in modules:
        for name in ("inject", "prompts"):
            sources[f"maemm/{name}"] = file_sha256(here.parents[1] / "maemm" / f"{name}.py")
    needed = {dep for name in names for dep in COMMON_DEPENDENCIES.get(name, ())} | set(common)
    for dep in sorted(needed):
        sources[f"common/{dep}"] = file_sha256(shared / f"{dep}.py")
    return sources


class Run:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config.validate()
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        config_path = self.root / "config.json"
        if config_path.exists() and read_json(config_path) != config.to_dict():
            raise ValueError("Run configuration differs; use a new run directory")
        write_json(config_path, config.to_dict())
        path = self.root / "provenance.json"
        if not path.exists():
            versions = {}
            for name in ("numpy", "torch", "transformers", "pyarrow", "huggingface_hub"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            commit, dirty = git_state()
            write_json(path, {"created_at": time.time(), "git_commit": commit, "git_dirty": dirty,
                              "judge_profile": config.judge_profile, "versions": versions, "stages": {}})

    # --- the append-only judge logs of evals.downstream.common.judge_client.run_requests, and plain JSON -----------
    def file(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def exists(self, relative):
        return (self.root / relative).exists()

    def read_json(self, relative):
        return read_json(self.root / relative)

    def write_json(self, relative, value):
        write_json(self.root / relative, value)

    def append_jsonl(self, relative, record):
        with open(self.file(relative), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False) + "\n")

    def read_jsonl(self, relative):
        if not self.exists(relative):
            return []
        with (self.root / relative).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def key(self, stage, inputs=None, modules=(), common=()):
        """The cache key of one operation: its name, the run's settings, the bytes of every source it
        executes through (`source_hashes`) and its inputs."""
        return digest([stage, self.config.to_dict(), source_hashes(modules, common), inputs])

    def cached(self, relative, key):
        path = self.root / relative
        if path.exists():
            value = read_json(path)
            if value.get("cache_key") == key:
                return value["data"]
        return None

    def save(self, relative, key, data):
        write_json(self.root / relative, {"cache_key": key, "data": data})
        return data

    def stage_done(self, stage, started, **details):
        with self._lock:
            path = self.root / "provenance.json"
            record = read_json(path)
            record["stages"][stage] = {"completed_at": time.time(), "seconds": time.time() - started, **details}
            record['package_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(__file__).parent.glob('*.py')}
            write_json(path, record)

    def save_arrays(self, relative, key, value):
        arrays = {}
        def encode(x):
            if isinstance(x, np.ndarray):
                name = f"a{len(arrays)}"
                arrays[name] = x
                return {"__array__": name}
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [encode(v) for v in x]
            return x
        encoded = encode(value)
        write_npz((self.root / relative).with_suffix(".npz"), **arrays)
        self.save(relative, key, encoded)
        return value

    def cached_arrays(self, relative, key):
        value = self.cached(relative, key)
        return None if value is None else read_arrays(self.root / relative, value)
