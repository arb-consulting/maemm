"""Run directories, resume keys (stage records, chained hashes), config.json and provenance, plus the shared
cost keys of coverage_and_costs.json. Imports no package config: a package passes its pins in."""

import collections.abc, dataclasses, hashlib, json, math, os, platform, subprocess, sys, time


def json_safe(obj):
    """`obj` with NaN/Inf floats replaced by None, recursively, so every write is strict JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class RunDir:
    def __init__(self, output_dir, run_id=None):
        # without run_id, output_dir is the run directory itself
        self.path = output_dir if run_id is None else os.path.join(output_dir, run_id)
        os.makedirs(self.path, exist_ok=True)

    def sub(self, name):
        p = os.path.join(self.path, name)
        os.makedirs(p, exist_ok=True)
        return p

    def file(self, rel):
        p = os.path.join(self.path, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def exists(self, rel):
        return os.path.exists(os.path.join(self.path, rel))

    def read_json(self, rel):
        with open(self.file(rel), encoding="utf-8") as h:
            return json.load(h)

    def write_json(self, rel, obj):
        tmp = self.file(rel) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as h:
            json.dump(json_safe(obj), h, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, self.file(rel))

    def append_jsonl(self, rel, obj):
        with open(self.file(rel), "a", encoding="utf-8") as h:
            h.write(json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False) + "\n")

    def read_jsonl(self, rel):
        """Every record of an append-only log; a bad line raises, except a last line of NUL bytes and
        whitespace (a killed append), which is skipped with a warning."""
        if not self.exists(rel):
            return []
        path = self.file(rel)
        with open(path, encoding="utf-8") as h:
            lines = h.readlines()
        out = []
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                if lineno == len(lines) and not line.strip("\x00 \t\r\n"):
                    print(f"[read_jsonl] {path}:{lineno}: torn trailing line ({len(line)} bytes of NUL/whitespace) "
                          f"skipped; the record was never written", file=sys.stderr, flush=True)
                    continue
                raise ValueError(f"{path}:{lineno}: not a JSON record ({e.msg} at column {e.colno})") from e
        return out


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def plain(v):
    """`v` made JSON-plain, recursively: a dataclass as its fields, any mapping as a dict, a tuple as a list."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, collections.abc.Mapping):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v


_CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _checkout_path(value):
    """True for a path-like or an absolute path inside this checkout, which describes the machine, not the
    run."""
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str) or not os.path.isabs(value):
        return False
    norm = os.path.normpath(value)
    return norm == _CHECKOUT or norm.startswith(_CHECKOUT + os.sep)


def config_constants(module):
    """Every public upper-case constant of a config module, JSON-plain, minus checkout paths."""
    return {k: plain(v) for k, v in vars(module).items()
            if k.isupper() and not k.startswith("_") and not _checkout_path(v)}


# args describing one invocation rather than the run
INVOCATION_ONLY = ("stage", "force", "shard", "device", "output_dir", "run_id", "arms", "n_shards")


def run_args(args, invocation_only=INVOCATION_ONLY):
    return {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in invocation_only}


EXPLICIT_OUTPUT_DIR = "<explicit>"


def recorded_output_dir(value):
    """`--output-dir` relative to the checkout root when inside it, else `<explicit>`."""
    if value is None:
        return None
    path = os.path.normpath(os.path.abspath(value))
    if path == _CHECKOUT or path.startswith(_CHECKOUT + os.sep):
        return os.path.relpath(path, _CHECKOUT)
    return EXPLICIT_OUTPUT_DIR


def invocation_args(args):
    """This invocation's resolved args for provenance/<stage>.json, with `output_dir` recorded safely."""
    out = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    if "output_dir" in out:
        out["output_dir"] = recorded_output_dir(out["output_dir"])
    return out


def saved_judge_profile(run):
    """The judge profile a run directory's config.json names, or None (`run` is a RunDir or a path)."""
    path = os.path.join(str(getattr(run, "path", run)), "config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as h:
        doc = json.load(h)
    block = doc["config"] if isinstance(doc.get("config"), dict) else doc
    return block.get("JUDGE_PROFILE", block.get("judge_profile"))


def refuse_profile_switch(run, profile):
    """Raise when a run directory started under one judge profile is invoked under another."""
    saved = saved_judge_profile(run)
    if saved is None or saved == profile:
        return
    raise RuntimeError(f"{getattr(run, 'path', run)} was started under the {saved!r} judge profile and "
                       f"this invocation runs {profile!r}; name another directory (or pass "
                       f"--judge-profile {saved})")


def write_config_once(run, config, args):
    """Write config.json = {"config", "args", "created_at", "updated_at"} unless both blocks match, printing
    the changed keys; returns whether it wrote. Both blocks are compared as JSON reads them back (string
    keys, lists for tuples). Resume is decided by stage hashes, not by this file."""
    doc = json.loads(json.dumps(json_safe({"config": plain(config), "args": run_args(args)}), ensure_ascii=False))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    prev = run.read_json("config.json") if run.exists("config.json") else None
    if prev is not None and prev.get("config") == doc["config"] and prev.get("args") == doc["args"]:
        return False
    if prev is None:
        created = now
    else:
        created = prev["created_at"]
        print(f"[config] config.json refreshed: {', '.join(_changed_keys(prev, doc))} differ from this "
              f"invocation; created_at {created} kept", flush=True)
    run.write_json("config.json", {**doc, "created_at": created, "updated_at": now})
    return True


_MISSING = object()


def _changed_keys(prev, doc):
    """The `config`/`args` keys whose values differ between `prev` and `doc`."""
    keys = set()
    for block in ("config", "args"):
        old = prev.get(block) or {}
        keys |= {k for k in set(old) | set(doc[block]) if old.get(k, _MISSING) != doc[block].get(k, _MISSING)}
    return sorted(keys)


# ---------------------------------------------------------------- coverage_and_costs.json's shared keys
ELAPSED_SECONDS_KEY = "elapsed_seconds"               # float: measured stage seconds, the render excluded
ELAPSED_UNMEASURED_KEY = "elapsed_unmeasured_stages"  # list[str]: stages that measured nothing anywhere
GPU_COST_KEYS = ("gpu_type", "gpu_seconds_measured", "gpu_stages_unmeasured",
                 "gpu_rate_per_hour", "gpu_rate_source", "gpu_cost_usd_estimated")


def elapsed_block(total_seconds, unmeasured):
    """The elapsed keys; `unmeasured` stages are listed, making the total a lower bound."""
    return {ELAPSED_SECONDS_KEY: total_seconds, ELAPSED_UNMEASURED_KEY: sorted(unmeasured)}


def launch_records(run):
    """Each launcher's provenance/modal*.json as `(name, record)`, sorted; a record another file lists under
    `sources` is dropped, so no seconds are counted twice."""
    raw = []
    directory = os.path.join(run.path, "provenance")
    if not os.path.isdir(directory):
        return raw
    for name in sorted(os.listdir(directory)):
        if name.startswith("modal") and name.endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as h:
                raw.append((name[:-5], json.load(h)))
    folded = set()
    for name, record in raw:
        for source in record.get("sources") or []:
            filename = source.get("file") if isinstance(source, dict) else None
            if isinstance(filename, str) and filename != f"{name}.json":
                folded.add(filename)
    return [(name, record) for name, record in raw if f"{name}.json" not in folded]


def _numbers(records, key):
    return [float(d[key]) for _n, d in records
            if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool)]


def launch_gpu_seconds(records):
    """The launches' measured container seconds, summed; None when no record has any. A lower bound."""
    vals = _numbers(records, "gpu_seconds_measured")
    return sum(vals) if vals else None


def launch_gpu_rate(records, provenance=None):
    """`(US$ per GPU-hour, source)`: the launcher's list rate, else its estimate over its seconds."""
    prov = provenance or {}
    rates = _numbers(records, "gpu_cost_per_hour") or _numbers([("p", prov)], "gpu_cost_per_hour")
    if rates:
        return rates[0], "launcher list rate"
    est, secs = sum(_numbers(records, "gpu_cost_usd_estimated")), sum(_numbers(records, "gpu_seconds_measured"))
    if est and secs:
        return est / secs * 3600.0, "launcher estimate / launcher seconds"
    if prov.get("gpu_cost_usd_estimated") and prov.get("gpu_seconds_measured"):
        return float(prov["gpu_cost_usd_estimated"]) / float(prov["gpu_seconds_measured"]) * 3600.0, \
            "launcher estimate / launcher seconds"
    return None, "unavailable"


def gpu_cost_block(gpu_type, gpu_seconds, unmeasured, rate_per_hour, rate_source):
    """The GPU_COST_KEYS block: seconds x rate, "unavailable" (never 0) when a term is missing."""
    return {
        "gpu_type": gpu_type if gpu_type is not None else "unavailable",
        "gpu_seconds_measured": gpu_seconds if gpu_seconds is not None else "unavailable",
        "gpu_stages_unmeasured": sorted(unmeasured),
        "gpu_rate_per_hour": rate_per_hour,
        "gpu_rate_source": rate_source,
        "gpu_cost_usd_estimated": (gpu_seconds / 3600.0 * rate_per_hour
                                   if (rate_per_hour is not None and gpu_seconds) else "unavailable"),
    }


def stage_done(run, stage, chash):
    rel = f"stages/{stage}.json"
    if not run.exists(rel):
        return False
    rec = run.read_json(rel)
    return rec.get("completed") is True and rec.get("config_hash") == chash


def mark_stage(run, stage, chash, extra=None, started=None):
    """Write `stages/<stage>.json` as completed under `chash`, with its seconds (from `started`, else
    `extra["seconds"]`) and a `ChainedHash`'s `upstream` digests."""
    extra = dict(extra or {})
    if started is not None:
        extra["seconds"] = time.time() - started
    elif not isinstance(extra.get("seconds"), (int, float)) or isinstance(extra.get("seconds"), bool):
        raise ValueError(f"stage {stage!r}: mark_stage needs `started` or a numeric `seconds` in the record; "
                         f"a record without one leaves the run's summed stage time a silent lower bound")
    if isinstance(chash, ChainedHash):
        extra["upstream"] = dict(chash.upstream)
    run.write_json(
        f"stages/{stage}.json",
        {
            "stage": stage,
            "config_hash": str(chash),
            "completed": True,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        },
    )


# ---------------------------------------------------------------- chaining a stage to the records it reads
# A stage's key includes the digest of each upstream stage record, so an upstream re-run invalidates it.


class StaleVerdict(RuntimeError):
    """A verdict about to be joined to a readout it was not given for; re-run the judge stage."""


class ChainedHash(str):
    """A resume key (a plain str) that also carries the upstream record digests, for `mark_stage`."""

    def __new__(cls, value, upstream):
        self = super().__new__(cls, value)
        self.upstream = dict(upstream)
        return self


def record_digest(run, name):
    """The digest of the parsed `stages/<name>.json`; None when absent or not completed."""
    rel = f"stages/{name}.json"
    if not run.exists(rel):
        return None
    rec = run.read_json(rel)
    return config_hash(rec) if rec.get("completed") is True else None


def stage_record_names(run, belongs):
    """Sorted names of completed stage records for which `belongs(name)` holds (e.g. a stage's shards)."""
    d = os.path.join(run.path, "stages")
    if not os.path.isdir(d):
        return []
    names = sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json") and belongs(n[:-5]))
    return [n for n in names if record_digest(run, n) is not None]


def chained_hash(settings, run, upstream):
    """A `ChainedHash` of `settings` plus the digest of each `upstream` record (None when absent)."""
    digests = {name: record_digest(run, name) for name in sorted(upstream)}
    return ChainedHash(config_hash({**settings, "upstream": digests}), digests)


def broken_links(run, name):
    """Sentences naming each link of `name`'s upstream chain whose recorded digest no longer matches."""
    out, seen = [], set()

    def walk(n):
        if n in seen:
            return
        seen.add(n)
        rel = f"stages/{n}.json"
        if not run.exists(rel):
            out.append(f"`{n}` has no record")
            return
        ups = run.read_json(rel).get("upstream")
        if ups is None:
            out.append(f"`{n}`'s record names no upstream digests (it was written under an unchained key)")
            return
        for up, was in sorted(ups.items()):
            now = record_digest(run, up)
            if now == was:
                walk(up)
            elif now is None:
                out.append(f"`{n}` was built over `{up}` as recorded at {was}, and `{up}` has no completed record now")
            else:
                out.append(f"`{n}` was built over `{up}` as recorded at {was}, and that record now reads {now}")

    walk(name)
    return out


# git fields and their environment overrides (set by remote launchers, whose images have no .git)
GIT_ENV = {"git_commit": "GIT_COMMIT", "git_branch": "GIT_BRANCH", "git_dirty": "GIT_DIRTY"}


def _git_local(cwd=None):
    def one(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = one(["status", "--porcelain"])
    return {
        "git_commit": one(["rev-parse", "HEAD"]),
        "git_branch": one(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def git_state():
    """`{"git_commit", "git_branch", "git_dirty"}`, from `GIT_ENV` variables first, else from git."""
    local = None
    out = {}
    for key, env_var in GIT_ENV.items():
        value = (os.environ.get(env_var) or "").strip()
        if not value:
            local = local or _git_local()
            out[key] = local[key]
        else:
            out[key] = (value.lower() in ("1", "true")) if key == "git_dirty" else value
    return out


RUNTIME_PACKAGES = ("torch", "transformers", "accelerate", "flash-linear-attention", "numpy")


def runtime_versions(packages=RUNTIME_PACKAGES):
    """`{"python": ..., package: version or None when not installed}`."""
    import importlib.metadata

    out = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_provenance(run, extra=None, stage=None, common_fields=None):
    """Record where a run or one stage came from: with `stage`, merge `extra`, the git state and runtime into
    provenance/<stage>.json (one writer per file); without, update provenance.json (git state: first write
    wins) with the interpreter, platform, `common_fields` and `extra`."""
    if stage is not None:
        rel = f"provenance/{stage}.json"
        prev = run.read_json(rel) if run.exists(rel) else {}
        prev.update(extra or {})
        prev["invocation_git"] = git_state()
        prev["runtime"] = runtime_versions()
        run.write_json(rel, prev)
        return
    prev = run.read_json("provenance.json") if run.exists("provenance.json") else {}
    for key, value in git_state().items():
        if prev.get(key) is None:
            prev[key] = value
    prev.update(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    prev.update(common_fields or {})
    prev.update(extra or {})
    run.write_json("provenance.json", prev)


def read_provenance(run):
    """provenance.json updated with every provenance/<stage>.json in name order (later wins)."""
    merged = dict(run.read_json("provenance.json")) if run.exists("provenance.json") else {}
    stage_dir = os.path.join(run.path, "provenance")
    if os.path.isdir(stage_dir):
        for name in sorted(os.listdir(stage_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(stage_dir, name), encoding="utf-8") as h:
                merged.update(json.load(h))
    return merged
