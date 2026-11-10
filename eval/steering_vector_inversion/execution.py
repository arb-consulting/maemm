"""A bounded pool shared by GPU stages; each worker owns the models of one role."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import threading

from .artifacts import digest

_worker = None
_worker_config = None


def _initialize(config, cache_dir, devices):
    global _worker_config
    _worker_config = (config, cache_dir, devices.get())


def _execute(payload):
    """One payload on this process's worker, built for the role the payload needs: a model worker (the
    clean base, and the inverter once asked) or the NLA verbalizer's worker, never both at once."""
    global _worker
    from . import nla
    role = 'nla' if payload.get('operation') == nla.OPERATION else 'model'
    if _worker is not None and _worker.role != role:
        _worker.close()
        _worker = None
    if _worker is None:
        from .config import Config
        config, cache_dir, device = _worker_config
        if role == 'nla':
            _worker = nla.NlaWorker(Config(**config), cache_dir, device)
        else:
            from .model import ModelWorker
            _worker = ModelWorker(Config(**config), cache_dir, device)
    return _worker.execute(payload)


class GPUExecutor:
    def __init__(self, config, cache_dir, max_workers=8, remote=None):
        self.stopping = threading.Event()
        # A local host runs one worker per card, at most eight; a remote executor's ceiling is its launcher's.
        if max_workers < 1 or (remote is None and max_workers > 8):
            raise ValueError("Use between one and eight GPU workers locally, and at least one remotely")
        if remote is not None:
            self.workers, self.execute = max_workers, remote
            self.pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            import torch
            available = torch.cuda.device_count()
            if not available:
                raise RuntimeError("No local CUDA GPU. Use the Modal wrapper or run this stage on a GPU host.")
            self.workers = min(max_workers, available)
            context = multiprocessing.get_context("spawn")
            devices = context.Queue()
            for index in range(self.workers):
                devices.put(f"cuda:{index}")
            self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                initializer=_initialize, initargs=(config.to_dict(), str(cache_dir), devices))
            self.execute = _execute

    def map(self, tasks):
        """Yield (task ID, result) on completion with bounded outstanding work."""
        tasks = iter(tasks)
        pending = {}
        def fill():
            if self.stopping.is_set():
                raise RuntimeError('GPU execution cancelled after a pipeline failure')
            while len(pending) < self.workers:
                try:
                    task_id, payload = next(tasks)
                except StopIteration:
                    break
                pending[self.pool.submit(self.execute, payload)] = task_id
        fill()
        while pending:
            if self.stopping.is_set():
                raise RuntimeError('GPU execution cancelled after a pipeline failure')
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = pending.pop(future)
                yield task_id, future.result()
            fill()

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def cancel(self):
        self.stopping.set()
        self.pool.shutdown(wait=False, cancel_futures=True)


def batch_identity(value):
    # NumPy arrays use byte hashes, avoiding large JSON expansions in cache keys.
    if hasattr(value, "dtype"):
        return {"array_digest": digest(value)}
    if isinstance(value, dict):
        return {k: batch_identity(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [batch_identity(v) for v in value]
    return value


#: The module of this package that answers a GPU operation beside the model worker's own; its bytes are in
#: that operation's batch keys. `generate`, `read`, `plain_steer` and `preflight` are the model worker's.
OPERATION_MODULES = {"nla": ("nla",), "lens": ("lens",), "corpus_search": ("corpus",)}


def batch_key(run, payload, namespace):
    """The content-addressed key of one GPU batch: the payload (arrays by their bytes) and the sources that
    answer it -- the model worker, the data module and the operation's own module."""
    extra = OPERATION_MODULES.get(payload.get("operation"), ()) if isinstance(payload, dict) else ()
    return run.key(namespace, batch_identity(payload), modules=("model", "data", *extra))


def gpu_tasks(run, executor, tasks, namespace):
    """(task id, result) for every task: cached batches first, the rest as the executor finishes them, each
    saved under its own key. A caller that needs an order sorts by task id."""
    prepared, pending = {}, []
    for task_id, payload in tasks:
        key = batch_key(run, payload, namespace)
        path = f"{namespace}/batches/{key}.json"
        prepared[task_id] = (key, path)
        cached = run.cached_arrays(path, key)
        if cached is None:
            pending.append((task_id, payload))
        else:
            yield task_id, cached
    if pending and executor is None:
        raise RuntimeError(f"{namespace} needs GPU work; select a GPU backend")
    for task_id, result in executor.map(pending) if pending else ():
        key, path = prepared[task_id]
        run.save_arrays(path, key, result)
        yield task_id, result


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]
