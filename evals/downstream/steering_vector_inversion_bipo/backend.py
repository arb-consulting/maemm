"""What answers one GPU payload on a Modal container or a CUDA host: the AxBench package's model worker or
NLA worker, plus the operations this package adds."""
NLA_STAGE = "nla"


def role_of_stage(stage):
    """`nla` for the verbalizer's stage, `model` for every other."""
    return "nla" if stage == NLA_STAGE else "model"


def role_of(payload):
    from . import nla_arm

    return "nla" if payload["operation"] == nla_arm.OPERATION else "model"


def build_worker(role, config, cache_dir):
    """The AxBench package's `NlaWorker` or `ModelWorker`."""
    from evals.downstream.steering_vector_inversion.nla import ROLES, NlaWorker

    if role not in ROLES:
        raise ValueError(f"A GPU worker is one of {ROLES}, not {role!r}")
    if role == "nla":
        return NlaWorker(config, cache_dir)
    from evals.downstream.steering_vector_inversion.model import ModelWorker
    return ModelWorker(config, cache_dir)


def route(worker, payload):
    """One payload on the worker whose role it needs, or refused. `code` is part of the cache key only."""
    from . import gpu, lens_arm, nla_arm, plain_steer_arm, retrieval_arm
    role, needed = getattr(worker, "role", "model"), role_of(payload)
    if role != needed:
        raise ValueError(f"The {payload['operation']} operation is answered by a {needed!r} worker and "
                         f"was sent to a {role!r} worker")
    for module in (nla_arm, plain_steer_arm, retrieval_arm, lens_arm):
        if payload["operation"] == module.OPERATION:
            return module.execute(worker, payload)
    if payload["operation"] in gpu.OPERATIONS:
        return gpu.execute(worker, payload)
    return worker.execute({k: v for k, v in payload.items() if k not in ("code", "vector_id", "arm")})


def local(config, cache_dir):
    """`route` over one worker built in this process for the role each payload needs: a CUDA host's
    `remote`. One payload at a time, since the operations hook and train the one shared model."""
    worker = None

    def execute(payload):
        nonlocal worker
        needed = role_of(payload)
        if worker is not None and worker.role != needed:
            worker.close()
            worker = None
        if worker is None:
            worker = build_worker(needed, config, cache_dir)
        return route(worker, payload)

    return execute
