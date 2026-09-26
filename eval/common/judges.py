"""The LLM judges of every evaluation package, in two profiles of one judge each: `sonnet` (Claude Sonnet 5,
the default) and `sol` (GPT-5.6 Sol, the steering packages' default, where Sonnet refused too often).

A package's config binds its judge at import (`JUDGES = judges(caps)`), so its `__main__` calls
`activate_from_argv(argv, default=...)` and a launcher `activate(profile)` before that import. The first
read of `REFERENCE_JUDGE` / `JUDGE_ORDER`, or the first `judges()` without an explicit profile, latches the
profile; activating another one afterwards raises.
"""

from eval.common.judge_client import JudgeSpec

PROFILES = {"sonnet": ("sonnet",), "sol": ("sol",)}
DEFAULT_PROFILE = "sonnet"
PROFILE_FLAG = "--judge-profile"

# `rates` are list prices in US$ per million input / output tokens. Reasoning is off, and no `temperature`
# is sent (not every endpoint supports it).
_SPECS = {
    "sonnet": dict(
        model="claude-sonnet-5",
        provider=None,
        # Extended thinking is on by default for this model; this is the one sampling field sent.
        sampling={"thinking": {"type": "disabled"}},
        label="Claude Sonnet 5",
        rates=(2.0, 10.0),
        transport="anthropic",
    ),
    "sol": dict(
        model="openai/gpt-5.6-sol",
        provider={"order": ["openai"], "allow_fallbacks": False, "max_price": {"prompt": 2.0, "completion": 10.0}},
        sampling={"reasoning": {"enabled": False}},
        label="GPT-5.6 Sol",
        rates=(2.0, 10.0),
    ),
}

_active = DEFAULT_PROFILE
_latched = None  # the profile this process has bound names or specs under, once it has


def _check(profile):
    if profile not in PROFILES:
        raise ValueError(f"unknown judge profile {profile!r}; one of {sorted(PROFILES)}")
    return profile


def activate(profile):
    """Make `profile` the one `judges()` and the judge names resolve against, unless another is latched."""
    global _active
    _check(profile)
    if _latched is not None and _latched != profile:
        raise RuntimeError(f"judge profile {_latched!r} is already in use in this process (a package config "
                           f"or a judge name was bound under it); activate({profile!r}) must come before "
                           f"the first import of a package config")
    _active = profile
    return profile


def active():
    return _active


def profile_from_argv(argv, default=DEFAULT_PROFILE):
    """The value of `--judge-profile` in `argv` (either spelling, the last winning), else `default`; read by
    hand, since it is needed before the package's parser exists."""
    profile = default
    args = list(argv)
    for i, arg in enumerate(args):
        if arg == PROFILE_FLAG:
            if i + 1 >= len(args):
                raise SystemExit(f"{PROFILE_FLAG} needs a value; one of {sorted(PROFILES)}")
            profile = args[i + 1]
        elif arg.startswith(PROFILE_FLAG + "="):
            profile = arg.split("=", 1)[1]
    if profile not in PROFILES:
        raise SystemExit(f"{PROFILE_FLAG} {profile!r}: one of {sorted(PROFILES)}")
    return profile


def activate_from_argv(argv, default=DEFAULT_PROFILE):
    return activate(profile_from_argv(argv, default))


def add_profile_argument(parser, default=DEFAULT_PROFILE):
    """The flag on a package's own parser, so `--help` lists it; pass the package's default profile."""
    parser.add_argument(PROFILE_FLAG, dest="judge_profile", choices=sorted(PROFILES), default=default,
                        help=f"which judge answers (default {default}): "
                             + "; ".join(f"{p} = {_SPECS[names[0]]['label']}" for p, names in PROFILES.items()))


def judges(max_tokens, profile=None):
    """{name: JudgeSpec} of a profile's judge with the package's token caps (an int, or a dict by `kind`).
    Without `profile`, the active one, which this call latches."""
    global _latched
    if profile is None:
        profile = _latched = _active
    return {name: JudgeSpec(name=name, max_tokens=max_tokens, **_SPECS[name]) for name in PROFILES[_check(profile)]}


def label(name):
    """The display name of a judge."""
    return _SPECS[name]["label"]


def rates(specs):
    """{model id: (input, output) US$ per million tokens}, the mapping `Ledger` prices requests with."""
    return {s.model: s.rates for s in specs.values()}


def __getattr__(name):
    global _latched
    if name in ("REFERENCE_JUDGE", "JUDGE_ORDER"):
        _latched = _active
        order = PROFILES[_active]
        return order if name == "JUDGE_ORDER" else order[0]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
