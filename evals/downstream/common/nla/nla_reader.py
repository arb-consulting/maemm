"""The Natural Language Autoencoder verbalizer as a reader of one activation.

The released checkpoint (`Pins.repo`) is the base with the verbalizer LoRA merged in, loaded as a causal
LM. The raw layer-42 activation is injected as h + ‖h‖·v/‖v‖ at the marker token after block 1; the prompt
is the sidecar's actor template through the chat template with thinking disabled; up to 200 new tokens,
sampled with the suite's settings named on every call. A cosine scores `full_text` (tags included); a judge
or the word rule reads `text` (the first closed tag pair's body). `*_trunc` are the same views of the first
`trunc` generated ids."""

import hashlib, json, os, re
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from maem.config import INJECT_LAYER

from evals.downstream.common import model_io
from evals.downstream.common.pins import NLA_REPO, NLA_REVISION

# Vendored sidecar: nla_meta.yaml is the contract, nla_meta.json its runtime rendering (see SOURCES.md).
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "nla_assets")

SIDECAR_SHA256 = {
    "nla_meta.yaml": "e0d411c0a58d8450b4888d3df00dd72b0c0fd95e46f79500d74a44eef959f69e",
    "nla_meta.json": "63b5ffdba64d334c04b5a9cc7f576bba1c562531669e942e95740a7255f49e0f",
}


def scrub_sidecar(text):
    """The checkpoint's `nla_meta.yaml` text as vendored here: the local corpus path cut to its file name,
    the creation time and producer commit dropped; every field the reader uses is kept (SOURCES.md)."""
    out = []
    for line in text.splitlines(keepends=True):
        if re.match(r"(created_at|git_commit):", line):
            continue
        out.append(re.sub(r"^(\s+corpus: )\S*/(\S+)", r"\1\2", line))
    return "".join(out)


SHIPPED_SAMPLING_FIELDS = ("do_sample", "temperature", "top_p", "top_k")


@dataclass(frozen=True)
class Pins:
    """Every verbalizer setting a caller may pin, defaulting to the packages' values."""

    repo: str = NLA_REPO
    revision: str = NLA_REVISION
    max_new: int = 200  # the model card's own invocation
    min_new: int = 0
    trunc: int = 64  # the budget row: each sample's first `trunc` generated ids, re-decoded
    gen_chunk: int = 32
    open_tag: str = "<explanation>"
    close_tag: str = "</explanation>"
    enable_thinking: bool = False
    score_max_length: int = 256  # the re-read window: the whole generation, the sink and slack
    min_close_rate: float = 0.8  # share of samples whose tags close; a lower one is reported, never raised on
    inject_layer: int = INJECT_LAYER
    n_samples: int = 8
    gen_seed: int = 1234
    temp: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    sidecar_sha256: "MappingProxyType[str, str]" = field(
        default_factory=lambda: MappingProxyType(dict(SIDECAR_SHA256))
    )

    def __post_init__(self):
        if not isinstance(self.sidecar_sha256, MappingProxyType):  # frozen=True does not freeze a dict
            object.__setattr__(self, "sidecar_sha256", MappingProxyType(dict(self.sidecar_sha256)))
        if self.max_new > self.score_max_length - 1:
            raise ValueError(f"max_new {self.max_new} does not fit a {self.score_max_length}-token re-read "
                             "window behind its sink")
        if not 0 < self.trunc <= self.max_new:
            raise ValueError(f"the budget row's {self.trunc} tokens are not a prefix of a {self.max_new}-token "
                             "generation")


PINS = Pins()


def pins_record(pins=PINS):
    """The pins a verbalizer stage's resume hash carries."""
    return [pins.repo, pins.revision, pins.max_new, pins.top_p, pins.top_k, pins.enable_thinking,
            pins.score_max_length]


def sidecar_path(name="nla_meta.json"):
    return os.path.join(ASSETS_DIR, name)


def sha256_of(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def load_sidecar(check_sha=True, pins=PINS):
    """The vendored sidecar as a dict, after checking both files' sha256 against the pins."""
    if check_sha:
        for name, want in pins.sidecar_sha256.items():
            got = sha256_of(sidecar_path(name))
            if got != want:
                raise RuntimeError(f"vendored {name} sha256 {got} != pinned {want}")
    with open(sidecar_path(), encoding="utf-8") as h:
        return json.load(h)


def contract(sidecar):
    """The fields the reader needs, in one place."""
    t = sidecar["tokens"]
    return {
        "injection_char": t["injection_char"],
        "injection_token_id": int(t["injection_token_id"]),
        "left_id": int(t["injection_left_neighbor_id"]),
        "right_id": int(t["injection_right_neighbor_id"]),
        "d_model": int(sidecar["extraction"]["d_model"]),
        "layer_index": int(sidecar["extraction"]["layer_index"]),
        "actor_template": sidecar["prompt_templates"]["actor"],
        # the NLA reference implementation resolve_target_scale: null / "none" / "raw" all mean "inject the raw vector's direction"
        "norm": (lambda r: None if r in (None, "none", "raw") else r)(sidecar["extraction"].get("norm")),
    }


def build_prompt_text(tok, con, pins=PINS):
    """The actor template as one user message through the chat template, thinking disabled."""
    content = con["actor_template"].format(injection_char=con["injection_char"])
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=pins.enable_thinking)


def marker_position(ids, con):
    """Position of the single marker with the sidecar's neighbours, else raise."""
    pos = [k for k, t in enumerate(ids) if t == con["injection_token_id"]]
    if len(pos) != 1:
        raise RuntimeError(f"expected exactly one marker id {con['injection_token_id']}, found {len(pos)}")
    p = pos[0]
    if p == 0 or p == len(ids) - 1:
        raise RuntimeError("marker at the prompt boundary")
    if ids[p - 1] != con["left_id"] or ids[p + 1] != con["right_id"]:
        raise RuntimeError(
            f"marker neighbours {(ids[p - 1], ids[p + 1])} != sidecar {(con['left_id'], con['right_id'])}"
        )
    return p


def prompt_ids(tok, con, pins=PINS):
    """(ids, marker position) of the rendered prompt; raises if the tokenizer disagrees with the sidecar."""
    enc = tok.encode(con["injection_char"], add_special_tokens=False)
    if list(enc) != [con["injection_token_id"]]:
        raise RuntimeError(f"tokenizer drift: {con['injection_char']!r} -> {enc}, sidecar {con['injection_token_id']}")
    ids = [int(t) for t in tok.encode(build_prompt_text(tok, con, pins), add_special_tokens=False)]
    return ids, marker_position(ids, con)


def make_karvonen_hook(vecs, position, device, dtype):
    """Forward hook adding ‖h_p‖·v̂ at `position`, prefill only, one vector per batch row (the arithmetic of
    maem.inject.make_inject_hook(mode="add", coeff=1.0))."""
    import torch

    V = torch.as_tensor(np.asarray(vecs, dtype=np.float32)).to(device)
    unit = torch.nn.functional.normalize(V, dim=-1)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        if h.shape[0] != unit.shape[0]:
            raise RuntimeError(f"inject batch {h.shape[0]} != {unit.shape[0]} vector rows")
        h = h.clone()
        base = h[:, position]
        h[:, position] = base + (base.norm(dim=-1, keepdim=True) * unit).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


def explanation_body(text, pins=PINS):
    """(body, closed): the stripped text between the first closed tag pair, else the whole stripped text."""
    m = re.search(re.escape(pins.open_tag) + r"(.*?)" + re.escape(pins.close_tag), text, re.DOTALL)
    return (m.group(1).strip(), True) if m else (text.strip(), False)


def judge_view(text, pins=PINS):
    """(text a judge or the word rule reads, closed): the body when the tags close, else the whole text
    without a leading open tag, which would otherwise identify this reader to a blind judge."""
    body, closed = explanation_body(text, pins)
    if not closed and body.startswith(pins.open_tag):
        body = body[len(pins.open_tag):].strip()
    return body, closed


def describe(gen_ids, tok, max_new=None, trunc=None, pins=PINS, stop_ids=None):
    """One generation (ids after the prompt) as a sample record: `model_io.describe`'s fields, `full_text`,
    `text` and `closed` (`judge_view`), and the same three over the first `trunc` content ids."""
    max_new = pins.max_new if max_new is None else max_new
    trunc = pins.trunc if trunc is None else trunc
    stops = {int(t) for t in (stop_ids if stop_ids else (tok.eos_token_id, tok.pad_token_id)) if t is not None}
    kept, cut = model_io.trim_at_stop(gen_ids, stops)
    full = tok.decode(kept[:cut], skip_special_tokens=True)
    text, closed = judge_view(full, pins)
    if cut > trunc:
        full_trunc = tok.decode(kept[:trunc], skip_special_tokens=True)
        text_trunc, closed_trunc = judge_view(full_trunc, pins)
    else:
        full_trunc, text_trunc, closed_trunc = full, text, closed
    return {
        "text": text,
        "full_text": full,
        "ids": kept,
        "n_tokens": cut,
        "closed": closed,
        "eos": len(kept) > cut,
        "capped": cut >= max_new,
        "text_trunc": text_trunc,
        "full_text_trunc": full_trunc,
        "closed_trunc": closed_trunc,
        "trunc": trunc,
    }


def close_rate(records):
    """Share of records whose tags closed, or None over no records (reported, never raised on)."""
    records = list(records)
    return (sum(1 for r in records if r["closed"]) / len(records)) if records else None


def download_checkpoint(cache_dir=None, pins=PINS):
    """Download the pinned revision's snapshot into the HF cache; returns the snapshot directory."""
    from huggingface_hub import snapshot_download

    return snapshot_download(pins.repo, revision=pins.revision, cache_dir=cache_dir)


def check_checkpoint(snapshot_dir, pins=PINS):
    """Raise unless the snapshot is the pinned revision and ships the vendored `nla_meta.yaml` once scrubbed;
    return a provenance record with its sha256 and the shipped (unused) sampling fields."""
    got = os.path.basename(os.path.normpath(snapshot_dir))
    if got != pins.revision:
        raise RuntimeError(f"verbalizer snapshot {got!r} is not the pinned revision {pins.revision!r}")
    path = os.path.join(snapshot_dir, "nla_meta.yaml")
    if not os.path.exists(path):
        raise RuntimeError(f"no {path}: the checkpoint ships its injection contract beside its weights")
    with open(path, encoding="utf-8") as h:
        scrubbed = hashlib.sha256(scrub_sidecar(h.read()).encode("utf-8")).hexdigest()
    sha, want = sha256_of(path), pins.sidecar_sha256["nla_meta.yaml"]
    if scrubbed != want:
        raise RuntimeError(f"the snapshot's nla_meta.yaml, scrubbed, has sha256 {scrubbed} != the vendored "
                           f"sidecar's {want}")
    shipped = None
    gpath = os.path.join(snapshot_dir, "generation_config.json")
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as h:
            gen = json.load(h)
        shipped = {k: gen.get(k) for k in SHIPPED_SAMPLING_FIELDS}
    return {"revision": got, "nla_meta_sha256": sha, "shipped_generation_config": shipped,
            "sampling": {"temp": pins.temp, "top_p": pins.top_p, "top_k": pins.top_k, "min_p": pins.min_p}}


def load_verbalizer(device, snapshot_dir, pins=PINS):
    """(model, tokenizer) of the checked snapshot, about 54 GB in bf16."""
    check_checkpoint(snapshot_dir, pins)
    mdl = model_io.load_model(device, snapshot_dir, None)
    return mdl, model_io.load_tokenizer(snapshot_dir, None)


def generate_explanations(
    mdl, tok, H42, ids, marker_pos, device, n_samples=None, seed=None, gen_chunk=None, greedy=True, pins=PINS,
    row_ids=None
):
    """`n_samples` samples (and one greedy) per row of the raw activations H42 [n, d], each recorded by
    `describe`."""
    import torch
    from maem.inject import get_layer, hooked

    n_samples = pins.n_samples if n_samples is None else n_samples
    seed = pins.gen_seed if seed is None else seed
    gen_chunk = pins.gen_chunk if gen_chunk is None else gen_chunk
    H = np.asarray(H42, dtype=np.float32)
    sub = get_layer(mdl, pins.inject_layer)
    stops = model_io.stop_token_ids(tok, mdl)
    sampling = dict(temp=pins.temp, top_p=pins.top_p, top_k=pins.top_k, min_p=pins.min_p,
                    max_new=pins.max_new, min_new=pins.min_new)

    def batch_fn(rows):
        batch = torch.tensor([list(ids)] * len(rows), device=device)
        return batch, torch.ones_like(batch), len(ids)

    def hook_fn(rows):
        return hooked(sub, make_karvonen_hook(H[rows], marker_pos, device, torch.bfloat16))

    def describe_fn(gen_row, prompt_len, tok_, max_new):
        return describe(gen_row[prompt_len:].tolist(), tok_, max_new=max_new, pins=pins, stop_ids=stops)

    return model_io.generate_batches(mdl, tok, H.shape[0], device, n_samples, seed, greedy, gen_chunk,
                                     sampling, batch_fn, hook_fn, describe_fn, row_ids)
