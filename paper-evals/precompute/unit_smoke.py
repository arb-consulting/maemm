# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.10.0", "transformers==5.15.0", "numpy==2.4.6", "pyyaml"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""CPU unit smoke for precompute/common.py -- no weights, no GPU, no network.

    uv run paper-evals/precompute/unit_smoke.py

Everything here is checked against an INDEPENDENT computation: hook placement against
`output_hidden_states`, the scorer against a hand-built mask and a direct einsum, the SAE against
the formula written out. A test that can only agree with the code it tests proves nothing, so each
check below was also run against a deliberately broken variant while it was written (see
SMOKES.md).

The real chat template is NOT exercised here (it needs a real tokenizer): the Modal `check` product
builds both prompts on the real 8B/27B tokenizers and asserts the marker is one token occurring
exactly once. What this file covers of the prompt path is the assertion behaviour.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import precompute.common as C  # noqa: E402

D = 64
N_LAYERS = 4  # >= 3 so that an inject layer, a strictly earlier layer and a later read layer exist
VOCAB = 128
SINK = 1
PAD = 0


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


def tiny_model(seed=0):
    """A randomly initialised Qwen3 with the real block structure, in fp32 on the cpu."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=D,
        intermediate_size=2 * D,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    return model


class FakeTok:
    """The tokenizer surface score_tokens uses: one token per character, right padding.

    Deliberately not a real tokenizer -- the point is that the expected ids of every test string
    are obvious by hand, so the mask/BOS/padding behaviour can be checked against a literal.
    """

    def __init__(self, bos=SINK, eos=2, pad=PAD):
        self.padding_side = "left"  # score_tokens must flip this to 'right' and restore it
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos, eos, pad

    def ids_of(self, text):
        return [(ord(ch) % (VOCAB - 8)) + 8 for ch in text]

    def __call__(
        self,
        batch,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=None,
        add_special_tokens=True,
    ):
        assert truncation, "the re-encode protocol truncates at SCORE_MAX_LENGTH"
        assert not add_special_tokens, "the re-encode protocol tokenizes the text ALONE"
        assert self.padding_side == "right", f"padding_side is {self.padding_side}, want 'right'"
        rows = [self.ids_of(t)[:max_length] for t in batch]
        if not padding and return_tensors is None:
            # encode_for_score's call: ids only, no padding and no tensors -- score_ids pads.
            return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]}
        assert return_tensors == "pt", f"unexpected return_tensors {return_tensors!r}"
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            am[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": am}


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------


def check_config():
    cfg = C.load_config()
    assert cfg["bases"]["qwen3-8b"]["read_layer"] == 27
    assert cfg["bases"]["qwen36-27b"]["d"] == 5120
    assert set(C.families_for(cfg, "2026-09-16_v1", "qwen3-8b")) == {"realact", "random", "sae"}, (
        "jlens is declared 27B-only, so the 8B set must not carry it"
    )
    assert "jlens" in C.families_for(cfg, "2026-09-16_v1", "qwen36-27b")

    # The max_new >= max_length - 1 guard (checklist item 7) must actually fire.
    import yaml

    raw = yaml.safe_load(C.CONFIG_PATH.read_text())
    raw["rollouts"]["max_new"] = C.SCORE_MAX_LENGTH  # one token too long
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "config.yaml"
        bad.write_text(yaml.safe_dump(raw))
        try:
            C.load_config(bad)
        except AssertionError as e:
            assert "SCORE_MAX_LENGTH" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("load_config accepted max_new = SCORE_MAX_LENGTH")


def check_paths():
    assert C.heldout_dir("qwen3-8b", "s") == "/vol/base/qwen3-8b/heldout/s"
    assert C.sae_dir("qwen36-27b/l42-1b") == "/vol/base/qwen36-27b/sae/l42-1b"
    assert C.scores_dir("qwen3-8b/2026-09-03_run1-rl", "s") == (
        "/vol/maemms/qwen3-8b/2026-09-03_run1-rl/scores/s"
    )
    try:
        C.split_key("no-slash", "maemm")
    except AssertionError as e:
        assert "exactly" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("split_key accepted a key without a base")


def check_read_hook():
    """read_resid returns the BLOCK OUTPUT, i.e. output_hidden_states[L + 1]."""
    model = tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])
    am = torch.ones_like(am_like := ids)
    del am_like
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
    for layer in range(N_LAYERS - 1):  # the last entry is post-final-norm, not a block output
        h, mask = C.read_resid(model, layer, {"input_ids": ids, "attention_mask": am}, pool="all")
        assert torch.allclose(h, hs[layer + 1].float(), atol=1e-5), (
            f"read hook at layer {layer} does not match output_hidden_states[{layer + 1}]; "
            f"max |diff| {(h - hs[layer + 1].float()).abs().max():.2e}"
        )
        assert mask.shape == ids.shape and mask.all()
    # A read at a layer that does not exist must fail loudly, not return the wrong tensor.
    try:
        C.read_resid(model, N_LAYERS + 3, {"input_ids": ids, "attention_mask": am})
    except IndexError:
        pass
    else:
        raise AssertionError("read_resid accepted a layer index past the end of the model")


def check_inject_hook():
    """Injection at block INJECT's output touches exactly one position, at exactly that layer."""
    model = tiny_model()
    inject, pos = 1, 3
    ids = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    am = torch.ones_like(ids)
    v = torch.randn(2, D)
    with torch.no_grad():
        clean = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
    hook = C.make_inject_hook([v[0:1], v[1:2]], [[pos], [pos]], 1.0, "cpu", torch.float32)
    with torch.no_grad(), C.hooked(C.get_layer(model, inject), hook):
        dirty = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states

    assert torch.allclose(clean[inject], dirty[inject], atol=1e-6), (
        "the layer BELOW the injection changed: the hook is on the wrong module"
    )
    diff = (clean[inject + 1] - dirty[inject + 1]).abs().sum(-1)  # [B, T]
    assert (diff[:, pos] > 0).all(), "the marker position did not change at the injection layer"
    other = torch.cat([diff[:, :pos], diff[:, pos + 1 :]], dim=1)
    assert other.max() == 0, (
        f"positions other than the marker changed at the injection layer (max {other.max():.2e})"
    )
    # Causality: later layers see the change at the marker and after it, never before it.
    later = (clean[inject + 2] - dirty[inject + 2]).abs().sum(-1)
    assert later[:, :pos].max() == 0 and (later[:, pos:] > 0).all(), (
        "the injection leaked to positions before the marker at a later layer"
    )

    # The formula itself: h[pos] += unit(v) * ||h[pos]|| * coeff.
    for b in range(2):
        base = clean[inject + 1][b, pos].float()
        want = base + torch.nn.functional.normalize(v[b], dim=-1) * base.norm() * 1.0
        assert torch.allclose(dirty[inject + 1][b, pos].float(), want, atol=1e-4), (
            f"row {b}: injected value is not base + unit(v)*||base||*coeff"
        )

    # The decode-step guard (seq_len 1 under the KV cache) must be a no-op.
    out = torch.zeros(2, 1, D)
    assert hook(None, None, out) is out, "the hook fired on a decode step"


def check_marker_norm():
    model = tiny_model()
    ids = [5, 6, 7, 8, 9]
    pos, inject = 3, 1
    with torch.no_grad():
        hs = model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            output_hidden_states=True,
        ).hidden_states
    want = hs[inject + 1][0, pos].float().norm().item()
    got = C.marker_norm(model, ids, pos, inject)
    assert abs(got - want) < 1e-4, f"marker_norm {got} != ||h[{pos}]|| at layer {inject} = {want}"


def check_score_tokens():
    model, tok = tiny_model(), FakeTok()
    read_layer = 2
    texts = ["abcde", "xy", "   ", "q" * 200]
    dirs = torch.randn(len(texts), D)
    out = C.score_tokens(model, tok, texts, dirs, read_layer, sbatch=2, device="cpu")

    assert tok.padding_side == "left", "score_tokens did not restore the tokenizer's padding_side"
    for k in ("cos", "norm", "keep", "ids"):
        assert out[k].shape == (len(texts), C.SCORE_WIDTH), f"{k} has shape {out[k].shape}"

    # keep: BOS never, padding never, content always. Widths are 5, 2, 1 (' ' substituted), 95.
    want_len = [5, 2, 1, C.SCORE_MAX_LENGTH]
    for i, n in enumerate(want_len):
        assert not out["keep"][i, 0], f"row {i}: the BOS sink is kept"
        assert out["keep"][i].sum().item() == n, (
            f"row {i}: kept {out['keep'][i].sum().item()} tokens, expected {n} "
            f"(text len {len(texts[i])}, truncation at {C.SCORE_MAX_LENGTH})"
        )
        assert out["keep"][i, 1 : n + 1].all() and not out["keep"][i, n + 1 :].any()
        assert out["ids"][i, 0].item() == SINK, f"row {i}: column 0 is not the sink"
        assert torch.isnan(out["cos"][i, n + 1 :]).all(), f"row {i}: padding is not NaN in cos"
        assert torch.isnan(out["norm"][i, 0]), f"row {i}: the sink carries a norm"
    assert out["ids"][1, 3].item() == -1, "padding token ids must be -1, not the pad id"
    assert out["ids"][0, 1:6].tolist() == tok.ids_of("abcde"), "row 0 ids are not the text's ids"
    kept = out["cos"][out["keep"]]  # NaN lives outside `keep`, so reduce over the kept entries only
    assert kept.max() <= 1.0 + 1e-5 and kept.min() >= -1.0 - 1e-5, "cos outside [-1, 1]"
    assert not torch.isnan(kept).any(), "a kept token has a NaN cosine"

    # Independent recomputation of row 0 from output_hidden_states.
    ids0 = torch.tensor([[SINK, *tok.ids_of("abcde")]])
    with torch.no_grad():
        hs = (
            model(input_ids=ids0, attention_mask=torch.ones_like(ids0), output_hidden_states=True)
            .hidden_states[read_layer + 1]
            .float()
        )
    want_cos = torch.einsum(
        "btd,bd->bt",
        torch.nn.functional.normalize(hs, dim=-1),
        torch.nn.functional.normalize(dirs[0:1], dim=-1),
    )[0]
    assert torch.allclose(out["cos"][0, 1:6], want_cos[1:6], atol=1e-5), (
        f"row 0 cos differs from the direct einsum by {(out['cos'][0, 1:6] - want_cos[1:6]).abs().max():.2e}"
    )
    assert torch.allclose(out["norm"][0, 1:6], hs[0, 1:6].norm(dim=-1), atol=1e-4)


def check_score_ids_is_score_tokens():
    """`score_tokens(texts)` IS `score_ids(the ids of those texts)` -- bit for bit.

    The two paths must not be able to drift: rollouts and the SAE repo windows come in as text, the
    GCG loop comes in as ids, and the paper compares their cosines directly. The id lists here are
    built INDEPENDENTLY of `common.encode_for_score` (straight off the fixture tokenizer, truncated
    and blank-substituted by hand), so this is a real cross-check and not a restatement of the
    wrapper. It also pins the two guards `score_ids` owes a caller that skips the tokenizer: an
    over-length row and an empty row must both raise rather than be silently re-shaped.
    """
    model, tok = tiny_model(), FakeTok()
    read_layer = 2
    texts = ["abcde", "xy", "   ", "q" * 200]
    dirs = torch.randn(len(texts), D)
    # hand-built: the whitespace row becomes " " (one id) and the long row truncates at 95
    by_hand = [tok.ids_of(t if t.strip() else " ")[: C.SCORE_MAX_LENGTH] for t in texts]
    assert [len(x) for x in by_hand] == [5, 2, 1, C.SCORE_MAX_LENGTH], (
        f"the fixture's own id lengths moved: {[len(x) for x in by_hand]}"
    )
    assert C.encode_for_score(tok, texts) == by_hand, (
        "encode_for_score does not reproduce the protocol's ids (blank substitution / truncation)"
    )

    a = C.score_tokens(model, tok, texts, dirs, read_layer, sbatch=2, device="cpu")
    b = C.score_ids(model, tok, by_hand, dirs, read_layer, sbatch=2, device="cpu")
    for k in ("cos", "norm", "keep", "ids"):
        same = torch.equal(torch.nan_to_num(a[k], nan=-12345.0), torch.nan_to_num(b[k], nan=-12345.0))
        assert same, (
            f"score_tokens and score_ids disagree on `{k}`: max |d| "
            f"{(a[k].float() - b[k].float()).abs().nan_to_num().max().item():.3e} -- the text path "
            f"and the id path are no longer one forward"
        )

    # A row above the re-encode truncation must be REFUSED, not silently truncated: the caller that
    # skips the tokenizer (gcg) would otherwise optimise a string longer than the one scored.
    try:
        C.score_ids(model, tok, [[7] * (C.SCORE_MAX_LENGTH + 1)], dirs[:1], read_layer, device="cpu")
    except AssertionError as e:
        assert "above the re-encode truncation" in str(e), f"wrong assert fired for an over-length row: {e}"
    else:
        raise AssertionError("score_ids accepted a row longer than SCORE_MAX_LENGTH")
    try:
        C.score_ids(model, tok, [[]], dirs[:1], read_layer, device="cpu")
    except AssertionError as e:
        assert "no tokens" in str(e), f"wrong assert fired for an empty row: {e}"
    else:
        raise AssertionError("score_ids accepted a row with no tokens")


def check_no_norm_filter():
    """The stored per-token values are UNFILTERED (Tomáš 2026-09-15): the 10x-nanmedian filter
    (eval_universal.py:71,145-147) belongs to reconstruction/stats.py, as an option.

    A token with an outsized residual norm is manufactured by scaling one embedding row, so this
    check goes red the moment score_tokens starts dropping such tokens.
    """
    model, tok = tiny_model(), FakeTok()
    text = "abcde"
    hot = tok.ids_of(text)[2]
    with torch.no_grad():
        model.model.embed_tokens.weight[hot] *= 400.0
    out = C.score_tokens(model, tok, [text], torch.randn(1, D), 2, device="cpu")
    nrm = out["norm"][0, 1:6]
    assert out["keep"][0, 1:6].all(), (
        "a token was dropped from `keep`: score_tokens must not apply the 10x-median norm filter"
    )
    ratio = (nrm[2] / nrm.median()).item()
    assert ratio > 10.0, f"the fixture failed to make an outlier: norm ratio {ratio:.1f} <= 10"
    assert not torch.isnan(out["cos"][0, 3]), "the outlier token has no stored cosine"


def check_agg():
    # Row 3 carries a FINITE value at a column that is NOT kept (a sink scoring 0.99): agg must
    # exclude it by the mask, not by relying on the NaN that score_tokens happens to store there.
    cos = torch.tensor(
        [
            [float("nan"), 0.1, 0.7, 0.3, float("nan")],
            [float("nan"), -0.9, float("nan"), float("nan"), float("nan")],
            [float("nan")] * 5,
            [0.99, 0.2, float("nan"), float("nan"), float("nan")],
        ]
    )
    keep = torch.tensor(
        [
            [False, True, True, True, False],
            [False, True, False, False, False],
            [False] * 5,
            [False, True, False, False, False],
        ]
    )
    best, arg = C.agg(cos, keep)
    assert torch.allclose(best, torch.tensor([0.7, -0.9, -1.0, 0.2]), atol=1e-6), best.tolist()
    # row 2 has no kept token: best -1.0 at column 0, which stats.py separates by keep.sum() == 0
    assert arg.tolist() == [2, 1, 0, 1], arg.tolist()


def check_eos_and_trim():
    class GenCfg:
        eos_token_id = [7, 11]

    class M:
        generation_config = GenCfg()

    tok = FakeTok(eos=7)
    assert C.eos_ids(tok, M()) == {7, 11}, "eos must be the UNION of tokenizer and generation config"

    class NoGen:
        pass

    assert C.eos_ids(tok, NoGen()) == {7}, "a model without a generation config contributes nothing"
    assert C.trim_at_stop([3, 4, 7, 9, 0], {7, 11}) == [3, 4, 7], "the stop token is kept"
    assert C.trim_at_stop([3, 4], {7}) == [3, 4], "a rollout without a stop token is unchanged"


def check_sae():
    """nn.Linear-shaped checkpoint: key aliases, the transpose, the gate, and the encode formula."""
    torch.manual_seed(1)
    d, f = 16, 32
    W_enc_lin = torch.randn(f, d)  # nn.Linear stores [out, in]
    W_dec_lin = torch.nn.functional.normalize(torch.randn(f, d), dim=-1).T.contiguous()  # [d, f]
    b_enc, b_dec = torch.randn(f), torch.randn(d)
    ckpt = {
        "encoder.weight": W_enc_lin,
        "decoder.weight": W_dec_lin,
        "encoder.bias": b_enc,
        "bias": b_dec,
        "threshold": torch.tensor(1.654),
        "unused": torch.zeros(3),
    }
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/ae.pt"
        torch.save(ckpt, path)
        sae = C.load_sae(path, d_model=d)
        assert sae.W_enc.shape == (d, f), f"W_enc must be [d, F], got {tuple(sae.W_enc.shape)}"
        assert sae.W_dec.shape == (f, d), f"W_dec must be [F, d], got {tuple(sae.W_dec.shape)}"
        assert abs(sae.threshold - 1.654) < 1e-6, "the learned BatchTopK gate was not kept"
        assert torch.allclose(sae.b_dec, b_dec), "the 'bias' alias did not map to b_dec"

        ids = [0, 5, 31]
        x = torch.randn(2, 3, d)
        want = torch.relu((x - b_dec) @ W_enc_lin.T[:, ids] + b_enc[ids])
        assert torch.allclose(C.sae_encode(sae, x, ids), want, atol=1e-5), (
            "sae_encode is not relu((x - b_dec) @ W_enc[:, ids] + b_enc[ids])"
        )
        dirs = C.sae_dirs(sae, ids)
        assert torch.allclose(dirs, torch.nn.functional.normalize(W_enc_lin[ids], dim=-1), atol=1e-6), (
            "the sae family direction must be the unit ENCODER column"
        )

        del ckpt["threshold"]
        torch.save(ckpt, path)
        try:
            C.load_sae(path, d_model=d)
        except AssertionError as e:
            assert "threshold" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("load_sae accepted a checkpoint with no gate threshold")


def check_io_and_outdir():
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "scan"
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        with C.OutDir(
            target,
            argv=["smoke"],
            commit="deadbeef",
            inputs={"corpus": "p0009"},
            provenance={"n": 3, "bo": 64, "seed": 1234, "sha": "abc"},
        ) as out:
            out.write_array("cos.f16", arr, "float16")
            out.write_jsonl("ids.jsonl", [{"row": i} for i in range(3)])
            out.note("written by the unit smoke")
            assert not target.exists(), "OutDir must not create the final path before it completes"
            assert out.tmp.exists() and ".tmp-" in out.tmp.name
        assert target.exists() and not out.tmp.exists(), "the temp dir was not renamed"

        idx = json.loads((target / "index.json").read_text())
        assert idx["cos.f16"] == {"kind": "array", "dtype": "float16", "shape": [3, 4], "bytes": 24}
        assert idx["ids.jsonl"]["rows"] == 3
        back = C.read_array(target / "cos.f16", "float16", (3, 4))
        assert np.array_equal(back.astype(np.float32), arr), "array round trip changed the values"
        assert C.read_jsonl(target / "ids.jsonl") == [{"row": i} for i in range(3)]
        readme = (target / "README.md").read_text()
        for needle in ("smoke", "deadbeef", "p0009", "seed: 1234", "cos.f16", "unit smoke"):
            assert needle in readme, f"README does not record {needle!r}:\n{readme}"

        try:
            with C.OutDir(target, argv=["smoke"]):
                pass
        except AssertionError as e:
            assert "--force" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("OutDir overwrote an existing directory without --force")

        with C.OutDir(target, force=True, argv=["smoke"]) as out:
            out.write_jsonl("ids.jsonl", [{"row": 99}])
        assert C.read_jsonl(target / "ids.jsonl") == [{"row": 99}], "--force did not replace"
        assert not (target / "cos.f16").exists(), "--force left a file from the previous version"

        # A failing writer leaves the temp dir behind and never renames.
        target2 = Path(td) / "boom"
        try:
            with C.OutDir(target2, argv=["smoke"]) as out:
                out.write_jsonl("x.jsonl", [{"a": 1}])
                raise RuntimeError("deliberate")
        except RuntimeError:
            pass
        assert not target2.exists() and out.tmp.exists(), (
            "a failed OutDir must leave <name>.tmp-<date>/ and no final directory"
        )


def check_sha256_of_weights():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "model-1.safetensors").write_bytes(b"abc")
        (p / "model-2.safetensors").write_bytes(b"def")
        (p / "README.md").write_text("not a weight file")
        got = C.sha256_of_weights(p)
        assert sorted(got["files"]) == ["model-1.safetensors", "model-2.safetensors"], got["files"]
        assert got["files"]["model-1.safetensors"]["sha256"] == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ), "sha256('abc')"
        (p / "model-2.safetensors").write_bytes(b"deg")
        assert C.sha256_of_weights(p)["sha256"] != got["sha256"], "the combined digest ignores content"


def check_prompt_assertions():
    """The real templates run in the Modal `check`; here only the guards are exercised."""

    class MultiTokTok:
        def encode(self, text, add_special_tokens=True):
            return [11, 12]

    try:
        C.marker_positions(MultiTokTok(), [11, 12, 13])
    except AssertionError as e:
        assert "single-token" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("marker_positions accepted a multi-token marker")

    class OneTokTok:
        def encode(self, text, add_special_tokens=True):
            return [11]

    assert C.marker_positions(OneTokTok(), [3, 11, 4]) == [1]
    try:
        C.marker_positions(OneTokTok(), [11, 3, 11])
    except AssertionError as e:
        assert "exactly one marker" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("marker_positions accepted two markers")
    try:
        C.prompt_ids(OneTokTok(), "nope", 27)
    except AssertionError as e:
        assert "unknown prompt" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("prompt_ids accepted an unregistered prompt name")


# ---------------------------------------------------------------------------------------------
# corpus geometry (stats.py pass A and scan.py pass B must agree on it, so it lives in common.py)
# ---------------------------------------------------------------------------------------------


def check_window_geometry():
    """windows_of against the stated rule, re-derived here rather than restated."""
    for n in list(range(1, 200)) + [256, 512, 1000, 1024]:
        w = C.windows_of(n)
        assert w[0][0] == 0, f"n={n}: the first window must start at 0, got {w[0]}"
        # every token is covered
        covered = set()
        for s, ln in w:
            assert 0 < ln <= C.SCAN_BLOCK, f"n={n}: window {(s, ln)} is not 1..{C.SCAN_BLOCK} tokens"
            assert s + ln <= n, f"n={n}: window {(s, ln)} runs past the document"
            covered |= set(range(s, s + ln))
        assert covered == set(range(n)), f"n={n}: windows do not cover the document"
        starts = [s for s, _ in w]
        assert starts == sorted(set(starts)), f"n={n}: starts not strictly increasing: {starts}"
        for a, b in zip(starts, starts[1:], strict=False):
            assert b - a == C.SCAN_STRIDE, f"n={n}: start gap {b - a} != stride"
        if n <= C.SCAN_BLOCK:
            assert w == [(0, n)], f"n={n}: a short document must be ONE window, got {w}"
        else:
            full = [x for x in w if x[1] == C.SCAN_BLOCK]
            assert full == w[: len(full)], f"n={n}: the partial window must be last: {w}"
            assert len(w) - len(full) <= 1, f"n={n}: at most one partial window, got {w}"
            # the partial window exists exactly when the last full window leaves a tail
            tail = full[-1][0] + C.SCAN_BLOCK < n
            assert tail == (len(w) > len(full)), f"n={n}: partial-window rule broken: {w}"
    # blocks_of went with the acts_1m store (dropped 2026-09-15); windows_of is the only geometry.
    try:
        C.windows_of(0)
    except AssertionError as e:
        assert "empty document" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("windows_of accepted an empty document")


def check_size_tags():
    """Tags are non-decreasing in stored order and each subset is a prefix -- the property every
    nested-size snapshot in stats.py and scan.py relies on."""
    import numpy as np

    sizes = [1, 2, 4]
    rng = np.random.default_rng(0)
    lens = rng.integers(1, 5000, 4000)
    cum, tags = 0, []
    for n in lens:
        if cum + n > sizes[-1] * 1_000_000:
            break
        tags.append(C.size_tag_of(cum, int(n), sizes))
        cum += int(n)
    assert tags == sorted(tags), "size tags are not non-decreasing in stored order"
    off = 0
    for s in sizes:
        tot = sum(int(n) for n, t in zip(lens, tags, strict=False) if t <= s)
        assert tot <= s * 1_000_000, f"subset {s}M holds {tot} tokens, over budget"
        assert tot >= off, "subsets are not nested"
        off = tot
    assert C.size_tag_of(0, 10, [1, 2]) == 1
    assert C.size_tag_of(0, 1_000_000, [1, 2]) == 1, "a document that exactly fills a size stays in it"
    assert C.size_tag_of(1, 1_000_000, [1, 2]) == 2
    assert C.size_tag_of(999_999, 10, [1, 2]) == 2, "a document that does not fit must move up a tag"
    assert C.size_tag_of(0, 3_000_000, [1, 2]) == 2, "an over-budget document is clamped to the top"


def check_quantiles_from_hist():
    """Histogram quantiles against np.quantile on the same sample, to within one bin width."""
    import numpy as np

    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0.1, 0.25, 200_000), -0.999, 0.999)
    bins = 512
    idx = np.clip(((x + 1) * (bins / 2)).astype(int), 0, bins - 1)
    hist = np.bincount(idx, minlength=bins)[None, :]
    qs = [0.5, 0.9, 0.99, 0.999]
    got = C.quantiles_from_hist(hist, qs)[0]
    want = np.quantile(x, qs)
    assert got.shape == (len(qs),), f"shape {got.shape}"
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) <= 2 / bins + 1e-9, f"histogram quantile {g} vs exact {w}"
    # an empty row returns the lower bound rather than raising
    assert float(C.quantiles_from_hist(np.zeros((1, bins), dtype=np.int64), [0.5])[0, 0]) == -1.0
    # convention: the quantile is the ceil(q*N)-th smallest value's bin upper edge
    h4 = np.array([[1, 1, 1, 1]], dtype=np.int64)  # 4 bins of width 0.5 over [-1, 1]
    assert float(C.quantiles_from_hist(h4, [0.6])[0, 0]) == 0.5, "q=0.6 of 4 items is the 3rd item"
    assert float(C.quantiles_from_hist(h4, [0.5])[0, 0]) == 0.0, "q=0.5 of 4 items is the 2nd item"
    assert float(C.quantiles_from_hist(h4, [0.25])[0, 0]) == -0.5
    # a single mass point lands in its own bin
    h = np.zeros((1, bins), dtype=np.int64)
    h[0, 384] = 7  # bin 384 covers (0.5, 0.50390625]
    v = float(C.quantiles_from_hist(h, [0.5])[0, 0])
    assert abs(v - 0.50390625) < 1e-6, v


def check_load_corpus():
    """load_corpus round trip and the tokens/docs consistency assert."""
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "base" / "b" / "corpus"
        d.mkdir(parents=True)
        lens = [5, 3, 7]
        C.write_array(d / "tokens.i32", np.arange(sum(lens)), "int32")
        off, rows = 0, []
        for i, n in enumerate(lens):
            rows.append({"doc": i, "offset": off, "len": n, "size_tag": 1, "part": 0, "row": i})
            off += n
        C.write_jsonl(d / "docs.jsonl", rows)
        toks, docs = C.load_corpus("b", str(Path(td)))
        assert len(toks) == 15 and len(docs) == 3
        assert list(toks[docs[2]["offset"] : docs[2]["offset"] + docs[2]["len"]]) == list(range(8, 15))
        assert C.corpus_sizes(docs) == [1]
        rows[-1]["len"] = 6  # now the docs no longer account for every token
        C.write_jsonl(d / "docs.jsonl", rows)
        try:
            C.load_corpus("b", str(Path(td)))
        except AssertionError as e:
            assert "tokens.i32" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("load_corpus accepted a docs.jsonl that does not cover the tokens")


def check_outdir_cost():
    """The README carries the gpu and the product's own cost (checklist item 84)."""
    with tempfile.TemporaryDirectory() as td:
        with C.outdir(
            Path(td) / "p",
            {"argv": ["x"], "repo_commit": "abc", "gpu": "H100", "usd_per_s": 1.0, "force": False},
        ) as od:
            od.write_json("a.json", {"k": 1})
        txt = (Path(td) / "p" / "README.md").read_text()
        assert "- gpu: H100" in txt, txt
        assert "- cost: $" in txt, txt
        assert "3600.00 $/h" in txt, txt


def check_best_of_k_means():
    """The per_target bo_<k> aggregation: disjoint groups of k, group max, mean of the maxima."""
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    got = C.best_of_k_means(vals, (1, 2, 4, 8, 16))
    # independent recomputation, written out rather than reusing the code under test
    assert got[1] == 3.5, f"bo1 must be the plain mean 3.5, got {got[1]}"
    assert got[2] == (1 + 3 + 5 + 7) / 4, f"bo2 must average the 4 pair maxima, got {got[2]}"
    assert got[4] == (3 + 7) / 2, f"bo4 must average the 2 quad maxima, got {got[4]}"
    assert got[8] == 7.0, f"bo8 must be the single max 7.0, got {got[8]}"
    assert 16 not in got, (
        "k > n must be SKIPPED, not clamped: a summary may not claim a bo it could not compute"
    )
    # groups are DISJOINT and CONSECUTIVE, so order matters -- a sorted copy scores differently
    assert C.best_of_k_means([7.0, 0.0, 6.0, 1.0], (2,))[2] == 6.5
    assert C.best_of_k_means([0.0, 1.0, 6.0, 7.0], (2,))[2] == 4.0
    # a remainder is dropped, not folded into the last group
    assert C.best_of_k_means([0.0, 9.0, 5.0], (2,))[2] == 9.0, "the odd tail must be dropped"
    try:
        C.best_of_k_means(vals, (0,))
    except AssertionError as e:
        assert "k >= 1" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("best_of_k_means accepted k = 0")


def check_parse_rows_and_gen_seed():
    assert C.parse_rows("", 5) == [0, 1, 2, 3, 4], "an empty --rows means every row"
    assert C.parse_rows("0-7", 16) == list(range(8))
    assert C.parse_rows("3,5,9-11", 16) == [3, 5, 9, 10, 11]
    assert C.parse_rows("2,2,1-2", 16) == [1, 2], "duplicates collapse"
    for bad, want in (("0-16", "outside"), ("5-3", "backwards"), ("99", "outside")):
        try:
            C.parse_rows(bad, 16)
        except AssertionError as e:
            assert want in str(e), f"--rows {bad!r}: wrong assert fired: {e}"
        else:
            raise AssertionError(f"parse_rows accepted {bad!r}")

    # flat = row * n + k, so the rule is chunking-independent at aligned boundaries
    assert C.gen_seed_for(1234, 0, 0, 64) == 1234000
    assert C.gen_seed_for(1234, 0, 32, 64) == 1234032
    assert C.gen_seed_for(1234, 4, 0, 64) == 1234256, "the 8B's 256-row call starts at target 4"
    assert C.gen_seed_for(1234, 1, 0, 64) != C.gen_seed_for(1234, 0, 0, 64)
    try:
        C.gen_seed_for(1234, 0, 64, 64)
    except AssertionError as e:
        assert "k=64" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("gen_seed_for accepted k == n")


def check_outdir_keep_existing_and_section():
    """rollouts/ accumulates one <set>.jsonl per run; a README section carries a real heading."""
    with tempfile.TemporaryDirectory() as td:
        args = {"argv": ["x"], "repo_commit": "abc", "gpu": "CPU", "usd_per_s": 0.0, "force": False}
        d = Path(td) / "rollouts"
        with C.outdir(d, args) as od:
            od.write_jsonl("setA.jsonl", [{"a": 1}])
        with C.outdir(d, args, keep_existing=True) as od:
            od.write_jsonl("setB.jsonl", [{"b": 2}])
            od.section("Methods", ["- line one", "- line two"])
        assert (d / "setA.jsonl").exists(), "keep_existing must preserve the sets already there"
        assert (d / "setB.jsonl").exists()
        idx = json.loads((d / "index.json").read_text())
        assert sorted(idx) == ["setA.jsonl", "setB.jsonl"], (
            f"index.json must list every set present and not itself, got {sorted(idx)}"
        )
        txt = (d / "README.md").read_text()
        assert "## Methods" in txt and "- line one" in txt, txt
        assert txt.index("## Methods") < txt.index("## Files"), "sections go before the file table"
        # without keep_existing the same path is still refused
        try:
            with C.outdir(d, args):
                pass
        except AssertionError as e:
            assert "--force" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("outdir overwrote an existing directory without --force")


def check_sha256_of_index():
    """The cheap identity for a sharded 52 GiB model: index.json content + shard NAMES and SIZES."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "model.safetensors.index.json").write_text('{"weight_map": {"a": "model-00001.safetensors"}}')
        (d / "model-00001.safetensors").write_bytes(b"x" * 100)
        (d / "model-00002.safetensors").write_bytes(b"y" * 200)
        a = C.sha256_of_index(d)
        assert a["bytes"] == 300, a["shards"]
        assert sorted(a["shards"]) == ["model-00001.safetensors", "model-00002.safetensors"]
        assert "not hashed" in a["kind"], a["kind"]
        # content changes of the same SIZE are invisible -- that is the documented weakness
        (d / "model-00001.safetensors").write_bytes(b"z" * 100)
        assert C.sha256_of_index(d)["sha256"] == a["sha256"], (
            "same sizes + same index must hash the same; if this ever changes, the README claim "
            "that shard CONTENT is not hashed is wrong"
        )
        # a size change, or an index change, must move it
        (d / "model-00001.safetensors").write_bytes(b"z" * 101)
        assert C.sha256_of_index(d)["sha256"] != a["sha256"], "a shard size change must change the digest"
        (d / "model-00001.safetensors").write_bytes(b"x" * 100)
        (d / "model.safetensors.index.json").write_text('{"weight_map": {"a": "model-00002.safetensors"}}')
        assert C.sha256_of_index(d)["sha256"] != a["sha256"], "an index change must change the digest"
        try:
            C.sha256_of_index(Path(td) / "nope")
        except AssertionError as e:
            assert "index.json" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("sha256_of_index accepted a directory with no index")


def check_vllm_finish_ids():
    """The vLLM -> HF row fixup: the engine drops the stop token, rollouts_hf keeps it.

    Checked against the rule written out, not against the function: a "stop" row must end in a
    stop id, a "length" row must be returned untouched, and the trim must cut at the FIRST stop.
    """
    stop = {7, 9}
    # (a) finish_reason "stop", engine dropped the token: stop_reason names which one
    ids, app = C.vllm_finish_ids([1, 2, 3], "stop", 9, stop, eos_fallback=7)
    assert ids == [1, 2, 3, 9] and app, f"the dropped stop token must be re-appended, got {ids}"
    # (b) a non-integer stop_reason (a stop STRING) falls back to the tokenizer eos
    ids, app = C.vllm_finish_ids([1, 2], "stop", "</s>", stop, eos_fallback=7)
    assert ids == [1, 2, 7] and app, f"a non-int stop_reason must fall back to eos, got {ids}"
    # (c) the engine already kept it: nothing is appended
    ids, app = C.vllm_finish_ids([1, 9], "stop", 9, stop, eos_fallback=7)
    assert ids == [1, 9] and not app, f"a row already ending in a stop id must be untouched: {ids}"
    # (d) finish_reason "length": no stop token exists, nothing is appended
    ids, app = C.vllm_finish_ids([1, 2, 3], "length", None, stop, eos_fallback=7)
    assert ids == [1, 2, 3] and not app, f"a length-capped row must be returned as is, got {ids}"
    # (e) trimming cuts at the FIRST stop token and keeps it (rl/rl.py:82-90)
    ids, app = C.vllm_finish_ids([1, 7, 2, 9], "length", None, stop, eos_fallback=7)
    assert ids == [1, 7], f"trimming must cut at the first stop token and keep it, got {ids}"
    # (f) an empty stop row still gets one token, so `finished` is never a lie
    ids, app = C.vllm_finish_ids([], "stop", 9, stop, eos_fallback=7)
    assert ids == [9] and app, f"an empty stop row must still carry its stop token, got {ids}"


def check_rename_lora_keys():
    """The 27B adapter rename vLLM's SUFFIX-only validation makes necessary (rl/rl.py:192-211)."""
    keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.31.mlp.down_proj.lora_B.weight",
    ]
    m = C.rename_lora_keys(keys, "qwen36-27b")
    for k in keys:
        want = k.replace("model.layers.", "model.language_model.layers.", 1)
        assert m[k] == want, f"27B rename wrong:\n  got  {m[k]}\n  want {want}"
        assert "model.language_model.layers." in m[k] and m[k] != k
    # the 8B is a plain CausalLM: the mapping must be the identity, or vLLM serves a WRONG adapter
    assert C.rename_lora_keys(keys, "qwen3-8b") == dict(zip(keys, keys, strict=True)), (
        "the 8B needs no rename; renaming it would break the suffix lookup the other way"
    )
    # a key that already names language_model is left alone (idempotent under a second pass)
    once = C.rename_lora_keys(keys, "qwen36-27b")
    twice = C.rename_lora_keys(list(once.values()), "qwen36-27b")
    assert list(twice.values()) == list(once.values()), f"the rename must be idempotent: {twice}"
    # only the FIRST occurrence of the prefix is rewritten
    odd = ["x.model.layers.0.model.layers.1.w"]
    assert C.rename_lora_keys(odd, "qwen36-27b")[odd[0]] == "x.model.language_model.layers.0.model.layers.1.w"


def check_maemms_for():
    """`compute: false` entries are skipped by every "all MAEMMs" iteration and kept by `check`."""
    cfg = C.load_config()
    all_keys = C.maemms_for(cfg, computable_only=False)
    run_keys = C.maemms_for(cfg)
    assert set(run_keys) < set(all_keys), "config must declare at least one compute: false MAEMM"
    for k in set(all_keys) - set(run_keys):
        assert cfg["maemms"][k].get("compute") is False, f"{k} was filtered but is not compute: false"
    for k in run_keys:
        assert cfg["maemms"][k].get("compute", True), f"{k} is compute: false but survived the filter"
    per_base = C.maemms_for(cfg, "qwen3-8b", computable_only=False)
    assert per_base and all(k.startswith("qwen3-8b/") for k in per_base), per_base
    assert set(per_base) | set(C.maemms_for(cfg, "qwen36-27b", computable_only=False)) == set(all_keys)


def check_strip_repo_sink():
    """The 8B sink strip: column 0 leaves the ids AND the activations, or nothing does.

    Built against a hand-made block rather than the real file: [F, W, T] with a known sink column
    and activations that are their own position index, so a strip that moved the two apart is
    visible as a shifted value, not only as a shifted shape.
    """
    F_, W, T = 3, 4, 5
    SINKID = 151645
    body = torch.arange(2, 2 + T).repeat(F_, W, 1)  # ordinary ids, none of them the sink
    acts = torch.arange(T).float().repeat(F_, W, 1)  # activation == position

    # sink-prefixed (the 8B file): column 0 goes from BOTH
    ids = body.clone()
    ids[..., 0] = SINKID
    out_ids, out_acts, frac = C.strip_repo_sink(ids, acts, SINKID, True)
    assert frac == 1.0, frac
    assert out_ids.shape == (F_, W, T - 1) and out_acts.shape == (F_, W, T - 1), (
        f"a strip must shorten ids and acts together, got {tuple(out_ids.shape)} / {tuple(out_acts.shape)}"
    )
    assert torch.equal(out_ids, body[..., 1:]), "the wrong ids column was dropped"
    assert torch.equal(out_acts, acts[..., 1:]), (
        "the activations were not shifted with the ids: after a strip, position i of the returned "
        "acts must be the activation of position i of the returned ids"
    )

    # not sink-prefixed (the 27B file): nothing moves, and a stray sink id is tolerated
    ids = body.clone()
    ids[0, 0, 0] = SINKID
    out_ids, out_acts, frac = C.strip_repo_sink(ids, acts, SINKID, False)
    assert out_ids.shape == (F_, W, T) and torch.equal(out_acts, acts), "nothing may move here"
    assert abs(frac - 1.0 / (F_ * W)) < 1e-9, frac

    # the config's declaration is ASSERTED against the data, both ways
    ids = body.clone()
    ids[..., 0] = SINKID
    ids[1, 2, 0] = 7  # one window out of twelve is not sink-prefixed
    try:
        C.strip_repo_sink(ids, acts, SINKID, True)
    except AssertionError as e:
        assert "prepends the sink token" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("a mixed block must NOT be stripped silently")
    ids[1, 2, 0] = SINKID
    try:
        C.strip_repo_sink(ids, acts, SINKID, False)
    except AssertionError as e:
        assert "NO sink prefix" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("an all-sink block declared sink_first: false must fail")

    # shape mismatch between ids and acts is the silent-misalignment case, so it is fatal
    try:
        C.strip_repo_sink(body, acts[..., :-1], SINKID, False)
    except AssertionError as e:
        assert "SAME shape" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("ids/acts of different shapes must not be accepted")


CHECKS = [
    check_config,
    check_paths,
    check_read_hook,
    check_inject_hook,
    check_marker_norm,
    check_score_tokens,
    check_score_ids_is_score_tokens,
    check_no_norm_filter,
    check_agg,
    check_eos_and_trim,
    check_sae,
    check_io_and_outdir,
    check_sha256_of_weights,
    check_prompt_assertions,
    check_window_geometry,
    check_size_tags,
    check_quantiles_from_hist,
    check_load_corpus,
    check_outdir_cost,
    check_best_of_k_means,
    check_parse_rows_and_gen_seed,
    check_outdir_keep_existing_and_section,
    check_sha256_of_index,
    check_vllm_finish_ids,
    check_rename_lora_keys,
    check_maemms_for,
    check_strip_repo_sink,
]


def run_all():
    import time as _time

    names = []
    for fn in CHECKS:
        t0 = _time.time()
        fn()
        print(f"[smoke] ok  {fn.__name__:<26} {_time.time() - t0:5.2f}s", flush=True)
        names.append(fn.__name__)
    print(f"[smoke] {len(names)}/{len(CHECKS)} checks passed", flush=True)
    return names


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_all()
