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
import os
import re
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

    # The centred pair goes through BOTH paths here, so the equivalence check covers `cos_centred`
    # too: adding a second cosine to score_ids without adding it to this key set would let the text
    # and id paths drift on it silently, which is the one thing this check exists to prevent.
    dirs_c = torch.randn(len(texts), D)
    mu = torch.randn(D)
    a = C.score_tokens(
        model, tok, texts, dirs, read_layer, sbatch=2, device="cpu", dirs_centred=dirs_c, mu=mu
    )
    b = C.score_ids(
        model, tok, by_hand, dirs, read_layer, sbatch=2, device="cpu", dirs_centred=dirs_c, mu=mu
    )
    for k in ("cos", "norm", "keep", "ids", "cos_centred"):
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

    # --- the per-run scoring window (the NLA arm scores at 256, not the protocol's 95) ---------
    WIDE = 256
    wide = C.score_tokens(model, tok, texts, dirs, read_layer, sbatch=2, device="cpu", max_length=WIDE)
    for k in ("cos", "norm", "keep", "ids"):
        assert wide[k].shape == (len(texts), WIDE + 1), (
            f"max_length={WIDE} must give [{len(texts)}, {WIDE + 1}] arrays, got {tuple(wide[k].shape)}"
        )
    # the SHORT rows are scored identically either way: a wider window changes only how far a long
    # row is allowed to run, never what a row inside both windows scores.
    short = slice(0, 3)
    for k in ("cos", "norm", "keep", "ids"):
        lhs = torch.nan_to_num(a[k][short], nan=-12345.0)
        rhs = torch.nan_to_num(wide[k][short, : C.SCORE_WIDTH], nan=-12345.0)
        assert torch.equal(lhs, rhs), (
            f"the {WIDE}-token window changed `{k}` on rows that fit inside the 95-token one: "
            f"widening must not move a short row's score"
        )
    # ...and the LONG row is the one that moves: 200 chars -> 95 ids at the default, 200 at WIDE.
    assert int(a["keep"][3].sum()) == C.SCORE_MAX_LENGTH, (
        f"the 200-character row must be truncated to {C.SCORE_MAX_LENGTH} ids by default, kept "
        f"{int(a['keep'][3].sum())}"
    )
    assert int(wide["keep"][3].sum()) == 200, (
        f"at max_length={WIDE} the 200-character row must keep all 200 ids, kept "
        f"{int(wide['keep'][3].sum())} -- the window is not actually being widened"
    )
    wide_ids = C.encode_for_score(tok, texts, WIDE)
    assert [len(x) for x in wide_ids] == [5, 2, 1, 200], (
        f"encode_for_score at max_length={WIDE} truncated somewhere it should not: "
        f"{[len(x) for x in wide_ids]}"
    )
    # score_ids' over-length guard moves with the window: 96 ids is refused at the default and
    # accepted at WIDE, which is what makes the two arms' guards the SAME guard.
    C.score_ids(
        model, tok, [[7] * (C.SCORE_MAX_LENGTH + 1)], dirs[:1], read_layer, device="cpu", max_length=WIDE
    )

    # score_width_of reads the width back out of a scores/ directory rather than the constant.
    with tempfile.TemporaryDirectory() as td:
        plain, widened = Path(td) / "plain", Path(td) / "wide"
        rows_of = ((plain, {"rows": [0], "n": 1}), (widened, {"rows": [0], "n": 1, "score_max_length": WIDE}))
        for d_, obj in rows_of:
            d_.mkdir()
            (d_ / "rows.json").write_text(json.dumps(obj))
        assert C.score_width_of(plain) == C.SCORE_WIDTH, (
            "a rows.json with no score_max_length is the protocol width"
        )
        assert C.score_width_of(widened) == WIDE + 1, "score_width_of must return score_max_length + 1"


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


def check_bo_ladder():
    """The per_target bo_<k> aggregation: the UNBIASED order statistic over all n rollouts.

    The reference is a Monte-Carlo estimate of E[max of k draws without replacement] from the same
    eight values, not a second call of the formula: an estimator checked only against its own
    algebra is checked against nothing. It is also asserted to DIFFER from the disjoint-group mean
    this replaced on 2026-09-23, since that is the regression the swap exists to make visible.
    """
    import itertools
    import math

    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    got = C.bo_ladder(vals, (1, 2, 4, 8, 16))
    assert got[1] == 3.5, f"bo1 must be the plain mean 3.5, got {got[1]}"
    assert got[8] == 7.0, f"bo8 = bo_n must be the single max 7.0, got {got[8]}"
    assert 16 not in got, (
        "k > n must be SKIPPED, not clamped: a summary may not claim a bo it could not compute"
    )
    # EXHAUSTIVE reference: the mean of max() over every k-subset of the eight, enumerated.
    for k in (2, 4):
        want = sum(max(c) for c in itertools.combinations(vals, k)) / math.comb(8, k)
        assert abs(got[k] - want) < 1e-12, (
            f"bo{k} = {got[k]} but the mean max over all {math.comb(8, k)} {k}-subsets is {want}"
        )
    # it is NOT the disjoint-group mean, which is what `best_of_k_means` returned until 2026-09-23
    assert abs(got[2] - (1 + 3 + 5 + 7) / 4) > 1e-3, "bo2 is still the disjoint-group mean"
    assert abs(got[4] - (3 + 7) / 2) > 1e-3, "bo4 is still the disjoint-group mean"
    # ORDER-FREE, unlike the disjoint-group estimator: the same multiset scores the same
    assert C.bo_ladder([7.0, 0.0, 6.0, 1.0], (2,))[2] == C.bo_ladder([0.0, 1.0, 6.0, 7.0], (2,))[2]
    # every draw is used: three values, k = 2 -> (9 + 9 + 5)/3, no remainder dropped
    assert abs(C.bo_ladder([0.0, 9.0, 5.0], (2,))[2] - (9 + 9 + 5) / 3) < 1e-12
    try:
        C.bo_ladder(vals, (0,))
    except AssertionError as e:
        assert "k >= 1" in str(e), f"wrong assert fired: {e}"
    else:
        raise AssertionError("bo_ladder accepted k = 0")


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


def _writer_args():
    return {"argv": ["x"], "repo_commit": "abc", "gpu": "CPU", "usd_per_s": 0.0, "force": False}


def _legacy_outdir_write(path, name, rows, barrier):
    """The PRE-2026-09-23 accumulating write, verbatim, for the mutation half of the two-writer
    check: copytree the whole product directory into `<name>.tmp-<date>`, write there, then rmtree
    the original and rename the temp over it (common.py:2689-2700, :2793-2795 at 6cdd429)."""
    import shutil
    import time as _t

    tmp = path.with_name(f"{path.name}.tmp-{_t.strftime('%Y-%m-%d')}")
    index = {}
    if path.exists():
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(path, tmp)
        if (tmp / "index.json").exists():
            index.update(json.loads((tmp / "index.json").read_text()))
        for stale in ("README.md", "index.json"):
            (tmp / stale).unlink(missing_ok=True)
            index.pop(stale, None)
    else:
        tmp.mkdir(parents=True)
    barrier.wait(timeout=60)
    C.write_jsonl(tmp / name, rows)
    index[name] = {"kind": "jsonl", "rows": len(rows)}
    barrier.wait(timeout=60)
    (tmp / "index.json").write_text(json.dumps(index, indent=1))
    (tmp / "README.md").write_text("# legacy\n")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def _two_writer_child(path, name, rows, barrier, legacy):
    """One of the two concurrent writers. Both barriers are inside the open product directory, so
    the two runs' enter / write / commit phases are forced to interleave."""
    path = Path(path)
    if legacy:
        _legacy_outdir_write(path, name, rows, barrier)
        return
    with C.outdir(path, _writer_args(), keep_existing=True) as od:
        barrier.wait(timeout=60)
        od.write_jsonl(name, rows)
        barrier.wait(timeout=60)


def _run_two_writers(td, legacy):
    """(files present, index keys, child exit codes) after two concurrent writers of one product."""
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    d = Path(td) / "rollouts"
    with C.outdir(d, _writer_args()) as od:  # the product already holds one set
        od.write_jsonl("set0.jsonl", [{"z": 0}])
    barrier = ctx.Barrier(2)
    procs = [
        ctx.Process(target=_two_writer_child, args=(str(d), f"set{i}.jsonl", [{"i": i}], barrier, legacy))
        for i in (1, 2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    codes = [p.exitcode for p in procs]
    files = sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []
    idx = sorted(json.loads((d / "index.json").read_text())) if (d / "index.json").exists() else []
    return files, idx, codes


def check_two_writers_into_one_product():
    """TWO CONCURRENT WRITERS of one accumulating product both survive -- and did not before.

    `SMOKES.md:4349-4356`: two `rollouts_vllm` runs of one MAEMM shared `rollouts.tmp-<date>` and
    the later rename silently discarded the earlier file. The additive write (common.OutDir) moves
    only this run's own files in, so disjoint writers cannot touch each other.

    The mutation half runs the SAME scenario through the pre-2026-09-23 copytree/rmtree/rename
    code (`_legacy_outdir_write`) and asserts it does NOT get both files into the index: a gate
    that has never been red is not a gate.
    """
    with tempfile.TemporaryDirectory() as td:
        files, idx, codes = _run_two_writers(td, legacy=False)
        assert codes == [0, 0], f"a writer failed: exitcodes {codes}"
        assert files == ["README.md", "index.json", "set0.jsonl", "set1.jsonl", "set2.jsonl"], files
        assert idx == ["set0.jsonl", "set1.jsonl", "set2.jsonl"], (
            f"the index must list the set already there and BOTH new ones, got {idx}"
        )
    print("  running the MUTATION (the legacy copytree write); the child traceback below "
          "is the defect being demonstrated, not a failure of this check", flush=True)
    with tempfile.TemporaryDirectory() as td:
        files, idx, codes = _run_two_writers(td, legacy=True)
        assert codes != [0, 0] or idx != ["set0.jsonl", "set1.jsonl", "set2.jsonl"], (
            f"the legacy copytree write kept both writers (files {files}, index {idx}, "
            f"exitcodes {codes}): the mutation did not apply, so this check proves nothing"
        )
        print(f"  mutation (legacy copytree write): exitcodes {codes}, index {idx}", flush=True)


def check_additive_removes_nothing_and_the_legacy_path_is_the_hazard():
    """The additive commit REMOVES NOTHING -- and an OLD writer beside it is still unsafe.

    Two separate claims, and running them together is the point.

    (a) STRUCTURAL, by ast: nothing on `OutDir._commit_additive`'s path calls `rmtree` or
        `copytree`, and `__exit__` reaches no `rmtree` at all. That is the whole guarantee the
        concurrent-writer fix rests on, and it is asserted rather than read off the diff.

    (b) EMPIRICAL, and it is a WARNING not a guarantee: a pre-2026-09-23 writer overlapping an
        additive one on the same directory can still lose its own file or die, because the hazard
        is ITS `rmtree(path)` + `rename(tmp, path)`, which the additive side cannot make safe from
        outside. Measured here: the legacy writer ends up with exitcode 1 (its rename hits a
        directory the additive writer has repopulated) and the directory is left holding neither
        writer's rows. THE OPERATIONAL RULE THAT FOLLOWS: while any job is still running on
        `6cdd429`, no new-code job may write the same MAEMM's `rollouts/`. Check for a
        `rollouts.tmp-*` sibling before launching. Once every writer is on this code, concurrent
        writers are safe -- `check_two_writers_into_one_product` is that case.
    """
    import ast
    import multiprocessing as mp

    src = (Path(__file__).resolve().parent / "common.py").read_text()
    tree = ast.parse(src)
    outdir = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "OutDir")
    for meth in ("_commit_additive", "__exit__"):
        fn = next(n for n in outdir.body if isinstance(n, ast.FunctionDef) and n.name == meth)
        calls = sorted({
            n.func.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("rmtree", "copytree")
        })
        assert not calls, (
            f"OutDir.{meth} calls {calls}: the additive commit removes nothing and copies nothing, "
            f"which is the entire reason two writers of one product can no longer destroy each "
            f"other (SMOKES.md:4349-4356)"
        )

    ctx = mp.get_context("fork")
    outcomes = []
    for legacy_first in (True, False):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "rollouts"
            with C.outdir(d, _writer_args()) as od:
                od.write_jsonl("set0.jsonl", [{"z": 0}])
            barrier = ctx.Barrier(2)
            order = [True, False] if legacy_first else [False, True]
            procs = [
                ctx.Process(target=_two_writer_child,
                            args=(str(d), f"set{i + 1}.jsonl", [{"i": i}], barrier, legacy))
                for i, legacy in enumerate(order)
            ]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join(120)
            files = sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []
            outcomes.append((legacy_first, [pr.exitcode for pr in procs], files))
    lost = [o for o in outcomes if o[1] != [0, 0] or len([f for f in o[2] if f.startswith("set")]) < 3]
    assert lost, (
        "a legacy writer beside an additive one came through intact in BOTH orders; either the "
        "legacy reimplementation here no longer reproduces the pre-2026-09-23 write, or the "
        "hazard is gone and this check (and the launch rule in its docstring) should be retired"
    )
    print(f"  mixed old/new overlap is UNSAFE, as expected: {outcomes}", flush=True)


def check_rollout_chunk_stem_and_read():
    """`--rows` chunks of ONE product under ONE run tag read back as ONE product.

    `rollout_chunk_stem` leaves a whole-set run at its historical path (byte-compatible with every
    product on the volume) and suffixes a chunk; `read_rollouts` concatenates the chunks, refuses
    a directory holding both shapes, and refuses chunks that overlap or disagree.
    """
    assert C.rollout_chunk_stem("s__vllm", "") == "s__vllm"
    assert C.rollout_chunk_stem("s__vllm", "0-7") == "s__vllm__rows0-7"
    assert C.rollout_chunk_stem("s__vllm", "3,5,9-11") == "s__vllm__rows3_5_9-11"
    try:
        C.rollout_chunk_stem("s__vllm", "0-7 ; rm")
    except AssertionError as e:
        assert "row spec" in str(e), e
    else:
        raise AssertionError("rollout_chunk_stem accepted a non-spec")

    def summ(rows, **kw):
        return {"engine": "vllm", "n": 2, "max_new": 8, "rows": rows, "set": "s", **kw}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name, rows in (("s__vllm__rows0-1", [0, 1]), ("s__vllm__rows2-3", [2, 3])):
            C.write_jsonl(d / f"{name}.jsonl", [{"row": r, "k": 0} for r in rows])
            (d / f"{name}.summary.json").write_text(json.dumps(summ(rows)))
        recs, rsum, src = C.read_rollouts(str(d), "s__vllm")
        assert [r["row"] for r in recs] == [0, 1, 2, 3], recs
        assert rsum["rows"] == [0, 1, 2, 3] and rsum["n_targets"] == 4, rsum
        assert rsum["chunks"] == ["s__vllm__rows0-1.jsonl", "s__vllm__rows2-3.jsonl"], rsum
        assert len(src) == 2
        # a whole-set file beside chunks of the same stem is a refusal, not a preference
        C.write_jsonl(d / "s__vllm.jsonl", [{"row": 0, "k": 0}])
        (d / "s__vllm.summary.json").write_text(json.dumps(summ([0])))
        try:
            C.read_rollouts(str(d), "s__vllm")
        except AssertionError as e:
            assert "claiming one product" in str(e), e
        else:
            raise AssertionError("read_rollouts preferred one shape over the other")
        (d / "s__vllm.jsonl").unlink()
        (d / "s__vllm.summary.json").unlink()
        # overlapping chunks
        (d / "s__vllm__rows2-3.summary.json").write_text(json.dumps(summ([1, 2, 3])))
        try:
            C.read_rollouts(str(d), "s__vllm")
        except AssertionError as e:
            assert "DISJOINT" in str(e), e
        else:
            raise AssertionError("read_rollouts merged overlapping chunks")
        # chunks that are two experiments
        (d / "s__vllm__rows2-3.summary.json").write_text(json.dumps(summ([2, 3], n=4)))
        try:
            C.read_rollouts(str(d), "s__vllm")
        except AssertionError as e:
            assert "one experiment" in str(e), e
        else:
            raise AssertionError("read_rollouts merged chunks that disagree on n")


def check_nla_min_new_override():
    """`nla.min_new` overrides the shared `rollouts:` block, and rollouts_nla reads it there.

    The verbalizer stops on its own well before the shared 16, which pads short <explanation>
    answers with continuation the checkpoint would not have produced (Tomas 2026-09-22). The
    shared block is never edited -- that would re-point every rollout product -- so the key is
    per-MAEMM with the shared value as the fallback.
    """
    cfg = C.load_config()
    nla_keys = [k for k, v in cfg["maemms"].items() if v.get("type") == "nla"]
    assert nla_keys, "no `type: nla` entry in config.yaml"
    for k in nla_keys:
        assert cfg["maemms"][k]["nla"].get("min_new") == 0, (
            f"{k}: nla.min_new must be 0 (config.yaml), got {cfg['maemms'][k]['nla'].get('min_new')!r}"
        )
    assert int(cfg["rollouts"]["min_new"]) == 16, (
        "the SHARED rollouts.min_new moved; it is never edited (every other arm reads it)"
    )
    src = (Path(__file__).resolve().parent / "rollouts_nla.py").read_text()
    assert '"min_new": nla["min_new"] if "min_new" in nla else rl["min_new"]' in src, (
        "rollouts_nla's sampling dict must take min_new from the `nla:` block with the shared "
        "`rollouts:` value as the fallback, or the config key above is inert"
    )
    # the fallback itself: an entry without the key still generates under the shared value
    spec = dict(cfg["maemms"][nla_keys[0]])
    spec["nla"] = {k: v for k, v in spec["nla"].items() if k != "min_new"}
    C._check_nla(nla_keys[0], spec, cfg["rollouts"])  # must not raise
    spec["nla"] = {**spec["nla"], "min_new": -1}
    try:
        C._check_nla(nla_keys[0], spec, cfg["rollouts"])
    except AssertionError as e:
        assert "non-negative int" in str(e), e
    else:
        raise AssertionError("_check_nla accepted a negative nla.min_new")


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


def check_sae_key_for():
    """WHICH SAE of a base: explicit wins, unknown is refused, and two SAEs refuse to be guessed.

    The last one is the whole point. `qwen36-27b` carries two since `sae2m`, and every call site
    used to write `keys[0]` or `assert len(keys) == 1` by hand -- one silently picking whichever
    came first in config.yaml, the others simply unable to run. A silent fallback here is
    invisible in every product's output, so it is asserted directly rather than through a product.
    """
    cfg = C.load_config()
    two = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == "qwen36-27b"]
    one = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == "qwen3-8b"]
    assert len(two) == 2 and len(one) == 1, (
        f"this check needs a two-SAE base and a one-SAE base; config has {two} and {one}"
    )
    assert C.sae_key_for(cfg, "qwen3-8b") == one[0], "a single-SAE base needs no --sae"
    for k in two:
        assert C.sae_key_for(cfg, "qwen36-27b", k) == k, "an explicit --sae must be honoured"
    try:
        C.sae_key_for(cfg, "qwen36-27b")
    except AssertionError as e:
        assert "pass --sae" in str(e), f"wrong assert fired for an ambiguous base: {e}"
    else:
        raise AssertionError("sae_key_for GUESSED between two SAEs instead of refusing")
    for bad in ("qwen36-27b/nope", one[0]):
        try:
            C.sae_key_for(cfg, "qwen36-27b", bad)
        except AssertionError as e:
            assert "is not one of" in str(e), f"wrong assert fired for --sae {bad!r}: {e}"
        else:
            raise AssertionError(f"sae_key_for accepted --sae {bad!r} on qwen36-27b")


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


def check_rollouts_nla_selftest():
    """`rollouts_nla`'s own pure-piece checks, so `--product unit` covers them too.

    They live in that module rather than here because they are about ITS recipe (the amp solve,
    the sidecar's marker contract, the explanation tags), not about common.py -- and because
    `uv run precompute/rollouts_nla.py --selftest` has to work on numpy alone, with no torch in
    the call path. This wrapper is the one place the two entry points meet.
    """
    from precompute import rollouts_nla

    names = rollouts_nla.selftest()
    assert len(names) == len(rollouts_nla.SELFTESTS), f"only {len(names)} nla selftests ran"



# ---------------------------------------------------------------------------------------------
# the 2026-09-21 conventions layer: a mu is a file, a set states its storage, a dictionary is
# named per row. Each check builds its own fixture on disk and compares against a numpy reference.
# ---------------------------------------------------------------------------------------------


def _conv_cfg(d=D):
    """A minimal cfg for the storage/centring helpers: they read bases[b].d and family_kinds only."""
    return {
        "bases": {"tb": {"d": d}},
        "family_kinds": {
            "realact": {"centrable": True, "kind": "activation"},
            "random": {"centrable": False, "kind": "synthetic"},
            "sae": {"centrable": False, "kind": "dictionary"},
        },
        "heldout": {},
        "modal": {"archive": "/nonexistent-archive"},
    }


def _write_set(dirpath: Path, rows, act, storage: dict, vecs=None):
    """A held-out set on disk: ids.jsonl, act.f32 (raw sets), vecs.f16 and storage.json."""
    import numpy as np

    dirpath.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(dirpath / "ids.jsonl", rows)
    if act is not None:
        C.write_array(dirpath / "act.f32", act, "float32")
    if vecs is None:
        vecs = act / np.maximum(np.linalg.norm(act, axis=1, keepdims=True), 1e-12)
    C.write_array(dirpath / "vecs.f16", vecs, "float16")
    with open(dirpath / C.STORAGE_FILE, "w") as fh:
        json.dump(storage, fh)
    return vecs


def check_storage_contract():
    """`dirs_for` on a RAW set: unit(act) at mu=none, unit(act - mu) where the family is centrable.

    The reference is numpy, computed from the same act.f32 but with the mean subtracted by hand,
    and the non-centrable rows are asserted IDENTICAL under both means -- which is the property
    that makes `centering: none` a no-op on an encoder column rather than a special case at seven
    call sites.
    """
    import numpy as np

    cfg = _conv_cfg()
    rng = np.random.default_rng(0)
    fams = ["realact", "realact", "random", "sae"]
    act = rng.normal(size=(4, D)).astype(np.float32) * 30.0
    rows = [{"row": i, "family": f, "id": i} for i, f in enumerate(fams)]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sdir = root / "base" / "tb" / "heldout" / "s1"
        _write_set(sdir, rows, act, {"storage": "raw"})
        mu = rng.normal(size=(D,)).astype(np.float32) * 3.0
        mupath = root / "mu.f32"
        C.write_array(mupath, mu, "float32")

        got_raw = C.dirs_for(cfg, "tb", str(sdir), None, str(root))
        want_raw = act / np.linalg.norm(act, axis=1, keepdims=True)
        assert np.abs(got_raw - want_raw).max() < 1e-6, (
            f"dirs_for at mu=none is not unit(act): max |d| {np.abs(got_raw - want_raw).max():.2e}"
        )

        got_c = C.dirs_for(cfg, "tb", str(sdir), str(mupath), str(root))
        cen = act - mu[None, :]
        want_c = cen / np.linalg.norm(cen, axis=1, keepdims=True)
        for i, fam in enumerate(fams):
            want = want_c[i] if fam == "realact" else want_raw[i]
            assert np.abs(got_c[i] - want).max() < 1e-6, (
                f"row {i} ({fam}) under mu: max |d| {np.abs(got_c[i] - want).max():.2e} -- a "
                f"{'centrable' if fam == 'realact' else 'non-centrable'} row was treated as the other"
            )
        assert np.array_equal(got_c[2:], got_raw[2:]), (
            "the non-centrable rows moved between mu=none and mu=<file>; nothing may be subtracted "
            "from an encoder column or a Gaussian draw"
        )
        # A raw set with no act.f32 is a broken set, not a set to guess about.
        (sdir / "act.f32").unlink()
        try:
            C.dirs_for(cfg, "tb", str(sdir), None, str(root))
        except AssertionError as e:
            assert "has no act.f32" in str(e), f"wrong assert for a raw set with no act.f32: {e}"
        else:
            raise AssertionError("dirs_for served a `storage: raw` set that has no act.f32")


def check_no_mean_reaches_a_row_without_a_raw_activation():
    """A mean is applied to exactly the `centrable` rows -- asserted, not merely performed.

    The set this is for is `2026-09-21_v3_ctrl`, which mixes 512 `random` Gaussian draws with the
    512 131k encoder columns in ONE `storage: raw` directory: its `act.f32` holds rows that are
    not activations at all. cos(h - mu, encoder column) is a one-sided number wearing a centred
    number's name, and `family_kinds:` is the only thing standing between the two.

    Both halves here: the guarantee holds on a ctrl-shaped set, and the assertion inside
    `dirs_for` FIRES when the family table is mutated to call a dictionary column centrable --
    a gate that has never been red is unevaluated.
    """
    import numpy as np

    cfg = _conv_cfg()
    rng = np.random.default_rng(23)
    fams = ["realact", "random", "sae", "random", "sae"]
    act = rng.normal(size=(len(fams), D)).astype(np.float32) * 30.0
    rows = [{"row": i, "family": f, "id": i} for i, f in enumerate(fams)]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sdir = root / "v3_ctrl_shaped"
        _write_set(sdir, rows, act, {"storage": "raw"})
        mupath = root / "mu.f32"
        C.write_array(mupath, rng.normal(size=(D,)).astype(np.float32) * 3.0, "float32")

        got = C.dirs_for(cfg, "tb", str(sdir), str(mupath), str(root))
        raw = C.dirs_for(cfg, "tb", str(sdir), None, str(root))
        for i, fam in enumerate(fams):
            moved = float(np.abs(got[i] - raw[i]).max())
            if fam == "realact":
                assert moved > 1e-3, f"row {i} ({fam}) did not move under a mean"
            else:
                assert moved == 0.0, (
                    f"row {i} ({fam}) MOVED under a mean ({moved:.2e}); it has no raw activation "
                    f"to centre and cos(h - mu, v) against it is one-sided"
                )

        # THE MUTATION: call the dictionary family centrable and the assertion must fire.
        bent = {**cfg, "family_kinds": {**cfg["family_kinds"],
                                        "sae": {"centrable": True, "kind": "dictionary"}}}
        moved_now = C.dirs_for(bent, "tb", str(sdir), str(mupath), str(root))
        assert np.abs(moved_now[2] - raw[2]).max() > 1e-3, (
            "the mutation did not apply: the sae row still did not move, so the check below "
            "would pass for the wrong reason"
        )
        # ... and with the table honest again, the post-condition catches a hand-bent array.
        real_centrable = C.family_centrable
        try:
            C.family_centrable = lambda cfg_, f: True  # the subtraction reaches every row
            try:
                C.dirs_for(cfg, "tb", str(sdir), str(mupath), str(root))
            except AssertionError as e:
                assert "not `centrable`" in str(e), f"wrong assert fired: {e}"
            else:
                raise AssertionError(
                    "dirs_for subtracted a mean from a non-centrable row and said nothing"
                )
        finally:
            C.family_centrable = real_centrable


def check_unit_set_is_served_as_shipped():
    """A stored `unit` set comes back UNCHANGED under every mu, and states that it has no mean.

    Until 2026-09-23 this branch carried a `mu_stored` / `family_mu` contract: the set named the
    mean each family was built under, `dirs_for` asserted the run's `mu` matched it, and an
    `unknown` mean was returned with a label. All of it is deleted (M0a). A stored unit direction
    cannot be moved to another mean without ||act||, so there was never anything the contract
    could DO except refuse -- and the refusal made a legacy set unreadable rather than readable at
    the one thing it is good for, the uncentred cosine. The replacement is: serve the rows as
    shipped, and let `score` write NaN for the centred cosine of such a set.

    Checked both ways round, because "unchanged under every mu" is the whole claim: two different
    means, and the declared keys refused at config load so the dead contract cannot come back.
    """
    import numpy as np

    cfg = _conv_cfg()
    rng = np.random.default_rng(1)
    rows = [{"row": 0, "family": "realact", "id": 0}, {"row": 1, "family": "sae", "id": 1}]
    v = rng.normal(size=(2, D)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mu_a, mu_b = root / "a.f32", root / "b.f32"
        for path in (mu_a, mu_b):
            C.write_array(path, rng.normal(size=(D,)).astype(np.float32), "float32")

        sdir = root / "unitset"
        _write_set(sdir, rows, None, {"storage": "unit"}, vecs=v)
        for mu in (None, str(mu_a), str(mu_b)):
            notes: list[str] = []
            got = C.dirs_for(cfg, "tb", str(sdir), mu, str(root), notes)
            assert np.abs(got - v).max() < 1e-3, (
                f"a `storage: unit` set moved under mu={mu!r}; its rows are the producer's and "
                f"nothing here may re-centre them"
            )
            assert any("no centred number" in n for n in notes), (
                f"the product README must SAY that this set has no centred reading, got {notes}"
            )

        # the deleted contract cannot be reintroduced through config without the loader saying so
        bad = {"storage": "unit", "families": {"realact": {"n": 1}}, "mu_stored": None}
        try:
            C._check_heldout_storage(cfg, "s1", bad)
        except AssertionError as e:
            assert "deleted on 2026-09-23" in str(e), f"wrong assert for a revived mu_stored: {e}"
        else:
            raise AssertionError("_check_heldout_storage accepted a `mu_stored:` key")

        # An OLD storage.json on the volume still carries the dead fields; they are IGNORED, not
        # a refusal -- no set on the volume is re-drawn for this change.
        legacy = root / "legacy"
        _write_set(legacy, rows, None,
                   {"storage": "unit", "mu_stored": str(mu_a), "family_mu": {"realact": "x"}},
                   vecs=v)
        got = C.set_storage(cfg, str(legacy), str(root))
        assert sorted(got) == ["source", "storage"] and got["storage"] == "unit", got

        # A directory with neither a storage.json nor a config entry has no stated contract.
        bare = root / "bare"
        bare.mkdir()
        C.write_jsonl(bare / "ids.jsonl", rows)
        try:
            C.set_storage(cfg, str(bare), str(root))
        except AssertionError as e:
            assert "carries no storage.json" in str(e), f"wrong assert for an undeclared dir: {e}"
        else:
            raise AssertionError("set_storage invented a contract for a directory that states none")


def check_two_cosines():
    """`score_ids` with both tensors: `cos` unchanged, `cos_centred` equal to a direct einsum.

    Bit-identity of `cos` is the load-bearing half -- gcg.py and sae_self.py call without the
    keywords and their numbers must not move by an ulp -- and the centred half is checked against
    hidden states read out of the model independently, not against the same code path.
    """
    model, tok = tiny_model(), FakeTok()
    read_layer = 2
    texts = ["abcde", "xy"]
    dirs, dirs_c = torch.randn(2, D), torch.randn(2, D)
    mu = torch.randn(D) * 0.3

    plain = C.score_tokens(model, tok, texts, dirs, read_layer, sbatch=1, device="cpu")
    both = C.score_tokens(
        model, tok, texts, dirs, read_layer, sbatch=1, device="cpu", dirs_centred=dirs_c, mu=mu
    )
    assert "cos_centred" not in plain, "a caller that asked for one cosine got two"
    for k in ("cos", "norm", "keep", "ids"):
        same = torch.equal(
            torch.nan_to_num(plain[k], nan=-12345.0), torch.nan_to_num(both[k], nan=-12345.0)
        )
        assert same, f"asking for cos_centred moved `{k}`; the first cosine must be bit-identical"

    ids0 = torch.tensor([[SINK, *tok.ids_of("abcde")]])
    with torch.no_grad():
        hs = (
            model(input_ids=ids0, attention_mask=torch.ones_like(ids0), output_hidden_states=True)
            .hidden_states[read_layer + 1]
            .float()
        )
    want = torch.einsum(
        "btd,bd->bt",
        torch.nn.functional.normalize(hs - mu, dim=-1),
        torch.nn.functional.normalize(dirs_c[0:1], dim=-1),
    )[0]
    got = both["cos_centred"][0, 1:6]
    assert torch.allclose(got, want[1:6], atol=1e-5), (
        f"cos_centred differs from the direct einsum by {(got - want[1:6]).abs().max():.2e}"
    )
    # A NaN direction row (a non-centrable family) must come back NaN, never a one-sided number.
    dirs_nan = dirs_c.clone()
    dirs_nan[1] = float("nan")
    out = C.score_tokens(
        model, tok, texts, dirs, read_layer, sbatch=1, device="cpu", dirs_centred=dirs_nan, mu=mu
    )
    assert torch.isnan(out["cos_centred"][1][out["keep"][1]]).all(), (
        "a NaN row of dirs_centred produced a number; a non-centrable family must stay NaN"
    )
    # The two keywords are a pair.
    try:
        C.score_ids(model, tok, [[3, 4]], dirs[:1], read_layer, device="cpu", mu=mu)
    except AssertionError as e:
        assert "TOGETHER or neither" in str(e), f"wrong assert for mu without dirs_centred: {e}"
    else:
        raise AssertionError("score_ids accepted mu with no dirs_centred")


def check_sae_key_selector():
    """A two-dictionary ids.jsonl selects only the asked-for `sae_key`, and only encoder rows.

    Selecting on the family label alone is not a near miss: every feature id below 131,072 is a
    valid index into a 2^21 encoder, so the wrong rows would be scored silently. The unkeyed case
    is asserted to be a no-op, because every set drawn before the field existed relies on that.
    """
    rows = [
        {"row": 0, "family": "sae", "sae_key": "b/two_m", "id": 10, "sae_side": "enc"},
        {"row": 1, "family": "sae", "sae_key": "b/two_m", "id": 11, "sae_side": "dec"},
        {"row": 2, "family": "sae", "sae_key": "b/one31k", "id": 10},
        {"row": 3, "family": "realact", "id": 99},
        # the legacy family label, but KEYED -- the unkeyed case is its own block below
        {"row": 4, "family": "sae2m_enc", "sae_key": "b/two_m", "id": 12},
    ]
    got = [r["row"] for r in C.sae_rows_of(rows, "b/two_m")]
    assert got == [0, 1, 4], f"sae_key filter picked {got}; the 131k row must not be in it"
    got = [r["row"] for r in C.sae_rows_of(rows, "b/one31k")]
    assert got == [2], f"sae_key filter picked {got} for the 131k dictionary"
    got = [r["row"] for r in C.sae_rows_of(rows, "b/two_m", side="enc")]
    assert got == [0, 4], f"the side filter picked {got}; row 1 is a decoder row"

    # UNKEYED ROWS -- the case that made the first version of this guard vacuous. Both production
    # sets carry `sae_key` on no row, and `r.get("sae_key", sae_key)` defaulted each of them to
    # match whatever was typed, so `--sae <the 2M> --set 2026-09-16_v1` selected all 512 of the
    # 131k rows and every id was a valid 2^21 index. Three outcomes now, and the middle one is the
    # only one that returns rows.
    unkeyed = [{"row": 0, "family": "sae", "id": 1}, {"row": 1, "family": "sae", "id": 2}]
    try:
        C.sae_rows_of(unkeyed, "b/two_m")
    except AssertionError as e:
        assert "declares no dictionary" in str(e) and "2 SAE rows" in str(e), (
            f"wrong assert for an undeclared unkeyed set: {e}"
        )
    else:
        raise AssertionError("sae_rows_of selected unkeyed rows with nothing declaring them")
    try:
        C.sae_rows_of(unkeyed, "b/two_m", declared="b/one31k")
    except AssertionError as e:
        assert "but this run asked for" in str(e), f"wrong assert for a declared mismatch: {e}"
    else:
        raise AssertionError("sae_rows_of selected unkeyed rows of the WRONG declared dictionary")
    assert [r["row"] for r in C.sae_rows_of(unkeyed, "b/two_m", declared="b/two_m")] == [0, 1], (
        "a set that DECLARES the dictionary asked for must still yield its unkeyed rows"
    )
    # Mixed: keyed rows are judged by their own key, unkeyed ones by the declaration, and the
    # result stays in row order.
    mixed = unkeyed + [{"row": 2, "family": "sae", "sae_key": "b/one31k", "id": 3}]
    assert [r["row"] for r in C.sae_rows_of(mixed, "b/two_m", declared="b/two_m")] == [0, 1]

    # The declaration itself: storage.json wins, then the `heldout:` entry, then None.
    cfg = C.load_config()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "2026-09-16_v1"
        d.mkdir()
        assert C.declared_sae_key(cfg, str(d)) == "qwen36-27b/l42-1b", (
            "config.yaml must declare which dictionary 2026-09-16_v1's sae ids index"
        )
        with open(d / C.STORAGE_FILE, "w") as fh:
            json.dump({"storage": "raw", "sae_key": "qwen36-27b/sae2m"}, fh)
        assert C.declared_sae_key(cfg, str(d)) == "qwen36-27b/sae2m", "storage.json must win"
        bare = Path(td) / "nowhere"
        bare.mkdir()
        assert C.declared_sae_key(cfg, str(bare)) is None


def check_family_kinds_table():
    """Every family of every configured held-out set has a `family_kinds:` entry, and the two
    `mus`-free invariants hold: `centrable` iff `kind: activation`, and no mu value is a bare name.

    load_config asserts these at load; this check states them over the REAL config so a new set or
    family added without its declaration fails here rather than at the first GPU call.
    """
    cfg = C.load_config()
    for set_name, spec in cfg["heldout"].items():
        for fam in spec["families"]:
            assert fam in cfg["family_kinds"], f"heldout {set_name} family {fam} has no family_kinds"
    for fam, fspec in cfg["family_kinds"].items():
        assert fspec["centrable"] == (fspec["kind"] == "activation"), fam
    # THE SCORING CONSTANT is a file path on every base, since every centred number in the
    # pipeline is taken about it and a base without one can report no centred cosine at all.
    for b in cfg["bases"]:
        mu = C.score_mu(cfg, b)
        assert str(mu).endswith(C.MU_SUFFIXES), (
            f"bases[{b}].whiten_mu = {mu!r} is not a {'/'.join(C.MU_SUFFIXES)} path -- a mu is a "
            f"FILE, never a name"
        )
    for key, spec in cfg["maemms"].items():
        if "mu" in spec:
            # Three legal states: null, a file path, or `unknown` -- "considered, not established",
            # which every run must override with --mu (mu_for refuses to pick one).
            assert (
                spec["mu"] is None
                or spec["mu"] == C.MU_UNKNOWN
                or str(spec["mu"]).endswith(C.MU_SUFFIXES)
            ), f"maemms[{key}].mu = {spec['mu']!r} is not null, {C.MU_UNKNOWN!r} or a file path"
    # Resolution: `{base}` expands, a relative path takes --root, an absolute one does not.
    assert C.resolve_mu_path("base/{base}/stats/mu.f32", "qq", "/r") == "/r/base/qq/stats/mu.f32"
    assert C.resolve_mu_path("/abs/mu.npy", "qq", "/r") == "/abs/mu.npy"
    for bad in ("stats_mu", "base/x/mu.bin", ""):
        try:
            C._check_mu_value(bad, "test", allow_unknown=False)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"_check_mu_value accepted {bad!r}, which is not a mu file")


def check_exact_solve_roundtrip():
    """`rollouts_nla.build_inputs(amp="exact")` recovers the RAW activation from a centred unit
    direction plus its stored `act_norm` -- the migration tool for an imported `storage: unit` set.

    Reference is the activation it was built from, not the solve restated: given act and mu, the
    row carries u = unit(act - mu) and act_norm = ||act||, and the solve must return act itself.
    """
    import numpy as np

    from precompute import rollouts_nla

    rng = np.random.default_rng(7)
    d = 64
    mu = rng.normal(size=(d,)).astype(np.float64) * 2.0
    act = rng.normal(size=(3, d)).astype(np.float64) * 40.0
    u = (act - mu) / np.linalg.norm(act - mu, axis=1, keepdims=True)
    rows = [{"act_norm": round(float(np.linalg.norm(a)), 3)} for a in act]
    x, info = rollouts_nla.build_inputs(u.astype(np.float32), rows, mu.astype(np.float32), "exact", 1.0)
    cos = np.einsum("nd,nd->n", x, act) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(act, axis=1)
    )
    assert cos.min() > 1 - 1e-4, f"the exact solve did not recover act: min cos {cos.min():.6f}"
    nrm = np.linalg.norm(x, axis=1)
    assert np.abs(nrm - np.linalg.norm(act, axis=1)).max() < 1e-2, (
        f"||x|| != the stored act_norm: max |d| {np.abs(nrm - np.linalg.norm(act, axis=1)).max():.2e}"
    )
    assert all(r["amp_used"] == "exact" and r["fallback"] is None for r in info), (
        f"a row fell back instead of solving: {info}"
    )
    # A row with no act_norm (an encoder column) has nothing to solve for and must SAY so.
    x2, info2 = rollouts_nla.build_inputs(
        u[:1].astype(np.float32), [{"act_norm": None}], mu.astype(np.float32), "exact", 1.0
    )
    assert info2[0]["fallback"] == "no_act_norm", info2




def check_heldout_v3_recovery():
    """`features/heldout_v3.recover_raw` returns the ACTIVATION, and refuses when it cannot.

    Its whole claim is that a row stored as `unit(act - mu)` plus `||act||` is recoverable, so
    the reference here is the activation it was built from -- never the solve restated. Two
    outcomes are pinned: the recovery and its two identities on a solvable draw, and the
    REFUSAL on a row whose norm is unreachable on the line `mu + R*u`, because silently keeping
    such a row would put a `mu`-shaped vector into `act.f32` under a raw contract.
    """
    import numpy as np

    from features import heldout_v3

    rng = np.random.default_rng(11)
    d = 64
    mu = rng.normal(size=(d,)).astype(np.float64) * 2.0
    act = rng.normal(size=(5, d)).astype(np.float64) * 40.0
    u = (act - mu) / np.linalg.norm(act - mu, axis=1, keepdims=True)
    a = np.linalg.norm(act, axis=1)
    x, info, ident = heldout_v3.recover_raw(u.astype(np.float32), a, mu.astype(np.float32))
    cos = np.einsum("nd,nd->n", x, act) / (np.linalg.norm(x, axis=1) * np.linalg.norm(act, axis=1))
    assert cos.min() > 1 - 1e-4, f"recover_raw did not return act: min cos {cos.min():.6f}"
    assert ident["min_cos_to_shipped"] > 1 - 1e-6 and ident["max_abs_norm_error"] < 1e-3, ident
    assert all(r["fallback"] is None for r in info), info
    # `||mu||` here is 2*sqrt(d) ~ 16; a row claiming ||act|| = 1e-3 is not on that line at all.
    try:
        heldout_v3.recover_raw(u[:1].astype(np.float32), np.array([1e-3]), mu.astype(np.float32))
    except AssertionError as e:
        # Either guard is a valid refusal -- the fallback count, or the ||act|| identity that
        # catches the `mu`-shaped vector the fallback would otherwise hand back. Both have to
        # go for a row to get through, which is what the mutation battery in SMOKES.md removes.
        assert "fell back" in str(e) or "the stored norm" in str(e), f"wrong refusal: {e}"
    else:
        raise AssertionError("recover_raw accepted a row the solve cannot reach")


def check_set_on_disk():
    """`modal_app._check_set_on_disk` opens a set and reconciles it with its declaration.

    Three outcomes, because `check` is the cheap gate in front of every GPU run: a set that is
    not on this root is SKIPPED (a smoke root carries two of the twelve declared sets, and
    failing on the other ten would make the gate useless), a set that agrees with its config
    entry passes, and a set that disagrees FAILS here rather than inside a family-keyed product
    an hour into an H200.
    """
    import numpy as np

    cfg = _conv_cfg()
    cfg["heldout"]["s1"] = {"storage": "raw",
                            "families": {"realact": {"n": 2}, "random": {"n": 2}}}
    fams = cfg["heldout"]["s1"]["families"]
    rows = [{"row": i, "family": f, "id": i}
            for i, f in enumerate(["realact", "realact", "random", "random"])]
    act = np.random.default_rng(1).normal(size=(4, D)).astype(np.float32) * 30.0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert C.check_set_on_disk(cfg, "tb", "s1", fams, str(root))["status"] == "absent"
        sdir = root / "base" / "tb" / "heldout" / "s1"
        _write_set(sdir, rows, act, {"storage": "raw"})
        got = C.check_set_on_disk(cfg, "tb", "s1", fams, str(root))
        assert got["status"] == "ok", got
        assert "storage raw" in got["detail"], got

        # The failure this exists for: the rows on the volume disagree with the declaration.
        cfg["heldout"]["s1"]["families"]["random"]["n"] = 3
        try:
            C.check_set_on_disk(cfg, "tb", "s1", {**fams}, str(root))
        except AssertionError as e:
            assert "declares" in str(e), f"wrong assert for a count mismatch: {e}"
        else:
            raise AssertionError("check accepted a set whose rows disagree with its config entry")
        cfg["heldout"]["s1"]["families"]["random"]["n"] = 2

        # A raw set must carry act.f32, and only a raw set may.
        (sdir / "act.f32").unlink()
        try:
            C.check_set_on_disk(cfg, "tb", "s1", fams, str(root))
        except AssertionError as e:
            assert "act.f32 is absent" in str(e), f"wrong assert for a raw set with no act: {e}"
        else:
            raise AssertionError("check accepted a `storage: raw` set with no act.f32")


def check_csr_gate_floor():
    """`sae_self._csr_at_argmax` tolerates the f16 STORAGE cast and nothing wider.

    `score` selects CSR entries with `a > gate` in fp32 and stores `a` as float16. Round-to-
    nearest puts every fp32 value just above the gate onto the f16 value NEAREST the gate, which
    for the 2M checkpoint's gate 1.682811975479126 is 1.6826171875 -- BELOW it. A correct write
    therefore reads back under the gate, and the original assert (`> gate`) failed on the cast.
    MEASURED: it killed a paid `sae_self` on `rl-last16` x `2026-09-21_v3_sae2m` after the
    forward, with 8,023 of 9,855,412 entries at that one value and no other value under the gate.

    Both directions are the check: the storage floor must PASS, and a value a hair below it --
    which no cast can produce, so it means the wrong dictionary or the wrong gate -- must FAIL.
    """
    import numpy as np

    from autointerp.sae_self import _csr_at_argmax

    gate = 1.682811975479126
    floor = float(np.float16(gate))
    assert floor < gate, "this check is vacuous unless float16(gate) really is below the gate"

    with tempfile.TemporaryDirectory() as td:
        def write(vals):
            C.write_array(f"{td}/sae_off.i64", np.arange(len(vals) + 1, dtype=np.int64), "int64")
            C.write_array(f"{td}/sae_idx.i32", np.full(len(vals), 7, dtype=np.int32), "int32")
            C.write_array(f"{td}/sae_val.f16", np.asarray(vals, dtype=np.float16), "float16")

        # what the cast really produces for activations just above the gate
        stored = [float(np.float16(gate + d)) for d in (1e-6, 1e-5, 1e-4)]
        assert set(stored) == {floor}, f"expected the cast to land on {floor}, got {stored}"
        write(stored)
        val, has = _csr_at_argmax(td, len(stored), 1, list(range(len(stored))), [7] * len(stored), gate)
        assert has.all() and abs(float(val.min()) - floor) < 1e-9, (val, has)

        # one f16 tick below the floor is NOT reachable by the cast: it must still trip
        bad = float(np.nextafter(np.float16(floor), np.float16(0)))
        write([floor, bad])
        try:
            _csr_at_argmax(td, 2, 1, [0, 1], [7, 7], gate)
        except AssertionError as e:
            assert "below float16" in str(e), f"wrong assert text: {e}"
        else:
            raise AssertionError("a CSR entry below the f16 storage floor was accepted")


def check_heldout_v3_ours_block():
    """`heldout_v3 --block ours` copies a CENTRABLE family, and only out of a raw source.

    Four properties, each of which was a live way to get eval 1's sanity block wrong:
      * the copy is byte-for-byte -- `act.f32` of the copied rows IS the source's, so the
        `--block ours` set and `2026-09-21_v1raw` are the same targets and not a re-draw;
      * `src_row` survives the renumbering, which is what makes an exclusion index typed against
        the v1 draw meaningful in the new block;
      * the six exclusions land on the rows they name and are RECORDED, not applied (the row
        count is unchanged, so every arm still pairs);
      * `--block ctrl` still REFUSES a centrable family, and `ours` still refuses a `storage:
        unit` source. The permission is the raw contract, not the block name.
    """
    import numpy as np

    from features import heldout_v3

    cfg = _conv_cfg()
    n_src = 24
    rows_src = [{"row": i, "family": "realact", "id": 1000 + i, "act_norm": 90.0 + i}
                for i in range(n_src)]
    act = np.random.default_rng(7).normal(size=(n_src, D)).astype(np.float32) * 30.0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "base" / "tb" / "heldout" / "v1raw"
        _write_set(src, rows_src, act, {"storage": "raw"})
        args = {"base": "tb", "root": str(root), "dirs_from": str(src), "rows": "0-15"}

        notes: list[str] = []
        rows, arr, contract, extra = heldout_v3._block_ctrl(
            cfg, args, notes, block="ours", allow_centrable=True, exclude=(3, 11))
        assert len(rows) == 16 and contract["storage"] == "raw", (len(rows), contract)
        assert np.array_equal(arr, act[:16]), "the copy is not the source's own act.f32 bytes"
        assert [r["src_row"] for r in rows] == list(range(16)), "src_row did not survive the copy"
        exc = extra["exclusions.json"]
        assert exc["excluded_rows"] == [3, 11] and exc["n_headline"] == 14, exc
        assert [r["row"] for r in rows if r["excluded"]] == [3, 11], "excluded flags are on the wrong rows"
        assert sum(r["excluded"] for r in rows) == 2 and len(rows) == 16, (
            "exclusions were APPLIED, not recorded: the block must keep all its rows so the arms pair")

        # an exclusion index outside the copied range is a typo, not a silent no-op
        try:
            heldout_v3._block_ctrl(cfg, args, [], block="ours", allow_centrable=True, exclude=(3, 99))
        except AssertionError as e:
            assert "not inside the copied range" in str(e), f"wrong assert: {e}"
        else:
            raise AssertionError("an exclusion index outside the block was accepted")

        # the permission is the CONTRACT: `ctrl` still refuses a centrable family ...
        try:
            heldout_v3._block_ctrl(cfg, args, [], block="ctrl")
        except AssertionError as e:
            assert "is centrable" in str(e), f"wrong assert for ctrl on a centrable family: {e}"
        else:
            raise AssertionError("--block ctrl copied a centrable family")

        # ... and `ours` refuses a source whose stored rows are already centred. (Mutating the
        # contract assert away makes this go red on the SECOND guard instead -- a `storage: unit`
        # set has no act.f32 to open -- which is red either way, and the message names the file.)
        unit_src = root / "base" / "tb" / "heldout" / "v1unit"
        vecs = act / np.linalg.norm(act, axis=1, keepdims=True)
        _write_set(unit_src, rows_src, None, {"storage": "unit"}, vecs=vecs)
        try:
            heldout_v3._block_ctrl(cfg, {**args, "dirs_from": str(unit_src)}, [],
                                   block="ours", allow_centrable=True, exclude=())
        except AssertionError as e:
            assert "has no act.f32 to copy" in str(e), f"wrong assert for a unit source: {e}"
        else:
            raise AssertionError("--block ours copied out of a `storage: unit` set")


def check_sae_column_reader():
    """`draw_sae2m._columns` is `common.load_sae`'s two matrices, sliced instead of cast whole.

    The encoder side has to be BIT-IDENTICAL or the 2M sets drawn before and after this change
    are not comparable; the decoder side has to be the feature's COLUMN of `decoder.weight`
    (nn.Linear stores `[out, in]`, so the decoder that maps F -> d is `[d, F]`) -- transposing
    the wrong way gives `d` rows of length F that still normalise, and would be silent.
    """
    import numpy as np
    import torch

    from features import draw_sae2m

    rng = np.random.default_rng(3)
    F, d = 37, 8
    enc = torch.tensor(rng.normal(size=(F, d)), dtype=torch.float32)   # encoder.weight [F, d]
    dec = torch.tensor(rng.normal(size=(d, F)), dtype=torch.float32)   # decoder.weight [d, F]
    dec = dec / dec.norm(dim=0, keepdim=True)                          # load_sae asserts unit rows
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ae.pt")
        torch.save({"encoder.weight": enc, "decoder.weight": dec,
                    "encoder.bias": torch.zeros(F), "bias": torch.zeros(d),
                    "threshold": torch.tensor(1.5)}, path)
        drawn = np.array([0, 5, 36, 12], dtype=np.int64)
        cols, gate, d_sae = draw_sae2m._columns(path, d, drawn, ("enc", "dec"))
        assert (gate, d_sae) == (1.5, F), (gate, d_sae)
        sae = C.load_sae(path, d, device="cpu", dtype=torch.float32, need_decoder=True)
        want_enc = torch.nn.functional.normalize(
            sae.W_enc[:, torch.as_tensor(drawn)].T.contiguous(), dim=-1)
        assert torch.equal(cols["enc"], want_enc), (
            "the sliced encoder side is not bit-identical to load_sae's: "
            f"max |d| {float((cols['enc'] - want_enc).abs().max()):.3e}")
        want_dec = torch.nn.functional.normalize(sae.W_dec[torch.as_tensor(drawn)], dim=-1)
        assert torch.allclose(cols["dec"], want_dec, atol=1e-6), (
            "the decoder side is not unit(W_dec[f]): "
            f"max |d| {float((cols['dec'] - want_dec).abs().max()):.3e}")
        # and the two sides are genuinely different vectors, so a copy-paste would be caught
        assert float((cols["enc"] * cols["dec"]).sum(1).abs().max()) < 0.99, (
            "enc and dec came out collinear on a random checkpoint -- one side is a copy")
    # A row emitted by the draw must SAY which side it is, or sae_rows_of defaults it to enc.
    src = (Path(__file__).resolve().parent.parent / "features/draw_sae2m.py").read_text()
    assert '"sae_side": sd' in src, "draw_sae2m emits no sae_side field"


def check_sae_self_side_flag():
    """`--sae-side` reaches `sae_self`'s row selector, refuses everywhere else, and moves the path.

    The gap it closes is eval 1's: `_sae_rows` pinned `side="enc"`, so the 512 `sae_side: dec`
    rows of `2026-09-21_v3_sae2m` had cosines from `score` and no activation metric at all
    (SMOKES.md 2026-09-21, "three things the results run must not get wrong", item 2). Four
    things have to hold and each is its own failure:

      1. the default is `enc`, bit-for-bit the selection this stage has always made -- every
         product on the volume was written under it and none of them may move;
      2. `dec` selects the OTHER block, not both and not the same one;
      3. the three corpus-side stages in that file refuse a non-default side rather than
         quietly writing a decoder row's numbers to the encoder row's path;
      4. `dec` writes to `sae_self__dec`, so the decoder run cannot land on the encoder product
         eval 1 already paid $0.4 for. This is checked on the SOURCE, because the only other way
         to see it is to run a 27B forward.
    """
    from autointerp.sae_self import sae_side_of

    rows = [
        {"row": 0, "family": "sae", "sae_key": "b/two_m", "id": 10, "sae_side": "enc"},
        {"row": 1, "family": "sae", "sae_key": "b/two_m", "id": 11, "sae_side": "enc"},
        {"row": 2, "family": "sae", "sae_key": "b/two_m", "id": 10, "sae_side": "dec"},
        {"row": 3, "family": "sae", "sae_key": "b/two_m", "id": 11, "sae_side": "dec"},
        # predates the field: reads as `enc` and must stay in the default selection
        {"row": 4, "family": "sae", "sae_key": "b/two_m", "id": 12},
    ]
    assert sae_side_of({}, "sae_self") == "enc", "an unset --sae-side must default to enc"
    assert sae_side_of({"sae_side": ""}, "build") == "enc", "an EMPTY side is the default, not a request"

    # THROUGH `_sae_rows`, not through `sae_rows_of` -- an earlier version of this check called
    # the selector directly and stayed green under the mutation that matters most, `_sae_rows`
    # pinning `side="enc"` and ignoring the flag it was just given. The fixture is a real set
    # directory under a temp root, because `_sae_rows` resolves the dictionary and the
    # declaration off disk.
    from autointerp.sae_self import _sae_rows

    cfg = C.load_config()
    base, sae_key = "qwen36-27b", "qwen36-27b/sae2m"
    with tempfile.TemporaryDirectory() as td:
        hdir = Path(C.heldout_dir(base, "fixture_sides", td))
        hdir.mkdir(parents=True)
        C.write_jsonl(hdir / "ids.jsonl", [{**r, "sae_key": sae_key} for r in rows])
        with open(hdir / C.STORAGE_FILE, "w") as fh:
            json.dump({"storage": "dirs_only", "sae_key": sae_key}, fh)
        args = {"base": base, "root": td, "heldout": "fixture_sides", "sae": sae_key}
        _meta, sel, feats, key, side = _sae_rows(cfg, args, "sae_self")
        assert (sel, feats, key, side) == ([0, 1, 4], [10, 11, 12], sae_key, "enc"), (
            f"the DEFAULT selection moved: {sel} / {feats} / {side}. Every sae_self product on "
            f"the volume was written under it and none of them may change meaning.")
        _meta, sel, feats, key, side = _sae_rows(cfg, {**args, "sae_side": "dec"}, "sae_self")
        assert (sel, feats, side) == ([2, 3], [10, 11], "dec"), (
            f"--sae-side dec selected {sel} / {feats} / {side}: it must be the decoder block "
            f"alone, and `_sae_rows` must PASS the side on rather than pinning `enc`.")

    for stage in ("random_pool", "examples_4m", "examples_docmax", "build"):
        try:
            sae_side_of({"sae_side": "dec"}, stage)
        except AssertionError as e:
            assert "is a `sae_self` flag" in str(e), f"wrong refusal for stage {stage}: {e}"
        else:
            raise AssertionError(f"stage {stage} accepted --sae-side dec; only sae_self may ask for it")
    try:
        sae_side_of({"sae_side": "encoder"}, "sae_self")
    except AssertionError as e:
        assert "must be `enc` or `dec`" in str(e), f"wrong refusal for a bad side: {e}"
    else:
        raise AssertionError("sae_side_of accepted a side that is neither enc nor dec")

    # ONE DIRECTION PER FLAT ROLLOUT ROW. `common.score_tokens` pairs `dirs[i]` with `texts[i]`
    # over the flattened [N, n] grid, and the encoder branch gets this for free because
    # `sae_dirs(sae, row_feats)` is already indexed that way. The decoder branch reads the set's
    # [N_set, d] array and has to EXPAND it, and the first version indexed the targets instead --
    # 512 directions for 32,768 texts. That cost ~$0.4 over three H200 containers, because it got
    # through the base load, the SAE load and the direction read before the scorer's own pairing
    # assert fired. Nothing here needs a GPU.
    import numpy as np

    from autointerp.sae_self import stored_dirs_of

    all_dirs = np.arange(12, dtype=np.float32).reshape(4, 3)   # set rows 0..3, d = 3
    flat = [{"row": r, "k": k} for r in (2, 0) for k in range(3)]   # 2 targets x 3 rollouts
    got = stored_dirs_of(all_dirs, flat)
    assert got.shape == (6, 3), (
        f"stored_dirs_of gave {got.shape} for {len(flat)} rollout texts: it must be one direction "
        f"PER FLAT (target, rollout) row, not one per target")
    assert np.array_equal(got, np.stack([all_dirs[2]] * 3 + [all_dirs[0]] * 3)), (
        f"stored_dirs_of did not keep `flat`'s own order:\n{got}")

    # The side is part of the product PATH, and `enc` keeps the historical name exactly.
    src = (Path(__file__).resolve().parent.parent / "autointerp/sae_self.py").read_text()
    assert 'side_suffix = "" if side == "enc" else f"__{side}"' in src, (
        "sae_self no longer puts the side in the product path: a `dec` run would --force over "
        "the `enc` product on the same scores directory")
    assert 'out = f"{sdir}/sae_self{side_suffix}{args.get(\'out_suffix\') or \'\'}"' in src, (
        "the side suffix is not in `out`")
    # And the decoder half must NOT be scored against the re-derived encoder column, which is a
    # different vector from the one its rollouts were generated with.
    assert 'if side == "enc":' in src and "C.dirs_for(cfg, base, hdir, None, root" in src, (
        "sae_self does not read the SET's stored vecs for a decoder row: `common.sae_dirs` is "
        "unit(W_enc[:, f]) and CHECK 1 would fire on the direction, not on the run")


def check_nla_arm_a_reads_the_body():
    """`build.run`'s rollout loop calls `explanation_token_mask` under `is_nla` -- the WIRING.

    `autointerp/selfcheck.check_nla_body_tokens` pins what the mask returns, and a mutation that
    deleted the call site entirely stayed GREEN against it: a covered function reached by nothing
    is the same as no fix at all. That mutation was the fifth of this session to come back green
    and the fifth to mean a mis-aimed test, so the call site gets its own check. `ast` rather than
    a run, because the branch needs a real NLA build on the volume to execute.

    The function named here is Ari's shared slicer (`precompute/rollouts_nla`), which the
    2026-09-21 rebase kept over this branch's own `nla_body_tokens`; the check is on the WIRING,
    so it stays valid whichever of the two reaches the call site -- rename it and it goes red.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "autointerp" / "build.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    guarded = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Name) and test.id == "is_nla"):
            continue
        if any(isinstance(c, ast.Call)
               and isinstance(c.func, ast.Name)
               and c.func.id == "explanation_token_mask"
               for c in ast.walk(node)):
            guarded = True
    assert guarded, (
        "build.run has no `if is_nla:` branch calling `explanation_token_mask`. Arm A would then "
        "show the explainer the FULL decode -- `<explanation>` tags and any preamble -- while arm "
        "B shows the stripped body, which is the defect Juan's review found: two NLA arms reading "
        "different text from one rollout, only one of them reading the description."
    )
    print("  build.run: arm A's rollout pool is sliced to the explanation body")


def check_sae_column_slice_is_the_dictionary():
    """`load_sae_columns` gives `sae_encode`/`sae_dirs` exactly what the full load would.

    The slice exists because `gcg --mode epo --sae qwen36-27b/sae2m` OOMed an H200 loading a
    43 GB W_dec it never reads. The risk it introduces is an INDEX one -- a column slice whose
    `W_enc[:, 0]` is feature 0 of the slice and feature 125750 of the dictionary -- and that
    failure is silent: a wrong encoder column still produces a plausible activation. So this
    builds a tiny SAE checkpoint, loads it both ways, and requires the two to agree BIT FOR BIT
    on the features the slice holds, plus a refusal on one it does not.
    """
    import torch

    d, f_all = 6, 11
    g = torch.Generator().manual_seed(4242)
    W = torch.randn(f_all, d, generator=g)          # nn.Linear stores [out, in]
    Wd = torch.nn.functional.normalize(torch.randn(f_all, d, generator=g), dim=1)
    ck = {"encoder.weight": W, "decoder.weight": Wd.T.contiguous(),
          "encoder.bias": torch.randn(f_all, generator=g),
          "bias": torch.randn(d, generator=g), "threshold": torch.tensor(1.25)}
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "sae.pt")
        torch.save(ck, path)
        want = [7, 2, 9]
        full = C.load_sae(path, d, need_decoder=False)
        part = C.load_sae_columns(path, d, want)
        assert part.d_sae == f_all, f"a slice must still report the dictionary width: {part.d_sae}"
        assert part.n_cols == len(want), part.n_cols
        assert part.W_dec is None, "the slice must not carry a decoder"
        h = torch.randn(5, d, generator=g)
        for ids in ([7], [9, 2], want):
            a_full = C.sae_encode(full, h, ids)
            a_part = C.sae_encode(part, h, ids)
            assert torch.equal(a_full, a_part), (
                f"sae_encode disagrees on {ids}: max |d| "
                f"{float((a_full - a_part).abs().max()):.3e} -- the slice is indexing the wrong "
                f"columns"
            )
            assert torch.equal(C.sae_dirs(full, ids), C.sae_dirs(part, ids)), (
                f"sae_dirs disagrees on {ids}"
            )
        assert torch.equal(full.b_dec, part.b_dec), "b_dec must come across whole"
        try:
            C.sae_encode(part, h, [3])
        except AssertionError as exc:
            assert "does not hold feature" in str(exc), f"refused for the wrong reason: {exc}"
        else:
            raise AssertionError(
                "a feature outside the slice was encoded, not refused -- which is the silent "
                "wrong-column failure this whole check exists for"
            )


def check_gcg_never_loads_the_full_dictionary():
    """`gcg/gcg.py` calls `load_sae_columns`, never `load_sae`.

    The product reads ONE encoder column per direction and its objective is a cosine to a
    direction off `vecs.f16`, so a full load is 43 GB of W_dec (at 2^21, fp32) that nothing
    touches -- and on the 2M dictionary it is not a waste but a hard OOM at setup. `ast` over the
    source rather than a grep, so a call spelled `C.load_sae(...)` inside a branch this smoke
    never runs is still caught on CPU instead of on an H200 four hours in.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "gcg" / "gcg.py").read_text()
    called = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if name in ("load_sae", "load_sae_columns"):
                called.add(name)
    assert "load_sae" not in called, (
        "gcg/gcg.py calls `load_sae`, which moves the WHOLE dictionary to the device including "
        "W_dec (43 GB in fp32 at 2^21 features). It reads one encoder column per direction: use "
        "`load_sae_columns`. This is the call that OOMed an H200 at setup on --sae qwen36-27b/sae2m."
    )
    assert "load_sae_columns" in called, (
        "gcg/gcg.py no longer loads any SAE -- if the activation block was removed, remove this "
        "check with it rather than leaving it green on nothing"
    )


def check_autointerp_main_forwards_every_flag():
    """Every `autointerp/modal_app.main` parameter reaches the container, or is named local-only.

    The same defect class D7 found on the precompute path, on the autointerp one: a flag can be
    added to the entrypoint signature and not to the `args` dict it builds, and then it parses,
    type-checks, appears in `--help`, and is silently dropped. MEASURED 2026-09-21: that is exactly
    what `--floor-source-arm` did on its first run -- the operator passed it, the container never
    saw it, and the run took the config default instead. Nothing raised; the only trace was a
    stdout line saying the fallback had fired when it should not have.

    `ast` rather than an import: this smoke runs on CPU without `modal`, and modal_app.py decorates
    at module scope.
    """
    import ast

    here = Path(__file__).resolve().parent.parent / "autointerp"
    tree = ast.parse((here / "modal_app.py").read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    sig = {a.arg for a in fn.args.args}
    # The dict literal assigned to `args` inside main -- the thing that crosses to the container.
    forwarded = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "args" and isinstance(node.value, ast.Dict):
                forwarded |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert forwarded, "autointerp/modal_app.main no longer builds `args` as a dict literal"
    # Handled by the entrypoint itself rather than sent: `stage` selects the function, `set` and
    # `heldout` are folded into one `heldout` key, and `dry_launch` returns before any `.remote()`.
    local_only = {"stage", "set", "heldout", "dry_launch"}
    missing = sorted(sig - forwarded - local_only)
    assert not missing, (
        f"autointerp/modal_app.main takes {missing} but never puts them in `args`, so passing one "
        f"on the command line changes nothing and the stage takes its default. Add it to the dict, "
        f"or to this check's `local_only` with the reason."
    )
    # Provenance the entrypoint ADDS rather than takes: the commit the image was built from and
    # the literal command line. They are in `args` on purpose and are not flags.
    stale = sorted(forwarded - sig - {"repo_commit", "argv"})
    assert not stale, (
        f"`args` forwards {stale}, which are not parameters of main -- a flag that can never be set"
    )
    print(f"  autointerp/modal_app: {len(sig)} flags, {len(forwarded)} forwarded, "
          f"{len(local_only)} local-only")


def check_spawn_mirrors_main():
    """`features/spawn.py`'s DEFAULTS and `modal_app.main`'s signature carry the SAME arguments.

    D7. `spawn.py` calls the Modal function directly and so bypasses every assert in the
    entrypoint; four knobs (ps_alpha, ps_prompt, ps_rule, subset) existed on one path and not the
    other, which means a run launched the other way silently took a default nobody chose. This
    parses modal_app.py with `ast` rather than importing it -- the CPU smoke has no `modal` -- and
    compares the two key sets, so the drift fails here instead of on an H200.
    """
    import ast

    here = Path(__file__).resolve().parent
    fn = next(
        n for n in ast.walk(ast.parse((here / "modal_app.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    sig = {a.arg for a in fn.args.args} - {"product"}

    # spawn.py imports `modal` at module scope and this smoke runs without it, so its two
    # module-level literals are read with ast rather than by importing the module.
    spawn_tree = ast.parse((here.parent / "features" / "spawn.py").read_text())
    found = {}
    for node in spawn_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("DEFAULTS", "_LOCAL_ONLY"):
                found[target.id] = ast.literal_eval(node.value)
    assert sorted(found) == ["DEFAULTS", "_LOCAL_ONLY"], (
        f"features/spawn.py no longer defines DEFAULTS and _LOCAL_ONLY as module-level literals "
        f"(found {sorted(found)}); this check reads them without importing modal"
    )
    have = set(found["DEFAULTS"])
    assert sig == have, (
        f"features/spawn.py and precompute/modal_app.py:main have drifted -- only in main: "
        f"{sorted(sig - have)}; only in spawn: {sorted(have - sig)}. Add it to both, or a run "
        f"launched the other way takes a default nobody chose (D7)."
    )
    for name in found["_LOCAL_ONLY"]:
        assert name in have, f"spawn._LOCAL_ONLY names {name!r}, which is not an argument"




# How many values each direction/target loader returns. MEASURED THE EXPENSIVE WAY 2026-09-21:
# `_realact` was given a third return value (the raw activations) and its `return` statement was
# not updated with it, which nothing on CPU could see -- `targets.run` needs a GPU -- so it
# surfaced as `ValueError: not enough values to unpack (expected 3, got 2)` after a 148-second
# H200 forward, with the product's temp directory left on the volume. These functions are the
# seam this branch kept moving, they are all GPU-only, and an arity is exactly the kind of thing
# `ast` can check for free.
RETURN_ARITY = {
    ("precompute/targets.py", "_realact"): 3,
    ("precompute/score.py", "_load_dirs"): 5,
    ("precompute/patchscopes.py", "_patch_check"): 3,
    ("precompute/rollouts_hf.py", "load_dirs"): 3,
    ("precompute/scan.py", "_load_targets"): 4,   # + the window-side mean under --centre
    ("precompute/centred.py", "_load_dirs"): 4,
    ("gcg/gcg.py", "_load_targets"): 2,
    ("autointerp/sae_self.py", "_sae_rows"): 5,
    ("precompute/common.py", "mu_for"): 2,
    # 2 since 2026-09-21: it hands back the shipped generation_config so the summary can carry it.
    # The one unpack site is `rollouts_nla.run`, on a GPU, after the weights are loaded.
    ("precompute/rollouts_nla.py", "check_sidecar"): 2,
}


def check_scores_dir_puts_the_tag_where_rollout_stem_does():
    """`scores_dir` and `rollout_stem` spell one (set, engine, tag) triple the SAME way.

    C7. `scores_dir` took no tag, so a tagged score run could only name itself through
    `--score-name <set>__<tag>`, and `rollout_stem` then appended `__<engine>` after the tag. The
    volume carries the mismatched pair: `rollouts/...__vllm__mu-none.jsonl` beside
    `scores/...__mu-none__vllm/`. `results.common.parse_scores_dir` already reads either order and
    its docstring records what the first, order-dependent version cost -- six of eval 1's arms
    labelled HF in the paper's CSV when they were vLLM. This check is on the WRITER, so the two
    cannot drift apart again.
    """
    import tempfile

    maemm, set_name = "qwen3-8b/2026-09-03_run1-rl", "s"
    for engine in C.ENGINES:
        for tag in ("", "mu-none"):
            want = C.rollout_stem(set_name, engine, tag)
            got = C.scores_dir(maemm, set_name, "/vol", engine, tag, write=True)
            assert got.endswith("/" + want), (
                f"scores_dir({engine!r}, tag={tag!r}) -> {got}, but rollout_stem spells the same "
                f"triple {want!r}. The score directory must follow the rollouts file it scored.")
    # untagged paths do not move: every product already on the volume keeps its name
    assert C.scores_dir(maemm, set_name) == "/vol/maemms/qwen3-8b/2026-09-03_run1-rl/scores/s"

    # THE READER'S FALLBACK, on a real directory rather than on the docstring's promise.
    with tempfile.TemporaryDirectory() as td:
        legacy = C.scores_dir(maemm, f"{set_name}__mu-none", td, "vllm", write=True)
        os.makedirs(legacy)
        got = C.scores_dir(maemm, set_name, td, "vllm", "mu-none")
        assert got == legacy, (
            f"a scores directory written under the OLD spelling is no longer found: asked for "
            f"tag `mu-none`, got {got}, the product is at {legacy}")
        # ...and the canonical path wins as soon as it exists
        canon = C.scores_dir(maemm, set_name, td, "vllm", "mu-none", write=True)
        os.makedirs(canon)
        assert C.scores_dir(maemm, set_name, td, "vllm", "mu-none") == canon, (
            "the legacy directory shadowed the canonical one")
    print("  scores_dir: __<engine>__<tag>, with the legacy order still readable")


def check_both_tag_axes_reach_the_scores_path():
    """`--run-tag` AND `--score-tag` are both in the scores directory name, and in that order.

    THE REBASE HAZARD THIS EXISTS FOR. `evals/pipeline` and `evals/pipeline-ood` each gave
    `scores_dir` a `tag` parameter, in the same position, with the same name, within a day of each
    other -- and meant different things by it:

      `--run-tag`   selects which rollouts FILE is scored;
      `--score-tag` scores ONE rollouts file again under a second convention (`cos_asym` is why).

    Keeping either side's resolution alone loses the other axis with NO error anywhere: the second
    run writes the first run's directory, `--force` replaces it, and the only trace is a README
    naming a different rollouts file. `score_tag_of` composes both; this check is on the
    composition and on every call site, because a call site that builds the tag by hand is the
    same defect wearing a different sleeve.
    """
    import ast

    assert C.score_tag_of({}) == "", "an untagged run must keep the historical bare path"
    assert C.score_tag_of({"run_tag": "mu-none"}) == "mu-none", (
        "the RUN tag vanished from the scores path: two rollouts files would score into one dir")
    assert C.score_tag_of({"score_tag": "asym"}) == "asym", (
        "the SCORE tag vanished: a re-score of one rollouts file would overwrite the first result")
    assert C.score_tag_of({"run_tag": "mu-none", "score_tag": "asym"}) == "mu-none__asym", (
        "the two axes must BOTH appear, run tag first -- every score of one rollouts file sorts "
        "together only if the run tag is the outer component")
    # whitespace is not a tag, and `rollout_stem` would accept a padded one into a path
    assert C.score_tag_of({"run_tag": "  ", "score_tag": "asym"}) == "asym"

    # the two axes must land in DIFFERENT directories, which is the whole point
    m, s = "qwen3-8b/2026-09-03_run1-rl", "s"
    seen = {
        C.scores_dir(m, s, "/vol", "vllm", C.score_tag_of(a), write=True)
        for a in ({}, {"run_tag": "r"}, {"score_tag": "t"}, {"run_tag": "r", "score_tag": "t"})
    }
    assert len(seen) == 4, f"four (run tag, score tag) combinations share {4 - len(seen) + 1} paths: {seen}"

    # EVERY call site composes it the same way. A `scores_dir(...)` whose tag argument is
    # `args.get("run_tag")` or `args.get("score_tag")` alone silently drops the other axis.
    root = Path(__file__).resolve().parent.parent
    checked = 0
    for rel in ("precompute/score.py", "precompute/centred.py", "autointerp/sae_self.py",
                "autointerp/build.py"):
        for node in ast.walk(ast.parse((root / rel).read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "scores_dir"):
                continue
            tag_arg = node.args[4] if len(node.args) > 4 else next(
                (k.value for k in node.keywords if k.arg == "tag"), None)
            if tag_arg is None:      # an untagged read of the historical path is legal
                continue
            checked += 1
            ok = (isinstance(tag_arg, ast.Call) and isinstance(tag_arg.func, ast.Attribute)
                  and tag_arg.func.attr == "score_tag_of")
            assert ok, (
                f"{rel}:{node.lineno} builds scores_dir's tag by hand instead of with "
                f"C.score_tag_of(args). One of --run-tag / --score-tag will be missing from the "
                f"path, and the run that collides with it will simply overwrite.")
    assert checked >= 4, f"only {checked} tagged scores_dir call sites found; the check is not reaching them"
    print(f"  scores path: run tag and score tag both in, {checked} call sites compose it")


def check_return_arities():
    """Every loader returns as many values as its callers unpack -- checked with `ast`, on CPU.

    Both halves, because either one alone passes while the pair is broken: every `return` in the
    function yields the declared number of values, AND every `a, b, ... = f(...)` anywhere under
    paper-evals/ unpacks that many. A single-name assignment (`x = f(...)`) is ignored; it is
    legal and says nothing about arity.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    for (rel, fname), want in RETURN_ARITY.items():
        tree = ast.parse((root / rel).read_text())
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname]
        assert len(fns) == 1, f"{rel}: expected exactly one def {fname}, found {len(fns)}"
        got = [
            len(n.value.elts) if isinstance(n.value, ast.Tuple) else 1
            for n in ast.walk(fns[0])
            if isinstance(n, ast.Return) and n.value is not None
        ]
        assert got and set(got) == {want}, (
            f"{rel}:{fname} returns {got} values, but RETURN_ARITY says {want} -- update both, or "
            f"a caller unpacking {want} fails only on a GPU, after the forward it paid for"
        )

    # A leading underscore means module-private, so its unpack sites are looked for in its OWN
    # file only. That is not pedantry: `_load_targets` is `scan`'s (4 values) AND `gcg`'s (2), and
    # `_load_dirs` is `score`'s (5) AND `centred`'s (3). A repo-wide match by bare name would
    # report every one of those as a mismatch.
    seen = {k: 0 for k in RETURN_ARITY}
    for path in sorted(root.rglob("*.py")):
        if "third_party" in path.parts or path.name == "unit_smoke.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, call = node.targets[0], node.value
            if not isinstance(target, ast.Tuple) or not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            for (rel, fname), want in RETURN_ARITY.items():
                if name != fname:
                    continue
                if fname.startswith("_") and path != root / rel:
                    continue  # a module-private name: only its own file can mean this function
                seen[(rel, fname)] += 1
                assert len(target.elts) == want, (
                    f"{path.relative_to(root)}:{node.lineno} unpacks {len(target.elts)} values "
                    f"from {fname}(), which returns {want}"
                )
    unused = [k for k, n in seen.items() if not n]
    assert not unused, f"RETURN_ARITY lists {unused}, which nothing unpacks any more -- stale entry"




def check_centred_uses_one_mu():
    """`centred.py` centres BOTH sides of its cosine on the run's mu, and names it.

    Structural, with `ast`, because the product itself needs a scores directory and a GPU run in
    front of it. It encodes what went wrong: `_load_dirs` resolved the TARGET through `--mu` while
    `:116` hardcoded `C.stats_mu`, so every `rl-last16` run compared `best_act - stats_mu` against
    `unit(act - whiten_mu)` and `centred.json` recorded the stats path either way. Two means, one
    name, and reconstruction/stats.py reads the result into the paper tables.
    """
    import ast

    src = (Path(__file__).resolve().parent / "centred.py").read_text()
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("stats_mu", "stats_dir")
    ]
    assert not calls, (
        f"precompute/centred.py calls {[c.func.attr for c in calls]} at line(s) "
        f"{[c.lineno for c in calls]}: the mean is the RUN's (common.mu_for -> load_mu), never one "
        f"named in this file, or the two sides of cos_centred_best part company again"
    )
    assert "C.load_mu(cfg, base, mu_val, root)" in src, (
        "centred.py no longer loads the run's own mu for the activation side"
    )
    assert 'C.mu_label(mu_val, base, root)' in src, (
        "centred.json must NAME the mean both sides used; reconstruction/stats.py reads this "
        "directory and a reader has to be able to tell two runs apart"
    )




def check_every_set_writer_writes_the_contract():
    """Every tool that DRAWS a held-out set writes its `storage.json`.

    `common.set_storage` refuses a directory that states no contract, and `common.py`'s own
    docstring promises "every set drawn after 2026-09-21 writes it" while the `set_storage` error
    tells the reader to "re-draw the set, which writes the contract itself". Both were false for
    `features/draw_sae2m.py` and `features/heldout_v2.py`, which wrote `ids.jsonl` and `vecs.f16`
    and nothing else -- so a set from either tool was born unreadable and needed a hand-written
    config entry that nothing told the author to write. `ast` keeps the promise honest.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    # FROM `C.SET_WRITERS`, not a second hand-kept list: the D6 tuple and this one had already
    # drifted (`draw_sae131k` arrived on main writing no contract at all, and `spawn.py` held a
    # third copy that knew about neither it nor `heldout_v3`). `heldout_v2` is not a D6 writer --
    # it predates the entrypoint assert -- so it is named separately.
    rels = {"targets": "precompute/targets.py", "draw_sae2m": "features/draw_sae2m.py",
            "draw_sae131k": "features/draw_sae131k.py", "heldout_v3": "features/heldout_v3.py"}
    unmapped = sorted(set(C.SET_WRITERS) - set(rels))
    assert not unmapped, (
        f"C.SET_WRITERS names {unmapped}, which this check has no source file for: a new set "
        f"writer must be added here, or it can be born writing no storage contract at all"
    )
    for rel in [rels[k] for k in C.SET_WRITERS] + ["features/heldout_v2.py"]:
        src = (root / rel).read_text()
        assert C.STORAGE_FILE in src, (
            f"{rel} draws a held-out set but never writes {C.STORAGE_FILE}: common.set_storage "
            f"will refuse every set it produces, and the advice 'the draw writes the contract "
            f"itself' is false for it"
        )
        # and the value has to be one of the declared kinds, not a free-text guess
        kinds = {
            n.value
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and n.value in C.STORAGE_KINDS
        }
        assert kinds, f"{rel} writes {C.STORAGE_FILE} but names no storage kind from {list(C.STORAGE_KINDS)}"




def check_draws_call_finish_with_its_signature():
    """`features/draw_*.py` call `draw_sae2m._finish` with the arity it actually has.

    The defect this is for arrived through a merge, not through an edit: `_finish` gained a
    `peak16` parameter on this branch (between `gated_full` and `cuts`), and `draw_sae131k.py` --
    written against the older 16-argument signature on main -- kept calling it positionally. Git
    reported no conflict, because the two changes are in different files. The call bound `cuts` to
    `peak16` and the meta dict to `cuts`, then raised `TypeError` on the missing `meta_extra`.

    Positional binding across a module boundary is the whole hazard, so the check counts
    positionals rather than trusting that a call which parses is a call that binds.
    """
    import ast
    import inspect

    from features.draw_sae2m import _finish

    params = list(inspect.signature(_finish).parameters)
    n_required = len([p for p in inspect.signature(_finish).parameters.values()
                      if p.default is inspect.Parameter.empty])
    root = Path(__file__).resolve().parent.parent
    seen = 0
    for path in sorted((root / "features").glob("draw_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_finish"):
                continue
            seen += 1
            n_pos = len(node.args)
            kw = {k.arg for k in node.keywords if k.arg}
            assert n_pos + len(kw) >= n_required, (
                f"{path.name}:{node.lineno} calls _finish with {n_pos} positional + {len(kw)} "
                f"keyword arguments; it takes {n_required} required, {params}. A short positional "
                f"call does not fail where it is written -- it binds every argument after the "
                f"missing one to the wrong parameter.")
            assert n_pos <= len(params), (
                f"{path.name}:{node.lineno} passes {n_pos} positional arguments to a _finish that "
                f"takes {len(params)}: {params}")
    assert seen >= 2, f"found only {seen} _finish call sites; the check is not reaching them"
    print(f"  _finish: {seen} call sites, all binding {n_required} parameters")


def check_no_draw_stamps_a_bare_sae_key():
    """No draw overwrites the full `<base>/<name>` sae_key with a bare dictionary name.

    `draw_sae131k.py` did: `_finish` stamped `sae_key: "qwen36-27b/l42-1b"` per row and the caller
    then replaced it with `"l42-1b"`. `common.sae_rows_of` matches the FULL key, so every row
    missed -- and because the rows ARE keyed, just wrongly, the unkeyed branch's loud assert never
    fired. `scan` would have run with `n_feat = 0`, which is not an error condition anywhere. A
    silent empty selection is the worst shape this class of defect takes, so it gets a check that
    names the shape rather than the one file that had it.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    for path in sorted((root / "features").glob("draw_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "sae_key"):
                    continue
                val = node.value
                assert not (isinstance(val, ast.Constant) and isinstance(val.value, str)), (
                    f"{path.name}:{node.lineno} assigns a LITERAL sae_key {val.value!r}. It must "
                    f"be the full `<base>/<name>` config key (C.sae_key_for), or common.sae_rows_of "
                    f"selects nothing and says nothing.")
    print("  draws: no literal sae_key overwrites the full config key")


def check_one_set_writers_tuple():
    """`modal_app` and `features/spawn` enforce D6 off the SAME tuple.

    Both had their own copy and the copies had drifted: modal_app knew `heldout_v3`, spawn did
    not, and neither knew `draw_sae131k`. `spawn.py` is the path that bypasses `modal_app.main`
    entirely -- which is how a set got onto the volume without ever being declared -- so its guard
    being the weaker of the two is the wrong way round.
    """
    import ast

    # ON THE COMPARISON, not on the text. The first version of this check asked whether the source
    # contained the string "C.SET_WRITERS" -- and stayed GREEN under the mutation that put the
    # literal tuple back in the assert, because the COMMENT above the assert names the constant.
    # A grep for a symbol finds the prose about the symbol too.
    root = Path(__file__).resolve().parent.parent
    for rel in ("precompute/modal_app.py", "features/spawn.py"):
        tree = ast.parse((root / rel).read_text())
        reads, literals = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for comp in node.comparators:
                if isinstance(comp, ast.Attribute) and comp.attr == "SET_WRITERS":
                    reads.append(node.lineno)
                # BOTH names, which is the D6 tuple's shape. `targets` alone appears in the
                # per-flag ownership asserts (`--arm` is a corpus/targets/ood_selfcheck flag),
                # and the first version of this check called one of those a second D6 tuple.
                if isinstance(comp, ast.Tuple) and {"targets", "draw_sae2m"} <= {
                    e.value for e in comp.elts if isinstance(e, ast.Constant)
                }:
                    literals.append(node.lineno)
        assert reads, (
            f"{rel}: no `in`/`not in` comparison against `C.SET_WRITERS`. The D6 guard has a "
            f"second copy of the tuple again, and a copy is a thing that drifts")
        assert not literals, (
            f"{rel}:{literals[0]} compares a product against a LITERAL set-writer tuple. There is "
            f"one such tuple and it lives in common.py")
        lits = [n for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SET_WRITERS" for t in n.targets)]
        assert not lits, f"{rel}:{lits[0].lineno} redefines SET_WRITERS; common.py owns it"
    assert "draw_sae131k" in C.SET_WRITERS, (
        "draw_sae131k WRITES a held-out set and --force rmtrees what is there; it must be under D6")
    print(f"  D6: one SET_WRITERS tuple, {list(C.SET_WRITERS)}")


def check_corpus_axis():
    """The corpus is an axis of a scan's OUTPUT PATH, of spawn's resolution, and of the refusal.

    H5/H6/H7 in one check, all three of which were "declared but not wired":
      * `scan_dir` / `sae_examples_dir` key by corpus as well as set, with "" resolving to today's
        path so nothing on the volume moves -- the plan scans ONE set over TWO corpora and both
        used to land on one directory, where the second destroys the first under --force;
      * `features/spawn.py` resolves `--corpus` itself instead of forwarding the key unresolved,
        which left `corpus_name` empty and every spawned scan silently on the default corpus;
      * a corpus whose declared window geometry is not the one all eleven `windows_of` sites cut
        at is REFUSED, not scanned at the wrong width under a README claiming the right one.
    """
    import ast

    cfg = C.load_config()
    assert C.scan_dir("b", "s") == C.scan_dir("b", "s", corpus_name="")
    assert C.scan_dir("b", "s", corpus_name="cc").endswith("/scan/s__cc")
    assert C.sae_examples_dir("b/x", "s", write=True, corpus_name="cc").endswith("/examples/s__cc")
    assert C.sae_examples_dir("b/x", "s", write=True) != C.sae_examples_dir(
        "b/x", "s", write=True, corpus_name="cc"
    ), "two corpora must not share one examples directory"

    assert C.corpus_key_of_dir(cfg, "") == "heldout16m", "the unnamed directory is the 16M corpus"
    assert C.corpus_key_of_dir(cfg, "train_parity_10m") == "celeste-train10m"
    assert C.corpus_key_of_dir(cfg, "nothing-like-this") == ""
    assert C.assert_corpus_geometry(cfg, "") == (C.SCAN_BLOCK, C.SCAN_STRIDE)
    # EVERY configured corpus must match the geometry the pipeline actually cuts at, or it cannot
    # be scanned. `celeste-train10m` declared 32/8 until 2026-09-23 and was refused here; it is
    # 64/16 now (nothing was built at 32/8 -- the corpus product is a geometry-free token stream).
    for key, spec in cfg["corpora"].items():
        got = C.assert_corpus_geometry(cfg, spec["dir"])
        assert got == (C.SCAN_BLOCK, C.SCAN_STRIDE), f"corpus {key}: {got}"
    # and the refusal itself still fires, on a corpus built to disagree
    bad = {**cfg, "corpora": {**cfg["corpora"],
                              "bad": {**cfg["corpora"]["celeste-train10m"],
                                      "dir": "bad_dir", "block": 32, "stride": 8}}}
    try:
        C.assert_corpus_geometry(bad, "bad_dir")
    except AssertionError as e:
        assert "declares window 32/8" in str(e), f"wrong assert for a mismatched geometry: {e}"
    else:
        raise AssertionError(
            "a corpus declaring 32/8 was accepted by a pipeline that cuts 64/16 everywhere"
        )

    spawn_src = (Path(__file__).resolve().parent.parent / "features" / "spawn.py").read_text()
    local_only = next(
        ast.literal_eval(n.value)
        for n in ast.parse(spawn_src).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_LOCAL_ONLY"
    )
    assert "corpus" in local_only, (
        "features/spawn.py forwards `corpus` to the container, where nothing reads it: the key "
        "must be resolved to a directory name on the client, as modal_app.main does"
    )
    assert "corpus_key_name" in spawn_src, "spawn.py does not resolve --corpus to a directory"


# ---------------------------------------------------------------------------------------------
# the OOD generalisation evaluation (infra/2026-09-18_ood-eval-design.md)
#
# A byte-level tokenizer is built HERE, from explicit byte pieces, rather than loaded: the point of
# the covariates is what happens when one character is split across several tokens, and a fixture
# that can only agree with the real tokenizer's own splitting would not test that.
# ---------------------------------------------------------------------------------------------


class FakeByteTok:
    """A byte-level BPE tokenizer over an explicit list of byte pieces; ids index that list."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def convert_ids_to_tokens(self, ids):
        return ["".join(C.BYTE_ENCODER[b] for b in self.pieces[int(i)]) for i in ids]

    def decode(self, ids):
        return b"".join(self.pieces[int(i)] for i in ids).decode("utf-8", errors="replace")


def check_byte_tables():
    """BYTE_ENCODER is a bijection on 0..255 and BYTE_DECODER inverts it."""
    assert len(C.BYTE_ENCODER) == 256, len(C.BYTE_ENCODER)
    assert len(set(C.BYTE_ENCODER.values())) == 256
    assert all(C.BYTE_DECODER[C.BYTE_ENCODER[b]] == b for b in range(256))
    # and it is the table the real tokenizers use: space -> 'Ġ', newline -> 'Ċ'
    assert C.BYTE_ENCODER[0x20] == "\u0120" and C.BYTE_ENCODER[0x0A] == "\u010a"
    tok = FakeByteTok([b"Dobr", b"\xc3\xbd"])
    assert C.check_token_bytes(tok, [0, 1]).startswith("token_bytes verified")


def check_token_covariates_czech():
    """A Latin word split across a byte boundary: 'Dobrý den' as Dobr | \xc3 | \xbd | ' den'."""
    tok = FakeByteTok([b"Dobr", b"\xc3", b"\xbd", b" den"])
    ids = [0, 1, 2, 3]
    c1 = C.token_covariates(tok, ids, 1, "Latin", unspaced=False)
    assert c1["byte_piece"] and not c1["whole_char"] and not c1["multi_char"], c1
    assert c1["char_type"] == "letter_arm", c1  # the y-acute the two pieces make
    assert c1["tok_class"] == "mid" and c1["n_subtokens"] == 3, c1
    assert c1["unitend_p"] == 2 and c1["unitend_rule"] == "wordend", c1
    c0 = C.token_covariates(tok, ids, 0, "Latin", unspaced=False)
    assert c0["tok_class"] == "first" and c0["multi_char"] and not c0["byte_piece"], c0
    c3 = C.token_covariates(tok, ids, 3, "Latin", unspaced=False)
    assert c3["tok_class"] == "word" and c3["n_subtokens"] == 1, c3
    assert c3["char_type"] == "space" and c3["unitend_rule"] == "none", c3
    # `char_type` is the FIRST character (the leading space); `char_type_body` skips it
    assert c3["char_type_body"] == "letter_arm", c3
    # the same window read as a Cyrillic arm: the y-acute is then another script's letter
    assert C.token_covariates(tok, ids, 1, "Cyrillic", unspaced=False)["char_type"] == "letter_other"


def check_token_covariates_thai():
    """An unspaced abugida with a vowel sign split across two tokens (the charend rule)."""
    pieces = [
        "\u0e2a\u0e27".encode(),  # SO SUA + WO WAEN, two whole characters in one token
        b"\xe0\xb8",  # the first two bytes of MAI HAN-AKAT
        b"\xb1",  # its third byte
        "\u0e2a\u0e14\u0e35".encode(),
    ]
    tok = FakeByteTok(pieces)
    ids = [0, 1, 2, 3]
    c1 = C.token_covariates(tok, ids, 1, "Thai", unspaced=True)
    assert c1["tok_class"] == "unspaced" and c1["n_subtokens"] is None, c1
    assert c1["byte_piece"] and c1["unitend_p"] == 2 and c1["unitend_rule"] == "charend", c1
    # MAI HAN-AKAT is a COMBINING MARK: str.isalpha() is False for it and common.is_letter is not,
    # which is the whole reason is_letter exists (an abugida is mostly marks)
    assert not "\u0e31".isalpha() and C.is_letter("\u0e31")
    assert c1["char_type"] == "letter_arm", c1
    c0 = C.token_covariates(tok, ids, 0, "Thai", unspaced=True)
    assert c0["multi_char"] and not c0["byte_piece"] and c0["unitend_rule"] == "none", c0
    c2 = C.token_covariates(tok, ids, 2, "Thai", unspaced=True)
    assert c2["byte_piece"] and c2["unitend_rule"] == "none", c2  # p already ends the character


def check_token_covariates_python():
    """Code: whitespace units over indentation, and a single-token word."""
    pieces = [b"def", b" foo", b"(", b"x", b"):", b"\n    ", b"return", b" x"]
    tok = FakeByteTok(pieces)
    ids = list(range(len(pieces)))
    c0 = C.token_covariates(tok, ids, 0, "Latin", unspaced=False)
    assert c0["tok_class"] == "word" and c0["n_subtokens"] == 1, c0
    c2 = C.token_covariates(tok, ids, 2, "Latin", unspaced=False)
    assert c2["tok_class"] == "mid" and c2["n_subtokens"] == 4, c2  # ' foo' '(' 'x' '):'
    assert c2["unitend_p"] == 4 and c2["char_type"] == "punct", c2
    c1 = C.token_covariates(tok, ids, 1, "Latin", unspaced=False)
    assert c1["char_type"] == "space" and c1["char_type_body"] == "letter_arm", c1
    c6 = C.token_covariates(tok, ids, 6, "Latin", unspaced=False)
    assert c6["tok_class"] == "word", c6  # the indentation token carries the whitespace
    assert C.code_like("def f(x):\n    return {1: 2};\nimport os\n")
    assert not C.code_like("Dobry den, jak se mate? Dnes je krasne pocasi a jdu ven.")


def check_script_fraction():
    assert C.script_fraction("Dobry den", "Latin") == 1.0
    assert C.script_fraction("\u0434\u0435\u043d\u044c", "Cyrillic") == 1.0
    assert C.script_fraction("\u0434\u0435\u043d\u044c", "Latin") == 0.0
    assert C.script_fraction("\u3053\u308c\u306f\u65e5\u672c", "Jpan") == 1.0  # kana + han
    assert C.script_fraction("\u3053\u308c\u306f\u65e5\u672c", "Han") < 1.0
    assert C.script_fraction("123 + 456 = 579", "Latin") != C.script_fraction("123", "Latin")  # nan != nan
    mixed = C.script_fraction("\u0e2a\u0e27\u0e31\u0e2a hello", "Thai")
    assert abs(mixed - 4 / 9) < 1e-9, mixed


def check_arm_perm_and_rng():
    """The permutation is a function of (arm, seed) only, and different arms differ."""
    import numpy as np

    a = C.arm_perm("tha_Thai", 1000, 20260918)
    assert (a == C.arm_perm("tha_Thai", 1000, 20260918)).all()
    assert not (a == C.arm_perm("python", 1000, 20260918)).all()
    assert not (a == C.arm_perm("tha_Thai", 1000, 20260919)).all()
    assert sorted(a.tolist()) == list(range(1000))
    r1 = C.arm_rng("tha_Thai", 20260918).integers(0, 1_000_000, 8)
    assert (r1 == C.arm_rng("tha_Thai", 20260918).integers(0, 1_000_000, 8)).all()
    assert not (r1 == C.arm_rng("python", 20260918).integers(0, 1_000_000, 8)).all()
    # the p/L stream must NOT be the permutation's stream
    assert not np.array_equal(r1[:4], C.arm_perm("tha_Thai", 1_000_000, 20260918)[:4])


def check_ood_config():
    """config.yaml's ood_arms and the two OOD held-out sets, as `check` would see them."""
    cfg = C.load_config()
    arms = C.ood_arms(cfg)
    assert len(arms) == 23, len(arms)
    fams = {}
    for spec in arms.values():
        fams[spec["family"]] = fams.get(spec["family"], 0) + 1
    assert fams == {"lang": 8, "ctrl": 2, "code": 8, "math": 4, "diag": 1}, fams
    # 2026-09-23 (eval plan M5): every ladder carries the 1/4/10 prefix -- 10 is the full run's
    # own-domain search size and 1/4 are the nested prefixes the per-target columns read -- and
    # the four arms of review R2 that reached 16M keep it on the end.
    assert all(s["sizes"][:3] == [1, 4, 10] for s in arms.values()), (
        {a: s["sizes"] for a, s in arms.items() if s["sizes"][:3] != [1, 4, 10]}
    )
    assert [a for a, s in arms.items() if s["sizes"] == [1, 4, 10, 16]] == [
        "tha_Thai",
        "ufw_en",
        "python",
        "owm",
    ], "the four 16M arms of review R2"
    assert [a for a, s in arms.items() if s["unspaced"]] == [
        "tha_Thai",
        "cmn_Hani",
        "jpn_Jpan",
        "ufw_zh",
    ], "the four unspaced arms of review R5"
    assert arms["formulas"]["sizes"] == [1, 4, 10]  # was [1]; 4 added so the prefix is uniform
    assert C.is_ood_set(cfg, "2026-09-18_ood_v1")
    assert not C.is_ood_set(cfg, "2026-09-16_v1")
    assert len(C.ood_set_arms(cfg, "2026-09-18_ood_v1")) == 23
    assert cfg["heldout"]["2026-09-18_ood_v1_unitend"]["variant_of"] == "2026-09-18_ood_v1"
    assert C.families_for(cfg, "2026-09-18_ood_v1", "qwen36-27b") == {}


def check_span_in_corpus():
    """targets._span_in_corpus finds a span and does not find a near miss."""
    import numpy as np

    from precompute.targets import _span_in_corpus

    toks = np.array([5, 1, 2, 3, 9, 1, 2, 4, 1, 2, 3, 7], dtype=np.int32)
    assert _span_in_corpus(toks, np.array([1, 2, 3], dtype=np.int32))
    assert _span_in_corpus(toks, np.array([1, 2, 4], dtype=np.int32))
    assert not _span_in_corpus(toks, np.array([1, 2, 5], dtype=np.int32))
    assert not _span_in_corpus(toks, np.array([3, 7, 7], dtype=np.int32))
    assert _span_in_corpus(toks, np.array([7], dtype=np.int32))
    assert not _span_in_corpus(np.array([1, 2], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))


def check_scan_masks_on_label():
    """`scan` must key the own-document mask on the corpus LABEL, never on `cname`.

    A target's `mask_corpus` says which corpus its `doc` indexes. The base's own English corpus is
    `cname == ""` and `label == "corpus"`, so a comparison against `cname` matches nothing for a
    realact target in a scan of that corpus and every one of them loses its own-document mask --
    silently, and in the direction that INFLATES the corpus-search baseline, which is the effect
    review R1 of the OOD design exists to remove. Checked on the source because the function needs
    a corpus, a set and a GPU to run, and the invariant is one token wide.
    """
    import re

    src = (Path(__file__).resolve().parent / "scan.py").read_text()
    m = re.search(r"keep_mask = torch\.tensor\(\[mc == (\w+) for mc in mask_corpus\]", src)
    assert m, "scan.py no longer builds keep_mask from mask_corpus in the expected shape"
    assert m.group(1) == "label", (
        f"scan.py compares mask_corpus against {m.group(1)!r}; it must be `label` "
        f"(= cname or 'corpus'), or the base's own corpus never matches its own targets"
    )
    # and nothing may append a label that could be empty for a corpus that HAS one
    appends = re.findall(r"mask_corpus\.append\((.+?)\)\s*(?:#|$)", src, re.M)
    assert appends, "scan.py no longer appends to mask_corpus"
    for a in appends:
        assert a.strip() in ('""', '"corpus"') or "or \"corpus\"" in a, (
            f"mask_corpus.append({a.strip()}) can yield an empty label for a corpus that has one; "
            f'append `... or "corpus"` so it is comparable with the scan\'s own label'
        )


def check_sae_key_for_rows():
    """A set with no SAE rows must need no `--sae`, and every SET-READING product must use that.

    `sae_key_for` refuses without `--sae` on a base with two dictionaries. That is right for a
    product about to look feature ids up in one of them and wrong for a set that has none: the OOD
    sets have no `sae` family, and `scan` and `score` each died on it AFTER the model load -- the
    scan after its topk.jsonl had already been paid for and renamed into place.

    The source half is the point: fixing `scan` and leaving `score` is how this cost two calls
    rather than one, so the products that read a SET are required to use the rows-aware helper.
    """
    cfg = C.load_config()
    assert C.sae_key_for_rows(cfg, "qwen36-27b", [{"family": "lang"}, {"family": "code"}]) == ""
    assert C.sae_key_for_rows(cfg, "qwen36-27b", []) == "", "an empty set names no dictionary"
    assert C.sae_key_for_rows(
        cfg, "qwen36-27b", [{"family": "sae"}], "qwen36-27b/l42-1b"
    ) == "qwen36-27b/l42-1b"
    try:
        C.sae_key_for_rows(cfg, "qwen36-27b", [{"family": "realact"}, {"family": "sae"}])
    except AssertionError:
        pass
    else:
        raise AssertionError("a set WITH sae rows must still refuse without --sae")

    # and `score._sae_for` ACTUALLY CALLED, on both row shapes. The source check below would have
    # passed while the call still died in the container on `.values()` of a list -- which is what
    # happened, on the relaunch after the source check was added.
    import precompute.score as S

    for shape in ([{"family": "lang"}], {0: {"family": "lang"}}, None, []):
        assert S._sae_for(cfg, {"base": "qwen36-27b"}, shape) == (None, ""), shape
    assert S._sae_for(cfg, {"base": "qwen36-27b", "no_sae": True}, [{"family": "sae"}]) == (None, "")
    try:
        S._sae_for(cfg, {"base": "qwen36-27b"}, [{"family": "sae"}])
    except AssertionError:
        pass
    else:
        raise AssertionError("score._sae_for must still refuse a set WITH sae rows and no --sae")

    here = Path(__file__).resolve().parent
    for name in ("scan.py", "score.py"):
        src = (here / name).read_text()
        assert "sae_key_for_rows(" in src, (
            f"{name} reads a held-out set and must resolve its dictionary with "
            f"`sae_key_for_rows`, not `sae_key_for`: a set with no sae rows would otherwise be "
            f"refused for want of a --sae it has no use for"
        )
        assert not re.search(r"(?<!_rows)C\.sae_key_for\(", src), (
            f"{name} still calls C.sae_key_for directly somewhere"
        )
    # and the examples notes must not be reachable with sae = None
    scan_src = (here / "scan.py").read_text()
    assert "_examples_notes(" in scan_src and "d_sae: int" in scan_src, (
        "scan.py's examples notes read sae.d_sae; they belong in a function called only under "
        "`if n_feat:`, or a set with no sae rows builds the f-string against None"
    )
    # `sae` is None when the set has no sae rows, so every read of `sae.d_sae` must sit under the
    # `if n_feat:` guard. The two that do are pinned by shape; a third line would be a new one.
    guarded = ("peak = C.read_array(", "_examples_notes(od, n_feat, sae.d_sae,")
    seen_guard = 0
    for ln in scan_src.split("\n"):
        t = ln.strip()
        if t.startswith("if n_feat:"):
            seen_guard = len(ln) - len(t)
        if "sae.d_sae" in t and not t.startswith(('"""', "#", "f\"", '"')):
            assert t.startswith(guarded), (
                f"scan.py reads `sae.d_sae` at an unguarded site: {t!r}. With no sae rows `sae` "
                f"is None there, and the scan dies AFTER its topk.jsonl has been paid for."
            )
            # INDENTATION, not just shape: dedenting the call out of the `if n_feat:` it sits
            # under is exactly the regression, and the line reads identically either way.
            indent = len(ln) - len(t)
            assert indent > seen_guard, (
                f"`{t[:48]}...` is indented {indent} but the `if n_feat:` guarding it is at "
                f"{seen_guard}: it is no longer inside the guard"
            )


def check_three_cosines():
    """`cos`, `cos_centred` and `cos_asym` from one forward, each against its own definition.

        cos          = cos(h,      unit(act))          both sides RAW
        cos_centred  = cos(h - mu, unit(act - mu))     both sides CENTRED
        cos_asym     = cos(h,      unit(act - mu))     the SCAN's convention

    The third is the one a corpus search can be differenced against: `scan` scores every window as
    `normalize(h) @ unit(act - mu)` (precompute/scan.py), and the paper's bo64 0.569 and corpus
    0.351 are both stated in it. Before this column a `storage: raw` set could not produce it --
    `dirs` is unit(act) there -- so a Δ mixed a doubly-centred MAEMM cosine with a singly-centred
    corpus one, and its magnitude meant nothing even though its sign did.

    Checked against an independent einsum on the SAME arrays, and required to DIFFER from the
    other two: a `cos_asym` that equalled `cos` would be the bug this exists to prevent.
    """
    torch.manual_seed(20260921)
    n, t, d = 3, 5, 16
    h = torch.randn(n, t, d)
    act = torch.randn(n, d)
    mu = torch.randn(d) * 0.4
    dirs = torch.nn.functional.normalize(act, dim=-1)
    dirs_c = torch.nn.functional.normalize(act - mu, dim=-1)
    f = torch.nn.functional.normalize
    cos = torch.einsum("btd,bd->bt", f(h, dim=-1), dirs)
    cos_c = torch.einsum("btd,bd->bt", f(h - mu, dim=-1), dirs_c)
    cos_a = torch.einsum("btd,bd->bt", f(h, dim=-1), dirs_c)
    # each against a hand-rolled reference, elementwise
    for i in range(n):
        for j in range(t):
            hv, a = h[i, j], act[i]
            u = lambda x: x / x.norm()  # noqa: E731
            assert abs(float(cos[i, j]) - float(u(hv) @ u(a))) < 1e-5
            assert abs(float(cos_c[i, j]) - float(u(hv - mu) @ u(a - mu))) < 1e-5
            assert abs(float(cos_a[i, j]) - float(u(hv) @ u(a - mu))) < 1e-5
    assert (cos - cos_a).abs().max() > 1e-3, "cos_asym must not collapse onto cos"
    assert (cos_c - cos_a).abs().max() > 1e-3, "cos_asym must not collapse onto cos_centred"
    # and the scan computes EXACTLY one of the two, by mode: without --centre the window side is
    # raw and the scan IS cos_asym; with --centre it subtracts the SAME scoring constant from the
    # window that `dirs_for` subtracted from the target, which is cos_centred.
    src = (Path(__file__).resolve().parent / "scan.py").read_text()
    assert "hc = h if wmu is None else h - wmu" in src, (
        "scan.py no longer selects its window side as `h` / `h - wmu`; cos_asym is defined to "
        "match the first and cos_centred the second, and the three must move together"
    )
    assert "cos = torch.nn.functional.normalize(hc, dim=-1) @ v.T" in src, (
        "scan.py no longer scores `normalize(hc) @ v.T`"
    )
    # AND the production expression itself, pinned by source. The arithmetic above is this
    # module's own einsum, so mutating `common.score_block`'s `cos_a` line does not move it --
    # that mutation SURVIVED the first version of this check. Scoring for real needs a model and
    # a tokenizer, so the shipped formula is pinned textually instead: `h` UNCENTRED against the
    # centred target. A source pin is weaker than an execution test and is here because the
    # execution test is not affordable in this file; it catches exactly the regression that the
    # einsum above cannot.
    csrc = (Path(__file__).resolve().parent / "common.py").read_text()
    assert 'cos_a = torch.einsum("btd,bd->bt", F.normalize(h.float(), dim=-1), dc)' in csrc, (
        "common.score_block's cos_asym must be normalize(h) against the CENTRED target dc -- "
        "`h.float() - mu_t` there would make it a second copy of cos_centred, and every Delta "
        "built on it would silently be the mismatched one again"
    )

    # score.py must ask for the column whenever it asks for the centred one
    ssrc = (Path(__file__).resolve().parent / "score.py").read_text()
    assert '["cos_centred", "cos_asym"]' in ssrc, (
        "score.py must request cos_asym alongside cos_centred: they share the `dirs_centred` "
        "gate, and one without the other is a scores directory that cannot be differenced"
    )


CHECKS = [
    check_three_cosines,
    check_sae_key_for_rows,
    check_scan_masks_on_label,
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
    check_bo_ladder,
    check_parse_rows_and_gen_seed,
    check_outdir_keep_existing_and_section,
    check_two_writers_into_one_product,
    check_additive_removes_nothing_and_the_legacy_path_is_the_hazard,
    check_rollout_chunk_stem_and_read,
    check_nla_min_new_override,
    check_sha256_of_index,
    check_vllm_finish_ids,
    check_rename_lora_keys,
    check_maemms_for,
    check_strip_repo_sink,
    check_sae_key_for,
    check_storage_contract,
    check_no_mean_reaches_a_row_without_a_raw_activation,
    check_unit_set_is_served_as_shipped,
    check_two_cosines,
    check_sae_key_selector,
    check_family_kinds_table,
    check_exact_solve_roundtrip,
    check_heldout_v3_recovery,
    check_set_on_disk,
    check_heldout_v3_ours_block,
    check_csr_gate_floor,
    check_sae_column_reader,
    check_sae_self_side_flag,
    check_spawn_mirrors_main,
    check_autointerp_main_forwards_every_flag,
    check_nla_arm_a_reads_the_body,
    check_sae_column_slice_is_the_dictionary,
    check_gcg_never_loads_the_full_dictionary,
    check_scores_dir_puts_the_tag_where_rollout_stem_does,
    check_both_tag_axes_reach_the_scores_path,
    check_return_arities,
    check_centred_uses_one_mu,
    check_every_set_writer_writes_the_contract,
    check_draws_call_finish_with_its_signature,
    check_no_draw_stamps_a_bare_sae_key,
    check_one_set_writers_tuple,
    check_corpus_axis,
    check_rollouts_nla_selftest,
    check_byte_tables,
    check_token_covariates_czech,
    check_token_covariates_thai,
    check_token_covariates_python,
    check_script_fraction,
    check_arm_perm_and_rng,
    check_ood_config,
    check_span_in_corpus,
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
