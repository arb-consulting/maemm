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
        _write_set(sdir, rows, act, {"storage": "raw", "mu_stored": None, "family_mu": {}})
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


def check_unit_set_refuses():
    """A stored `unit` set is served at ITS mean, refuses another, and LABELS an unknown one.

    The three outcomes are the whole design: a known mismatch is a wrong number waiting to happen
    and must raise; a match is the legacy path that has to keep reproducing every number measured
    before 2026-09-21; an `unknown` mean is Celeste's imported families, which are run as shipped
    with a label rather than refused (plan §1.4).
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

        sdir = root / "known"
        _write_set(sdir, rows, None, {"storage": "unit", "mu_stored": None,
                                      "family_mu": {"realact": str(mu_a), "sae": None}}, vecs=v)
        got = C.dirs_for(cfg, "tb", str(sdir), str(mu_a), str(root))
        assert np.abs(got - v).max() < 1e-3, "the set's own mean must return the stored rows"
        try:
            C.dirs_for(cfg, "tb", str(sdir), str(mu_b), str(root))
        except AssertionError as e:
            assert "cannot be re-centred" in str(e), f"wrong assert for a mismatched mean: {e}"
        else:
            raise AssertionError("dirs_for served a `storage: unit` set under the WRONG mean")

        udir = root / "unknown"
        _write_set(udir, rows, None, {"storage": "unit", "mu_stored": None,
                                      "family_mu": {"realact": C.MU_UNKNOWN, "sae": None}}, vecs=v)
        notes: list[str] = []
        got = C.dirs_for(cfg, "tb", str(udir), str(mu_b), str(root), notes)
        assert np.abs(got - v).max() < 1e-3, "an `unknown` mean must still return the stored rows"
        assert any(C.MU_UNKNOWN in n and "realact" in n for n in notes), (
            f"an `unknown` family must be LABELLED into the product README, got notes {notes}"
        )

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
            json.dump({"storage": "raw", "mu_stored": None, "sae_key": "qwen36-27b/sae2m"}, fh)
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
        for fam, mu in (spec.get("family_mu") or {}).items():
            assert mu is None or mu == C.MU_UNKNOWN or str(mu).endswith(C.MU_SUFFIXES), (
                f"heldout {set_name} family_mu[{fam}] = {mu!r} is not null, {C.MU_UNKNOWN!r} or a "
                f"{'/'.join(C.MU_SUFFIXES)} path -- a mu is a FILE, never a name"
            )
    for fam, fspec in cfg["family_kinds"].items():
        assert fspec["centrable"] == (fspec["kind"] == "activation"), fam
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
    ("precompute/scan.py", "_load_targets"): 3,
    ("precompute/centred.py", "_load_dirs"): 4,
    ("gcg/gcg.py", "_load_targets"): 2,
    ("autointerp/sae_self.py", "_sae_rows"): 4,
    ("precompute/common.py", "mu_for"): 2,
}


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
    # file only. That is not pedantry: `_load_targets` is `scan`'s (3 values) AND `gcg`'s (2), and
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
    check_sae_key_for,
    check_storage_contract,
    check_unit_set_refuses,
    check_two_cosines,
    check_sae_key_selector,
    check_family_kinds_table,
    check_exact_solve_roundtrip,
    check_spawn_mirrors_main,
    check_return_arities,
    check_centred_uses_one_mu,
    check_rollouts_nla_selftest,
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
