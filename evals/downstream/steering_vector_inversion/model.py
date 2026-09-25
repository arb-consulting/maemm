"""The model worker: the clean base and the inverter on one card, clean reads, generation, and the preflight.

Loading, injection and generation are `evals/downstream/common/model_io.py`; this module is the package's job shape.
`self.base` is the clean Qwen3.6-27B: every read, the corpus search, the lens, `base_l1` and the steered
model run on it. `self.inverter` (the full-parameter fine-tune under evaluation) only generates MAEM's
arm, and is loaded on first use so a worker that only reads never holds it. The NLA verbalizer has its
own worker (`nla.NlaWorker`).
"""
import time

import numpy as np
import torch

from evals.downstream.common import model_io, plain_steer
from evals.downstream.common.model_io import addition_hook

from maem.inject import get_layer, hooked, make_inject_hook, read_resid

from . import corpus, lens, nla
from .config import NORM_FILTER_MULT, PREFLIGHT_REPEATS, PREFLIGHT_RTOL, PREFLIGHT_WARMUP

#: Which model a generation job runs on, as its `model` key names it.
MODELS = ('inverter', 'base')


def research_prompt(tokenizer):
    from maem.config import READ_LAYER
    from maem.prompts import build_prompt_ids, marker_positions
    if READ_LAYER != 42:
        raise ValueError('The upstream research prompt must describe layer 42')
    ids, positions = build_prompt_ids(tokenizer)
    if positions != [len(ids)-1] or marker_positions(tokenizer, ids) != positions:
        raise ValueError('The trained marker must be the unique final prompt token')
    return ids


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    result = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=add_generation_prompt, enable_thinking=False)
    ids = result['input_ids'] if hasattr(result, 'keys') else result
    while ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError('Expected a single nonempty tokenized chat')
    return list(ids)


class CleanReads:
    """The preflight's clean-read check: the same sources read `repeats` times on the clean base, and again
    after the base and the inverter have generated (`later`). The first read after `warmup` is the
    reference; every later read must be within `rtol` x the reference mean's L2 norm, per source. Whether
    the reads were exact is recorded beside the verdict and decides nothing."""

    def __init__(self, read, repeats=PREFLIGHT_REPEATS, warmup=PREFLIGHT_WARMUP, rtol=PREFLIGHT_RTOL):
        if not 0 <= warmup < repeats:
            raise ValueError('The clean-read series needs a read after its warm-up')
        self.read, self.repeats, self.warmup, self.rtol = read, repeats, warmup, rtol
        self.reference, self.previous, self.series, self.later_reads = None, None, [], {}

    def _read(self):
        reads = self.read()
        if any(r['status'] != 'ok' for r in reads):
            raise ValueError('Preflight source read failed')
        return reads

    @staticmethod
    def difference(name, left, right):
        """The largest absolute difference between two reads' mean activations and the largest relative
        one -- a source's absolute difference over the L2 norm of `left`'s mean -- and whose each is."""
        each = [float(np.max(np.abs(b['mean']-a['mean']))) for a, b in zip(left, right, strict=True)]
        norms = [float(np.linalg.norm(a['mean'])) for a in left]
        relative = [d/n if n else (0. if d == 0 else float('inf')) for d, n in zip(each, norms)]
        # a NaN is the maximum, so it can never read as 0 or as within the line
        worst, worst_rel = int(np.argmax(each)), int(np.argmax(relative))
        return {name: each[worst], f'{name}_record': worst,
                f'{name}_rel': relative[worst_rel], f'{name}_rel_record': worst_rel}

    def repeat(self):
        """The back-to-back series. Returns the reference read."""
        reads = [self._read() for _ in range(self.repeats)]
        self.reference, self.previous = reads[self.warmup], reads[-1]
        for k, this in enumerate(reads):
            self.series.append(
                {'repeat': k, 'warmup': k < self.warmup} | self.difference('to_first', reads[0], this)
                | (self.difference('to_previous', reads[k-1], this) if k else
                   dict.fromkeys(('to_previous', 'to_previous_record', 'to_previous_rel',
                                  'to_previous_rel_record')))
                | self.difference('to_reference', self.reference, this))
        return self.reference

    def later(self, name):
        """One more read, after something else has run on the card. Returns it."""
        this = self._read()
        self.later_reads[name] = (self.difference('to_reference', self.reference, this)
                                  | self.difference('to_previous', self.previous, this))
        self.previous = this
        return this

    def record(self):
        warm = [row for row in self.series if row['warmup']]
        judged = [row for row in self.series if not row['warmup']] + list(self.later_reads.values())
        passed = all(row['to_reference_rel'] <= self.rtol for row in judged)
        inexact = sum(row['to_reference'] != 0 for row in judged)
        return {'status': 'passed' if passed else 'failed_clean_read_consistency',
                'exact': inexact == 0, 'n_inexact_reads': inexact,
                'max_relative_difference': float(np.max([row['to_reference_rel'] for row in judged])),
                'warmup_differences': any(row['to_reference'] != 0 for row in warm),
                'clean_reads': {'repeats': self.repeats, 'warmup': self.warmup, 'rtol': self.rtol,
                                'reference_repeat': self.warmup, 'series': self.series,
                                'later': self.later_reads}}


class ModelWorker:
    role = 'model'

    def __init__(self, config, cache_dir, device="cuda:0"):
        from transformers import GenerationConfig
        self.config, self.device, self.cache_dir = config, torch.device(device), cache_dir
        self.versions = nla.runtime_versions()
        self.base, self.tokenizer = model_io.load_base(str(self.device), config.model,
                                                       config.model_revision,
                                                       cache_dir=str(cache_dir), token=False)
        # Every snippet is read behind an EOS sink, so a tokenizer without one cannot score at all.
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Standalone scoring requires an EOS sink")
        self.base.requires_grad_(False)
        self._inverter = None
        if get_layer(self.base, config.read_layer) is None:
            raise ValueError("Missing read layer")
        self.prompt_ids = research_prompt(self.tokenizer)
        self.generation_config = GenerationConfig.from_pretrained(
            config.model, revision=config.model_revision, cache_dir=str(cache_dir), token=False)
        self.eos_ids = model_io.stop_token_ids(self.tokenizer, self.generation_config)
        text_config = getattr(self.base.config, "text_config", self.base.config)
        self.max_context = text_config.max_position_embeddings
        if text_config.hidden_size != config.hidden_size:
            raise ValueError("Model width does not match the specified activation space")

    @property
    def inverter(self):
        """The model under evaluation, loaded the first time a job generates on it, with the base's
        tokenizer; its generation config must agree with the base's field by field."""
        if self._inverter is None:
            model = model_io.load_inverter(str(self.device), self.config.inverter,
                                           self.config.inverter_revision,
                                           cache_dir=str(self.cache_dir), token=False)
            model.requires_grad_(False)
            model_io.refuse_unless_generation_agrees(self.generation_config, model)
            self._inverter = model
        return self._inverter

    def close(self):
        """Give both checkpoints back, for a process that is about to host the verbalizer's worker."""
        self.base = model_io.free_model(self.base)
        if self._inverter is not None:
            self._inverter = model_io.free_model(self._inverter)

    def gen_model(self, name):
        """The model a generation job's `model` key names."""
        if name not in MODELS:
            raise ValueError(f"A generation job runs on {' or '.join(MODELS)}, not {name!r}")
        return self.inverter if name == "inverter" else self.base

    def _batch(self, sequences, side="right"):
        return model_io.pad_batch(sequences, self.tokenizer.pad_token_id, self.device, side)

    @torch.inference_mode()
    def read(self, records):
        """One clean-base read per record, behind an EOS sink: the mean layer-42 activation over the tokens
        whose residual norm is at most `NORM_FILTER_MULT` x the text's median (`n_retained` of `n_tokens`)."""
        output = [None] * len(records)
        rows = []
        for index, row in enumerate(records):
            content = row.get("token_ids")
            if content is None:
                content = self.tokenizer.encode(row["text"], add_special_tokens=False)
            if not content or not row.get("text", "").strip():
                output[index] = {"status": "unavailable", "reason": "empty_snippet"}
                continue
            ids = [self.tokenizer.eos_token_id, *content]
            if len(ids) > self.max_context:
                output[index] = {"status": "unavailable", "reason": "full_text_exceeds_context"}
                continue
            rows.append((index, ids))
        rows.sort(key=lambda row: (len(row[1]), row[0]))
        for offset in range(0, len(rows), self.config.read_batch_size):
            selected = rows[offset:offset + self.config.read_batch_size]
            batch = self._batch([r[1] for r in selected])
            h, _ = read_resid(self.base, self.config.read_layer, batch, pool="all")
            for i, (index, ids) in enumerate(selected):
                tokens = h[i, 1:len(ids)].float()
                if not torch.isfinite(tokens).all():
                    output[index] = {"status": "unavailable", "reason": "nonfinite_activations"}
                    continue
                norms = tokens.norm(dim=-1)
                kept = tokens[(norms <= NORM_FILTER_MULT * norms.median()) & (norms > 0)]
                if not len(kept):
                    output[index] = {"status": "unavailable", "reason": "no_retained_tokens"}
                    continue
                output[index] = {"status": "ok", "n_tokens": len(ids) - 1, "n_retained": len(kept),
                                 "mean": kept.mean(0).cpu().numpy()}
            del h, batch
        return output

    @torch.inference_mode()
    def generate(self, jobs):
        """One batch of inversion jobs sharing a model and a decoding mode: the research prompt, the direction
        injected at the marker on `inject_layer`, one seed per row (none when greedy)."""
        if not jobs:
            return []
        which, greedy = jobs[0]["model"], jobs[0]["greedy"]
        if any((j["kind"], j["model"], j["greedy"]) != ("inversion", which, greedy) for j in jobs):
            raise ValueError("A generation batch must be inversion jobs sharing a model and decoding mode")
        sequences = [self.prompt_ids] * len(jobs)
        maximum = self.config.max_new_tokens
        suffixes = model_io.generate_rows(
            self.gen_model(which), self.tokenizer, sequences,
            np.array([j["direction"] for j in jobs]), self.device, "marker", self.config.inject_layer,
            coefficients=None, seeds=None if greedy else [j["seed"] for j in jobs], greedy=greedy,
            max_new=maximum, min_new=self.config.min_new_tokens, eos_ids=self.eos_ids)
        result = []
        for job, ids, prompt in zip(jobs, suffixes, sequences):
            stop = next((i for i, t in enumerate(ids) if t in self.eos_ids), None)
            raw_ids = ids if stop is None else ids[:stop + 1]
            raw_text = self.tokenizer.decode(raw_ids, skip_special_tokens=True)
            view_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)[:maximum]
            result.append({k: v for k, v in job.items() if k != "direction"} | {
                "status": "ok", "raw_text": raw_text, "raw_token_ids": raw_ids,
                "text": self.tokenizer.decode(view_ids, skip_special_tokens=False), "token_ids": view_ids,
                "prompt_token_ids": prompt, "eos_terminated": stop is not None,
                "length_capped": stop is None and len(raw_ids) >= maximum})
        return result

    def execute(self, payload):
        started = time.time()
        operation = payload["operation"]
        if operation == "generate":
            data = self.generate(payload["jobs"])
        elif operation == "read":
            data = self.read(payload["records"])
        elif operation == plain_steer.OPERATION:
            result = plain_steer.generate(self, payload)
            data = {k: result[k] for k in ("rows", "sink_id", "sink_token", "eos_ids", "read_layer")}
        elif operation == corpus.OPERATION:
            data = corpus.execute(self, payload)
        elif operation == nla.OPERATION:
            nla.refuse_on_model_worker(operation)
        elif operation == lens.OPERATION:
            data = lens.execute(self, payload)
        elif operation == 'preflight':
            data = self.preflight(payload['records'])
        else:
            raise ValueError(f"Unknown GPU operation {operation}")
        return {"data": data, "gpu_seconds": time.time() - started,
                "runtime_versions": self.versions,
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"}

    @torch.inference_mode()
    def preflight(self, records):
        """A hardware check before any benchmark work: the clean-read series (`CleanReads`), the layer-1
        marker injection and the layer-42 addition checked against their formulas, a factor-zero addition
        equal to the unhooked base, and one inverter row followed by more clean reads. The hook checks
        assert; the status is the clean reads'."""
        from transformers import GenerationConfig
        def backend_flags():
            return {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                    'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'matmul_precision': torch.get_float32_matmul_precision(),
                    'training_modules': sum(m.training for m in self.base.modules()),
                    'forward_hooks': sum(len(m._forward_hooks) for m in self.base.modules())}
        initial_flags = backend_flags()
        # The clean-read series comes first, on a card nothing else has run on: what it records is how
        # the first reads of a container compare with its later ones. `reads` is its reference.
        clean_reads = CleanReads(lambda: self.read(records))
        reads = clean_reads.repeat()
        v = torch.as_tensor(reads[0]['mean'] - reads[1]['mean'], device=self.device)
        if not torch.isfinite(v).all() or v.norm() == 0:
            raise ValueError('Preflight contrast is unusable')
        v = torch.nn.functional.normalize(v, dim=0)[None]
        batch = self._batch([self.prompt_ids])
        clean, _ = read_resid(self.base, self.config.read_layer, batch, pool='all')
        norm_hook = make_inject_hook([v], [[len(self.prompt_ids)-1]], 1., self.device, torch.bfloat16)
        norm_checks = []
        def checked_norm(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            before = h.clone()
            changed = norm_hook(module, inputs, output)
            after = changed[0] if isinstance(changed, tuple) else changed
            expected = before.clone()
            direction = torch.nn.functional.normalize(v.to(h.dtype), dim=-1)
            expected[:, -1] += direction * before[:, -1].norm(dim=-1, keepdim=True)
            norm_checks.append(bool(torch.equal(after, expected)))
            return changed
        with hooked(get_layer(self.base, self.config.inject_layer), checked_norm):
            injected, _ = read_resid(self.base, self.config.read_layer, batch, pool='all')
        l1_change = float((injected-clean).abs().max())
        assert norm_checks == [True] and l1_change > 0, 'Layer-1 injection did not match its formula'
        settings = GenerationConfig(do_sample=False, min_new_tokens=3, max_new_tokens=4,
            eos_token_id=self.eos_ids, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        plain = self.base.generate(**batch, generation_config=settings)
        zero_hook = addition_hook(v, torch.zeros(1, device=self.device), batch['attention_mask'].bool())
        with hooked(get_layer(self.base, self.config.read_layer), zero_hook):
            zero = self.base.generate(**batch, generation_config=settings)
        assert torch.equal(plain, zero), 'Factor-zero generation differs from the unhooked base'
        policies = {}
        for policy in ('response_only', 'prompt_and_response'):
            mask = batch['attention_mask'].bool() if policy == 'prompt_and_response' else torch.zeros_like(batch['input_ids'], dtype=torch.bool)
            if policy == 'response_only':
                mask[:, -1] = True
            calls = []
            coefficient = torch.tensor([10.], device=self.device)
            hook = addition_hook(v, coefficient, mask)
            def checked_addition(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                before = h.clone()
                expected_mask = mask if not calls else torch.ones(h.shape[:2], dtype=torch.bool, device=self.device)
                changed = hook(module, inputs, output)
                after = changed[0] if isinstance(changed, tuple) else changed
                expected = before + expected_mask[:, :, None] * (v.float()*coefficient[:, None]).to(h.dtype)[:, None]
                assert torch.equal(after, expected), 'Layer-42 addition did not match its position policy'
                calls.append(h.shape[1])
                return changed
            with hooked(get_layer(self.base, self.config.read_layer), checked_addition):
                self.base.generate(**batch, generation_config=settings)
            assert calls[0] == len(self.prompt_ids) and len(calls) >= 3 and all(n == 1 for n in calls[1:]), 'Missing cached decoding checks'
            policies[policy] = calls
        clean_reads.later('after_base')
        base_flags = backend_flags()
        # One row on the inverter, then the same sources read again: generation on the second checkpoint
        # must leave no active injection hook behind and must not move what the clean base reads.
        job = dict(concept_id=-1, condition='preflight', sample_id=-1, greedy=True,
                   kind='inversion', model='inverter', direction=v[0].cpu().numpy(), seed=0)
        sample = self.generate([job])[0]
        clean_reads.later('after_inverter')
        clean_reads.later('final')
        return clean_reads.record() | {
            'backend_flags': {'initial': initial_flags, 'after_base': base_flags, 'final': backend_flags()},
            'source_read_counts': [r['n_tokens'] for r in reads],
            'layer1_formula_matches': True, 'layer1_downstream_max_abs_change': l1_change,
            'factor_zero_matches_unhooked': True, 'layer42_forward_lengths': policies,
            'research_prompt_token_ids': self.prompt_ids,
            'inverter_sample': sample, 'peak_allocated_gb': torch.cuda.max_memory_allocated(self.device)/1e9}
