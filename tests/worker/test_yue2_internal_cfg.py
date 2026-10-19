# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Internal CFG uses native rows, one sample and parent-only client outputs."""

import pytest
import torch
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import SamplerOutput
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.model_executor.models.yue2.request import SEMANTIC_ARGS, prepare_song
from vllm_omni.model_executor.models.yue2.runtime import Yue2Backbone, Yue2ForCausalLM, Yue2Scheduler, Yue2SchedulerBase
from vllm_omni.model_executor.models.yue2.yue2 import ABC_END, CODEC_OFFSET, CODEC_SIZE, MUSIC_END, MUSIC_START
from vllm_omni.request import OmniRequest
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def encode(text):
    return list(text.encode())


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1, 21, True, "2"])
def test_invalid_guidance(scale):
    with pytest.raises(ValueError, match="cfg_scale"):
        prepare_song(encode, "style", "lyrics", cfg_scale=scale)


@pytest.mark.parametrize("cot,abc", [("off", None), ("full", "CDEF"), ("melody", None)])
def test_internal_negative_uses_native_registry_and_exact_abc(monkeypatch, mocker, cot, abc):
    prompt, params = prepare_song(encode, "style", "lyrics", cot=cot, abc=abc, cfg_scale=1.5)
    scheduler = object.__new__(Yue2Scheduler)
    scheduler._pairs, scheduler._request_pairs, scheduler.requests = {}, {}, {}
    scheduler._internal_ids = set()
    scheduler._rejected = {}
    scheduler.scheduler_config = mocker.Mock(spec=SchedulerConfig, enable_chunked_prefill=False)
    scheduler.max_model_len = 24576
    scheduler.block_size = 16
    scheduler.max_num_running_reqs = scheduler.max_num_active_reqs = 4
    scheduler.max_num_scheduled_tokens = 24576
    scheduler._capacity = 4096
    monkeypatch.setattr(OmniARScheduler, "add_request", lambda self, r: self.requests.update({r.request_id: r}))
    request = OmniRequest("song", prompt["prompt_token_ids"], params, None)
    scheduler.add_request(request)
    if cot == "melody":
        assert list(scheduler.requests) == ["song"]
        request.yue2_abc_ids = [42, 43]
        request.sampling_params.extra_args = params.extra_args[SEMANTIC_ARGS]
        scheduler._add_negative(request)
    scheduler._add_negative(request)
    assert set(scheduler.requests) == {"song", "song-yue2-negative"}
    child = scheduler.requests["song-yue2-negative"]
    assert child.sampling_params.extra_args["yue2_cfg_parent"] == "song"
    assert child.sampling_params.extra_args["yue2_skip_synthesis"]
    assert "yue2_cfg_parent" not in request.sampling_params.extra_args
    assert child.sampling_params.extra_args is not request.sampling_params.extra_args
    if cot == "melody":
        assert child.prompt_token_ids[-4:] == [42, 43, ABC_END, MUSIC_START]
    if abc is not None:
        assert child.prompt_token_ids[-6:] == encode(abc) + [ABC_END, MUSIC_START]
    assert child.max_tokens == request.max_tokens
    monkeypatch.setattr(
        OmniARScheduler, "finish_requests", lambda self, ids, status: [self.requests.pop(i) for i in ids]
    )
    stopped = scheduler.finish_requests("song", RequestStatus.FINISHED_ABORTED)
    assert [r.request_id for r in stopped] == ["song"]
    assert not scheduler.requests and not scheduler._pairs


def test_guidance_precedes_one_native_sample_and_copies_token(monkeypatch):
    model = Yue2ForCausalLM.__new__(Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._step_rows = [("negative", 5, 1), ("parent", 20, 1)]
    model._cfg_step = [{"yue2_cfg_parent": "parent", "yue2_cfg_scale": 1.5}, {}]
    model._step_discard = [False, False]
    logits = torch.full((2, CODEC_OFFSET + CODEC_SIZE), -torch.inf, dtype=torch.bfloat16)
    logits[0, MUSIC_END:] = 0.25
    logits[1, MUSIC_END:] = 0.5

    def sample(self, guided, metadata):
        assert self._step_discard == [True, False]
        assert (guided[1, MUSIC_END:] == 0.625).all()
        assert torch.isneginf(guided[:, :MUSIC_END]).all()
        return SamplerOutput(sampled_token_ids=torch.tensor([[0], [123]], dtype=torch.int32), logprobs_tensors=None)

    monkeypatch.setattr(Yue2Backbone, "sample", sample)
    result = model.sample(logits, None)
    assert result.sampled_token_ids.tolist() == [[123], [123]]
    assert model._step_discard == [False, False]
    model._step_rows[1] = ("missing", 20, 1)
    with pytest.raises(RuntimeError, match="without its parent"):
        model.sample(logits, None)


def test_child_outputs_hidden_but_worker_cleanup_preserved(monkeypatch, mocker):
    scheduler = object.__new__(Yue2Scheduler)
    scheduler._internal_ids = {"negative"}
    scheduler._rejected = {}
    scheduler.requests = {}
    scheduler.finished_req_ids = {"negative", "parent"}
    batch = EngineCoreOutputs(
        outputs=[OmniEngineCoreOutput(request_id=i, new_token_ids=[]) for i in ("negative", "parent")],
        finished_requests={"negative", "parent"},
    )
    monkeypatch.setattr(Yue2SchedulerBase, "update_from_output", lambda *args: {0: batch})
    scheduler.update_from_output(mocker.Mock(spec=OmniSchedulerOutput, num_scheduled_tokens={}), None)
    assert [o.request_id for o in batch.outputs] == ["parent"]
    assert batch.finished_requests == {"parent"}
    assert scheduler.finished_req_ids == {"negative", "parent"}
    assert not scheduler._internal_ids


@pytest.mark.asyncio
async def test_http_cfg_detection_and_sampling(mocker):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    assert detect_tts_model_type("yue2", "Yue2ForCausalLM") == "yue2"
    ctx = SpeechServingContext(
        server=None,
        engine_client=mocker.Mock(
            spec=AsyncOmni,
            vllm_config=mocker.Mock(spec=VllmConfig, model_config=mocker.Mock(spec=ModelConfig, max_model_len=16384)),
        ),
    )
    adapter = Yue2Adapter(ctx)
    adapter._cached_tokenizer = mocker.Mock(spec=YuE2TextTokenizer, encode=encode)
    request = OpenAICreateSpeechRequest(
        input="lyrics",
        instructions="style",
        seed=42,
        max_new_tokens=200,
        extra_params={"cot": "full", "cfg_scale": 1.5},
    )
    assert adapter.validate(request) is None
    built = await adapter.build(request, [], False)
    semantic = built.prompt["additional_information"]["yue2_song_sampling"][SEMANTIC_ARGS]
    assert semantic["yue2_cfg_scale"] == 1.5 and semantic["yue2_cfg_needs_abc"]


def test_existing_model_and_pipeline_names_resolve_to_unified_path():
    from importlib import import_module
    from pathlib import Path

    import yaml

    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert [key for key in OMNI_PIPELINES if key.startswith("yue2")] == ["yue2"]
    assert [key for key in _OMNI_MODELS if key.startswith("Yue2")] == ["Yue2ForCausalLM"]
    assert _OMNI_MODELS["Yue2ForCausalLM"] == ("yue2", "runtime", "Yue2ForCausalLM")
    pipeline = OMNI_PIPELINES["yue2"]
    assert pipeline.default_deploy_config_name == "yue2.yaml"
    module, name = pipeline.stages[0].scheduler_cls.rsplit(".", 1)
    assert getattr(import_module(module), name) is Yue2Scheduler
    deploy = yaml.safe_load((Path(__file__).parents[2] / "vllm_omni/deploy/yue2.yaml").read_text())
    assert deploy["pipeline"] == "yue2"
    module, name = deploy["stages"][0]["worker_cls"].rsplit(".", 1)
    worker = getattr(import_module(module), name)
    from vllm_omni.model_executor.models.yue2.runtime import Yue2ModelRunner

    assert worker.model_runner_cls is Yue2ModelRunner
    family, module, name = _OMNI_MODELS["Yue2ForCausalLM"]
    assert getattr(import_module(f"vllm_omni.model_executor.models.{family}.{module}"), name) is Yue2ForCausalLM
    assert detect_tts_model_type("yue2", "Yue2ForCausalLM") == "yue2"


@pytest.mark.parametrize("paired", [False, True])
def test_unpaired_admission_stays_native(monkeypatch, paired):
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    from vllm_omni.core.sched.omni_cfg_ar_scheduler import OmniCFGARScheduler

    scheduler = object.__new__(Yue2Scheduler)
    scheduler._pairs = {"pair": {}} if paired else {}
    scheduler._rejected = {}
    scheduler.requests = {}
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    called = []

    def advance(name):
        def run(self, throttle_prefills=False):
            called.append(name)
            return SchedulerOutput.make_empty()

        return run

    monkeypatch.setattr(OmniARScheduler, "schedule", advance("native"))
    monkeypatch.setattr(OmniCFGARScheduler, "schedule", advance("atomic"))
    scheduler.schedule()
    assert called == ["atomic" if paired else "native"]


@pytest.mark.parametrize("limit", ["slots", "prefill", "kv", "chunked"])
def test_unfit_cfg_returns_request_error_without_admitting_negative(monkeypatch, mocker, limit):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
    from vllm.v1.engine import FinishReason

    prompt, params = prepare_song(encode, "style", "lyrics", cfg_scale=1.5)
    request = OmniRequest("bad", prompt["prompt_token_ids"], params, None)
    scheduler = object.__new__(Yue2Scheduler)
    scheduler._pairs, scheduler._request_pairs = {}, {}
    scheduler._internal_ids, scheduler._rejected = set(), {}
    scheduler.requests = {"bad": request}
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.waiting.add_request(request)
    scheduler.scheduler_config = mocker.Mock(spec=SchedulerConfig, enable_chunked_prefill=limit == "chunked")
    scheduler.max_model_len, scheduler.block_size = 24576, 16
    scheduler.max_num_running_reqs = scheduler.max_num_active_reqs = 1 if limit == "slots" else 4
    scheduler.max_num_scheduled_tokens = 1 if limit == "prefill" else 24576
    scheduler._capacity = 1 if limit == "kv" else 4096
    scheduler._add_negative(request)
    assert not scheduler._pairs and not scheduler._internal_ids
    assert list(scheduler._rejected) == ["bad"]

    def finish(self, ids, status):
        assert set(ids) == {"bad"} and status == RequestStatus.FINISHED_ERROR
        request.status = status
        self.requests.pop("bad")
        self.waiting.remove_requests([request])
        return [request]

    monkeypatch.setattr(OmniARScheduler, "finish_requests", finish)
    monkeypatch.setattr(Yue2SchedulerBase, "update_from_output", lambda *args: {})
    output = scheduler.update_from_output(mocker.Mock(spec=OmniSchedulerOutput, num_scheduled_tokens={}), None)
    assert output[0].outputs[0].finish_reason == FinishReason.ERROR
    assert "capacity" in output[0].outputs[0].stop_reason
    assert not scheduler.requests and not scheduler._rejected and not scheduler.waiting


def test_cfg1_sample_calls_original_sampler_directly(monkeypatch):
    model = Yue2ForCausalLM.__new__(Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._cfg_step = [{}, None]
    expected = SamplerOutput(torch.tensor([[2], [3]]), None)
    monkeypatch.setattr(Yue2Backbone, "sample", lambda *args: expected)
    assert model.sample(None, None) is expected
