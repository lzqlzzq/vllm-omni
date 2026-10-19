# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One native request owns ABC, semantic and acoustic completion."""

import pytest
from vllm import SamplingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.request import Request, RequestStatus

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.model_executor.models.yue2.request import AUTO_ABC, SEMANTIC_ARGS, prepare_song
from vllm_omni.model_executor.models.yue2.runtime import Yue2SchedulerBase
from vllm_omni.model_executor.models.yue2.yue2 import ABC_END, ABC_SAMPLING, MUSIC_START, SEMANTIC_SAMPLING
from vllm_omni.tokenizers.yue2_prompt import abc_prefix_ids, semantic_prefix_ids
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def encode(text):
    return list(text.encode())


@pytest.mark.parametrize(
    "cot,abc", [("off", None), ("melody", "CDEF"), ("full", "CDEF"), ("melody", None), ("full", None)]
)
def test_song_prefix_and_independent_phase_presets(cot, abc):
    prompt, params = prepare_song(encode, "style", "lyrics", cot=cot, abc=abc, max_frames=200, max_abc_tokens=32)
    automatic = cot != "off" and abc is None
    expected = (
        abc_prefix_ids(encode, "style", "lyrics", cot)
        if automatic
        else semantic_prefix_ids(encode, "style", "lyrics", cot, None if abc is None else encode(abc))
    )
    assert prompt["prompt_token_ids"] == expected
    args = params.extra_args
    assert bool(args.get(AUTO_ABC)) == automatic
    assert params.max_tokens == (32 if automatic else 202)
    assert args["yue2_temperature"] == (ABC_SAMPLING if automatic else SEMANTIC_SAMPLING)["temperature"]
    if automatic:
        assert args[SEMANTIC_ARGS]["yue2_temperature"] == SEMANTIC_SAMPLING["temperature"]
        assert args[SEMANTIC_ARGS]["yue2_seed"] == args["yue2_seed"]
        assert args[SEMANTIC_ARGS]["yue2_max_audio_frames"] == 200
        assert not args[SEMANTIC_ARGS]["yue2_skip_synthesis"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cot": "bad"},
        {"abc": "CDEF"},
        {"cot": "full", "abc": ""},
        {"max_frames": True},
        {"max_abc_tokens": 0},
        {"seed": -1},
        {"cot": "full", "max_model_len": 100},
    ],
)
def test_bad_song_rejected_before_admission(kwargs):
    with pytest.raises(ValueError):
        prepare_song(encode, "style", "lyrics", **kwargs)


@pytest.mark.parametrize("eos", [True, False])
def test_abc_tokens_are_hidden_and_budget_rolls_to_semantic(eos):
    prompt, params = prepare_song(encode, "style", "lyrics", cot="full", max_abc_tokens=2)
    request = Request("song", prompt["prompt_token_ids"], params, None)
    scheduler = object.__new__(Yue2SchedulerBase)
    scheduler.max_model_len = 24576
    assert scheduler._update_request_with_output(request, [42]) == ([], False)
    assert not hasattr(request, "yue2_next_prefix")
    assert scheduler._update_request_with_output(request, [ABC_END if eos else 43]) == ([], False)
    abc = [42] if eos else [42, 43]
    assert request.yue2_abc_ids == abc
    assert request.yue2_abc_truncated == (not eos)
    assert request.yue2_next_prefix == prompt["prompt_token_ids"] + abc + [ABC_END, MUSIC_START]
    assert not request.is_finished()


def test_invalid_or_overlong_abc_fails_only_request():
    for tokens, context in (([ABC_END], 24576), ([MUSIC_START], 24576), ([42], 1)):
        prompt, params = prepare_song(encode, "style", "lyrics", cot="full", max_abc_tokens=1)
        request = Request("song", prompt["prompt_token_ids"], params, None)
        scheduler = object.__new__(Yue2SchedulerBase)
        scheduler.max_model_len = context
        assert scheduler._update_request_with_output(request, tokens) == ([], True)
        assert request.status == RequestStatus.FINISHED_ERROR


@pytest.mark.asyncio
async def test_http_existing_adapter_accepts_automatic_abc(mocker):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    ctx = SpeechServingContext(
        server=None,
        engine_client=mocker.Mock(
            spec=AsyncOmni,
            vllm_config=mocker.Mock(spec=VllmConfig, model_config=mocker.Mock(spec=ModelConfig, max_model_len=24576)),
        ),
    )
    request = OpenAICreateSpeechRequest(
        input="lyrics",
        instructions="style",
        seed=42,
        max_new_tokens=200,
        extra_params={"cot": "full", "abc_max_tokens": 32},
    )
    adapter = Yue2Adapter(ctx)
    assert detect_tts_model_type("yue2", "Yue2ForCausalLM") == "yue2"
    adapter._cached_tokenizer = mocker.Mock(spec=YuE2TextTokenizer, encode=encode)
    assert adapter.validate(request) is None
    defaults = [SamplingParams(extra_args={"unchanged": True})]
    built = await adapter.build(request, defaults, False)
    params = adapter.apply_sampling_overrides(defaults, request, built.prompt)
    assert params[0].extra_args[AUTO_ABC] and params[0].extra_args["yue2_seed"] == 42
    assert params[0].max_tokens == 32
    assert defaults[0].extra_args == {"unchanged": True}
    request.extra_params["cfg_scale"] = 1.01
    assert adapter.validate(request) is None


@pytest.mark.parametrize(
    "abc_truncated,semantic_truncated", [(True, False), (False, True), (True, True), (False, False)]
)
def test_speech_truncation_includes_either_phase(abc_truncated, semantic_truncated):
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    adapter = object.__new__(Yue2Adapter)
    metadata: dict[str, bool] = {}
    adapter.collect_response_metadata(
        {"meta.abc_truncated": abc_truncated, "meta.truncated": semantic_truncated}, metadata
    )
    assert metadata["audio_truncated"] == (abc_truncated or semantic_truncated)
