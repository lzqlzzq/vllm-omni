# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint locations and independent revisions for YuE2."""

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import LoadConfig, ModelConfig, VllmConfig

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter
from vllm_omni.model_executor.models.yue2 import yue2 as y
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer
from vllm_omni.transformers_utils.configs.yue2 import Yue2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tokenizer_local_directory(tmp_path, mocker):
    merge = tmp_path / "qwen.tiktoken"
    merge.write_text("test")
    init = mocker.patch.object(YuE2TextTokenizer, "__init__", return_value=None)
    YuE2TextTokenizer.from_pretrained(tmp_path)
    init.assert_called_once_with(str(merge))


def test_tokenizer_hf_revision(mocker):
    cached = mocker.patch("transformers.utils.hub.cached_file", return_value="/cache/qwen.tiktoken")
    mocker.patch.object(YuE2TextTokenizer, "__init__", return_value=None)
    YuE2TextTokenizer.from_pretrained("m-a-p/YuE2-3B", revision="model-sha", cache_dir="/cache", token=True)
    cached.assert_called_once_with(
        "m-a-p/YuE2-3B", "qwen.tiktoken", revision="model-sha", cache_dir="/cache", token=True
    )


def test_missing_local_tokenizer(tmp_path):
    with pytest.raises(OSError, match="qwen.tiktoken"):
        YuE2TextTokenizer.from_pretrained(tmp_path)


def test_adapter_uses_model_revision_not_placeholder_tokenizer(mocker):
    config = mocker.Mock(spec=ModelConfig, model="m-a-p/YuE2-3B", revision="model-sha", hf_token=True)
    engine = mocker.Mock(
        spec=AsyncOmni,
        vllm_config=mocker.Mock(spec=VllmConfig, model_config=config, load_config=LoadConfig(download_dir="/cache")),
    )
    adapter = Yue2Adapter(SpeechServingContext(server=None, engine_client=engine))
    load = mocker.patch.object(YuE2TextTokenizer, "from_pretrained")
    assert adapter._tokenizer() is adapter._tokenizer()
    load.assert_called_once_with("m-a-p/YuE2-3B", revision="model-sha", token=True, cache_dir="/cache")


@pytest.mark.parametrize(
    "path,legacy,expected",
    [
        ("m-a-p/YuE2-Vae", "/legacy", "m-a-p/YuE2-Vae"),
        ("/models/vae", "/legacy", "/models/vae"),
        (None, "/legacy", "/legacy"),
        (None, None, "m-a-p/YuE2-Vae"),
    ],
)
def test_model_vae_revision_and_credentials(path, legacy, expected, monkeypatch, mocker):
    if legacy is None:
        monkeypatch.delenv("YUE2_VAE", raising=False)
    else:
        monkeypatch.setenv("YUE2_VAE", legacy)
    model = object.__new__(y.Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = Yue2Config(yue2_vae_path=path, yue2_vae_revision="vae-sha")
    model.vllm_config = mocker.Mock(
        spec=VllmConfig,
        model_config=mocker.Mock(spec=ModelConfig, revision="different-model-sha", hf_token=True),
        load_config=LoadConfig(download_dir="/cache"),
    )
    model._vae = None
    model._vae_device = None
    vae = mocker.Mock(spec=y.YuE2VAE)
    load = mocker.patch.object(y.YuE2VAE, "from_pretrained", return_value=vae)
    assert model._vae_model(torch.device("cpu")) is vae
    load.assert_called_once_with(
        expected, revision="vae-sha", token=True, cache_dir="/cache", decoder_only=True, device="cpu"
    )
    assert model._vae_model(torch.device("cpu")) is vae
    assert load.call_count == 1


@pytest.mark.parametrize("from_hub", [False, True], ids=["local", "hf"])
def test_vae_checkpoint_loading(tmp_path, mocker, from_hub):
    config = y.YuE2VAEConfig(
        latent_dim=2,
        downsampling_ratio=2,
        decoder_config=dict(out_channels=2, channels=4, c_mults=[1], strides=[2], latent_dim=2),
    )
    original = y.YuE2VAE(config, decoder_only=True).float()
    config.save_pretrained(tmp_path)
    save_file(original.state_dict(), tmp_path / "model.safetensors")
    api = mocker.patch("vllm_omni.transformers_utils.repo_utils.hf_api")
    api.return_value.snapshot_download.return_value = str(tmp_path)
    loaded = y.YuE2VAE.from_pretrained(
        "m-a-p/YuE2-Vae" if from_hub else tmp_path, revision="vae-sha", decoder_only=True
    )
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    if from_hub:
        assert api.return_value.snapshot_download.call_args.kwargs["revision"] == "vae-sha"
    else:
        api.assert_not_called()


def test_vae_config_round_trip():
    config = Yue2Config(yue2_vae_path="m-a-p/YuE2-Vae", yue2_vae_revision="vae-sha")
    restored = Yue2Config.from_dict(config.to_dict())
    assert restored.yue2_vae_path == "m-a-p/YuE2-Vae"
    assert restored.yue2_vae_revision == "vae-sha"
