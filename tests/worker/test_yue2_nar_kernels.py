# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NAR activation fallback and compile-time output contract."""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.yue2 import yue2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cpu_post_attention_preserves_reference(monkeypatch, available, dtype):
    monkeypatch.setattr(yue2, "_USE_FLASHINFER_SILU", available)
    torch.manual_seed(73)
    x, attn, w_o, norm, gate_up, down = [
        torch.randn(*shape, dtype=dtype) for shape in [(7, 32), (7, 4, 8), (32, 32), (32,), (96, 32), (32, 48)]
    ]
    residual = x + F.linear(attn.flatten(1), w_o)
    gate, up = F.linear(yue2._rms(residual, norm, 1e-6), gate_up).chunk(2, dim=-1)
    expected = residual + F.linear(F.silu(gate) * up, down)
    actual = yue2._nar_post_attention(x, attn, w_o, norm, gate_up, down, 1e-6)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(7, 12288), (2, 7, 12288)])
def test_flashinfer_fake_output_contract(shape):
    x = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    out = yue2._flashinfer_silu_and_mul(x)
    assert out.shape == (*shape[:-1], shape[-1] // 2)
    assert out.dtype == x.dtype and out.device == x.device
    assert out.is_contiguous()


@pytest.mark.parametrize("capture_fails", [False, True])
def test_side_stream_capture_reclaims_cache_once(monkeypatch, mocker, capture_fails):
    engine = object.__new__(yue2.CachedNAR)
    engine.device = torch.device("cuda:0")
    engine.pool = (0, 1)
    engine.graph = None
    engine.state_in = torch.zeros(2, 64)
    engine.time_in = torch.zeros(())
    state = torch.ones_like(engine.state_in)
    calls = mocker.Mock()
    graph = mocker.Mock(spec=torch.cuda.CUDAGraph)
    for name in ("capture_begin", "capture_end", "replay"):
        calls.attach_mock(getattr(graph, name), name)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "side")
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: "default")
    monkeypatch.setattr(torch.accelerator, "synchronize", calls.synchronize)
    monkeypatch.setattr(torch.accelerator, "empty_cache", calls.empty_cache)
    engine._velocity = mocker.Mock(return_value=state.clone())
    if capture_fails:
        engine._velocity.side_effect = RuntimeError("capture failed")
        with pytest.raises(RuntimeError, match="capture failed"):
            engine.velocity(state, 0.5)
    else:
        assert torch.equal(engine.velocity(state, 0.5), state)
        assert torch.equal(engine.velocity(state, 0.25), state)
        assert torch.equal(engine.state_in, state)
        assert engine.time_in.item() == 0.25
    expected = [
        mocker.call.synchronize(engine.device),
        mocker.call.empty_cache(),
        mocker.call.capture_begin(pool=engine.pool, capture_error_mode="thread_local"),
        mocker.call.capture_end(),
    ]
    if not capture_fails:
        expected += [mocker.call.replay(), mocker.call.replay()]
    assert calls.mock_calls == expected
    engine._velocity.assert_called_once()
