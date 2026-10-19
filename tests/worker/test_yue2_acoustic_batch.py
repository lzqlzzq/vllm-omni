# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-input independent-request oracle for packed NAR batching."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.yue2 import batched_nar, yue2
from vllm_omni.model_executor.models.yue2.batched_synthesis import BatchedSynthesisQueue, decode_tiles_iter
from vllm_omni.transformers_utils.configs.yue2 import Yue2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def tiny_model(monkeypatch, mocker):
    torch.manual_seed(721)

    def kernels():
        return yue2._nar_pre_attention, yue2._nar_post_attention

    monkeypatch.setattr(yue2, "_nar_kernels", kernels)
    monkeypatch.setattr(batched_nar, "_nar_kernels", kernels)
    monkeypatch.setattr(yue2, "_flash_attention_fn", lambda: None)
    config = Yue2Config(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        rms_norm_eps=1e-6,
    )
    layers = nn.ModuleList([yue2._NARLayer(config) for _ in range(2)])
    fused = yue2.fuse_nar_projections(layers)
    rotary = yue2._RotaryEmbedding(8, 1000000.0)
    return mocker.Mock(
        spec=yue2.Yue2ForCausalLM,
        config=config,
        vae2llm=nn.Linear(64, 32),
        llm2vae=nn.Linear(32, 64),
        time_embedder=yue2._TimestepEmbedder(32),
        latent_pos_embed=yue2._AudioPositionEmbedding(64, 32),
        rotary=rotary,
        model=mocker.Mock(norm=nn.RMSNorm(32, eps=1e-6)),
        vocab_size=100,
        max_position_embeddings=64,
        max_latent_frames=64,
        _timestep_shift=1.0,
        nar_weights=lambda: fused,
    )


@pytest.fixture(autouse=True)
def fixed_prefill(monkeypatch):
    def prefill(self):
        if self.keys is None:
            self.keys, self.values = (x.clone() for x in self.chunk.test_condition)
        else:
            for target, condition in zip((self.keys, self.values), self.chunk.test_condition):
                target[:, : self.ar_length].copy_(condition[:, : self.ar_length])

    monkeypatch.setattr(yue2.CachedNAR, "_prefill", prefill)


def inputs(lengths):
    chunks, conditions = [], []
    for ar, frames in lengths:
        chunks.append(yue2.Chunk(ar_tokens=list(range(ar)), noise=torch.randn(frames, 64)))
        conditions.append(tuple(torch.randn(2, ar + frames + 2, 2, 8) for _ in range(2)))
    for chunk, condition in zip(chunks, conditions):
        chunk.test_condition = condition
    return chunks, conditions


@pytest.mark.parametrize("lengths", [[(5, 4), (5, 4)], [(3, 2), (7, 5)], [(7, 2), (3, 5), (4, 3), (6, 4)]])
@pytest.mark.parametrize("varlen", [False, True])
@torch.inference_mode()
def test_batched_solver_matches_independent_requests(tiny_model, lengths, varlen):
    chunks, _ = inputs(lengths)
    expected = [yue2.CachedNAR(tiny_model, c).solve(3) for c in chunks]
    engine = batched_nar.BatchedCachedNAR(tiny_model, chunks)

    def varlen_oracle(q, k, v, *, cu_seqlens_q, cu_seqlens_k, **kwargs):
        # Independent attention on the packed spans checks both gather indices
        # and cumulative boundaries; CUDA tests use the real flash kernel.
        outputs = []
        for qa, qb, ka, kb in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:]):
            outputs.append(yue2._attention(q[qa:qb], k[ka:kb], v[ka:kb], causal=False))
        return torch.cat(outputs)

    yue2._drain(engine.prefill_iter())
    if varlen:
        engine.batch_flash = varlen_oracle
    actual = yue2._drain(engine.solve_iter(3)).split(engine.lengths)
    for index, value in enumerate(expected):
        torch.testing.assert_close(actual[index], value, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_request_order_and_lengths_do_not_leak_between_songs(tiny_model):
    chunks, conditions = inputs([(3, 2), (7, 5)])
    reference = batched_nar.BatchedCachedNAR(tiny_model, chunks)
    yue2._drain(reference.prefill_iter())
    first = yue2._drain(reference.solve_iter(2)).split(reference.lengths)
    # A drastically different neighbour and reversed order must not affect
    # request zero. This catches missing batch/length masks and bad offsets.
    chunks[1].noise.mul_(100)
    for value in conditions[1]:
        value.mul_(100)
    reverse = batched_nar.BatchedCachedNAR(tiny_model, chunks[::-1])
    yue2._drain(reverse.prefill_iter())
    second = yue2._drain(reverse.solve_iter(2)).split(reverse.lengths)
    torch.testing.assert_close(first[0], second[1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
@torch.inference_mode()
def test_vae_tile_batch_matches_original_crops_and_cancellation(batch_size):
    torch.manual_seed(731)
    vae = object.__new__(yue2.YuE2VAE)
    nn.Module.__init__(vae)
    vae.config = yue2.YuE2VAEConfig(
        latent_dim=64,
        audio_channels=2,
        downsampling_ratio=16,
        decoder_config=dict(channels=8, c_mults=[1, 2], strides=[4, 4], latent_dim=64),
    )
    vae.decoder = yue2.OobleckDecoder(channels=8, latent_dim=64, c_mults=(1, 2), strides=(4, 4))
    vae.eval()
    latents = [torch.randn(n, 64) for n in (48, 48, 65, 17)]
    originals = [z.clone() for z in latents]
    expected = [vae.decode_tiled(z.T[None], core_frames=16, halo_frames=16)[0] for z in latents]
    actual = yue2._drain(decode_tiles_iter(vae, latents, batch_size, 16))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    actual = yue2._drain(decode_tiles_iter(vae, latents, batch_size, 16, {1}))
    assert not actual[1].count_nonzero()
    for i in (0, 2, 3):
        torch.testing.assert_close(actual[i], expected[i], atol=2e-6, rtol=2e-5)
    for latent, original in zip(latents, originals):
        assert torch.equal(latent, original)


def test_batch_member_cancellation_preserves_survivor_and_mapping(mocker):
    queue = BatchedSynthesisQueue(None, 2, 2, 1024)
    a, b, c = [yue2._RequestState(x, mocker.Mock(spec=yue2._RowConstants)) for x in ("a", "b", "c")]
    queue.members = [a, b]
    queue.waiting = [c]
    queue.sizes = [2, 3]
    cancelled = []
    source = torch.tensor([[1, 1, 2, 2, 2], [1, 1, 2, 2, 2]])
    queue.active = mocker.Mock(
        spec=yue2._SynthesisJob,
        done=lambda: True,
        finish=lambda: (source, False),
        cancel=lambda: cancelled.append(True),
    )
    queue.cancel_request("c")
    queue.cancel_request("a")
    assert not cancelled and not queue.waiting
    ready = queue.poll()
    assert set(ready) == {"b"} and torch.equal(ready["b"]["audio"], torch.full((2, 3), 2))
    assert ready["b"]["audio"].is_contiguous() and ready["b"]["audio"]._base is None
    source.zero_()
    assert torch.equal(ready["b"]["audio"], torch.full((2, 3), 2))
    assert not queue.active and not queue.members
    queue.members = [a, b]
    queue.active = mocker.Mock(spec=yue2._SynthesisJob, cancel=lambda: cancelled.append(True))
    queue.cancelled = set()
    queue.cancel_request("a")
    queue.cancel_request("b")
    assert cancelled == [True] and queue.active is None and not queue.members


def test_batch_window_deadline_full_cancel_and_drain(monkeypatch, mocker):
    from vllm_omni.model_executor.models.yue2 import batched_synthesis as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module, "_SynthesisJob", lambda *a: mocker.Mock(start=lambda stream: None))
    monkeypatch.setattr(BatchedSynthesisQueue, "stream", lambda self: None)
    queue = BatchedSynthesisQueue(None, 2, 1, 1024, batch_wait_ms=100)

    def submit(name):
        state = yue2._RequestState(name, mocker.Mock(spec=yue2._RowConstants), history=[yue2.CODEC_OFFSET])
        queue.submit(state)
        return state

    submit("a")
    queue.pump()
    assert queue.active is None
    now[0] += 0.05
    submit("b")
    queue.pump()
    assert [s.request_id for s in queue.members] == ["a", "b"]
    queue.active, queue.members = None, []
    submit("c")
    now[0] += 0.05
    submit("d")
    queue.cancel_request("c")
    now[0] += 0.06
    queue.pump()
    assert queue.active is None  # d keeps its own arrival deadline.
    now[0] += 0.05
    queue.pump()
    assert [s.request_id for s in queue.members] == ["d"]  # no AR arrivals needed.
    queue.active, queue.members = None, []
    submit("e")
    now[0] += 1
    queue.pump()
    assert [s.request_id for s in queue.members] == ["e"]  # prior job time counts.
    for cap, wait in [(1, 100), (2, 0)]:
        queue = BatchedSynthesisQueue(None, cap, 1, 1024, batch_wait_ms=wait)
        submit("immediate")
        queue.pump()
        assert queue.active is not None
    for invalid in [-1, float("nan"), float("inf"), True, "1"]:
        with pytest.raises(ValueError):
            BatchedSynthesisQueue(None, 2, 1, 1024, batch_wait_ms=invalid)


@pytest.mark.parametrize("frames", [7, 20000])
def test_token_cost_matches_real_chunk_kv_lengths(frames, mocker):
    state = yue2._RequestState(
        "cost", mocker.Mock(spec=yue2._RowConstants), prefix_ids=[1, 2, 3], history=[yue2.CODEC_OFFSET] * frames
    )
    queue = BatchedSynthesisQueue(None, 4, 1, 1024, batch_max_tokens=100)
    queue.submit(state)
    chunks = yue2.song_chunks(state.prefix_ids, [0] * frames, 17)
    assert state.acoustic_batch_tokens == max(len(c.ar_tokens) + len(c.noise) + 2 for c in chunks)


def test_token_budget_preserves_fifo_and_flushes_without_wait(monkeypatch, mocker):
    from vllm_omni.model_executor.models.yue2 import batched_synthesis as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(module, "_SynthesisJob", lambda *a: mocker.Mock(start=lambda stream: None))
    monkeypatch.setattr(BatchedSynthesisQueue, "stream", lambda self: None)
    queue = BatchedSynthesisQueue(None, 4, 1, 1024, batch_wait_ms=1000, batch_max_tokens=30)

    def submit(name, frames):
        state = yue2._RequestState(
            name, mocker.Mock(spec=yue2._RowConstants), prefix_ids=[1], history=[yue2.CODEC_OFFSET] * frames
        )
        queue.submit(state)

    # Costs 10, 24, 10: do not skip the middle request to fill the first batch.
    submit("a", 3)
    submit("b", 10)
    submit("c", 3)
    queue.pump()
    assert [s.request_id for s in queue.members] == ["a"]
    assert [s.request_id for s in queue.waiting] == ["b", "c"]
    queue.active, queue.members = None, []
    queue.cancel_request("b")
    queue.pump()
    assert queue.active is None  # c can still wait for a compatible neighbour.
    submit("d", 8)  # Cost 20: exact budget, flush without waiting.
    queue.pump()
    assert [s.request_id for s in queue.members] == ["c", "d"]
    queue.active, queue.members = None, []
    submit("oversize", 20)
    submit("tail", 3)
    queue.pump()
    assert [s.request_id for s in queue.members] == ["oversize"]
    assert [s.request_id for s in queue.waiting] == ["tail"]


@pytest.mark.parametrize("invalid", [-1, 1.5, float("nan"), float("inf"), True, "30", None])
def test_token_budget_rejects_invalid_config(invalid):
    with pytest.raises(ValueError, match="token budget"):
        BatchedSynthesisQueue(None, 4, 1, 1024, batch_max_tokens=invalid)


@torch.inference_mode()
def test_packed_prefill_writes_final_storage_and_close_keeps_views_alive(tiny_model, monkeypatch):
    chunks, conditions = inputs([(3, 2), (7, 5), (4, 3)])
    original = yue2.CachedNAR._prefill
    members = []

    def observe(self):
        members.append(self)
        original(self)

    monkeypatch.setattr(yue2.CachedNAR, "_prefill", observe)
    engine = batched_nar.BatchedCachedNAR(tiny_model, chunks)
    assert len(list(engine.prefill_iter())) == len(chunks)
    for member, start, end, condition in zip(members, engine.key_offsets, engine.key_offsets[1:], conditions):
        for target, packed, expected in zip((member.keys, member.values), (engine.keys, engine.values), condition):
            assert target.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
            assert target.data_ptr() == packed[:, start:end].data_ptr()
            torch.testing.assert_close(target[:, : member.ar_length], expected[:, : member.ar_length], rtol=0, atol=0)
    for member in members:
        member.close()
    result = yue2._drain(engine.solve_iter(2))
    assert torch.isfinite(result).all()
    engine.close()
    assert engine.keys is None and engine.values is None


def test_supplied_condition_storage_rejects_wrong_shape(tiny_model):
    chunks, _ = inputs([(3, 2)])
    wrong = torch.zeros(2, 8, 2, 8)
    with pytest.raises(ValueError, match="Condition KV storage"):
        yue2.CachedNAR(tiny_model, chunks[0], kv=(wrong, wrong.clone()))


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
