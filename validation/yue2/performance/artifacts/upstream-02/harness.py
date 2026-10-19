"""Matched upstream two-call / candidate same-request ABC→audio benchmark."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import threading
import subprocess
import time
from pathlib import Path
from tps import metrics as content_tps
from types import SimpleNamespace

import numpy as np
import pynvml
import torch
from vllm import SamplingParams
from vllm_omni import AsyncOmni
from vllm_omni.tokenizers.yue2_prompt import abc_ids_from_generated, abc_prefix_ids, semantic_prefix_ids
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer
from vllm_omni.model_executor.models.yue2.yue2 import ABC_END, CODEC_OFFSET, CODEC_SIZE
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

ROOT = Path(__file__).resolve().parents[1]
ARM = os.environ['YUE2_ARM']
SOURCE = ROOT / os.environ.get('YUE2_SOURCE', 'source' if ARM == 'candidate' else 'upstream')
if ARM == 'candidate':
    from vllm_omni.model_executor.models.yue2.queued import Yue2QueueWorker
    from vllm_omni.model_executor.models.yue2.request import prepare_song
else:
    Yue2QueueWorker = GPUARWorker


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def worker_result(result):
    assert isinstance(result,list) and len(result)==1 and isinstance(result[0],list) and len(result[0])==1,result
    assert isinstance(result[0][0],dict) and 'error' not in result[0][0],result
    return result[0][0]


class BenchWorker(Yue2QueueWorker):
    def bench_mps(self):
        if os.environ['YUE2_MPS'] == 'off':
            assert not os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
            pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()
            assert str(os.getpid()) in pids, (os.getpid(), pids)
            queue = self.model_runner.model.sample.__self__._synthesis
            actual = queue.stream().priority
            expected = 0 if os.environ['YUE2_SOURCE'] == 'equal' else torch.cuda.Stream.priority_range()[1]
            assert actual == expected
            from vllm_omni.model_executor.models.yue2.yue2 import _SynthesisJob
            assert _SynthesisJob.MAX_INFLIGHT == 2
            from vllm_omni.model_executor.models.yue2 import yue2
            fi_enabled=getattr(yue2,'_USE_FLASHINFER_SILU',False)
            assert fi_enabled == (ARM=='candidate')
            alloc_env={k:os.environ.get(k) for k in ['PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF']}
            assert not any(alloc_env.values()),alloc_env
            return dict(gpu_name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                        fa3_available=yue2._flash_attention_fn() is not None, allocator_env=alloc_env,
                        flashinfer_silu=fi_enabled, mode='off', worker_pid=os.getpid(), gpu_pids=pids,
                        acoustic_priority=actual, priority_range=torch.cuda.Stream.priority_range(),
                        max_inflight=_SynthesisJob.MAX_INFLIGHT)
        def control(command):
            return subprocess.check_output(['nvidia-cuda-mps-control'], input=command+'\n', text=True, timeout=10).strip()
        servers = control('get_server_list').split()
        assert servers and all(pid.isdigit() for pid in servers), servers
        clients = {server: control('get_client_list '+server).split() for server in servers}
        assert str(os.getpid()) in {pid for group in clients.values() for pid in group}, (os.getpid(), clients)
        return dict(mode='on', worker_pid=os.getpid(), pipe=os.environ['CUDA_MPS_PIPE_DIRECTORY'], clients=clients)

    def bench_heat(self, seconds):
        # Top up cold-card warmup only; never keep heating an already hot card.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        begin = time.perf_counter()
        try:
            if pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) < 83:
                a = torch.full((4096, 4096), 0.0001, device="cuda", dtype=torch.bfloat16)
                b = torch.empty_like(a)
                while time.perf_counter() - begin < seconds:
                    if pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) >= 83:
                        break
                    for _ in range(16):
                        torch.mm(a, a, out=b)
                    torch.cuda.synchronize()
            return dict(elapsed_s=time.perf_counter()-begin,
                        temperature_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        finally:
            pynvml.nvmlShutdown()

    def bench_reset(self, observe=False):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.batches = []
        self.nar_batches = []
        if observe and ARM == 'candidate':
            queue = self.model_runner.model.sample.__self__._synthesis
            from vllm_omni.model_executor.models.yue2 import batched_synthesis
            original_nar = batched_synthesis.BatchedCachedNAR
            def make_nar(*args, **kwargs):
                engine = original_nar(*args, **kwargs)
                self.nar_batches.append(dict(t=time.perf_counter(), frames=engine.lengths,
                                            query_tokens=engine.nar_length, kv_tokens=engine.key_offsets[-1]))
                return engine
            batched_synthesis.BatchedCachedNAR = make_nar
            original = queue.pump
            def pump():
                previous = queue.active
                original()
                if queue.active is not None and previous is not queue.active:
                    self.batches.append(dict(t=time.perf_counter(), ids=[s.request_id for s in queue.members],
                                             frames=[len(s.history) for s in queue.members],
                                             prefix_tokens=[len(s.prefix_ids) for s in queue.members],
                                             budget_tokens=[getattr(s,'acoustic_batch_tokens',None) for s in queue.members]))
            queue.pump = pump
        return self.bench_memory()

    def bench_memory(self):
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                    batches=getattr(self, 'batches', []), nar_batches=getattr(self, 'nar_batches', []),
                    allocator={k:torch.cuda.memory_stats().get(k,0) for k in ['num_alloc_retries','num_ooms']},
                    model_states=len(self.model_runner.model.sample.__self__._states))


def hashes():
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((SOURCE/'vllm_omni').rglob('*')) if p.suffix in ('.py','.yaml')}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=9000)
    parser.add_argument('--max-abc', type=int, default=4096)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--thermal-warmup', type=float, default=0)
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.label
    out.mkdir(parents=True, exist_ok=False)
    (out/'harness.py').write_bytes(Path(__file__).read_bytes())
    assets = json.loads((ROOT/'evidence/assets.json').read_text())
    rows = [json.loads(s) for s in (ROOT/'evidence/reproduction_manifest.jsonl').read_text().splitlines()]
    tokenizer = YuE2TextTokenizer.from_pretrained(assets[0]['repo'], revision=assets[0]['revision'])
    spec = importlib.util.spec_from_file_location('official_end2end', SOURCE/'examples/offline_inference/yue2/end2end.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    defaults = SimpleNamespace(resolve_sampling_params_list=lambda _: [SamplingParams(temperature=1.0)])
    before = hashes()
    assert {k:v for k,v in before.items() if not k.endswith('/_version.py')}==json.loads((ROOT/f'evidence/{ARM}-source.json').read_text())
    write(out/'source-hashes.json', before)
    write(out/'protocol.json', dict(vars(args), arm=ARM, model=assets[:2], source=str(SOURCE),
                                   harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/'config.yaml').write_bytes(Path(args.config).read_bytes())
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.is_set():
            samples.append([time.perf_counter(), pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                            pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                            pynvml.nvmlDeviceGetPowerUsage(handle)])
            stop.wait(0.1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    engine = None
    try:
        engine = AsyncOmni(model=assets[0]['path'], revision=assets[0]['revision'],
            deploy_config=args.config, stage_overrides={'0': {'worker_cls': 'bench_e2e.BenchWorker',
                'hf_overrides': {'yue2_vae_path': assets[1]['path'], 'yue2_vae_revision': assets[1]['revision']}}})
        gate = asyncio.Semaphore(args.concurrency)
        async def generate(prompt, params, rid):
            result = None
            async for result in engine.generate(prompt, request_id=rid, sampling_params_list=[params]):
                pass
            assert result is not None, rid
            return result
        async def song(index, warm=False):
            row = rows[(190 + index) % 192] if warm else rows[(args.offset + index * 5) % 192]
            seed = row['candidate_ar_seeds'][index % 2]
            rid = f"{'warm' if warm else 'measure'}_{index}_{row['prompt_index']}_{seed}"
            queued = time.perf_counter()
            async with gate:
                begin = time.perf_counter()
                if ARM == 'candidate':
                    prompt, params = prepare_song(tokenizer.encode, row['style'], row['lyrics'], cot='full',
                        seed=seed, max_frames=args.max_frames, max_abc_tokens=args.max_abc)
                    result = await generate(prompt, params, rid)
                    mm = result.multimodal_output
                    abc = mm['abc_token_ids']
                    meta = mm.get('meta') or {k[5:]: v for k,v in mm.items() if k.startswith('meta.')}
                    abc_truncated = bool(torch.as_tensor(meta.get('abc_truncated', False)).item())
                else:
                    prefix = abc_prefix_ids(tokenizer.encode, row['style'], row['lyrics'], 'full')
                    params = helper.sampling_params(defaults, phase='abc', seed=seed,
                                                   max_frames=args.max_frames, prompt_ids=prefix)
                    params.max_tokens = args.max_abc
                    result = await generate({'prompt_token_ids': prefix}, params, rid+'_abc')
                    tokens = result.outputs[0].token_ids
                    abc_truncated = ABC_END not in tokens
                    abc = abc_ids_from_generated(tokens)
                    prefix = semantic_prefix_ids(tokenizer.encode, row['style'], row['lyrics'], 'full', abc_ids=abc)
                    params = helper.sampling_params(defaults, phase='semantic', seed=seed,
                                                   max_frames=args.max_frames, prompt_ids=prefix)
                    result = await generate({'prompt_token_ids': prefix}, params, rid)
                    mm = result.multimodal_output
                    meta = mm.get('meta') or {k[5:]: v for k,v in mm.items() if k.startswith('meta.')}
                end = time.perf_counter()
            audio = mm.get('audio', mm.get('model_outputs'))
            tokens = result.outputs[0].token_ids
            frames = sum(CODEC_OFFSET <= t < CODEC_OFFSET + CODEC_SIZE for t in tokens)
            assert audio.shape == (2, frames*1920-64), (rid, audio.shape, frames)
            assert not torch.as_tensor(meta['error']).item(), rid
            assert await asyncio.to_thread(lambda: bool(torch.isfinite(audio).all())), rid
            item = dict(id=rid, prompt=row['prompt_index'], seed=seed, queued=queued, start=begin,end=end,
                        e2el_s=end-begin, offered_e2el_s=end-queued, frames=frames,audio_s=audio.shape[-1]/48000,
                        abc_tokens=len(abc),abc_truncated=abc_truncated,
                        truncated=bool(torch.as_tensor(meta['truncated']).item()))
            item['rtf'] = item['e2el_s']/item['audio_s']
            item['semantic_sha256'] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            item['abc_sha256'] = hashlib.sha256(json.dumps(torch.as_tensor(abc).tolist()).encode()).hexdigest()
            with (out/('warmup.jsonl' if warm else 'requests.jsonl')).open('a') as f:
                f.write(json.dumps(item)+'\n')
            print(json.dumps(item), flush=True)
            return item
        await asyncio.gather(*(song(i, True) for i in range(args.warmup)))
        heat = None
        if args.thermal_warmup:
            heat = await engine.collective_rpc('bench_heat', args=(args.thermal_warmup,), timeout=args.thermal_warmup+120)
        mps_before = await engine.collective_rpc('bench_mps')
        worker_result(mps_before)
        write(out/'mps-before.json', mps_before)
        initial = await engine.collective_rpc('bench_reset', args=(args.observe,))
        worker_result(initial)
        begin = time.perf_counter()
        results = await asyncio.gather(*(song(i) for i in range(args.count)))
        end = max(r['end'] for r in results)
        memory = await engine.collective_rpc('bench_memory')
        mps_after = await engine.collective_rpc('bench_mps')
        worker_result(mps_after)
        final_memory=worker_result(memory)
        write(out/'final-memory.json', final_memory)
        assert final_memory['model_states']==0
        # Cache reclamation retries are expected under a tight device budget.
        assert final_memory['allocator']['num_ooms']==0, final_memory
        write(out/'mps-after.json', mps_after)
        elapsed = end-min(r['start'] for r in results)
        summary = dict(complete=True, arm=ARM, count=len(results), elapsed_s=elapsed,
                       throughput_audio_s_s=sum(r['audio_s'] for r in results)/elapsed,
                       throughput_requests_s=len(results)/elapsed,
                       initial_memory=initial, memory=memory, thermal_precondition=heat,
                       device_peak_all_bytes=max(v[1] for v in samples),
                       device_peak_measured_bytes=max(v[1] for v in samples if begin<=v[0]<=end),
                       device_sample_columns=['monotonic_s','used_bytes','gpu_busy_pct','temperature_c','sm_mhz','clock_reasons','power_mw'],
                       measured_start=begin, measured_end=end,
                       abc_truncated=sum(r['abc_truncated'] for r in results),
                       semantic_truncated=sum(r['truncated'] for r in results))
        summary.update(content_tps(results, elapsed))
        summary['tps_scope'] = 'Aggregate retained ABC + semantic content tokens per full E2E makespan; no prompts/control tokens/negative CFG or ODE work.'
        for key in ['e2el_s','offered_e2el_s','rtf']:
            summary[key] = {f'p{q}':float(np.percentile([r[key] for r in results],q)) for q in [50,95]}
        assert hashes()==before
        write(out/'source-after.json', hashes())
        write(out/'summary.json',summary)
        print('SUMMARY',json.dumps(summary),flush=True)
    finally:
        if engine is not None:
            engine.close()
        stop.set()
        thread.join()
        write(out/'device-memory.json',samples)
        pynvml.nvmlShutdown()

if __name__=='__main__':
    asyncio.run(main())
