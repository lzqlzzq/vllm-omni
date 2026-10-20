"""Matched old/new PR E2E benchmark without step or synthesis hooks."""
import argparse
import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.yue2 import yue2
from vllm_omni.model_executor.models.yue2.request import prepare_song
from vllm_omni.model_executor.models.yue2.runtime import Yue2Worker
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class BenchWorker(Yue2Worker):
    def bench_reset(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return self.bench_snapshot()

    def bench_snapshot(self):
        model = self.model_runner.model.sample.__self__
        queue = model._synthesis
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                    allocator={k: torch.cuda.memory_stats().get(k, 0) for k in ('num_alloc_retries', 'num_ooms')},
                    model_states=len(model._states), active=queue.active is not None, waiting=len(queue.waiting),
                    fa3_available=yue2._flash_attention_fn() is not None, acoustic_priority=queue.stream().priority)


def worker_result(value):
    assert len(value) == 1 and len(value[0]) == 1 and isinstance(value[0][0], dict), value
    assert 'error' not in value[0][0], value
    return value[0][0]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['old-pr', 'new-pr'], required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--max-frames', type=int, default=9000)
    p.add_argument('--max-abc', type=int, default=4096)
    args = p.parse_args()
    out = ROOT / 'artifacts' / args.label
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / ('source-' + args.arm)
    expected = json.loads((ROOT / 'evidence/variant-sources.json').read_text())[args.arm]
    expected.update(json.loads((ROOT / 'evidence/runtime-sources.json').read_text())[args.arm])
    plan = json.loads((ROOT / 'evidence/plan.json').read_text())
    selection = plan['original_sequence_indices']
    assert args.count == len(selection) and args.warmup == plan['warmup']
    def verify():
        assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == sha for name, sha in expected.items())
    verify()
    assets = json.loads((ROOT / 'evidence/assets.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'evidence/reproduction_manifest.jsonl').read_text().splitlines()]
    tokenizer = YuE2TextTokenizer.from_pretrained(assets[0]['path'], revision=assets[0]['revision'])
    config = ROOT / 'config.yaml'
    write(out / 'protocol.json', dict(vars(args), concurrency=2, assets=assets,
                                     source=str(source), commit=plan['refs'][args.arm], sequence_indices=selection,
                                     stage_probes=False, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                                     harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out / 'config.yaml').write_bytes(config.read_bytes())
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
        engine = AsyncOmni(model=assets[0]['path'], revision=assets[0]['revision'], deploy_config=str(config),
            stage_overrides={'0': {'worker_cls': 'sync_bench.BenchWorker', 'tokenizer': assets[0]['path'],
                'hf_overrides': {'yue2_vae_path': assets[1]['path'], 'yue2_vae_revision': assets[1]['revision']}}})
        gate = asyncio.Semaphore(2)
        async def song(index, warm=False):
            original_index = index if warm else selection[index]
            row = rows[(190 + index) % 192] if warm else rows[(1 + original_index * 5) % 192]
            seed = row['candidate_ar_seeds'][original_index % 2]
            rid = f"{'warm' if warm else 'measure'}_{index}_{row['prompt_index']}_{seed}"
            async with gate:
                begin = time.perf_counter()
                prompt, params = prepare_song(tokenizer.encode, row['style'], row['lyrics'], cot='full',
                    seed=seed, max_frames=args.max_frames, max_abc_tokens=args.max_abc)
                result = None
                async for result in engine.generate(prompt, request_id=rid, sampling_params_list=[params]):
                    pass
                end = time.perf_counter()
            assert result is not None, rid
            mm = result.multimodal_output
            meta = mm.get('meta') or {k[5:]: v for k, v in mm.items() if k.startswith('meta.')}
            audio = mm['audio']
            tokens = result.outputs[0].token_ids
            frames = sum(yue2.CODEC_OFFSET <= t < yue2.CODEC_OFFSET + yue2.CODEC_SIZE for t in tokens)
            assert not torch.as_tensor(meta['error']).item(), rid
            assert audio.shape == (2, frames * 1920 - 64), (rid, audio.shape, frames)
            assert await asyncio.to_thread(lambda: bool(torch.isfinite(audio).all())), rid
            abc = torch.as_tensor(mm['abc_token_ids']).tolist()
            item = dict(id=rid, prompt=row['prompt_index'], seed=seed, start=begin, end=end, e2el_s=end-begin,
                        frames=frames, audio_s=audio.shape[-1]/48000, abc_tokens=len(abc),
                        abc_truncated=bool(torch.as_tensor(meta.get('abc_truncated', False)).item()),
                        truncated=bool(torch.as_tensor(meta['truncated']).item()),
                        semantic_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        abc_sha256=hashlib.sha256(json.dumps(abc).encode()).hexdigest())
            item['rtf'] = item['e2el_s'] / item['audio_s']
            with (out / ('warmup.jsonl' if warm else 'requests.jsonl')).open('a') as f:
                f.write(json.dumps(item) + '\n')
            print(json.dumps(item), flush=True)
            return item
        await asyncio.gather(*(song(i, True) for i in range(args.warmup)))
        initial = worker_result(await engine.collective_rpc('bench_reset'))
        write(out / 'initial.json', initial)
        begin = time.perf_counter()
        results = await asyncio.gather(*(song(i) for i in range(args.count)))
        final = worker_result(await engine.collective_rpc('bench_snapshot'))
        write(out / 'worker.json', final)
        assert final['model_states'] == 0 and not final['active'] and final['waiting'] == 0, final
        assert final['allocator']['num_ooms'] == initial['allocator']['num_ooms'], final
        elapsed = max(r['end'] for r in results) - min(r['start'] for r in results)
        summary = dict(complete=True, arm=args.arm, count=len(results), elapsed_s=elapsed,
                       throughput_audio_s_s=sum(r['audio_s'] for r in results)/elapsed,
                       total_content_tps=sum(r['frames']+r['abc_tokens'] for r in results)/elapsed,
                       semantic_tps=sum(r['frames'] for r in results)/elapsed,
                       measured_start=begin, measured_end=max(r['end'] for r in results),
                       peak_device_bytes=max(s[1] for s in samples if begin <= s[0] <= max(r['end'] for r in results)),
                       peak_allocated=final['peak_allocated'], peak_reserved=final['peak_reserved'],
                       allocator=final['allocator'])
        for key in ('e2el_s', 'rtf'):
            summary[key] = {f'p{q}': float(np.percentile([r[key] for r in results], q)) for q in (50,95)}
        verify()
        assert config.read_bytes() == (out / 'config.yaml').read_bytes()
        write(out / 'summary.json', summary)
        print('SUMMARY', json.dumps(summary), flush=True)
    finally:
        if engine is not None:
            engine.close()
        stop.set()
        thread.join()
        write(out / 'device.json', samples)
        pynvml.nvmlShutdown()


if __name__ == '__main__':
    asyncio.run(main())
