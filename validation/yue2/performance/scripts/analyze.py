"""Independently validate the frozen upstream/candidate E2E comparison."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from tps import metrics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan = read(ROOT / "evidence/plan.json")
for name, expected in plan["runtime_files"].items():
    assert sha(ROOT / name) == expected, name
manifest = rows(ROOT / "evidence/reproduction_manifest.jsonl")
results, reference = [], None
assert sys.argv[1:]
for label in sys.argv[1:]:
    folder = ROOT / "artifacts" / label
    s, p = read(folder / "summary.json"), read(folder / "protocol.json")
    assert s["complete"] and s["count"] == p["count"] == 32 and p["concurrency"] == 2
    assert not p["observe"] and p["warmup"] == 2 and p["thermal_warmup"] == 0
    assert p["offset"] == 1 and p["max_abc"] == 4096 and p["max_frames"] == 9000
    assert p["harness_sha256"] == sha(folder / "harness.py") == sha(ROOT / "scripts/bench_e2e.py")
    assert (folder / "config.yaml").read_bytes() == (ROOT / p["config"]).read_bytes()
    before = read(folder / "source-hashes.json")
    assert before == read(folder / "source-after.json")
    assert {k: v for k, v in before.items() if not k.endswith('/_version.py')} == read(ROOT / f"evidence/{p['arm']}-source.json")
    for filename, count, warm in [("requests.jsonl", 32, False), ("warmup.jsonl", 2, True)]:
        data = rows(folder / filename)
        expected = set()
        for index in range(count):
            row = manifest[(190 + index) % 192 if warm else (1 + index * 5) % 192]
            seed = row["candidate_ar_seeds"][index % 2]
            expected.add((f"{'warm' if warm else 'measure'}_{index}_{row['prompt_index']}_{seed}", row["prompt_index"], seed))
        assert len(data) == count
        assert {(r["id"], r["prompt"], r["seed"]) for r in data} == expected
    data = rows(folder / "requests.jsonl")
    elapsed = max(r["end"] for r in data) - min(r["start"] for r in data)
    assert math.isclose(elapsed, s["elapsed_s"], abs_tol=1e-6)
    assert math.isclose(sum(r['audio_s'] for r in data) / elapsed, s['throughput_audio_s_s'], abs_tol=1e-9)
    assert math.isclose(len(data) / elapsed, s['throughput_requests_s'], abs_tol=1e-9)
    for r in data:
        assert math.isclose(r['audio_s'], (r['frames'] * 1920 - 64) / 48000)
        assert math.isclose(r['rtf'], r['e2el_s'] / r['audio_s'])
    for key in ['e2el_s', 'rtf']:
        for q in [50, 95]:
            assert math.isclose(float(np.percentile([r[key] for r in data], q)), s[key][f'p{q}'], abs_tol=1e-8)
    tps = metrics(data, elapsed)
    for key, value in tps.items():
        assert math.isclose(value, s[key], abs_tol=1e-8)
    for name in ['mps-before.json', 'mps-after.json']:
        status = read(folder / name)[0][0]
        assert status['mode'] == 'off' and str(status['worker_pid']) in status['gpu_pids']
        assert status['acoustic_priority'] == status['priority_range'][1] and status['max_inflight'] == 2
        assert status['flashinfer_silu']==(p['arm']=='candidate')
    mem = s['memory'][0][0]
    assert mem == read(folder / 'final-memory.json')
    assert mem['allocator']['num_ooms'] == 0
    assert mem['model_states'] == 0 and not mem['batches'] and not mem['nar_batches']
    dev = [v for v in read(folder / 'device-memory.json') if s['measured_start'] <= v[0] <= s['measured_end']]
    assert dev and max(v[1] for v in dev) == s['device_peak_measured_bytes']
    data.sort(key=lambda r: r['id'])
    if reference is None:
        reference = data
    result = dict(label=label, arm=p['arm'], elapsed_s=elapsed,
                  throughput=s['throughput_audio_s_s'], request_throughput=s['throughput_requests_s'],
                  e2el=s['e2el_s'], rtf=s['rtf'], **tps,
                  peak_gib=s['device_peak_measured_bytes'] / 2**30,
                  allocated_gib=mem['peak_allocated'] / 2**30, reserved_gib=mem['peak_reserved'] / 2**30,
                  allocator_initial=s['initial_memory'][0][0]['allocator'], allocator_final=mem['allocator'],
                  total_audio_s=sum(r['audio_s'] for r in data), abc_truncated=s['abc_truncated'], semantic_truncated=s['semantic_truncated'],
                  temperature=[min(v[3] for v in dev), max(v[3] for v in dev)], sm_median=float(np.median([v[4] for v in dev])),
                  thermal_fraction=sum(bool(v[5] & (32 | 64)) for v in dev) / len(dev),
                  returned_token_sequence_equal_to_first=sum(a['semantic_sha256'] == b['semantic_sha256'] for a, b in zip(data, reference)),
                  abc_equal_to_first=sum(a['abc_sha256'] == b['abc_sha256'] for a, b in zip(data, reference)))
    results.append(result)
output = dict(complete=True, results=results, scope='4090 frozen upstream/final unified main path ABBA; each process has 32 measured + 2 warmups; no batch/step/Nsight observers; full E2E content TPS, not phase-local decode TPS.')
(ROOT / 'artifacts/analysis.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
