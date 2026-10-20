"""Recompute complete old/new PR ABBA results from per-request records."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / 'remote'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q / 100
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] * (hi - at) + values[hi] * (at - lo) if lo != hi else values[lo]


def metrics(data):
    elapsed = max(r['end'] for r in data) - min(r['start'] for r in data)
    return dict(elapsed_s=elapsed, audio_s=sum(r['audio_s'] for r in data),
                throughput_audio_s_s=sum(r['audio_s'] for r in data) / elapsed,
                total_content_tps=sum(r['frames'] + r['abc_tokens'] for r in data) / elapsed,
                semantic_tps=sum(r['frames'] for r in data) / elapsed,
                **{key: {f'p{q}': percentile([r[key] for r in data], q) for q in [50, 95]}
                   for key in ['e2el_s', 'rtf']})


def analyze():
    plan = read(ROOT / 'evidence/plan.json')
    manifest = rows(REMOTE / 'evidence/reproduction_manifest.jsonl')
    harness_sha = hashlib.sha256((ROOT / 'scripts/sync_bench.py').read_bytes()).hexdigest()
    assert (REMOTE / 'evidence/controller.exit').read_text().strip() == '0'
    results, reference, configs = [], None, []
    for label in plan['order']:
        folder = REMOTE / 'artifacts' / label
        s, p, worker = [read(folder / f'{name}.json') for name in ['summary', 'protocol', 'worker']]
        arm = label.rsplit('-', 1)[0]
        assert s['complete'] and s['count'] == p['count'] == plan['count']
        assert p['arm'] == arm and p['commit'] == plan['refs'][arm]
        assert p['harness_sha256'] == harness_sha and not p['stage_probes']
        assert p['concurrency'] == 2 and p['warmup'] == 2
        assert p['max_abc'] == 4096 and p['max_frames'] == 9000
        config = (folder / 'config.yaml').read_bytes()
        assert hashlib.sha256(config).hexdigest() == p['config_sha256']
        configs.append(config)
        for filename, count, warm in [('warmup', 2, True), ('requests', 8, False)]:
            data = rows(folder / f'{filename}.jsonl')
            assert len(data) == count
            expected = set()
            for index in range(count):
                original = index if warm else plan['original_sequence_indices'][index]
                row = manifest[(190 + index) % 192 if warm else (1 + original * 5) % 192]
                seed = row['candidate_ar_seeds'][original % 2]
                expected.add((f"{'warm' if warm else 'measure'}_{index}_{row['prompt_index']}_{seed}", row['prompt_index'], seed))
            assert {(r['id'], r['prompt'], r['seed']) for r in data} == expected
            for r in data:
                assert math.isclose(r['audio_s'], (r['frames'] * 1920 - 64) / 48000)
                assert math.isclose(r['rtf'], r['e2el_s'] / r['audio_s'])
                assert math.isclose(r['e2el_s'], r['end'] - r['start'])
        m = metrics(data)
        for key in ['elapsed_s', 'throughput_audio_s_s', 'total_content_tps', 'semantic_tps']:
            assert math.isclose(m[key], s[key], rel_tol=1e-9), (label, key)
        for key in ['e2el_s', 'rtf']:
            for q in ['p50', 'p95']:
                assert math.isclose(m[key][q], s[key][q], rel_tol=1e-9)
        assert worker['model_states'] == 0 and not worker['active'] and worker['waiting'] == 0
        assert worker['allocator']['num_ooms'] == 0
        assert worker['peak_allocated'] == s['peak_allocated'] and worker['peak_reserved'] == s['peak_reserved']
        dev = [v for v in read(folder / 'device.json') if s['measured_start'] <= v[0] <= s['measured_end']]
        assert dev and max(v[1] for v in dev) == s['peak_device_bytes']
        identity = {r['id']: (r['frames'], r['abc_sha256'], r['semantic_sha256']) for r in data}
        if reference is None:
            reference = identity
        results.append(dict(label=label, arm=arm, **m, device_peak_GiB=s['peak_device_bytes'] / 2**30,
                            allocated_peak_GiB=s['peak_allocated'] / 2**30, reserved_peak_GiB=s['peak_reserved'] / 2**30,
                            allocator=worker['allocator'], temperature_c=[min(v[3] for v in dev), max(v[3] for v in dev)],
                            sm_median_mhz=statistics.median(v[4] for v in dev),
                            thermal_fraction=sum(bool(v[5] & (32 | 64)) for v in dev) / len(dev),
                            power_mean_w=statistics.mean(v[6] for v in dev) / 1000,
                            identical_outputs=sum(identity[k] == reference[k] for k in reference),
                            truncated=sum(r['truncated'] or r['abc_truncated'] for r in data)))
    assert all(c == configs[0] for c in configs)
    pairs = []
    for index in [1, 2]:
        old = next(r for r in results if r['label'] == f'old-pr-0{index}')
        new = next(r for r in results if r['label'] == f'new-pr-0{index}')
        change = {k: (new[k] / old[k] - 1) * 100 for k in ['throughput_audio_s_s', 'total_content_tps']}
        for k in ['e2el_s', 'rtf']:
            for q in ['p50', 'p95']:
                change[f'{k}_{q}'] = (new[k][q] / old[k][q] - 1) * 100
        pairs.append(dict(pair=index, new_vs_old_percent=change,
                          device_peak_GiB_delta=new['device_peak_GiB'] - old['device_peak_GiB']))
    return dict(complete=True, results=results, pairs=pairs,
                scope='Eight fixed long songs per run; complete old/new/new/old order; descriptive small-sample comparison.')


if __name__ == '__main__':
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isclose(percentile([1, 2, 3, 4], 95), 3.85)
    output = analyze()
    (ROOT / 'evidence/analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
