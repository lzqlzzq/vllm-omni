#!/usr/bin/env python3
"""Verify published bytes and recompute reported metrics; no GPU or third-party dependencies."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), (actual, expected)


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def verify(root):
    manifest = read(root / 'SHA256.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    perf = root / 'performance'
    plan = read(perf / 'evidence/plan.json')
    for name, expected in plan['runtime_files'].items():
        assert hashlib.sha256((perf / name).read_bytes()).hexdigest() == expected, name
    inputs = rows(perf / 'evidence/reproduction_manifest.jsonl')
    labels = ['upstream-01', 'candidate-01', 'candidate-02', 'upstream-02']
    display = ['main 1', 'PR 1', 'PR 2', 'main 2']
    for label, shown in zip(labels, display):
        folder = perf / 'artifacts' / label
        summary, protocol = read(folder / 'summary.json'), read(folder / 'protocol.json')
        data = rows(folder / 'requests.jsonl')
        assert summary['complete'] and len(data) == summary['count'] == protocol['count'] == 32
        assert protocol['concurrency'] == 2 and protocol['warmup'] == 2 and not protocol['observe']
        assert protocol['max_abc'] == 4096 and protocol['max_frames'] == 9000
        assert read(folder / 'source-hashes.json') == read(folder / 'source-after.json')
        assert hashlib.sha256((folder / 'harness.py').read_bytes()).hexdigest() == protocol['harness_sha256']
        for filename, count, warm in [('requests.jsonl', 32, False), ('warmup.jsonl', 2, True)]:
            values = rows(folder / filename)
            expected = set()
            for i in range(count):
                entry = inputs[(190 + i) % 192 if warm else (1 + i * 5) % 192]
                seed = entry['candidate_ar_seeds'][i % 2]
                expected.add((f"{'warm' if warm else 'measure'}_{i}_{entry['prompt_index']}_{seed}", entry['prompt_index'], seed))
            assert len(values) == count and {(r['id'], r['prompt'], r['seed']) for r in values} == expected
        elapsed = max(r['end'] for r in data) - min(r['start'] for r in data)
        close(elapsed, summary['elapsed_s'])
        close(sum(r['audio_s'] for r in data) / elapsed, summary['throughput_audio_s_s'])
        close(sum(r['frames'] + r['abc_tokens'] for r in data) / elapsed, summary['total_content_tps'])
        for row in data:
            close(row['audio_s'], (row['frames'] * 1920 - 64) / 48000)
            close(row['rtf'], row['e2el_s'] / row['audio_s'])
        for metric in ['e2el_s', 'rtf']:
            for q in [50, 95]:
                close(percentile([r[metric] for r in data], q), summary[metric][f'p{q}'])
        device = [r for r in read(folder / 'device-memory.json') if summary['measured_start'] <= r[0] <= summary['measured_end']]
        assert max(r[1] for r in device) == summary['device_peak_measured_bytes']
        mem = summary['memory'][0][0]
        assert mem['allocator']['num_ooms'] == 0 and mem['model_states'] == 0
        print(shown, 'audio/s=', summary['throughput_audio_s_s'], 'TPS=', summary['total_content_tps'])
    scoring = root / 'wsb/scoring'
    full, extra = scoring / 'nar1-full-scores-01', scoring / 'nar1-additional-01'
    s, t = read(full / 'summary.json'), read(extra / 'summary.json')
    assert s['complete'] and t['complete'] and not s['errors'] and not t['errors']
    assert s['selected_candidates'] == t['selected_candidates'] and len(s['selected_candidates']) == 192
    key = lambda r: (r['prompt_index'], r['candidate_index'], r['audio_sha256'])
    chosen = {key(r) for r in s['selected_candidates']}
    all_rows, extra_rows = rows(full / 'per-song-numeric.jsonl'), rows(extra / 'per-song-numeric.jsonl')
    assert len(all_rows) == 384 and len({key(r) for r in all_rows}) == 384
    assert {(r['prompt_index'], r['candidate_index']) for r in all_rows} == {(p, c) for p in range(192) for c in range(2)}
    selected = [r for r in all_rows if key(r) in chosen]
    assert len(selected) == len(extra_rows) == 192 and {key(r) for r in extra_rows} == chosen
    # The evaluator retains the input row's pending status; metric status is authoritative.
    assert all(m['status'] == 'complete' for r in all_rows + extra_rows for m in r['metrics'].values())
    per = [r['metrics']['per'] for r in selected]
    scores = s['standard_selected']['groups']['all']
    close(sum(r['edits'] for r in per) / sum(r['reference_phonemes'] for r in per), scores['per']['per_micro'])
    for metric, value in scores['songbench']['means'].items():
        close(sum(r['metrics']['songbench']['scores'][metric] for r in selected) / 192, value)
    for metric in ['mulan', 'allmusiccaps']:
        close(sum(r['metrics'][metric]['scores'][metric] for r in extra_rows) / 192, t['standard_selected']['groups']['all'][metric]['means'][metric])
    assert sum(r['metrics']['per']['truncated'] for r in all_rows) == 11
    assert sum(r['metrics']['per']['truncated_repeats'] for r in all_rows) == 38
    assert sum(r['truncated'] for r in per) == 3
    generated = read(root / 'wsb/generation/generation_report.json')
    assert generated['complete'] and len(generated['results']) == generated['expected_candidates'] == 384
    print('Verified:', len(manifest), 'file hashes; 128 measured songs; 384 WSB outputs; 192 shared selections.')


if __name__ == '__main__':
    verify(ROOT)
