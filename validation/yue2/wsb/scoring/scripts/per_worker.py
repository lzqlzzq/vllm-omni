"""Frozen PER worker with its redundant parent-process sync removed in memory."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import sys


def without_parent_sync(source):
    anchor = '            torch.cuda.synchronize(); elapsed = time.perf_counter()-started'
    assert source.count(anchor) == 1
    return source.replace(anchor, '            elapsed = time.perf_counter()-started')


if __name__ == '__main__':
    entry = Path(sys.argv[1]).resolve()
    expected = Path('/root/autodl-tmp/yue2-wsb-4090d-20261016-01/wsb_metrics/wsb_eval.py')
    assert entry == expected and sys.argv[2] == '_worker'
    assert sys.argv[sys.argv.index('--metric') + 1] == 'per'
    report = Path(sys.argv[sys.argv.index('--jobs') + 1]).resolve().parent
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location('frozen_wsb_api', entry)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.verify_release()
    from mot_infer import eval_worker
    original = inspect.getsource(eval_worker.run)
    executed = without_parent_sync(original)
    (report / 'per-worker.original.py').write_text(original)
    target = report / 'per-worker.executed.py'
    target.write_text(executed)
    provenance = {
        'scope': 'Only PER parent synchronization after blocking ASR results; all scorer parameters unchanged',
        'frozen_release_verified': True,
        'original_run_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'executed_run_sha256': hashlib.sha256(executed.encode()).hexdigest(),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parent_context_preinitialization': False,
        'official_source_files_modified': False,
    }
    (report / 'per-worker-execution.json').write_text(json.dumps(provenance, indent=2))
    print('PER_EXTERNAL_EXECUTION_OVERRIDE', json.dumps(provenance), flush=True)
    exec(compile(executed, str(target), 'exec'), eval_worker.__dict__)
    sys.argv = sys.argv[1:]
    runpy.run_path(str(entry), run_name='__main__')
