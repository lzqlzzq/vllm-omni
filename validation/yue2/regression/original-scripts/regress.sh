#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/yue2-repatch-20261019
old=/root/autodl-tmp/yue2-wsb-4090d-20261016-01
mkdir -p "$root/evidence"
exec > "$root/evidence/regress.log" 2>&1
trap 'rc=$?; echo "$rc" > "$root/evidence/regress.exit"; date -Is; exit "$rc"' EXIT
while [ ! -f "$root/evidence/restore-env.exit" ]; do sleep 10; done
[ "$(cat "$root/evidence/restore-env.exit")" = 0 ]
export PATH="$root/venv/bin:$old/bin:$PATH" UV_CACHE_DIR="$root/uv-cache" TMPDIR="$root/tmp"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$root/source"
python - <<'PY'
import json,hashlib,pathlib
root=pathlib.Path('..')
m=json.loads((root/'source-manifest.json').read_text())
for name,h in m.items(): assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==h,name
print('SOURCE_SHA_VERIFIED',len(m))
PY
VLLM_OMNI_VERSION_OVERRIDE=0.0.0+yue2.repatch VLLM_TARGET_DEVICE=cuda uv pip install --python "$root/venv/bin/python" --no-deps -e .
uv pip check --python "$root/venv/bin/python"
python -m pytest tests/worker/test_yue2_*.py tests/entrypoints/openai_api/test_tts_adapter.py tests/core/sched/test_omni_cfg_ar_scheduler.py -m 'core_model and cpu' --run-level core_model -q > "$root/evidence/yue2-cpu.log" 2>&1
python -m pytest tests/worker/test_gpu_ar_model_runner.py tests/core/test_prefix_cache.py tests/engine/test_output_processor.py tests/core/sched/test_omni_ar_scheduler_logprobs.py -m 'core_model and cpu' --run-level core_model -q > "$root/evidence/shared-cpu.log" 2>&1
printf 'REGRESS_COMPLETE\n'
