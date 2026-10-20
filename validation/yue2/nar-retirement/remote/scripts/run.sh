#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/yue2-retire-perf-39268
base=/root/autodl-tmp/yue2-sync-39268
cd "$root"
exec > evidence/controller.log 2>&1
trap 'rc=$?; echo "$rc" > /root/autodl-tmp/yue2-retire-perf-39268/evidence/controller.exit; date -Is; exit "$rc"' EXIT
export PATH="$base/venv/bin:/usr/local/cuda/bin:$PATH"
export HF_HOME="$base/hf-cache" HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$base/tmp"
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF CUDA_MPS_PIPE_DIRECTORY YUE2_PROBE_OUT
for label in old-pr-01 new-pr-01 new-pr-02 old-pr-02; do
    arm=${label%-*}
    export PYTHONPATH="$root/source-$arm:$root/scripts"
    python - <<'PY'
import pynvml
pynvml.nvmlInit()
assert not pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(0))
pynvml.nvmlShutdown()
PY
    printf 'START %s %s\n' "$label" "$(date -Is)"
    python scripts/sync_bench.py --arm "$arm" --label "$label" --count 8 --warmup 2 > "evidence/$label.log" 2>&1
    printf 'COMPLETE %s %s\n' "$label" "$(date -Is)"
done
printf 'ABBA_COMPLETE %s\n' "$(date -Is)"
