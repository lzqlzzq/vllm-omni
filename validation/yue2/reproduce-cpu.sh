#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run the two recorded CPU suites using the active Python environment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'Usage: bash %s IMPLEMENTATION_CHECKOUT NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

checkout=$(cd -- "$1" && pwd)
if [[ ! -f "$checkout/tests/worker/test_yue2_acoustic_batch.py" ]]; then
    printf 'Not a YuE2 implementation checkout: %s\n' "$checkout" >&2
    exit 2
fi
python_bin=$(command -v python)

# Refuse existing output so a failed rerun cannot overwrite prior evidence.
mkdir -- "$2"
output=$(cd -- "$2" && pwd)
trap 'status=$?; printf "%s\n" "$status" > "$output/exit-code"' EXIT

cd "$checkout"
git rev-parse HEAD > "$output/commit.txt"
git status --porcelain=v1 --untracked-files=all > "$output/worktree-status.txt"
git diff HEAD --binary > "$output/tracked-changes.patch"

export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

"$python_bin" - > "$output/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys

versions = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": sorted(
        (package.metadata["Name"], package.version)
        for package in importlib.metadata.distributions()
    ),
}
print(json.dumps(versions, indent=2))
PY

run_suite() {
    local name=$1
    local status=0
    shift

    printf 'Running %s CPU regressions. Log: %s/%s.log\n' "$name" "$output" "$name"
    "$python_bin" -m pytest "$@" \
        -m 'core_model and cpu' --run-level core_model -q \
        > "$output/$name.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$output/$name.exit"
    if [[ $status -ne 0 ]]; then
        printf '%s CPU regressions failed (exit %s). See %s/%s.log\n' \
            "$name" "$status" "$output" "$name" >&2
    fi
    return "$status"
}

run_suite yue2 \
    tests/worker/test_yue2_*.py \
    tests/entrypoints/openai_api/test_tts_adapter.py \
    tests/core/sched/test_omni_cfg_ar_scheduler.py

run_suite shared \
    tests/worker/test_gpu_ar_model_runner.py \
    tests/core/test_prefix_cache.py \
    tests/engine/test_output_processor.py \
    tests/core/sched/test_omni_ar_scheduler_logprobs.py

printf 'CPU regressions completed. Logs: %s\n' "$output"
