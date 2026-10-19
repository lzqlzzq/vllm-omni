#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Reconstruct and run the measured f9143976 snapshots in a new directory.
set -euo pipefail

: "${YUE2_RUN_DIR:?Set an unused absolute output directory}"
: "${YUE2_MODEL:?Set the pinned YuE2-3B snapshot directory}"
: "${YUE2_VAE:?Set the pinned legacy VAE snapshot directory}"
: "${HF_HOME:?Set the populated Hugging Face cache}"
: "${CUDA_HOME:?Set the CUDA 13 toolkit directory}"

python_bin=$(command -v python)
package=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$package" rev-parse --show-toplevel)
base_commit=f9143976b688cf1e78796ee080d7294f6e019960

if [[ "$YUE2_RUN_DIR" != /* ]]; then
    printf 'YUE2_RUN_DIR must be absolute: %s\n' "$YUE2_RUN_DIR" >&2
    exit 2
fi
# Refuse existing output so a failed rerun cannot overwrite prior evidence.
mkdir -p -- "$(dirname "$YUE2_RUN_DIR")"
mkdir -- "$YUE2_RUN_DIR"
cp -R "$package/performance/scripts" \
    "$package/performance/configs" \
    "$package/performance/evidence" \
    "$YUE2_RUN_DIR/"
mkdir "$YUE2_RUN_DIR/artifacts"

for source in upstream source; do
    mkdir "$YUE2_RUN_DIR/$source"
    git -C "$repo" archive "$base_commit" \
        | tar -xf - -C "$YUE2_RUN_DIR/$source"
done
git -C "$YUE2_RUN_DIR/upstream" apply "$package/performance/evidence/shared-loading.patch"
git -C "$YUE2_RUN_DIR/source" apply "$package/performance/evidence/candidate.patch"

"$python_bin" - "$YUE2_RUN_DIR" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
for arm, source in [("upstream", "upstream"), ("candidate", "source")]:
    expected = json.loads((root / f"evidence/{arm}-source.json").read_text())
    for name, digest in expected.items():
        actual = hashlib.sha256((root / source / name).read_bytes()).hexdigest()
        assert actual == digest, name

assets = json.loads((root / "evidence/assets.json").read_text())
for item, key in zip(assets, ["YUE2_MODEL", "YUE2_VAE"]):
    model_path = Path(os.environ[key]).resolve()
    item["path"] = str(model_path)
    with (model_path / "model.safetensors").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], key

(root / "evidence/assets.json").write_text(json.dumps(assets, indent=2) + "\n")
print("Historical main/PR source and model weight hashes verified.")
PY

export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 YUE2_MPS=off
unset LD_LIBRARY_PATH CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

# Keep archived labels so source hashes and the original analyzers remain valid.
cd "$YUE2_RUN_DIR"
for label in upstream-01 candidate-01 candidate-02 upstream-02; do
    export YUE2_ARM=${label%-*}
    if [[ "$YUE2_ARM" = candidate ]]; then
        export YUE2_SOURCE=source
        shown=PR
    else
        export YUE2_SOURCE=upstream
        shown=main
    fi
    export PYTHONPATH="$YUE2_RUN_DIR/$YUE2_SOURCE:$YUE2_RUN_DIR/scripts"
    printf 'Running %s (%s)\n' "$shown" "$label"
    "$python_bin" scripts/bench_e2e.py \
        --label "$label" \
        --config "configs/$YUE2_ARM.yaml" \
        --count 32 --concurrency 2 --offset 1 --warmup 2 \
        > "artifacts/$label.log" 2>&1
done

"$python_bin" scripts/analyze.py \
    upstream-01 candidate-01 candidate-02 upstream-02 > artifacts/analysis.log
"$python_bin" scripts/compare.py > artifacts/comparison.log
