#!/usr/bin/env bash
set -euo pipefail
export YUE2_ARM=candidate YUE2_SOURCE=source
source /root/autodl-tmp/yue2-rebase-abba-20261014-01/scripts/env.sh
exec > artifacts/validation.log 2>&1
trap 'code=$?; printf "EXIT %s %s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
cd source
python -m pytest tests/worker/test_gpu_ar_model_runner.py tests/core/test_prefix_cache.py tests/engine/test_output_processor.py -m 'core_model and cpu' --run-level core_model -q > ../artifacts/upstream-contracts.log 2>&1
cd ..
for slots in 1 2; do
    SMOKE_SLOTS="$slots" python scripts/smoke.py > "artifacts/gpu-$slots.log" 2>&1
done
python scripts/http_smoke.py --label http > artifacts/http.log 2>&1
printf 'VALIDATION_COMPLETE %s\n' "$(date -Is)"
