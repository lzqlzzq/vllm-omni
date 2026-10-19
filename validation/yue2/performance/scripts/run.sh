#!/usr/bin/env bash
set -euo pipefail
export YUE2_ARM=candidate YUE2_SOURCE=source
source /root/autodl-tmp/yue2-rebase-abba-20261014-01/scripts/env.sh
exec > artifacts/controller.log 2>&1
trap 'code=$?; printf "EXIT %s %s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
test -f artifacts/gpu-1.json
test -f artifacts/gpu-2.json
python scripts/preflight.py
for arm in upstream candidate; do
    export YUE2_ARM="$arm"
    if [[ "$arm" == candidate ]]; then export YUE2_SOURCE=source; else export YUE2_SOURCE=upstream; fi
    source scripts/env.sh
    python scripts/bench_e2e.py --label "smoke-$arm" --config "configs/$arm.yaml" --count 4 --concurrency 2 --offset 1 --warmup 1 --max-abc 128 --max-frames 256 > "artifacts/smoke-$arm.log" 2>&1
    printf 'SMOKE_COMPLETE %s %s\n' "$arm" "$(date -Is)"
done
for label in upstream-01 candidate-01 candidate-02 upstream-02; do
    arm=${label%-*}
    export YUE2_ARM="$arm"
    if [[ "$arm" == candidate ]]; then export YUE2_SOURCE=source; else export YUE2_SOURCE=upstream; fi
    source scripts/env.sh
    printf 'FORMAL_START %s %s\n' "$label" "$(date -Is)"
    python scripts/bench_e2e.py --label "$label" --config "configs/$arm.yaml" --count 32 --concurrency 2 --offset 1 --warmup 2 > "artifacts/$label.log" 2>&1
    printf 'FORMAL_COMPLETE %s %s\n' "$label" "$(date -Is)"
done
python scripts/analyze.py upstream-01 candidate-01 candidate-02 upstream-02 > artifacts/analysis.log 2>&1
python scripts/compare.py > artifacts/comparison.log 2>&1
printf 'ABBA_COMPLETE %s\n' "$(date -Is)"
