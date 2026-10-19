#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/yue2-wsb-4090d-20261016-01/scripts/env.sh
exec > >(tee -a evidence/evaluator-assets.log) 2>&1
trap 'rc=$?; echo "$rc" > evidence/evaluator-assets.exit; date -Is; exit "$rc"' EXIT
HF_HUB_OFFLINE=0 python wsb_metrics/wsb_eval.py prepare --metrics per,songbench,mulan,allmusiccaps --download hf-mirror
