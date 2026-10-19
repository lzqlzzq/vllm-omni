#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/yue2-wsb-4090d-20261016-01/scripts/env.sh
trap 'rc=$?; echo "$rc" > evidence/canary.exit; date -Is; exit "$rc"' EXIT
[ "$(cat evidence/setup.exit)" = 0 ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
python scripts/check_capture.py
python scripts/generate_wsb.py --label nar1-canary-01 --prompt-index 0 --prompt-index 100
