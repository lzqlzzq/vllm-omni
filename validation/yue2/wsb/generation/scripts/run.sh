#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/yue2-wsb-4090d-20261016-01/scripts/env.sh
exec > >(tee -a evidence/controller.log) 2>&1
trap 'rc=$?; echo "$rc" > evidence/controller.exit; date -Is; exit "$rc"' EXIT
echo $$ > evidence/controller.pid
date -Is > evidence/controller.started
[ "$(cat evidence/setup.exit)" = 0 ]
[ "$(cat evidence/system.exit)" = 0 ]
step() {
  echo "$1" > evidence/stage
  date -Is
  local name="$1"; shift
  "$@" > "evidence/$name.log" 2>&1
}
nvidia-smi -q > evidence/gpu-before.txt
python - <<'ENV'
import importlib.metadata as m
import json, os, platform
from pathlib import Path
Path('evidence/runtime.json').write_text(json.dumps(dict(
    python=platform.python_version(), platform=platform.platform(),
    torch=m.version('torch'), vllm=m.version('vllm'),
    flashinfer=m.version('flashinfer-python'),
    environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','CUDA_MPS_PIPE_DIRECTORY']}
),indent=2)+'\n')
ENV
step canary bash scripts/canary.sh
python - <<'CHECK'
import json
from pathlib import Path
p=Path('artifacts/nar1-canary-01')
r=json.loads((p/'generation_report.json').read_text())
assert r['complete'] and len(r['results']) == r['expected_candidates'] == 4
CHECK
step full-generation python scripts/generate_wsb.py --label nar1-full-01
while screen -ls | grep -Eq 'yue2-wsb-assets|yue2-wsb-evaluator-env'; do sleep 30; done
step prepare-evaluator env HF_HUB_OFFLINE=0 python wsb_metrics/wsb_eval.py prepare --metrics per,songbench --download hf-mirror --install
step canary-score python wsb_metrics/wsb_eval.py score --input artifacts/nar1-canary-01/evaluation_inputs.jsonl --output artifacts/nar1-canary-scores-01 --mode shard --metrics per,songbench
python - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('artifacts/nar1-canary-scores-01/summary.json').read_text())
assert r['complete'] and r['candidate_count'] == 4, r
CHECK
step full-score python wsb_metrics/wsb_eval.py score --input artifacts/nar1-full-01/evaluation_inputs.jsonl --output artifacts/nar1-full-scores-01 --mode standard --metrics per,songbench
step prepare-additional env HF_HUB_OFFLINE=0 python wsb_metrics/wsb_eval.py prepare --metrics mulan,allmusiccaps --download hf-mirror
mkdir artifacts/nar1-additional-01
ln -s ../nar1-full-scores-01/audio artifacts/nar1-additional-01/audio
step additional-score python wsb_metrics/wsb_eval.py score --input artifacts/nar1-full-01/evaluation_inputs.jsonl --per-results artifacts/nar1-full-scores-01/per_song.jsonl --output artifacts/nar1-additional-01 --metrics mulan,allmusiccaps
echo nar1-complete > evidence/stage
step nar2-comparison python scripts/nar2_compare.py
echo complete > evidence/stage
