#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/yue2-wsb-4090d-20261016-01
audit=/root/autodl-tmp/yue2-wsb-acceptance-20261019
python="$root/wsb_metrics/.venvs/metrics/bin/python"
exec > "$audit/evidence/nar1-scores.log" 2>&1
trap 'rc=$?; echo "$rc" > "$audit/evidence/nar1-scores.exit"; date -Is; exit "$rc"' EXIT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1
[ "$(cat "$audit/evidence/per-no-parent-sync.exit")" = 0 ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
"$python" - "$root" "$audit" <<'PY'
import json,sys
from pathlib import Path
root,audit=map(Path,sys.argv[1:])
source=root/'artifacts/nar1-full-01'
report=json.loads((source/'generation_report.json').read_text())
assert report['complete'] and len(report['results'])==384
longest=sorted(report['results'],key=lambda x:(-x['frames'],x['id']))[:8]
ids={x['id'] for x in longest}
rows=[json.loads(x) for x in (source/'evaluation_inputs.jsonl').read_text().splitlines()]
selected=[]
for row in rows:
    path=(source/row['path']).resolve()
    if path.name in ids:
        selected.append({**row,'path':str(path)})
assert len(selected)==8
(audit/'longest8.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
(audit/'evidence/longest8-selection.json').write_text(json.dumps({'rule':'largest frames, tie by id; capacity check only; full384 unchanged','songs':longest},indent=2))
PY
echo per-longest8 > "$audit/evidence/stage"
"$python" "$root/wsb_metrics/wsb_eval.py" score --input "$audit/longest8.jsonl" \
  --output "$audit/per-longest8-01" --mode shard --metrics per --per-python "$audit/scripts/per-python"
"$python" - "$audit/per-longest8-01/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); assert s['complete'] and s['candidate_count']==8
PY
echo nar1-full-score > "$audit/evidence/stage"
date -Is
"$python" "$root/wsb_metrics/wsb_eval.py" score \
  --input "$root/artifacts/nar1-full-01/evaluation_inputs.jsonl" \
  --output "$audit/nar1-full-scores-01" --mode standard --metrics per,songbench \
  --per-python "$audit/scripts/per-python"
echo nar1-selected-additional > "$audit/evidence/stage"
mkdir -p "$audit/nar1-additional-01"
ln -s ../nar1-full-scores-01/audio "$audit/nar1-additional-01/audio"
"$python" "$root/wsb_metrics/wsb_eval.py" score \
  --input "$root/artifacts/nar1-full-01/evaluation_inputs.jsonl" \
  --per-results "$audit/nar1-full-scores-01/per_song.jsonl" \
  --output "$audit/nar1-additional-01" --metrics mulan,allmusiccaps
echo nar1-scores-complete > "$audit/evidence/stage"
