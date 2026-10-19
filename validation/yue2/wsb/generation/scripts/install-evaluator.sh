#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/yue2-wsb-4090d-20261016-01
exec > >(tee -a evidence/install-evaluator.log) 2>&1
trap 'rc=$?; echo "$rc" > evidence/install-evaluator.exit; date -Is; exit "$rc"' EXIT
echo 'bd550b36f1fc43cad3254a9897dd3a078e50c4eb9e8d4f295ac4b5e81d1fc4fd  evaluator-preview.tar.gz' | sha256sum -c -
tar --no-same-owner -xzf evaluator-preview.tar.gz
mkdir -p wsb_metrics/.runtime
[ -e wsb_metrics/.runtime/python ] || ln -s "$PWD/python" wsb_metrics/.runtime/python
mkdir -p evaluator-install-inputs
cp wsb_metrics/configs/*.installed.lock.txt evaluator-install-inputs/
cp wsb_metrics/vendor/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl evaluator-install-inputs/
cp wsb_metrics/bin/uv bin/evaluator-uv
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_CACHE_DIR="$PWD/wsb_metrics/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/python" UV_LINK_MODE=copy TMPDIR="$PWD/tmp"
uv="$PWD/bin/evaluator-uv"
for role in eval metrics; do
  python="$PWD/wsb_metrics/.venvs/$role/bin/python"
  [ -x "$python" ] || "$uv" venv --python "$PWD/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12" "${python%/bin/python}"
  if [ "$role" = metrics ]; then
    "$uv" pip install --python "$python" --no-deps evaluator-install-inputs/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
  fi
  "$uv" pip install --python "$python" -r "evaluator-install-inputs/$role.installed.lock.txt"
done
