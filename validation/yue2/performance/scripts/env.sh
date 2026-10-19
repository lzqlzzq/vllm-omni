export YUE2_ARM=${YUE2_ARM:?arm required}
export YUE2_SOURCE=${YUE2_SOURCE:?source required}
source /root/autodl-tmp/yue2-4090-e2e-20261012-01/scripts/env.sh
export YUE2_ROOT=/root/autodl-tmp/yue2-rebase-abba-20261014-01
export PYTHONPATH="$YUE2_ROOT/$YUE2_SOURCE:$YUE2_ROOT/scripts"
cd "$YUE2_ROOT"
