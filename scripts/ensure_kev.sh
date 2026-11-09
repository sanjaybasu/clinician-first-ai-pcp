#!/bin/zsh
# Start the local Kev server (System-1 judge) on :8009 if it is not running, and wait until it answers.
# KEV_DIR is a checkout of https://github.com/jaredpalmer/kev (scripts/setup_vendor.sh makes one at the pinned commit).
K=${KEV_DIR:-~/waymark-local/packaging/omission-risk-set-nba/tools/kev}
LOG=${COMPASS_PCP_DATA:-~/waymark-local/packaging/compass-pcp-data}/kev_serve.log
mkdir -p "$(dirname $LOG)"
[ -d "$K" ] || { echo "no Kev checkout at $K; run scripts/setup_vendor.sh or set KEV_DIR" >&2; exit 1; }
pgrep -f "kev.serve --run jaredpalmer/kev-4b" >/dev/null || (cd $K && nohup uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009 > $LOG 2>&1 &)
until curl -s -m 2 -o /dev/null localhost:8009/docs; do sleep 5; done
echo "kev up on :8009"
