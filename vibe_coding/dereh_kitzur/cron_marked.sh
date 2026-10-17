#!/usr/bin/env bash
# Nightly refresh of the שבילים מסומנים layer (called from crontab).
# build_marked.py --publish downloads Geofabrik's Israel extract (rebuilt from
# the live OSM database around 23:20 UTC), pulls out every marked hiking route
# and puts data/marked.json in the data repo, where the app reads it. A night
# with no change to any trail makes no commit.
set -uo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/marked_cron.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') marked =====" >>"$LOG"
/usr/bin/env python3 "$DIR/build_marked.py" --publish >>"$LOG" 2>&1
STATUS=$?
# Geofabrik or GitHub was down: tomorrow's run picks it all up.
echo "exit: $STATUS" >>"$LOG"
# The log grows by three lines a night; keep the last few months.
tail -n 400 "$LOG" >"$LOG.tmp" && mv "$LOG.tmp" "$LOG"
exit 0
