#!/usr/bin/env bash
# Unattended tidy of the committee-meetings board on [[ועדות המועצה]]
# (called from crontab). The rows themselves are typed by hand; this only
# moves meetings that have already happened into the archive section.
# Logs to committees_cron.log in this directory so failures are inspectable.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/committees_cron.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$LOG"
/usr/bin/env python3 "$DIR/update_committees.py" "$@" >>"$LOG" 2>&1
echo "exit: $? (ok)" >>"$LOG"

# Second job on the same page: rebuild the table of plenum sittings from the
# municipality's protocol archive. Kept in this script rather than its own
# cron line so the two edits to [[ועדות המועצה]] never race each other.
# `set -e` is relaxed for it: a municipality site that is down for a morning
# must not look like a failure of the archiving above, which already ran.
echo "--- ישיבות מליאה ---" >>"$LOG"
set +e
/usr/bin/env python3 "$DIR/council_protocols.py" >>"$LOG" 2>&1
echo "protocols exit: $?" >>"$LOG"
