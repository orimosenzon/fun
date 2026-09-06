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
