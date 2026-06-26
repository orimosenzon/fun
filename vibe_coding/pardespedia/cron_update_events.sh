#!/usr/bin/env bash
# Unattended refresh of the pardespedia events guide (called from crontab).
# Logs to events_cron.log in this directory so failures are inspectable.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/events_cron.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$LOG"
/usr/bin/env python3 "$DIR/update_events.py" "$@" >>"$LOG" 2>&1
echo "exit: $? (ok)" >>"$LOG"
