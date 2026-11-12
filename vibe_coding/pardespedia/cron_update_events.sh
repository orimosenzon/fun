#!/usr/bin/env bash
# Unattended refresh of the pardespedia events guide (called from crontab).
# Logs to events_cron.log in this directory so failures are inspectable.
# A failure is also posted to the bot's talk page (once per streak) by
# cron_alert.py, because a silent log let the board go stale for four nights.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/events_cron.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$LOG"
/usr/bin/env python3 "$DIR/update_events.py" "$@" >>"$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "exit: 0 (ok)" >>"$LOG"
  /usr/bin/env python3 "$DIR/cron_alert.py" ok events >>"$LOG" 2>&1
else
  echo "exit: $rc (FAILED)" >>"$LOG"
  /usr/bin/env python3 "$DIR/cron_alert.py" fail events "$LOG" >>"$LOG" 2>&1
fi
exit $rc
