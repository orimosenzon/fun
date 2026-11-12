#!/usr/bin/env bash
# Monthly refresh of the [[פרדספדיה]] article: charts + numbers table
# (crontab, 1st of the month). Failures go to the bot's talk page via cron_alert.py.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/about_cron.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$LOG"
/usr/bin/env python3 "$DIR/update_about_page.py" >>"$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "exit: 0 (ok)" >>"$LOG"
  /usr/bin/env python3 "$DIR/cron_alert.py" ok about >>"$LOG" 2>&1
else
  echo "exit: $rc (FAILED)" >>"$LOG"
  /usr/bin/env python3 "$DIR/cron_alert.py" fail about "$LOG" >>"$LOG" 2>&1
fi
exit $rc
