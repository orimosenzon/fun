#!/usr/bin/env bash
# Daily check of the moshava's objection windows (called from crontab).
# watch_objections.py writes reports/objections/<date>.md only when something
# opened, is about to shut, shut, or entered deposit; on those days this also
# pops a desktop notification. Read-only against everything but its own state.
set -uo pipefail
# cron starts with no locale, and notify-send then rejects Hebrew outright.
export LANG=C.UTF-8 LC_ALL=C.UTF-8
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/objections_cron.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') objections =====" >>"$LOG"
OUT="$(/usr/bin/env python3 "$DIR/watch_objections.py" 2>>"$LOG")"
STATUS=$?
printf '%s\n' "$OUT" >>"$LOG"
if [ $STATUS -ne 0 ]; then
  # Usually the planning service's firewall. Tomorrow's run compares against
  # the last good state, so a missed day loses nothing but a day.
  echo "exit: $STATUS (service unreachable?)" >>"$LOG"
  exit 0
fi

REPORT="$(printf '%s\n' "$OUT" | sed -n 's/^REPORT=//p' | tail -1)"
NEWS="$(printf '%s\n' "$OUT" | sed -n 's/^NEWS=//p' | tail -1)"
if [ -n "$REPORT" ] && [ "${NEWS:-0}" -gt 0 ]; then
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
  notify-send -a "דרך קיצור" "חלונות התנגדות: $NEWS עדכונים" \
    "$(grep -m3 '^- ' "$REPORT" | sed 's/\[[^]]*\]([^)]*)//g; s/\*\*//g')" \
    2>>"$LOG" || true
  echo "REPORT READY: $REPORT" >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
