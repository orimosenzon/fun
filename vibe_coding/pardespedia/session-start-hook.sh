#!/usr/bin/env bash
# Session-start hook for pardespedia.
# Picked up automatically by the generic session-start skill, which runs any
# <project-root>/session-start-hook.sh it finds. Here: refresh the culture &
# events guide so a new session always opens on fresh data.
# Prints ONE concise Hebrew status line (the skill folds it into the briefing).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

out="$(cd "$DIR" && timeout 180 python3 update_events.py 2>&1)" || {
  echo "לוח אירועי התרבות: העדכון נכשל (ראה events_cron.log)."
  exit 0
}

if echo "$out" | grep -q "Edited:"; then
  rev="$(echo "$out" | grep -oE 'rev [0-9]+' | tail -1)"
  n="$(echo "$out" | grep -oE '[0-9]+ events after' | grep -oE '^[0-9]+')"
  echo "לוח אירועי התרבות עודכן (${rev}, ${n} אירועים)."
elif echo "$out" | grep -q "No change"; then
  echo "לוח אירועי התרבות נבדק — אין שינוי."
else
  echo "לוח אירועי התרבות: ההרצה הושלמה."
fi
