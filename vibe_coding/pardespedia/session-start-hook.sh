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

# סבב השיפור הדו-שבועי (cron_sweep.sh): סבב באמצע, או סיכום שעוד לא נקרא.
sweep_out="$(cd "$DIR" && timeout 20 python3 sweep_cron.py status 2>/dev/null)" || sweep_out=""
if echo "$sweep_out" | grep -q "^active run"; then
  echo "סבב השיפור הדו-שבועי באמצע: $(echo "$sweep_out" | sed -n 's/^active run \([^:]*\): done \([0-9]*\), pending \([0-9]*\).*/\1, \2 אצוות גמורות, \3 ממתינות/p')."
fi
latest="$(ls -d "$HOME"/.cache/pardespedia_sweep/cron/runs/*/SUMMARY.md 2>/dev/null | tail -1)"
if [ -n "$latest" ] && [ ! -e "$latest.seen" ]; then
  echo "סבב שיפור הסתיים, יש סיכום שלא נקרא: $latest (אחרי שעוברים עליו: touch $latest.seen)."
fi

# בדיקת הודעות ממפעילי מערכת שטרם נענו בדף השיחה של הבוט (ראה check_bot_talk.py).
# רק דגל בשלב זה — הטיפול בפועל נעשה ע"י הסשן החי (Claude), לא ע"י ה-hook.
talk_out="$(cd "$DIR" && timeout 30 python3 check_bot_talk.py 2>&1)" || talk_out=""
if echo "$talk_out" | grep -q "^SYSOP_USER="; then
  sysop="$(echo "$talk_out" | sed -n 's/^SYSOP_USER=//p' | tail -1)"
  echo "יש הודעה ממתינה ממפעיל המערכת '${sysop}' בדף השיחה של הבוט — טרם נענתה."
fi
