#!/usr/bin/env bash
# Nightly refresh of the פסטיבל דרך הנדיב layer (called from crontab).
# build_hanadiv.py --publish pulls the festival's API, places every venue and
# puts data/hanadiv.json in the data repo, where the app reads it; a night
# with no change makes no commit. A new venue the script cannot place gets no
# pin, so those nights pop a desktop notification naming it, for a FIXES entry.
# After the festival there is nothing left to refresh and the run is a no-op.
set -uo pipefail
# cron starts with no locale, and notify-send then rejects Hebrew outright.
export LANG=C.UTF-8 LC_ALL=C.UTF-8
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/hanadiv_cron.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') hanadiv =====" >>"$LOG"

UNTIL="$(python3 -c 'import json; print(json.load(open("web/data/hanadiv.json"))["festival"]["until"])' 2>/dev/null)"
if [ -n "$UNTIL" ] && [[ "$(date +%F)" > "$UNTIL" ]]; then
  echo "the festival ended $UNTIL; nothing to do" >>"$LOG"
  exit 0
fi

ERR="$(mktemp)"
OUT="$(/usr/bin/env python3 "$DIR/build_hanadiv.py" --publish 2>"$ERR")"
STATUS=$?
printf '%s\n' "$OUT" >>"$LOG"
cat "$ERR" >>"$LOG"
if [ $STATUS -ne 0 ]; then
  # The festival's API or GitHub was down; tomorrow's run picks it all up.
  echo "exit: $STATUS" >>"$LOG"
  rm -f "$ERR"
  exit 0
fi

# Venues known to be on no open map (see the docstring); anything else
# unplaced is new and wants a FIXES entry.
KNOWN='חוות החזון הירוק|הנורית|קריית החינוך|ירושלים 52|שפת הים 13'
NEW="$(sed -n 's/^  ! no position for //p' "$ERR" | grep -Ev "$KNOWN" | sort -u)"
rm -f "$ERR"
if [ -n "$NEW" ]; then
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
  notify-send -a "דרך קיצור" -- "דרך הנדיב: כתובות חדשות בלי מיקום" "$NEW" 2>>"$LOG" || true
  echo "UNPLACED: $(printf '%s' "$NEW" | tr '\n' ';')" >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
