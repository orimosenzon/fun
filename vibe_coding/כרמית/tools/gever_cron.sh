#!/usr/bin/env bash
# Hourly (crontab): Carmit's requests in the "גבר גבר" tab of the admin page. When there is a new one, Claude makes
# the change in the app, publishes it, and answers her in the tab (see gever_prompt.md). With nothing new, this is one
# Firestore read and nothing else.
set -uo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8   # cron starts with no locale; Hebrew in notify-send and git needs one
export PATH="/home/ori/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=/home/ori/fun
STATE="$HOME/.cache/carmit_gever"
mkdir -p "$STATE"
LOG="$STATE/cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"
cd "$DIR"

notify() {
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
  notify-send -a "כרמית" -- "גבר גבר" "$1" 2>>"$LOG" || true
}

exec 9>"$STATE/lock"
flock -n 9 || { echo "$(date '+%F %T') previous run still going" >>"$LOG"; exit 0; }

COUNT="$(python3 tools/gever.py count 2>>"$LOG" | sed -n 's/^COUNT=//p')"
[ "${COUNT:-0}" -gt 0 ] || exit 0   # the common case: nothing new, nothing logged

echo "===== $(date '+%Y-%m-%d %H:%M:%S') gever: $COUNT new =====" >>"$LOG"
# Ori's own unfinished work in the project, or another branch: wait, rather than commit it along with the change.
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain -- "$DIR/app" "$DIR/firebase" "$DIR/tools")"
if [ "$BRANCH" != master ] || [ -n "$DIRTY" ]; then
  echo "waiting: branch=$BRANCH, uncommitted in the project:" >>"$LOG"
  printf '%s\n' "$DIRTY" >>"$LOG"
  # Once every few hours is enough of a reminder.
  if [ ! -f "$STATE/nagged" ] || [ -n "$(find "$STATE/nagged" -mmin +240)" ]; then
    notify "יש $COUNT בקשות מכרמית, אבל בכרמית יש שינויים לא מקומטים (או שלא על master). קומיט, והן יטופלו בשעה הבאה."
    touch "$STATE/nagged"
  fi
  exit 0
fi
rm -f "$STATE/nagged"

PENDING="$(python3 tools/gever.py pending 2>>"$LOG")" || { echo "could not read the requests" >>"$LOG"; exit 0; }
IDS="$(printf '%s' "$PENDING" | python3 -c 'import json,sys; print(" ".join(g["id"] for g in json.load(sys.stdin)))')"
[ -n "$IDS" ] || exit 0
python3 tools/gever.py status $IDS --to working >>"$LOG" 2>&1

PROMPT="$(cat tools/gever_prompt.md)

\`\`\`json
$PENDING
\`\`\`"
"$CLAUDE" -p "$PROMPT" \
  --model opus \
  --permission-mode acceptEdits \
  --allowedTools "Read,Write,Edit,Grep,Glob,Bash" \
  --disallowedTools "Bash(git push --force*)" "Bash(git push -f*)" "Bash(git reset --hard*)" "Bash(git add -A*)" "Bash(git commit -a*)" "Bash(rm -rf*)" "WebFetch" "WebSearch" \
  --max-budget-usd 5.00 \
  --no-session-persistence \
  >>"$LOG" 2>&1 && echo "agent ok" >>"$LOG" || echo "agent failed, see above" >>"$LOG"

RELEASED="$(python3 tools/gever.py release $IDS 2>>"$LOG")"
printf '%s\n' "$RELEASED" >>"$LOG"
if printf '%s' "$RELEASED" | grep -q STUCK; then
  notify "בקשה של כרמית נתקעה פעמיים. הלוג: $LOG"
else
  notify "טופלו $COUNT בקשות מכרמית. התשובות בלשונית גבר גבר, הלוג: $LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
