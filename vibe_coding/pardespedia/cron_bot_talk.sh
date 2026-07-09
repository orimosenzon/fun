#!/usr/bin/env bash
# Unattended check of the bot's own talk page for unanswered sysop requests
# (called from crontab). Unlike cron_enrich.sh, this stage IS allowed to edit
# the wiki (Bash + the pardespedia CLI) — sysop-authored requests are the
# trigger, and the prompt instructs conservative, reversible-only action.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/bot_talk_cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') bot_talk =====" >>"$LOG"

CHECK_OUT="$(/usr/bin/env python3 "$DIR/check_bot_talk.py" 2>>"$LOG")"
printf '%s\n' "$CHECK_OUT" >>"$LOG"
COUNT="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^COUNT=//p' | tail -1)"
COUNT="${COUNT:-0}"
SYSOP_USER="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^SYSOP_USER=//p' | tail -1)"
REV="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^REV=//p' | tail -1)"

if [ "$COUNT" -gt 0 ] && [ -x "$CLAUDE" ]; then
  echo "pending sysop message from '$SYSOP_USER' (rev $REV) — invoking agent" >>"$LOG"
  PROMPT="$(cat "$DIR/bot_talk_prompt.txt")

הודעה ממתינה מאת: $SYSOP_USER (רוויזיה $REV) בדף שיחת_משתמש:אורי מוסנזון בוט."
  "$CLAUDE" -p "$PROMPT" \
    --model sonnet \
    --permission-mode acceptEdits \
    --allowedTools "Read,Write,Edit,Grep,Glob,Bash,WebSearch,WebFetch" \
    --max-budget-usd 2.00 \
    --no-session-persistence \
    >>"$LOG" 2>&1 && echo "agent stage ok" >>"$LOG" \
    || echo "agent stage failed (non-fatal) — see log; message still awaiting reply" >>"$LOG"
else
  echo "no pending sysop message; nothing to do." >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
