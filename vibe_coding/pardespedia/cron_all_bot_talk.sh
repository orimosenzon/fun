#!/usr/bin/env bash
# Hourly check across ALL user-talk pages the bot has participated in (not
# just the bot's own talk page) for unanswered replies, then invokes an agent
# to respond. Supersedes cron_bot_talk.sh / check_bot_talk.py, which only
# ever watched שיחת_משתמש:אורי מוסנזון בוט and missed replies posted on other
# users' own talk pages (e.g. after cron_contributor_comments.sh starts a
# thread there). See check_all_bot_talk.py docstring for the detection logic.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/all_bot_talk_cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') all_bot_talk =====" >>"$LOG"

CHECK_OUT="$(/usr/bin/env python3 "$DIR/check_all_bot_talk.py" --hours 26 2>>"$LOG")"
printf '%s\n' "$CHECK_OUT" >>"$LOG"
COUNT="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^COUNT=//p' | tail -1)"
COUNT="${COUNT:-0}"
PAGES="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^PAGES=//p' | tail -1)"

if [ "$COUNT" -gt 0 ] && [ -n "${PAGES:-}" ] && [ -x "$CLAUDE" ]; then
  echo "$COUNT pending page(s): $PAGES — invoking agent" >>"$LOG"
  PAGES_LIST="$(printf '%s' "$PAGES" | tr '|' '\n' | sed 's/^/* /')"
  PROMPT="$(cat "$DIR/all_bot_talk_prompt.txt")
$PAGES_LIST"
  "$CLAUDE" -p "$PROMPT" \
    --model sonnet \
    --permission-mode acceptEdits \
    --allowedTools "Read,Write,Edit,Grep,Glob,Bash,WebSearch,WebFetch" \
    --max-budget-usd 2.00 \
    --no-session-persistence \
    >>"$LOG" 2>&1 && echo "agent stage ok" >>"$LOG" \
    || echo "agent stage failed (non-fatal) — see log; message(s) still awaiting reply" >>"$LOG"
else
  echo "no pending replies; nothing to do." >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
