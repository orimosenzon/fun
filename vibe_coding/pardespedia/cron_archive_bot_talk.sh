#!/usr/bin/env bash
# Daily archiving of the bot's user-talk page: threads with no new message in
# the last 7 days move to the numbered archive subpage, so the live page shows
# only what is actually current. Pure Python, no agent and no API cost.
#
# Timed at 06:20 to sit clear of the other daily jobs (enrich 06:30,
# contributor comments 06:50) and away from the hourly cron_all_bot_talk.sh,
# which fires on the hour and also edits this page.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/archive_bot_talk_cron.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') archive_bot_talk =====" >>"$LOG"
if /usr/bin/env python3 "$DIR/archive_bot_talk.py" --days 7 >>"$LOG" 2>&1; then
  echo "exit: 0 (ok)" >>"$LOG"
else
  echo "exit: $? (failed — talk page left untouched)" >>"$LOG"
fi

# Keep the log from growing without bound.
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 1000000 ]; then
  tail -c 400000 "$LOG" >"$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
exit 0
