#!/usr/bin/env bash
# Recurring all-articles sweep, every 14 days (Ori's request, 24/11/2026).
# Repeats the one-off sweep of 23/11/2026: an agent per thematic batch checks
# each article against the web and edits it live through sweep_edit.py (edit
# conflicts fail instead of overwriting, pages a human edited in the last 6h
# are skipped, >30% shrink is refused). Like cron_contributor_comments.sh this
# publishes directly, by Ori's choice; the safety rails are in sweep_edit.py.
#
# Runs DAILY from crontab; sweep_cron.py decides whether a run is due (14 days
# after the previous one started) and hands out the next batch. A full sweep
# is ~31 batches, so each night does at most MAX_BATCHES and starts no new
# batch after STOP_HOUR; the run spreads over several nights and resumes by
# itself if the computer was off.
#
#   python3 sweep_cron.py status     # where the current run stands
# Reports: ~/.cache/pardespedia_sweep/cron/runs/<date>/reports/, and SUMMARY.md
# at the end of a run (everything that needs Ori, in one file).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/sweep_cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"
MODEL="${SWEEP_MODEL:-opus}"
BUDGET="${SWEEP_BUDGET_USD:-10.00}"   # per batch
MAX_BATCHES="${SWEEP_MAX_BATCHES:-6}" # per night
STOP_HOUR="${SWEEP_STOP_HOUR:-8}"     # no new batch from this hour on
LOCK="$HOME/.cache/pardespedia_sweep/cron/lock"
mkdir -p "$(dirname "$LOCK")"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') sweep: previous night still running, skip =====" >>"$LOG"
  exit 0
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') sweep =====" >>"$LOG"
failed=0
for ((i = 0; i < MAX_BATCHES; i++)); do
  if [ "$(date +%-H)" -ge "$STOP_HOUR" ]; then
    echo "past ${STOP_HOUR}:00, stopping for tonight" >>"$LOG"
    break
  fi
  NEXT="$(/usr/bin/env python3 "$DIR/sweep_cron.py" next 2>>"$LOG")" || { failed=1; break; }
  printf '%s\n' "$NEXT" >>"$LOG"
  BATCH="$(printf '%s\n' "$NEXT" | sed -n 's/^BATCH=//p')"
  [ -n "$BATCH" ] || break
  FILE="$(printf '%s\n' "$NEXT" | sed -n 's/^FILE=//p')"
  REPORT="$(printf '%s\n' "$NEXT" | sed -n 's/^REPORT=//p')"
  WORK="$(printf '%s\n' "$NEXT" | sed -n 's/^WORK=//p')"
  PREV="$(printf '%s\n' "$NEXT" | sed -n 's/^PREV=//p')"
  mkdir -p "$WORK"
  PROMPT="$(cat "$DIR/sweep_prompt.txt")
- batch name: $BATCH
- titles (JSON list): $FILE
- work dir <WORK>: $WORK
- report <REPORT>: $REPORT
- previous sweep (PREV): $PREV
- today: $(date +%F)"
  echo "batch $BATCH ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$FILE") articles) -> $MODEL" >>"$LOG"
  if "$CLAUDE" -p "$PROMPT" \
      --model "$MODEL" \
      --permission-mode acceptEdits \
      --allowedTools "Read,Write,Edit,Grep,Glob,Bash,WebSearch,WebFetch,ToolSearch" \
      --max-budget-usd "$BUDGET" \
      --no-session-persistence \
      </dev/null >>"$LOG" 2>&1 && [ -s "$REPORT" ]; then
    python3 "$DIR/sweep_cron.py" done "$BATCH" >>"$LOG" 2>&1
    echo "batch $BATCH ok" >>"$LOG"
  else
    python3 "$DIR/sweep_cron.py" fail "$BATCH" >>"$LOG" 2>&1
    echo "batch $BATCH failed (will retry later in the run)" >>"$LOG"
    failed=1
  fi
done

if [ "$failed" -eq 1 ]; then
  python3 "$DIR/cron_alert.py" fail sweep "$LOG" >>"$LOG" 2>&1
else
  python3 "$DIR/cron_alert.py" ok sweep >>"$LOG" 2>&1
fi
echo "exit: 0 (ok)" >>"$LOG"
