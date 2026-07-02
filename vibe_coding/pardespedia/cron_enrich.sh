#!/usr/bin/env bash
# Report-first enrichment of NEW pardespedia articles (called from crontab).
# Stage 1 (Python): detect new articles since last run, write a triage report.
# Stage 2 (headless Claude, cost-capped, read-only vs the wiki): fill the report
# with proposed enrichment (text / candidate images / candidate video).
# Never edits the wiki — Bash is intentionally NOT in the allowed tools.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/enrich_cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') enrich =====" >>"$LOG"

# --- Stage 1: detect ---
SCAN_OUT="$(/usr/bin/env python3 "$DIR/enrich_scan.py" 2>>"$LOG")"
printf '%s\n' "$SCAN_OUT" >>"$LOG"
REPORT="$(printf '%s\n' "$SCAN_OUT" | sed -n 's/^REPORT=//p' | tail -1)"
COUNT="$(printf '%s\n' "$SCAN_OUT" | sed -n 's/^COUNT=//p' | tail -1)"
COUNT="${COUNT:-0}"

# --- Stage 2: agentic drafts (only if there are new articles) ---
if [ "$COUNT" -gt 0 ] && [ -n "${REPORT:-}" ] && [ -x "$CLAUDE" ]; then
  echo "drafting enrichment for $COUNT new article(s) -> $REPORT" >>"$LOG"
  PROMPT="$(cat "$DIR/enrich_prompt.txt")

קובץ הדוח למילוי: $REPORT"
  "$CLAUDE" -p "$PROMPT" \
    --model sonnet \
    --permission-mode acceptEdits \
    --allowedTools "Read,Write,Edit,Grep,Glob,WebSearch,WebFetch" \
    --max-budget-usd 2.00 \
    --no-session-persistence \
    >>"$LOG" 2>&1 && echo "draft stage ok" >>"$LOG" \
    || echo "draft stage failed (non-fatal) — report still has the triage table" >>"$LOG"
  echo "REPORT READY: $REPORT" >>"$LOG"
else
  echo "no new articles (COUNT=$COUNT); nothing to draft." >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
