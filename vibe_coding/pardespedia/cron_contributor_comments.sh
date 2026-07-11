#!/usr/bin/env bash
# Auto-comment on active users' recent contributions (called from crontab).
# Unlike cron_enrich.sh, this is NOT report-first: the agentic stage has
# Bash and posts directly to live user talk pages. That's an explicit product
# choice (2026-07-11) — see contributor_scan.py's docstring for the safety
# rails that exist instead of human approval (per-user state, MIN_EDITS
# threshold, excluded accounts, capped users-per-run).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/contributor_cron.log"
CLAUDE="${CLAUDE_BIN:-/home/ori/.local/bin/claude}"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') contributor-comments =====" >>"$LOG"

# --- Stage 1: detect (deterministic, no LLM) ---
SCAN_OUT="$(/usr/bin/env python3 "$DIR/contributor_scan.py" 2>>"$LOG")"
printf '%s\n' "$SCAN_OUT" >>"$LOG"
REPORT="$(printf '%s\n' "$SCAN_OUT" | sed -n 's/^REPORT=//p' | tail -1)"
COUNT="$(printf '%s\n' "$SCAN_OUT" | sed -n 's/^COUNT=//p' | tail -1)"
COUNT="${COUNT:-0}"

# --- Stage 2: agentic — reads real content, composes, and POSTS ---
if [ "$COUNT" -gt 0 ] && [ -n "${REPORT:-}" ] && [ -x "$CLAUDE" ]; then
  echo "commenting on $COUNT user(s) -> $REPORT" >>"$LOG"
  PROMPT="$(cat "$DIR/contributor_prompt.txt")$REPORT"
  "$CLAUDE" -p "$PROMPT" \
    --model sonnet \
    --permission-mode acceptEdits \
    --allowedTools "Read,Bash,Grep,Glob" \
    --max-budget-usd 3.00 \
    --no-session-persistence \
    >>"$LOG" 2>&1 && echo "comment stage ok" >>"$LOG" \
    || echo "comment stage failed (non-fatal) — see log; those edits' state was already advanced (see contributor_scan.py docstring)" >>"$LOG"
else
  echo "no users to comment on (COUNT=$COUNT); nothing to do." >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
