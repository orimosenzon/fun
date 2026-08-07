#!/usr/bin/env bash
# Report-first enrichment of NEW pardespedia articles (called from crontab).
# Stage 1 (Python): detect new articles since last run, write a triage report.
# Stage 2 (headless Claude, cost-capped, read-only vs the wiki): fill the report
# with proposed enrichment (text / candidate images / candidate video), and a
# machine-readable plan JSON alongside it.
# Stage 3 (Python): download the candidates and check them — image dimensions,
# face detection, and whether each YouTube id actually resolves to the claimed
# video. Read-only against the wiki too.
#
# Nothing here publishes. Stage 2 has no Bash on purpose, and stage 3 only runs
# --prepare. Putting an image on the wiki needs two judgment calls a cron job
# has no business making — is the licence honest, and is there a face in the
# frame that did not consent — so publishing is a separate manual command:
#   python3 enrich_apply.py --list    <plan>
#   python3 enrich_apply.py --approve <plan> --article "ערך"
#   python3 enrich_apply.py --apply   <plan>
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
  mkdir -p "$DIR/enrich_plans"
  PLAN="$DIR/enrich_plans/$(basename "${REPORT%.md}").json"
  echo "drafting enrichment for $COUNT new article(s) -> $REPORT" >>"$LOG"
  PROMPT="$(cat "$DIR/enrich_prompt.txt")

קובץ הדוח למילוי: $REPORT
קובץ התוכנית (JSON) לכתיבה: $PLAN"
  "$CLAUDE" -p "$PROMPT" \
    --model sonnet \
    --permission-mode acceptEdits \
    --allowedTools "Read,Write,Edit,Grep,Glob,WebSearch,WebFetch" \
    --max-budget-usd 2.00 \
    --no-session-persistence \
    >>"$LOG" 2>&1 && echo "draft stage ok" >>"$LOG" \
    || echo "draft stage failed (non-fatal) — report still has the triage table" >>"$LOG"
  echo "REPORT READY: $REPORT" >>"$LOG"

  # --- Stage 3a: fetch and vet the candidates (still no wiki writes) ---
  if [ -s "$PLAN" ]; then
    /usr/bin/env python3 "$DIR/enrich_apply.py" "$PLAN" --prepare >>"$LOG" 2>&1 \
      && echo "PLAN READY (needs your approval): $PLAN" >>"$LOG" \
      || echo "prepare stage failed (non-fatal) — plan is still on disk" >>"$LOG"
  else
    echo "no plan JSON written by the draft stage; nothing to prepare." >>"$LOG"
  fi
else
  echo "no new articles (COUNT=$COUNT); nothing to draft." >>"$LOG"
fi
echo "exit: 0 (ok)" >>"$LOG"
