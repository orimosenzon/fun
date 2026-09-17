#!/usr/bin/env bash
# Nightly scan of the shared Google Drive folder (called from crontab):
# new committee posters go onto [[ועדות המועצה]], everything else new is
# reported on the bot's talk page. Logs to drive_scan_cron.log here.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LOG="$DIR/drive_scan_cron.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >>"$LOG"
/usr/bin/env python3 "$DIR/drive_scan.py" "$@" >>"$LOG" 2>&1
echo "exit: $?" >>"$LOG"
