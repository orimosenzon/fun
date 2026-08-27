#!/usr/bin/env python3
"""Archive stale threads off the bot's user-talk page, into monthly archives.

Keeps the talk page readable: a thread stays on the live page only while
someone has written in it within the last N days (default 7). Everything older
moves, verbatim, to a monthly archive subpage such as
``שיחת משתמש:אורי מוסנזון בוט/ארכיון אוגוסט 2026``.

Two different timestamps drive the two different decisions, on purpose:

* **whether** a thread is archived — its *last* signature, so an old question
  answered yesterday stays on the live page;
* **where** it is archived to — its *first* signature, so a thread lands in the
  month it was opened, which is the month a reader remembers it from.

The block of text before the first heading (the archive box) is regenerated on
every run, so the list of archives and their thread counts stay accurate. A
thread with no parseable timestamp is never archived — it is reported in the
log instead, because guessing an age would risk hiding a live request.

Usage:
    python3 archive_bot_talk.py --dry-run     # show what would move
    python3 archive_bot_talk.py               # do it (no edit if nothing moves)
    python3 archive_bot_talk.py --days 14     # different retention window
"""
import argparse
import datetime as dt
import re
import sys

from wiki_client import WikiClient

TALK_PAGE = "שיחת משתמש:אורי מוסנזון בוט"
ARCHIVE_PREFIX = TALK_PAGE + "/ארכיון "

ARCHIVE_HEADER = (
    "''זהו ארכיון של דף השיחה של [[משתמש:אורי מוסנזון בוט|אורי מוסנזון בוט]]. "
    "אין לערוך דף זה; להערות חדשות אנא פנו "
    "ל[[שיחת משתמש:אורי מוסנזון בוט|דף השיחה הפעיל]].''\n"
)

MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
          "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
MONTH_NUM = {"ב" + name: i + 1 for i, name in enumerate(MONTHS)}

# 14:01, 15 באוגוסט 2026 (IDT)
SIG_RE = re.compile(r"(\d{1,2}):(\d{2}),\s*(\d{1,2})\s+(ב\S+?)\s+(\d{4})")
HEADING_RE = re.compile(r"^\s*==(?!=)\s*(.+?)\s*(?<!=)==\s*$")


def now_local():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None)
    except Exception:
        return dt.datetime.now()


def timestamps(text):
    """Every signature time in a thread, in the order they appear."""
    out = []
    for hh, mm, day, month, year in SIG_RE.findall(text):
        if month not in MONTH_NUM:
            continue
        try:
            out.append(dt.datetime(int(year), MONTH_NUM[month], int(day),
                                   int(hh), int(mm)))
        except ValueError:
            continue
    return out


def split_threads(wikitext):
    """-> (header_block, [(title, full_thread_text), ...])"""
    lines = wikitext.split("\n")
    starts = [i for i, ln in enumerate(lines) if HEADING_RE.match(ln)]
    if not starts:
        return wikitext, []
    header = "\n".join(lines[:starts[0]])
    threads = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        title = HEADING_RE.match(lines[start]).group(1)
        threads.append((title, "\n".join(lines[start:end]).rstrip("\n")))
    return header, threads


def archive_title(ts):
    return f"{ARCHIVE_PREFIX}{MONTHS[ts.month - 1]} {ts.year}"


def list_archives(client):
    """Existing monthly archive subpages, oldest first."""
    found = []
    for page in client.list_pages(prefix=ARCHIVE_PREFIX.split(":", 1)[1],
                                  namespace=3):
        title = page if isinstance(page, str) else page.get("title", "")
        if not title.startswith(ARCHIVE_PREFIX):
            continue
        label = title[len(ARCHIVE_PREFIX):]
        parts = label.split()
        if len(parts) == 2 and parts[0] in MONTHS and parts[1].isdigit():
            found.append((int(parts[1]), MONTHS.index(parts[0]) + 1, title, label))
    found.sort()
    return found


def count_threads(client, title):
    _, threads = split_threads(client.get_page(title)["wikitext"])
    return len(threads)


def build_box(rows):
    """The archive table at the top of the talk page, newest month first."""
    out = ['{| class="wikitable" style="background:#f8f9fa; margin-bottom:1em"',
           "! ארכיוני דף זה !! שיחות"]
    for title, label, n in rows:
        out.append("|-")
        out.append(f"| [[{title}|{label}]] || {n}")
    out.append("|}")
    return "\n".join(out) + "\n"


def write_archive(client, title, new_threads, dry_run):
    """Merge threads into a monthly archive, kept in chronological order."""
    page = client.get_page(title)
    existing = split_threads(page["wikitext"])[1] if page["exists"] else []
    merged = existing + [t for t in new_threads
                         if t[0] not in {e[0] for e in existing}]
    merged.sort(key=lambda tb: (timestamps(tb[1]) or [dt.datetime.max])[0])
    text = ARCHIVE_HEADER + "\n" + "\n\n".join(b for _, b in merged) + "\n"
    if dry_run:
        print(f"    would write {title}: {len(merged)} threads, {len(text)} chars")
        return len(merged)
    client.edit_page(title, text,
                     summary=f"ארכוב אוטומטי: {len(new_threads)} שיחות מדף השיחה")
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="keep threads touched within this many days (default 7)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cutoff = now_local() - dt.timedelta(days=args.days)
    print(f"cutoff: {cutoff:%Y-%m-%d %H:%M} (last {args.days} days stay)")

    client = WikiClient()
    client.login()

    page = client.get_page(TALK_PAGE)
    if not page["exists"]:
        print("talk page does not exist; nothing to do.")
        return
    header, threads = split_threads(page["wikitext"])
    if not threads:
        print("no level-2 threads found; nothing to do.")
        return

    keep, by_month = [], {}
    for title, body in threads:
        ts = timestamps(body)
        if not ts:
            keep.append((title, body))
            print(f"  KEEP (no timestamp, not archiving blind): {title}")
        elif ts[-1] >= cutoff:
            keep.append((title, body))
            print(f"  KEEP  {ts[-1]:%Y-%m-%d %H:%M}  {title}")
        else:
            target = archive_title(ts[0])
            by_month.setdefault(target, []).append((title, body))
            print(f"  MOVE  {ts[-1]:%Y-%m-%d %H:%M}  -> {target[-14:]}  {title}")

    moved = sum(len(v) for v in by_month.values())
    print(f"{len(keep)} stay, {moved} to archive across {len(by_month)} month(s)")

    counts = {}
    for target, items in sorted(by_month.items()):
        counts[target] = write_archive(client, target, items, args.dry_run)

    # The box is rebuilt every run, not only when something moves, so a stale
    # archive list or thread count heals itself on the next daily pass.
    rows = []
    for _, _, title, label in list_archives(client):
        rows.append((title, label, counts.get(title) or count_threads(client, title)))
    for title in by_month:                          # created moments ago
        if not any(r[0] == title for r in rows):
            rows.append((title, title[len(ARCHIVE_PREFIX):], counts[title]))
    rows.sort(key=lambda r: (int(r[1].split()[1]), MONTHS.index(r[1].split()[0])),
              reverse=True)

    new_talk = build_box(rows)
    if keep:
        new_talk += "\n" + "\n\n".join(b for _, b in keep) + "\n"

    if new_talk.strip() == page["wikitext"].strip():
        print("talk page already up to date; no edit.")
        return
    if args.dry_run:
        print("--- DRY RUN: talk page box would become ---")
        print(build_box(rows))
        return

    summary = (f"ארכוב אוטומטי של {moved} שיחות ללא פעילות מעל {args.days} ימים"
               if moved else "רענון תיבת הארכיונים")
    client.edit_page(TALK_PAGE, new_talk, summary=summary)
    print(f"talk page updated ({moved} archived)")


if __name__ == "__main__":
    sys.exit(main())
