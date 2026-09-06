#!/usr/bin/env python3
"""Keep the committee-meetings board on [[ועדות המועצה]] to the coming month.

Unlike the culture board, this one has no feed behind it: the municipality
publishes a roster of committees but never the dates they meet, so the rows are
typed by hand (עירית רתם asked for exactly this, 6/9/2026). The script's whole
job is therefore the one thing a person should not have to remember: a meeting
that has happened moves out of "ישיבות ועדה קרובות" and down into "ישיבות
שהתקיימו", so the board always shows what is still ahead.

Deliberately conservative, because a human edits the same table:
  * only whole rows move, and their text is copied verbatim.
  * a row whose date it cannot read is left exactly where it is, and logged.
    Better a stale row somebody notices than a row this script eats.
  * anything outside the two tables is untouched, get-first as always.

Usage:
    python3 update_committees.py [--dry-run] [--days 31]
"""

import argparse
import datetime as dt
import re
import sys

from wiki_client import WikiClient

PAGE = 'ועדות המועצה'
UPCOMING = 'ישיבות ועדה קרובות'
PAST = 'ישיבות שהתקיימו'

EMPTY_UPCOMING = ("''כרגע לא ידוע על ישיבות ועדה מתוכננות בחודש הקרוב. "
                  "מי שיודע על ישיבה מוזמן להוסיף אותה, או לפנות לחברי המועצה.''")
EMPTY_PAST = "''טרם תועדו ישיבות בדף זה.''"

DATE_RE = re.compile(r'data-sort-value="(\d{4}-\d{2}-\d{2})"')


def split_table(section: str):
    """(before, header, rows, after) for the first wikitable in a section."""
    m = re.search(r'^\{\|[^\n]*\n', section, re.M)
    if not m:
        return None
    start = m.start()
    end = section.find('\n|}', start)
    if end < 0:
        return None
    body = section[m.end():end]
    lines = body.split('\n')
    header = lines[0] if lines and lines[0].startswith('!') else ''
    rest = '\n'.join(lines[1:] if header else lines)
    rows = [r.strip('\n') for r in rest.split('|-') if r.strip()]
    return section[:start] + m.group(0), header, rows, section[end:]


def section_bounds(text: str, heading: str):
    m = re.search(r'^==\s*%s\s*==\s*$' % re.escape(heading), text, re.M)
    if not m:
        return None
    nxt = re.search(r'^==[^=]', text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start() if nxt else len(text))


def rebuild(section: str, parts, rows, empty_note):
    pre, header, _, post = parts
    if rows:
        body = '\n|-\n'.join([''] + rows).lstrip('\n')
        table = pre + (header + '\n' if header else '') + body + post
        # drop a leftover "nothing here yet" line under a table that has rows
        return re.sub(r'\n+' + re.escape(empty_note), '', table)
    table = pre + (header + '\n' if header else '') + post
    if empty_note not in table:
        table = table.rstrip() + '\n\n' + empty_note + '\n'
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=31,
                    help='how far ahead the board looks (default: a month)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    today = dt.date.today()
    horizon = today + dt.timedelta(days=args.days)

    client = WikiClient()
    if not args.dry_run:
        client.login()
    text = client.get_page(PAGE)['wikitext']

    up_b, past_b = section_bounds(text, UPCOMING), section_bounds(text, PAST)
    if not up_b or not past_b:
        print('לא נמצאו שני הסעיפים בדף. לא נגעתי בכלום.', file=sys.stderr)
        return 1

    up_sec, past_sec = text[up_b[0]:up_b[1]], text[past_b[0]:past_b[1]]
    up_parts, past_parts = split_table(up_sec), split_table(past_sec)
    if not up_parts or not past_parts:
        print('לא נמצאה טבלה באחד הסעיפים. לא נגעתי בכלום.', file=sys.stderr)
        return 1

    keep, moved, unreadable, beyond = [], [], 0, 0
    for row in up_parts[2]:
        m = DATE_RE.search(row)
        if not m:
            unreadable += 1
            keep.append(row)                     # never eat what we cannot read
            continue
        d = dt.date.fromisoformat(m.group(1))
        if d < today:
            moved.append(row)
        else:
            if d > horizon:
                beyond += 1
            keep.append(row)

    keep.sort(key=lambda r: DATE_RE.search(r).group(1) if DATE_RE.search(r) else '9999')
    past_rows = moved + past_parts[2]
    past_rows.sort(key=lambda r: DATE_RE.search(r).group(1) if DATE_RE.search(r) else '0000',
                   reverse=True)

    print('ישיבות קרובות: %d | הועברו לארכיון: %d | מעבר לחלון: %d | שורות לא קריאות: %d'
          % (len(keep), len(moved), beyond, unreadable), file=sys.stderr)
    if unreadable:
        print('  שורה בלי data-sort-value תקין הושארה במקומה ולא הועברה.', file=sys.stderr)

    if not moved and not args.dry_run:
        print('אין מה להעביר.', file=sys.stderr)
        return 0

    new = (text[:up_b[0]] + rebuild(up_sec, up_parts, keep, EMPTY_UPCOMING)
           + text[up_b[1]:past_b[0]] + rebuild(past_sec, past_parts, past_rows, EMPTY_PAST)
           + text[past_b[1]:])

    if args.dry_run:
        print(new)
        return 0

    client.edit_page(PAGE, new,
                     summary='העברת %d ישיבות שהתקיימו לסעיף הארכיון' % len(moved))
    return 0


if __name__ == '__main__':
    sys.exit(main())
