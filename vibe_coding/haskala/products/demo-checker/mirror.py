"""mirror.py — keep the "checked files" tab in newest-first order, with links.

Avishai maintains a second tab in the responses spreadsheet holding the same
submissions in reverse chronological order — newest at the top — plus one extra
column, "link to analyzed Doc". Forms only ever appends to the bottom of its own
tab and cannot be told otherwise, so that ordering has to be maintained by us.

WHY THIS TAB AND NOT COLUMN I OF THE RESPONSES TAB
──────────────────────────────────────────────────
The plan doc says "a link to the file is added to the response Google sheet in
column I", which read as the Forms tab. The sheet itself settles it: "Form
Responses 1" has eight columns and stops, while "checked files" has those same
eight plus "link to analyzed Doc" in column I. Column I of the mirror is the
column he meant. Writing to the Forms tab instead would have put a stray header
next to teachers' answers and left his own column empty.

REBUILT, NOT APPENDED
─────────────────────
Every scan rewrites the whole data range. Appending would be cheaper, but
newest-first means each new row goes at the TOP, so an append is really an
insert-and-shift — and one interrupted run leaves the tab permanently
misordered with nothing to detect it. A full rewrite is idempotent: whatever
state the tab is in, one scan puts it right.

Only A2 downwards is written. Row 1 is Avishai's, formatting and all, and is
left untouched.

WHERE THE LINKS COME FROM
─────────────────────────
Harvested from the tab's own previous contents, keyed on the same material as
main.row_key. Nothing else is read to build them: the sheet is its own memory,
so a rebuild cannot lose a link learned on an earlier run, and links learned in
this run are simply overlaid on top.
"""
from __future__ import annotations

import logging

log = logging.getLogger("demo-checker")

TAB = "checked files"

# 0-based index of the link column within the mirror: I.
LINK_COL = 8

# The header Avishai wrote there. Checked, not written — if it ever stops
# matching, the layout has changed under us and a rewrite could overwrite a
# column of his rather than ours.
LINK_HEADER = "link to analyzed doc"


def _key(row: list[str], colmap: dict[str, int], form_schema) -> str:
    """The row_key material for one mirrored row.

    The mirror's A–H are the Forms tab's A–H, so the same colmap parses both.
    Imported through the caller rather than at module scope to keep this file
    free of the grading imports."""
    p = form_schema.parse_row(row, colmap)
    return "|".join([p.get("timestamp", ""), p.get("email", "").lower(),
                     ",".join(p.get("file_ids", []))])


def read_known_links(sheets_write, sheet_id: str, colmap: dict, form_schema) -> dict[str, str]:
    """{row key: link} already present in the tab, so a rebuild never loses one."""
    try:
        got = sheets_write.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=f"'{TAB}'!A1:Z1000").execute()
    except Exception as e:
        log.warning("could not read the %r tab (%s: %s) — treating it as empty",
                    TAB, type(e).__name__, str(e)[:200])
        return {}

    values = got.get("values", [])
    if not values:
        return {}

    header = values[0]
    if len(header) <= LINK_COL or header[LINK_COL].strip().lower() != LINK_HEADER:
        log.warning("the %r tab's column %s is %r, not %r — not writing links "
                    "into a column that may be someone else's",
                    TAB, chr(ord("A") + LINK_COL),
                    header[LINK_COL] if len(header) > LINK_COL else "(missing)",
                    LINK_HEADER)
        return {}

    known = {}
    for row in values[1:]:
        if len(row) > LINK_COL and (row[LINK_COL] or "").strip():
            known[_key(row, colmap, form_schema)] = row[LINK_COL].strip()
    log.debug("%d known link(s) already in the %r tab", len(known), TAB)
    return known


def rebuild(sheets_write, sheet_id: str, rows: list[list[str]], colmap: dict,
            links: dict[str, str], form_schema) -> bool:
    """Rewrite the tab: same submissions, newest first, links in column I.

    `rows` is the Forms tab's data rows in sheet order (oldest first); `links`
    is {row key: link} learned this run, overlaid on what the tab already knew.
    Never raises — a spreadsheet that is one scan out of date is not worth
    failing a graded submission over."""
    known = read_known_links(sheets_write, sheet_id, colmap, form_schema)
    if not known and not links:
        log.debug("no links known yet — the mirror will still be reordered")
    known.update(links)

    body = []
    for row in reversed(rows):                       # newest first
        padded = list(row) + [""] * (LINK_COL - len(row))
        body.append(padded[:LINK_COL] + [known.get(_key(row, colmap, form_schema), "")])

    try:
        # Clear first: a deleted or edited response can leave the tab longer
        # than the source, and update() alone would strand those rows below.
        sheets_write.spreadsheets().values().clear(
            spreadsheetId=sheet_id, range=f"'{TAB}'!A2:Z1000", body={}).execute()
        if body:
            sheets_write.spreadsheets().values().update(
                spreadsheetId=sheet_id, range=f"'{TAB}'!A2",
                valueInputOption="USER_ENTERED",     # so links render as links
                body={"values": body}).execute()
    except Exception as e:
        log.warning("could not rebuild the %r tab (%s: %s)",
                    TAB, type(e).__name__, str(e)[:200])
        return False

    log.info("rebuilt the %r tab: %d row(s), newest first, %d with a report link",
             TAB, len(body), sum(1 for r in body if r[LINK_COL]))
    return True
