"""sheet_link.py — write the graded report's link back into the responses sheet.

Avishai's spec says "a link to the file is added to the response Google sheet in
column I". This does that, with one deliberate difference: the column is found
by its header text and only *defaults* to I, rather than being pinned there.

WHY NOT JUST COLUMN I
─────────────────────
Everything else in this product matches columns by header text, for a reason
that has already cost us once: Avishai owns the form and keeps editing it, and
an inserted question shifts every column after it without anything looking
wrong. Writing to a fixed index I would, after one new question, start
overwriting a column of teachers' answers — silently, and in a sheet nobody
reads closely. So the first run writes a header into the first free column
(which today is I) and every run after that finds it by name.

WHY ITS OWN SCOPE TIER
──────────────────────
Writing needs `spreadsheets`, not the `spreadsheets.readonly` the service reads
with, and domain-wide delegation is all-or-nothing *per token request*. Bundling
this into EXTRA_SCOPES alongside gmail.send would mean that until an admin adds
the write scope, the mail scope fails too — and mail is the only way anything
reaches the teacher. So this asks for its own token, and a denial costs exactly
this feature and nothing else.
"""
from __future__ import annotations

import logging

log = logging.getLogger("demo-checker")

# What the header of our column says. Matched case-insensitively.
LINK_HEADER = "Bdika report"

# Where the header is written the first time, if it is not already present.
# Column I — the first free column in the demo form's sheet, and what Avishai's
# document asks for. Only ever used to place a *new* header.
DEFAULT_COLUMN_INDEX = 8   # 0-based: A=0 … I=8


def _a1(col_index: int, row: int) -> str:
    """(0-based column, 1-based row) → an A1 reference like 'I7'."""
    letters = ""
    n = col_index
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return f"{letters}{row}"


def find_or_create_column(sheets_write, sheet_id: str, header_row: list[str]) -> int | None:
    """0-based index of the link column, creating its header if absent.

    Returns None if the header could not be written, so the caller can carry on
    without it — a missing link in a spreadsheet is not worth failing a graded
    submission over."""
    for idx, head in enumerate(header_row):
        if (head or "").strip().lower() == LINK_HEADER.lower():
            return idx

    idx = max(DEFAULT_COLUMN_INDEX, len(header_row))
    try:
        sheets_write.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=_a1(idx, 1),
            valueInputOption="RAW",
            body={"values": [[LINK_HEADER]]},
        ).execute()
    except Exception as e:
        log.warning("could not add the %r header at column %s (%s: %s)",
                    LINK_HEADER, _a1(idx, 1), type(e).__name__, str(e)[:200])
        return None
    log.info("added the %r header at %s", LINK_HEADER, _a1(idx, 1))
    return idx


def write_link(sheets_write, sheet_id: str, row_num: int, col_index: int,
               link: str, tag: str = "") -> bool:
    """Put one report link in one cell. True if it stuck; never raises.

    row_num is the sheet's own 1-based row, so it lines up with what a person
    sees when they open the sheet."""
    cell = _a1(col_index, row_num)
    try:
        sheets_write.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=cell,
            valueInputOption="USER_ENTERED",   # so Sheets renders it as a link
            body={"values": [[link]]},
        ).execute()
    except Exception as e:
        log.warning("%s could not write the report link to %s (%s: %s)",
                    tag, cell, type(e).__name__, str(e)[:200])
        return False
    log.info("%s wrote the report link to %s", tag, cell)
    return True
