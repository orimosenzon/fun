"""pilot_check.py — read the real responses sheet and the real folders, grade nothing.

The dress rehearsal. Every failure mode that a new form introduces shows up
here, at zero cost: a column the schema cannot find, a folder shared read-only,
a link that is not a folder, files in formats we cannot read, an exam paper that
is not detected as one. Those are the things that would otherwise be discovered
by a teacher not receiving a report.

It writes nothing, creates nothing, and makes no model calls. The only thing it
cannot tell you is whether the grades are any good.

    RESPONSES_SHEET_ID=1arj… python pilot_check.py

Prerequisite: application-default credentials.  ./connect --adc
"""
from __future__ import annotations

import os
import sys

from googleapiclient.discovery import build

import drive_folder
import form_schema
import main

# Rough per-page cost of the grading pass (rotation + OCR + a share of the
# evaluation) on Claude Sonnet 5. Deliberately the high end — an estimate that
# flatters us is worse than useless.
COST_PER_PAGE = 0.06


def main_():
    sheet_id = os.environ.get("RESPONSES_SHEET_ID") or main.RESPONSES_SHEET_ID
    if not sheet_id:
        print("RESPONSES_SHEET_ID is not set.", file=sys.stderr)
        return 2

    creds = main.get_workspace_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    print("── identity ──")
    print(f"   impersonating : {main.WORKSPACE_SUBJECT}")
    print(f"   mail scope    : {'granted' if main._extra_scopes_granted else 'NOT granted'}")
    print(f"   acting as     : "
          f"{drive.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']}")

    print(f"\n── responses sheet {sheet_id} ──")
    values = sheets.spreadsheets().values().get(
        spreadsheetId=sheet_id, range="A:ZZ").execute().get("values", [])
    if not values:
        print("   the sheet is empty — no submissions yet")
        return 0
    header, rows = values[0], values[1:]
    print(f"   {len(rows)} response row(s)")
    print("   columns:")
    for i, h in enumerate(header):
        print(f"      {i:2}  {h!r}")

    print("\n── column mapping ──")
    try:
        colmap = form_schema.map_columns(header)
    except ValueError as e:
        # The single most likely failure with a form we did not build, and the
        # one that takes down every row at once rather than degrading.
        print(f"   ❌ {e}")
        return 1
    for field, idx in sorted(colmap.items(), key=lambda kv: kv[1]):
        print(f"   ✓ {field:14} ← column {idx}  {header[idx]!r}")
    unmapped = [h for i, h in enumerate(header) if i not in set(colmap.values())]
    for h in unmapped:
        print(f"   · (ignored)     {h!r}")

    total_pages = 0
    for row_num, row in enumerate(rows, start=2):
        params = form_schema.parse_row(row, colmap)
        print(f"\n{'═' * 70}\n── sheet row {row_num} ──")
        for k in ("timestamp", "email", "teacher_name", "module", "rubric_id",
                  "folder_id", "comments"):
            print(f"   {k:14}: {params.get(k)!r}")
        task = (params.get("instructions") or "").strip()
        print(f"   {'task on form':14}: "
              f"{(task[:120] + '…') if len(task) > 120 else (task or '(none)')!r}")

        refusal = main._admit(params, f"[row {row_num}]")
        if refusal is not None:
            print(f"   ❌ would not be graded: {refusal}")
            continue

        print("\n   ── folder ──")
        try:
            meta = drive_folder.check_access(drive, params["folder_id"])
        except drive_folder.FolderError as e:
            print(f"   ❌ {e}")
            continue
        caps = meta.get("capabilities") or {}
        print(f"   name           : {meta.get('name')!r}")
        print(f"   canAddChildren : {caps.get('canAddChildren')}  "
              f"{'✓ we can write results back' if caps.get('canAddChildren') else '✗ READ-ONLY'}")

        work, other = drive_folder.list_candidates(drive, params["folder_id"])
        works, contexts = [], []
        for f in work:
            data = drive_folder.download(drive, f["id"])
            typed, why = drive_folder.looks_typed(data, f["ext"])
            pages = drive_folder.page_count(data, f["ext"])
            (contexts if typed else works).append((f, pages, why))

        print(f"\n   context documents ({len(contexts)}) — read for the task, not graded:")
        for f, pages, why in contexts:
            print(f"      • {f['name']!r}  ({pages}p — {why})")
        if not contexts and not task:
            print("      ⚠️  none, and no task on the form either — reports will say")
            print("          the assignment was not supplied and Content cannot be judged")

        print(f"\n   student work ({len(works)}):")
        for i, (f, pages, why) in enumerate(works):
            mark = "→ WOULD GRADE NOW" if i < main.MAX_WORKS_PER_FOLDER else "  (deferred)"
            print(f"      {mark}  {f['name']!r}  ({pages}p)")
        for f in other:
            print(f"      ✗ skipped  {f['name']!r} — {f['reason']}")

        selected = works[:main.MAX_WORKS_PER_FOLDER]
        pages = sum(p for _f, p, _w in selected)
        total_pages += pages
        print(f"\n   would grade {len(selected)} work(s) / {pages} page(s)"
              f"  ≈ ${pages * COST_PER_PAGE:.2f}")

    print(f"\n{'═' * 70}")
    print(f"TOTAL if run for real: {total_pages} page(s) ≈ ${total_pages * COST_PER_PAGE:.2f}")
    print("Nothing was written, created or graded.")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
