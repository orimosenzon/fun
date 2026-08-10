"""טסטים ל-demo-checker — מיפוי הטופס, שכבת ההעלאה, והמסירה במייל.

בלי רשת, בלי Sheets, בלי Drive אמיתי, בלי קריאות למודל.

מריצים עם pytest אם הוא מותקן, אחרת:  python3 tests/test_demo_checker.py
"""
import os
import sys
import base64
from email import message_from_bytes
from email.policy import default as _email_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

import form_schema as fs  # noqa: E402
import mailer  # noqa: E402
import sheet_link  # noqa: E402
import upload  # noqa: E402


# הכותרות האמיתיות של "Bdika Demo- English", כפי שנקראו מגיליון התשובות
# ב-8/8/2026. אם הטסטים כאן עוברים והשירות עדיין נופל — הבעיה בהרשאות או
# ברשת, לא במיפוי.
LIVE_HEADER = [
    "Timestamp",
    "Email Address",
    "First name",
    "Last name",
    "Use rubric ",
    "Assignment description",
    "Upload student's work here",
    "Comments and requests",
]

FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
FILE_URL = f"https://drive.google.com/open?id={FILE_ID}"


def _row(**over):
    row = {
        "Timestamp": "8/8/2026 10:15:00",
        "Email Address": "teacher@someschool.org.il",
        "First name": "Dana",
        "Last name": "Levi",
        "Use rubric ": "Module G 16582",
        "Assignment description": "Write about a place you love.",
        "Upload student's work here": FILE_URL,
        "Comments and requests": "",
    }
    row.update(over)
    return [row[h] for h in LIVE_HEADER]


# ─── the live form ──────────────────────────────────────────────────────────

def test_live_demo_form_header_maps():
    """Locked against the real sheet. form-checker shipped a form whose folder
    question was spelled "לתקייה" and every response was silently rejected;
    this is the test that would have caught it."""
    cm = fs.map_columns(LIVE_HEADER)
    assert cm == {"timestamp": 0, "email": 1, "first_name": 2, "last_name": 3,
                  "rubric": 4, "instructions": 5, "files": 6, "comments": 7}


def test_upload_column_is_required():
    """The whole product is the upload. A form without it must fail loudly at
    map time rather than grade nothing, quietly, forever."""
    without = [h for h in LIVE_HEADER if h != "Upload student's work here"]
    try:
        fs.map_columns(without)
    except ValueError as e:
        assert "files" in str(e)
    else:
        raise AssertionError("a form with no upload column was accepted")


def test_folder_link_is_not_a_field_here():
    assert "folder_link" not in fs.FIELDS
    assert not hasattr(fs, "extract_folder_id")


def test_parse_row_reads_the_upload():
    p = fs.parse_row(_row(), fs.map_columns(LIVE_HEADER))
    assert p["file_ids"] == [FILE_ID]
    assert p["email"] == "teacher@someschool.org.il"
    assert p["teacher_name"] == "Dana Levi"
    assert p["rubric_id"] == "moe-writing-module-g"
    assert p["module"] == "g"
    assert p["instructions"] == "Write about a place you love."


def test_file_id_url_shapes():
    """Forms has changed the upload URL shape more than once."""
    for cell in (f"https://drive.google.com/file/d/{FILE_ID}/view",
                 f"https://drive.google.com/open?id={FILE_ID}",
                 f"https://drive.google.com/d/{FILE_ID}",
                 FILE_ID):
        assert fs.extract_file_ids(cell) == [FILE_ID], cell


def test_missing_upload_parses_to_no_files():
    """An empty cell must not become a bogus id — it has to reach the
    teacher-facing "no file attached" message intact."""
    p = fs.parse_row(_row(**{"Upload student's work here": ""}),
                     fs.map_columns(LIVE_HEADER))
    assert p["file_ids"] == []


# ─── the upload layer ───────────────────────────────────────────────────────

def test_ext_for_accepts_what_we_can_render():
    assert upload.ext_for("application/pdf", "a.pdf") == ".pdf"
    assert upload.ext_for("image/jpeg", "a.jpeg") == ".jpg"
    assert upload.ext_for("image/png", "a.png") == ".png"


def test_ext_for_refuses_heic_and_word():
    assert upload.ext_for("image/heic", "IMG_1234.HEIC") is None
    assert upload.ext_for(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "essay.docx") is None


def test_report_name_is_readable_and_sorts_by_time():
    """Keeping only the digits of "8/8/2026 10:15:00" gives "882026101500",
    which is unreadable and ambiguous about where the date ends."""
    n = upload.report_name("Vacation essay.pdf", "8/8/2026 10:15:00")
    assert n == "Vacation essay_bdika_2026-08-08_1015.docx"
    assert upload.report_name("a.pdf", "") == "a_bdika.docx"
    assert upload.report_name("", "") == "work_bdika.docx"


def test_page_count_without_rendering():
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    data = doc.tobytes()
    assert upload.page_count(data, ".pdf") == 3
    assert upload.page_count(b"not a pdf", ".pdf") == 1
    assert upload.page_count(b"", ".jpg") == 1


class _FakeDrive:
    """Enough of the Drive client for fetch(): metadata and a media download."""

    def __init__(self, meta, data=b"%PDF-1.4 fake"):
        self._meta, self._data = meta, data

    def files(self):
        return self

    def get(self, **kw):
        return self

    def get_media(self, **kw):
        return self

    def execute(self):
        return self._meta


def _fetch(monkeypatched_bytes, meta):
    drive = _FakeDrive(meta)
    upload._download = lambda *a, **k: monkeypatched_bytes
    return upload.fetch(drive, meta["id"])


def test_fetch_refuses_heic_with_instructions_a_teacher_can_follow():
    try:
        _fetch(b"x", {"id": FILE_ID, "name": "IMG_9021.HEIC", "mimeType": "image/heic"})
    except upload.UploadError as e:
        assert "HEIC" in str(e) and "Most Compatible" in str(e)
    else:
        raise AssertionError("a HEIC upload was accepted")


def test_fetch_returns_bytes_for_a_pdf():
    entry = _fetch(b"%PDF-1.4 fake",
                   {"id": FILE_ID, "name": "essay.pdf", "mimeType": "application/pdf"})
    assert entry["ext"] == ".pdf" and entry["data"] == b"%PDF-1.4 fake"


# ─── delivery ───────────────────────────────────────────────────────────────

class _FakeGmail:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId=None, body=None):
        self._body = body
        return self

    def execute(self):
        if self.fail:
            raise RuntimeError("403 insufficient scope")
        self.sent.append(self._body)
        return {"id": "msg1"}


def _send(params, gmail=None, docx=b"PK\x03\x04fake-docx"):
    gmail = gmail or _FakeGmail()
    ok = mailer.send_report(gmail, "ori@bdika.net", params, docx,
                            "report.docx", "MoE Module G", tag="[t]")
    if not gmail.sent:
        return ok, None
    raw = base64.urlsafe_b64decode(gmail.sent[0]["raw"])
    # policy=default is what makes this an EmailMessage with get_body(); the
    # legacy Message the compat policy returns has no such method.
    return ok, message_from_bytes(raw, policy=_email_policy)


def _text(msg):
    return msg.get_body(preferencelist=("plain",)).get_content()


_PARAMS = {"email": "teacher@someschool.org.il", "teacher_name": "Dana Levi",
           "instructions": "Write about a place you love."}


def test_report_is_attached_not_linked():
    """The demo's whole delivery model: a teacher who has never heard of us
    should not have to click through a Drive permission prompt."""
    ok, msg = _send(_PARAMS)
    assert ok
    names = [p.get_filename() for p in msg.walk() if p.get_filename()]
    assert names == ["report.docx"]


def test_mail_carries_the_feedback_link():
    """A graded report we never hear back about is the demo failing silently."""
    _, msg = _send(_PARAMS)
    body = _text(msg)
    assert mailer.FEEDBACK_FORM_URL in body


def test_mail_flags_a_missing_assignment():
    """The report hedges the content score when the task is unknown; if the
    mail does not say why, that hedge reads as the tool being vague."""
    _, with_task = _send(_PARAMS)
    _, without = _send({**_PARAMS, "instructions": ""})
    assert "did not describe the assignment" not in _text(with_task)
    assert "did not describe the assignment" in _text(without)


def test_send_report_never_raises_on_a_gmail_failure():
    """Returning False is what lets main.py park the report and retry the row;
    an exception here would lose a report we already paid for."""
    ok, _ = _send(_PARAMS, gmail=_FakeGmail(fail=True))
    assert ok is False


def test_failure_mail_says_what_to_do_next():
    gmail = _FakeGmail()
    mailer.send_failure(gmail, "ori@bdika.net", _PARAMS,
                        "The file you uploaded is HEIC.", tag="[t]")
    body = _text(message_from_bytes(
        base64.urlsafe_b64decode(gmail.sent[0]["raw"]), policy=_email_policy))
    assert "HEIC" in body and "submit the form again" in body


# ─── writing back to Drive and the sheet ────────────────────────────────────

class _RecordingDrive:
    """Records create/list/permission calls so the write path can be asserted
    without touching Drive."""

    def __init__(self, existing_folder=None):
        self.created, self.permissions_granted = [], []
        self._existing = existing_folder
        self._pending = None

    def files(self):
        return self

    def permissions(self):
        return _RecordingPermissions(self)

    def list(self, **kw):
        self._pending = ("list", kw)
        return self

    def create(self, body=None, media_body=None, **kw):
        self._pending = ("create", body)
        return self

    def execute(self):
        kind, payload = self._pending
        if kind == "list":
            files = [{"id": self._existing, "name": upload.OUTPUT_FOLDER_NAME}] \
                if self._existing else []
            return {"files": files}
        self.created.append(payload)
        return {"id": f"new-{len(self.created)}", "webViewLink": "http://x"}


class _RecordingPermissions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, fileId=None, body=None, **kw):
        self._call = (fileId, body, kw)
        return self

    def execute(self):
        self.parent.permissions_granted.append(self._call)
        return {"id": "perm1"}


def test_output_folder_is_reused_not_recreated():
    """A new "Bdika (1)" per submission would bury the uploads folder."""
    drive = _RecordingDrive(existing_folder="existing-folder-id")
    assert upload.ensure_output_folder(drive, "parent") == "existing-folder-id"
    assert drive.created == []


def test_output_folder_is_created_when_absent():
    drive = _RecordingDrive()
    upload.ensure_output_folder(drive, "parent-id")
    assert drive.created[0]["name"] == "Bdika"
    assert drive.created[0]["mimeType"] == upload.FOLDER_MIME
    assert drive.created[0]["parents"] == ["parent-id"]


def test_report_is_written_as_an_editable_google_doc():
    """A .docx in Drive opens read-only until explicitly converted, and the
    whole point of sharing it is that the teacher can correct the grade."""
    drive = _RecordingDrive()
    upload.write_report(drive, "folder", "essay_bdika_2026-08-10_1015.docx", b"PK\x03\x04")
    body = drive.created[0]
    assert body["mimeType"] == upload.GDOC_MIME
    assert body["name"] == "essay_bdika_2026-08-10_1015"   # no .docx suffix


def test_share_grants_editor_without_a_drive_notification():
    """Drive's own notification would arrive alongside ours, for one submission."""
    drive = _RecordingDrive()
    assert upload.share(drive, "doc1", "teacher@x.org", role="writer") is True
    file_id, body, kw = drive.permissions_granted[0]
    assert (file_id, body["role"], body["emailAddress"]) == ("doc1", "writer", "teacher@x.org")
    assert kw["sendNotificationEmail"] is False


def test_share_failure_never_raises():
    """Usually just an address with no Google account behind it. A graded report
    already in the folder must not be lost over that."""
    class _Boom(_RecordingDrive):
        def permissions(self):
            raise RuntimeError("no such user")
    assert upload.share(_Boom(), "doc1", "nobody@nowhere", role="writer") is False


class _FakeSheets:
    def __init__(self, fail=False):
        self.writes, self.fail = [], fail

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def update(self, spreadsheetId=None, range=None, body=None, **kw):
        self._call = (range, body)
        return self

    def execute(self):
        if self.fail:
            raise RuntimeError("403 insufficient scope")
        self.writes.append(self._call)
        return {}


def test_link_column_is_found_by_header_not_by_position():
    """Everything else maps by header text for a reason: Avishai keeps editing
    the form, and a fixed index I would start overwriting a real answer column
    the first time he inserts a question."""
    header = LIVE_HEADER + ["Something new", "Bdika report"]
    assert sheet_link.find_or_create_column(_FakeSheets(), "sid", header) == 9


def test_link_column_defaults_to_I_on_the_first_run():
    sheets = _FakeSheets()
    assert sheet_link.find_or_create_column(sheets, "sid", LIVE_HEADER) == 8
    assert sheets.writes[0] == ("I1", {"values": [["Bdika report"]]})


def test_link_column_returns_none_without_the_write_scope():
    """Grading must carry on; a missing spreadsheet link is not worth failing a
    submission over."""
    assert sheet_link.find_or_create_column(_FakeSheets(fail=True), "sid", LIVE_HEADER) is None


def test_write_link_targets_the_row_a_person_would_see():
    sheets = _FakeSheets()
    assert sheet_link.write_link(sheets, "sid", 7, 8, "http://doc", tag="[t]") is True
    assert sheets.writes[0][0] == "I7"


def test_write_link_failure_never_raises():
    assert sheet_link.write_link(_FakeSheets(fail=True), "sid", 7, 8, "http://doc") is False


def test_mail_carries_the_doc_link_when_there_is_one():
    _, without = _send(_PARAMS)
    assert "Google Doc you can edit" not in _text(without)
    gmail = _FakeGmail()
    mailer.send_report(gmail, "ori@bdika.net", _PARAMS, b"PK\x03\x04", "r.docx",
                       "MoE Module G", tag="[t]", doc_link="http://doc/1")
    body = _text(message_from_bytes(
        base64.urlsafe_b64decode(gmail.sent[0]["raw"]), policy=_email_policy))
    assert "http://doc/1" in body
    # Both, not either — the attachment is the no-friction path.
    msg = message_from_bytes(base64.urlsafe_b64decode(gmail.sent[0]["raw"]),
                             policy=_email_policy)
    assert [p.get_filename() for p in msg.walk() if p.get_filename()] == ["r.docx"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
