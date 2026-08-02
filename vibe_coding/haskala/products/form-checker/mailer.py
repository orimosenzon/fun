"""mailer.py — deliver the report to the teacher who submitted the form.

Delivery is the whole reason this product differs from oris-scanner. There, a
report lands in a Drive folder the teacher already has open in Classroom. Here
the teacher filled in a form and closed the tab; if nothing arrives in their
inbox, as far as they are concerned nothing happened.

Mail is sent as the delegated Workspace user (main.TEACHER_EMAIL), so it arrives
from a real bdika.net address rather than a no-reply nobody recognises.
"""
from __future__ import annotations

import base64
import logging
from email.message import EmailMessage

log = logging.getLogger("form-checker-mailer")

_DOCX_MIME = ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document")


def _send(gmail_service, message: EmailMessage, tag: str) -> bool:
    """True if Gmail accepted the message.

    Never raises. A delivery failure must not fail the response — the report is
    already graded and main.py has a Drive fallback; losing the mail costs a
    link, losing the run costs another full round of model calls."""
    try:
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = gmail_service.users().messages().send(
            userId="me", body={"raw": raw}).execute()
        log.info("%s emailed %s (message id=%s)", tag, message["To"], sent.get("id"))
        return True
    except Exception as e:
        # Overwhelmingly a 403 from a delegation entry without gmail.send. Say
        # so explicitly — the fix is one admin-console edit, and this log line
        # is where it gets noticed.
        log.warning("%s could not email %s (%s: %s) — falling back to Drive delivery. "
                    "If this is a 403, add https://www.googleapis.com/auth/gmail.send "
                    "to the service account's domain-wide delegation in the Workspace "
                    "admin console.",
                    tag, message["To"], type(e).__name__, str(e)[:200])
        return False


def send_report(gmail_service, sender: str, params: dict, docx_bytes: bytes,
                filename: str, rubric_name: str, tag: str = "") -> bool:
    """Mail the graded report back, .docx attached."""
    msg = EmailMessage()
    msg["To"] = params["email"]
    msg["From"] = sender
    msg["Subject"] = "הבדיקה מוכנה — %s" % (params.get("school") or "תרגיל באנגלית")

    greeting = f"שלום {params['teacher_name']}," if params.get("teacher_name") else "שלום,"
    lines = [
        greeting,
        "",
        "התרגיל ששלחת בטופס נבדק, והדוח מצורף כקובץ Word.",
        "",
        f"מחוון: {rubric_name}",
    ]
    if params.get("grade_level"):
        lines.append(f"שכבת גיל: {params['grade_level']}")
    lines += [
        "",
        "הדוח כולל את הציון לכל קריטריון, הערות מסומנות בגוף הטקסט, "
        "וסריקת המקור בסופו.",
        "",
        "הבדיקה בוצעה אוטומטית והיא בגדר המלצה — שיקול הדעת נשאר אצלך.",
        "",
        "בהצלחה,",
        "השכלה",
    ]
    msg.set_content("\n".join(lines))
    msg.add_attachment(docx_bytes, maintype=_DOCX_MIME[0], subtype=_DOCX_MIME[1],
                       filename=filename)
    return _send(gmail_service, msg, tag)


def send_failure(gmail_service, sender: str, params: dict, reason: str,
                 tag: str = "") -> bool:
    """Tell the teacher their submission could not be graded, and why.

    Worth the extra call: the alternative is a teacher who submitted a HEIC
    photo from an iPhone waiting indefinitely for a report that is never coming,
    and concluding the product is broken. `reason` is written for a teacher to
    read, not for a log."""
    msg = EmailMessage()
    msg["To"] = params["email"]
    msg["From"] = sender
    msg["Subject"] = "לא הצלחנו לבדוק את התרגיל ששלחת"

    greeting = f"שלום {params['teacher_name']}," if params.get("teacher_name") else "שלום,"
    msg.set_content("\n".join([
        greeting,
        "",
        "קיבלנו את הטופס ששלחת, אבל לא הצלחנו לבדוק את התרגיל:",
        "",
        f"    {reason}",
        "",
        "אפשר לשלוח את הטופס שוב אחרי שמתקנים את זה.",
        "",
        "השכלה",
    ]))
    return _send(gmail_service, msg, tag)
