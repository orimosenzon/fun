"""mailer.py — deliver the report to the teacher who submitted the form.

Delivery is the whole reason this product differs from oris-scanner. There, a
report lands in a Drive folder the teacher already has open in Classroom. Here
the teacher filled in a form and closed the tab; if nothing arrives in their
inbox, as far as they are concerned nothing happened.

Mail is sent as the delegated Workspace user (main.WORKSPACE_SUBJECT), so it arrives
from a real bdika.net address rather than a no-reply nobody recognises.
"""
from __future__ import annotations

import base64
import logging
from email.message import EmailMessage

log = logging.getLogger("form-checker-mailer")


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


def send_summary(gmail_service, sender: str, params: dict, results: dict,
                 rubric_name: str, output_link: str, tag: str = "") -> bool:
    """Tell the teacher the folder has been checked, and what happened to each file.

    Nothing is attached. The reports are already Word files in the teacher's own
    folder — attaching copies would double every report and leave two versions to
    keep straight. What the mail carries instead is the part the folder cannot
    show: which files were graded, which were treated as the task rather than as
    student work, and which failed and why.

    That per-file accounting is the point. A teacher who shares thirty scans and
    gets back twenty-eight reports will not diff the two lists by hand; if we do
    not tell them which two are missing, they will assume the product silently
    lost them.
    """
    graded = results.get("graded") or []
    failed = results.get("failed") or []
    skipped = results.get("skipped") or []
    context = results.get("context") or []
    deferred = results.get("deferred") or []
    already = results.get("already", 0)

    msg = EmailMessage()
    msg["To"] = params["email"]
    msg["From"] = sender
    msg["Subject"] = ("הבדיקה מוכנה — %d עבודות" % len(graded) if graded
                      else "לא נבדקו עבודות בתיקייה ששיתפת")

    greeting = f"שלום {params['teacher_name']}," if params.get("teacher_name") else "שלום,"
    lines = [greeting, ""]

    if graded:
        lines += [
            "בדקנו את התיקייה ששיתפת. הדוחות ממתינים לך בתוך התיקייה, "
            "בתת-התיקייה Bdika — קובץ Word לכל עבודה:",
            "",
            f"    {output_link}",
            "",
            f"מחוון: {rubric_name}",
            "",
            # The single most important sentence in this mail. The report shows
            # a number per problem, and a number in a document looks like a
            # grade. It is a suggestion nobody has approved — the maths UI has a
            # screen where the teacher edits and confirms the points, and a form
            # submission has no such screen. Said here as well as on the report
            # itself, because a teacher who forwards the file has already
            # skipped the cover page.
            "⚠️  הניקוד בדוחות הוא הצעה של המערכת ולא ציון סופי. הוא נועד לחסוך "
            "לך את המעבר הראשון על המחברות — עברו עליו, תקנו ואשרו לפני שהוא "
            "מגיע לתלמידים.",
            "",
            "נבדקו:",
        ] + [f"    ✓ {name}" for name in graded]
    else:
        lines += ["עברנו על התיקייה ששיתפת, אבל לא היה בה משהו חדש לבדוק."]

    if not results.get("task_known", True) and graded:
        # Said before the file lists, not buried after them: it changes how the
        # whole batch should be read, and a teacher who misses it will take an
        # incomplete Content mark for a final one.
        # Softer than the English product's equivalent, and deliberately so: a
        # missing מחוון is a degraded grade, not an unscoreable one. The model
        # falls back to the points printed beside each problem on the page, so
        # the marks are still anchored to something the teacher wrote — just to
        # the exam paper rather than to their marking scheme.
        lines += [
            "",
            "ℹ️  לא קיבלנו מחוון, אז הניקוד נקבע לפי הנקודות המודפסות ליד כל "
            "תרגיל בדף (ובאין כאלה — 10 נקודות לתרגיל). אם החלוקה שלך שונה, "
            "אפשר להדביק את המחוון בטופס בשליחה הבאה ולקבל ניקוד מדויק יותר.",
        ]

    if deferred:
        lines += [
            "",
            "נשארו לבדיקה בהמשך (%d) — אנחנו בשלב הרצה ובודקים עד %d עבודות "
            "בכל שליחה:" % (len(deferred), len(graded) + len(failed)),
        ] + [f"    · {name}" for name in deferred] + [
            "כדי לבדוק את הבאות בתור, פשוט שלחו את הטופס שוב עם אותה תיקייה — "
            "מה שכבר נבדק לא ייבדק שוב.",
        ]

    if already:
        lines += ["", "כבר נבדקו בעבר ולכן דילגנו עליהן (%d):" % already,
                  "    הדוחות שלהן כבר נמצאים בתת-התיקייה Bdika."]

    if context:
        lines += ["", "זוהו כמסמכי רקע ולא כעבודות תלמידים:"]
        lines += [f"    • {name} — {kind}" for name, kind in context]

    if skipped:
        lines += ["", "דילגנו עליהם:"]
        lines += [f"    • {name} — {reason}" for name, reason in skipped]

    if failed:
        lines += ["", "לא הצלחנו לבדוק:"]
        lines += [f"    ✗ {name} — {reason}" for name, reason in failed]

    lines += [
        "",
        "הבדיקה בוצעה אוטומטית והיא בגדר המלצה — שיקול הדעת נשאר אצלך.",
        "",
        "בהצלחה,",
        "השכלה",
    ]
    msg.set_content("\n".join(lines))
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
