"""mailer.py — deliver the report to the teacher who submitted the form.

Delivery is the whole reason this product exists separately. oris-scanner writes
into a Drive folder the teacher already has open in Classroom, and form-checker
writes back into the folder the teacher shared. Here the teacher uploaded one
file and closed the tab: there is no folder, so **mail is the only way anything
reaches them**. A delivery failure is a total failure of the submission, not the
loss of a convenience — which is why main.py escalates a mail failure to the log
as an error and marks the row for retry rather than done.

Mail is sent as the delegated Workspace user (main.TEACHER_EMAIL), so it arrives
from a real bdika.net address rather than a no-reply nobody recognises.

The report goes out as an attachment rather than a link. A teacher trying the
demo has no relationship with us yet; a Drive link would ask them to click
through a permission prompt from a domain they do not recognise, and that is the
step where a demo gets abandoned.
"""
from __future__ import annotations

import base64
import logging
import os
from email.message import EmailMessage

log = logging.getLogger("demo-checker-mailer")

_DOCX_MIME = ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document")

# Avishai's "Feedback to Bdika Demo- English" survey, in the bdika.net Workspace.
# Overridable so a test run can point somewhere harmless — a live outreach link
# in a smoke test is a real response in the real sheet.
FEEDBACK_FORM_URL = os.environ.get(
    "FEEDBACK_FORM_URL",
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSfiVGuDf1xNdwcpVoZNAESLbb6kqmxQq-INS44KubQAB_lo1Q/viewform")


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
                filename: str, rubric_name: str, tag: str = "",
                doc_link: str = "") -> bool:
    """Mail the graded report back, .docx attached, and ask for feedback.

    The wording is Avishai's, from the "Bdika demo form" tab of the plan doc —
    he owns the teacher relationship and signs the mail, so the copy is his and
    changes to it should come from him. Two things were added on top of his
    draft, both marked below: a line naming the attachment, because his text
    assumes the Doc link is the delivery and that fails for a submitter with no
    Google account, and the missing-assignment note.

    The feedback link is the reason the demo exists at all — a graded report we
    never hear back about tells us nothing."""
    name = params.get("teacher_name") or "there"
    msg = EmailMessage()
    msg["To"] = params["email"]
    msg["From"] = sender
    msg["Subject"] = f"Your checked document is ready + quick feedback, {name}!"

    lines = [
        f"Dear {name},",
        "",
        "We've reviewed your submission, and your checked document is ready for you.",
    ]
    if doc_link:
        lines += [
            "",
            "View Your Checked Document:",
            doc_link,
        ]
    # Not in Avishai's draft, and added rather than left out: his text assumes
    # the Doc link is the delivery. A submitter with no Google account cannot
    # open it at all, and the attachment is what makes this product work with no
    # permission prompt — so it is mentioned rather than silently attached.
    lines += [
        "",
        "The same report is attached to this email as a Word document, in case "
        "you would rather not open Drive.",
    ]
    if not (params.get("instructions") or "").strip():
        # Say it here too, not only inside the report. A teacher who skipped the
        # assignment field has no reason to expect the content score to be
        # qualified, and will read a hedged Content mark as the tool being vague.
        lines += [
            "",
            "One note: you did not describe the assignment on the form, so we "
            "could not judge whether the student answered the question. "
            "Everything else — vocabulary, language use, mechanics — was graded "
            "normally.",
        ]
    lines += [
        "",
        "Since we're tweaking this process to fit your personal teaching needs, "
        "your thoughts mean the world to us:",
        "",
        "Your Feedback:",
        FEEDBACK_FORM_URL,
        "",
        "Thanks again for testing this out with us. We'd love to see you use "
        "this tool in the coming school year, so let's definitely stay in touch! "
        "If you ever have ideas, questions, or just want to chat about how it's "
        "working for you, hit reply anytime.",
        "",
        "Best,",
        "",
        "Avishai Chelouche",
        "Bdika team member",
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
    msg["Subject"] = "We could not grade the work you uploaded"

    greeting = f"Hello {params['teacher_name']}," if params.get("teacher_name") else "Hello,"
    msg.set_content("\n".join([
        greeting,
        "",
        "We received your submission, but could not grade it:",
        "",
        f"    {reason}",
        "",
        "You are welcome to submit the form again once that is sorted out.",
        "",
        "Haskala",
    ]))
    return _send(gmail_service, msg, tag)
