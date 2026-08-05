"""form_schema.py — the bridge between a Google Form and the grading pipeline.

A Form's responses sheet is a table of free text whose column headers are
whatever the teacher-facing question titles happen to say. This module turns one
such row into the typed parameters core.check_pages needs.

WHY MATCH COLUMNS BY HEADER TEXT AND NOT BY POSITION
────────────────────────────────────────────────────
Avishai owns the form and will keep editing it — reordering questions, inserting
one, rewording another. Column *indices* shift silently when he does. Reading
"column D" would then hand the rubric column's text to the grade-level
parameter and grade every submission against the wrong rubric, with nothing in
the logs looking wrong. Matching on the header text turns that same edit into
either a correct match or a loud failure, which is the only pair of outcomes
worth having.

Matching is deliberately fuzzy — normalise away case, punctuation, Hebrew
niqqud and the "(optional)" suffixes Forms likes to add, then look for known
keywords. A question reworded from "בית ספר" to "באיזה בית ספר את/ה מלמד/ת?"
still matches. What it cannot survive is a question that drops the keyword
entirely, which is why print_form_template() exists: run it, hand the output to
Avishai, and the form is built against the mapping rather than reverse-engineered
after the fact.

Ambiguity is resolved by longest-alias-wins. That matters more than it looks:
"מחוון לבדיקה" and "התרגיל לבדיקה" (the upload) both contain "לבדיקה", and the
naive first-match-wins rule maps whichever column comes first.

HOW THIS DIFFERS FROM form-checker's COPY
─────────────────────────────────────────
Same machinery, different vocabulary — the column matching, normalisation and
id extraction below are unchanged. What changed is everything downstream of
them, because math grading takes a different shape of input:

  • No rubric dropdown. The English product picks one of four bundled Ministry
    rubrics by Bagrut module code. Maths has no such national scale, and
    math_core takes a **free-text** מחוון instead — so the form asks for the
    מחוון as a paragraph, or the teacher drops it in the folder as a file.
  • A school-solution question the English form has no equivalent of. Avishai
    added it to the maths form on 2026-08-05: the teacher attaches the school's
    own worked solution, so the model knows what the expected answer and method
    look like instead of re-deriving every problem itself. An English
    composition has no such artefact.
  • No exercise-language question. The maths prompt and its feedback are Hebrew.
  • A missing task description is far less damaging here. An English composition
    graded without knowing the prompt gets an invented Content score; a maths
    page carries the problems printed on it, so the model reads the task off the
    scan itself.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from haskala_grading import math_core

log = logging.getLogger("math-form-checker")


# ─── field definitions ──────────────────────────────────────────────────────
# key, aliases (matched as normalised substrings, Hebrew and English), required.
#
# Required means "the form is unusable without it": we cannot grade with no
# files, cannot dedup with no timestamp, and cannot deliver with no address. A
# missing *optional* column is normal — not every form will ask for a school —
# and simply leaves that parameter unset.

FIELDS: dict[str, dict] = {
    "timestamp": {
        "aliases": ["חותמת זמן", "timestamp", "תאריך שליחה"],
        "required": True,
        "hint": "Forms adds this column itself — do not create it by hand.",
    },
    "email": {
        "aliases": ["כתובת אימייל", "כתובת מייל", "כתובת דואר אלקטרוני",
                    "email address", "e-mail", "אימייל", "מייל", "email"],
        "required": True,
        "hint": "Turn on 'Collect email addresses' in the form settings; "
                "this is where the report is sent.",
    },
    "folder_link": {
        "aliases": ["link to folder", "קישור לתיקייה", "קישור לתיקיה",
                    "shared folder", "drive folder", "folder link", "תיקייה",
                    "תיקיה", "folder"],
        "required": True,
        "hint": "Short answer or paragraph. The teacher pastes a Drive folder "
                "link; the folder must be shared with the service account (see "
                "SHARE_WITH below) with EDITOR access, because results are "
                "written back into it.",
    },
    "files": {
        # "פתרונות התלמידים" is listed here, not under `solution`, and the two
        # are separated by alias length rather than by luck: this phrase is
        # sixteen characters and beats every solution alias, so a form that
        # words the student-upload question that way still maps to the work
        # files. "העלאת" is here for the same family of titles, which say
        # "upload" without ever saying "files".
        "aliases": ["המבחנים לבדיקה", "התרגיל לבדיקה", "העלאת קבצים", "קבצים",
                    "פתרונות התלמידים", "עבודות התלמידים", "מחברות",
                    "סריקות", "העלאת", "upload", "files", "exercise files",
                    "צרף", "צרפו"],
        "required": False,
        "hint": "Optional. A File upload question still works and is graded "
                "alongside the folder, but folder sharing is the primary path — "
                "it is what lets results be written back where the teacher can "
                "see them, and what makes a whole class one submission.",
    },
    "first_name": {
        "aliases": ["first name", "שם פרטי"],
        "required": False,
        "hint": "Short answer.",
    },
    "last_name": {
        "aliases": ["last name", "שם משפחה"],
        "required": False,
        "hint": "Short answer.",
    },
    "teacher_name": {
        "aliases": ["שם המורה", "שם מלא", "שמך", "teacher name", "your name", "שם"],
        "required": False,
        "hint": "Short answer. Only needed if the form asks for the name as one "
                "field instead of first_name + last_name.",
    },
    "comments": {
        # Deliberately not a bare "הערות". A responses sheet grows human-added
        # columns — "הערות פנימיות", "ציון ידני" — and a one-word alias claims
        # them, which would quietly feed our own notes to the model as though the
        # teacher had written them.
        "aliases": ["comments and requests", "comments", "requests",
                    "הערות ובקשות", "הערות מהמורה"],
        "required": False,
        "hint": "Paragraph. Free text from the teacher — appended to the "
                "מחוון as clearly-fenced context, so a request like "
                "'these are weak students, be gentle' actually reaches the model.",
    },
    "school": {
        "aliases": ["בית הספר", "בית ספר", "מוסד", "school"],
        "required": False,
        "hint": "Short answer.",
    },
    "grade_level": {
        "aliases": ["שכבת הגיל", "שכבה", "כיתה", "גיל התלמידים", "גיל",
                    "grade level", "age group", "grade"],
        "required": False,
        "hint": "Short answer or dropdown, e.g. 'כיתה ט'.",
    },
    "rubric": {
        "aliases": ["מחוון הבדיקה", "מחוון לבדיקה", "המחוון", "מחוון",
                    "לפי איזו רובריקה", "רובריקה", "ניקוד", "rubric",
                    "marking scheme", "grading key"],
        "required": False,
        "hint": "Paragraph — FREE TEXT, not a dropdown. The teacher pastes the "
                "marking scheme ('סעיף א: 5 נק', סעיף ב: 10 נק'…'), or writes "
                "that it is attached in the folder. It conditions points_max / "
                "points_earned per problem. Left empty, the model scores by the "
                "points printed on the page, defaulting to 10 per problem.",
    },
    "solution": {
        # Deliberately NOT a bare "פתרון". The student-upload question is very
        # plausibly worded "העלאת פתרונות התלמידים", and "פתרון" is a substring
        # of "פתרונות" — a bare alias would win that column on length and hand
        # the entire class's work to the model as the reference solution, which
        # would mark every student correct. Every alias here is qualified.
        # The first three are the title Avishai actually used, confirmed
        # 2026-08-05 from the Drive folder Forms created for the question
        # ("התרגיל והפתרון המוצע (File responses)"). Note it bundles the
        # exercise *and* the proposed solution into one upload, which is why
        # the prompt calls the attachment "the exam and its solution" rather
        # than "the solution" — see math_core._SOLUTION_INTRO.
        "aliases": ["התרגיל והפתרון המוצע", "הפתרון המוצע", "פתרון מוצע",
                    "פתרון בית הספר", "פתרון בית ספר", "פתרון בית־ספר",
                    "הפתרון של בית הספר", "פתרון המורה", "פתרון מורה",
                    "פתרון לדוגמה", "פתרון מלא", "הפתרון הרשמי",
                    "school solution", "reference solution", "model solution",
                    "answer key", "official solution"],
        "required": False,
        "hint": "File upload (preferred) or paragraph. The school's own worked "
                "solution, optionally together with the exam paper. Either "
                "question type works — the answer is inspected at parse time: "
                "Drive links become solution files, prose becomes solution "
                "text, and 'מצורף בתיקייה' sends us to the folder.",
    },
    "instructions": {
        "aliases": ["הוראות המשימה", "תיאור התרגיל", "מה התלמידים התבקשו",
                    "המשימה", "הוראות", "נושא המבחן", "assignment",
                    "instructions", "task", "the task", "question",
                    "the question", "what were the students asked", "prompt"],
        "required": False,
        "hint": "Paragraph, optional. Context about the exam — topic, chapter, "
                "what was allowed. Unlike the English product this is NOT what "
                "the grade is measured against: the problems are printed on the "
                "scan and the model reads them from there.",
    },
    "model": {
        "aliases": ["מודל", "model"],
        "required": False,
        "hint": "Optional/advanced — leave it off the teacher-facing form "
                "unless you want teachers choosing providers. Default is "
                "Sonnet 5; 'Opus' is the quality escalation.",
    },
    "passphrase": {
        "aliases": ["סיסמה", "סיסמת גישה", "קוד גישה", "passphrase", "access code"],
        "required": False,
        "hint": "Short answer. Only needed if FORM_CHECKER_PASSPHRASE is set on "
                "the service — see the cost guardrails in main.py.",
    },
}

# A מחוון answer that says "it's in the folder" rather than pasting the scheme
# itself. Matched loosely — teachers phrase this a dozen ways.
RUBRIC_IN_FOLDER_HINTS = ("צירפתי", "מצורף", "בתיקייה", "בתיקיה", "בקבצים",
                          "attached", "in the folder", "see folder")

# Filename fragments that mark a file in the shared folder as the school's
# reference solution rather than a student's work.
#
# This list earns its keep twice. Feeding the reference to the model as though
# it were student work would score the teacher's own answers — a perfect grade
# under some student's name — and it is not a hypothetical: a *handwritten*
# school solution is exactly what a maths teacher scans, and nothing else in the
# pipeline can tell that handwriting apart from a pupil's.
#
# Here a bare "פתרון" IS wanted, unlike in the field aliases above: these match
# filenames the teacher chose for one file, not question titles, and "פתרון.pdf"
# is the overwhelmingly common name for exactly this file.
SOLUTION_NAME_HINTS = ("פתרון", "פיתרון", "תשובות", "מחברת המורה",
                       "solution", "answers", "answer key", "teacher")


def looks_like_solution(name: str) -> bool:
    """True when a folder filename names the school's reference solution.

    Deliberately not applied to the מחוון: a marking scheme is read as text and
    a solution is read as pages, and a file called 'מחוון ופתרון' is better
    treated as the solution — the scheme survives in the pages either way."""
    return any(normalise(h) in normalise(name) for h in SOLUTION_NAME_HINTS)


def points_to_folder(label: str) -> bool:
    """True when a free-text answer says 'it's attached' rather than containing
    the thing itself. Shared by the מחוון and the school-solution questions —
    teachers answer both the same way, and there is no reason for two lists."""
    if not label:
        return False
    norm = normalise(label)
    return any(normalise(h) in norm for h in RUBRIC_IN_FOLDER_HINTS)


def rubric_is_in_folder(label: str) -> bool:
    """True when the מחוון answer points at the folder instead of containing the
    scheme. The caller then looks for a typed document in the folder and uses
    its text as the rubric."""
    return points_to_folder(label)


# ─── grading with no מחוון ───────────────────────────────────────────────────
# Unlike the English product there is no NO_TASK_NOTICE here, and that asymmetry
# is deliberate rather than an omission.
#
# An English composition graded without its prompt gets a Content score invented
# from a topic the model inferred out of the composition itself — a confident,
# unearned mark that has to be suppressed explicitly. A maths page has the
# problems printed on it. The model reads the task off the scan, so a missing
# form field costs context, not correctness.
#
# The מחוון is the analogous risk, and ANALYSIS_PROMPT already handles it: with
# no scheme supplied it scores by the points printed beside each problem, and
# falls back to 10 only when the page shows nothing. That is a stated rule with
# a visible default, not a silent guess, so it needs no counter-notice here.


def resolve_model_choice(label: str) -> str:
    """Model dropdown label → math_core.MODELS key, default math_core.DEFAULT_MODEL.

    Accepts the key ("opus"), the human label the form shows ("Claude Opus 4.8 ·
    איכות מרבית"), or a bare family name a teacher might type."""
    norm = normalise(label)
    if not norm:
        return math_core.DEFAULT_MODEL

    # Exact matches only, and in this order. Substring matching on keys and
    # aliases was the obvious implementation and it is wrong: the alias "gpt"
    # is a substring of "gpt-9 ultra", so a teacher naming a model we do not
    # have would silently be graded on Azure GPT-4.1-mini instead of falling
    # back to the default. An unrecognised name must land on the default and
    # say so, never on whichever alias happens to be a substring of it.
    if norm in math_core.MODELS:
        return norm
    if norm in _MODEL_ALIASES:
        return _MODEL_ALIASES[norm]

    # The human label the form actually shows, e.g. "Claude Opus 4.8 · איכות
    # מרבית". Compared whole, so it cannot fire on a fragment either.
    for key, human in math_core.MODELS.items():
        if normalise(human) == norm:
            return key

    log.warning("model %r not recognised — falling back to %s",
                label, math_core.DEFAULT_MODEL)
    return math_core.DEFAULT_MODEL


# Spellings a teacher might plausibly type. Keys of math_core.MODELS are always
# accepted; this is only for the aliases. Note "sonnet" alone is deliberately
# NOT mapped: math_core has both a sonnet5 and a sonnet (4.6) key, and guessing
# which one a bare "sonnet" meant would hand someone a different model without
# a word. Left unmapped it produces a warning and grades with the default.
_MODEL_ALIASES = {
    "sonnet 5": "sonnet5",
    "claude sonnet 5": "sonnet5",
    "opus 4.8": "opus",
    "claude opus": "opus",
    "anthropic": "sonnet5",
    "gemini": "gemini-flash",
    "gemini flash": "gemini-flash",
    "flash lite": "gemini-lite",
    "groq": "groq-scout",
    "llama": "groq-scout",
    "azure": "azure-gpt41-mini",
    "gpt": "azure-gpt41-mini",
}


# ─── normalisation ──────────────────────────────────────────────────────────

_NIQQUD = re.compile(r"[֑-ׇ]")          # Hebrew cantillation + vowel points
_PUNCT = re.compile(r"[^\w֐-׿؀-ۿ]+")  # keep letters/digits/Hebrew/Arabic


def normalise(s: str) -> str:
    """Lowercased, unaccented, punctuation-collapsed form used for all matching.

    Strips Hebrew niqqud (a teacher may paste a vowelised title), the * Forms
    appends to required questions, and the parenthesised notes it appends to
    file-upload columns."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = _NIQQUD.sub("", s)
    s = _PUNCT.sub(" ", s.lower())
    return " ".join(s.split())


def _best_field(header: str) -> tuple[str | None, int]:
    """(field key, matched alias length) for the field this header best fits.

    Longest alias wins — see the module docstring on שפת התרגיל vs התרגיל."""
    norm = normalise(header)
    if not norm:
        return None, 0
    best_key, best_len = None, 0
    for key, spec in FIELDS.items():
        for alias in spec["aliases"]:
            na = normalise(alias)
            if na and na in norm and len(na) > best_len:
                best_key, best_len = key, len(na)
    return best_key, best_len


def map_columns(header_row: list[str]) -> dict[str, int]:
    """{field key: column index} for one responses-sheet header row.

    Raises ValueError naming every missing required field — the caller turns
    that into a loud failure rather than grading half a row, because a form
    whose email column cannot be found has nowhere to send the report anyway.

    When two columns match the same field, the stronger match wins and the other
    is left unmapped; both are logged, since it usually means the form has two
    similar questions and one of them is being ignored.
    """
    mapping: dict[str, int] = {}
    strength: dict[str, int] = {}
    for idx, header in enumerate(header_row):
        key, score = _best_field(header)
        if not key:
            log.debug("column %d %r matched no known field — ignored", idx, header)
            continue
        if key in mapping and strength[key] >= score:
            log.warning("column %d %r also looks like %r, which column %d already "
                        "matched more strongly — ignoring this one",
                        idx, header, key, mapping[key])
            continue
        if key in mapping:
            log.warning("column %d %r matches %r more strongly than column %d — "
                        "using this one instead", idx, header, key, mapping[key])
        mapping[key] = idx
        strength[key] = score

    missing = [k for k, spec in FIELDS.items() if spec["required"] and k not in mapping]
    if missing:
        raise ValueError(
            "the responses sheet is missing required column(s): %s. Headers seen: %s. "
            "Run `python form_schema.py --template` for the question titles the form "
            "is expected to use." % (", ".join(missing), header_row))
    return mapping


# ─── row → grading parameters ───────────────────────────────────────────────

_FILE_ID_PATTERNS = [
    re.compile(r"/file/d/([\w-]{20,})"),
    re.compile(r"/d/([\w-]{20,})"),
    re.compile(r"[?&]id=([\w-]{20,})"),
    re.compile(r"/open\?id=([\w-]{20,})"),
]


def extract_file_ids(cell: str) -> list[str]:
    """Drive file ids out of a file-upload answer.

    Forms writes these as comma-separated links, and the exact URL shape has
    changed more than once (/open?id=, /file/d/, /d/), so every known form is
    tried. A bare id pasted by hand is accepted too. Order is preserved and
    duplicates dropped — page order is the order the teacher uploaded in, and
    the report reads wrong if it is shuffled."""
    if not cell:
        return []
    ids: list[str] = []
    for chunk in re.split(r"[,\s]+", str(cell)):
        if not chunk:
            continue
        found = None
        for pat in _FILE_ID_PATTERNS:
            m = pat.search(chunk)
            if m:
                found = m.group(1)
                break
        if not found and re.fullmatch(r"[\w-]{20,}", chunk):
            found = chunk          # a bare id
        if found and found not in ids:
            ids.append(found)
    return ids


_FOLDER_ID_PATTERNS = [
    re.compile(r"/folders/([\w-]{15,})"),
    re.compile(r"/drive/u/\d+/folders/([\w-]{15,})"),
    re.compile(r"[?&]id=([\w-]{15,})"),
]


def extract_folder_id(cell: str) -> str:
    """Drive folder id out of whatever the teacher pasted, "" if there is none.

    Teachers paste a share link, a browser address-bar URL (which carries the
    /u/0/ account index), or occasionally the bare id. All three are accepted;
    the trailing ?usp=sharing is ignored because the patterns stop at the id.

    Only the first id found is returned. A teacher who pastes two links has
    given us an ambiguous answer, and grading the first one silently is better
    than grading both — the second would be someone else's folder, and writing
    results into it is not a mistake worth making automatically.
    """
    if not cell:
        return ""
    text = str(cell).strip()
    for pat in _FOLDER_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    if re.fullmatch(r"[\w-]{15,}", text):
        return text          # a bare id
    return ""


def build_rubric(params: dict) -> str:
    """The free-text מחוון handed to math_core.analyze_page().

    math_core appends this to the *user* turn, not the cached system prompt, so
    per-teacher text conditions the scoring without busting the prompt cache.

    Three sources are merged, in descending authority:
      1. the מחוון the teacher pasted on the form (or the text of the מחוון file
         found in their folder, which the caller passes in as folder_rubric);
      2. class context — grade level, school — which sets the expectation bar;
      3. the teacher's free-text comments.

    Context is appended rather than prepended so the teacher's own scheme stays
    the first thing the model reads. Each block is fenced and labelled: a
    required free-text field is as likely to say "תודה!" as anything useful, and
    an unlabelled pleasantry reads as part of the marking scheme.
    """
    parts: list[str] = []
    scheme = (params.get("rubric_text") or "").strip()
    if scheme:
        parts.append(scheme)

    context = []
    if params.get("grade_level"):
        context.append(f"שכבת גיל: {params['grade_level']}")
    if params.get("school"):
        context.append(f"בית ספר: {params['school']}")
    if params.get("instructions"):
        context.append(f"על המבחן: {params['instructions']}")
    if context:
        parts.append("הקשר כיתתי (לא חלק מהמחוון):\n" + "\n".join(context))

    if params.get("comments"):
        parts.append("הערות מהמורה (לא חלק מהמחוון):\n" + params["comments"])
    return "\n\n".join(parts)


def parse_row(row: list[str], colmap: dict[str, int]) -> dict:
    """One responses row → the parameters the maths grading pipeline takes.

    Short rows are normal: Sheets truncates trailing empty cells, so a response
    that skipped the last few optional questions comes back shorter than the
    header. Every read is bounds-checked for that reason."""
    def cell(key: str) -> str:
        idx = colmap.get(key)
        if idx is None or idx >= len(row):
            return ""
        return (row[idx] or "").strip()

    rubric_answer = cell("rubric")
    in_folder = points_to_folder(rubric_answer)

    # The school-solution question is read without caring which *type* of
    # question Avishai built, because both are reasonable and we do not control
    # the form. A File-upload answer is a list of Drive links; a paragraph
    # answer is either the solution itself or a note that it is attached. Asking
    # the cell what it contains handles all three, and means a teacher who
    # switches the question from upload to paragraph next term breaks nothing.
    solution_answer = cell("solution")
    solution_ids = extract_file_ids(solution_answer)
    solution_in_folder = points_to_folder(solution_answer) and not solution_ids

    # "Assignment description" is a *File upload* question on Avishai's maths
    # form, not the paragraph the English form uses (confirmed 2026-08-05 from
    # the folder Forms created for it). Read as text, its answer is a Drive URL,
    # and build_rubric would paste "על המבחן: https://drive.google.com/..." into
    # the marking scheme — a line of noise presented to the model as context
    # about the exam. Nothing would error and no log would look wrong.
    instructions_answer = cell("instructions")
    instructions_ids = extract_file_ids(instructions_answer)
    if instructions_ids:
        log.info("the task-description answer is %d uploaded file(s), not text — "
                 "keeping the link(s) out of the rubric", len(instructions_ids))

    # One name field, however the form spells it. A form asking First + Last
    # gives us two columns; an older one gives a single "שם המורה". Joining here
    # means everything downstream — the greeting, the report filename — reads
    # one key and does not care which form produced the row.
    name = " ".join(p for p in (cell("first_name"), cell("last_name")) if p)
    if not name:
        name = cell("teacher_name")

    return {
        "timestamp": cell("timestamp"),
        "email": cell("email"),
        "teacher_name": name,
        "first_name": cell("first_name"),
        "last_name": cell("last_name"),
        "comments": cell("comments"),
        "folder_id": extract_folder_id(cell("folder_link")),
        "folder_link": cell("folder_link"),
        "school": cell("school"),
        "grade_level": cell("grade_level"),
        "instructions": "" if instructions_ids else instructions_answer,
        # Parsed and carried, but deliberately not fed to the model yet: whether
        # the exam paper should join the reference pages is Avishai's call, not
        # a default we should pick silently. Available the moment that is decided.
        "instructions_file_ids": instructions_ids,
        "passphrase": cell("passphrase"),
        # An answer that merely points at the folder is not itself a scheme —
        # keeping it out of rubric_text stops "מצורף בתיקייה" being handed to
        # the model as though it were the marking criteria.
        "rubric_text": "" if in_folder else rubric_answer,
        "rubric_in_folder": in_folder,
        "rubric_label": rubric_answer,
        "model_key": resolve_model_choice(cell("model")),
        "file_ids": extract_file_ids(cell("files")),
        # The school's reference solution. Exactly one of these three is
        # meaningful per submission, and all three are carried so the caller can
        # say in the summary mail which one it used — "no solution was found" and
        # "the solution file could not be read" are different problems for the
        # teacher, and indistinguishable if we collapse them here.
        "solution_file_ids": solution_ids,
        # A note that the solution is attached is not itself a solution. Keeping
        # it out of solution_text stops "מצורף בתיקייה" being shown to the model
        # as the school's worked answer — the same trap the מחוון field has.
        "solution_text": "" if (solution_ids or solution_in_folder) else solution_answer,
        "solution_in_folder": solution_in_folder,
        "solution_label": solution_answer,
    }



# ─── the form-building aid ──────────────────────────────────────────────────

def print_form_template() -> None:
    """Print the question titles and dropdown options the form should use.

    This is the intended way to build the form: run it, hand the output over,
    and the schema and the form agree by construction. Reverse-engineering the
    mapping from a form that already exists is the path that produces silent
    mismatches.
    """
    print("Google Form — questions math-form-checker knows how to read")
    print("=" * 62)
    print("\nHeaders are matched on keywords, so a question may be reworded freely")
    print("as long as it still contains one of the keywords listed.\n")
    for key, spec in FIELDS.items():
        req = "REQUIRED" if spec["required"] else "optional"
        print(f"── {key}  ({req})")
        print(f"   suggested title : {spec['aliases'][0]}")
        print(f"   keywords        : {', '.join(spec['aliases'][:5])}")
        print(f"   note            : {spec['hint']}")
        print()

    print("=" * 62)
    print("The מחוון question is FREE TEXT (paragraph) — there is no dropdown.")
    print("Maths has no national rubric scale to pick from, so the teacher")
    print("pastes the scheme itself, e.g.:\n")
    print("   סעיף א: 5 נק' · סעיף ב: 10 נק' · סעיף ג: 15 נק'")
    print("   הורדה של נקודה אחת על טעות חישוב, שתיים על טעות שיטתית.\n")
    print("An answer that instead points at the folder — any of:")
    print(f"   {', '.join(RUBRIC_IN_FOLDER_HINTS)}")
    print("— makes us look for a typed מחוון document inside the shared folder")
    print("and use its text instead.\n")
    print("(no answer → the model scores by the points printed beside each")
    print(" problem on the page, defaulting to 10 per problem)\n")

    print("=" * 62)
    print("The פתרון בית הספר question accepts EITHER question type.")
    print("File upload is preferred — a maths solution is diagrams and")
    print("derivations, and its pages are sent to the model as images.")
    print("A paragraph answer works too and is sent as text.\n")
    print("Whatever the teacher attaches is shown to the model as the school's")
    print("solution, explicitly not as the pupil's work, with the rule that a")
    print("different valid method still earns full marks.\n")
    print("A file in the shared folder is used when the question went")
    print("unanswered, matched on its name:")
    print(f"   {', '.join(SOLUTION_NAME_HINTS)}")
    print("Those names are also excluded from the student-work pile — a")
    print("handwritten school solution would otherwise be graded as a pupil's.\n")
    print("NOTE — form file uploads land in the FORM OWNER's Drive, so the form")
    print("must be owned by the same Workspace as the service account or the")
    print("attachment cannot be read at all.\n")

    print("Dropdown options for the model question, if you expose it at all:\n")
    for key, human in math_core.MODELS.items():
        mark = "  ← default" if key == math_core.DEFAULT_MODEL else ""
        print(f"   • {human}{mark}")
    print()


if __name__ == "__main__":
    import sys
    if "--template" in sys.argv:
        print_form_template()
    else:
        print("usage: python form_schema.py --template")
