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
"שפת התרגיל" (exercise language) and "התרגיל לבדיקה" (the upload) both contain
"תרגיל", and the naive first-match-wins rule maps whichever column comes first.

HOW THIS DIFFERS FROM form-checker's COPY
─────────────────────────────────────────
Same form, one structural change: the teacher uploads a file instead of sharing
a folder. So `files` is required here and `folder_link` does not exist at all,
where form-checker has it the other way round. Dropping the folder field rather
than leaving it optional is deliberate — an unused optional field invites the
next reader to wire it up, and there is nowhere in this product for a folder to
go, since the report leaves by email.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from haskala_grading import core

log = logging.getLogger("form-checker")


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
    "files": {
        "aliases": ["upload student's work", "התרגיל לבדיקה", "העלאת קבצים",
                    "קבצים", "סריקות", "upload", "files", "exercise files",
                    "צרף", "צרפו"],
        "required": True,
        "hint": "File upload question, and the whole point of this form. One "
                "scan of one piece of work; the graded report comes back by "
                "email. Uploads land in the Drive of whoever owns the form, so "
                "the form must be owned inside the bdika.net Workspace.",
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
        "hint": "Paragraph. Free text from the teacher — passed to the model as "
                "context alongside the task instructions, so a request like "
                "'these are weak students, be gentle' actually reaches it.",
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
        "aliases": ["לפי איזו רובריקה", "רובריקה", "מחוון", "rubric"],
        "required": False,
        "hint": "Dropdown. Use exactly the options printed below.",
    },
    "instructions": {
        "aliases": ["הוראות המשימה", "תיאור התרגיל", "מה התלמידים התבקשו",
                    "המשימה", "הוראות", "assignment", "instructions", "task",
                    "the task", "writing task", "question", "the question",
                    "what were the students asked", "prompt"],
        "required": False,
        "hint": "Paragraph. What the students were actually asked to do — this "
                "is what the model grades the answer against.",
    },
    "exercise_lang": {
        "aliases": ["שפת התרגיל", "באיזו שפה", "שפה", "language"],
        "required": False,
        "hint": "Dropdown: אנגלית / עברית / ערבית. Defaults to English.",
    },
    "model": {
        "aliases": ["מודל", "model"],
        "required": False,
        "hint": "Optional/advanced — leave it off the teacher-facing form "
                "unless you want teachers choosing providers.",
    },
    "passphrase": {
        "aliases": ["סיסמה", "סיסמת גישה", "קוד גישה", "passphrase", "access code"],
        "required": False,
        "hint": "Short answer. Only needed if FORM_CHECKER_PASSPHRASE is set on "
                "the service — see the cost guardrails in main.py.",
    },
}

# The rubric dropdown option that means "the rubric is one of the files I
# uploaded" rather than one of the bundled ones.
RUBRIC_FROM_UPLOAD = "מחוון שצירפתי בקבצים"

# ─── grading with no task in hand ───────────────────────────────────────────
# The form asks for the assignment, but "Assignment description" is optional and
# a demo submission often arrives without it. There is no folder here to find an
# exam paper in, so this notice fires more often in this product than in
# form-checker. The work is graded anyway — language, vocabulary and mechanics
# do not depend on knowing the topic.
#
# What must not happen is a silent guess. Without this notice the model infers a
# topic from the composition itself and then scores the composition as perfectly
# on-topic against its own inference, which produces a confident, unearned
# Content mark. Saying so explicitly is what converts an invisible wrong answer
# into a visible "cannot be determined" — and the teacher, who does know the
# task, can supply the judgement we cannot.
NO_TASK_NOTICE = """\
NOTE — THE ASSIGNMENT WAS NOT SUPPLIED.
The teacher did not describe the task on the form. You are therefore grading
without knowing what the student was asked to write. Accordingly:

• Do NOT judge, reward or penalise topic relevance. You cannot know the topic,
  and you must NOT infer it from the composition and then score the composition
  against your own inference.
• Score CONTENT AND ORGANIZATION on what can still be judged without the task:
  development of ideas, coherence, structure, introduction and conclusion.
• In the CONTENT AND ORGANIZATION feedback, state plainly, as the first
  sentence: "Cannot determine whether the student answered the question — the
  assignment was not supplied."
• Repeat that limitation in the overall feedback, so the teacher knows the mark
  is incomplete rather than final.
• Every other criterion — vocabulary, language use, mechanics — is unaffected by
  the missing task and should be scored normally."""

# ─── Bagrut module codes → bundled rubric ───────────────────────────────────
# Avishai's form labels its dropdown options the way the Ministry names the
# exams — "Module G 16582" — and neither half of that matches a bundled rubric's
# name or id. Matching on the *code* is what makes the mapping unambiguous:
# 16582 is Module G and nothing else, whereas the letter alone is one character
# that appears all over a label.
#
# This is not cosmetic. Before this map existed every one of the four options
# fell through resolve_rubric_choice to core.DEFAULT_RUBRIC_ID, an ESL 9th-grade
# CEFR rubric — so a Module G composition worth 40 points was graded out of a
# completely different scale, with only a log warning to say so.
_MODULE_RUBRICS: dict[str, str] = {
    "c": "moe-writing-module-c",
    "d": "moe-writing-module-d",
    "f": "moe-writing-module-f",
    "g": "moe-writing-module-g",
}

# The leading zero is optional in practice — the Ministry writes 016382 on some
# sheets and 16382 on others, and teachers copy whichever they saw.
_MODULE_CODES: dict[str, str] = {
    "16382": "c",
    "16484": "d",
    "16584": "f",
    "16582": "g",
}

_MODULE_CODE_RE = re.compile(r"\b0?(\d{5})\b")
_MODULE_LETTER_RE = re.compile(r"(?:\bmodule|מודול)\s+([cdfg])\b")


def resolve_module(label: str) -> str | None:
    """Bagrut module letter ('c'/'d'/'f'/'g') for a rubric dropdown label.

    Code first, letter second: the code is unique, the letter is not — a label
    reading "Module G 16582" would also match a naive search for "d" inside the
    word "Module". Returns None when the label names no module at all, which is
    the normal case for a form that offers our own rubrics instead.
    """
    if not label:
        return None
    m = _MODULE_CODE_RE.search(str(label))
    if m and m.group(1) in _MODULE_CODES:
        return _MODULE_CODES[m.group(1)]
    m = _MODULE_LETTER_RE.search(normalise(label))
    if m:
        return m.group(1)
    return None

# Exercise-language dropdown labels → core.LANGS keys. Accepts the Hebrew name,
# the native name and the ISO code, since all three show up in practice.
_LANG_LABELS: dict[str, str] = {}
for _code, _meta in core.LANGS.items():
    _LANG_LABELS[_code] = _code
    _LANG_LABELS[_meta["name_he"]] = _code
    _LANG_LABELS[_meta["name_native"]] = _code
_LANG_LABELS.update({"english": "en", "hebrew": "he", "arabic": "ar"})


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


# form-checker's extract_folder_id lives here in its copy. It is gone on
# purpose: this product never resolves a folder, and a helper that only the
# deleted field used is the kind of thing that gets re-wired by mistake.


def resolve_rubric_choice(label: str) -> tuple[str | None, bool]:
    """(bundled rubric id, use_uploaded_rubric).

    (None, False) means "no explicit choice" — the caller falls back to
    core.DEFAULT_RUBRIC_ID. (None, True) means the teacher said the rubric is
    among the files they uploaded, which routes those files through
    core.classify_document() instead."""
    norm = normalise(label)
    if not norm:
        return None, False
    if normalise(RUBRIC_FROM_UPLOAD) in norm or "צירפתי" in label or "attached" in norm:
        return None, True

    # Before any name matching: a Bagrut module code names exactly one rubric.
    module = resolve_module(label)
    if module:
        rubric_id = _MODULE_RUBRICS[module]
        if core.load_rubric(rubric_id):
            return rubric_id, False
        log.error("rubric choice %r is Bagrut module %s, but %s is not bundled — "
                  "falling back to %s, which grades on a different scale. Add the "
                  "rubric JSON to haskala_grading/rubrics/.",
                  label, module.upper(), rubric_id, core.DEFAULT_RUBRIC_ID)
        return None, False

    for r in core.list_rubrics():
        if normalise(r["name"]) == norm or normalise(r["id"]) == norm:
            return r["id"], False
    # Substring, for a dropdown label carrying a parenthesised note.
    for r in core.list_rubrics():
        if normalise(r["name"]) in norm or normalise(r["id"]) in norm:
            return r["id"], False
    log.warning("rubric choice %r matches no bundled rubric — falling back to %s. "
                "Fix the form's dropdown options (see --template).",
                label, core.DEFAULT_RUBRIC_ID)
    return None, False


def resolve_lang(label: str) -> str:
    """Exercise-language dropdown label → core.LANGS key, default English."""
    norm = normalise(label)
    if not norm:
        return core.DEFAULT_EXERCISE_LANG
    for key, code in _LANG_LABELS.items():
        if normalise(key) and normalise(key) in norm:
            return code
    log.warning("language %r not recognised — defaulting to %s",
                label, core.DEFAULT_EXERCISE_LANG)
    return core.DEFAULT_EXERCISE_LANG


def resolve_model_choice(label: str) -> str:
    """Model dropdown label → core.MODELS key, default core.DEFAULT_MODEL.

    Accepts the key, an alias from core._MODEL_ALIASES, or the human label the
    form shows ("Claude (Sonnet 5)")."""
    norm = normalise(label)
    if not norm:
        return core.DEFAULT_MODEL
    if norm in core.MODELS:
        return norm
    for key, human in core.MODELS.items():
        if normalise(key) in norm or normalise(human) in norm:
            return key
    for alias, key in core._MODEL_ALIASES.items():
        if normalise(alias) in norm:
            return key
    log.warning("model %r not recognised — falling back to %s", label, core.DEFAULT_MODEL)
    return core.DEFAULT_MODEL


def parse_row(row: list[str], colmap: dict[str, int]) -> dict:
    """One responses row → the parameters the grading pipeline takes.

    Short rows are normal: Sheets truncates trailing empty cells, so a response
    that skipped the last few optional questions comes back shorter than the
    header. Every read is bounds-checked for that reason."""
    def cell(key: str) -> str:
        idx = colmap.get(key)
        if idx is None or idx >= len(row):
            return ""
        return (row[idx] or "").strip()

    rubric_id, rubric_from_upload = resolve_rubric_choice(cell("rubric"))

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
        "module": resolve_module(cell("rubric")),
        "school": cell("school"),
        "grade_level": cell("grade_level"),
        "instructions": cell("instructions"),
        "passphrase": cell("passphrase"),
        "rubric_id": rubric_id,
        "rubric_from_upload": rubric_from_upload,
        "rubric_label": cell("rubric"),
        "exercise_lang": resolve_lang(cell("exercise_lang")),
        "model_key": resolve_model_choice(cell("model")),
        "file_ids": extract_file_ids(cell("files")),
    }


def build_question(params: dict) -> str:
    """The 'question' passed to core.check_pages: the teacher's own instructions
    plus the context they gave about the class.

    The context matters to the grade — "כיתה ז" and "כיתה יב" deserve very
    different expectations of the same paragraph, and the rubric alone does not
    say which one this is. It is appended rather than prepended so the teacher's
    own wording stays the first thing the model reads."""
    parts = []
    if params.get("instructions"):
        parts.append(params["instructions"])
    else:
        parts.append(NO_TASK_NOTICE)
    context = []
    if params.get("module"):
        # The module is the single most useful thing we know about the
        # expectation level: a Module C task and a Module G task are the same
        # genre written by students two Bagrut levels apart.
        context.append(f"בחינת בגרות באנגלית, מודול {params['module'].upper()}")
    if params.get("grade_level"):
        context.append(f"שכבת גיל: {params['grade_level']}")
    if params.get("school"):
        context.append(f"בית ספר: {params['school']}")
    if context:
        parts.append("הקשר כיתתי (לא חלק מהוראות המשימה):\n" + "\n".join(context))
    if params.get("comments"):
        # Kept last and clearly fenced. It is free text a teacher was *required*
        # to fill in, so it is as likely to say "thanks!" as anything useful —
        # labelling it stops the model reading a pleasantry as task instructions.
        parts.append("הערות מהמורה (לא חלק מהוראות המשימה):\n" + params["comments"])
    return "\n\n".join(parts)


# ─── the form-building aid ──────────────────────────────────────────────────

def print_form_template() -> None:
    """Print the question titles and dropdown options the form should use.

    This is the intended way to build the form: run it, hand the output over,
    and the schema and the form agree by construction. Reverse-engineering the
    mapping from a form that already exists is the path that produces silent
    mismatches.
    """
    print("Google Form — questions form-checker knows how to read")
    print("=" * 60)
    print("\nHeaders are matched on keywords, so a question may be reworded freely")
    print("as long as it still contains one of the keywords listed.\n")
    for key, spec in FIELDS.items():
        req = "REQUIRED" if spec["required"] else "optional"
        print(f"── {key}  ({req})")
        print(f"   suggested title : {spec['aliases'][0]}")
        print(f"   keywords        : {', '.join(spec['aliases'][:5])}")
        print(f"   note            : {spec['hint']}")
        print()

    print("=" * 60)
    print("Dropdown options for the rubric question.\n")
    print("Bagrut modules — matched on the exam CODE, so the wording around it")
    print("is free ('Module G 16582', 'מודול G — 16582' and '16582' all work):\n")
    for letter, rubric_id in _MODULE_RUBRICS.items():
        code = next(c for c, l in _MODULE_CODES.items() if l == letter)
        rubric = core.load_rubric(rubric_id)
        status = "" if rubric else "   ← NOT BUNDLED, would fall back to the default"
        print(f"   • Module {letter.upper()} {code}{status}")
    print("\nOr any of our own rubrics, matched on the name:\n")
    for r in core.list_rubrics():
        print(f"   • {r['name']}")
    print(f"   • {RUBRIC_FROM_UPLOAD}")
    print(f"\n(no answer → the bundled default, {core.DEFAULT_RUBRIC_ID})\n")

    print("Dropdown options for the language question:\n")
    for code, meta in core.LANGS.items():
        print(f"   • {meta['name_he']}  ({meta['name_native']}, {code})")
    print(f"\n(no answer → {core.DEFAULT_EXERCISE_LANG})\n")

    print("Dropdown options for the model question, if you expose it at all:\n")
    for key, human in core.MODELS.items():
        mark = "  ← default" if key == core.DEFAULT_MODEL else ""
        print(f"   • {human}{mark}")
    print()


if __name__ == "__main__":
    import sys
    if "--template" in sys.argv:
        print_form_template()
    else:
        print("usage: python form_schema.py --template")
