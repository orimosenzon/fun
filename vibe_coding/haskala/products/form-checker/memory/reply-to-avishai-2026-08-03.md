# Draft reply to Avishai — 2026-08-03

Hi Avishai,

Your flow is right, and one detail in it is doing more work than it looks like.

**Why the Classroom assignment folder specifically is the correct choice.** When
a student submits, Classroom transfers ownership of the file to the teacher —
students drop to viewers until the work is returned. That is what makes the
sharing work at all: the teacher actually owns the contents of that folder, so
they can grant us access to them. If we asked teachers to share an ordinary
Drive folder of student-owned files, sharing the folder would not reliably give
us the files inside it. So your version is not just tidier, it is the one that
functions.

**One correction.** You wrote that checking begins "via the Google Classroom
API". It can't — the Classroom API is scoped to course membership, so if we
can't be added to the class across domains, we have no API access to it either.
The checking has to be pure Drive, which is what we have now built. Nothing is
lost: our existing Classroom-API product keeps serving same-domain schools like
Shamir, and this new one is the cross-domain path. Flagging it in case other
parts of your plan assume Classroom API access.

**The blocker to check before we go further.** Cross-domain Drive sharing is an
admin policy on the *teacher's* side, not ours:

    Admin console → Drive and Docs → Sharing settings
                  → Sharing outside of organization → Allowlisted domains

Many education domains restrict this, and a teacher cannot change it. So "a user
from another domain can get Editor access to a drive" is true only where the
מחוז permits it.

**Can you test this in two minutes?** From your Shamir account, share any folder
with an address at bdika.net and see whether Google lets you. If it works, your
domain is permissive and we know the flow is viable there. If it is blocked, we
know the real ask is "your IT allowlists bdika.net once" — which we would much
rather discover now than in front of a school.

**A timing trap for teachers.** Clicking **Return** in Classroom hands ownership
back to the students. If a teacher returns the work before our check has run,
the files revert and our access can break mid-batch. The rule is: submit the
form → get the reports → *then* return.

**Two things we need from you for the form:**

1. **Link the responses spreadsheet** (Form → Responses → the Sheets icon) and
   send us its id — the long token in the sheet's URL between `/d/` and `/edit`.
   Nothing can run until we have it.
2. **Add the sharing instruction to the form.** Right now the form asks for a
   folder link but never says who to share it with, or that Editor access is
   required. A teacher who shares read-only gets nothing. Suggested text for the
   "Link to Folder" question:

   > Before submitting, share the folder with `ori@bdika.net` with **Editor**
   > access — not "Viewer". Editor is required because we write the checked
   > results back into your folder.

   `ori@bdika.net` is the address you already shared with, so nothing needs to
   move for now. We will switch to a dedicated service account later — a
   personal address on a form that circulates is a bad idea long term — and when
   we do, folders will need resharing. Better before the link spreads than
   after.

**The question you're adding is the most valuable change on this list.** Call it
"The task" or "What were the students asked to write?" — we match all the
obvious phrasings — and make it a paragraph, not required.

Without it we cannot say whether a student answered what was asked, and that is
a whole criterion: "fully on topic" is 8 of the 40 points in Module G. When it
is missing we still check the work — vocabulary, language use and mechanics do
not depend on knowing the topic — but the Content section of the report says
plainly *"Cannot determine whether the student answered the question — the
assignment was not supplied"*, and the model is explicitly instructed **not** to
guess the topic from the composition and then score the composition as being on
that topic. A confident wrong answer would be worse than an honest gap.

Equally good alternative, if teachers would rather not retype it: **put the exam
paper in the folder.** We detect typed documents, do not grade them as student
work, and read the task out of them.

One small suggestion: **"Comments and requests" is currently required**, so
every teacher must type something even when they have nothing to say. Worth
making it optional.

**While we are testing, we check only 3 works per folder.** Not a rejection — a
teacher who shares a class of thirty gets three reports and an explanation. To
get the next three, they resubmit the form with the same folder; already-checked
work is never rechecked, so each submission advances by three. We will raise
this once we have read the first reports and know the real cost and quality.
Resubmitting the same folder is always free and safe.

Ori
