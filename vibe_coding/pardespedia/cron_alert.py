"""Tell a human when an unattended pardespedia cron job breaks.

The events board failed four nights in a row (9/11 to 12/11/2026) and nobody
knew: the error sat in events_cron.log and the main page quietly kept an old
board. This posts to the bot's talk page instead, where Ori and the sysops look.

    python3 cron_alert.py fail <job> <logfile>   # after a failed run
    python3 cron_alert.py ok   <job>             # after a successful run

One post per failure streak, not one per night: the first failure opens a
section, the following failures stay quiet, and the first success after them
adds a short "back to normal" line under the same section. The streak lives
in cron_alert_state.json.

The edit summary starts with "התראת cron", which is in HOUSEKEEPING_PREFIXES
of check_all_bot_talk.py, so this post never hides a pending human question.
"""

import datetime as dt
import json
import os
import re
import sys

from wiki_client import WikiClient

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "cron_alert_state.json")
TALK = "שיחת משתמש:אורי מוסנזון בוט"
SUMMARY = "התראת cron"
TAIL_LINES = 12

JOB_NAMES = {
    "events": "לוח אירועי התרבות בעמוד הראשי",
    "about": "הגרפים וטבלת הנתונים בערך פרדספדיה",
}


def load_state():
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state):
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)


def last_run_tail(logfile):
    """The tail of the last run in the log (runs start with '===== ')."""
    try:
        with open(logfile, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return "(הלוג לא נמצא)"
    run = text.rsplit("\n===== ", 1)[-1]
    lines = [l for l in run.splitlines()
             if l.strip() and "outside the vocabulary" not in l]
    tail = "\n".join(lines[-TAIL_LINES:])
    # no live external links in the post: the spam filter would reject them
    return re.sub(r"https?://", "", tail)


def client():
    c = WikiClient()
    c.login()
    return c


def post(c, section_title, text):
    """Append a new section (MediaWiki section=new: nothing existing is touched)."""
    r = c.session.post("https://pardespedia.info/api.php", data={
        "action": "edit", "title": TALK, "section": "new",
        "sectiontitle": section_title, "text": text,
        "summary": f"{SUMMARY}: {section_title}",
        "token": c._csrf_token(), "format": "json"}).json()
    if "error" in r:
        raise RuntimeError(r["error"].get("info"))


def append_to_section(c, section_title, text):
    """Add a line under an existing section, found by its heading."""
    page = c.get_page(TALK)
    content = page.get("wikitext") or ""
    head = f"== {section_title} =="
    i = content.find(head)
    if i == -1:
        post(c, section_title + " (המשך)", text)
        return
    j = content.find("\n== ", i + len(head))
    end = len(content) if j == -1 else j
    new = content[:end].rstrip() + "\n\n" + text + "\n" + content[end:]
    c.edit_page_spam_safe(TALK, new, summary=f"{SUMMARY}: {section_title}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("fail", "ok"):
        print(__doc__)
        sys.exit(2)
    mode, job = sys.argv[1], sys.argv[2]
    state = load_state()
    cur = state.get(job)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    name = JOB_NAMES.get(job, job)

    if mode == "fail":
        if cur:  # streak already reported
            cur["count"] += 1
            save_state(state)
            print(f"cron_alert: {job} still failing ({cur['count']}), already reported")
            return
        title = f"תקלה בעדכון האוטומטי: {name} ({now[:10]})"
        tail = last_run_tail(sys.argv[3] if len(sys.argv) > 3 else "")
        text = (f"העדכון הלילי של '''{name}''' נכשל ב-{now}, והדף נשאר בגרסתו הקודמת "
                f"עד שזה יתוקן. לא אכתוב כאן שוב בכל לילה; כשהעדכון יחזור לעבוד "
                f"תתווסף כאן שורה.\n\nסוף הלוג (<code>{job}_cron.log</code>):\n"
                f"<pre>{tail}</pre>\n~~~~")
        post(client(), title, text)
        state[job] = {"since": now, "count": 1, "title": title}
        save_state(state)
        print(f"cron_alert: reported failure of {job}")
    else:
        if not cur:
            return
        text = (f":{now}: העדכון חזר לעבוד, אחרי {cur['count']} הרצות שנכשלו "
                f"מאז {cur['since']}. ~~~~")
        append_to_section(client(), cur["title"], text)
        del state[job]
        save_state(state)
        print(f"cron_alert: reported recovery of {job}")


if __name__ == "__main__":
    main()
