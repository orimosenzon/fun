"""Monthly refresh of the [[פרדספדיה]] article: its three charts and the
"פרדספדיה במספרים" table (cron, 1st of the month; asked by Ori 12/11/2026,
after the charts had stayed frozen at August 2026).

    python3 update_about_page.py            # regenerate, upload, edit
    python3 update_about_page.py --dry-run  # regenerate and print, touch nothing

What it touches, and only that:
  * the three chart files, uploaded as a new version of the same file names,
    so the article and its captions do not change;
  * inside the section "== פרדספדיה במספרים ==", the line "נכון ל..." and the
    wikitable under it. The rest of the article is prose people wrote; a
    section that cannot be found is reported, not rebuilt.
"""

import datetime as dt
import os
import re
import sys

import stats_charts as sc
from wiki_client import API_URL, WikiClient

ARTICLE = "פרדספדיה"
OUT = os.path.expanduser("~/.cache/pardespedia_about")
FILES = {  # local chart -> file on the wiki
    "pardespedia_growth.png": "גרף מספר הערכים בפרדספדיה.png",
    "pardespedia_edits.png": "גרף קצב העריכה בפרדספדיה.png",
    "pardespedia_contributors.png": "גרף תורמים מובילים בפרדספדיה.png",
}
HE_MONTHS = ["", "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי",
             "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
SECTION = "== פרדספדיה במספרים =="


def site_stats(client):
    s = client.session.get(API_URL, params={
        "action": "query", "meta": "siteinfo", "siprop": "statistics",
        "format": "json"}).json()["query"]["statistics"]
    sysops = client.session.get(API_URL, params={
        "action": "query", "list": "allusers", "augroup": "sysop", "aulimit": 500,
        "format": "json"}).json()["query"]["allusers"]
    # the bot is a sysop since 4/9/2026; the table counts people
    return s, len([u for u in sysops if u["name"] != sc.BOT_USER])


def table(stats, sysops, editors, human, bot):
    total = human + bot
    rows = [
        ("ערכים במרחב השם הראשי", stats["articles"]),
        ("דפים באתר בסך הכול (כולל קטגוריות, תבניות ודפי שיחה)", stats["pages"]),
        ("קבצי תמונה", stats["images"]),
        ("עריכות מאז ההקמה", stats["edits"]),
        ("חשבונות רשומים", stats["users"]),
        ("חשבונות שביצעו לפחות עריכה אחת", editors),
        ("עורכים פעילים בחודש האחרון", stats["activeusers"]),
        ("מפעילי מערכת", sysops),
    ]
    out = ['{| class="wikitable"', "! נתון !! כמות"]
    for name, val in rows:
        out += ["|-", f"| {name} || {val:,}"]
    out += ["|-", f"| חלוקת העריכות || {round(100 * human / total)}% בני אדם, "
            f"{round(100 * bot / total)}% הבוט", "|}"]
    return "\n".join(out)


def splice(text, new_table, today):
    i = text.find(SECTION)
    if i == -1:
        raise RuntimeError(f"section not found: {SECTION}")
    j = text.find("\n==", i + len(SECTION))
    end = len(text) if j == -1 else j
    sec = text[i:end]
    sec2 = re.sub(r"נכון ל[^\n:]*:", f"נכון ל{HE_MONTHS[today.month]} {today.year}:", sec, count=1)
    sec2, n = re.subn(r"\{\|.*?\n\|\}", lambda m: new_table, sec2, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("stats table not found inside the section")
    return text[:i] + sec2 + text[end:]


def upload_new_version(client, local, title):
    with open(local, "rb") as fh:
        r = client.session.post(API_URL, data={
            "action": "upload", "filename": title, "ignorewarnings": "1",
            "comment": "עדכון חודשי אוטומטי של הגרף (update_about_page.py)",
            "token": client._csrf_token(), "format": "json"},
            files={"file": (title, fh, "image/png")}).json()
    res = r.get("upload", {}).get("result")
    if res != "Success":
        # identical bytes come back as a 'duplicate'/'exists' warning; not an error
        print(f"  {title}: {r.get('upload', {}).get('warnings') or r.get('error')}")
    else:
        print(f"  uploaded new version: {title}")


def main():
    dry = "--dry-run" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    sc.setup_font()
    client = WikiClient()
    if not dry:
        client.login()

    creations = sc.fetch_creations(client)
    human, bot, per_user = sc.fetch_revisions(client)
    sc.chart_growth(creations, f"{OUT}/pardespedia_growth.png")
    sc.chart_edits(human, bot, f"{OUT}/pardespedia_edits.png")
    sc.chart_contributors(per_user, f"{OUT}/pardespedia_contributors.png")

    stats, sysops = site_stats(client)
    editors = len([u for u in per_user if u != "?"])
    today = dt.date.today()
    new_table = table(stats, sysops, editors, sum(human.values()), sum(bot.values()))
    text = client.get_page(ARTICLE)["wikitext"]
    new = splice(text, new_table, today)

    if dry:
        print(new_table)
        print(f"charts in {OUT}; article unchanged (dry run)")
        return
    for local, title in FILES.items():
        upload_new_version(client, f"{OUT}/{local}", title)
    if new != text:
        client.edit_page_spam_safe(ARTICLE, new,
                                   summary="עדכון חודשי אוטומטי: טבלת הנתונים והגרפים")
    else:
        print("article table unchanged")


if __name__ == "__main__":
    main()
