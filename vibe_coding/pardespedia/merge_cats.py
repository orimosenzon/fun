#!/usr/bin/env python3
"""Merge duplicate wiki categories: retag members, leave a redirect behind.

The bot has no delete right, so a merged-away category becomes a redirect to
its survivor — which is the better outcome anyway: an editor who types the old
name still lands in the right place.

The community cluster had grown four near-synonyms ("קהילות", "פעילות קהילתית",
"יוזמה קהילתית", "שרותי קהילה"); "יום השכנים" sat in two of them at once, which
is what a duplicate category looks like from the inside. The surviving scheme is
three axes that cannot overlap:

    קהילות            — מי:    groups of people
    יוזמה קהילתית     — מה:    initiatives, projects, activities
    מרחבים קהילתיים   — איפה:  physical spaces

"פעילות קהילתית" cut across all three, so its members are sorted by hand below
rather than swept into one bucket: a place already carrying "מרחבים קהילתיים"
just drops the redundant tag, while a genuine initiative moves to
"יוזמה קהילתית".

Follows the project rule for every page touched: get the current wikitext,
make one targeted substitution, send the full page back. Never reconstruct.
"""
import re
import sys

from wiki_client import WikiClient

API = "https://pardespedia.info/api.php"

# category -> survivor. Every member is retagged.
MERGES = [
    ("אנשים הקבורים בכרכור", "אישים הקבורים בפרדס חנה-כרכור"),
]

# "פעילות קהילתית" is dissolved, not merged: each member goes where it belongs.
DISSOLVE = "פעילות קהילתית"
DISSOLVE_TARGET = "יוזמה קהילתית"       # where the category page redirects to
# members that are already filed correctly elsewhere — just drop the extra tag
DISSOLVE_DROP = {
    "בית יובל",                # מרחבים קהילתיים — a place
    "הפרדס - מרכז צעירים",     # מרחבים קהילתיים — a place
    "מרכז קהילתי צמרת",        # מרחבים קהילתיים — a place
    "יום השכנים",              # already in יוזמה קהילתית — the duplicate itself
    "דורון קדמיאל",            # a person: אישים + אקטיביזם already say it
}


def members(c, cat):
    r = c.session.get(API, params={
        "action": "query", "list": "categorymembers",
        "cmtitle": f"קטגוריה:{cat}", "cmlimit": "500", "format": "json"}).json()
    return [m["title"] for m in r["query"]["categorymembers"]]


def _tag(name):
    return re.compile(r"\[\[\s*(?:קטגוריה|Category)\s*:\s*" + re.escape(name) +
                      r"\s*(\|[^\]]*)?\]\]")


def retag(text, old, new):
    """Swap one category tag, preserving any sort key. Returns (text, changed).

    When `new` is None — or already present on the page — the old tag is
    dropped instead, so a merge never leaves the same category twice.
    """
    pat = _tag(old)
    if not pat.search(text):
        return text, False
    if new is None or _tag(new).search(text):
        return re.sub(r"\n{3,}", "\n\n", pat.sub("", text)), True
    return pat.sub(lambda m: f"[[קטגוריה:{new}{m.group(1) or ''}]]", text), True


def apply(c, title, old, new, summary, dry):
    page = c.get_page(title)
    if not page["exists"]:
        print(f"  ! missing: {title}")
        return
    text, changed = retag(page["wikitext"], old, new)
    if not changed:
        print(f"  - no tag found in {title}")
        return
    action = "drop" if new is None else f"→ {new}"
    print(f"  {'would ' if dry else ''}{action}: {title}")
    if not dry:
        c.edit_page(title, text, summary=summary)


def redirect(c, cat, target, dry):
    print(f"  {'would ' if dry else ''}redirect: קטגוריה:{cat} → {target}")
    if not dry:
        c.edit_page(f"קטגוריה:{cat}", f"#הפניה [[:קטגוריה:{target}]]\n",
                    summary=f"קטגוריה מאוחדת אל [[:קטגוריה:{target}]]")


def main():
    dry = "--dry-run" in sys.argv
    c = WikiClient()
    c.login()

    for old, new in MERGES:
        titles = members(c, old)
        print(f"\n=== {old} → {new}  ({len(titles)} members)")
        summary = f"איחוד קטגוריות: [[:קטגוריה:{old}]] ← [[:קטגוריה:{new}]]"
        for t in titles:
            apply(c, t, old, new, summary, dry)
        redirect(c, old, new, dry)

    titles = members(c, DISSOLVE)
    print(f"\n=== {DISSOLVE} — פירוק ({len(titles)} members)")
    summary = (f"איחוד טקסונומיה: [[:קטגוריה:{DISSOLVE}]] פורקה לטובת "
               f"[[:קטגוריה:{DISSOLVE_TARGET}]] / [[:קטגוריה:מרחבים קהילתיים]]")
    for t in titles:
        target = None if t.split(":")[-1] in DISSOLVE_DROP else DISSOLVE_TARGET
        apply(c, t, DISSOLVE, target, summary, dry)
    redirect(c, DISSOLVE, DISSOLVE_TARGET, dry)

    # שרותי קהילה was merged into פעילות קהילתית earlier this session; now that
    # פעילות קהילתית is itself a redirect, repoint it — MediaWiki does not
    # follow a double redirect.
    redirect(c, "שרותי קהילה", DISSOLVE_TARGET, dry)


if __name__ == "__main__":
    main()
