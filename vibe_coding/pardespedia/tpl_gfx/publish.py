#!/usr/bin/env python3
"""
מעלה את הגרסאות הגרפיות של התבניות.

  python3 tpl_gfx/publish.py sandbox            # לדפי המשנה של הבוט + דף ההדגמה
  python3 tpl_gfx/publish.py live NAME [NAME..] # לתבניות עצמן, רק אלה שנקראו בשמן

העלאה ל-live מסרבת אם התבנית החיה השתנתה מאז שנשמר העותק ב-tpl_gfx/baseline/
(כלומר מישהו ערך אותה בינתיים), כדי לא לדרוס עבודה של עורך אחר.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build
from wiki_client import WikiClient

SUMMARY_SB = "ארגז חול: הצעה לאלמנטים גרפיים בתבניות (בוטי)"


def main():
    mode = sys.argv[1]
    client = WikiClient()
    client.login()
    if mode == "sandbox":
        for name, text in build.TEMPLATES.items():
            client.edit_page(build.SANDBOX + name, build.render(text, "sandbox"), summary=SUMMARY_SB)
        demo = open(os.path.join(HERE, "demo.wiki")).read().replace("{{@@", "{{" + build.SANDBOX)
        client.edit_page(build.SANDBOX.rstrip("/"), demo, summary=SUMMARY_SB)
        return
    if mode == "live":
        names = sys.argv[2:]
        if not names:
            sys.exit("לציין שמות תבניות במפורש")
        for name in names:
            text = build.render(build.TEMPLATES[name], "live")
            title = "תבנית:" + name
            page = client.get_page(title)
            base = os.path.join(HERE, "baseline", name.replace("/", "__") + ".wiki")
            if page["exists"]:
                if not os.path.exists(base):
                    sys.exit(f"אין baseline ל-{name}; לא דורסים תבנית קיימת בלי עותק להשוואה")
                if open(base).read().strip() != page["wikitext"].strip():
                    sys.exit(f"{name}: התבנית החיה השתנתה מאז ה-baseline. לעצור ולמזג ביד.")
            if page["exists"] and "<noinclude>" not in text and "<noinclude>" in page["wikitext"]:
                # שינינו רק את החלק המוצג; התיעוד שכתב העורך נשאר כמו שהוא
                text += page["wikitext"][page["wikitext"].index("<noinclude>"):]
            client.edit_page(title, text, summary="אלמנטים גרפיים בתבנית (בוטי)")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
