#!/usr/bin/env python3
"""
check_balances.py — בודק יתרת/שימוש בקונסולות החיוב של ספקי ה-LLM, ע"י
הנעת דפדפן Chrome אמיתי שמחובר לחשבונות שלך (פרופיל Playwright ייעודי).

זרימה:
  1. פעם אחת:   ./venv/bin/python check_balances.py --login
     נפתח דפדפן; התחבר לכל קונסולה (כולל 2FA), ואז Enter בטרמינל. העוגיות נשמרות.
  2. שוטף:      ./venv/bin/python check_balances.py
     נכנס לכל דף חיוב, מצלם ל-shots/, ושופך את הטקסט הגלוי (לחידוד selectors).
  3. headless:  הוסף --headless כשהחילוץ כבר יציב.

הערה: רק ל-Anthropic יש "יתרה" אמיתית (prepaid credits). ל-Groq יש קרדיט גם.
ל-Gemini/Azure אין מושג יתרה — חיוב ענן מצטבר; נכללים רק כדמפ-עזר.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(HERE, "pw-profile")   # עוגיות/סשן — לא לקומיש!
SHOTS_DIR = os.path.join(HERE, "shots")

# כל ספק: (מפתח, שם תצוגה, URL של דף החיוב/שימוש)
PROVIDERS = [
    ("anthropic", "Anthropic (Claude) — prepaid credits",
     "https://console.anthropic.com/settings/billing"),
    ("groq", "Groq — credits",
     "https://console.groq.com/settings/billing"),
    ("gemini", "Google AI Studio (Gemini) — אין יתרה, שימוש בלבד",
     "https://aistudio.google.com/usage"),
    ("azure", "Azure OpenAI — חיוב מנוי (Cost Management)",
     "https://portal.azure.com/#view/Microsoft_Azure_CostManagement"),
]


def launch(p, headless: bool):
    """Chrome אמיתי עם פרופיל מתמיד; נופל ל-Chromium מובנה אם chrome לא נמצא."""
    common = dict(user_data_dir=PROFILE_DIR, headless=headless,
                  viewport={"width": 1400, "height": 900})
    try:
        return p.chromium.launch_persistent_context(channel="chrome", **common)
    except Exception as e:
        print(f"(channel=chrome נכשל: {e} — נופל ל-Chromium מובנה)", file=sys.stderr)
        return p.chromium.launch_persistent_context(**common)


def do_login() -> None:
    with sync_playwright() as p:
        ctx = launch(p, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        for _, name, url in PROVIDERS:
            print(f"→ פותח: {name}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                print(f"  (טעינה נכשלה: {e})")
            input("    התחבר/אמת שהדף מוצג, ואז Enter כדי לעבור לבא… ")
        print("נשמר. מעכשיו הרץ בלי --login.")
        ctx.close()


def do_check(headless: bool) -> None:
    os.makedirs(SHOTS_DIR, exist_ok=True)
    with sync_playwright() as p:
        ctx = launch(p, headless=headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        for key, name, url in PROVIDERS:
            print(f"\n{'='*70}\n{name}\n{url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(2500)  # זמן לרינדור client-side
            except Exception as e:
                print(f"  שגיאת ניווט: {e}")
                continue
            shot = os.path.join(SHOTS_DIR, f"{key}.png")
            page.screenshot(path=shot, full_page=True)
            body = page.inner_text("body") if page.query_selector("body") else ""
            # שורות לא-ריקות בלבד, חתוך כדי לא להציף
            lines = [l.strip() for l in body.splitlines() if l.strip()]
            snippet = "\n".join(lines[:60])
            print(f"  צילום: {shot}")
            print("  --- טקסט גלוי (60 שורות ראשונות) ---")
            print(textwrap.indent(snippet, "  | "))
        ctx.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true",
                    help="מצב התחברות חד-פעמי (דפדפן גלוי)")
    ap.add_argument("--headless", action="store_true",
                    help="הרצה ללא חלון (רק כשהחילוץ יציב)")
    args = ap.parse_args()
    if args.login:
        do_login()
    else:
        do_check(headless=args.headless)


if __name__ == "__main__":
    main()
