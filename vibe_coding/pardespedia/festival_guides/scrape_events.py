#!/usr/bin/env python3
"""Scrape the אמנות במושבה 2026 events programme (the ~94 WP posts) into JSON.

Each event post carries the same badge strip: day, start, end, venue, price,
how to register, phone. Output: pa/events.json
"""
import concurrent.futures as cf
import html
import json
import os
import re
import sys

from scrape_pardesart import get, sitemap, txt

OUT = os.path.dirname(os.path.abspath(__file__)) + "/pa"

DAYS = ("חמישי", "שישי", "שבת")
TIME = re.compile(r"^\d{1,2}:\d{2}\s*-?\s*$|^\d{1,2}:\d{2}$")
REG_WORDS = ("הרשמה", "לא נדרשת", "הכניסה חופשית", "קישור להרשמה")
PRICE = re.compile(r"(ללא עלות|חופשי|\d+\s*(?:ש[״\"']ח|₪))")


def parse_event(url):
    t = get(url)
    rec = {"url": url}

    m = re.search(r"<h1[^>]*>(.*?)</h1>", t, re.S)
    rec["artist"] = txt(m.group(1)) if m else ""
    # everything we want sits after the title; before it is nav + breadcrumb
    body = t[m.end():] if m else t

    # The page's JSON-LD is the reliable source for the two taxonomies:
    # articleSection is the event type, keywords is the festival day.
    m = re.search(r'"articleSection":\[(.*?)\]', t)
    rec["kind"] = html.unescape(m.group(1).strip('"')) if m else ""
    m = re.search(r'"keywords":\[(.*?)\]', t)
    kw = html.unescape(m.group(1)).replace('"', "").split(",") if m else []

    # The theme leaves most <p> tags unclosed, so a lazy <p>...</p> match would
    # swallow the whole page. Bound each paragraph at the next <p or </p>.
    paras = []
    for m in re.finditer(r"<p[^>]*>", body):
        rest = body[m.end():]
        stop = min([x for x in (rest.find("<p"), rest.find("</p>")) if x != -1] or [len(rest)])
        s = txt(rest[:stop])
        if len(s) > 25 and "cookie" not in s.lower() \
                and "פסטיבל אמנות עיצוב וקראפט" not in s \
                and "מפה דיגיטלית" not in s and "מס' אמן" not in s:
            paras.append(s)
    rec["description"] = paras[0] if paras else ""

    # badge strip
    spans, seen = [], set()
    for m in re.finditer(r"<span[^>]*>([^<]{1,120})</span>", body):
        s = txt(m.group(1))
        if s and s not in seen:
            seen.add(s)
            spans.append(s)

    day = times = venue = price = reg = phone = ""
    for d in kw:
        if d.strip() in DAYS:
            day = d.strip()
    for s in spans:
        if not day and s in DAYS:
            day = s
        elif TIME.match(s):
            times = (times + " " + s).strip()
        elif not phone and re.fullmatch(r"0\d{1,2}-?\d{7}", s.replace(" ", "")):
            phone = s
        elif not price and PRICE.search(s) and len(s) < 25:
            price = s
        elif not reg and any(w in s for w in REG_WORDS):
            reg = s
        elif not venue and ("," in s or re.search(r"\d", s)) and 8 < len(s) < 90 \
                and not s.startswith("©") and "פסטיבל" not in s \
                and "אמנות במושבה" not in s and s != rec["artist"]:
            venue = s
    rec["day"] = day
    rec["time"] = re.sub(r"\s*-\s*", "–", times).strip("– ")
    rec["venue"] = venue
    rec["price"] = price
    rec["registration"] = reg
    rec["phone"] = phone
    return rec


def main():
    urls = [u for u in sitemap("post") if re.search(r"co\.il/[^/]+/?$", u)]
    print(f"{len(urls)} event posts", file=sys.stderr)
    recs = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(parse_event, u): u for u in urls}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                recs.append(fut.result())
            except Exception as e:
                print(f"  FAIL {futs[fut]}: {e}", file=sys.stderr)
            if i % 25 == 0:
                print(f"  {i}/{len(urls)}", file=sys.stderr)
    order = {d: i for i, d in enumerate(DAYS)}
    recs.sort(key=lambda r: (order.get(r["day"], 9), r["time"], r["artist"]))
    with open(f"{OUT}/events.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    print(f"  wrote {OUT}/events.json ({len(recs)})", file=sys.stderr)


if __name__ == "__main__":
    main()
