#!/usr/bin/env python3
"""Scrape the אמנות במושבה 2026 programme from pardesart.co.il into JSON.

Pulls every artist and culinary entry from the site's own sitemaps, so the run
is reproducible: if the organisers add or move a participant, re-running picks
it up. Output: pa/artists.json, pa/food.json.
"""
import concurrent.futures as cf
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/pa"


def get(url, tries=3):
    url = urllib.parse.quote(url, safe=":/?&=%#")
    for n in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
        except Exception as e:
            if n == tries - 1:
                raise
            time.sleep(2 * (n + 1))


def sitemap(name):
    t = get(f"https://www.pardesart.co.il/{name}-sitemap.xml")
    return re.findall(r"<loc>(.*?)</loc>", t)


def txt(s):
    """Strip tags from an HTML fragment and collapse whitespace."""
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


# an <a> whose visible label we care about (website / facebook / instagram)
LINK_LABELS = {"אתר": "website", "פייסבוק": "facebook", "אינסטגרם": "instagram",
               "טיקטוק": "tiktok", "יוטיוב": "youtube", "וואטסאפ": "whatsapp"}
ACCESS_WORDS = ("נגיש", "לא נגיש", "כניסה נגישה", "נגיש לכסא גלגלים",
                "נגיש חלקית", "לא נגיש לכסא גלגלים")


def parse_entry(url, kind):
    t = get(url)
    rec = {"url": url, "kind": kind}

    m = re.search(r"<h1[^>]*>(.*?)</h1>", t, re.S)
    rec["name"] = txt(m.group(1)) if m else ""

    # breadcrumb: ... > אמניות/ים > <category> > <name>
    crumbs = re.findall(r"<span[^>]*>\s*<a[^>]*>\s*<span[^>]*>(.*?)</span>", t, re.S)
    rec["breadcrumb"] = [txt(c) for c in crumbs]

    # taxonomy chips (main craft, then sub-crafts). Food pages have no crafts of
    # their own, so the only matches there come from the site-wide nav: drop them.
    cats = []
    if kind == "artist":
        tax = re.findall(r'href="https://www\.pardesart\.co\.il/artist_(?:main)?crafts/([^"]+)/"[^>]*>(.*?)</a>', t, re.S)
        seen = set()
        for _slug, label in tax:
            label = txt(label)
            if label and label not in seen:
                seen.add(label)
                cats.append(label)
    rec["categories"] = cats

    # address: the Waze deep link carries it as ?q=, but some entries use Waze's
    # short form (waze.com/ul/<id>) and then only the visible label has it.
    m = re.search(r'href="https://waze\.com/ul\?q=([^"&]+)', t)
    q = urllib.parse.unquote(m.group(1)) if m else ""
    m = re.search(r"<h2[^>]*>\s*<a[^>]*waze\.com[^>]*>(.*?)</a>", t, re.S)
    rec["address_label"] = txt(m.group(1)) if m else ""
    rec["address"] = q or (rec["address_label"] + ", פרדס חנה-כרכור" if rec["address_label"] else "")

    # phone: an <a> with the phone icon, or a bare 0xx-xxxxxxx run
    phones = re.findall(r">(0\d{1,2}[-\s]?\d{7})<", t)
    rec["phone"] = phones[0].strip() if phones else ""

    # labelled outbound links
    links = {}
    for href, label in re.findall(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', t, re.S):
        lab = txt(label)
        if lab in LINK_LABELS and "pardesart.co.il" not in href:
            links.setdefault(LINK_LABELS[lab], href)
    rec["links"] = links

    # accessibility + free-text notes sit in the same strip as the links
    BOILER = ("אמנות במושבה. פסטיבל אמנות עיצוב וקראפט. פרדס חנה כרכור",
              "Designed by NotFromHere",
              "Designed by NotFromHere Developed by Digital Guru",
              "כל הזכויות שמורות לעמותת אמנות פרדס חנה כרכור © 2026",
              "לקריאה נוספת >>")
    notes, access = [], ""
    for m in re.finditer(r"<h2[^>]*>(.*?)</h2>", t, re.S):
        s = txt(m.group(1))
        if not s or s == "|" or s == rec["address_label"] or s == rec["name"]:
            continue
        if s in BOILER or s in notes:
            continue
        notes.append(s)
    # The strip under the address is a row of badges: phone, social links, an
    # accessibility statement and (for some) whether they open on Shabbat.
    badges = []
    for m in re.finditer(r"<span[^>]*>([^<]{2,60})</span>", t):
        s = txt(m.group(1))
        if s and s not in badges and s != rec["phone"]:
            badges.append(s)
    shabbat = ""
    for s in badges:
        if "נגיש" in s and not access:
            access = s
        elif "שבת" in s and not shabbat:
            shabbat = s
    rec["accessibility"] = access
    rec["shabbat"] = shabbat

    # description: the paragraphs of the body section
    body = re.split(r"</section>", t)
    paras = []
    for chunk in body:
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", chunk, re.S):
            s = txt(m.group(1))
            if len(s) > 40 and "cookie" not in s.lower():
                paras.append(s)
    # de-dup while preserving order
    seen, desc = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p)
            desc.append(p)
    rec["description"] = "\n\n".join(desc).strip()

    # Food pages carry no <p> body: their blurb and their street address are both
    # h2s in the same strip, so split them apart by length.
    if kind == "food":
        addr = [n for n in notes if len(n) < 40 and re.search(r"\d", n)]
        if addr and not rec["address"]:
            rec["address_label"] = addr[0]
            rec["address"] = addr[0] + ", פרדס חנה-כרכור"
        blurb = [n for n in notes if len(n) >= 40]
        if blurb and not rec["description"]:
            rec["description"] = "\n\n".join(blurb)
        notes = [n for n in notes if n not in addr and n not in blurb]
    rec["notes"] = notes

    # images (skip the site-wide ad banners under /2020/02/)
    imgs = re.findall(r'https://www\.pardesart\.co\.il/wp-content/uploads/(?!2020/02/)[^"\' ]+?\.(?:jpg|jpeg|png|webp)', t)
    imgs = [i for i in dict.fromkeys(imgs) if not re.search(r"-\d+x\d+\.", i)]
    rec["images"] = imgs[:8]
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    for kind, smap in (("artist", "artist"), ("food", "food")):
        urls = sitemap(smap)
        urls = [u for u in urls if re.search(rf"/{kind}/[^/]+/?$", u)]
        print(f"{kind}: {len(urls)} pages", file=sys.stderr)
        recs = []
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(parse_entry, u, kind): u for u in urls}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                try:
                    recs.append(fut.result())
                except Exception as e:
                    print(f"  FAIL {futs[fut]}: {e}", file=sys.stderr)
                if i % 20 == 0:
                    print(f"  {i}/{len(urls)}", file=sys.stderr)
        recs.sort(key=lambda r: r["name"])
        with open(f"{OUT}/{kind}s.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        print(f"  wrote {OUT}/{kind}s.json ({len(recs)})", file=sys.stderr)


if __name__ == "__main__":
    main()
