#!/usr/bin/env python3
"""Curated pictures and clips for the layers this app builds from outside data.

The layers under "מהעולם" are rebuilt from OpenStreetMap on every run, so a
photo somebody attaches in the app cannot live inside them - it goes in the
side-car, ``data/media.json``. The other half of the story is this: media the
*build* puts there, hand-picked once and resolved fresh on every build.

Two sources, and each is asked a different question:

  Wikimedia Commons   a file title, resolved to a thumbnail, a display copy and
                      the credit the picture is offered under
  YouTube oEmbed      a video id, answered "does this still exist and may it
                      still be embedded"

Written out as fixed titles and ids rather than searched for at build time, on
purpose: a search that returns something different next month would silently
put a different picture on the map, and the point of a curated handful is that
somebody looked at each one. What *is* asked afresh every build is whether each
one is still there, so a picture deleted from Commons or a video taken down
drops out rather than becoming a broken tile.

Shared by ``build_curitiba.py`` and ``build_houten.py``, which had the same
hundred lines each until 8/9/2026.
"""

import json
import re
import time
import urllib.parse
import urllib.request

COMMONS = "https://commons.wikimedia.org/w/api.php"
OEMBED = "https://www.youtube.com/oembed"

UA = "derech-kitzur/build (github.com/orimosenzon/fun)"


def http_json(url, what):
    """One GET that answers JSON, or None. Never fatal: a build with no
    pictures is a worse layer, and a build that died is no layer."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as err:                     # network, 404, rate limit
        print(f"  {what}: {err}", flush=True)
        return None


def commons_photos(titles):
    """Resolve Commons file titles to a thumbnail, a display copy and a credit.

    Two renditions and never the original: several of these are 5000 px wide and
    weigh megabytes, and the app would fetch that to fill a phone screen. The
    thumbnailer takes any width, so 500 for the strip and 1600 for the lightbox
    matches exactly what `build_data.py` writes for the initiative's own photos.

    Credit is read off Commons rather than written here, because that is the
    condition the pictures are offered under and a copy of it would go stale.
    """
    if not titles:
        return {}

    def ask(width):
        url = COMMONS + "?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": "|".join(titles),
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": width,
        })
        doc = http_json(url, "ויקישיתוף")
        pages = (doc or {}).get("query", {}).get("pages", {}) or {}
        return {p["title"]: (p.get("imageinfo") or [{}])[0]
                for p in pages.values() if p.get("title")}

    small, large = ask(500), ask(1600)
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()

    # Commons hangs `?utm_source=…&utm_campaign=imageinfo` off every url the API
    # hands out. Dropping it is not only tidiness: it is a campaign tag that
    # would be sent by every visitor's browser on every load, and it says
    # nothing this app needs the image server to know.
    plain = lambda u: u.split("?", 1)[0] if u else u

    out = {}
    for title in titles:
        thumb, full = small.get(title, {}), large.get(title, {})
        if not thumb.get("thumburl"):
            print(f"  חסרה תמונה בוויקישיתוף: {title}", flush=True)
            continue
        meta = full.get("extmetadata", thumb.get("extmetadata", {})) or {}
        artist = strip(meta.get("Artist", {}).get("value"))
        lic = strip(meta.get("LicenseShortName", {}).get("value"))
        credit = " · ".join(x for x in (artist, lic) if x)
        out[title] = {
            "thumb": plain(thumb["thumburl"]),
            "full": plain(full.get("thumburl") or thumb["thumburl"]),
            "cap": credit or "ויקישיתוף",
        }
    return out


def youtube_ok(vid):
    """Does this video still exist and still allow embedding.

    oEmbed is the free way to ask: it needs no key, and it answers 404 both for
    a video that was taken down and for one whose owner has switched embedding
    off - which are different things to a person and the same thing to this map,
    because both come out as a tile that plays nothing.
    """
    url = OEMBED + "?" + urllib.parse.urlencode({
        "format": "json", "url": f"https://www.youtube.com/watch?v={vid}"})
    doc = http_json(url, f"יוטיוב {vid}")
    return bool(doc and doc.get("title"))


def video_entry(vid):
    return {"yt": vid, "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}


def attach_media(wanted, skip=False):
    """Hang the curated picture and clip on the items that asked for one.

    `wanted` is a list of (item, spec) pairs - a list and not a dictionary
    keyed by item, because the items are plain dicts and a dict cannot be a key.
    A spec carries `file`, `yt`, or both; each half drops out on its own if its
    source no longer has it, and the item keeps whatever survived. The picture
    goes first, because a strip that opens with a YouTube thumbnail reads as a
    video gallery.

    An item may appear in `wanted` more than once, which is how a layer or a
    segment ends up carrying two clips: the entries simply append in order.
    """
    if skip or not wanted:
        return
    titles = sorted({w["file"] for _, w in wanted if w.get("file")})
    photos = commons_photos(titles)
    checked = {}
    for it, want in wanted:
        shot = photos.get(want.get("file"))
        if shot:
            it["photos"].append(dict(shot))
        # `yt` is one id or a list of them. A whole town's layer earns two
        # clips and a single path earns one, and accepting both shapes here
        # keeps the rules readable at their own end.
        clips = want.get("yt") or []
        for vid in [clips] if isinstance(clips, str) else clips:
            if vid not in checked:
                checked[vid] = youtube_ok(vid)
                time.sleep(0.4)                  # one clip at a time, politely
            if checked[vid]:
                it["photos"].append(video_entry(vid))
