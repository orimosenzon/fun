#!/usr/bin/env python3
"""Print map of the Hanadiv festival venues, made to be drawn on.

The festival team wants a paper map to scribble routes and notes over with
markers, so the design goal is the opposite of the app's: a pale, quiet base
with the streets named, big numbered pins for the 21 venues, and nothing else
competing with a felt-tip. Page 2 is the schedule per venue so a number on the
map can be looked up without a phone.

Why Chromium and not matplotlib/Pillow: the base map is MapLibre over the same
OpenFreeMap tiles the app uses, the labels are Hebrew (bidi, real fonts) and a
browser already does all of that. The page is rendered with Playwright and
printed to PDF, so the legend text stays vector and the map canvas is embedded
as a bitmap at DPR x OVERZOOM = 3.4 device pixels per CSS px, ~320dpi on paper.

The base style is OpenFreeMap's positron, patched for print: Hebrew-only street
names (positron prints "Latin Hebrew" pairs), minor street names from zoom 13
instead of 15, no road-number shields, darker label ink, wider streets. The
map is drawn OVERZOOM times larger than its slot and scaled down with CSS,
because the tiles carry residential street names only from zoom 14 and the
festival fits a page at 13.3; see OVERZOOM below. An invisible symbol layer at
each pin keeps street names from being placed under the pins.

    python3 build_hanadiv_print.py            # web/print/hanadiv_A4.pdf + _A3.pdf
    python3 build_hanadiv_print.py --paper A3 # one size only
    python3 build_hanadiv_print.py --png      # also a PNG of the map page, for a quick look
"""

import argparse
import copy
import datetime as dt
import html
import json
import pathlib
import re
import statistics
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
WEB = ROOT / "web"
SRC = WEB / "data" / "hanadiv.json"
OUT = WEB / "print"

STYLE_URL = "https://tiles.openfreemap.org/styles/positron"
MAPLIBRE = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl"
RTL_PLUGIN = (WEB / "vendor" / "mapbox-gl-rtl-text.min.js")

# Landscape, in mm. The legend column is a fixed share so A3 and A4 keep the
# same composition; only the map's zoom (and so its label density) differs.
PAPER = {"A4": (297, 210), "A3": (420, 297)}
DPR = 2

# The map is laid out OVERZOOM times larger than its slot and scaled back down
# with a CSS transform. The whole festival fits one frame at zoom ~13.3, but
# the vector tiles only carry the names of residential streets from zoom 14,
# and those are the streets the venues are on. 1.7x lifts the frame to ~14.1.
# Every size inside the map (pins, label text, scale bar) is multiplied by it so
# that on paper nothing changes except the label density.
OVERZOOM = 1.7

# Festival brand-ish red, chosen for one reason: it is not a colour anybody
# will reach for in a marker pack (those are green, blue, black, orange).
PIN = "#c2185b"

DAY = {3: "ה'", 4: "ו'", 5: "ש'"}          # weekday() -> Hebrew day letter
DAY_NAME = {3: "חמישי", 4: "שישי", 5: "שבת"}


# --- data -------------------------------------------------------------------

def load(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=20) as r:
            return json.load(r)
    return json.loads(pathlib.Path(src).read_text())


def strip_day(name):
    """'יוגה תרפיה · יום חמישי' -> 'יוגה תרפיה'. The day is its own column."""
    return re.sub(r"\s*·\s*יום\s+(ראשון|שני|שלישי|רביעי|חמישי|שישי|שבת)\s*$", "", name)


def venues(data):
    """One entry per address, west to east, numbered from 1.

    The app spreads stacked pins a few metres apart so they can be tapped; the
    centroid of a stack is the address itself, near enough for print. Numbering
    west to east means neighbouring numbers are neighbours on the map, which is
    the only property a number order can usefully have here.
    """
    by = {}
    for p in data["places"]:
        addr, _, where = p["address"].partition(" · ")
        v = by.setdefault(addr, {"addr": addr, "wheres": [], "events": [], "lat": [], "lng": []})
        if where and where not in v["wheres"]:
            v["wheres"].append(where)
        v["events"].append(p)
        v["lat"].append(p["geo"]["lat"])
        v["lng"].append(p["geo"]["lng"])
    out = []
    for v in by.values():
        v["lat"] = statistics.fmean(v["lat"])
        v["lng"] = statistics.fmean(v["lng"])
        v["events"].sort(key=lambda e: (e["date"], e["hour"]))
        for e in v["events"]:
            d = dt.date.fromisoformat(e["date"])
            e["day"] = DAY.get(d.weekday(), "")
            e["dm"] = f"{d.day}.{d.month}"
            e["title"] = strip_day(e["name"])
        v["days"] = sorted({e["day"] for e in v["events"]}, key=list(DAY.values()).index)
        out.append(v)
    out.sort(key=lambda v: v["lng"])
    for i, v in enumerate(out, 1):
        v["n"] = i
    return out


# --- base style -------------------------------------------------------------

HEBREW_NAME = ["coalesce", ["get", "name:nonlatin"], ["get", "name"], ["get", "name:latin"]]


def print_style(k=OVERZOOM):
    """Positron, tuned for paper: Hebrew labels, more of them, darker ink.

    Pixel sizes are in the overzoomed frame, so they carry the factor k; a
    text-size of 9*k is 9 CSS px, about 2.4mm, once the frame is scaled down.
    """
    # OpenFreeMap answers 403 to urllib's default User-Agent; curl and browsers are fine.
    req = urllib.request.Request(STYLE_URL, headers={"User-Agent": "derech-kitzur-print/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        s = json.load(r)
    s = copy.deepcopy(s)
    for l in s["layers"]:
        lid, layout, paint = l["id"], l.setdefault("layout", {}), l.setdefault("paint", {})
        if l["type"] == "symbol" and "text-field" in layout and "shield" not in lid:
            layout["text-field"] = HEBREW_NAME
        if lid == "highway-name-minor":
            l["minzoom"] = 13
            layout["text-size"] = 9 * k
            layout["symbol-spacing"] = 160 * k       # repeat along a long street
            layout["text-padding"] = 1
            paint["text-color"] = "#333"
            paint["text-halo-width"] = 1.2 * k
        elif lid == "highway-name-major":
            layout["text-size"] = 10.5 * k
            layout["symbol-spacing"] = 200 * k
            paint["text-color"] = "#222"
            paint["text-halo-width"] = 1.2 * k
        elif lid == "highway-name-path":
            l["minzoom"] = 14.5
            layout["text-size"] = 8.5 * k
        elif lid in ("highway-shield-non-us", "highway-shield-us-interstate", "road_shield_us"):
            # Line symbols restart their spacing at every tile edge, so a road
            # number lands once per 80mm whatever the spacing says. Locals walk
            # by street names, not by 6502.
            layout["visibility"] = "none"
        elif lid == "highway_minor":
            paint["line-color"] = "hsl(0,0%,80%)"
            paint["line-width"] = ["interpolate", ["exponential", 1.55], ["zoom"], 13, 2.4 * k, 20, 20 * k]
        elif lid == "highway_path":
            paint["line-color"] = "hsl(0,0%,82%)"
            paint["line-width"] = ["interpolate", ["exponential", 1.2], ["zoom"], 13, 1 * k, 20, 10 * k]
        elif lid in ("highway_major_casing", "highway_motorway_casing"):
            paint["line-color"] = "rgb(190,190,190)"
        elif lid == "building":
            # Building footprints are the marker's worst enemy: grey confetti.
            paint["fill-color"] = "hsl(0,0%,93%)"
            paint["fill-outline-color"] = "hsl(0,0%,91%)"
        elif lid in ("label_town", "label_village", "label_other"):
            # 'פרדס חנה-כרכור' in 14pt across the middle of the map, no thanks.
            layout["visibility"] = "none"
    return s


# --- page -------------------------------------------------------------------

def esc(s):
    return html.escape(str(s or ""), quote=True)


def legend_html(vs):
    rows = []
    for v in vs:
        n = len(v["events"])
        count = "אירוע אחד" if n == 1 else f"{n} אירועים"
        days = " ".join(v["days"])
        where = esc(v["wheres"][0]) if v["wheres"] else ""
        rows.append(f"""
      <li>
        <span class="num">{v['n']}</span>
        <span class="txt"><b>{esc(v['addr'])}</b><small>{where}</small></span>
        <span class="cnt">{count}<small>{days}</small></span>
      </li>""")
    return "\n".join(rows)


def schedule_html(vs, groups):
    colour = {g["name"]: g["color"] for g in groups}
    blocks = []
    for v in vs:
        rows = []
        for e in v["events"]:
            c = colour.get(e["group"], "#607d8b")
            aud = e.get("audience", "")
            aud = {"לכל המשפחה": "משפחה", "רק למבוגרים": "מבוגרים", "רק לילדים": "ילדים"}.get(aud, aud)
            rows.append(f"""
        <tr>
          <td class="d">{e['day']} {e['dm']}</td>
          <td class="h">{esc(e['hour'])}</td>
          <td class="t"><span class="g" style="--c:{c}"></span>{esc(e['title'])}</td>
          <td class="a">{esc(aud)}</td>
        </tr>""")
        where = " · ".join(esc(w) for w in v["wheres"])
        blocks.append(f"""
    <section>
      <h3><span class="num">{v['n']}</span> {esc(v['addr'])}<small>{where}</small></h3>
      <table>{''.join(rows)}
      </table>
    </section>""")
    chips = "".join(f'<span class="chip"><span class="g" style="--c:{g["color"]}"></span>{esc(g["name"])}</span>'
                    for g in groups)
    return "\n".join(blocks), chips


def page_html(data, vs, style, paper):
    w, h = PAPER[paper]
    fest = data.get("festival", {})
    d0 = dt.date.fromisoformat(fest.get("from", "2026-10-29"))
    d1 = dt.date.fromisoformat(fest.get("until", "2026-10-31"))
    title = fest.get("name", "פסטיבל דרך הנדיב")
    dates = f"{DAY_NAME.get(d0.weekday(), '')}–{DAY_NAME.get(d1.weekday(), '')}, {d0.day}–{d1.day} באוקטובר {d1.year}"
    stamp = dt.date.today().strftime("%d.%m.%Y")
    legend = legend_html(vs)
    schedule, chips = schedule_html(vs, data.get("groups", []))
    pins = json.dumps([{"n": v["n"], "lng": v["lng"], "lat": v["lat"]} for v in vs], ensure_ascii=False)
    rtl = "data:text/javascript;base64," + __import__("base64").b64encode(RTL_PLUGIN.read_bytes()).decode()
    n_ev, n_v = len(data["places"]), len(vs)

    # Page geometry in mm. The map slot is what the reader sees; the map element
    # itself is OVERZOOM times bigger and transformed down into the slot.
    m, head, foot = 7, 14, 6
    legend_mm = round(w * 0.235)
    slot_w = w - 2 * m - legend_mm - 4
    slot_h = h - 2 * m - head - 3 - foot
    k = OVERZOOM
    pin_mm = 6.4 if paper == "A4" else 7.2
    # Type is set in points, so on A3 the legend would sit small in a big
    # column. CSS zoom scales the text blocks; the column heights divide it out.
    fs = 1.0 if paper == "A4" else 1.3

    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<title>{esc(title)} · מפה להדפסה</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{MAPLIBRE}.css">
<style>
  @page {{ size: {w}mm {h}mm; margin: 0; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {w}mm; height: {h * 2}mm; overflow: hidden;
               font-family: Heebo, 'Noto Sans Hebrew', 'DejaVu Sans', sans-serif;
               color: #111; background: #fff; }}
  .page {{ position: relative; width: {w}mm; height: {h}mm; overflow: hidden;
           break-after: page; page-break-after: always; }}
  .page:last-child {{ break-after: auto; page-break-after: auto; }}

  /* ---- page 1: the map --------------------------------------------------- */
  :root {{ --m: {m}mm; --legend: {legend_mm}mm; --head: {head}mm; --k: {k}; --pin: calc({pin_mm}mm * var(--k)); }}
  header {{ position: absolute; top: var(--m); right: var(--m); left: var(--m); height: var(--head);
            display: flex; align-items: baseline; gap: 4mm; border-bottom: .5mm solid #111; }}
  header h1 {{ font-size: 22pt; font-weight: 900; letter-spacing: -.2pt; line-height: 1; }}
  header .dates {{ font-size: 12pt; font-weight: 500; }}
  header .sub {{ margin-inline-start: auto; font-size: 9pt; color: #555; }}
  /* Everything inside #map is in the overzoomed frame: sizes carry var(--k). */
  #map {{ position: absolute; top: calc(var(--m) + var(--head) + 3mm); right: calc(var(--m) + var(--legend) + 4mm);
          width: {slot_w * k:.2f}mm; height: {slot_h * k:.2f}mm;
          transform: scale({1 / k:.5f}); transform-origin: top right;
          border: calc(.3mm * var(--k)) solid #999; background: #f7f7f5; }}
  #leader {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; }}
  #leader line {{ stroke: {PIN}; stroke-width: calc(.35mm * var(--k)); }}
  #leader circle {{ fill: {PIN}; }}
  /* No box-shadow: Chromium exports a blurred shadow to PDF as a rectangular
     bitmap, and phone viewers that drop its alpha show a grey square behind
     every pin (Yoav's first comment). A white border and a solid outline are
     plain vector shapes and survive any viewer. */
  .pin {{ width: var(--pin); height: var(--pin); border-radius: 50%;
          background: {PIN}; color: #fff; border: calc(var(--pin) * .07) solid #fff;
          outline: calc(var(--pin) * .045) solid {PIN};
          font-size: calc(var(--pin) * .52); font-weight: 700; line-height: 1;
          display: flex; align-items: center; justify-content: center; font-variant-numeric: tabular-nums; }}
  .maplibregl-ctrl-scale {{ font-family: Heebo, sans-serif; font-size: calc(8pt * var(--k)); color: #111; direction: ltr;
                            background: rgba(255,255,255,.85); border: calc(.3mm * var(--k)) solid #111; border-top: 0;
                            padding: 0 calc(1.5mm * var(--k)); }}
  .maplibregl-ctrl-bottom-left {{ bottom: calc(2mm * var(--k)); left: calc(2mm * var(--k)); }}
  .maplibregl-ctrl {{ margin: 0 !important; }}
  .north {{ position: absolute; top: calc(var(--m) + var(--head) + 5mm); left: calc(var(--m) + 2mm); z-index: 2;
            width: 9mm; text-align: center; font-size: 7pt; font-weight: 700; line-height: 1; color: #111; }}
  .north svg {{ display: block; width: 9mm; height: 9mm; margin-bottom: .5mm; }}

  aside {{ position: absolute; top: calc(var(--m) + var(--head) + 3mm); right: var(--m); width: var(--legend);
           bottom: calc(var(--m) + {foot}mm); overflow: hidden; }}
  aside h2 {{ font-size: 10.5pt; font-weight: 700; margin-bottom: 1.5mm; }}
  aside ol {{ list-style: none; zoom: {fs}; }}
  aside li {{ display: flex; align-items: center; gap: 1.8mm; padding: .6mm 0; border-bottom: .15mm solid #ddd; }}
  aside .num {{ flex: none; width: 5.4mm; height: 5.4mm; border-radius: 50%; background: {PIN}; color: #fff;
                font-size: 7.5pt; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
  aside .txt {{ flex: 1; min-width: 0; line-height: 1.1; }}
  aside .txt b {{ display: block; font-size: 8.5pt; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  aside .txt small {{ display: block; font-size: 6.5pt; color: #666; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  aside .cnt {{ flex: none; text-align: left; font-size: 7pt; color: #333; line-height: 1.1; }}
  aside .cnt small {{ display: block; font-size: 6.5pt; color: #777; letter-spacing: .3pt; }}
  footer {{ position: absolute; bottom: var(--m); right: var(--m); left: var(--m); height: 5mm;
            display: flex; align-items: flex-end; justify-content: space-between; font-size: 6.5pt; color: #666; }}
  footer a {{ color: inherit; text-decoration: none; }}

  /* ---- page 2: the schedule ---------------------------------------------- */
  .list {{ padding: var(--m); }}
  .list header {{ position: static; margin-bottom: 3mm; }}
  .list .cols {{ column-count: 4; column-gap: 6mm; column-fill: auto; zoom: {fs};
                 height: calc(({h}mm - 2 * var(--m) - var(--head) - 12mm) / {fs}); }}
  section {{ break-inside: avoid; margin-bottom: 3mm; }}
  section h3 {{ font-size: 9.5pt; font-weight: 700; display: flex; align-items: center; gap: 1.6mm;
                border-bottom: .3mm solid #111; padding-bottom: .6mm; margin-bottom: .6mm; flex-wrap: wrap; }}
  section h3 .num {{ width: 5.2mm; height: 5.2mm; border-radius: 50%; background: {PIN}; color: #fff;
                     font-size: 7.5pt; display: inline-flex; align-items: center; justify-content: center; flex: none; }}
  section h3 small {{ font-size: 7pt; font-weight: 400; color: #666; flex-basis: 100%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 8pt; }}
  td {{ padding: .5mm .8mm .5mm 0; vertical-align: top; border-bottom: .12mm solid #e5e5e5; }}
  td.d {{ white-space: nowrap; width: 11mm; color: #333; }}
  td.h {{ white-space: nowrap; width: 9mm; font-variant-numeric: tabular-nums; }}
  td.t {{ font-weight: 500; }}
  td.a {{ white-space: nowrap; width: 12mm; color: #777; font-size: 7pt; text-align: left; }}
  .g {{ display: inline-block; width: 2.2mm; height: 2.2mm; border-radius: 50%; background: var(--c);
        margin-inline-end: 1.2mm; vertical-align: -.1mm; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 1.5mm 4mm; font-size: 7.5pt; color: #333; margin-bottom: 3mm; zoom: {fs}; }}
  .chip {{ white-space: nowrap; }}
</style></head><body>

<div class="page">
  <header>
    <h1>{esc(title)}</h1>
    <span class="dates">{esc(dates)}</span>
    <span class="sub">{n_ev} אירועים ב-{n_v} מקומות · מפת המקומות</span>
  </header>
  <div id="map"></div>
  <div class="north"><svg viewBox="0 0 20 20"><path d="M10 1 L14 17 L10 13 L6 17 Z" fill="#111"/></svg>צפון</div>
  <aside>
    <h2>המקומות, ממערב למזרח</h2>
    <ol>{legend}
    </ol>
  </aside>
  <footer>
    <span>הלוח המלא בעמוד הבא · באפליקציה: <a href="https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/?layers=hanadiv">orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/?layers=hanadiv</a></span>
    <span>נתונים: 2026.hanadiv.org · מפה: © OpenStreetMap contributors, OpenFreeMap · הופק ב-{stamp} על ידי דרך קיצור</span>
  </footer>
</div>

<div class="page list">
  <header>
    <h1>{esc(title)}</h1>
    <span class="dates">לוח האירועים לפי מקום</span>
    <span class="sub">המספרים כמו במפה · הצבע לפי סוג האירוע</span>
  </header>
  <div class="chips">{chips}</div>
  <div class="cols">{schedule}
  </div>
</div>

<script src="{MAPLIBRE}.js"></script>
<script>
(async () => {{
  const PINS = {pins};
  const K = {k};
  const STYLE = {json.dumps(style, ensure_ascii=False)};
  // Hebrew street names come out backwards without the shaping plugin. It is
  // inlined so the page has no file:// dependency (see build_og.py on origins).
  await maplibregl.setRTLTextPlugin("{rtl}", true);

  const map = new maplibregl.Map({{
    container: 'map', style: STYLE, attributionControl: false, interactive: false,
    preserveDrawingBuffer: true, fadeDuration: 0, pixelRatio: window.devicePixelRatio,
  }});
  map.addControl(new maplibregl.ScaleControl({{ maxWidth: 180 * K, unit: 'metric' }}), 'bottom-left');

  const el = document.getElementById('map');
  // The pin's size in the map's own (overzoomed, untransformed) pixels.
  const pinPx = {pin_mm} * K * 96 / 25.4;

  const b = new maplibregl.LngLatBounds();
  PINS.forEach(p => b.extend([p.lng, p.lat]));
  map.fitBounds(b, {{ padding: pinPx * 1.6, duration: 0 }});

  // The pins are HTML, so the map's label placement does not know they exist
  // and happily puts a street name under one. An invisible symbol the size of
  // a pin, in the topmost layer (placed first, so it wins), makes the street
  // names keep clear; a name that loses here reappears further along its
  // street, which is where the reader can see it anyway.
  map.on('load', () => {{
    map.addSource('pins', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
      features: PINS.map(p => ({{ type: 'Feature', geometry: {{ type: 'Point', coordinates: [p.lng, p.lat] }}, properties: {{}} }})) }} }});
    map.addLayer({{ id: 'pin-space', type: 'symbol', source: 'pins',
      layout: {{ 'text-field': 'OO', 'text-font': ['Noto Sans Regular'], 'text-size': pinPx * .8,
                 'text-padding': pinPx * .12, 'text-allow-overlap': true, 'text-ignore-placement': false }},
      paint: {{ 'text-opacity': 0 }} }});
  }});

  const markers = PINS.map(p => {{
    const node = document.createElement('div');
    node.className = 'pin'; node.textContent = p.n;
    return new maplibregl.Marker({{ element: node, anchor: 'center' }}).setLngLat([p.lng, p.lat]).addTo(map);
  }});

  // Two venues 75m apart are 13px apart at this zoom and the pins would sit on
  // top of each other. Push overlapping pins apart in screen space and draw a
  // leader back to the true spot for any that had to move noticeably.
  function spread() {{
    const pts = PINS.map(p => {{ const q = map.project([p.lng, p.lat]); return {{ x: q.x, y: q.y, ox: q.x, oy: q.y }}; }});
    const min = pinPx * 1.15;
    for (let it = 0; it < 200; it++) {{
      let moved = false;
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {{
        const a = pts[i], c = pts[j];
        let dx = c.x - a.x, dy = c.y - a.y, d = Math.hypot(dx, dy);
        if (d >= min) continue;
        if (d < 1e-3) {{ dx = 1; dy = 0; d = 1; }}
        const push = (min - d) / 2 + .25;
        a.x -= dx / d * push; a.y -= dy / d * push; c.x += dx / d * push; c.y += dy / d * push;
        moved = true;
      }}
      if (!moved) break;
    }}
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.id = 'leader';
    pts.forEach((p, i) => {{
      const dx = p.x - p.ox, dy = p.y - p.oy, d = Math.hypot(dx, dy);
      markers[i].setOffset([dx, dy]);
      if (d < 3 * K) return;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', p.ox); line.setAttribute('y1', p.oy);
      line.setAttribute('x2', p.x - dx / d * pinPx / 2); line.setAttribute('y2', p.y - dy / d * pinPx / 2);
      svg.appendChild(line);
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', p.ox); dot.setAttribute('cy', p.oy); dot.setAttribute('r', 2.2 * K);
      svg.appendChild(dot);
    }});
    el.appendChild(svg);
  }}

  map.once('idle', async () => {{
    spread();
    await document.fonts.ready;
    window.__zoom = map.getZoom();
    window.__ready = true;
  }});
}})();
</script>
</body></html>"""


# --- render -----------------------------------------------------------------

def render(html_text, paper, pdf_path, png_path=None):
    from playwright.sync_api import sync_playwright
    w, h = PAPER[paper]
    px = lambda mm: round(mm * 96 / 25.4)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page(viewport={"width": px(w), "height": px(h)}, device_scale_factor=DPR)
        page.on("console", lambda m: print("  [browser]", m.text) if m.type in ("error", "warning") else None)
        page.set_content(html_text, wait_until="load")
        page.wait_for_function("window.__ready === true", timeout=120_000)
        zoom = page.evaluate("window.__zoom")
        if png_path:
            page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": px(w), "height": px(h)})
        page.pdf(path=str(pdf_path), width=f"{w}mm", height=f"{h}mm", print_background=True,
                 prefer_css_page_size=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        b.close()
    return zoom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="hanadiv.json path or URL")
    ap.add_argument("--paper", choices=sorted(PAPER), action="append", help="default: both")
    ap.add_argument("--png", action="store_true", help="also write a PNG of the map page")
    ap.add_argument("--html", action="store_true", help="also keep the page source next to the PDF")
    a = ap.parse_args()

    data = load(a.src)
    vs = venues(data)
    print(f"{len(data['places'])} events at {len(vs)} venues, "
          f"{data.get('festival', {}).get('from')}..{data.get('festival', {}).get('until')}")
    style = print_style()
    OUT.mkdir(exist_ok=True)
    for paper in a.paper or sorted(PAPER):
        page = page_html(data, vs, style, paper)
        pdf = OUT / f"hanadiv_{paper}.pdf"
        png = OUT / f"hanadiv_{paper}.png" if a.png else None
        if a.html:
            (OUT / f"hanadiv_{paper}.html").write_text(page)
        zoom = render(page, paper, pdf, png)
        print(f"  {pdf.relative_to(ROOT)}  {PAPER[paper][0]}x{PAPER[paper][1]}mm  zoom {zoom:.2f}  {pdf.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
