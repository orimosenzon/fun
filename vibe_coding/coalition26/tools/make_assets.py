"""מייצר את האייקון ואת תמונת התצוגה המקדימה (og.png) מתוך data.js.
הרצה: python3 tools/make_assets.py  (מתוך תיקיית הפרויקט; דורש playwright)
"""
import asyncio, json, math, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent

def parties():
    src = (ROOT / 'data.js').read_text(encoding='utf-8')
    out = []
    for m in re.finditer(r"\{ id: '(\w+)',\s+name: '([^']+)'.*?seats: (\d+).*?color: \['(#\w+)'", src, re.S):
        out.append({'id': m[1], 'name': m[2], 'seats': int(m[3]), 'color': m[4]})
    return out

# סדר הגושים, מימין לשמאל, כמו במצב הפתיחה של האפליקציה
ORDER = ['L', 'O', 'RZ', 'S', 'U', 'R', 'J', 'YB', 'D', 'B', 'Y']

def arc_path(cx, cy, r0, r1, a0, a1):
    # פלח של טבעת בין הזוויות a0..a1 (0 = ימין, pi = שמאל)
    p = lambda r, a: (cx + r * math.cos(a), cy - r * math.sin(a))
    x0, y0 = p(r1, a0); x1, y1 = p(r1, a1); x2, y2 = p(r0, a1); x3, y3 = p(r0, a0)
    return (f'M{x0:.2f} {y0:.2f} A{r1} {r1} 0 0 0 {x1:.2f} {y1:.2f} '
            f'L{x2:.2f} {y2:.2f} A{r0} {r0} 0 0 1 {x3:.2f} {y3:.2f}Z')

def favicon(ps):
    by = {p['id']: p for p in ps}
    segs, a = [], 0.0
    gap = 0.045
    for pid in ORDER:
        span = math.pi * by[pid]['seats'] / 120
        segs.append(f'<path d="{arc_path(32, 46, 13, 27, a + gap / 2, a + span - gap / 2)}" fill="{by[pid]["color"]}"/>')
        a += span
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="14" fill="#141414"/>'
            + ''.join(segs) +
            '<rect x="31" y="14" width="2" height="34" rx="1" fill="#fff"/></svg>')

def og_html(ps):
    by = {p['id']: p for p in ps}
    # 120 מושבים בשש שורות, כמו באפליקציה
    rows, r0, r1 = 6, 0.42, 1.0
    radii = [r0 + (r1 - r0) * i / (rows - 1) for i in range(rows)]
    counts = [round(120 * r / sum(radii)) for r in radii]
    counts[-1] += 120 - sum(counts)
    seats = []
    for r, k in zip(radii, counts):
        for j in range(k):
            seats.append((math.pi * j / (k - 1), r))
    seats.sort(key=lambda s: (s[0], -s[1]))
    owners = [pid for pid in ORDER for _ in range(by[pid]['seats'])]
    R, cx, cy = 250, 300, 330
    dots = ''.join(
        f'<circle cx="{cx + R * r * math.cos(t):.1f}" cy="{cy - R * r * math.sin(t):.1f}" r="11" fill="{by[o]["color"]}"/>'
        for (t, r), o in zip(seats, owners))
    b61 = (seats[59][0] + seats[60][0]) / 2
    line = (f'<line x1="{cx + R * .3 * math.cos(b61):.1f}" y1="{cy - R * .3 * math.sin(b61):.1f}" '
            f'x2="{cx + R * 1.1 * math.cos(b61):.1f}" y2="{cy - R * 1.1 * math.sin(b61):.1f}" '
            'stroke="#141414" stroke-width="4" stroke-dasharray="8 6"/>'
            f'<text x="{cx + R * 1.19 * math.cos(b61):.1f}" y="{cy - R * 1.19 * math.sin(b61):.1f}" '
            'text-anchor="middle" font-size="30" font-weight="800" fill="#141414">61</text>')
    return f'''<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;800&display=swap" rel="stylesheet">
<style>
body{{margin:0;width:1200px;height:630px;background:#f4f2ec;font-family:Rubik,sans-serif;color:#141414;overflow:hidden;position:relative}}
.txt{{position:absolute;right:70px;top:78px;width:520px}}
.eyebrow{{font-size:26px;color:#7d7b75;font-weight:500}}
h1{{font-size:84px;line-height:1.02;margin:14px 0 22px;font-weight:800;letter-spacing:-1px}}
p{{font-size:30px;line-height:1.35;margin:0;color:#3d3c39}}
.blocs{{display:flex;gap:34px;margin-top:34px}}
.blocs div{{display:flex;flex-direction:column}}
.blocs b{{font-size:48px;font-weight:800;line-height:1}}
.blocs span{{font-size:20px;color:#7d7b75}}
svg{{position:absolute;left:30px;top:70px}}
.n61{{position:absolute;left:316px;top:400px;font-size:30px;font-weight:800;transform:translateX(-50%)}}
.cta{{position:absolute;left:70px;bottom:46px;font-size:24px;color:#7d7b75}}
</style></head><body>
<svg width="600" height="380" viewBox="0 0 600 380">{dots}{line}</svg>
<div class="cta">בנו קואליציה בעצמכם ←</div>
<div class="txt">
<div class="eyebrow">הבחירות לכנסת ה-26</div>
<h1>מי יקים את<br>הממשלה?</h1>
<p>כל הדרכים להגיע ל-61,<br>ומה כל אחת תעלה לנו</p>
<div class="blocs"><div><b>53</b><span>גוש נתניהו</span></div><div><b>19</b><span>המפלגות הערביות</span></div><div><b>48</b><span>גוש השינוי</span></div></div>
</div></body></html>'''

async def main():
    from playwright.async_api import async_playwright
    ps = parties()
    assert sum(p['seats'] for p in ps) == 120, ps
    svg = favicon(ps)
    (ROOT / 'favicon.svg').write_text(svg, encoding='utf-8')
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1200, 'height': 630})
        await pg.set_content(og_html(ps))
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=str(ROOT / 'og.png'))
        for size, name in [(180, 'apple-touch-icon.png'), (32, 'favicon-32.png')]:
            ip = await b.new_page(viewport={'width': size, 'height': size})
            sized = svg.replace('<svg ', '<svg width="%d" height="%d" ' % (size, size))
            await ip.set_content('<body style="margin:0">' + sized + '</body>')
            await ip.screenshot(path=str(ROOT / name), omit_background=True)
        await b.close()
    print('favicon.svg, favicon-32.png, apple-touch-icon.png, og.png')

asyncio.run(main())
