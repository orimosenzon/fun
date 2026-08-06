"""בדיקה שה-CSS המוצע מתקן את הגלילה האופקית — בכמה דפים וכמה רחבי מסך."""
from playwright.sync_api import sync_playwright

OUT = "/tmp/claude-1000/-home-ori-fun-vibe-coding/a20f9320-3500-4fab-967d-d1e46d909d89/scratchpad"
CSS = open(f"{OUT}/fix.css", encoding="utf-8").read()

PAGES = {
    "עמוד ראשי": "https://pardespedia.info/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99",
    "פרדספדיה (טבלה+גרפים)": "https://pardespedia.info/wiki/%D7%A4%D7%A8%D7%93%D7%A1%D7%A4%D7%93%D7%99%D7%94",
    "ריפבליק השקד": "https://pardespedia.info/wiki/%D7%A8%D7%99%D7%A4%D7%91%D7%9C%D7%99%D7%A7_%D7%94%D7%A9%D7%A7%D7%93",
    "האינדקס של הדפים": "https://pardespedia.info/wiki/%D7%94%D7%90%D7%99%D7%A0%D7%93%D7%A7%D7%A1_%D7%A9%D7%9C_%D7%94%D7%93%D7%A4%D7%99%D7%9D",
}

PROBE = """() => {
  const de = document.documentElement;
  const bad = [];
  for (const e of document.querySelectorAll('body *')) {
    if (e.offsetWidth > de.clientWidth + 1) {
      bad.push(e.tagName.toLowerCase() + '.' +
               (e.className||'').toString().slice(0,30) + ' w=' + e.offsetWidth);
    }
  }
  return {clientWidth: de.clientWidth, scrollWidth: de.scrollWidth,
          overflowPx: de.scrollWidth - de.clientWidth, bad: bad.slice(0,4)};
}"""


def check(pw, url, width, css, shot=None):
    b = pw.chromium.launch(channel="chrome")
    ctx = b.new_context(viewport={"width": width, "height": 800}, device_scale_factor=2,
                        is_mobile=True, has_touch=True,
                        user_agent=("Mozilla/5.0 (Linux; Android 13; SM-S911B) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120 Mobile Safari/537.36"))
    pg = ctx.new_page()
    pg.goto(url, wait_until="load", timeout=60000)
    if css:
        pg.add_style_tag(content=css)
    pg.wait_for_timeout(900)
    d = pg.evaluate(PROBE)
    if shot:
        pg.screenshot(path=f"{OUT}/{shot}.png")
    b.close()
    return d


with sync_playwright() as pw:
    for width in (360, 412):
        print(f"\n{'='*66}\nרוחב מסך {width}px\n{'='*66}")
        for name, url in PAGES.items():
            before = check(pw, url, width, None)
            after = check(pw, url, width, CSS,
                          shot=f"fixed_{width}_{name.split()[0]}" if width == 360 else None)
            f_b = "✗ %dpx" % before["overflowPx"] if before["overflowPx"] > 1 else "✓ תקין"
            f_a = "✗ %dpx" % after["overflowPx"] if after["overflowPx"] > 1 else "✓ תקין"
            print(f"  {name:26s} לפני: {f_b:10s} → אחרי: {f_a}")
            if after["overflowPx"] > 1:
                print(f"    {'':26s} נותרו חורגים: {after['bad']}")
