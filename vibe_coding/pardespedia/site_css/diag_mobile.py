"""אבחון גלישה אופקית בעמוד הראשי של פרדספדיה במסך טלפון."""
import sys
from playwright.sync_api import sync_playwright

URL = "https://pardespedia.info/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99"
OUT = "/tmp/claude-1000/-home-ori-fun-vibe-coding/a20f9320-3500-4fab-967d-d1e46d909d89/scratchpad"
EXTRA_CSS = sys.argv[1] if len(sys.argv) > 1 else None
TAG = sys.argv[2] if len(sys.argv) > 2 else "before"

PROBE = """() => {
  const de = document.documentElement;
  const vw = window.innerWidth;
  const out = {
    innerWidth: vw,
    scrollWidth: de.scrollWidth,
    clientWidth: de.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
    scrollLeftInitial: de.scrollLeft,
    overflow: []
  };
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    // חורג מימין (מעבר לרוחב החלון) או משמאל (לפני 0)
    if (r.right > vw + 1 || r.left < -1) {
      out.overflow.push({
        tag: el.tagName.toLowerCase(),
        cls: (el.className || '').toString().slice(0, 60),
        left: Math.round(r.left), right: Math.round(r.right),
        width: Math.round(r.width),
        attrW: el.getAttribute && el.getAttribute('width'),
        txt: (el.textContent || '').trim().slice(0, 40)
      });
    }
  }
  return out;
}"""


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(
            viewport={"width": 360, "height": 780},
            device_scale_factor=2, is_mobile=True, has_touch=True,
            user_agent=("Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"),
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="load", timeout=60000)
        if EXTRA_CSS:
            page.add_style_tag(content=open(EXTRA_CSS, encoding="utf-8").read())
        page.wait_for_timeout(1500)

        data = page.evaluate(PROBE)
        print(f"innerWidth      = {data['innerWidth']}")
        print(f"scrollWidth     = {data['scrollWidth']}  (html)")
        print(f"clientWidth     = {data['clientWidth']}")
        print(f"body.scrollWidth= {data['bodyScrollWidth']}")
        extra = data['scrollWidth'] - data['clientWidth']
        print(f"--> גלישה אופקית: {extra}px" if extra > 1 else "--> אין גלישה אופקית ✓")
        print(f"\nחריגות ({len(data['overflow'])}):")
        seen = set()
        for o in data["overflow"][:40]:
            key = (o["tag"], o["cls"], o["width"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  <{o['tag']} class='{o['cls']}'> w={o['width']} "
                  f"left={o['left']} right={o['right']} attrW={o['attrW']} | {o['txt']!r}")

        page.screenshot(path=f"{OUT}/mobile_{TAG}.png", full_page=False)
        print(f"\nצילום: {OUT}/mobile_{TAG}.png")
        browser.close()


main()
