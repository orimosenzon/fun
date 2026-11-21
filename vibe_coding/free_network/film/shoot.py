"""The camera: photographs the screens of the staged network (node film/stage.mjs first).

  python3 film/shoot.py            every still the film uses, into film/shots/
Stills are taken at twice the screen's sharpness, so the film can move in on a detail.
"""
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SHOTS = HERE / "shots"
stage = json.loads((HERE / ".stage.json").read_text())
SITE = stage["site"]


def page_for(browser, who, width, height, phone=False, welcome=False):
    context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=2, locale="he-IL",
                                  timezone_id="Asia/Jerusalem", bypass_csp=True, is_mobile=phone, has_touch=phone, reduced_motion="reduce")
    name, value = stage["sessions"][who].split("=", 1)
    context.add_cookies([{"name": name, "value": value, "url": SITE}])
    if welcome:
        context.add_init_script("try { sessionStorage.setItem('welcome', '1') } catch {}")
    return context.new_page()


def settle(page, ms=900):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def boxes(page, **selectors):
    """Where things are on the still, in still pixels, so the film can point at them."""
    out = {}
    for key, selector in selectors.items():
        box = page.locator(selector).first.bounding_box()
        if box:
            out[key] = [round(box[k] * 2) for k in ("x", "y", "width", "height")]
    return out


def main():
    SHOTS.mkdir(exist_ok=True)
    marks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()

        page = page_for(browser, "dana", 1360, 860, welcome=True)
        page.goto(SITE + "/")
        settle(page)
        page.screenshot(path=SHOTS / "welcome.png")
        marks["welcome"] = boxes(page, face=".welcome .avatar", name=".welcome .field", google=".welcome .google-button", go=".welcome button[type=submit]", card=".welcome")

        page = page_for(browser, "dana", 1360, 860)
        page.goto(SITE + "/")
        settle(page, 4500)
        page.screenshot(path=SHOTS / "home.png")
        marks["home"] = boxes(page, rule=".feed-rule", composer=".composer", first=".feed .post", globe=".globe-side", update=".update-side")

        # One column, as long as the whole feed, for the film to travel down. The news lines above
        # the composer are left out: the film has its own moment for the update.
        page = page_for(browser, "dana", 990, 860)
        page.goto(SITE + "/")
        settle(page, 3000)
        page.add_style_tag(content=".update-top { display: none !important; }")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "feed.png", full_page=True)
        marks["feed"] = boxes(page, rule=".feed-rule", composer=".composer", video=".post .video", end=".feed-end")
        marks["feed"]["posts"] = [[round(v * 2) for v in (b["x"], b["y"], b["width"], b["height"])] for b in (post.bounding_box() for post in page.locator(".feed .post").all())]
        marks["feed"]["days"] = [[round(v * 2) for v in (b["x"], b["y"], b["width"], b["height"])] for b in (day.bounding_box() for day in page.locator(".feed .day").all())]
        # The same page with a post being written, first for friends and then for everyone.
        page.locator(".composer textarea").click()
        page.keyboard.type("היי לכולם, הצטרפתי. מי בא לשוק ביום שישי?")
        page.wait_for_timeout(400)
        page.screenshot(path=SHOTS / "compose.png", full_page=True)
        marks["compose"] = boxes(page, composer=".composer", audience=".composer .audience", reach=".composer .reach", publish=".composer button[type=submit]",
                                 friends=".composer .audience label >> nth=0", public=".composer .audience label >> nth=1")
        page.locator(".composer .audience label").nth(1).click()
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "compose_public.png", full_page=True)

        page = page_for(browser, "dana", 990, 860)
        page.goto(f"{SITE}/p/{stage['garden']}")
        settle(page, 1500)
        page.screenshot(path=SHOTS / "post.png", full_page=True)
        marks["post"] = boxes(page, thread=".thread", link=".thread a.url", post=".post", form=".thread > .comment-form")
        marks["post"]["comments"] = [[round(v * 2) for v in (b["x"], b["y"], b["width"], b["height"])] for b in (c.bounding_box() for c in page.locator(".thread .comment").all())]

        page = page_for(browser, "dana", 990, 860)
        page.goto(SITE + "/friends")
        settle(page)
        page.screenshot(path=SHOTS / "friends0.png")
        page.locator("input[type=search]").fill("עמית")
        page.wait_for_timeout(900)
        page.screenshot(path=SHOTS / "friends.png")
        marks["friends"] = boxes(page, search=".search", found=".people .person", ask=".people .person button", card=".people")

        page = page_for(browser, "dana", 990, 860)
        page.goto(SITE + "/globe")
        settle(page, 3500)
        page.screenshot(path=SHOTS / "globe.png")
        marks["globe"] = boxes(page, globe=".globe", card=".globe-page")

        page = page_for(browser, "dana", 990, 1160)
        page.goto(SITE + "/update")
        settle(page, 4000)
        page.screenshot(path=SHOTS / "update.png")
        marks["update"] = boxes(page, card=".update-page")

        page = page_for(browser, "dana", 390, 844, phone=True)
        page.goto(SITE + "/")
        settle(page, 2500)
        page.add_style_tag(content=".update-top { display: none !important; }")
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "phone.png")

        page = page_for(browser, "ori", 990, 1100)
        page.goto(SITE + "/admin")
        settle(page)
        page.screenshot(path=SHOTS / "admin.png", full_page=True)

        browser.close()
    sizes = {}
    for still in SHOTS.glob("*.png"):
        data = still.read_bytes()
        sizes[still.stem] = [int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")]
    (SHOTS / "marks.json").write_text(json.dumps({"marks": marks, "sizes": sizes}, indent=1))
    print(json.dumps(sizes))


main()
