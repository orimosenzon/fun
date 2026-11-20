# python3 tools/browser_test.py: open the game page in Chrome, start, skip the intro, pick one
# player and go on; saves reference/browser.png. Needs the server on port 8934 (run from arcade/).
import sys, time
from playwright.sync_api import sync_playwright
url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8934/mule/index.html'
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--autoplay-policy=no-user-gesture-required'])
    pg = b.new_page(viewport={'width': 1100, 'height': 900})
    errs = []
    pg.on('console', lambda m: errs.append(m.text) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(url)
    pg.wait_for_selector('#start:not([hidden])', timeout=10000)
    pg.click('#start')
    time.sleep(4)
    pg.keyboard.press('Space', delay=300); time.sleep(13)
    for _ in range(3): pg.keyboard.press('F5', delay=150); time.sleep(0.7)
    pg.keyboard.press('F7', delay=150); time.sleep(2.5)
    pg.keyboard.press('Space', delay=300); time.sleep(3)
    info = pg.evaluate('''() => { const m = mule.machine; return { frames: m.vic.frames, secs: m.cycles / m.model.hz, audio: mule.audio.ctx && mule.audio.ctx.state, vol: m.sid.reg[0x18] & 15, pc: m.cpu.pc.toString(16) }; }''')
    print(info)
    pg.screenshot(path='reference/browser.png')
    print('errors:', errs)
    b.close()
