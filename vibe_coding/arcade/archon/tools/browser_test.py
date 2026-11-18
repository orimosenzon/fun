# python3 tools/browser_test.py: open the page in Chrome, start, play a little,
# save reference/browser.png. Needs the local server on port 8933.
import sys, time
from playwright.sync_api import sync_playwright
url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8933/index.html'
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--autoplay-policy=no-user-gesture-required'])
    pg = b.new_page(viewport={'width': 1100, 'height': 900})
    errs = []
    pg.on('console', lambda m: errs.append(m.text) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(url)
    pg.wait_for_selector('#start:not([hidden])', timeout=10000)
    pg.click('#start')
    time.sleep(13)
    pg.keyboard.press('Escape', delay=150); time.sleep(3)
    pg.keyboard.press('F7', delay=150); time.sleep(4)
    pg.keyboard.down('ArrowRight'); time.sleep(0.5); pg.keyboard.up('ArrowRight'); time.sleep(0.5)
    info = pg.evaluate('''() => { const m = archon.machine; return { frames: m.vic.frames, secs: m.cycles / m.model.hz, audio: archon.audio.ctx && archon.audio.ctx.state, rate: archon.audio.sampleRate, cursorX: m.vic.reg[2], vol: m.sid.reg[0x18] & 15 }; }''')
    print(info)
    pg.screenshot(path='reference/browser.png')
    print('errors:', errs)
    b.close()
