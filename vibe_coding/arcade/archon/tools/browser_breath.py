# python3 tools/browser_breath.py [url]: in Chrome, bring the dragon into a fight and
# breathe fire with the keyboard. Saves reference/browser_breath.png. Server on 8933.
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
    # Dragon next to the knight in the cursor's row (board at $BD7C).
    pg.evaluate('() => { const r = archon.machine.ram, B = 0xbd7c; r[B + 5 * 9 + 8] = 0x80; r[B + 4 * 9 + 2] = 0x1d; }')
    def hold(key, secs): pg.keyboard.down(key); time.sleep(secs); pg.keyboard.up(key); time.sleep(0.4)
    x = lambda s: pg.evaluate(f'() => archon.machine.vic.reg[{s * 2}]')
    def move_to(sprite, want):
        pg.keyboard.down('ArrowRight')
        for _ in range(400):
            if x(sprite) >= want: break
            time.sleep(0.005)
        pg.keyboard.up('ArrowRight'); time.sleep(0.5)
    move_to(1, 96); hold('Space', 0.2); move_to(0, 124); hold('Space', 0.2)
    time.sleep(6)
    print('armed:', pg.evaluate('() => !!archon.breath'), pg.evaluate('() => archon.breath && archon.breath.state'))
    pg.keyboard.down('ArrowLeft'); pg.keyboard.press('KeyX', delay=60); pg.keyboard.up('ArrowLeft')
    time.sleep(0.6)
    print('breathing:', pg.evaluate('() => archon.breath.state.active'), 'audio', pg.evaluate('() => archon.audio.ctx.state'))
    pg.screenshot(path='reference/browser_breath.png')
    time.sleep(1.5)
    print('after:', pg.evaluate('() => ({ active: archon.breath.state.active, wait: archon.breath.state.wait, burns: archon.breath.state.burns })'))
    print('errors:', errs)
    b.close()
