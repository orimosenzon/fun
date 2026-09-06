"""Test the copy-link button against the live wiki, before it goes site-wide."""
import sys
from playwright.sync_api import sync_playwright

JS = open('/tmp/claude-1000/-home-ori-fun-vibe-coding/faef8717-883a-4b9b-8365-a9787c0ae42f/scratchpad/copylink.js', encoding='utf-8').read()
CSS = open('/tmp/claude-1000/-home-ori-fun-vibe-coding/faef8717-883a-4b9b-8365-a9787c0ae42f/scratchpad/copylink.css', encoding='utf-8').read()

CASES = [
    ('https://pardespedia.info/?curid=1980', True,  'ערך תוכן'),
    ('https://pardespedia.info/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99', True, 'עמוד ראשי'),
    ('https://pardespedia.info/wiki/Special:RecentChanges', False, 'דף מיוחד'),
    ('https://pardespedia.info/index.php?title=%D7%9E%D7%95%D7%97%D7%9E%D7%93_%D7%9B%D7%94%D7%9F&action=history', False, 'היסטוריה'),
]

fails = []
with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(permissions=['clipboard-read', 'clipboard-write'],
                         viewport={'width': 1200, 'height': 900})
    for url, expect, label in CASES:
        pg = ctx.new_page()
        pg.goto(url, wait_until='domcontentloaded', timeout=45000)
        pg.add_style_tag(content=CSS)
        pg.evaluate(JS)
        pg.wait_for_timeout(400)
        present = pg.locator('#pp-copy-link').count() > 0
        status = 'OK ' if present == expect else 'FAIL'
        if present != expect:
            fails.append(label)
        print('%s  %-12s כפתור=%s (ציפייה=%s)' % (status, label, present, expect))

        if present and expect:
            pg.click('#pp-copy-link')
            pg.wait_for_timeout(300)
            copied = pg.evaluate('navigator.clipboard.readText()')
            label_after = pg.inner_text('#pp-copy-link')
            print('      הועתק : %s' % copied)
            print('      אורך  : %d תווים   טקסט הכפתור: %s' % (len(copied), label_after))
            if '%' in copied or not copied.startswith('https://pardespedia.info/wiki/'):
                fails.append(label + ' (כתובת שגויה)')
            if '✓' not in label_after:
                fails.append(label + ' (אין משוב)')
            # does the copied URL actually resolve?
            chk = ctx.new_page()
            r = chk.goto(copied, wait_until='domcontentloaded', timeout=45000)
            print('      נטען  : HTTP %s — %s' % (r.status, chk.title().split('–')[0].strip()))
            if r.status != 200:
                fails.append(label + ' (הכתובת לא נטענת)')
            chk.close()
        pg.close()

    # visual check
    pg = ctx.new_page()
    pg.goto('https://pardespedia.info/?curid=1980', wait_until='networkidle', timeout=45000)
    pg.add_style_tag(content=CSS)
    pg.evaluate(JS)
    pg.wait_for_timeout(500)
    pg.screenshot(path='/tmp/claude-1000/-home-ori-fun-vibe-coding/faef8717-883a-4b9b-8365-a9787c0ae42f/scratchpad/btn_desktop.png', clip={'x':0,'y':0,'width':1200,'height':520})
    pg.set_viewport_size({'width': 390, 'height': 800})
    pg.wait_for_timeout(400)
    pg.screenshot(path='/tmp/claude-1000/-home-ori-fun-vibe-coding/faef8717-883a-4b9b-8365-a9787c0ae42f/scratchpad/btn_mobile.png', clip={'x':0,'y':0,'width':390,'height':520})
    pg.close()
    br.close()

print('\n' + ('כל הבדיקות עברו' if not fails else 'נכשלו: %s' % fails))
sys.exit(1 if fails else 0)
