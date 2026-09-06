"""Verify the copy-link button as a real visitor sees it: nothing injected."""
import sys
from playwright.sync_api import sync_playwright

CASES = [
    ('https://pardespedia.info/?curid=1980', True,  'ערך תוכן'),
    ('https://pardespedia.info/wiki/%D7%9E%D7%95%D7%97%D7%9E%D7%93_%D7%9B%D7%94%D7%9F', True, 'ערך אחר'),
    ('https://pardespedia.info/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99', True, 'עמוד ראשי'),
    ('https://pardespedia.info/wiki/Special:RecentChanges', False, 'דף מיוחד'),
    ('https://pardespedia.info/index.php?title=%D7%9E%D7%95%D7%97%D7%9E%D7%93_%D7%9B%D7%94%D7%9F&action=history', False, 'היסטוריה'),
    ('https://pardespedia.info/wiki/%D7%A7%D7%98%D7%92%D7%95%D7%A8%D7%99%D7%94:%D7%A2%D7%9E%D7%95%D7%AA%D7%95%D7%AA', False, 'קטגוריה'),
]

fails, errors = [], []
with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(permissions=['clipboard-read', 'clipboard-write'],
                         viewport={'width': 1200, 'height': 900})
    ctx.on('weberror', lambda e: errors.append(str(e.error)))

    for url, expect, label in CASES:
        pg = ctx.new_page()
        pg.on('pageerror', lambda e: errors.append('%s: %s' % (label, e)))
        pg.goto(url, wait_until='networkidle', timeout=60000)
        pg.wait_for_timeout(600)
        present = pg.locator('#pp-copy-link').count() > 0
        ok = present == expect
        if not ok:
            fails.append(label)
        print('%s  %-11s כפתור=%-5s (ציפייה=%s)' % ('OK ' if ok else 'FAIL', label, present, expect))

        if present and expect:
            styled = pg.eval_on_selector('#pp-copy-link',
                'el => getComputedStyle(el).borderRadius')
            pg.click('#pp-copy-link')
            pg.wait_for_timeout(400)
            copied = pg.evaluate('navigator.clipboard.readText()')
            after = pg.inner_text('#pp-copy-link')
            print('      הועתק: %s  (%d תווים)' % (copied, len(copied)))
            print('      משוב : %-12s  border-radius: %s' % (after, styled))
            if '%' in copied:
                fails.append(label + ' — כתובת מקודדת')
            if styled in ('', '0px'):
                fails.append(label + ' — ה-CSS לא נטען')
            if '✓' not in after:
                fails.append(label + ' — אין משוב')
        pg.close()

    pg = ctx.new_page()
    pg.goto('https://pardespedia.info/?curid=1980', wait_until='networkidle', timeout=60000)
    pg.wait_for_timeout(600)
    pg.screenshot(path='live_desktop.png', clip={'x': 0, 'y': 0, 'width': 1200, 'height': 430})
    pg.set_viewport_size({'width': 390, 'height': 800})
    pg.wait_for_timeout(500)
    pg.screenshot(path='live_mobile.png', clip={'x': 0, 'y': 0, 'width': 390, 'height': 430})
    pg.close()
    br.close()

if errors:
    print('\nשגיאות JS בדף:')
    for e in dict.fromkeys(errors):
        print('  ', e)

print('\n' + ('הכול עובד באתר החי' if not fails and not errors else 'בעיות: %s %s' % (fails, errors)))
sys.exit(1 if fails or errors else 0)
