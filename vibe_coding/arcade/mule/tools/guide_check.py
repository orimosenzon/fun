# python3 tools/guide_check.py: open how.html in Chrome, photograph every walkthrough step that has
# markers (to check they sit on the right things) and the whole page. Output in reference/guide_check/.
import os, sys
from playwright.sync_api import sync_playwright
from PIL import Image
out = os.path.join(os.path.dirname(__file__), '..', 'reference', 'guide_check'); os.makedirs(out, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome')
    pg = b.new_page(viewport={'width': 1200, 'height': 900})
    errs = []
    pg.on('console', lambda m: errs.append(m.text) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('http://127.0.0.1:8934/mule/how.html')
    n = pg.evaluate('STEPS.length'); shots = []
    for i in range(n):
        pg.evaluate(f'showStep({i})'); pg.wait_for_timeout(120)
        if pg.evaluate(f'(STEPS[{i}].marks || []).length'):
            f = os.path.join(out, f'step{i + 1:02d}.png'); pg.locator('#shot').screenshot(path=f); shots.append(f)
    ims = [Image.open(f) for f in shots]; w, h = ims[0].size; cols = 4
    for k in range(0, len(ims), 8):
        part = ims[k:k + 8]; sheet = Image.new('RGB', (cols * w, ((len(part) + cols - 1) // cols) * h))
        for j, im in enumerate(part): sheet.paste(im, ((j % cols) * w, (j // cols) * h))
        sheet.save(os.path.join(out, f'sheet{k // 8}.png'))
    pg.evaluate('showStep(11)'); pg.screenshot(path=os.path.join(out, 'page.png'), full_page=True)
    pg.set_viewport_size({'width': 390, 'height': 800}); pg.screenshot(path=os.path.join(out, 'phone.png'), full_page=True)
    print('steps', n, 'marked', [os.path.basename(f) for f in shots], 'errors:', errs)
    b.close()
