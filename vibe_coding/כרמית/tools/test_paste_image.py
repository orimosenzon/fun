"""Drives admin.html with stubbed Firebase to test pasting a picture (Ctrl+V) into the riddle and letter
dialogs: the first paste fills the riddle picture, the next fills the answer picture when that one shows,
a paste over a zone goes to that zone, a text paste stays text, and with no dialog open nothing happens.
Usage: python3 tools/test_paste_image.py http://localhost:8767/vibe_coding/כרמית/app/ [screenshot-dir]"""
import base64
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
PNG = base64.b64encode((Path(__file__).resolve().parent.parent / "app" / "favicon.png").read_bytes()).decode()

# Fires a paste event carrying a PNG file, the way the browser does after "copy image".
PASTE = """(png) => {
  const bytes = Uint8Array.from(atob(png), (c) => c.charCodeAt(0));
  const dt = new DataTransfer();
  dt.items.add(new File([bytes], 'image.png', { type: 'image/png' }));
  document.activeElement.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
}"""

with sync_playwright() as p:
    b = p.chromium.launch()
    a = b.new_context(viewport={"width": 1200, "height": 1000}).new_page()
    stub(a)
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")

    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    check("Ctrl+V" in a.inner_text("#image-drop"), "the riddle picture zone mentions pasting")
    a.mouse.move(5, 5)
    a.evaluate(PASTE, PNG)
    a.wait_for_selector("#image-preview:not(.hidden)")
    check(a.locator("#answer-preview.hidden").count() == 1, "first paste: the riddle picture, not the answer")

    a.check('input[name="answer-type"][value="image"]')
    a.mouse.move(5, 5)
    a.evaluate(PASTE, PNG)
    a.wait_for_selector("#answer-preview:not(.hidden)")
    check(True, "second paste: the empty answer picture")

    # Pasting over a filled zone replaces that one, whatever is empty.
    a.evaluate("document.getElementById('image-preview').src = ''")
    a.hover("#image-drop")
    a.evaluate(PASTE, PNG)
    a.wait_for_function("document.getElementById('image-preview').src.startsWith('data:image')")
    check(True, "paste over a zone fills that zone")

    # Pasting text into the question box still pastes text.
    a.fill("#question", "")
    a.focus("#question")
    a.evaluate("""() => { const dt = new DataTransfer(); dt.setData('text/plain', 'x');
      document.activeElement.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true })); }""")
    check(a.locator("#image-drop.filled").count() == 1, "a text paste leaves the pictures alone")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/paste-riddle.png")
    a.click("#dialog-close")
    a.wait_for_selector("#riddle-dialog", state="hidden")

    # No dialog open: a pasted picture does nothing.
    a.evaluate(PASTE, PNG)
    check(a.locator("#letter-preview:not(.hidden)").count() == 0, "no dialog: paste ignored")
    b.close()
print("ok")
