"""Photos where they were taken (24/10/2026), in a real Chrome.

Until then every upload re-encoded photos on a canvas and the location the
phone wrote into them was lost, so the balloon spread them evenly along the
line. Now the location is read off the original (web/exif.js) and kept as
`at`. This checks the pieces:

  * Exif.locate reads a JPEG's GPS, and returns nothing for the zeroed block
    Android's picker leaves behind;
  * the likeness test that matches an original to its uploaded copy says yes
    to the same photo re-encoded as a small WebP, and no to another photo;
  * a trail whose photos carry `at` shows them on the map as round thumbnails
    at those spots, and a tap opens that very photo.

    python3 tests/test_photo_places.py

Needs Playwright with Google Chrome (channel='chrome') and Pillow.
"""
import base64, io, os, subprocess, sys, time
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)


def rational(x):
    return (round(x * 1000000), 1000000)


def jpeg(seed, gps):
    """A 1200x900 picture with structure (so two seeds differ), carrying the
    given GPS block: (lat, lng), or 'zero' for a redacted one, or None."""
    img = Image.effect_mandelbrot((1200, 900), (-2 + seed * .3, -1.2, 1 + seed * .3, 1.2), 60 + seed * 40).convert('RGB')
    exif = Image.Exif()
    exif[0x0110] = 'test'
    if gps:
        lat, lng = (0, 0) if gps == 'zero' else gps
        def dms(v):
            d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
            return (d, m, s)
        exif.get_ifd(0x8825).update({
            1: 'N', 2: dms(lat), 3: 'E', 4: dms(lng)})
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=90, exif=exif)
    return base64.b64encode(buf.getvalue()).decode()


srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(.6)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome')
        ctx = b.new_context(viewport={'width': 1280, 'height': 860})
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof Layers !== "undefined" && Layers.visibleSegments().length > 0', timeout=30000)
        page.wait_for_function('() => typeof map !== "undefined" && map && map.loaded()', timeout=30000)

        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')
        page.evaluate('''() => { window.blobOf = (b64, type) => {
            const s = atob(b64), a = new Uint8Array(s.length);
            for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
            return new Blob([a], { type: type || 'image/jpeg' }); }; }''')

        # ---- EXIF ----
        at = page.evaluate('(b) => Exif.locate(blobOf(b))', jpeg(0, (32.4731, 34.9707)))
        check('GPS read off a JPEG', at and abs(at[0] - 32.4731) < 1e-5 and abs(at[1] - 34.9707) < 1e-5, at)
        at = page.evaluate('(b) => Exif.locate(blobOf(b))', jpeg(0, 'zero'))
        check('a zeroed GPS block is no location', at is None, at)
        at = page.evaluate('(b) => Exif.locate(blobOf(b))', jpeg(0, None))
        check('no GPS block is no location', at is None, at)
        at = page.evaluate('() => Exif.locate(new Blob(["not a picture"]))')
        check('garbage is no location, and does not throw', at is None, at)

        # ---- matching an original to its upload ----
        # The upload path: shrink to 1600 WebP 0.82, then the 500px thumbnail.
        score = page.evaluate('''async ([a, b]) => {
            const enc = (blob, longest) => new Promise((done) => {
              const img = new Image();
              img.onload = () => {
                const k = Math.min(1, longest / Math.max(img.width, img.height));
                const c = document.createElement('canvas');
                c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
                c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
                c.toBlob(done, 'image/webp', 0.78);
              };
              img.src = URL.createObjectURL(blob);
            });
            const orig = blobOf(a), other = blobOf(b);
            const thumb = await enc(await enc(orig, 1600), 500);
            const s1 = await imageSignature(URL.createObjectURL(orig), false);
            const s2 = await imageSignature(URL.createObjectURL(thumb), false);
            const s3 = await imageSignature(URL.createObjectURL(other), false);
            return { same: likeness(s1, s2), other: likeness(s1, s3) };
        }''', [jpeg(0, (32.47, 34.97)), jpeg(3, None)])
        check(f'the same photo re-encoded matches ({score["same"]:.3f})', score['same'] >= 0.9, score)
        check(f'another photo does not ({score["other"]:.3f})', score['other'] < 0.9, score)

        # ---- the map ----
        # A trail with photos gets locations along its own line, in memory.
        tid = page.evaluate('''() => {
            const t = Layers.visibleSegments().find((s) => s.path && s.path.length > 10
              && (s.photos || []).filter((p) => !p.yt).length >= 3);
            const ph = t.photos.filter((p) => !p.yt);
            ph.forEach((p, k) => { p.at = t.path[Math.floor((k + .5) / ph.length * t.path.length)]; });
            return t.id;
        }''')
        n_at = page.evaluate('(id) => Layers.item(id).photos.filter((p) => p.at).length', tid)
        page.evaluate('(id) => select(id, true)', tid)
        page.wait_for_timeout(1500)
        pins = page.locator('.photo-pin.own')
        check(f'the selected trail shows its {n_at} photos on the map', pins.count() == n_at, pins.count())

        # Each pin sits where its photo says.
        off = page.evaluate('''(id) => {
            const it = Layers.item(id); let worst = 0;
            document.querySelectorAll('.photo-pin.own').forEach((n) => {
              const r = n.getBoundingClientRect(), m = map.getContainer().getBoundingClientRect();
              const x = r.left + r.width / 2 - m.left, y = r.top + r.height / 2 - m.top;
              const best = Math.min(...it.photos.filter((p) => p.at).map((p) => {
                const q = map.project([p.at[1], p.at[0]]); return Math.hypot(q.x - x, q.y - y); }));
              worst = Math.max(worst, best);
            });
            return worst;
        }''', tid)
        check('every pin is on its spot', off < 3, off)
        page.screenshot(path=os.path.join(OUT, 'shot_photo_pins.png'))

        # A tap opens that photo.
        want = page.evaluate('''(id) => {
            const it = Layers.item(id), i = it.photos.findIndex((p) => p.at);
            return { i, src: it.photos[i].full };
        }''', tid)
        pins.first.click()
        page.wait_for_timeout(400)
        lb = page.evaluate('() => ({ open: !el("lightbox").hidden, i: lbIndex })')
        check('a tap on a pin opens the lightbox on that photo', lb['open'] and lb['i'] == want['i'], (lb, want))
        page.evaluate('() => closeLightbox()')

        page.evaluate('() => deselect()')
        page.wait_for_timeout(300)
        page.evaluate('() => map.jumpTo({ zoom: 13 })')
        page.wait_for_timeout(800)
        check('nothing selected and zoomed out: no pins', page.locator('.photo-pin').count() == 0,
              page.locator('.photo-pin').count())
        first = page.evaluate('(id) => Layers.item(id).photos.find((p) => p.at).at', tid)
        page.evaluate('(a) => map.jumpTo({ center: [a[1], a[0]], zoom: 17 })', first)
        page.wait_for_timeout(800)
        check('close in, the pins come back without a selection', page.locator('.photo-pin').count() > 0,
              page.locator('.photo-pin').count())

        check('no page errors', not errors, errors)
        b.close()
finally:
    srv.terminate()

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED: {fails}'))
sys.exit(1 if fails else 0)
