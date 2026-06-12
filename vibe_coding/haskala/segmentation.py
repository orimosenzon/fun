"""Line segmentation for handwritten exercise pages.

Replaces the old writing_region/projection coupling with a self-contained
detector + a numbered-overlay protocol:

1. `slope_deskew` levels the text by voting on connected-component centres
   (works to ±24°, far beyond brute-force image rotation, and is immune to
   the long pen strokes that used to fool profile-std deskew).
2. `detect_lines` returns candidate line boxes. It is allowed to
   over-detect: empty ruled lines or desk clutter may get boxes.
3. `draw_overlay` numbers the boxes on the page sent to OCR; the model maps
   every transcribed line to a box number. Bad boxes are simply never
   referenced — over-detection costs nothing, and the off-by-one global
   shifts of the old architecture cannot happen by construction.

All detection runs on a ≤1600px-wide copy; returned boxes are in the
coordinates of the image passed in.
"""
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

log = logging.getLogger("haskala")

WORK_WIDTH = 1600


# --------------------------------------------------------------- ink layer

def paper_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of the paper region.

    Brightness alone can't separate a white page from a light wooden desk
    (their histograms overlap), but the desk/background is *coloured* while
    paper is grey — so the cut is low saturation AND reasonably bright,
    then the connected blob that dominates the image-centre box.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    p90 = np.percentile(gray, 90)
    paper = (sat < 0.18) & (gray > 0.55 * p90)
    lbl, n = ndimage.label(paper)
    if n == 0:
        return np.ones(gray.shape, dtype=bool)
    h, w = gray.shape
    centre = lbl[h // 3:2 * h // 3, w // 3:2 * w // 3]
    vals = centre[centre > 0]
    if vals.size:
        centre_lbl = int(np.bincount(vals).argmax())
    else:
        sizes = ndimage.sum(paper, lbl, range(1, n + 1))
        centre_lbl = 1 + int(np.argmax(sizes))
    mask = lbl == centre_lbl
    mask = ndimage.binary_fill_holes(mask)
    # erode a bit so the page-edge shadow/fold stays out
    mask = ndimage.binary_erosion(mask, iterations=max(3, min(gray.shape) // 200))
    return mask


def sauvola(gray: np.ndarray, window: int = 41, k: float = 0.25) -> np.ndarray:
    """Adaptive binarization → True where ink. Handles shadows/faint pencil."""
    g = gray.astype(np.float32)
    mean = ndimage.uniform_filter(g, window)
    sqmean = ndimage.uniform_filter(g * g, window)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    thresh = mean * (1 + k * (std / 128.0 - 1))
    return g < thresh


def remove_ruling(ink: np.ndarray):
    """Erase printed notebook ruling; return (clean ink, ruling mask).

    Ruling = pixels in long thin horizontal runs, on rows where such runs
    cover much of the page width (a student's underline doesn't). Pixels
    that also sit in a tall vertical run (a letter stroke crossing the
    rule) are exempt, so thin-pen handwriting survives.
    """
    long_h = ndimage.binary_opening(ink, structure=np.ones((1, 45), dtype=bool))
    cols = np.where(long_h.any(axis=0))[0]
    span = (cols[-1] - cols[0]) if cols.size else ink.shape[1]
    row_cov = long_h.sum(axis=1)
    rule_rows = row_cov > 0.30 * max(span, 1)
    rule_rows = ndimage.binary_dilation(rule_rows, iterations=2)
    tall = ndimage.binary_opening(ink, structure=np.ones((6, 1), dtype=bool))
    ruling = long_h & rule_rows[:, None] & ~tall
    return ink & ~ruling, ruling


def _components(ink: np.ndarray, min_px: int = 12):
    """Connected components → [(x0, y0, x1, y1, cx, cy, mass, label_id)]."""
    lbl, n = ndimage.label(ink)
    if not n:
        return [], lbl
    objs = ndimage.find_objects(lbl)
    masses = ndimage.sum(ink, lbl, range(1, n + 1))
    cys, cxs = zip(*ndimage.center_of_mass(ink, lbl, range(1, n + 1)))
    out = []
    for i, sl in enumerate(objs):
        if masses[i] < min_px:
            continue
        out.append((
            sl[1].start, sl[0].start, sl[1].stop, sl[0].stop,
            float(cxs[i]), float(cys[i]), float(masses[i]), i + 1,
        ))
    return out, lbl


def _ink_and_comps(img: Image.Image):
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    ink, ruling = remove_ruling(sauvola(gray) & paper_mask(img))

    # Ruled paper hands us the writing column for free: the printed lines
    # span exactly the student page. Clip ink to their x-range so an
    # adjacent book page / spiral binding can't fuse into the text rows.
    # Only when the ruling is *extensive* — several rules over much of the
    # page; a lone underline or border stroke must not clip anything.
    rule_row_idx = np.where(ruling.any(axis=1))[0]
    n_rules = 0
    if rule_row_idx.size:
        n_rules = 1 + int((np.diff(rule_row_idx) > 5).sum())
    if (
        n_rules >= 5
        and (rule_row_idx[-1] - rule_row_idx[0]) > 0.4 * ink.shape[0]
    ):
        col_counts = ruling.sum(axis=0)
        strong = np.where(col_counts >= max(2, 0.25 * col_counts.max()))[0]
        if strong.size > 200:
            x0 = max(0, int(strong[0]) - 10)
            x1 = min(ink.shape[1], int(strong[-1]) + 10)
            clipped = np.zeros_like(ink)
            clipped[:, x0:x1] = ink[:, x0:x1]
            ink = clipped
    comps, lbl = _components(ink)
    return ink, comps, lbl


# ------------------------------------------------------------------ skew

def estimate_slope(comps, max_slope: float = 0.45) -> float:
    """Dominant text-line slope (dy/dx) via histogram-sharpness voting.

    Project component centres onto y' = cy - s*cx for candidate slopes; the
    slope that makes the projected histogram sharpest (mass concentrated in
    few bins = level lines) wins. Component-based, so it sees through page
    texture and works far beyond the ±10° brute-force rotation cap.
    """
    if len(comps) < 8:
        return 0.0
    cxs = np.array([c[4] for c in comps])
    cys = np.array([c[5] for c in comps])
    masses = np.array([c[6] for c in comps])
    wts = np.minimum(masses, np.percentile(masses, 90))

    def score(s: float) -> float:
        proj = cys - s * cxs
        hist, _ = np.histogram(
            proj, bins=max(10, int(np.ptp(proj) / 12) or 10), weights=wts
        )
        return float((hist.astype(np.float64) ** 2).sum())

    best_s = 0.0
    best = score(0.0)
    for s in np.arange(-max_slope, max_slope + 1e-9, 0.02):
        sc = score(float(s))
        if sc > best:
            best, best_s = sc, float(s)
    for s in np.arange(best_s - 0.02, best_s + 0.0201, 0.002):
        sc = score(float(s))
        if sc > best:
            best, best_s = sc, float(s)
    return best_s


def _work_copy(img: Image.Image) -> tuple[Image.Image, float]:
    """Detection-scale copy + the factor mapping work coords → img coords."""
    if img.width <= WORK_WIDTH:
        return img, 1.0
    r = WORK_WIDTH / img.width
    return img.resize((WORK_WIDTH, int(img.height * r)), Image.LANCZOS), 1 / r


def estimate_skew_angle(img: Image.Image) -> float:
    """Degrees to rotate (PIL convention) so the text lines come out level."""
    work, _ = _work_copy(img)
    _, comps, _ = _ink_and_comps(work)
    s = estimate_slope(comps)
    return float(np.degrees(np.arctan(s)))


def slope_deskew(img: Image.Image) -> tuple[Image.Image, float]:
    """Level the text lines by the component-vote slope estimate."""
    angle = estimate_skew_angle(img)
    if abs(angle) < 0.3:
        return img, 0.0
    out = img.rotate(
        -angle, resample=Image.BILINEAR, expand=True, fillcolor=(255, 255, 255)
    )
    return out, angle


# ----------------------------------------------------------------- lines

def _estimate_pitch(ink: np.ndarray) -> int:
    """Dominant line pitch via autocorrelation of the y ink-profile."""
    prof = ink.sum(axis=1).astype(np.float32)
    prof = np.convolve(prof, np.ones(15) / 15, mode="same")
    centred = prof - prof.mean()
    ac = np.correlate(centred, centred, "full")[len(centred) - 1:]
    lo, hi = 25, min(220, len(ac) - 1)
    if hi <= lo:
        return 60
    return lo + int(np.argmax(ac[lo:hi]))


def _prom(prof: np.ndarray, i: int) -> float:
    """Topographic prominence of peak i in a 1D profile."""
    left_min = right_min = prof[i]
    j = i
    while j > 0 and prof[j] <= prof[i]:
        left_min = min(left_min, prof[j])
        j -= 1
    j = i
    while j < len(prof) - 1 and prof[j] <= prof[i]:
        right_min = min(right_min, prof[j])
        j += 1
    return float(prof[i] - max(left_min, right_min))


def detect_lines(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Candidate line boxes (x0, y0, x1, y1) in img coords, top-to-bottom.

    Free peak-picking on the y-profile of the cleaned ink — no forced line
    count. May over-detect (empty ruled rows, teacher marks); the numbered
    overlay makes that harmless.
    """
    work, scale = _work_copy(img)
    h, w = work.height, work.width
    ink, comps, lbl = _ink_and_comps(work)
    if not comps:
        return []

    # letter ink = pixels of components at least letter-high. Ruling dashes
    # that slipped past remove_ruling are flat components; thin-pen letters
    # still stand several px tall, so this keeps them (a vertical-opening
    # mask would not — a 1px-wide slanted stroke has no 6px vertical run).
    letter_ids = [c[7] for c in comps if (c[3] - c[1]) >= 6]
    lut = np.zeros(int(lbl.max()) + 1, dtype=bool)
    lut[letter_ids] = True
    letter_ink = lut[lbl]

    # pitch from letter ink only: on a page full of empty ruled lines the
    # surviving dashes dominate the raw autocorrelation with a wrong period
    pitch_src = letter_ink if letter_ink.sum() > 0.2 * ink.sum() else ink
    pitch = _estimate_pitch(pitch_src)

    prof = ink.sum(axis=1).astype(np.float32)
    win = max(9, (int(pitch * 0.3) | 1))
    prof = np.convolve(prof, np.ones(win) / win, mode="same")
    floor = prof.max() * 0.04
    prof[prof < floor] = 0.0

    cands = [
        i for i in range(1, h - 1)
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0
    ]
    cands.sort(key=lambda i: -_prom(prof, i))
    min_dist = max(14, int(pitch * 0.45))
    peaks: list[int] = []
    for c in cands:
        if all(abs(c - p) >= min_dist for p in peaks):
            peaks.append(c)
    peaks.sort()
    if not peaks:
        return []

    # cut at the deepest point between consecutive peaks; edge bands extend
    # to where the profile dies out (capped at one pitch from the peak)
    bounds = []
    first_fade = peaks[0]
    while first_fade > 0 and prof[first_fade] > 0 and peaks[0] - first_fade < pitch:
        first_fade -= 1
    bounds.append(first_fade)
    for a, b in zip(peaks, peaks[1:]):
        bounds.append(a + int(np.argmin(prof[a:b])))
    last_fade = peaks[-1]
    while last_fade < h - 1 and prof[last_fade] > 0 and last_fade - peaks[-1] < pitch:
        last_fade += 1
    bounds.append(last_fade)

    rows = []
    for i in range(len(peaks)):
        y0, y1 = bounds[i], bounds[i + 1]
        band = ink[y0:y1]
        mass = int(band.sum())
        xs = np.where(band.any(axis=0))[0]
        if not xs.size:
            continue
        rows.append([int(xs[0]), y0, int(xs[-1]) + 1, y1, mass])
    if not rows:
        return []

    # noise filters: a real text row carries real ink, real height (page-
    # edge shadows binarize into long flat slivers) and letter-height ink
    # (an empty ruled line whose rule survived as dashes has none)
    masses = sorted(r[4] for r in rows)
    med_mass = masses[len(masses) // 2]
    min_h = max(12, 0.3 * pitch)
    out = [
        r for r in rows
        if r[4] >= max(80, 0.10 * med_mass)
        and (r[3] - r[1]) >= min_h
        and int(letter_ink[r[1]:r[3], r[0]:r[2]].sum()) >= 0.25 * r[4]
    ]

    # cluster filter: split rows into y-clusters at big empty gaps; remote
    # AND light clusters are background junk (desk clutter, book spines),
    # while a real header/footer paragraph carries real mass
    if len(out) >= 2:
        clusters: list[list] = [[out[0]]]
        for r in out[1:]:
            if r[1] - clusters[-1][-1][3] > 3 * pitch:
                clusters.append([r])
            else:
                clusters[-1].append(r)
        body_mass = max(sum(r[4] for r in c) for c in clusters)
        out = [
            r for c in clusters
            if sum(r2[4] for r2 in c) >= 0.05 * body_mass
            for r in c
        ]

    log.info(
        "detect_lines: %d boxes pitch=%d work=%dx%d", len(out), pitch, w, h
    )
    return [
        (
            int(r[0] * scale), int(r[1] * scale),
            int(r[2] * scale), int(r[3] * scale),
        )
        for r in out
    ]


# ---------------------------------------------------------------- overlay

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(img: Image.Image, boxes) -> Image.Image:
    """The page with numbered boxes drawn on it — what OCR actually sees."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 800)
    fsize = max(22, img.width // 60)
    font = _font(fsize)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        d.rectangle([x0, y0, x1, y1], outline=(40, 110, 255), width=lw)
        label = str(i + 1)
        tw = int(d.textlength(label, font=font))
        tx = max(2, x0 - tw - 12)
        ty = (y0 + y1) // 2 - fsize // 2
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + fsize + 2],
                    fill=(255, 255, 255))
        d.text((tx, ty), label, fill=(0, 150, 0), font=font)
    return out


def crop_box(img: Image.Image, box, pad: int = 12) -> Image.Image:
    x0, y0, x1, y1 = box
    return img.crop((
        max(0, x0 - pad), max(0, y0 - pad),
        min(img.width, x1 + pad), min(img.height, y1 + pad),
    ))
