"""Segmentation lab — visual comparison of line-cutting approaches.

Usage:
    python3 segment_lab/lab.py <file> [page_num]

Writes to segment_lab/out/:
    <stem>_pre.png        rotated+deskewed page (what segmentation sees)
    <stem>_baseline.png   current 1D-projection cuts (needs N; uses v2 count)
    <stem>_v2.png         new 2D blob-based line boxes
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


# ---------------------------------------------------------------- loading

def load_page(path: str, page: int = 0) -> Image.Image:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    pages = app.file_to_page_images(data, ext)
    return pages[page]


def preprocess(img: Image.Image) -> Image.Image:
    img, rot = app.auto_rotate(img)
    if img.width > 1600:
        r = 1600 / img.width
        img = img.resize((1600, int(img.height * r)), Image.LANCZOS)
    img, angle = slope_deskew(img)
    print(f"  preprocess: rotate={rot} slope_deskew={angle:.1f}")
    return img


# ---------------------------------------------------------------- v2 detector

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
    # fill holes (ink inside the page punches holes in the paper mask)
    mask = ndimage.binary_fill_holes(mask)
    # erode a bit so the page-edge shadow/fold stays out
    it = max(3, min(gray.shape) // 200)
    mask = ndimage.binary_erosion(mask, iterations=it)
    return mask


def sauvola(gray: np.ndarray, window: int = 41, k: float = 0.25) -> np.ndarray:
    """Adaptive binarization → True where ink. Handles shadows / faint pencil."""
    g = gray.astype(np.float32)
    mean = ndimage.uniform_filter(g, window)
    sqmean = ndimage.uniform_filter(g * g, window)
    var = np.maximum(sqmean - mean * mean, 0)
    std = np.sqrt(var)
    R = 128.0
    thresh = mean * (1 + k * (std / R - 1))
    return g < thresh


def remove_ruling(ink: np.ndarray, page_w: int):
    """Erase printed notebook ruling; return (clean ink, ruling mask).

    Ruling = pixels in long thin horizontal runs. Handwriting that crosses
    the ruling survives: pixels that also sit in a tall vertical run (a
    letter stroke) are exempt. The ruling mask is kept because its x-extent
    marks the student page's writing column — a free, reliable page anchor.
    """
    long_h = ndimage.binary_opening(
        ink, structure=np.ones((1, 45), dtype=bool)
    )
    # A printed rule lives on a row where long runs cover much of the page
    # width; a student's underline / long pen stroke doesn't. Without this
    # row gate, thin-pen handwriting (same stroke width as the rule) that
    # sits on the rule gets erased with it.
    cols = np.where(long_h.any(axis=0))[0]
    span = (cols[-1] - cols[0]) if cols.size else ink.shape[1]
    row_cov = long_h.sum(axis=1)
    rule_rows = row_cov > 0.30 * max(span, 1)
    rule_rows = ndimage.binary_dilation(rule_rows, iterations=2)
    tall = ndimage.binary_opening(ink, structure=np.ones((6, 1), dtype=bool))
    ruling = long_h & rule_rows[:, None] & ~tall
    clean = ink & ~ruling
    return clean, ruling, tall


def estimate_slope(comps, max_slope: float = 0.45) -> float:
    """Dominant text-line slope (dy/dx) via histogram-sharpness voting.

    Project component centres onto y' = cy - s*cx for candidate slopes; the
    slope that makes the projected histogram sharpest (mass concentrated in
    few bins = level lines) wins. Component-based, so it sees through page
    texture and works far beyond the ±10° brute-force image-rotation cap.
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


def _ink_and_comps(img: Image.Image):
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    pm = paper_mask(img)
    ink, ruling, tall = remove_ruling(sauvola(gray) & pm, gray.shape[1])

    # Ruled paper hands us the writing column for free: the printed lines
    # span exactly the student page. Clip ink to their x-range so an
    # adjacent book page / spiral binding can't fuse into the text rows.
    # Only when the ruling is *extensive* — several rules spread over much
    # of the page; a lone underline or border stroke must not clip anything.
    rule_row_idx = np.where(ruling.any(axis=1))[0]
    n_rules = 0
    if rule_row_idx.size:
        n_rules = 1 + int((np.diff(rule_row_idx) > 5).sum())
    rules_extensive = (
        n_rules >= 5
        and (rule_row_idx[-1] - rule_row_idx[0]) > 0.4 * ink.shape[0]
    )
    if rules_extensive:
        col_counts = ruling.sum(axis=0)
        # robust extent: ignore stray long strokes elsewhere
        strong = np.where(col_counts >= max(2, 0.25 * col_counts.max()))[0]
        if strong.size > 200:
            x0 = max(0, int(strong[0]) - 10)
            x1 = min(ink.shape[1], int(strong[-1]) + 10)
            clipped = np.zeros_like(ink)
            clipped[:, x0:x1] = ink[:, x0:x1]
            ink = clipped
    comps, lbl = _components(ink)
    aux = {"rules_extensive": rules_extensive, "tall": tall & (ink > 0)}
    return ink, comps, lbl, aux


def slope_deskew(img: Image.Image) -> tuple[Image.Image, float]:
    """Level the text lines by the component-vote slope estimate."""
    _, comps, _, _ = _ink_and_comps(img)
    s = estimate_slope(comps)
    angle = float(np.degrees(np.arctan(s)))
    if abs(angle) < 0.3:
        return img, 0.0
    # y' = y - s*x becomes level after rotating the image by -angle with
    # PIL's convention (positive = counter-clockwise in screen coords).
    out = img.rotate(-angle, resample=Image.BILINEAR, expand=True,
                     fillcolor=(255, 255, 255))
    return out, angle


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


def _components(ink: np.ndarray, min_px: int = 12):
    """Connected components → list of (x0, y0, x1, y1, cx, cy, mass).

    Also returns the label array so callers can slice oversized components.
    """
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


def _slice_tall_component(ink, lbl, c, pitch):
    """Cut one over-tall component into line-high pseudo-components.

    Slices along the valleys of the component's own row profile. Each slice
    gets its own centroid and x-extent, so a frame contributes thin strips
    while bridged text contributes proper per-line pieces.
    """
    x0, y0, x1, y1, _, _, _, comp_id = c
    mask = lbl[y0:y1, x0:x1] == comp_id
    prof = mask.sum(axis=1).astype(np.float32)
    prof_s = np.convolve(prof, np.ones(9) / 9, mode="same")
    bh = y1 - y0
    n_cuts = max(1, int(round(bh / pitch)) - 1)
    # deepest valleys, pitch-separated, away from the edges
    lo, hi = int(pitch * 0.4), bh - int(pitch * 0.4)
    cuts: list[int] = []
    if hi > lo:
        for i in sorted(range(lo, hi), key=lambda i: prof_s[i]):
            if all(abs(i - p) >= 0.7 * pitch for p in cuts):
                cuts.append(i)
            if len(cuts) == n_cuts:
                break
    edges = [0] + sorted(cuts) + [bh]
    out = []
    ys, xs = np.nonzero(mask)
    for a, b in zip(edges, edges[1:]):
        sel = (ys >= a) & (ys < b)
        if sel.sum() < 12:
            continue
        sy, sx = ys[sel], xs[sel]
        out.append((
            x0 + int(sx.min()), y0 + int(sy.min()),
            x0 + int(sx.max()) + 1, y0 + int(sy.max()) + 1,
            x0 + float(sx.mean()), y0 + float(sy.mean()),
            float(sel.sum()), comp_id,
        ))
    return out


def detect_lines_v2(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return line boxes (x0, y0, x1, y1), top-to-bottom.

    binarize → paper mask → drop ruling/specks → connected components →
    *line tracking*: walk components left-to-right, assign each to the line
    whose locally-tracked y it continues. The local estimate follows slope
    and waviness that a global horizontal projection smears away.
    """
    h, w = img.height, img.width
    ink, comps, _lbl, aux = _ink_and_comps(img)
    if not comps:
        return []
    # letter ink = pixels of components at least letter-high. Ruling dashes
    # that slipped past remove_ruling are flat components; thin-pen letters
    # still stand several px tall, so this keeps them (a vertical-opening
    # mask would not — a 1px-wide slanted stroke has no 6px vertical run).
    letter_ids = {c[7] for c in comps if (c[3] - c[1]) >= 6}
    lut = np.zeros(int(_lbl.max()) + 1, dtype=bool)
    lut[list(letter_ids)] = True
    letter_ink = lut[_lbl]

    # pitch from letter ink only: on a page full of empty ruled lines the
    # surviving dashes dominate the raw autocorrelation with a wrong period
    pitch_src = letter_ink if letter_ink.sum() > 0.2 * ink.sum() else ink
    pitch = _estimate_pitch(pitch_src)

    # 1D profile over the *cleaned* ink. The classic failure modes of a
    # projection profile (margins, background objects, ruling, slope) were
    # all removed upstream, which is what makes it reliable here.
    prof = ink.sum(axis=1).astype(np.float32)
    win = max(9, (int(pitch * 0.3) | 1))
    prof = np.convolve(prof, np.ones(win) / win, mode="same")
    floor = prof.max() * 0.04
    prof[prof < floor] = 0.0

    # free peak picking (no forced count): all local maxima, then greedy
    # prominence-ordered selection with a min spacing under the pitch, so
    # tightly-packed small writing isn't merged away
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

    # noise filter: a real text row carries real ink and real height —
    # page-edge shadows binarize into long flat slivers
    masses = sorted(r[4] for r in rows)
    if not masses:
        return []
    med_mass = masses[len(masses) // 2]
    min_h = max(12, 0.3 * pitch)
    out = [
        r for r in rows
        if r[4] >= max(80, 0.10 * med_mass) and (r[3] - r[1]) >= min_h
    ]

    # phantom-row filter: an empty ruled line whose printed rule survived as
    # dashes has ink but almost no letter-height ink; real text always does
    out = [
        r for r in out
        if int(letter_ink[r[1]:r[3], r[0]:r[2]].sum()) >= 0.25 * r[4]
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

    return [(r[0], r[1], r[2], r[3]) for r in out]


# ---------------------------------------------------------------- rendering

def draw_baseline(img: Image.Image, n: int) -> Image.Image:
    segs = app.segment_into_lines(img, n)
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for i, (a, b) in enumerate(segs):
        d.line([(0, a), (out.width, a)], fill=(255, 0, 0), width=3)
        d.text((10, (a + b) // 2), str(i + 1), fill=(255, 0, 0))
    if segs:
        d.line([(0, segs[-1][1]), (out.width, segs[-1][1])], fill=(255, 0, 0), width=3)
    return out


def draw_boxes(img: Image.Image, boxes) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        d.rectangle([x0, y0, x1, y1], outline=(0, 140, 255), width=3)
        d.text((max(0, x0 - 28), (y0 + y1) // 2 - 8), str(i + 1), fill=(255, 0, 0))
    return out


def run(path: str, page: int = 0, force_rot: int | None = None):
    stem = Path(path).stem.replace(" ", "_") + (f"_p{page}" if page else "")
    print(f"== {stem}")
    img = load_page(path, page)
    if force_rot:
        img = img.rotate(-force_rot, expand=True, fillcolor=(255, 255, 255))
    img = preprocess(img)
    img.save(OUT / f"{stem}_pre.png")
    boxes = detect_lines_v2(img)
    print(f"  v2: {len(boxes)} lines")
    draw_boxes(img, boxes).save(OUT / f"{stem}_v2.png")
    draw_baseline(img, len(boxes)).save(OUT / f"{stem}_baseline.png")


if __name__ == "__main__":
    run(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 0,
        int(sys.argv[3]) if len(sys.argv) > 3 else None,
    )
