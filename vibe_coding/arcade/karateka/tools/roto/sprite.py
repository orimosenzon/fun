"""Rotoscoping, step 2: turn an extracted frame into a draft sprite on the Apple II pixel grid.

Reads reference/roto/work/<name>/NN.png + NN_cls.png and prints a text grid:
  .  nothing      #  white (clothes)     s  skin      H  hair      o  black inside the figure (belt)
  c  a crease or shadow the camera saw inside the clothes (a hint for where to draw a line)
The draft is a starting point: the final art is edited by hand in tools/roto/art/.

usage: venv/bin/python tools/roto/sprite.py <name> <frame> <scale> <ax,ay> [--grid W,H] [--at gx,gy] [--png out.png]
  scale   video pixels per sprite pixel
  ax,ay   a point in the crop (usually where the supporting foot meets the floor) ...
  --at    ... and the grid cell it lands on (default 37,47: under the fighter, on the ground row)
"""
import os, sys
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'reference', 'roto')
PALETTE = [(0, 0, 0), (255, 255, 255), (255, 68, 253), (20, 245, 60), (20, 207, 253), (255, 106, 60)]


def opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def draft(name, frame, scale, ax, ay, W=84, H=48, gx=37, gy=47):
    d = os.path.join(ROOT, 'work', name)
    img = cv2.imread(os.path.join(d, f'{frame:02d}.png')); cls = cv2.imread(os.path.join(d, f'{frame:02d}_cls.png'), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    person = (cls > 0).astype(np.uint8)
    person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(person)           # keep the biggest blob
    if n > 1: person = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ff = person.copy(); cv2.floodFill(ff, None, (0, 0), 2); person[ff == 0] = 1  # fill holes
    hair = (cls == 1) & (person > 0); skin = np.isin(cls, (2, 3)) & (person > 0)
    near_hair = cv2.dilate(hair.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    dark = (gray < 75) & (person > 0) & ~near_hair & ~skin
    local = cv2.blur(np.where(person > 0, gray, 0), (21, 21)) / np.maximum(cv2.blur(person.astype(np.float32), (21, 21)), 1e-3)
    crease = (gray < local - 22) & (person > 0) & ~dark & ~near_hair & ~skin

    def cell(mask):  # area average of a mask over every sprite pixel
        m = np.zeros((H, W), np.float32)
        for r in range(H):
            y0, y1 = ay + (r - gy) * scale, ay + (r - gy + 1) * scale
            for c in range(W):
                x0, x1 = ax + (c - gx) * scale, ax + (c - gx + 1) * scale
                a = mask[max(0, int(round(y0))):max(0, int(round(y1))), max(0, int(round(x0))):max(0, int(round(x1)))]
                m[r, c] = a.mean() if a.size else 0
        return m
    P, Hh, S, D, C = (cell(x.astype(np.float32)) for x in (person, hair, skin, dark, crease))
    g = [['.'] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if P[r, c] < 0.5: continue
            p = P[r, c]
            g[r][c] = 'H' if Hh[r, c] / p > 0.5 else 's' if S[r, c] / p > 0.45 else 'o' if D[r, c] / p > 0.45 else 'c' if C[r, c] / p > 0.5 else '#'
    return [''.join(r) for r in g]


def to_bits(rows, anchor=21):
    """text grid -> rows of 0/1/None (None = transparent), the way the game will show it"""
    out = []
    for row in rows:
        out.append([None if ch == '.' else 0 if ch in 'oc' else ((c - anchor) & 1 ^ 1) if ch == 's' else 1 for c, ch in enumerate(row)])
    return out


def render(rows, scale=8, bg=0, x0=0):
    """Apple II hi-res colours (high bit set), on a black or lit background"""
    H, W = len(rows), len(rows[0]); bits = to_bits(rows)
    img = np.zeros((H, W, 3), np.uint8)
    for r in range(H):
        b = [bg if v is None else v for v in bits[r]]
        for c in range(W):
            x = c + x0; l = b[c - 1] if c else 0; rr = b[c + 1] if c + 1 < W else 0
            if b[c]: col = 1 if (l or rr) else (5 if x & 1 else 4)
            else: col = (4 if x & 1 else 5) if (l and rr) else 0
            img[r, c] = PALETTE[col][::-1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    name, frame, scale, a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    ax, ay = map(float, a.split(',')); W, H = map(int, opt('--grid', '84,48').split(',')); gx, gy = map(int, opt('--at', '37,47').split(','))
    rows = draft(name, frame, scale, ax, ay, W, H, gx, gy)
    print('\n'.join(rows))
    png = opt('--png', None)
    if png:
        src = cv2.imread(os.path.join(ROOT, 'work', name, f'{frame:02d}.png'))
        x0, y0 = int(ax - gx * scale), int(ay - gy * scale)
        M = np.float32([[8 / scale, 0, -x0 * 8 / scale], [0, 8 / scale, -y0 * 8 / scale]])
        ref = cv2.warpAffine(src, M, (W * 8, H * 8))
        cv2.imwrite(png, np.hstack([ref, render(rows)]))
