#!/usr/bin/env python3
"""Facts derived from hood_maps/hoods.json for the neighbourhood articles:
who borders whom (with compass direction) and which streets lie inside.

    python3 hood_facts.py            # print everything
"""
import json, math, os
from hood_maps import DIR, MX, MY, centroid

def zones():
    """Every clickable zone: (name, article, rings, parent name)."""
    hoods = json.load(open(os.path.join(DIR, 'hoods.json'), encoding='utf-8'))['hoods']
    out = []
    for h in hoods:
        if h.get('parts'):
            out += [dict(name=p['name'], article=p['article'], rings=p['rings'], parent=h['name']) for p in h['parts']]
        else:
            out.append(dict(name=h['name'], article=h['article'], rings=h['rings'], parent=h['name']))
    return out

def xy(p): return ((p[1] - 34.975) * MX, (p[0] - 32.47) * MY)

def seg_d(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)

def dense(ring, step=15):
    pts = [xy(p) for p in ring]; out = []
    for a, b in zip(pts, pts[1:] + pts[:1]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        out += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)]
    return out

def shared(z1, z2, tol=45):
    """Metres of z1's outline that run within `tol` of z2's outline, and where."""
    a = [p for r in z1['rings'] for p in dense(r)]
    b = [xy(p) for r in z2['rings'] for p in r]
    segs = list(zip(b, b[1:] + b[:1]))
    near = [p for p in a if any(seg_d(p, s, e) < tol for s, e in segs)]
    if not near: return 0, None
    return len(near) * 15, (sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near))

DIRS = ['ממזרח', 'מצפון-מזרח', 'מצפון', 'מצפון-מערב', 'ממערב', 'מדרום-מערב', 'מדרום', 'מדרום-מזרח']

def neighbours(min_len=120):
    zs = zones(); out = {}
    for z in zs:
        c = xy(centroid(max(z['rings'], key=len))); lst = []
        for o in zs:
            if o is z: continue
            L, mid = shared(z, o)
            if L >= min_len:
                ang = math.degrees(math.atan2(mid[1] - c[1], mid[0] - c[0])) % 360
                lst.append((o['name'], o['article'], DIRS[int((ang + 22.5) // 45) % 8], L))
        out[z['name']] = lst
    return out

def inside(pt, ring):
    x, y = pt; c = False
    for (ay, ax), (by, bx) in zip(ring, ring[1:] + ring[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax: c = not c
    return c

def streets():
    ways = json.load(open(os.path.join(DIR, 'roads.json'), encoding='utf-8'))['ways']
    zs = zones(); tot = {}; per = {z['name']: {} for z in zs}
    for w in ways:
        if not w['name'] or w['k'] == 'rail': continue
        for a, b in zip(w['g'], w['g'][1:]):
            L = math.hypot((a[0] - b[0]) * MY, (a[1] - b[1]) * MX); m = ((a[1] + b[1]) / 2, (a[0] + b[0]) / 2)
            tot[w['name']] = tot.get(w['name'], 0) + L
            for z in zs:
                if any(inside(m, r) for r in z['rings']):
                    per[z['name']][w['name']] = per[z['name']].get(w['name'], 0) + L; break
    # a street belongs to a neighbourhood when a real stretch of it runs inside
    return {n: sorted((s for s, L in d.items() if L >= 60 and L >= 0.25 * tot[s]), key=lambda s: -d[s]) for n, d in per.items()}

if __name__ == '__main__':
    nb = neighbours(); st = streets()
    for n in nb:
        print(n, '|', ', '.join('%s (%s, %dמ)' % (a, d, L) for a, _, d, L in nb[n]))
        print('    רחובות:', ', '.join(st[n]))
