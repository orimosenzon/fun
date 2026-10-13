"""מושך מודל גבהים (DEM) של האזור.

אריחי Terrarium הפתוחים של AWS (Mapzen), בזום 15, כלומר בערך 4 מטר לפיקסל.
המקור בישראל הוא בעיקר SRTM ברזולוציה של 30 מטר, כך שהפרטים עדינים
פחות ממה שהפיקסלים מרמזים, אבל הגבעה של כרכור והשיפוע לכיוון מערב נמצאים שם.
שומר ל-data/raw/dem.npz: מערך גבהים במטרים ומיפוי לקואורדינטות.
"""
import io
import math
import pathlib

import numpy as np
import requests
from PIL import Image

from fetch_osm import BBOX, ROOT

Z = 15
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OUT = ROOT / "data" / "raw" / "dem.npz"
CACHE = ROOT / "data" / "raw" / "dem_tiles"


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def main():
    s, w, n, e = BBOX
    x0, y0 = (int(v) for v in tile_xy(n, w, Z))
    x1, y1 = (int(v) for v in tile_xy(s, e, Z))
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for ty in range(y0, y1 + 1):
        row = []
        for tx in range(x0, x1 + 1):
            f = CACHE / f"{Z}_{tx}_{ty}.png"
            if not f.exists():
                r = requests.get(URL.format(z=Z, x=tx, y=ty), timeout=60)
                r.raise_for_status()
                f.write_bytes(r.content)
            a = np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGB")).astype(np.float64)
            row.append(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        rows.append(np.hstack(row))
    h = np.vstack(rows).astype(np.float32)
    np.savez_compressed(OUT, h=h, z=Z, x0=x0, y0=y0)
    print(f"DEM {h.shape} ({(x1-x0+1)*(y1-y0+1)} אריחים), גבהים {h.min():.1f}–{h.max():.1f} מ' -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
