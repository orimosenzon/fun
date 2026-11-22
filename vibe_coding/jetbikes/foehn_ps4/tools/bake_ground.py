#!/usr/bin/env python3
"""Bake the ground photographs of the web build into one raw file for the native build.

The console has no JPEG decoder to count on, and GLES2 has at least 8 texture units in a fragment
shader, of which the ground already uses 4. So the five photographs become four textures:
rock, grass, gravel (colour, RGB) and one RGBA texture that carries both normal maps
(rock in RG, grass in BA: only x and y of a normal map are read).

Rows are written bottom-up, which is what the web build gets from UNPACK_FLIP_Y.

  python3 tools/bake_ground.py [source dir] [output file]
"""
import struct
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "foehn" / "assets" / "ground"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "assets" / "ground.bin"
SIZE = 1024


def load(name, mode):
    im = Image.open(SRC / f"{name}.jpg").convert(mode)
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    return im.transpose(Image.FLIP_TOP_BOTTOM)


def linear_mean(im):
    """Mean colour in linear light: what the shader divides the photograph by."""
    lut = [((v / 255 / 12.92) if v <= 10 else ((v / 255 + 0.055) / 1.055) ** 2.4) for v in range(256)]
    out = []
    for band in im.split():
        h = band.histogram()
        out.append(sum(lut[v] * h[v] for v in range(256)) / (SIZE * SIZE))
    return out


def main():
    parts = []
    for name in ("rock", "grass", "gravel"):
        im = load(name, "RGB")
        m = linear_mean(im)
        print(f"{name:7s} linear mean {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}")
        parts.append((3, im.tobytes()))
    rn, gn = load("rock_n", "RGB").split(), load("grass_n", "RGB").split()
    parts.append((4, Image.merge("RGBA", (rn[0], rn[1], gn[0], gn[1])).tobytes()))
    with open(OUT, "wb") as f:
        f.write(b"FGT1" + struct.pack("<II", SIZE, len(parts)))
        for ch, data in parts:
            f.write(struct.pack("<I", ch))
            f.write(data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
