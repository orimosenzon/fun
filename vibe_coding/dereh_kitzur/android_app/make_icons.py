#!/usr/bin/env python3
"""Build the Android app's launcher icons and splash images from the web
app's own icon (web/icon-512.png, made by build_icons.py), so the app on the
home screen is the same picture as the web app added to it.

Android's adaptive icon is a 108dp square of which only the middle 72dp is
ever shown, cut to whatever shape the launcher likes. The web icon is
full-bleed with its line inside the middle 80 %, so it goes in as the
foreground at exactly the visible 72dp: the launcher's mask trims the green
corners and never reaches the line.

    python3 android_app/make_icons.py
"""

import pathlib

from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
ICON = HERE.parent / "web" / "icon-512.png"
RES = HERE / "android" / "app" / "src" / "main" / "res"

DENSITY = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}
BG = "#1e5e22"          # behind the foreground; seen only at the very edge


def round_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=255)
    return m


def main():
    src = Image.open(ICON).convert("RGBA")

    for name, k in DENSITY.items():
        d = RES / f"mipmap-{name}"
        # Adaptive foreground: 108dp canvas, the icon over the visible 72dp.
        fg_size, vis = round(108 * k), round(72 * k)
        fg = Image.new("RGBA", (fg_size, fg_size), BG)
        off = (fg_size - vis) // 2
        fg.paste(src.resize((vis, vis), Image.LANCZOS), (off, off))
        fg.save(d / "ic_launcher_foreground.png")

        # Legacy icons for launchers older than Android 8.
        s = round(48 * k)
        icon = src.resize((s, s), Image.LANCZOS)
        sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        sq.paste(icon, (0, 0), round_mask(s, round(s * 0.18)))
        sq.save(d / "ic_launcher.png")
        rd = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        rd.paste(icon, (0, 0), round_mask(s, s // 2))
        rd.save(d / "ic_launcher_round.png")

    (RES / "values" / "ic_launcher_background.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
        f'    <color name="ic_launcher_background">{BG}</color>\n</resources>\n')

    # Splash (Android 11 and older; 12+ draws its own from the launcher icon):
    # every existing splash.png is redrawn at its own size, white with the
    # icon in the middle, matching the white the app itself starts on.
    for f in RES.glob("drawable*/splash.png"):
        w, h = Image.open(f).size
        s = round(min(w, h) * 0.32)
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(src.resize((s, s), Image.LANCZOS), ((w - s) // 2, (h - s) // 2),
                     round_mask(s, round(s * 0.22)))
        canvas.save(f)
    print("icons and splash written under", RES)


if __name__ == "__main__":
    main()
