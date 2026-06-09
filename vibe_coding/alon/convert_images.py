#!/usr/bin/env python3
"""המרת תמונות DNG (RAW) של הגיטרות ל-JPG לאתר.

לכל DNG מחלצים את ה-thumbnail המוטמע (מהיר, רזולוציה גבוהה).
אם אין thumbnail מתאים — מפענחים את ה-RAW המלא.
נוצרות שתי גרסאות: מלאה (עד 1600px) ו-thumbnail (600px).
"""
import os
import io
import sys
import rawpy
from PIL import Image, ImageOps

SRC_ROOT = "/home/ori/tmp/אלון/gitars/גיטרות 1"
DST_ROOT = "/home/ori/fun/vibe_coding/alon/images"

# שם תיקיית מקור -> שם תיקיית יעד (אנגלית, נקי ל-URL)
GUITARS = {
    "רוקינגברד קטנה": "rockingbird",
    "גיטרה זברווד כולל צוואר": "zebrawood",
}

FULL_MAX = 1600
THUMB_MAX = 600
FULL_Q = 86
THUMB_Q = 80


def load_image(dng_path):
    """מחזיר תמונת PIL במגמת RGB מתוך קובץ DNG."""
    with rawpy.imread(dng_path) as raw:
        # ניסיון 1: thumbnail מוטמע (מהיר ואיכותי)
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
                # אם ה-thumbnail גדול מספיק — משתמשים בו
                if max(img.size) >= FULL_MAX:
                    return ImageOps.exif_transpose(img).convert("RGB")
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass
        # ניסיון 2: פענוח RAW מלא
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        return Image.fromarray(rgb)


def save_variant(img, path, max_side, quality):
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
    return im.size


def main():
    manifest = {}
    for src_name, dst_name in GUITARS.items():
        src_dir = os.path.join(SRC_ROOT, src_name)
        dst_dir = os.path.join(DST_ROOT, dst_name)
        thumb_dir = os.path.join(dst_dir, "thumbs")
        os.makedirs(thumb_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".dng"))
        out_files = []
        for i, fname in enumerate(files, 1):
            stem = f"{i:02d}"
            src = os.path.join(src_dir, fname)
            full_out = os.path.join(dst_dir, f"{stem}.jpg")
            thumb_out = os.path.join(thumb_dir, f"{stem}.jpg")
            print(f"[{dst_name}] {fname} -> {stem}.jpg ...", flush=True)
            try:
                img = load_image(src)
                fs = save_variant(img, full_out, FULL_MAX, FULL_Q)
                ts = save_variant(img, thumb_out, THUMB_MAX, THUMB_Q)
                out_files.append(f"{stem}.jpg")
                print(f"    full={fs} thumb={ts}", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr, flush=True)
        manifest[dst_name] = out_files
    print("\n=== MANIFEST ===")
    for k, v in manifest.items():
        print(f"{k}: {len(v)} images -> {v}")


if __name__ == "__main__":
    main()
