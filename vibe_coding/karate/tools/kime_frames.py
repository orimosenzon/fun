#!/usr/bin/env python3
"""מוצא את נקודות העצירה (קימה) בסרטון קאטה ומרכיב גיליון של פריים אחד לכל תנועה.
שימוש: python3 tools/kime_frames.py ref/kata/heian1.mp4 out.png [--thr 0.35] [--cols 6] [--start s] [--end s]
"""
import sys, subprocess, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('out')
ap.add_argument('--fps', type=float, default=15)
ap.add_argument('--thr', type=float, default=0.30)   # חלק מהשיא שמתחתיו נחשב "עומד"
ap.add_argument('--minstill', type=float, default=0.20)
ap.add_argument('--cols', type=int, default=6)
ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float, default=1e9)
ap.add_argument('--w', type=int, default=240)
a = ap.parse_args()

W, H = 160, 120
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', a.video, '-vf', f'fps={a.fps},scale={W}:{H},format=gray', '-f', 'rawvideo', '-'], capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
k = np.ones(3) / 3; m = np.convolve(diff, k, 'same')
t = np.arange(len(m)) / a.fps
sel = (t >= a.start) & (t <= a.end)
peak = np.percentile(m[sel], 95)
still = (m < a.thr * peak) & sel
# קטעי עמידה
segs = []; i = 0
while i < len(still):
    if still[i]:
        j = i
        while j < len(still) and still[j]: j += 1
        if (j - i) / a.fps >= a.minstill: segs.append((i, j))
        i = j
    else: i += 1
# פריים בתחילת כל קטע עמידה (אחרי 2 פריימים, כשהתנועה נעצרה)
times = [(s + min(2, (e - s) // 2)) / a.fps for s, e in segs]
print(len(times), 'עצירות:', ' '.join(f'{x:.1f}' for x in times))
imgs = []
for x in times:
    png = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{x:.2f}', '-i', a.video, '-frames:v', '1', '-vf', f'scale={a.w}:-1', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    import io; imgs.append(Image.open(io.BytesIO(png)).convert('RGB'))
if not imgs: sys.exit()
w, h = imgs[0].size; cols = a.cols; rows = (len(imgs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * h), 'white'); d = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception: font = None
for n, (im, x) in enumerate(zip(imgs, times)):
    X, Yy = (n % cols) * w, (n // cols) * h
    sheet.paste(im, (X, Yy)); d.rectangle([X, Yy, X + 74, Yy + 22], fill='black')
    d.text((X + 3, Yy + 1), f'{n+1} {x:.1f}', fill='yellow', font=font)
sheet.save(a.out)
