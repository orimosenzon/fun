#!/usr/bin/env python3
"""גיליון פריימים במרווח קבוע מסרטון: python3 tools/frames_grid.py video out.png [fps=1.5] [cols=10] [w=170] [start] [end]"""
import sys, subprocess, io
from PIL import Image, ImageDraw, ImageFont
v, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 10
w = int(sys.argv[5]) if len(sys.argv) > 5 else 170
st = float(sys.argv[6]) if len(sys.argv) > 6 else 0
en = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', v], capture_output=True, text=True).stdout)
en = min(en, dur - 0.05)
# חיתוך: הדמות במרכז הפריים, מורידים שוליים
raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(st), '-to', str(en), '-i', v, '-vf', f'fps={fps},crop=iw*0.62:ih*0.92:iw*0.19:ih*0.04,scale={w}:-2', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
imgs = []; i = 0
while True:
    j = raw.find(b'\x89PNG', i + 1)
    chunk = raw[i:j if j > 0 else None]
    if chunk: imgs.append(Image.open(io.BytesIO(chunk)).convert('RGB'))
    if j < 0: break
    i = j
W, H = imgs[0].size; rows = (len(imgs) + cols - 1) // cols
sh = Image.new('RGB', (cols * W, rows * H), 'white'); d = ImageDraw.Draw(sh)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
for n, im in enumerate(imgs):
    X, Y = (n % cols) * W, (n // cols) * H
    sh.paste(im, (X, Y)); d.rectangle([X, Y, X + 40, Y + 15], fill='black'); d.text((X + 2, Y), f'{st + n / fps:.1f}', fill='yellow', font=font)
sh.save(out); print(len(imgs), 'frames', sh.size)
