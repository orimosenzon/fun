# python3 tools/sheet.py <prefix> [columns] [scale]: contact sheet of reference/shots/<prefix>_*.png with labels.
import sys, glob, os
from PIL import Image, ImageDraw
pre = sys.argv[1]; cols = int(sys.argv[2]) if len(sys.argv) > 2 else 4; sc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
here = os.path.join(os.path.dirname(__file__), '..', 'reference')
files = sorted(glob.glob(os.path.join(here, 'shots', pre + '_*.png')))
ims = [Image.open(f) for f in files]
w, h = int(ims[0].width * sc), int(ims[0].height * sc)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * h))
d = ImageDraw.Draw(sheet)
for i, (f, im) in enumerate(zip(files, ims)):
    x, y = (i % cols) * w, (i // cols) * h
    sheet.paste(im.resize((w, h), Image.NEAREST), (x, y))
    d.rectangle([x, y, x + 34, y + 11], fill=(0, 0, 0)); d.text((x + 2, y), os.path.basename(f)[len(pre) + 1:-4], fill=(255, 255, 0))
out = os.path.join(here, 'sheet_' + pre + '.png'); sheet.save(out); print(out, len(ims))
