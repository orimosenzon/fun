# python3 tools/sheet.py <name> [cols]: contact sheet of reference/shots/<name>_*.png
import sys, glob
from PIL import Image
name = sys.argv[1]; cols = int(sys.argv[2]) if len(sys.argv) > 2 else 4
fs = sorted(glob.glob(f'reference/shots/{name}_*.png'))
ims = [Image.open(f) for f in fs]
w, h = ims[0].size[0] // 2, ims[0].size[1] // 2
rows = (len(ims) + cols - 1) // cols
s = Image.new('RGB', (cols * w, rows * h))
for i, im in enumerate(ims): s.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
s.save(f'reference/sheet_{name}.png'); print(len(ims), 'shots', s.size)
