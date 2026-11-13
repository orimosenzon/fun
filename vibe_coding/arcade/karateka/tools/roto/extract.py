"""Rotoscoping, step 1: pull frames out of a reference video and find the person in them.

For every requested time it saves, under reference/roto/work/<name>/:
  NN.png       the cropped frame
  NN_cls.png   class map: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other
  NN.json      33 pose landmarks in crop pixels (x, y, z, visibility)
  sheet.jpg    all frames side by side with the classes and the skeleton drawn on top

usage: venv/bin/python tools/roto/extract.py <name> <video> <x,y,w,h> <t0> <t1> <fps> [--flip]
(run with reference/roto/venv; the models are in reference/roto/models)
"""
import json, os, sys
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'reference', 'roto')
name, video, box, t0, t1, fps = sys.argv[1:7]
flip = '--flip' in sys.argv
x, y, w, h = map(int, box.split(','))
t0, t1, fps = float(t0), float(t1), float(fps)
out = os.path.join(ROOT, 'work', name); os.makedirs(out, exist_ok=True)

seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ROOT, 'models', 'selfie_multiclass_256x256.tflite')),
    output_category_mask=True))
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ROOT, 'models', 'pose_landmarker_heavy.task')),
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))

cap = cv2.VideoCapture(video if os.path.exists(video) else os.path.join(ROOT, 'vid', video))
COL = np.array([[0, 0, 0], [255, 0, 255], [0, 160, 255], [0, 80, 255], [255, 255, 255], [0, 255, 0]], np.uint8)  # BGR
tiles, n, t = [], 0, t0
while t <= t1 + 1e-6:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, frame = cap.read()
    if not ok: break
    crop = frame[y:y + h, x:x + w]
    if flip: crop = crop[:, ::-1]
    crop = np.ascontiguousarray(crop)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    cls = seg.segment(img).category_mask.numpy_view().copy().reshape(h, w)
    res = pose.detect(img)
    lm = [[p.x * w, p.y * h, p.z * w, p.visibility] for p in res.pose_landmarks[0]] if res.pose_landmarks else []
    cv2.imwrite(os.path.join(out, f'{n:02d}.png'), crop)
    cv2.imwrite(os.path.join(out, f'{n:02d}_cls.png'), cls)
    json.dump({'t': t, 'landmarks': lm}, open(os.path.join(out, f'{n:02d}.json'), 'w'))
    vis = (crop * 0.45 + COL[cls] * 0.55).astype(np.uint8)
    for a, b in [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)]:
        if lm: cv2.line(vis, (int(lm[a][0]), int(lm[a][1])), (int(lm[b][0]), int(lm[b][1])), (0, 0, 255) if a % 2 else (255, 128, 0), 2)
    cv2.putText(vis, f'{n} t={t:.2f}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(np.hstack([crop, vis])); n += 1; t += 1 / fps
cols = 3
rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
sheet = np.vstack(rows); s = min(1.0, 2000 / sheet.shape[1])
cv2.imwrite(os.path.join(out, 'sheet.jpg'), cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
print(n, 'frames ->', out)
