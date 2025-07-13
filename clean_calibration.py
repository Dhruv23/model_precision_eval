import os
import cv2
import shutil
import numpy as np

SOURCE_DIR = "../coco/calibration_int8"
BAD_DIR = "../coco/calibration_bad"

os.makedirs(BAD_DIR, exist_ok=True)

for fname in os.listdir(SOURCE_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(SOURCE_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        shutil.move(path, os.path.join(BAD_DIR, fname))
        continue
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    stds = [img[c].std() for c in range(3)]
    if any(std < 0.01 for std in stds):
        print(f"[REMOVED] {fname} | std={stds}")
        shutil.move(path, os.path.join(BAD_DIR, fname))
