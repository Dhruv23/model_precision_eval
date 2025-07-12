import os
import cv2
import random
import shutil

SOURCE_DIR = "../coco/val2017"
CALIB_DIR = "../coco/calibration_int8"
IMG_SIZE = (640, 640)
NUM_SAMPLES = 100


os.makedirs(CALIB_DIR, exist_ok=True)

valid_filenames = []
for fname in os.listdir(SOURCE_DIR):
    path = os.path.join(SOURCE_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"Unreadable image: {fname}")
        continue
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Non-RGB image: {fname}")
        continue
    valid_filenames.append(fname)

print(f"Total valid RGB images: {len(valid_filenames)}")

if len(valid_filenames) < NUM_SAMPLES:
    raise ValueError(f"Not enough valid images. Found {len(valid_filenames)}, need {NUM_SAMPLES}.")

sampled_filenames = random.sample(valid_filenames, NUM_SAMPLES)


for fname in sampled_filenames:
    src_path = os.path.join(SOURCE_DIR, fname)
    dst_path = os.path.join(CALIB_DIR, fname)
    shutil.move(src_path, dst_path)

print(f"Moved {NUM_SAMPLES} images to {CALIB_DIR}")
