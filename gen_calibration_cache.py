import os, numpy as np, cv2, tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ONNX = "yolov5s.onnx"
CALIB_DIR = "../coco/calibration_int8"
CACHE_FILE = "yolov5s_int8.cache"
BATCH_SIZE = 8
INPUT_SHAPE = (3, 640, 640)

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, cache_file, batch_size):
        self.last_cache = None  # Add to __init__
        super().__init__()
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.images = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg",".png"))
        ]
        assert len(self.images) >= batch_size
        self.device_input = cuda.mem_alloc(
            int(batch_size * np.prod(INPUT_SHAPE) * np.float32().nbytes)
        )

        self.idx = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.images):
            return None
        batch = np.zeros((self.batch_size, *INPUT_SHAPE), dtype=np.float32)
        for i in range(self.batch_size):
            img = cv2.imread(self.images[self.idx + i])
            img = cv2.resize(img, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2,0,1))
            batch[i] = img
        cuda.memcpy_htod(self.device_input, batch.ravel())
        self.idx += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return open(self.cache_file, "rb").read()

    
    def write_calibration_cache(self, cache):
        self.last_cache = cache  # Save in memory
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def main():
    builder = trt.Builder(TRT_LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, TRT_LOGGER)
    with open(ONNX, "rb") as f:
        parser.parse(f.read())
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<30)

    calibrator = EntropyCalibrator(CALIB_DIR, CACHE_FILE, BATCH_SIZE)
    cfg.int8_calibrator = calibrator

    print("→ Calibrating INT8 (this writes the cache)...")
    engine_data = builder.build_serialized_network(net, cfg)
    if calibrator.last_cache:
        print(f"[INFO] Manually writing calibration cache to {cache_file}")
        with open(cache_file, "wb") as f:
            f.write(calibrator.last_cache)
    else:
        print("[ERROR] Calibration completed but no cache was saved.")
    if engine_data:
        print(f"✔ Calibration cache saved to {CACHE_FILE}")
    else:
        print("✘ Calibration failed")

if __name__ == "__main__":
    main()
