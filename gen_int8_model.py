import os
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# type: ignore

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ONNX_MODEL_PATH = "yolov5s.onnx"
CALIBRATION_DIR = "coco/calibration_int8"
CACHE_FILE = "yolov5s_int8_calibration.cache"
BATCH_SIZE = 8
INPUT_SHAPE = (3, 640, 640)  # CHW

class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, batch_size, input_shape):
        super().__init__()
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * np.prod(input_shape).item() * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return img

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.image_paths):
            return None
        batch = [self.preprocess(p) for p in self.image_paths[self.index : self.index + self.batch_size]]
        batch_np = np.ascontiguousarray(np.stack(batch))
        cuda.memcpy_htod(self.device_input, batch_np)
        self.index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(CACHE_FILE, "wb") as f:
            f.write(cache)

def build_int8_engine():
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_MODEL_PATH, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = ImageEntropyCalibrator(CALIBRATION_DIR, BATCH_SIZE, INPUT_SHAPE)
    config.int8_calibrator = calibrator

    print("Building INT8 engine with calibration...")
    engine = builder.build_serialized_network(network, config)
    serialized_engine = builder.build_serialized_network(network, config)
    with open("yolov5s_int8.engine", "wb") as f:
        f.write(serialized_engine)

        
    print("Done. Calibration cache saved to:", CACHE_FILE)

if __name__ == "__main__":
    build_int8_engine()
