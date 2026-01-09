import cv2
import numpy as np
import onnxruntime as ort

class AnimeGAN:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def process(self, img):
        h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        out = self.session.run(None, {self.input_name: img})[0]

        out = out[0].transpose(1, 2, 0)
        out = ((out + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        return out
