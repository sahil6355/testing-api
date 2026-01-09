import os
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from animegan import AnimeGAN
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ allow frontend requests

MODEL_PATH = "model/animeganv2.onnx"
gan = None  # lazy load


def get_model():
    global gan
    if gan is None:
        print("🔄 Loading AnimeGAN model...")
        gan = AnimeGAN(MODEL_PATH)
        print("✅ Model loaded")
    return gan


@app.route("/")
def home():
    return "AnimeGAN API is LIVE"


@app.route("/api/image-to-cartoon", methods=["POST"])
def image_to_cartoon():
    print("🔥 API CALLED")

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Image file required"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "message": "Invalid image"}), 400

    # 🔹 Resize for Render safety
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gan_model = get_model()
    cartoon = gan_model.process(img)

    success, buffer = cv2.imencode(".jpg", cartoon, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return jsonify({"success": False, "message": "Encode failed"}), 500

    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype="image/jpeg"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ USE PORT
    app.run(host="0.0.0.0", port=port)
