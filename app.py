from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from animegan import AnimeGAN
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = "model/animeganv2.onnx"
gan = AnimeGAN(MODEL_PATH)

@app.route("/api/image-to-cartoon", methods=["POST"])
def image_to_cartoon():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Image file required"}), 400

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "message": "Invalid image"}), 400

    cartoon = gan.process(img)

    _, buffer = cv2.imencode(".jpg", cartoon, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype="image/jpeg"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)