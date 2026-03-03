from flask import Flask, render_template, request, jsonify
import numpy as np
import time
import os
from datetime import datetime
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------
# 1️⃣ Import & Configuration
# ---------------------------------------------------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

model = None
model_error = None
MODEL_PATH = "malaria_model.h5"
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ---------------------------------------------------
# 2️⃣ Model Loading
# ---------------------------------------------------

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print("✅ Model Loaded Successfully")
        print("Input Shape:", model.input_shape)
        print("Output Shape:", model.output_shape)
    else:
        raise FileNotFoundError("Model file not found.")

except Exception as e:
    model_error = str(e)
    print("❌ Model Loading Failed:", model_error)

# ---------------------------------------------------
# 3️⃣ Image Preprocessing Function
# ---------------------------------------------------

def prepare_image(image, target_size=(128, 128)):

    image = Image.open(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# ---------------------------------------------------
# 4️⃣ Prediction Route
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return "Model not loaded", 500

    file = request.files["file"]

    start_time = time.time()

    image = prepare_image(file)
    prediction = model.predict(image, verbose=0)

    probability = float(prediction[0][0])
    predicted_class = 1 if probability > 0.5 else 0
    confidence = probability if predicted_class == 1 else (1 - probability)

    processing_time = round(time.time() - start_time, 2)

    return render_template(
        "report.html",
        prediction=CLASS_NAMES[predicted_class],
        confidence=round(confidence * 100, 2),
        parasitized=round((1 - probability) * 100, 2),
        uninfected=round(probability * 100, 2),
        processing_time=processing_time,
        timestamp=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
# ---------------------------------------------------
# 5️⃣ Home Route
# ---------------------------------------------------

@app.route("/")
def home():

    template_path = os.path.join("templates", "index.html")

    if os.path.exists(template_path):
        return render_template("index.html")
    else:
        return """
        <h1>Malaria Detection Server Running</h1>
        <p>Go to /predict to test API</p>
        """

# ---------------------------------------------------
# 6️⃣ Health Check Route
# ---------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "server": "running",
        "model_loaded": model is not None
    })

# ---------------------------------------------------
# 7️⃣ Entry Point
# ---------------------------------------------------

if __name__ == "__main__":

    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    print("🚀 Starting Flask Server...")
    print("Model Status:", "✅ Loaded" if model else "❌ Not Loaded")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)