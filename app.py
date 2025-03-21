from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow.lite as tflite
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the TensorFlow Lite model
TFLITE_MODEL_PATH = "CDM.tflite"
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image for TFLite model
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Increase contrast
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Convert to float32
    return img_array

# Function to predict currency authenticity using TFLite model
def predict_currency(image_path, threshold=0.7):
    preprocessed_image = load_and_preprocess_image(image_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return "Real Currency" if prediction >= threshold else "Fake Currency"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            result = predict_currency(file_path)
            return render_template("index.html", result=result, image=file.filename)

    return render_template("index.html", result=None, image=None)

if __name__ == "__main__":
    app.run(debug=True)
