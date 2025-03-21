from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = load_model("CDM.h5")

# Function to preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")  
    img = ImageEnhance.Contrast(img).enhance(2.0)  
    img = img.resize(target_size)  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Function to predict currency authenticity
def predict_currency(image_path, threshold=0.7):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)[0][0]
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
