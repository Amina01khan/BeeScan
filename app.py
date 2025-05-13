from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load models and labels
model_validation = load_model(os.path.join(APP_ROOT, 'keras_model.h5'))
model_diagnosis = load_model(os.path.join(APP_ROOT, 'keras_model (1).h5'))

labels_validation = [line.strip() for line in open(os.path.join(APP_ROOT, 'labels.txt'))]
labels_diagnosis = [line.strip() for line in open(os.path.join(APP_ROOT, 'labels (1).txt'))]labels_diagnosis = [line.strip() for line in open(os.path.join(APP_ROOT, 'labels.txt'))]






def preprocess_image(image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    normalized = (img_array / 127.5) - 1
    return np.expand_dims(normalized, axis=0)

def get_recommendation(label):
    if "Deformed Wing" in label:
        return [
            "Control Varroa mites, as they are the primary vector for DWV",
            "Use approved miticides or organic treatments",
            "Improve hive ventilation to reduce stress",
            "Consider requeening if severely affected",
            "Isolate affected hives"
        ]
    elif "Varroa Mite" in label:
        return [
            "Apply mite treatments (formic acid, oxalic acid, etc.)",
            "Use biotechnical methods like drone brood removal",
            "Monitor mite levels regularly",
            "Ensure hive ventilation",
            "Break brood cycle if needed"
        ]
    elif "Chalkbrood" in label:
        return [
            "Improve hive ventilation",
            "Destroy infected combs",
            "Requeen with chalkbrood-resistant stock",
            "Strengthen colony if weak",
            "Keep hive dry but insulated"
        ]
    elif "Healthy" in label:
        return [
            "Continue regular inspections",
            "Maintain ventilation",
            "Ensure adequate food supply",
            "Monitor for behavior changes"
        ]
    else:
        return [
            "Consult a beekeeping expert",
            "Take more samples",
            "Isolate affected hive",
            "Research condition online",
            "Check other hives for spread"
        ]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        image = Image.open(file.stream)
        processed = preprocess_image(image)

        # Stage 1: Validation
        pred_val = model_validation.predict(processed)[0]
        label_val = labels_validation[np.argmax(pred_val)]
        confidence_val = float(np.max(pred_val))

        if not label_val.startswith("0"):
            return render_template("index.html", invalid=True, label=label_val, confidence=confidence_val)

        # Stage 2: Diagnosis
        pred_diag = model_diagnosis.predict(processed)[0]
        idx = np.argmax(pred_diag)
        label_diag = labels_diagnosis[idx]
        confidence_diag = float(pred_diag[idx])
        recommendations = get_recommendation(label_diag)

        return render_template(
            "index.html",
            result=True,
            image_uploaded=True,
            label=label_diag,
            confidence=confidence_diag,
            recommendations=recommendations
        )

    return render_template("index.html")
