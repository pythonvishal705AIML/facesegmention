import cv2
import numpy as np
import torch
from torch import nn
from flask import Flask, render_template, request, send_file
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import io
import os

app = Flask(__name__)

#import models for face and segmentation
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

device = "cuda" if torch.cuda.is_available() else "cpu"
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device)

#labels(classes) from model documentation
id2label = {
    0: "background", 1: "skin", 2: "nose", 3: "eye_g", 4: "l_eye",
    5: "r_eye", 6: "l_brow", 7: "r_brow", 8: "l_ear", 9: "r_ear",
    10: "mouth", 11: "u_lip", 12: "l_lip", 13: "hair", 14: "hat",
    15: "ear_r", 16: "neck_l", 17: "neck", 18: "cloth"
}

#function for face detection annd count
def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)

#function for segmentation
def segment_face(image_pil):
    inputs = image_processor(images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
    )

    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    target_classes = [k for k in id2label.keys() if k not in [0, 18, 16, 17]]
    mask = np.isin(labels, target_classes).astype(np.uint8)

    image_np = np.array(image_pil)

    if image_np.shape[-1] == 3:
        alpha_channel = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8) * 255
        image_np = np.concatenate([image_np, alpha_channel], axis=-1)

    image_np[..., 3] = mask * 255

    return image_np  

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Read image 
        image_pil = Image.open(file.stream).convert("RGB")
        image_np = np.array(image_pil)

        face_count = detect_faces(image_np)
        if face_count != 1:
            return render_template("index.html", error=f"Detected {face_count} faces. Please upload an image with exactly one face.")

        
        segmented_image = segment_face(image_pil)

        # Convert NumPy to PIL and return dynamically
        segmented_pil = Image.fromarray(segmented_image)
        img_io = io.BytesIO()
        segmented_pil.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 
