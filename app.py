import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Classes (must match your dataset order)
class_names = ["Black Spot", "Canker", "Greening", "Healthy", "Melanose"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("best_citrus_leaf_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("index.html", filename=filename, prediction=prediction)
    return render_template("index.html")

@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
