from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
import shutil
import torch
import os
from torchvision import transforms, models
import torch.nn as nn

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if needed (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend HTML
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = Path("index.html")
    return index_path.read_text()

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "outputs_pytorch/best_model.pt"

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
CLASS_NAMES = ckpt["classes"]
IMG_SIZE = ckpt["img_size"]
num_classes = len(CLASS_NAMES)

def build_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_model(num_classes)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

def predict_image(image_path):
    H, W = IMG_SIZE[1], IMG_SIZE[0]
    tfm = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    idx = probs.argmax()
    category = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    probabilities = {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}

    return category, confidence, probabilities

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / file.filename

        # Save uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        category, confidence, probabilities = predict_image(file_path)

        # Remove file after prediction
        file_path.unlink()

        return JSONResponse(content={
            "category": category,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
