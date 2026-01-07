from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from PIL import Image
import io
import os
import requests
import sys

from app.model import predict

MODEL_PATH = "model/vit_meme_model.pth"

def download_model():
    model_url = os.getenv("MODEL_URL")
    if not model_url:
        print("Warning: MODEL_URL not set. Cannot auto-download model.")
        return

    print(f"Downloading model from {model_url}...")
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Check if it's a Google Drive link (gdown handles these best)
        if "drive.google.com" in model_url:
            import gdown
            gdown.download(model_url, MODEL_PATH, quiet=False, fuzzy=True)
        else:
            # Fallback for standard direct links
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        print("Model downloaded successfully!")
    except ImportError:
        print("Error: gdown not installed but Google Drive link detected.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # We don't exit here, so the app can still start (but will fail on predict)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Attempting download...")
        download_model()
    else:
        print(f"Model found at {MODEL_PATH}.")
    yield

app = FastAPI(title="Bengali Political Meme Classifier", lifespan=lifespan)

# Mount Static Files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/static/index.html") as f:
        return f.read()

@app.post("/predict")
async def predict_meme(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label = predict(image)
    return {"prediction": label}
