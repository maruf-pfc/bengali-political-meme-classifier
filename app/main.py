from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model import predict

app = FastAPI(title="Bengali Political Meme Classifier")

@app.get("/")
def root():
    return {"message": "ViT Meme Classifier API is running"}

@app.post("/predict")
async def predict_meme(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label = predict(image)
    return {"prediction": label}
