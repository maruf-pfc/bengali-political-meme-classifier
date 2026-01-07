# Bengali Political Meme Classifier

This is a **ViT-based Image Classifier** trained to distinguish between **Political** and **Non-Political** Bengali memes. The application is built using **FastAPI** and **PyTorch**.

## 🚀 Features
- **FastAPI** backend for high-performance inference.
- **Vision Transformer (ViT)** model for state-of-the-art image classification.
- **Premium Web UI**: Glassmorphism design with drag-and-drop support.
- **Auto-Model Download**: Automatically fetches the model from Google Drive on startup.
- **Dockerized** for easy deployment.
- **Swagger UI** for interactive API testing.

![Demo](assets/demo.webp)

---

## 🎨 Web Interface
The project now includes a beautiful web interface.
1.  Open the app URL.
2.  Drag and drop your image.
3.  Get instant results!

---


## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- Virtual Environment (recommended)
- **Model File**: Ensure `vit_meme_model.pth` is in the `model/` directory, OR set `MODEL_URL` to download it automatically.

### Deployment Checklist
- [ ] Model file (`model/vit_meme_model.pth`) is present OR
- [ ] `MODEL_URL` environment variable is set to a direct download link.
- [ ] Environment variables (if any) are set.
- [ ] `.gitignore` is configured to exclude `env/`, `__pycache__/`, etc.


### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maruf-pfc/bengali-political-meme-classifier.git
   cd bengali-political-meme-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Linux/Mac
   # \env\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 1221 --reload
   ```

5. **Access the API:**
   - Swagger Documentation: [http://localhost:1221/docs](http://localhost:1221/docs)
   - Health Check: [http://localhost:1221/](http://localhost:1221/)

---

## 🏋️ Training the Model
Since the model file is not included (due to size), you can train it yourself using the provided dataset.

1.  **Prepare Dataset:**
    - Download your dataset.
    - Organize it into two folders: `Political` and `NonPolitical`.
    - Example structure:
      ```
      dataset/
      ├── Political/
      │   ├── image1.jpg
      │   └── ...
      └── NonPolitical/
          ├── image1.jpg
          └── ...
      ```

2.  **Run Training:**
    ```bash
    # Activate your env first!
    python train.py --data_dir /path/to/dataset --epochs 3
    ```

3.  **Result:**
    - The model will be saved to `model/vit_meme_model.pth`.
    - You can now run the app!

---

## 🐳 Deployment on Coolify

This project is ready for deployment on **Coolify** or any Docker-based platform.

### Deployment Steps
1. **Push to GitHub/GitLab.**
2. **Add a new Service in Coolify:**
   - Select **Source**: Git Repository.
   - Choose this repository.
   - **Build Pack**: Dockerfile.
   - **Port**: `1221`.
3. **Deploy!** 🚀

### Docker Commands (Manual)

**Build the image:**
```bash
docker build -t meme-classifier .
```

**Run the container:**
```bash
docker run -d -p 1221:1221 --name meme-app meme-classifier
```


---

## ❓ Troubleshooting

### Error 522 / Connection Timed Out
If you see a **Cloudflare 522** error or the app takes forever to start:
- **Cause**: The app is downloading the 300MB+ model file on the *first* startup. This can take 2-5 minutes depending on the server's speed.
- **Solution**:
    1.  **Wait**: Give it 5-10 minutes.
    2.  **Check Logs**: If possible, check the container logs (`docker logs meme-app`) to see the download progress.
    3.  **Persist Storage**: Ensure the `model/` directory is mounted as a volume (already configured in `docker-compose.yaml`). This way, it only downloads ONCE. Future restarts will be instant.

---

## 📡 API Endpoints

### `GET /`
Returns a welcome message and health status.

### `POST /predict`
Upload an image to get a classification (Political vs. Non-Political).

**Example Request:**
```bash
curl -X 'POST' \
  'http://localhost:1221/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

**Example Response:**
```json
{
  "prediction": "Political"
}
```
