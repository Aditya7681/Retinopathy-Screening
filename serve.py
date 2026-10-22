import io
from pathlib import Path
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from model import create_model
from transforms import val_test_transform

app = FastAPI(
    title="Diabetic Retinopathy Screening API",
    description="ResNet-18 model with Weighted Cross-Entropy (best_model_weighted.pth)",
    version="1.0.0"
)

# Enable CORS for browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent / "best_model_weighted.pth"

# Load model weights into memory
model = create_model(num_classes=5)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    print(f"Loaded weights from {MODEL_PATH}")
else:
    print(f"Warning: {MODEL_PATH} not found. Running with uninitialized weights.")

model = model.to(DEVICE)
model.eval()


@app.get("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_file": str(MODEL_PATH.name),
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply standard 224x224 val_test_transform
        tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=1)[0].cpu().item())

        # Combine Stage 3 (Severe) and Stage 4 (Proliferative DR) into single "Severe" class
        combined_idx = 3 if pred_idx >= 3 else pred_idx
        combined_label = "Severe" if pred_idx >= 3 else CLASSES[pred_idx]
        severe_prob = probs[3] + probs[4]

        combined_probs = {
            "No DR": round(probs[0], 4),
            "Mild": round(probs[1], 4),
            "Moderate": round(probs[2], 4),
            "Severe": round(severe_prob, 4)
        }

        return {
            "prediction_index": combined_idx,
            "prediction_label": combined_label,
            "confidence": round(severe_prob if combined_idx == 3 else probs[combined_idx], 4),
            "probabilities": combined_probs,
            "raw_5_class_probabilities": {
                CLASSES[i]: round(probs[i], 4) for i in range(len(CLASSES))
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process image: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)
