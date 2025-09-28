from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from src.classifier.entity.config_entity import PredictionConfig
from src.classifier.pipeline.predict import Prediction_Pipeline
from src.classifier.utils.class_mapping import all_labels  # if you want class names
import torch
import os
import uvicorn

app = FastAPI()

# ----- Model Config ------------------
config = PredictionConfig(
    model_path=Path("artifacts/best_model/best_skin_model_full.pth"),
    device="cpu",
)

try:
    predictor = Prediction_Pipeline(config=config)
except Exception as e:
    print(f"❌ Error initializing Prediction_Pipeline: {e}")
    predictor = None

# --------Endpoints ------------------
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get predicted class probabilities.
    """
    if predictor is None:
        return {"error": "Model not loaded"}

    temp_path = Path(f"temp_{file.filename}")# Save uploaded file to temp
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        probs = predictor.predict(str(temp_path))
        if probs is None:
            return {"error": "Prediction failed"}

        ## ----tensor
        probs_tensor = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs

        # Top-3 predictions
        top3_probs, top3_indices = torch.topk(probs_tensor, k=3, dim=-1)
        top3_classes = [all_labels[i] for i in top3_indices.tolist()[0]]

        # cleaning the temp file
        if temp_path.exists():
            os.remove(temp_path)

        return {
            "top3_classes": top3_classes,
            "top3_probabilities": top3_probs.tolist()[0],
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_path.exists():
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.01", port=8000)