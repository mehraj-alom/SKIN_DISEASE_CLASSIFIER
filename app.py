# app.py

import os
from pathlib import Path
import streamlit as st
from PIL import Image
import torch

from src.classifier import logger
from src.classifier.entity.config_entity import PredictionConfig
from src.classifier.pipeline.predict import Prediction_Pipeline
from src.classifier.utils.class_mapping import all_labels  # list of class labels

# ------------------ Directories ------------------
ARTIFACT_DIR = Path("artifacts/streamlit")
TEMP_DIR = Path("temp")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Streamlit UI ------------------
st.title("Skin Disease Classifier")
st.write("Upload an image to predict the disease class.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to temp directory
    temp_image_path = TEMP_DIR / uploaded_file.name
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    image = Image.open(temp_image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        try:
            # ------------------ Config & Pipeline ------------------
            config = PredictionConfig(
                model_path="artifacts/best_model/best_skin_model_full.pth",  # path to your trained model
                device="cpu"  # or "cuda" if GPU available
            )
            predictor = Prediction_Pipeline(config=config)

            # ------------------ Prediction ------------------
            probs = predictor.predict(temp_image_path)  # Pass the file path
            if probs is None:
                st.error("❌ Prediction failed.")
            else:
                # Convert to tensor for top-k
                probs_tensor = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs

                # Top-3 predictions
                top3_probs, top3_indices = torch.topk(probs_tensor, k=3, dim=-1)
                top3_classes = [all_labels[i] for i in top3_indices.tolist()[0]]

                # Display results
                st.subheader("Top-3 Predictions")
                for cls, prob in zip(top3_classes, top3_probs.tolist()[0]):
                    st.write(f"**{cls}** : {prob*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

        finally:
            # ------------------ Cleanup ------------------
            if temp_image_path.exists():
                temp_image_path.unlink()
