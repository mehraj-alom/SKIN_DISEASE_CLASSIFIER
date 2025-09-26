# app.py

import os
from pathlib import Path
import streamlit as st
from PIL import Image
import torch

from src.classifier import logger
from src.classifier.entity.config_entity import PredictionConfig
from src.classifier.pipeline.predict import Prediction_Pipeline
from src.classifier.utils.class_mapping import all_labels

# ------------------ Directories ------------------
ARTIFACT_DIR = Path("artifacts/stramlit")
TEMP_DIR = Path("temp")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Streamlit UI ------------------
st.title("Skin Disease Classifier")
st.write("Upload an image to predict the disease class.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        try:
            # ------------------ Config & Pipeline ------------------
            config = PredictionConfig(
                model_path=Path("artifacts/best_model/best_skin_model_full.pth"),
                device="cpu"
            )
            predictor = Prediction_Pipeline(config=config)

            if predictor.model is None:
                st.error("❌ Model failed to load. Check model_path.")
            else:
                # ------------------ Prediction ------------------
                probs = predictor.predict(image)  # pass PIL.Image directly

                if probs is None:
                    st.error("❌ Prediction failed.")
                else:
                    probs_tensor = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs

                    top3_probs, top3_indices = torch.topk(probs_tensor, k=3, dim=-1)
                    top3_classes = [all_labels[i] for i in top3_indices.tolist()[0]]

                    st.subheader("Top-3 Predictions")
                    for cls, prob in zip(top3_classes, top3_probs.tolist()[0]):
                        st.write(f"**{cls}** : {prob*100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
