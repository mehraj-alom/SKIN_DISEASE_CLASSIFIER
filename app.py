import os
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from src.classifier import logger
from src.classifier.entity.config_entity import PredictionConfig
from src.classifier.pipeline.predict import Prediction_Pipeline
from src.classifier.utils.class_mapping import all_labels
from conditions import conditions_info

ARTIFACT_DIR = Path("artifacts/streamlit")
TEMP_DIR = Path("temp")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Skin Disease Classifier",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

video_url = "https://cdn.pixabay.com/vimeo/392238854/dna-Helix-6973.mp4?width=1280&hash=9dfeb1aebf4b3b9c699351b37cfc065f13e4de32"  # direct mp4 link

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}
    video.background {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        opacity: 0.35;
        filter: blur(2px);
    }}
    .main-title {{
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px #000;
    }}
    .warning-box {{
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    </style>
    <video autoplay loop muted playsinline class="background">
        <source src="{video_url}" type="video/mp4">
    </video>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>AI-Powered Skin Disease Classifier 🩺</h1>", unsafe_allow_html=True)
st.markdown("<div class='warning-box'>⚠️ This app is for educational purposes only — Not for real medical use.</div>", unsafe_allow_html=True)
st.write("Upload a skin image to get AI-based predictions (top 3 possible conditions).")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        try:
            config = PredictionConfig(
                model_path=Path("artifacts/best_model/best_skin_model_full.pth"),
                device="cpu"
            )
            predictor = Prediction_Pipeline(config=config)
            if predictor.model is None:
                st.error("Model failed to load. Check model_path.")
            else:
                probs = predictor.predict(image)
                if probs is None:
                    st.error("Prediction failed.")
                else:
                    probs_tensor = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs
                    top3_probs, top3_indices = torch.topk(probs_tensor, k=3, dim=-1)
                    top3_classes = [all_labels[i] for i in top3_indices.tolist()[0]]
                    st.markdown("###  Top-3 Predictions:")
                    for cls, prob in zip(top3_classes, top3_probs.tolist()[0]):
                        st.write(f"**{cls}** : {prob*100:.2f}%")
                        if cls in conditions_info:
                            info = conditions_info[cls]
                            with st.expander(f" Details for {cls}"):
                                st.markdown(f"**Risk:** {info['risk']}")
                                st.markdown(f"**Description:** {info['description']}")
                                st.markdown("**Recommendations:**")
                                for rec in info["recommendations"]:
                                    st.markdown(f"- {rec}")
                        else:
                            st.info(f"No extra info available for {cls}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 AI Skin Health Assistant | Educational use only</p>",
    unsafe_allow_html=True
)
