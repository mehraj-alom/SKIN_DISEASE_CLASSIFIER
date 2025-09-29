# 🩺 Skin Disease Classifier - AI-Powered Dermatology Assistant
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
> **⚠️ IMPORTANT DISCLAIMER**
: This project is developed strictly for **research and
educational purposes only**. It should **NEVER** be used for real medical diagnosis. Always
consult qualified healthcare professionals for medical advice and diagnosis.
## 📋 Table of Contents
- [Overview](#overview)
- [The Problem](#the-problem)
- [Solution](#solution)
- [Features](#features)
- [Model Performance](#model-performance)
- [Supported Skin Conditions](#supported-skin-conditions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Challenges Faced](#challenges-faced)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
## 🎯 Overview
This project demonstrates how **Artificial Intelligence can assist in dermatology** by helping
doctors diagnose skin conditions faster and more efficiently. Built as a proof-of-concept, this skin
disease classifier showcases the potential of deep learning in medical image analysis while
emphasizing the importance of human expertise in final diagnosis.
**Version**: 1.0.0
**Status**: Educational/Research Project
## 🔍 The Problem
Skin diseases affect millions of people worldwide, and early diagnosis is crucial for effective
treatment. However:
- Dermatologists often face high patient volumes
- Initial screening can be time-consuming
- Access to specialists may be limited in remote areas
- Visual diagnosis requires extensive experience
## 💡 Solution
This AI-powered classifier can:
- Analyze skin condition images in seconds
- Provide preliminary classification across 27 different skin conditions
- Assist medical professionals in faster preliminary screening
- Serve as an educational tool for medical students
- Demonstrate the practical application of deep learning in healthcare
**Note**: This is a first version (v1) focused on demonstrating capability rather than achieving
perfect accuracy. The goal is to show potential, not to replace medical professionals.
## ✨ Features
- **27 Skin Condition Classifications**: Covers a wide range of common and rare skin diseases
- **Transfer Learning**: Utilizes EfficientNet-B2 for efficient and accurate predictions
- **Hyperparameter Optimization**: Automated tuning using Optuna for optimal model
performance
- **RESTful API**: FastAPI backend for easy integration
- **Interactive Web UI**: Streamlit-based frontend for user-friendly interaction
- **Dockerized Deployment**: Easy deployment with Docker containerization
- **Class Imbalance Handling**: Implements WeightedRandomSampler (70% improvement)
## 📊 Model Performance
### Overall Metrics
- **Accuracy**: 71%
- **Macro Average Precision**: 64%
- **Macro Average Recall**: 76%
- **Macro Average F1-Score**: 68%
- **Weighted Average F1-Score**: 71%
### Detailed Classification Report
| Skin Condition | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Acne | 0.90 | 0.87 | 0.88 | 138 |
| Actinic keratosis | 0.86 | 0.81 | 0.84 | 148 |
| Atopic Dermatitis | 0.35 | 0.55 | 0.43 | 73 |
| Bullous Disease Photos | 0.64 | 0.65 | 0.64 | 117 |
| Chickenpox | 0.67 | 0.97 | 0.79 | 32 |
| Dermatitis | 0.40 | 0.56 | 0.47 | 71 |
| Dermatofibroma | 0.88 | 0.97 | 0.92 | 29 |
| Dry Skin | 0.83 | 0.74 | 0.78 | 100 |
| Eczema | 0.72 | 0.43 | 0.54 | 320 |
| Exanthems And Drug Eruptions | 0.47 | 0.65 | 0.54 | 139 |
| Hair Loss (Alopecia) | 0.26 | 0.87 | 0.40 | 30 |
| Herpes | 0.43 | 0.84 | 0.57 | 85 |
| Hidradenitis Suppurativa | 0.65 | 0.92 | 0.76 | 12 |
| Light Diseases & Pigmentation Disorders | 0.51 | 0.63 | 0.56 | 89 |
| Lupus & Connective Tissue Diseases | 0.48 | 0.74 | 0.58 | 109 |
| Nail Fungus & Other Nail Diseases | 0.96 | 0.79 | 0.87 | 890 |
| Oily Skin | 0.81 | 0.93 | 0.87 | 145 |
| Other diseases | 0.24 | 0.42 | 0.30 | 12 |
| Cutaneous Larva Migrans | 0.83 | 1.00 | 0.91 | 15 |
| Poison Ivy & Contact Dermatitis | 0.21 | 0.58 | 0.31 | 12 |
| Psoriasis, Lichen Planus & Related | 0.68 | 0.39 | 0.50 | 411 |
| Rashes | 0.72 | 0.75 | 0.73 | 154 |
| Rosacea | 0.83 | 0.92 | 0.87 | 64 |
| Shingles | 0.90 | 0.90 | 0.90 | 20 |
| Urticaria (Hives) | 0.46 | 0.82 | 0.59 | 33 |
| Vitiligo | 0.88 | 0.78 | 0.83 | 100 |
| Warts | 0.81 | 0.94 | 0.87 | 144 |

### Model Architecture
- **Base Model**: EfficientNet-B2 (Transfer Learning)
- **Training Platform**: Kaggle (Free GPU)
- **Hyperparameter Tuning**: Optuna (Google Colab)
- **Framework**: PyTorch
**Note**: Initial attempts with training from scratch resulted in poor performance and required
excessive computational resources. Transfer learning with EfficientNet-B2 (selected through
Optuna optimization) provided the best balance of performance and efficiency.

## 🏥 Supported Skin Conditions
The classifier can identify 27 different skin conditions:
1. Acne
2. Actinic Keratosis
3. Atopic Dermatitis
4. Bullous Disease
5. Chickenpox
6. Dermatitis
7. Dermatofibroma
8. Dry Skin
9. Eczema
10. Exanthems and Drug Eruptions
11. Hair Loss (Alopecia and Other Hair Diseases)
12. Herpes
13. Hidradenitis Suppurativa
14. Light Diseases and Disorders of Pigmentation
15. Lupus and Other Connective Tissue Diseases
16. Nail Fungus and Other Nail Diseases
17. Oily Skin
18. Other Diseases
19. Cutaneous Larva Migrans
20. Poison Ivy and Other Contact Dermatitis
21. Psoriasis, Lichen Planus and Related Diseases
22. Rashes
23. Rosacea
24. Shingles
25. Urticaria (Hives)
26. Vitiligo
27. Warts

## ️ Tech Stack
### Machine Learning & Data Science
- **PyTorch** - Deep learning framework
- **TorchVision** - Image transformations and models
- **EfficientNet-B2** - Transfer learning backbone
- **Optuna** - Hyperparameter optimization
- **scikit-learn** - Metrics and evaluation
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
### Backend & API
- **FastAPI** - High-performance REST API
- **Pydantic** - Data validation
### Frontend
- **Streamlit** - Interactive web application
### Image Processing
- **PIL (Pillow)** - Image handling
### Deployment
- **Docker** - Containerization
- **Python 3.8+** - Programming language
### Development & Training
- **Kaggle** - Model training (Free GPU)
- **Google Colab** - Hyperparameter tuning

## 📁 Project Structure
```
SKIN_DISEASE_CLASSIFIER/
│
├── artifacts/ # Duplicate folder (to be cleaned)
├── artifacts/ # Model artifacts
│ └── best_model.pth # Trained model weights
│
├── config/
│ └── config.yaml # Configuration file
│
├── fastapi_backend/
│ ├── main.py # FastAPI application
│ └── schemas.py # Pydantic schemas
│
├── model_creation_notebook/
│ ├── training.ipynb # Model training notebook (Kaggle)
│ └── optuna-hyp.ipynb # Hyperparameter tuning (Colab)
│
├── src/
│ └── classifier/
│ ├── constants/
│ │ └── constants.py # Project constants
│ ├── entity/
│ │ └── config_entity.py # Configuration entities
│ ├── pipeline/
│ │ ├── predict.py # Prediction pipeline
│ │ └── preprocess.py # Preprocessing pipeline
│ ├── utils/
│ │ └── utilities.py # Utility functions
│ └── __init__.py # Logger initialization
│
├── .dockerignore # Docker ignore file
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── app.py # Streamlit application
├── conditions.py # Skin condition definitions
├── dockerfile # Docker configuration
├── local_testing.py # Local testing script
├── requirements.txt # Python dependencies
└── setup.py # Package setup file
```

## 🚀 Installation
### Option 1: Using Git (Recommended for Development)
```bash
# Clone the repository
git clone https://github.com/mehraj-alom/SKIN_DISEASE_CLASSIFIER.git
cd SKIN_DISEASE_CLASSIFIER
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Download the trained model (if not included in repo)
# Place the model.pth file in the artifacts/ folder
```


### Option 2: Using Docker (Recommended for Production)
<!-- ```bash
# Pull the Docker image
docker pull <mehrajalom>/skinv0:latest
# Run the container
docker run -p 8501:8501 -p 8000:8000
<your-dockerhub-username>/skin-disease-classifier:latest
``` --> #Will be available from the future versions

<!-- ### Building Docker Image Locally
```bash
# Build the image
docker build -t skin-disease-classifier .
# Run the container
docker run -p 8501:8501 -p 8000:8000 skin-disease-classifier -->
#Will be available from the future versions
```

## 💻 Usage
### Running the Streamlit App
```bash
# Start the Streamlit web interface
streamlit run app.py
```
The app will be available at `http://localhost:8501`
### Running the FastAPI Backend
```bash
# Start the FastAPI server
python -m fastapi_backend.main
```
The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Local Testing
```bash
# Run local tests
python local_testing.py
```


## 🎓 Dataset
**Source**: [Kaggle - Image Dataset for Skin
Diseases](https://www.kaggle.com/datasets/sd20co001/image-dataset-for-skindiseases-dry-oily￾normalskin)
**Characteristics**:
- Publicly available and freely accessible
- Contains images for around 47 different skin conditions but i reduced the classes to 27
- Includes various skin types and conditions
- **Class Imbalance**: Significant variation in sample sizes across classes (addressed with
WeightedRandomSampler)
## 🚧 Challenges Faced
### 1. Class Imbalance
- **Problem**: Significant disparity in the number of samples across different skin conditions
- **Solution**: Implemented WeightedRandomSampler, achieving 70% improvement in handling
minority classes
- **Impact**: Better model generalization across all conditions
### 2. Computational Resources
- **Problem**: Notebook timeouts on Kaggle and Google Colab
- **Solution**:
- Used Kaggle's free GPU for main training
- Utilized Google Colab for hyperparameter tuning
- Implemented checkpointing to save progress
### 3. Medical Domain Knowledge
- **Problem**: Limited medical/dermatology background
- **Solution**:
- Researched each condition using Google and ChatGPT
- Studied medical literature and dermatology resources
- Gained foundational understanding before model development
- Consulted medical students(friends) for condition characteristics
### 4. Model Architecture Selection
- **Problem**: Training from scratch was resource-intensive and produced poor results
- **Solution**:
- Switched to transfer learning approach
- Used Optuna for automated architecture and hyperparameter search
- Selected EfficientNet-B2 for optimal balance of accuracy and efficiency
## 🔮 Future Roadmap
### Version 2.0 (Planned)
- **Vision Transformers (ViT)**: Implement state-of-the-art transformer architectures
- **Enhanced Performance**: Target 80%+ accuracy
- **Better Generalization**: Improved performance on minority classes
### Version 3.0 (Planned)
- **Object Detection**: Implement YOLO/Faster R-CNN for locating affected skin areas
- **Multi-lesion Detection**: Identify multiple conditions in a single image
- **Bounding Box Visualization**: Highlight specific problem areas
### Version 4.0 (Planned)
- **Image Segmentation**: Pixel-level classification using U-Net/Mask R-CNN
- **Detailed Area Analysis**: Precise mapping of affected regions
- **Severity Assessment**: Quantify the extent of skin conditions
### Version 5.0 (Planned)
- **Advanced Optimization**: Ensemble methods, attention mechanisms
- **Performance Boost**: Target 90%+ accuracy
- **Real-time Processing**: Optimized inference speed
- **Mobile Deployment**: Lightweight models for mobile devices

## 🤝 Contributing
We welcome contributions from the community! Whether you're a junior developer learning ML
or a senior expert, your help is appreciated.
### For Junior Developers & Students
- Want to learn about medical AI? This is a great project to start!
- Looking to collaborate and build your portfolio? Let's work together!
- Interested in contributing to open-source healthcare projects? Join us!
**How to reach out**:
- Open an issue on GitHub
- Comment on the repository
- Send a DM with your ideas or questions
### For Senior Developers & Medical Professionals
Your guidance would be invaluable! If you can spare some time to:
- Review our approach and suggest improvements
- Share best practices in medical ML
- Provide mentorship to the team
- Contribute domain expertise in dermatology
**Your help will be forever appreciated!** We're building this for educational purposes and to
demonstrate AI's potential in healthcare. Any guidance, code review, or suggestions are
welcome

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
## 📄 License
This project is licensed under the MIT License - see more about the [LICENSE] in google
## ⚠️ Medical Disclaimer
**CRITICAL**: This software is intended for **educational and research purposes ONLY**. It is
NOT:
- A medical device
- A diagnostic tool for clinical use
- A replacement for professional medical advice
- Certified or approved by any medical authority
**Always consult qualified healthcare professionals** for medical diagnosis, treatment, and
advice. Never make medical decisions based solely on this software's output.
## 🙏 Acknowledgments
- **Kaggle** for providing free GPU resources for model training
- **Google Colab** for computational resources for hyperparameter tuning
- The **Kaggle dataset creator** for making the skin disease dataset publicly available
- The **open-source community** for excellent tools and libraries
- **PyTorch and EfficientNet teams** for the amazing deep learning frameworks
- **Medical professionals** who create educational resources about dermatology
## 📞 Contact & Support
- **GitHub Issues**: For bug reports and feature requests
[![LinkedIn](https://cdn-icons-png.flaticon.com/512/174/174857.png)](https://www.linkedin.com/in/mehraj-alom-tapadar-b1abb025b)

- **Repository**:
[SKIN_DISEASE_CLASSIFIER](https://github.com/mehraj-alom/SKIN_DISEASE_CLASSIFIER)
- **Discussions**: Open for collaboration and questions
---
**Note**: This is Version 1.0 - a proof of concept demonstrating AI's potential in dermatology
assistance. The focus is on showing capability and encouraging collaboration rather than
production-ready accuracy. Together, we can improve this tool and contribute to the future of
AI-assisted healthcare! 🚀
**Star ⭐ this repository if you find it useful!**