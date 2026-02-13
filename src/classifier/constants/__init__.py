from pathlib import Path

MODEL_FILE_PATH = Path("artifacts/best_model/best_skin_model (10).pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")  