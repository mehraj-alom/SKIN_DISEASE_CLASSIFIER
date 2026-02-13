import os
from pathlib import Path
from typing import Union, IO
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torch import load as torch_load

from src.classifier.pipeline.preprocess import Transform
from src.classifier import logger
from src.classifier.entity.config_entity import PredictionConfig

class Prediction_Pipeline:
    """
    A pipeline for loading a trained PyTorch model and performing image predictions.

    This class supports:
    - File paths (str or Path)
    - In-memory file-like objects (e.g., Streamlit UploadedFile)
    - PIL.Image instances

    Parameters
    ----------
    config : PredictionConfig
        Configuration object containing model path, device, and optional filename.

    Attributes
    ----------
    config : PredictionConfig
        Configuration object provided during initialization.
    model_path : str
        Path to the trained PyTorch model file.
    device : str
        Device identifier for PyTorch computation (e.g., 'cpu' or 'cuda').
    filename : str | None
        Optional model filename from the configuration.
    model : torch.nn.Module | None
        Loaded PyTorch model, or `None` if loading failed.
    transform : callable | None
        Function to preprocess input images for prediction.
    """

    def __init__(self, config: PredictionConfig):
        """
        Initialize the prediction pipeline with configuration parameters.

        Raises
        ------
        TypeError
            If `config` is not an instance of `PredictionConfig`.
        """
        if not isinstance(config, PredictionConfig):
            raise TypeError(f"Expected config to be PredictionConfig, got {type(config)}")

        self.config = config
        self.model_path = config.model_path
        self.device = config.device
        self.filename = getattr(config, "filename", None)

        # Load the PyTorch model
        self.model = self.load_model()
        if self.model is None:
            logger.warning("Model not loaded. Predictions will fail until a valid model is provided.")

        # Set up transform
        try:
            self.transform = Transform.preprocess_image
        except AttributeError:
            logger.error("Transform.preprocess_image not found. Ensure Transform is correctly implemented.")
            self.transform = None

    def load_model(self) -> Union[torch.nn.Module, None]:
        """
        Load a PyTorch model from the specified file path and move it to the configured device.

        Returns
        -------
        torch.nn.Module | None
            Loaded PyTorch model on the specified device, or `None` if loading fails.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return None
        try:
            model = torch_load(self.model_path, map_location=self.device, weights_only=False)
            if hasattr(model, "to"):
                model = model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            return None

    def predict(self, image_input: Union[str, Path, IO, Image.Image]) -> Union[np.ndarray, None]:
        """
        Predict class probabilities for a given image.

        Supports:
        - File path as str or Path
        - File-like objects (e.g., Streamlit UploadedFile)
        - PIL.Image instances

        Parameters
        ----------
        image_input : Union[str, Path, IO, PIL.Image.Image]
            The input image to predict.

        Returns
        -------
        np.ndarray | None
            Probability distribution over classes as a NumPy array, or `None` if
            prediction fails due to missing files, preprocessing errors, or model issues.
        """
        if self.model is None:
            logger.error("Model is not loaded. Prediction aborted.")
            return None
        if self.transform is None:
            logger.error("Transform function not available. Prediction aborted.")
            return None

        try:
            # If input is a file path
            if isinstance(image_input, (str, Path)):
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return None
                image_tensor = self.transform(str(image_input)).unsqueeze(0).to(self.device)

            # If input is a PIL.Image instance
            elif isinstance(image_input, Image.Image):
                image_tensor = self.transform(image_input).unsqueeze(0).to(self.device)

            # If input is a file-like object (Streamlit UploadedFile)
            else:
                image = Image.open(image_input)
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        except (UnidentifiedImageError, ValueError, OSError) as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None

        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = nn.Softmax(dim=1)(output)
            logger.info(f"Prediction successful.")
            return probabilities.cpu().numpy()
        except Exception as e:
            logger.exception(f"Model inference failed: {e}")
            return None
