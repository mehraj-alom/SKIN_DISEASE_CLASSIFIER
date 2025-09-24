import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from pathlib import Path
from torch import load as torch_load
from src.classifier.pipeline.preprocess import Transform
from src.classifier import logger


class Prediction_Pipeline:
    """
    Pipeline for loading a trained PyTorch model and making predictions on input images.

    Attributes
    ----------
    model_path : str
        Path to the saved PyTorch model file.
    filename : str
        Name of the input image file (not directly used in this class).
    device : str
        Device to run the model on ('cpu' or 'cuda').
    model : torch.nn.Module
        The loaded PyTorch model.
    transform : callable
        Preprocessing function to prepare images for prediction.

    Methods
    -------
    predict(image_path: str) -> np.ndarray
        Loads and preprocesses an image, performs inference, and returns class probabilities.
    """

    def __init__(self, model_path: str,
                 filename : str ,
                 device: str = "cpu"):
        """
        Initializes the Prediction_Pipeline with model path, image filename, and device.

        Parameters
        ----------
        model_path : str
            Path to the saved PyTorch model file.
        filename : str
            Name of the input image file.
        device : str, optional
            Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()
        self.transform = Transform.preprocess_image

    def load_model(self):
        """
        Loads the PyTorch model from the specified path and moves it to the selected device.

        Returns
        -------
        torch.nn.Module
            The loaded model.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        RuntimeError
            If the model cannot be loaded.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return None
        try:
            model = torch_load(self.model_path, map_location=self.device)
            if hasattr(model, 'to'):
                model = model.to(self.device)
            return model
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            return None

    def predict(self, image_path: str) -> np.ndarray:
        """
        Preprocesses the input image, performs model inference, and returns class probabilities.

        Parameters
        ----------
        image_path : str
            Path to the input image file.

        Returns
        -------
        np.ndarray
            Array of class probabilities predicted by the model.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        ValueError
            If the image cannot be opened or preprocessing fails.
        RuntimeError
            If model inference fails.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        try:
            image_tensor = self.transform(image_path).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, ValueError, OSError) as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None
        if self.model is None:
            logger.error("Model is not loaded. Prediction aborted.")
            return None
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = nn.Softmax(dim=1)(output)
            return probabilities.cpu().numpy()
        except Exception as e:
            logger.exception(f"Model inference failed: {e}")
            return None