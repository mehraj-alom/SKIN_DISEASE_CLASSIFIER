import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from pathlib import Path
from torch import load as torch_load


class Prediction_Pipeline:
    def __init__(self, model_path: str,
                 filename : str ,
                 device: str = 'cpu'):
        
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()
        self.transform = self.get_transform()
    
    def predict(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probabilities = nn.Softmax(dim=1)(output)
        
        return probabilities.cpu().numpy()