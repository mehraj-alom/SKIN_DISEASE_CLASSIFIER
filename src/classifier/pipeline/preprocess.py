from torchvision.transforms import v2 as transform
from src.classifier.constants import IMAGENET_MEAN, IMAGENET_STD
import torch
from PIL import Image

class Transform:
    """
    Utility class for image preprocessing transformations used in the prediction pipeline.
    """

    @staticmethod
    def preprocess_image(image):
        """
        Loads and preprocesses an image for model prediction.

        Handles:
        - Loading from file path or PIL Image
        - Converting grayscale to RGB
        - Resizing, tensor conversion, normalization

        Parameters
        ----------
        image : str or PIL.Image.Image
            Path to image file or PIL Image object.

        Returns
        -------
        torch.Tensor
            The preprocessed image tensor.

        Raises
        ------
        ValueError
            If the input is not a valid image or preprocessing fails.
        """
        # Load image if a path is provided
        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Cannot open image: {e}")
        # Convert grayscale to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            transform_pipeline = transform.Compose([
                transform.Resize((380, 380)),
                transform.ToImage(),
                transform.ToDtype(torch.float32, scale=True),
                transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            return transform_pipeline(image)
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {e}")