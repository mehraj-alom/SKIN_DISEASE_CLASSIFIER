from torchvision.transforms import v2 as transform
from src.classifier.constants import IMAGENET_MEAN, IMAGENET_STD

class transform:
    
    @staticmethod
    def get_transform():
        return transform.Compose([
            transform.Resize((224, 224)),
            transform.ToTensor(),
            transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])