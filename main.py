from src.classifier.pipeline.predict import Prediction_Pipeline
from src.classifier.pipeline.preprocess import Transform
from src.classifier.entity.config_entity import PredictionConfig
from src.classifier import logger
from pathlib import Path
from src.classifier.utils.class_mapping import cls_to_idx, all_labels
import torch 


CURRENT_STAGE = "Preprocess"

try: 
    logger.info(f">>>>> stage {CURRENT_STAGE} started <<<<<")
    # Example image path for preprocessing
    image_path = "test_data/1Dermatofibroma.jpeg"
    tensor = Transform.preprocess_image(image_path)
    logger.info(f"Preprocessing successful. Tensor shape: {tensor.shape}")
    logger.info(f">>>>> stage {CURRENT_STAGE} completed <<<<<\n\nx=================================x")
except Exception as e:
    logger.exception(e)
    raise e

CURRENT_STAGE = "Preprocess"
try: 
    logger.info(f">>>>> stage {CURRENT_STAGE} started <<<<<")
    
    # Path to the image you want to predict
    image_path = "test_data/1Dermatofibroma.jpeg"
    
    # Preprocess image
    tensor = Transform.preprocess_image(image_path)
    logger.info(f"Preprocessing successful. Tensor shape: {tensor.shape}")

    logger.info(f">>>>> stage {CURRENT_STAGE} completed <<<<<\n\nx=================================x")
except Exception as e:
    logger.exception(e)
    raise e


CURRENT_STAGE = "Prediction"
try:
    logger.info(f">>>>> stage {CURRENT_STAGE} started <<<<<")

    config = PredictionConfig(
        model_path=Path("artifacts/best_model/best_skin_model_full.pth"),
        filename="test_image.jpg", 
        device="cpu"  
    )
    
    # Initialize prediction pipeline
    prediction_pipeline = Prediction_Pipeline(config)
    
    # Get probabilities
    probs = prediction_pipeline.predict()
    
    if probs is not None:
        # Convert to tensor if necessary
        probs_tensor = torch.tensor(probs) if not isinstance(probs, torch.Tensor) else probs
        
        # Get top 2 predictions
        top2_probs, top2_indices = torch.topk(probs_tensor, k=2, dim=-1)
        top2_classes = [all_labels[i] for i in top2_indices.tolist()[0]]
    
        top2_results = {
            "classes": top2_classes,
            "probabilities": top2_probs.tolist()[0]
        }
        
        logger.info(f"Top 2 predicted classes: {top2_results['classes']}")
        logger.info(f"Top 2 probabilities: {top2_results['probabilities']}")
        logger.info(f"Prediction successful. Full probabilities: {probs}")
        
    else:
        logger.error("Prediction failed.")
    
    logger.info(f">>>>> stage {CURRENT_STAGE} completed <<<<<\n\nx=================================x")
    
except Exception as e:
    logger.exception(e)
    raise e