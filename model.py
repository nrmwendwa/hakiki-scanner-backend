"""
Model loading and inference logic for face authenticity detection.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import settings

logger = logging.getLogger(__name__)


class FaceAuthenticityModel:
    """
    Wrapper for the EfficientNet face authenticity detection model.
    
    This model classifies images into three categories:
    - real: Genuine/real faces
    - suspicious: Potentially manipulated or unclear
    - fake: AI-generated or synthetic faces
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            model_path: Path to the model file. Uses config if not provided.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If CUDA is requested but not available
        """
        self.device = self._get_device()
        self.model = None
        self.model_path = Path(model_path) if model_path else settings.get_model_path()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self._load_model()
        self._setup_transforms()
        logger.info(f"Model loaded successfully from {self.model_path} on device: {self.device}")

    def _get_device(self) -> torch.device:
        """Determine the device to use (CUDA or CPU)."""
        if settings.DEVICE.lower() == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> None:
        """Load the model checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint saved with state_dict
                self.model = self._create_model()
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Alternative checkpoint format
                self.model = self._create_model()
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume it's a direct state_dict or full model
                try:
                    self.model = checkpoint if isinstance(checkpoint, nn.Module) else self._create_model()
                    if not isinstance(checkpoint, nn.Module):
                        self.model.load_state_dict(checkpoint)
                except Exception:
                    self.model = self._create_model()
                    self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def _create_model(self) -> nn.Module:
        """
        Create an EfficientNet model for 3-class classification.
        Adjust this if your model architecture is different.
        """
        try:
            from torchvision.models import efficientnet_b0
            
            model = efficientnet_b0(weights=None)
            # Replace the classifier for 3 classes
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 3)
            )
            return model
        except Exception as e:
            logger.error(f"Failed to create model architecture: {str(e)}")
            raise

    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image) -> dict:
        """
        Make a prediction on an image.
        
        Args:
            image: PIL Image object
        
        Returns:
            Dictionary containing:
            - verdict: "real", "suspicious", or "fake"
            - confidence: confidence percentage (0-100)
            - scores: dict with individual class scores
        
        Raises:
            ValueError: If image is invalid
        """
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess image
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                scores = probs[0].cpu().numpy()
            
            # Map class indices to labels
            class_names = ["real", "suspicious", "fake"]
            predicted_idx = np.argmax(scores)
            
            # Convert to percentages
            scores_dict = {
                class_names[i]: float(scores[i] * 100) for i in range(3)
            }
            
            verdict = class_names[predicted_idx]
            confidence = float(scores[predicted_idx] * 100)
            
            return {
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "scores": {
                    "real": round(scores_dict["real"], 2),
                    "suspicious": round(scores_dict["suspicious"], 2),
                    "fake": round(scores_dict["fake"], 2),
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")


# Global model instance
_model_instance: Optional[FaceAuthenticityModel] = None


def get_model() -> FaceAuthenticityModel:
    """Get or create the global model instance (lazy loading)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FaceAuthenticityModel()
    return _model_instance


def load_model() -> FaceAuthenticityModel:
    """Load the model at startup."""
    global _model_instance
    _model_instance = FaceAuthenticityModel()
    return _model_instance
