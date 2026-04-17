"""
Model service for loading and running inference with the EfficientNet model

This module handles:
- Loading the pre-trained EfficientNet model
- Image preprocessing and validation
- Running inference on images
- Resource cleanup
"""

import io
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model loading and inference
    
    Handles loading an EfficientNet-based 3-class classifier and running predictions
    on uploaded images to detect real vs AI-generated vs suspicious faces.
    """

    def __init__(self, model_path: str = "models/efficientnet3class_full_model.pth"):
        """
        Initialize the model service

        Args:
            model_path: Path to the saved PyTorch model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        logger.info(f"Initializing ModelService...")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        self._validate_model_path()
        self._load_model()
        self._setup_transforms()

    def _validate_model_path(self) -> None:
        """Validate that model file exists"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Please ensure MODEL_PATH environment variable is set correctly."
            )
        logger.info(f"✓ Model file found at {self.model_path}")

    def _load_model(self) -> None:
        """Load the EfficientNet model from file
        
        Supports loading from:
        - Full model objects (torch.nn.Module)
        - State dicts (model weights only)
        - Checkpoint dicts with 'state_dict' or 'model_state_dict' key
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            logger.info("Loading model checkpoint...")
            
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    # Checkpoint with explicit model_state_dict key
                    self._build_and_load_model(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    # Checkpoint with state_dict key
                    self._build_and_load_model(checkpoint["state_dict"])
                else:
                    # Assume it's a state_dict directly
                    self._build_and_load_model(checkpoint)
            else:
                # Full model object
                self.model = checkpoint
                self.model.to(self.device)
                self.model.eval()

            logger.info("✓ Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _build_and_load_model(self, state_dict) -> None:
        """Build EfficientNet model and load weights
        
        Args:
            state_dict: Model weights dictionary
        """
        try:
            from torchvision import models

            logger.info("Building EfficientNet B0 architecture...")
            self.model = models.efficientnet_b0(weights=None)
            
            # Modify classifier for 3 classes
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_features, 3)
            
            # Load state dict
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ Model architecture built and weights loaded")
            
        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            raise

    def _setup_transforms(self) -> None:
        """Setup image transforms for the model
        
        Uses standard ImageNet normalization values:
        - Mean: [0.485, 0.456, 0.406]
        - Std: [0.229, 0.224, 0.225]
        """
        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("✓ Image transforms setup complete")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on an image

        Args:
            image_bytes: Image as bytes (from file upload)

        Returns:
            Dictionary with:
            - verdict: "real", "suspicious", or "fake"
            - confidence: confidence percentage (0-100)
            - scores: dict with confidence for each class

        Raises:
            ValueError: If image is invalid or processing fails
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.debug(f"Image loaded: size={image.size}, mode={image.mode}")

            # Validate image size
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError(
                    f"Image is too small: {image.size}. Minimum size is 32x32 pixels."
                )

            if image.size[0] > 4096 or image.size[1] > 4096:
                raise ValueError(
                    f"Image is too large: {image.size}. Maximum size is 4096x4096 pixels."
                )

            # Transform and prepare batch
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            logger.debug(f"Image tensor shape: {image_tensor.shape}")

            # Run inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = F.softmax(logits, dim=1)
                scores = probabilities.cpu().numpy()[0]

            # Extract predictions
            # Class order: FAKE (index 0), OTHERS (index 1), REAL (index 2)
            # Alphabetically sorted from ImageFolder training
            class_names = ["fake", "suspicious", "real"]
            predicted_class_idx = np.argmax(scores)
            predicted_class = class_names[predicted_class_idx]
            confidence = float(scores[predicted_class_idx] * 100)

            # Create scores dictionary
            scores_dict = {
                "real": float(scores[2] * 100),      # Index 2 = REAL
                "suspicious": float(scores[1] * 100), # Index 1 = OTHERS
                "fake": float(scores[0] * 100),       # Index 0 = FAKE
            }

            logger.debug(
                f"Prediction - Class: {predicted_class}, Confidence: {confidence:.2f}%, "
                f"Scores: {scores_dict}"
            )

            return {
                "verdict": predicted_class,
                "confidence": round(confidence, 2),
                "scores": {k: round(v, 2) for k, v in scores_dict.items()},
            }

        except Image.UnidentifiedImageError as e:
            logger.error(f"Invalid image file: {e}")
            raise ValueError("Invalid or corrupted image file. Please upload a valid image.")
        except Image.DecompressionBombError as e:
            logger.error(f"Decompression bomb detected: {e}")
            raise ValueError("Image is too large or corrupted.")
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise ValueError(f"Failed to process image: {str(e)}")

    def cleanup(self) -> None:
        """Cleanup resources and free memory"""
        try:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                logger.info("✓ Model cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
