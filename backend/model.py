import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoofDetectorModel(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for Voice Spoofing Detection.
    designed to work with MFCC inputs of shape (batch, 1, n_mfcc, time_steps).
    """
    def __init__(self, n_mfcc=40, num_classes=2):
        super(SpoofDetectorModel, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle variable time lengths if needed
        # (though we fix length in preprocessing, this adds robustness)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, n_mfcc, time_steps)
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelHandler:
    """
    Singleton class to handle model loading and inference.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """
        Loads the pretrained model weights from checkpoint.
        If no weights found, initializes a fresh model (for demo purposes).
        """
        self.model = SpoofDetectorModel().to(self.device)
        self.model.eval()
    
        # Path to the checkpoint
        model_path = os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")
    
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
    
                # Handle checkpoint vs raw weights
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Loaded model_state_dict from checkpoint.")
                else:
                    state_dict = checkpoint
                    logger.info("Loaded raw model weights.")
    
                self.model.load_state_dict(state_dict)
                logger.info(f"Model weights successfully loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}. Using random initialization.")
        else:
            logger.warning(f"Model weights not found at {model_path}. Using random initialization for demonstration.")


    def predict(self, features: np.ndarray):
        """
        Performs inference on the given features.
        
        Args:
            features (np.ndarray): MFCC features of shape (n_mfcc, time_steps).
            
        Returns:
            dict: { "label": "Genuine"|"Spoofed", "confidence": float }
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        # Prepare input tensor: Add batch and channel dimensions -> (1, 1, n_mfcc, time_steps)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Assuming Class 0 = Genuine, Class 1 = Spoofed
            spoofed_prob = probabilities[0][1].item()
            genuine_prob = probabilities[0][0].item()
            
            prediction = torch.argmax(probabilities, dim=1).item()
            
            label = "Spoofed" if prediction == 1 else "Genuine"
            confidence = spoofed_prob if prediction == 1 else genuine_prob
            
            # Make sure we return a Python float, not a Tensor
            return {
                "label": label,
                "confidence": float(confidence),
                "details": {
                    "genuine_prob": float(genuine_prob),
                    "spoofed_prob": float(spoofed_prob)
                }
            }
