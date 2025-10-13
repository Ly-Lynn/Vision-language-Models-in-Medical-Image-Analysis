"""
Base classes for vision and text encoders
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class TextEncoder(nn.Module, ABC):
    """Abstract base class for text encoders"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Encode text inputs
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for input
            return_features: Whether to return raw features or normalized embeddings
            
        Returns:
            Text embeddings
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the encoder"""
        pass
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights - default implementation"""
        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


class VisionEncoder(nn.Module, ABC):
    """Abstract base class for vision encoders"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Encode image inputs
        
        Args:
            images: Input image tensor
            return_features: Whether to return raw features or normalized embeddings
            
        Returns:
            Image embeddings or logits
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the encoder"""
        pass
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights - default implementation"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.load_state_dict(state_dict, strict=False)
        print("✅ Pretrained weights loaded successfully")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized feature embeddings"""
        return self.forward(x, return_features=True)
