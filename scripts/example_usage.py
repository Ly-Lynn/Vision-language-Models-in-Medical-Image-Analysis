#!/usr/bin/env python3
"""
Example usage of VisionLanguageModel base class with MedCLIP and BioMedCLIP
"""

import torch
from PIL import Image
from modules.models.medclip import MedCLIPModel
from modules.models.biomedclip import BioMedCLIPModel


def demo_medclip():
    """Demo using MedCLIP with base class"""
    print("=== Demo MedCLIP ===")
    
    # Initialize model
    model = MedCLIPModel(
        text_encoder_type='bert',
        vision_encoder_type='vit'
    )
    
    # Print model info
    print("Model info:", model.get_model_info())
    
    # Demo text encoding
    texts = ["A chest X-ray showing normal lungs", "Medical image analysis"]
    text_embeddings = model.encode_text(texts=texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Demo image encoding (simulated)
    dummy_image = torch.randn(1, 3, 224, 224)
    image_embeddings = model.encode_image(dummy_image)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Demo forward pass
    outputs = model.forward(
        pixel_values=dummy_image,
        texts=texts,
        return_loss=False
    )
    print(f"Forward outputs keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")


def demo_biomedclip():
    """Demo using BioMedCLIP with base class"""
    print("\n=== Demo BioMedCLIP ===")
    
    # Initialize model
    model = BioMedCLIPModel()
    
    # Print model info
    print("Model info:", model.get_model_info())
    
    # Demo text encoding
    texts = ["A chest X-ray showing normal lungs", "Medical image analysis"]
    text_embeddings = model.encode_text(texts=texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Demo image encoding (simulated)
    dummy_image = torch.randn(1, 3, 224, 224)
    image_embeddings = model.encode_image(dummy_image)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Demo forward pass
    outputs = model.forward(
        pixel_values=dummy_image,
        texts=texts,
        return_loss=False
    )
    print(f"Forward outputs keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")


def demo_checkpoint_loading():
    """Demo loading checkpoint with base class"""
    print("\n=== Demo Checkpoint Loading ===")
    
    # Initialize model
    model = MedCLIPModel()
    
    # Demo loading checkpoint (will fail since file doesn't exist)
    model.load_checkpoint("nonexistent_checkpoint.pth", strict=False)
    
    # Demo with empty checkpoint path
    model.load_checkpoint("", strict=False)


if __name__ == "__main__":
    print("🚀 Demo Vision-Language Models with Base Class")
    
    try:
        demo_medclip()
        demo_biomedclip()
        demo_checkpoint_loading()
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
