#!/usr/bin/env python3
"""
Inference script for medical vision-language models
"""

import argparse
import yaml
import torch
import json
import sys
import os
from PIL import Image
import numpy as np

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.models.factory import create_model
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path: str, model_type: str = 'medclip'):
    """Preprocess image for inference"""
    # This is a simplified preprocessing - you should use the actual preprocessing
    # from your dataset modules
    image = Image.open(image_path).convert('RGB')
    
    # Basic preprocessing (you should replace this with proper preprocessing)
    from torchvision import transforms
    
    if model_type == 'medclip':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # biomedclip or others
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    return pixel_values


def run_zero_shot_classification(model, image_path: str, class_names: list, 
                                templates: list = None, device: str = 'cpu'):
    """Run zero-shot classification on a single image"""
    logger.info(f"🔍 Running zero-shot classification on {image_path}")
    
    # Preprocess image
    model_type = getattr(model, 'model_name', 'medclip')
    pixel_values = preprocess_image(image_path, model_type).to(device)
    
    # Create text prompts
    if templates is None:
        templates = ['this is a photo of {}']
    
    text_prompts = []
    for class_name in class_names:
        for template in templates:
            text_prompts.append(template.format(class_name))
    
    with torch.no_grad():
        # Encode image
        if hasattr(model, 'encode_image'):
            image_features = model.encode_image(pixel_values, normalize=True)
        else:
            logger.error("Model does not have encode_image method")
            return None
        
        # Encode text
        if hasattr(model, 'encode_text'):
            text_features = model.encode_text(text_prompts, normalize=True)
        else:
            logger.error("Model does not have encode_text method")
            return None
        
        # Compute similarities
        similarities = torch.matmul(image_features, text_features.t())
        
        # Aggregate similarities per class if multiple templates
        if len(templates) > 1:
            similarities = similarities.view(1, len(class_names), len(templates))
            logits = similarities.mean(dim=-1)
        else:
            logits = similarities
        
        # Get predictions
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx].item()
    
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_probabilities': {
            class_names[i]: probabilities[0, i].item() 
            for i in range(len(class_names))
        }
    }
    
    return results


def run_text_image_similarity(model, image_path: str, text_query: str, device: str = 'cpu'):
    """Compute similarity between image and text"""
    logger.info(f"🔍 Computing similarity between image and text")
    
    # Preprocess image
    model_type = getattr(model, 'model_name', 'medclip')
    pixel_values = preprocess_image(image_path, model_type).to(device)
    
    with torch.no_grad():
        # Encode image
        if hasattr(model, 'encode_image'):
            image_features = model.encode_image(pixel_values, normalize=True)
        else:
            logger.error("Model does not have encode_image method")
            return None
        
        # Encode text
        if hasattr(model, 'encode_text'):
            text_features = model.encode_text([text_query], normalize=True)
        else:
            logger.error("Model does not have encode_text method")
            return None
        
        # Compute similarity
        similarity = torch.matmul(image_features, text_features.t()).item()
    
    results = {
        'image_path': image_path,
        'text_query': text_query,
        'similarity_score': similarity
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with medical vision-language model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--task', type=str, choices=['classification', 'similarity'],
                       default='classification', help='Inference task')
    parser.add_argument('--text', type=str,
                       help='Text query for similarity task')
    parser.add_argument('--output_file', type=str, default='inference_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"📋 Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = create_model(**model_config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"📥 Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    logger.info(f"🏗️ Model ready for inference")
    
    # Run inference based on task
    if args.task == 'classification':
        # Get class names from config
        inference_config = config.get('inference', {})
        class_names = inference_config.get('class_names', ['Normal', 'Abnormal'])
        templates = inference_config.get('templates')
        
        results = run_zero_shot_classification(
            model, args.image, class_names, templates, device
        )
        
        if results:
            logger.info("📊 Classification Results:")
            logger.info(f"  Predicted Class: {results['predicted_class']}")
            logger.info(f"  Confidence: {results['confidence']:.4f}")
            logger.info("  Class Probabilities:")
            for class_name, prob in results['class_probabilities'].items():
                logger.info(f"    {class_name}: {prob:.4f}")
    
    elif args.task == 'similarity':
        if not args.text:
            logger.error("❌ Text query required for similarity task")
            return
        
        results = run_text_image_similarity(model, args.image, args.text, device)
        
        if results:
            logger.info("📊 Similarity Results:")
            logger.info(f"  Image: {results['image_path']}")
            logger.info(f"  Text: {results['text_query']}")
            logger.info(f"  Similarity Score: {results['similarity_score']:.4f}")
    
    # Save results
    if results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {args.output_file}")
    
    logger.info("🎉 Inference completed!")


if __name__ == "__main__":
    main()
