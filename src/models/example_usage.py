"""
Example usage of the model factory and base classes
"""

import torch
from PIL import Image
from factory import ModelFactory, create_medclip, create_biomedclip


def example_basic_usage():
    """Example of basic model creation and usage"""
    print("=" * 60)
    print("Basic Model Usage Examples")
    print("=" * 60)
    
    # Example 1: Create MedCLIP model using factory
    print("\n1. Creating MedCLIP model:")
    medclip = ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        pretrained=False  # Set to True to download pretrained weights
    )
    print(f"   Created: {type(medclip).__name__}")
    
    # Example 2: Create BioMedCLIP model using convenience function
    print("\n2. Creating BioMedCLIP model:")
    biomedclip = create_biomedclip()
    print(f"   Created: {type(biomedclip).__name__}")
    
    # Example 3: Encode text and images
    print("\n3. Encoding text and images:")
    
    # Dummy image tensor
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Encode with BioMedCLIP
    text = "chest x-ray showing pneumonia"
    text_features = biomedclip.encode_text(text)
    image_features = biomedclip.encode_image(dummy_image)
    
    print(f"   Text features shape: {text_features.shape}")
    print(f"   Image features shape: {image_features.shape}")
    
    # Compute similarity
    similarity = (image_features @ text_features.t()).softmax(dim=-1)
    print(f"   Similarity score: {similarity.item():.4f}")


def example_zero_shot_classification():
    """Example of zero-shot classification"""
    print("\n" + "=" * 60)
    print("Zero-Shot Classification Example")
    print("=" * 60)
    
    # Define classes for classification
    class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Lung Cancer']
    
    # Create zero-shot classifier using MedCLIP
    print("\n1. Creating MedCLIP zero-shot classifier:")
    medclip_classifier = ModelFactory.create_zeroshot_classifier(
        model_type='medclip',
        class_names=class_names,
        templates=['a chest x-ray showing {}'],
        ensemble=False
    )
    print(f"   Created classifier for {len(class_names)} classes")
    
    # Create zero-shot classifier using BioMedCLIP
    print("\n2. Creating BioMedCLIP zero-shot classifier:")
    biomedclip_classifier = ModelFactory.create_zeroshot_classifier(
        model_type='biomedclip',
        class_names=class_names,
        templates=['this is a chest x-ray showing {}'],
        ensemble=True
    )
    print(f"   Created classifier for {len(class_names)} classes")
    
    # Perform classification on dummy image
    print("\n3. Performing classification:")
    dummy_batch = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    # Classify with BioMedCLIP
    biomedclip_model = create_biomedclip()
    bio_classifier = ModelFactory.create_classifier(
        model=biomedclip_model,
        task_type='zeroshot',
        class_names=class_names
    )
    
    # Note: Real classification would require proper prompt inputs
    # This is just to demonstrate the structure
    print(f"   Input batch shape: {dummy_batch.shape}")
    print(f"   Classes: {class_names}")


def example_supervised_classification():
    """Example of supervised classification with fine-tuning"""
    print("\n" + "=" * 60)
    print("Supervised Classification Example")
    print("=" * 60)
    
    # Create supervised classifier for binary classification
    print("\n1. Creating binary classifier:")
    binary_classifier = ModelFactory.create_supervised_classifier(
        model_type='biomedclip',
        num_classes=1,  # Binary classification
        task_mode='binary',
        freeze_encoder=True  # Freeze encoder, only train classifier head
    )
    print(f"   Created binary classifier")
    
    # Create supervised classifier for multi-class classification
    print("\n2. Creating multi-class classifier:")
    multiclass_classifier = ModelFactory.create_supervised_classifier(
        model_type='medclip',
        num_classes=4,  # 4 classes
        task_mode='multiclass',
        freeze_encoder=False  # Fine-tune entire model
    )
    print(f"   Created 4-class classifier")
    
    # Create supervised classifier for multi-label classification
    print("\n3. Creating multi-label classifier:")
    multilabel_classifier = ModelFactory.create_supervised_classifier(
        model_type='biomedclip',
        num_classes=14,  # 14 different conditions
        task_mode='multilabel',
        freeze_encoder=True
    )
    print(f"   Created 14-label classifier")
    
    # Example forward pass
    print("\n4. Forward pass example:")
    dummy_batch = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (4,))  # Binary labels
    
    # This would normally be done in training loop
    outputs = binary_classifier(
        pixel_values=dummy_batch,
        labels=dummy_labels,
        return_loss=True
    )
    
    print(f"   Logits shape: {outputs['logits'].shape}")
    if 'loss_value' in outputs:
        print(f"   Loss: {outputs['loss_value'].item():.4f}")


def example_model_comparison():
    """Example comparing different models"""
    print("\n" + "=" * 60)
    print("Model Comparison Example")
    print("=" * 60)
    
    # Create both models
    print("\n1. Creating models for comparison:")
    
    medclip = create_medclip(variant='base', pretrained=False)
    biomedclip = create_biomedclip()
    
    print(f"   MedCLIP: {type(medclip).__name__}")
    print(f"   BioMedCLIP: {type(biomedclip).__name__}")
    
    # Compare on same inputs
    print("\n2. Comparing embeddings:")
    
    dummy_image = torch.randn(1, 3, 224, 224)
    texts = [
        "normal chest x-ray",
        "chest x-ray showing pneumonia",
        "chest x-ray showing covid-19"
    ]
    
    # Get embeddings from both models
    with torch.no_grad():
        # BioMedCLIP embeddings
        bio_image_feats = biomedclip.encode_image(dummy_image)
        bio_text_feats = biomedclip.encode_text(texts)
        
        # MedCLIP embeddings (would need proper tokenization)
        # This is simplified for demonstration
        med_image_feats = medclip.encode_image(dummy_image)
        
    print(f"   BioMedCLIP image embedding: {bio_image_feats.shape}")
    print(f"   BioMedCLIP text embedding: {bio_text_feats.shape}")
    print(f"   MedCLIP image embedding: {med_image_feats.shape}")
    
    # Compute similarities
    bio_similarities = (bio_image_feats @ bio_text_feats.t()).softmax(dim=-1)
    
    print("\n3. BioMedCLIP predictions:")
    for i, text in enumerate(texts):
        print(f"   {text}: {bio_similarities[0][i].item():.4f}")


def example_custom_integration():
    """Example of custom integration with existing pipeline"""
    print("\n" + "=" * 60)
    print("Custom Integration Example")
    print("=" * 60)
    
    # Show how to integrate with existing dataset/dataloader
    print("\n1. Integration with custom dataset:")
    
    # Create model
    model = create_biomedclip()
    
    # Create classifier wrapper
    classifier = ModelFactory.create_classifier(
        model=model,
        task_type='zeroshot',
        class_names=['Normal', 'Abnormal'],
        ensemble=True
    )
    
    print(f"   Model ready for integration with dataloaders")
    
    # Example batch processing
    print("\n2. Batch processing example:")
    
    # Simulate a batch from dataloader
    batch = {
        'pixel_values': torch.randn(8, 3, 224, 224),
        'labels': torch.randint(0, 2, (8,))
    }
    
    # Process batch
    with torch.no_grad():
        # For zero-shot, we'd need prompt_inputs
        # This is simplified
        image_features = model.encode_image(batch['pixel_values'])
        
    print(f"   Processed batch of {len(batch['pixel_values'])} images")
    print(f"   Output shape: {image_features.shape}")


def main():
    """Run all examples"""
    print("\n" + "🏥" * 30)
    print(" MEDICAL VISION-LANGUAGE MODELS - USAGE EXAMPLES")
    print("🏥" * 30)
    
    # Show available models
    print("\n📦 Available Models and Classifiers:")
    ModelFactory.print_registry()
    
    # Run examples
    example_basic_usage()
    example_zero_shot_classification()
    example_supervised_classification()
    example_model_comparison()
    example_custom_integration()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    
    # Usage tips
    print("\n💡 Tips for using the models:")
    print("1. Use ModelFactory for consistent model creation")
    print("2. Choose between MedCLIP and BioMedCLIP based on your data")
    print("3. For zero-shot: no training needed, just define classes")
    print("4. For supervised: freeze encoder for faster training")
    print("5. Use ensemble=True for better zero-shot performance")


if __name__ == "__main__":
    main()
