"""
Example usage of the model factory and base classes
"""

import torch
from PIL import Image
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep


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
        pretrained=True  # Set to True to download pretrained weights
    )
    print(f"   Created: {type(medclip).__name__}")
    
    # Example 2: Create BioMedCLIP model using convenience function
    print("\n2. Creating BioMedCLIP model:")
    biomedclip = create_biomedclip()
    print(f"   Created: {type(biomedclip).__name__}")
    
    # Example 3: Create EnTreP model using convenience function
    print("\n3. Creating Entrep model:")
    entrep = create_entrep(
        text_encoder='clip',
        vision_encoder='clip'
    )
    print(f"   Created: {type(entrep).__name__}")
    print(f"   Text encoder: CLIP, Vision encoder: CLIP")
    
    # Example 4: Encode text and images
    print("\n4. Encoding text and images:")
    
    # Dummy image tensor
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Encode with BioMedCLIP
    text = "chest x-ray showing pneumonia"
    text_features = biomedclip.encode_text(text)
    image_features = biomedclip.encode_image(dummy_image)
    
    print(f"   BioMedCLIP - Text features shape: {text_features.shape}")
    print(f"   BioMedCLIP - Image features shape: {image_features.shape}")
    
    # Compute similarity
    similarity = (image_features @ text_features.t()).softmax(dim=-1)
    print(f"   BioMedCLIP - Similarity score: {similarity.item():.4f}")
    
    # Encode with EnTreP
    try:
        entrep_text_features = entrep.encode_text(text)
        entrep_image_features = entrep.encode_image(dummy_image)
        
        print(f"   EnTreP - Text features shape: {entrep_text_features.shape}")
        print(f"   EnTreP - Image features shape: {entrep_image_features.shape}")
        
        # Compute similarity
        entrep_similarity = (entrep_image_features @ entrep_text_features.t()).softmax(dim=-1)
        print(f"   EnTreP - Similarity score: {entrep_similarity.item():.4f}")
    except Exception as e:
        print(f"   EnTreP encoding error: {e}")


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
    
    # Create zero-shot classifier using EnTreP
    print("\n3. Creating EnTreP zero-shot classifier:")
    try:
        entrep_classifier = ModelFactory.create_zeroshot_classifier(
            model_type='entrep',
            class_names=class_names,
            templates=['a medical image showing {}'],
            ensemble=True
        )
        print(f"   Created classifier for {len(class_names)} classes")
    except Exception as e:
        print(f"   Error creating EnTreP classifier: {e}")
    
    # Perform classification on dummy image
    print("\n4. Performing classification:")
    dummy_batch = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    # Classify with BioMedCLIP
    biomedclip_model = create_biomedclip()
    bio_classifier = ModelFactory.create_classifier(
        model=biomedclip_model,
        task_type='zeroshot',
        class_names=class_names
    )
    
    # Classify with EnTreP
    try:
        entrep_model = create_entrep()
        entrep_classifier = ModelFactory.create_classifier(
            model=entrep_model,
            task_type='zeroshot',
            class_names=class_names
        )
        print(f"   ✅ EnTreP classifier created successfully")
    except Exception as e:
        print(f"   ❌ EnTreP classifier error: {e}")
    
    print(f"   Input batch shape: {dummy_batch.shape}")
    print(f"   Classes: {class_names}")
    print(f"   Available models: MedCLIP, BioMedCLIP, EnTreP")

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
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    
    # Usage tips
    print("\n💡 Tips for using the models:")
    print("1. Use ModelFactory for consistent model creation")
    print("2. For zero-shot: no training needed, just define classes")
    print("3. Use ensemble=True for better zero-shot performance")
    print("4. Available models: MedCLIP, BioMedCLIP, EnTreP")
    print("5. EnTreP supports different vision encoders: 'clip', 'endovit', 'dinov2'")
    print("6. EnTreP supports different text encoders: 'clip' or 'none'")


if __name__ == "__main__":
    main()
