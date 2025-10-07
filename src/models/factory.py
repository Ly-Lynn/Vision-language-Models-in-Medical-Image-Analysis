"""
Factory for creating vision-language models and classifiers
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn

# Import base classes
from .model import (
    BaseVisionLanguageModel,
    BaseClassifier,
    BaseZeroShotClassifier,
    BaseSupervisedClassifier,
    BasePromptLearner
)

# Import concrete implementations
from .medclip import (
    MedCLIPModel,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
    PromptClassifier,
    SuperviseClassifier,
    PromptTuningClassifier
)
from .biomedclip import (
    BioMedCLIPModel,
    BioMedCLIPClassifier,
    BioMedCLIPFeatureExtractor
)

# Import constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SUPPORTED_MODELS, DEFAULT_TEMPLATES


class ModelFactory:
    """
    Factory class for creating vision-language models and classifiers
    """
    
    # Registry of base models
    MODEL_REGISTRY = {
        'medclip': {
            'base': MedCLIPModel,
            'vision_resnet': MedCLIPVisionModel,
            'vision_vit': MedCLIPVisionModelViT,
        },
        'biomedclip': {
            'base': BioMedCLIPModel,
        }
    }
    
    # Registry of classifiers
    CLASSIFIER_REGISTRY = {
        'medclip': {
            'zeroshot': PromptClassifier,
            'supervised': SuperviseClassifier,
            'prompt_tuning': PromptTuningClassifier,
        },
        'biomedclip': {
            'zeroshot': BioMedCLIPClassifier,
            'supervised': BioMedCLIPFeatureExtractor,
        }
    }
    
    # Default configurations
    DEFAULT_CONFIGS = {
        'medclip': {
            'vision_cls': MedCLIPVisionModelViT,  # Default to ViT
            'logit_scale_init_value': 0.07,
            'checkpoint': None,
        },
        'biomedclip': {
            'model_name': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'context_length': 256,
            'device': None,
            'checkpoint': None,
        }
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str = 'medclip',
        variant: str = 'base',
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> BaseVisionLanguageModel:
        """
        Create a vision-language model
        
        Args:
            model_type: 'medclip' or 'biomedclip'
            variant: Model variant ('base', 'vision_resnet', 'vision_vit')
            checkpoint: Path to model checkpoint
            device: Device to load model on
            pretrained: Whether to load pretrained weights
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model instance
        """
        # Validate model type
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_variants = cls.MODEL_REGISTRY[model_type]
        
        if variant not in model_variants:
            raise ValueError(f"Unknown variant '{variant}' for {model_type}. Available: {list(model_variants.keys())}")
        
        # Get model class
        model_class = model_variants[variant]
        
        # Prepare configuration
        config = cls.DEFAULT_CONFIGS.get(model_type, {}).copy()
        config.update(kwargs)
        
        if checkpoint:
            config['checkpoint'] = checkpoint
        
        # Create model based on type
        if model_type == 'medclip':
            if variant == 'base':
                # Create full MedCLIP model
                vision_cls = config.pop('vision_cls', MedCLIPVisionModelViT)
                model = model_class(
                    vision_cls=vision_cls,
                    checkpoint=config.pop('checkpoint', None),
                    vision_checkpoint=config.pop('vision_checkpoint', None),
                    **config
                )
                
                # Load pretrained weights if requested
                if pretrained and checkpoint is None:
                    model.from_pretrained()
                    
            else:
                # Create vision-only model
                model = model_class(
                    checkpoint=config.pop('checkpoint', None),
                    medclip_checkpoint=config.pop('medclip_checkpoint', None),
                    **config
                )
                
        elif model_type == 'biomedclip':
            # Create BioMedCLIP model
            if device:
                config['device'] = device
                
            model = model_class(**config)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move to device if specified
        if device and hasattr(model, 'to'):
            model = model.to(device)
        
        return model
    
    @classmethod
    def create_classifier(
        cls,
        model: Optional[BaseVisionLanguageModel] = None,
        model_type: Optional[str] = None,
        task_type: str = 'zeroshot',
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        freeze_encoder: bool = True,
        ensemble: bool = False,
        **kwargs
    ) -> BaseClassifier:
        """
        Create a classifier based on a vision-language model
        
        Args:
            model: Pre-initialized model (if None, will create one)
            model_type: Type of model to use ('medclip' or 'biomedclip')
            task_type: 'zeroshot', 'supervised', or 'prompt_tuning'
            num_classes: Number of classes (for supervised)
            class_names: List of class names (for zero-shot)
            freeze_encoder: Whether to freeze encoder (for supervised)
            ensemble: Whether to use prompt ensembling (for zero-shot)
            **kwargs: Additional classifier-specific arguments
            
        Returns:
            Classifier instance
        """
        # Create model if not provided
        if model is None:
            if model_type is None:
                raise ValueError("Either 'model' or 'model_type' must be provided")
            model = cls.create_model(model_type=model_type, **kwargs.pop('model_kwargs', {}))
        else:
            # Infer model type from model instance
            if isinstance(model, MedCLIPModel):
                model_type = 'medclip'
            elif isinstance(model, BioMedCLIPModel):
                model_type = 'biomedclip'
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
        
        # Validate task type
        if model_type not in cls.CLASSIFIER_REGISTRY:
            raise ValueError(f"No classifiers available for model type: {model_type}")
            
        classifier_types = cls.CLASSIFIER_REGISTRY[model_type]
        
        if task_type not in classifier_types:
            raise ValueError(f"Task type '{task_type}' not available for {model_type}. Available: {list(classifier_types.keys())}")
        
        # Get classifier class
        classifier_class = classifier_types[task_type]
        
        # Create classifier based on task type
        if task_type == 'zeroshot':
            if class_names is None:
                raise ValueError("class_names required for zero-shot classification")
                
            classifier = classifier_class(
                medclip_model=model if model_type == 'medclip' else None,
                biomedclip_model=model if model_type == 'biomedclip' else None,
                ensemble=ensemble,
                **kwargs
            )
            
        elif task_type == 'supervised':
            if num_classes is None:
                raise ValueError("num_classes required for supervised classification")
                
            if model_type == 'medclip':
                # For MedCLIP supervised classifier
                classifier = classifier_class(
                    vision_model=model.vision_model if hasattr(model, 'vision_model') else model,
                    num_class=num_classes,
                    mode=kwargs.pop('mode', 'multiclass'),
                    **kwargs
                )
            else:
                # For BioMedCLIP supervised classifier
                classifier = classifier_class(
                    biomedclip_model=model,
                    num_classes=num_classes,
                    freeze_encoder=freeze_encoder,
                    **kwargs
                )
                
        elif task_type == 'prompt_tuning':
            if model_type != 'medclip':
                raise ValueError(f"Prompt tuning not available for {model_type}")
                
            classifier = classifier_class(
                medclip_model=model,
                n_context=kwargs.pop('n_context', 4),
                class_specific_context=kwargs.pop('class_specific_context', False),
                num_class=num_classes,
                mode=kwargs.pop('mode', 'multiclass'),
                ensemble=ensemble,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return classifier
    
    @classmethod
    def create_zeroshot_classifier(
        cls,
        model_type: str = 'medclip',
        class_names: List[str] = None,
        templates: Optional[List[str]] = None,
        ensemble: bool = True,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseClassifier:
        """
        Convenience method to create a zero-shot classifier
        
        Args:
            model_type: 'medclip' or 'biomedclip'
            class_names: List of class names
            templates: Optional custom templates
            ensemble: Whether to use prompt ensembling
            checkpoint: Model checkpoint path
            device: Device to load model on
            **kwargs: Additional arguments
            
        Returns:
            Zero-shot classifier instance
        """
        # Create base model
        model = cls.create_model(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            **kwargs.pop('model_kwargs', {})
        )
        
        # Get default template if not provided
        if templates is None:
            templates = [DEFAULT_TEMPLATES.get(model_type, DEFAULT_TEMPLATES['general'])]
        
        # Create classifier
        return cls.create_classifier(
            model=model,
            model_type=model_type,
            task_type='zeroshot',
            class_names=class_names,
            ensemble=ensemble,
            templates=templates,
            **kwargs
        )
    
    @classmethod
    def create_supervised_classifier(
        cls,
        model_type: str = 'medclip',
        num_classes: int = None,
        task_mode: str = 'multiclass',
        freeze_encoder: bool = True,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseClassifier:
        """
        Convenience method to create a supervised classifier
        
        Args:
            model_type: 'medclip' or 'biomedclip'
            num_classes: Number of output classes
            task_mode: 'binary', 'multiclass', or 'multilabel'
            freeze_encoder: Whether to freeze encoder weights
            checkpoint: Model checkpoint path
            device: Device to load model on
            **kwargs: Additional arguments
            
        Returns:
            Supervised classifier instance
        """
        # Create base model
        model = cls.create_model(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            **kwargs.pop('model_kwargs', {})
        )
        
        # Create classifier
        return cls.create_classifier(
            model=model,
            model_type=model_type,
            task_type='supervised',
            num_classes=num_classes,
            mode=task_mode,
            freeze_encoder=freeze_encoder,
            **kwargs
        )
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """
        Get list of available models and variants
        
        Returns:
            Dictionary {model_type: [available_variants]}
        """
        return {
            model_type: list(variants.keys())
            for model_type, variants in cls.MODEL_REGISTRY.items()
        }
    
    @classmethod
    def get_available_classifiers(cls) -> Dict[str, List[str]]:
        """
        Get list of available classifiers
        
        Returns:
            Dictionary {model_type: [available_task_types]}
        """
        return {
            model_type: list(task_types.keys())
            for model_type, task_types in cls.CLASSIFIER_REGISTRY.items()
        }
    
    @classmethod
    def print_registry(cls):
        """Print information about available models and classifiers"""
        print("🏭 Model Factory Registry")
        print("=" * 50)
        
        print("\n🤖 Available Models:")
        for model_type, variants in cls.MODEL_REGISTRY.items():
            print(f"  {model_type}: {list(variants.keys())}")
        
        print("\n🎯 Available Classifiers:")
        for model_type, task_types in cls.CLASSIFIER_REGISTRY.items():
            print(f"  {model_type}: {list(task_types.keys())}")
        
        print(f"\n📚 Supported Model Types: {SUPPORTED_MODELS}")


# Convenience functions
def create_medclip(
    variant: str = 'base',
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create MedCLIP model
    
    Args:
        variant: 'base', 'vision_resnet', or 'vision_vit'
        pretrained: Whether to load pretrained weights
        checkpoint: Optional checkpoint path
        **kwargs: Additional arguments
        
    Returns:
        MedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='medclip',
        variant=variant,
        pretrained=pretrained,
        checkpoint=checkpoint,
        **kwargs
    )


def create_biomedclip(
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create BioMedCLIP model
    
    Args:
        checkpoint: Optional checkpoint path
        device: Device to load model on
        **kwargs: Additional arguments
        
    Returns:
        BioMedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='biomedclip',
        checkpoint=checkpoint,
        device=device,
        **kwargs
    )


def demo_factory():
    """Demo usage of ModelFactory"""
    print("🏭 Model Factory Demo")
    print("=" * 50)
    
    # Print registry
    ModelFactory.print_registry()
    
    # Test model creation
    print("\n📦 Testing Model Creation:")
    
    models_to_test = [
        ('medclip', 'base'),
        ('biomedclip', 'base'),
    ]
    
    for model_type, variant in models_to_test:
        try:
            model = ModelFactory.create_model(
                model_type=model_type,
                variant=variant,
                pretrained=False  # Don't download weights for demo
            )
            print(f"  ✅ {model_type} {variant}: {type(model).__name__}")
        except Exception as e:
            print(f"  ❌ {model_type} {variant}: {e}")
    
    # Test classifier creation
    print("\n🎯 Testing Classifier Creation:")
    
    classifiers_to_test = [
        ('medclip', 'zeroshot', ['normal', 'pneumonia']),
        ('biomedclip', 'zeroshot', ['normal', 'covid']),
    ]
    
    for model_type, task_type, class_names in classifiers_to_test:
        try:
            # Create model first
            model = ModelFactory.create_model(
                model_type=model_type,
                pretrained=False
            )
            
            # Create classifier
            if task_type == 'zeroshot':
                classifier = ModelFactory.create_classifier(
                    model=model,
                    task_type=task_type,
                    class_names=class_names,
                    ensemble=True
                )
            else:
                classifier = ModelFactory.create_classifier(
                    model=model,
                    task_type=task_type,
                    num_classes=len(class_names)
                )
                
            print(f"  ✅ {model_type} {task_type}: {type(classifier).__name__}")
            
        except Exception as e:
            print(f"  ❌ {model_type} {task_type}: {e}")
    
    # Test convenience functions
    print("\n🔧 Testing Convenience Functions:")
    
    try:
        # Test MedCLIP creation
        medclip = create_medclip(pretrained=False)
        print(f"  ✅ create_medclip: {type(medclip).__name__}")
        
        # Test BioMedCLIP creation
        biomedclip = create_biomedclip()
        print(f"  ✅ create_biomedclip: {type(biomedclip).__name__}")
        
        # Test zero-shot classifier
        zs_classifier = ModelFactory.create_zeroshot_classifier(
            model_type='medclip',
            class_names=['normal', 'abnormal'],
            ensemble=True
        )
        print(f"  ✅ Zero-shot classifier: {type(zs_classifier).__name__}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n✅ Factory Demo completed!")


if __name__ == "__main__":
    demo_factory()
