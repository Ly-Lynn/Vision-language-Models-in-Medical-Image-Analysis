"""
Complete testing
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
    PromptClassifier,
    SuperviseClassifier,
    PromptTuningClassifier
)
from .biomedclip import (
    BioMedCLIPModel,
    BioMedCLIPClassifier,
    BioMedCLIPFeatureExtractor
)
from .entrep import (
    ENTRepModel
)

# Import constants
from ..utils.constants import SUPPORTED_MODELS, DEFAULT_TEMPLATES
from ..utils import logging_config

logger = logging_config.get_logger(__name__)

class ModelFactory:
    """
    Factory class for creating vision-language models and classifiers
    """
    
    # Registry of base models
    MODEL_REGISTRY = {
        'medclip': {
            'base': MedCLIPModel,
        },
        'biomedclip': {
            'base': BioMedCLIPModel,
        },
        'entrep': {
            'base': ENTRepModel,
        }
    }
    
    # Registry of classifiers
    CLASSIFIER_REGISTRY = {
        'medclip': {
            'zeroshot': PromptClassifier
        },
        'biomedclip': {
            'zeroshot': BioMedCLIPClassifier
        },
        'entrep': {
            'zeroshot': ENTRepModel
        }
    }
    
    # Default configurations
    DEFAULT_CONFIGS = {
        'medclip': {
            'text_encoder_type': 'bert',
            'vision_encoder_type': 'vit',  # 'resnet' or 'vit'
            'logit_scale_init_value': 0.07,
            'checkpoint': None,
        },
        'biomedclip': {
            'model_name': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'context_length': 256,
            'checkpoint': None,
        },
        'entrep': {
            'text_encoder_type': 'clip',
            'vision_encoder_type': 'dinov2',
            'feature_dim': 768,
            'dropout': 0.1,
            'num_classes': 7,
            'freeze_backbone': False,
            'vision_checkpoint': None,
            'text_checkpoint': None,
            'logit_scale_init_value': 0.07,
            'pretrained': True,
        }
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str = 'medclip',
        variant: str = 'base',
        checkpoint: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> BaseVisionLanguageModel:
        """
        Create a vision-language model
        
        Args:
            model_type: 'medclip' or 'biomedclip'
            variant: Model variant ('base', 'vision_resnet', 'vision_vit')
            checkpoint: Path to model checkpoint
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
            # Create MedCLIP model with flexible encoders
            model = model_class(**config)
            
            # Load pretrained weights if requested
            if pretrained and checkpoint is None:
                model.from_pretrained()
                
        elif model_type == 'biomedclip':
            # Create BioMedCLIP model
            model = model_class(**config)
            
        elif model_type == 'entrep':
            # Create ENTRep model
            model = model_class(**config)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move to device if specified
        if hasattr(model, 'to'):
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
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
            elif isinstance(model, ENTRepModel):
                model_type = 'entrep'
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
                
            if model_type == 'medclip':
                classifier = classifier_class(
                    medclip_model=model,
                    ensemble=ensemble,
                    **kwargs
                )
            elif model_type == 'biomedclip':
                classifier = classifier_class(
                    biomedclip_model=model,
                    ensemble=ensemble,
                    **kwargs
                )
            elif model_type == 'entrep':
                classifier = classifier_class(
                    entrep_model=model,
                    num_classes=num_classes,
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
            **kwargs: Additional arguments
            
        Returns:
            Zero-shot classifier instance
        """
        # Create base model
        model = cls.create_model(
            model_type=model_type,
            checkpoint=checkpoint,
            **kwargs
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
            num_classes=len(class_names),
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
            **kwargs: Additional arguments
            
        Returns:
            Supervised classifier instance
        """
        # Create base model
        model = cls.create_model(
            model_type=model_type,
            checkpoint=checkpoint,
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
        """logger.info information about available models and classifiers"""
        logger.info("🏭 Model Factory Registry")
        logger.info("=" * 50)
        
        logger.info("🤖 Available Models:")
        for model_type, variants in cls.MODEL_REGISTRY.items():
            logger.info(f"  {model_type}: {list(variants.keys())}")
        
        logger.info("🎯 Available Classifiers:")
        for model_type, task_types in cls.CLASSIFIER_REGISTRY.items():
            logger.info(f"  {model_type}: {list(task_types.keys())}")
        
        logger.info(f"📚 Supported Model Types: {SUPPORTED_MODELS}")


# Convenience functions
def create_medclip(
    text_encoder: str = 'bert',
    vision_encoder: str = 'vit',
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create MedCLIP model
    
    Args:
        text_encoder: 'bert' (currently only BERT supported)
        vision_encoder: 'resnet' or 'vit'
        pretrained: Whether to load pretrained weights
        checkpoint: Optional checkpoint path
        **kwargs: Additional arguments
        
    Returns:
        MedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        text_encoder_type=text_encoder,
        vision_encoder_type=vision_encoder,
        pretrained=pretrained,
        checkpoint=checkpoint,
        **kwargs
    )


def create_biomedclip(
    checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create BioMedCLIP model
    
    Args:
        checkpoint: Optional checkpoint path
        **kwargs: Additional arguments
        
    Returns:
        BioMedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='biomedclip',
        checkpoint=checkpoint,
        **kwargs
    )

# Wrapper functions for easier import
def create_model(model_type: str = 'medclip', **kwargs):
    """Create a model using ModelFactory"""
    return ModelFactory.create_model(model_type=model_type, **kwargs)

def create_entrep(
    text_encoder: str = 'clip',
    vision_encoder: str = 'clip',
    vision_checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create ENTRep model
    
    Args:
        text_encoder: 'clip' or 'none'
        vision_encoder: 'clip', 'endovit', or 'dinov2'
        vision_checkpoint: Path to vision encoder checkpoint
        **kwargs: Additional arguments
        
    Returns:
        ENTRep model instance
        
    Note:
        If vision_checkpoint is provided and text_encoder is 'none', 
        creates a vision-only model using wrapper (DinoV2Model/EntVitModel)
        to ensure checkpoint loads correctly (compatible with ENTRep/model_factory.py)
    """
    # Vision-only mode với checkpoint
    if vision_checkpoint and text_encoder == 'none':
        from .entrep import ENTRepModel
        return ENTRepModel(
            vision_encoder_type=vision_encoder,
            vision_checkpoint=vision_checkpoint,
            **kwargs
        )
    
    # Full ENTRep model (text + vision)
    return ModelFactory.create_model(
        model_type='entrep',
        variant='base',
        text_encoder_type=text_encoder,
        vision_encoder_type=vision_encoder,
        vision_checkpoint=vision_checkpoint,
        **kwargs
    )


if __name__ == "__main__":
    demo_factory()
