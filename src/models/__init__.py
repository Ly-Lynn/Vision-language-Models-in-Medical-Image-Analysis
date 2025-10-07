"""
Vision-Language Models for Medical Image Analysis

This module provides implementations of various vision-language models
for medical image analysis, including MedCLIP and BioMedCLIP.
"""

# Base classes
from .model import (
    BaseVisionLanguageModel,
    BaseClassifier,
    BaseZeroShotClassifier,
    BaseSupervisedClassifier,
    BasePromptLearner
)

# MedCLIP models
from .medclip import (
    MedCLIPModel,
    MedCLIPTextModel,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
    PromptClassifier,
    SuperviseClassifier,
    PromptTuningClassifier,
    PartiallyFixedEmbedding
)

# BioMedCLIP models
from .biomedclip import (
    BioMedCLIPModel,
    BioMedCLIPClassifier,
    BioMedCLIPFeatureExtractor
)

# Factory
from .factory import (
    ModelFactory,
    create_medclip,
    create_biomedclip
)

# Version
__version__ = "0.1.0"

# Define what should be imported with "from models import *"
__all__ = [
    # Base classes
    'BaseVisionLanguageModel',
    'BaseClassifier',
    'BaseZeroShotClassifier',
    'BaseSupervisedClassifier',
    'BasePromptLearner',
    
    # MedCLIP
    'MedCLIPModel',
    'MedCLIPTextModel',
    'MedCLIPVisionModel',
    'MedCLIPVisionModelViT',
    'PromptClassifier',
    'SuperviseClassifier',
    'PromptTuningClassifier',
    
    # BioMedCLIP
    'BioMedCLIPModel',
    'BioMedCLIPClassifier',
    'BioMedCLIPFeatureExtractor',
    
    # Factory
    'ModelFactory',
    'create_medclip',
    'create_biomedclip',
]
