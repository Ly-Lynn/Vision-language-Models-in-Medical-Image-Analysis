"""
Trainer module for Vision-Language Model training

Supports:
- ENTRep Model
- MedCLIP Model
- BioMedCLIP Model
"""

from .entrep import (
    VisionLanguageTrainer,
    ENTRepTrainer,  # Backward compatibility
    create_trainer_for_entrep,
    create_trainer_for_medclip,
    create_trainer_for_biomedclip,
)

__all__ = [
    'VisionLanguageTrainer',
    'ENTRepTrainer',
    'create_trainer_for_entrep',
    'create_trainer_for_medclip',
    'create_trainer_for_biomedclip',
]
