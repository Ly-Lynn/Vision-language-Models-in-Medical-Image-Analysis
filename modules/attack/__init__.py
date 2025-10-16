"""
Vision-Language Models for Medical Image Analysis

This module provides implementations of various vision-language models
for medical image analysis, including MedCLIP and BioMedCLIP.
"""

# Base classes
from .attack import (
    BaseAttack,
    RandomSearchAttack,
    # NESAttack,
    # ESAttack,
    # CMAESAttack
)

# MedCLIP models
from .evaluator import (
    EvaluatePerturbation,
)

# BioMedCLIP models
from .util import (
    clamp_eps,
    project_delta,
)

# Factory


# Version
__version__ = "0.1.0"

# Define what should be imported with "from models import *"
__all__ = [
    BaseAttack,
    RandomSearchAttack,
    # NESAttack,
    # ESAttack,
    # CMAESAttack,
    
    EvaluatePerturbation,
    
    clamp_eps,
    project_delta,

]
