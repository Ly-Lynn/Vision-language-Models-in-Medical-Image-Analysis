# Import evaluators
from .evaluator import (
    Evaluator,  # Original evaluator
    BaseEvaluator,
    ZeroShotEvaluator,
    TextToImageRetrievalEvaluator
)

# Import models
from .models import *

# Import datasets  
from .dataset import *

from logging_config import setup_logging
setup_logging(level='INFO')


# Version
__version__ = "0.1.0"

# Define what should be imported with "from src import *"
__all__ = [
    # Evaluators
    'Evaluator',
    'BaseEvaluator', 
    'ZeroShotEvaluator',
    'TextToImageRetrievalEvaluator',
    
    # Models (imported from models module)
    'BaseVisionLanguageModel',
    'BaseClassifier',
    'BaseZeroShotClassifier',
    'BaseSupervisedClassifier',
    'MedCLIPModel',
    'BioMedCLIPModel',
    'ModelFactory',
    'create_medclip',
    'create_biomedclip',
    
    # Datasets (imported from dataset module)
    'BaseMedicalDataset',
    'BaseContrastiveDataset',
    'BaseClassificationDataset',
    'COVIDDataset',
    'RSNADataset',
    'MIMICContrastiveDataset',
    'DatasetFactory',
    'create_covid_dataloader',
    'create_rsna_dataloader',
    'create_mimic_dataloader'
]