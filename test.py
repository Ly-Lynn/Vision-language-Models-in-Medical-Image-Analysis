from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator

from tqdm import tqdm
import numpy as np

dataset_name = "rsna"
model_type = 'medclip'
transform = MODEL_TRANSFORMS[model_type]


DATA_ROOT = '/data2/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_data'
dataset = DatasetFactory.create_dataset(
    dataset_name=dataset_name,
    model_type=model_type,
    data_root=DATA_ROOT,
    transform=None
)

medclip = ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        pretrained=True  # Set to True to download pretrained weights
    )

img, label = dataset[0]
a = dataset._process_class_prompts(5)
print(a)




