from torchvision import transforms


# Model types
BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'
BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

# Image processing
IMG_SIZE = 224
IMG_MEAN = 0.5862785803043838
IMG_STD = 0.27950088968644304

BIOMEDCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMEDCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Dataset paths
DEFAULT_DATA_ROOT = 'local_data'

# COVID Dataset
COVID_TASKS = [
    'Normal',
    'COVID',
]

# MIMIC Dataset
MIMIC_TASKS = [
    'Normal',
    'Abnormal',
]

# RSNA Dataset  
RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]



MIMIC_CLASS_PROMPTS = {
    'Normal': {
        'adjective': ['clear', 'normal', 'healthy'],
        'description': ['chest', 'lungs', 'findings'],
        'subtype': ['x-ray', 'radiograph', 'image'],
        'location': ['', 'bilateral', 'throughout'],
    },
    'Abnormal': {
        'adjective': ['abnormal', 'pathological', 'irregular'],
        'description': ['findings', 'opacity', 'infiltrate'],
        'subtype': ['consolidation', 'effusion', 'pneumonia'],
        'location': ['in lung', 'bilateral', 'unilateral'],
    }
}

COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy', 'confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    },
    'Normal': {
        'adjective': ['clear', 'normal', 'healthy'],
        'description': ['chest', 'lungs', 'findings'],
        'subtype': ['x-ray', 'radiograph', 'image'],
        'location': ['', 'bilateral', 'throughout'],
    }
}

RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone", 
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    },
    'Normal': {
        'adjective': ['clear', 'normal', 'healthy'],
        'description': ['chest', 'lungs', 'findings'], 
        'subtype': ['x-ray', 'radiograph', 'image'],
        'location': ['', 'bilateral', 'throughout'],
    }
}

# Model weights
WEIGHTS_NAME = 'pytorch_model.bin'

# Pretrained model URLs
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'

# Dataset configurations
DATASET_CONFIGS = {
    'mimic': {
        'tasks': MIMIC_TASKS,
        'class_prompts': MIMIC_CLASS_PROMPTS,
        'data_files': {
            'train': 'mimic-train-meta.csv',
            'test': 'mimic-test-meta.csv',
            'finetune': 'mimic-finetune-meta.csv'
        },
        'mode': 'binary'
    },
    'covid': {
        'tasks': COVID_TASKS,
        'class_prompts': COVID_CLASS_PROMPTS,
        'data_files': {
            'train': 'covid-train-meta.csv',
            'test': 'covid-test-meta.csv',
            'small': 'covid-0.1-train-meta.csv'
        },
        'mode': 'binary'
    },
    'rsna': {
        'tasks': RSNA_TASKS,
        'class_prompts': RSNA_CLASS_PROMPTS,
        'data_files': {
            'train': 'stage_2_train_labels.csv',
            'test': 'rsna-balanced-test-meta.csv'
        },
        'mode': 'binary'
    }
}

# Supported model types
SUPPORTED_MODELS = ['medclip', 'biomedclip']

# Default templates for text prompts
DEFAULT_TEMPLATES = {
    'medclip': 'this is a photo of ',
    'biomedclip': 'this is a chest x-ray showing ',
    'general': 'this is an image of '
}

        # return transforms.Compose([
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        # ])


MODEL_TRANSFORMS = {
    'medclip': transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ]
    ),
    'biomedclip': transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=BIOMEDCLIP_MEAN,
                std=BIOMEDCLIP_STD
            ),
        ]
    ),
}


