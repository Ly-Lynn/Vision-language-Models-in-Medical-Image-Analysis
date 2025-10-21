"""
ENTRep dataset for endoscopic image classification tasks
"""

import os
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from open_clip import get_tokenizer
from datasets import load_from_disk
from PIL import Image

from .base import BaseContrastiveDataset, BaseClassificationDataset, BaseCollator
from ..utils.constants import (
    DATASET_CONFIGS,
    ENTREP_TASKS, DEFAULT_TEMPLATES
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ENTREPDataset(BaseContrastiveDataset):
    def __init__(
        self,
        data_root: str = 'local_data/entrep',
        split: str = 'train',
        model_type: str = 'entrep',
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        super().__init__(data_root, split, model_type, transform=transform, **kwargs)
        
        self.df = self._load_data()
    def create_csv(self) -> pd.DataFrame:
        train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
        random_state = 22520691

        df = pd.read_csv(os.path.join(self.data_root, 'entrep-data.csv'))
        for index, row in df.iterrows():
            row['image_path'] = os.path.join(self.data_root, 'images', row['image_path'])
            df.loc[index, 'image_path'] = row['image_path']
        df.to_csv(os.path.join(self.data_root, 'entrep-data.csv'), index=False)

        nose_df = df[df['nose'] == 1]
        nose_df = nose_df.sample(frac=1, random_state=random_state)

        vocal_throat_df = df[df['vocal-throat'] == 1]
        vocal_throat_df = vocal_throat_df.sample(frac=1, random_state=random_state)

        ear_df = df[df['ear'] == 1]
        ear_df = ear_df.sample(frac=1, random_state=random_state)

        throat_df = df[df['throat'] == 1]
        throat_df = throat_df.sample(frac=1, random_state=random_state)

        # Split into train, test, and validation sets
        nose_train = nose_df.iloc[:int(len(nose_df) * train_ratio)]
        nose_test = nose_df.iloc[int(len(nose_df) * train_ratio):int(len(nose_df) * (train_ratio + test_ratio))]
        nose_val = nose_df.iloc[int(len(nose_df) * (train_ratio + test_ratio)):]

        vocal_throat_train = vocal_throat_df.iloc[:int(len(vocal_throat_df) * train_ratio)]
        vocal_throat_test = vocal_throat_df.iloc[int(len(vocal_throat_df) * train_ratio):int(len(vocal_throat_df) * (train_ratio + test_ratio))]
        vocal_throat_val = vocal_throat_df.iloc[int(len(vocal_throat_df) * (train_ratio + test_ratio)):]

        ear_train = ear_df.iloc[:int(len(ear_df) * train_ratio)]
        ear_test = ear_df.iloc[int(len(ear_df) * train_ratio):int(len(ear_df) * (train_ratio + test_ratio))]
        ear_val = ear_df.iloc[int(len(ear_df) * (train_ratio + test_ratio)):]

        throat_train = throat_df.iloc[:int(len(throat_df) * train_ratio)]
        throat_test = throat_df.iloc[int(len(throat_df) * train_ratio):int(len(throat_df) * (train_ratio + test_ratio))]
        throat_val = throat_df.iloc[int(len(throat_df) * (train_ratio + test_ratio)):]

        # Create train, test, and validation sets
        train_df = pd.concat([nose_train, vocal_throat_train, ear_train, throat_train])
        test_df = pd.concat([nose_test, vocal_throat_test, ear_test, throat_test])
        val_df = pd.concat([nose_val, vocal_throat_val, ear_val, throat_val])

        # Save to CSV
        train_df = train_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)
        test_df = test_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)
        val_df = val_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)

        train_df.to_csv(os.path.join(self.data_root, 'entrep-train-meta.csv'), index=True)
        test_df.to_csv(os.path.join(self.data_root, 'entrep-test-meta.csv'), index=True)
        val_df.to_csv(os.path.join(self.data_root, 'entrep-val-meta.csv'), index=True)
    def _load_data(self) -> pd.DataFrame:
        """Load ENTREP data from CSV file"""
        def download_entrep_dataset():    
            if gdown is None:
                logger.error("gdown not installed. Please install with: pip install gdown")
                return False
                
            url_id = "1r2mIaytuvHQc5D77BAuIfkLeW9jGXjed"
            entrep_output = os.path.join(self.data_root, "entrep_dataset.zip")
            logger.info("Downloading ENTREP dataset from Google Drive...")
            
            try:
                gdown.download(id=url_id, output=entrep_output, quiet=False)
                with zipfile.ZipFile(entrep_output, 'r') as zip_ref:
                    zip_ref.extractall(self.data_root)
                os.remove(entrep_output) 
                self.create_csv()
                return True
            except Exception as e:
                logger.error(f"Failed to download ENTREP dataset: {e}")
                return False
        
        os.makedirs(self.data_root, exist_ok=True)
        entrep_data_path = self.data_root
        
        # Check if required files exist
        train_csv_path = os.path.join(entrep_data_path, "entrep-train-meta.csv")
        test_csv_path = os.path.join(entrep_data_path, "entrep-test-meta.csv")
        val_csv_path = os.path.join(entrep_data_path, "entrep-val-meta.csv")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path) or not os.path.exists(val_csv_path):
            logger.info(f"ENTREP data not found in {self.data_root}, preparing data")
            if not download_entrep_dataset():
                logger.warning("Failed to download ENTREP data")
        
        