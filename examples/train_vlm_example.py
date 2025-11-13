"""
Example: Training Vision-Language Models with VisionLanguageTrainer

Demonstrates how to train:
1. ENTRep Model
2. MedCLIP Model
3. BioMedCLIP Model

Usage:
    python train_vlm_example.py --model entrep
    python train_vlm_example.py --model medclip
    python train_vlm_example.py --model biomedclip
"""

import sys
import argparse
sys.path.append('..')

import torch
from modules.trainer import VisionLanguageTrainer
from modules.models.entrep import ENTRepModel
from modules.models.medclip import MedCLIPModel
from modules.models.biomedclip import BioMedCLIPModel


def get_config_for_entrep():
    """Configuration for ENTRep training"""
    return {
        'model_type': 'entrep',
        'dataset': {
            'dataset_name': 'entrep',
            'dataset_type': 'contrastive',
            'task_type': 'contrastive',
            'data_root': '../modules/local_data',
            'model_type': 'entrep',
            'batch_size': 32,
            'num_workers': 4,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        },
        'training': {
            'num_epochs': 50,
            'val_every': 1,
            'use_amp': True,
            'plot_every': 5
        },
        'num_epochs': 50,
        'val_every': 1,
        'save_every': 5,
    }


def get_config_for_medclip():
    """Configuration for MedCLIP training"""
    return {
        'model_type': 'medclip',
        'dataset': {
            'dataset_name': 'mimic',
            'dataset_type': 'contrastive',
            'task_type': 'contrastive',
            'data_root': '../modules/local_data',
            'model_type': 'medclip',
            'batch_size': 32,
            'num_workers': 4,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-5,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        },
        'training': {
            'num_epochs': 50,
            'val_every': 1,
            'use_amp': True,
            'plot_every': 5
        },
        'num_epochs': 50,
        'val_every': 1,
        'save_every': 5,
    }


def get_config_for_biomedclip():
    """Configuration for BioMedCLIP training"""
    return {
        'model_type': 'biomedclip',
        'dataset': {
            'dataset_name': 'mimic',
            'dataset_type': 'contrastive',
            'task_type': 'contrastive',
            'data_root': '../modules/local_data',
            'model_type': 'biomedclip',
            'batch_size': 16,  # Smaller batch for larger model
            'num_workers': 4,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-5,  # Lower learning rate for fine-tuning
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-7
        },
        'training': {
            'num_epochs': 30,
            'val_every': 1,
            'use_amp': True,
            'plot_every': 5
        },
        'num_epochs': 30,
        'val_every': 1,
        'save_every': 5,
    }


def train_entrep():
    """Train ENTRep model"""
    print("=" * 80)
    print("🚀 Training ENTRep Model")
    print("=" * 80)
    
    # Initialize model
    print("\n📦 Initializing ENTRep model...")
    model = ENTRepModel(
        vision_backbone='dinov2',
        text_encoder='clinicalbert',
        feature_dim=512
    )
    print(f"✅ Model initialized")
    
    # Get config
    config = get_config_for_entrep()
    
    # Create trainer
    print("\n🔧 Creating trainer...")
    trainer = VisionLanguageTrainer(
        model=model,
        config=config,
        output_dir='./checkpoints',
        experiment_name='entrep_contrastive',
        use_wandb=False,  # Set to True to enable WandB logging
        seed=42
    )
    print(f"✅ Trainer created")
    
    # Start training
    print("\n🎓 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")


def train_medclip():
    """Train MedCLIP model"""
    print("=" * 80)
    print("🚀 Training MedCLIP Model")
    print("=" * 80)
    
    # Initialize model
    print("\n📦 Initializing MedCLIP model...")
    model = MedCLIPModel(
        vision_backbone='resnet50'
    )
    print(f"✅ Model initialized")
    
    # Get config
    config = get_config_for_medclip()
    
    # Create trainer
    print("\n🔧 Creating trainer...")
    trainer = VisionLanguageTrainer(
        model=model,
        config=config,
        output_dir='./checkpoints',
        experiment_name='medclip_contrastive',
        use_wandb=False,  # Set to True to enable WandB logging
        seed=42
    )
    print(f"✅ Trainer created")
    
    # Start training
    print("\n🎓 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")


def train_biomedclip():
    """Train BioMedCLIP model"""
    print("=" * 80)
    print("🚀 Training BioMedCLIP Model")
    print("=" * 80)
    
    # Initialize model
    print("\n📦 Initializing BioMedCLIP model...")
    model = BioMedCLIPModel(
        model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    print(f"✅ Model initialized")
    
    # Get config
    config = get_config_for_biomedclip()
    
    # Create trainer
    print("\n🔧 Creating trainer...")
    trainer = VisionLanguageTrainer(
        model=model,
        config=config,
        output_dir='./checkpoints',
        experiment_name='biomedclip_finetune',
        use_wandb=False,  # Set to True to enable WandB logging
        seed=42
    )
    print(f"✅ Trainer created")
    
    # Start training
    print("\n🎓 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Vision-Language Models')
    parser.add_argument(
        '--model',
        type=str,
        default='entrep',
        choices=['entrep', 'medclip', 'biomedclip'],
        help='Model to train (default: entrep)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only initialize model and trainer without training'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🏋️  Vision-Language Model Training")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80 + "\n")
    
    if args.model == 'entrep':
        if args.dry_run:
            print("Dry run mode - skipping training")
            config = get_config_for_entrep()
            model = ENTRepModel(vision_backbone='dinov2', text_encoder='clinicalbert')
            trainer = VisionLanguageTrainer(model=model, config=config, use_wandb=False)
            print("✅ Model and trainer initialized successfully")
        else:
            train_entrep()
            
    elif args.model == 'medclip':
        if args.dry_run:
            print("Dry run mode - skipping training")
            config = get_config_for_medclip()
            model = MedCLIPModel(vision_backbone='resnet50')
            trainer = VisionLanguageTrainer(model=model, config=config, use_wandb=False)
            print("✅ Model and trainer initialized successfully")
        else:
            train_medclip()
            
    elif args.model == 'biomedclip':
        if args.dry_run:
            print("Dry run mode - skipping training")
            config = get_config_for_biomedclip()
            model = BioMedCLIPModel()
            trainer = VisionLanguageTrainer(model=model, config=config, use_wandb=False)
            print("✅ Model and trainer initialized successfully")
        else:
            train_biomedclip()


if __name__ == '__main__':
    main()

