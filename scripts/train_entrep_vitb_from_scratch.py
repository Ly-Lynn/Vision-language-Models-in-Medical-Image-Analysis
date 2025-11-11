"""
Train ENTRep with ViT-Base from scratch (NO pretrained weights)
Sử dụng IMG_MEAN và IMG_STD từ constants.py
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.factory import create_model
from modules.trainer.entrep import ENTRepTrainer
from modules.utils.logging_config import get_logger
from modules.utils import constants

logger = get_logger(__name__)


def create_config():
    """Tạo config cho training ENTRep với ViT-Base từ đầu"""
    return {
        'model': {
            'model_type': 'entrep',
            'text_encoder_type': 'clip',
            'vision_encoder_type': 'dinov2',
            'model_name': 'dinov2_vitb14',  # Sẽ dùng ViT-B làm fallback
            'feature_dim': 768,
            'dropout': 0.1,
            'dropout_rate': 0.3,
            'num_classes': 4,
            'freeze_backbone': False,
            'checkpoint': None,  # KHÔNG load checkpoint
            'vision_checkpoint': None,  # KHÔNG load vision checkpoint
            'text_checkpoint': None,
            'logit_scale_init_value': 0.07,
            'pretrained': False,  # QUAN TRỌNG: Không sử dụng pretrained weights
        },
        'dataset': {
            'dataset_name': 'entrep',
            'dataset_type': 'contrastive',
            'data_root': 'local_data/entrep',
            'model_type': 'entrep',
            'batch_size': 32,
            'num_workers': 4,
            'tokenizer_name': 'openai/clip-vit-base-patch32'
        },
        'training': {
            'num_epochs': 100,
            'val_every': 1,
            'save_every': 5,
            'use_amp': True,
            'plot_every': 5
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        },
        'loss': {
            'type': 'contrastive'
        },
        'experiment': {
            'seed': 42,
            'output_dir': './checkpoints',
            'use_wandb': False,
            'wandb_project': 'entrep-vitb-from-scratch'
        }
    }


def main():
    # Debug: Print script name
    import os
    script_name = os.path.basename(__file__)
    logger.info("=" * 70)
    logger.info(f"🔍 DEBUG: Running script: {script_name}")
    logger.info("=" * 70)
    
    parser = argparse.ArgumentParser(
        description='Train ENTRep with ViT-Base from scratch (NO pretrained weights)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints/entrep_vitb_from_scratch',
        help='Output directory'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='entrep_vitb_from_scratch',
        help='Experiment name'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases'
    )
    
    args = parser.parse_args()
    
    # Load hoặc tạo config
    if args.config and os.path.exists(args.config):
        logger.info(f"📋 Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.info("📋 Using default config")
        config = create_config()
    
    # Override config với command line args
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['optimizer']['lr'] = args.learning_rate
    if args.use_wandb:
        config['experiment']['use_wandb'] = True
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    # FORCE pretrained=False và checkpoint=None cho from_scratch mode
    logger.info("")
    logger.info("🔧 DEBUG - Config BEFORE forcing:")
    logger.info(f"   pretrained = {config['model'].get('pretrained')}")
    logger.info(f"   checkpoint = {config['model'].get('checkpoint')}")
    
    config['model']['pretrained'] = False
    config['model']['checkpoint'] = None
    
    logger.info("🔧 DEBUG - Config AFTER forcing:")
    logger.info(f"   pretrained = {config['model']['pretrained']}")
    logger.info(f"   checkpoint = {config['model']['checkpoint']}")
    logger.info("")
    
    logger.info("=" * 70)
    logger.info("🚀 TRAINING ENTRep - ViT-Base FROM SCRATCH")
    logger.info("=" * 70)
    logger.info(f"⚠️  Pretrained = {config['model']['pretrained']} (training from random initialization)")
    logger.info(f"⚠️  Checkpoint = {config['model']['checkpoint']}")
    logger.info("=" * 70)
    
    # Tạo model sử dụng factory
    logger.info("🏗️ Creating ENTRep model...")
    logger.info(f"   Model config: pretrained={config['model']['pretrained']}, checkpoint={config['model']['checkpoint']}")
    
    # Extract pretrained flag to pass explicitly (avoid default override)
    model_config = config['model'].copy()
    pretrained_flag = model_config.pop('pretrained', False)  # Default False for scratch
    checkpoint_path = model_config.pop('checkpoint', None)
    logger.info(f"   Calling create_model with: pretrained={pretrained_flag}, checkpoint={checkpoint_path}")
    
    model = create_model(
        pretrained=pretrained_flag,    # ← Explicitly pass pretrained
        checkpoint=checkpoint_path,     # ← Explicitly pass checkpoint
        **model_config
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   📊 Total parameters: {total_params:,}")
    logger.info(f"   📊 Trainable parameters: {trainable_params:,}")
    
    # Tạo trainer
    logger.info("🎓 Creating trainer...")
    trainer = ENTRepTrainer(
        model=model,
        config=config,
        output_dir=config['experiment']['output_dir'],
        experiment_name=args.experiment_name,
        use_wandb=config['experiment']['use_wandb'],
        wandb_project=config['experiment']['wandb_project'],
        seed=config['experiment']['seed']
    )
    
    # Bắt đầu training
    logger.info("🚀 Starting training...")
    trainer.train()
    
    logger.info("🎉 Training completed!")


if __name__ == '__main__':
    main()

