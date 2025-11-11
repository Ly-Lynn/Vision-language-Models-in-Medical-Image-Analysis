"""
Train ENTRep with ViT-Base pretrained
Sử dụng pretrained checkpoint từ pretrained/entrep_vit_b
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


def create_config(pretrained_path: str):
    """Tạo config cho training ENTRep với ViT-Base pretrained"""
    return {
        'model': {
            'model_type': 'entrep',
            'text_encoder_type': 'clip',
            'vision_encoder_type': 'dinov2',
            'model_name': 'dinov2_vitb14',
            'feature_dim': 768,
            'dropout': 0.1,
            'dropout_rate': 0.3,
            'num_classes': 4,
            'freeze_backbone': False,
            'checkpoint': pretrained_path,  # LOAD pretrained checkpoint
            'vision_checkpoint': None,
            'text_checkpoint': None,
            'logit_scale_init_value': 0.07,
            'pretrained': True,  # Sử dụng pretrained weights
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
            'lr': 1e-5,  # Lower learning rate for finetuning
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
            'wandb_project': 'entrep-vitb-pretrained'
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train ENTRep with ViT-Base pretrained weights'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='pretrained/entrep_vit_b',
        help='Path to pretrained checkpoint'
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
        default='./checkpoints/entrep_vitb_pretrained',
        help='Output directory'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='entrep_vitb_pretrained',
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
        default=1e-5,
        help='Learning rate (lower for finetuning)'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra pretrained path
    if not os.path.exists(args.pretrained_path):
        logger.error(f"❌ Pretrained checkpoint not found: {args.pretrained_path}")
        logger.info("💡 Please make sure the pretrained model is in the pretrained/ directory")
        sys.exit(1)
    
    # Load hoặc tạo config
    if args.config and os.path.exists(args.config):
        logger.info(f"📋 Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.info("📋 Using default config with pretrained weights")
        config = create_config(args.pretrained_path)
    
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
    
    # Update pretrained path in config
    config['model']['checkpoint'] = args.pretrained_path
    
    # Log thông tin quan trọng
    logger.info("=" * 70)
    logger.info("🚀 TRAINING ENTRep - ViT-Base PRETRAINED")
    logger.info("=" * 70)
    logger.info(f"📦 Pretrained checkpoint: {args.pretrained_path}")
    logger.info(f"📊 Transform settings:")
    logger.info(f"   IMG_MEAN = {constants.IMG_MEAN}")
    logger.info(f"   IMG_STD = {constants.IMG_STD}")
    logger.info(f"   IMG_SIZE = {constants.IMG_SIZE}")
    logger.info(f"✅ Pretrained = True (loading from checkpoint)")
    logger.info("=" * 70)
    
    # Tạo model sử dụng factory
    logger.info("🏗️ Creating ENTRep model with pretrained weights...")
    model = create_model(**config['model'])
    
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

