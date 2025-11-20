"""
Training script for BioMedCLIP model using VisionLanguageTrainer

All configurations are loaded from YAML config file.

Usage:
    python scripts/train_biomedclip.py --config configs/biomedclip_finetune.yaml
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.biomedclip import BioMedCLIPModel
from modules.trainer import VisionLanguageTrainer
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train BioMedCLIP model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train_biomedclip.py
  
  # Train with custom config
  python scripts/train_biomedclip.py --config configs/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/biomedclip_finetune.yaml',
        help='Path to config file (default: configs/biomedclip_finetune.yaml)'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 BioMedCLIP Training")
    logger.info("=" * 80)
    
    # Load config
    logger.info(f"\n📝 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Extract model config
    model_config = config.get('model', {})
    model_name = model_config.get('model_name', 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    checkpoint = model_config.get('checkpoint', None)
    vision_pretrained = model_config.get('vision_pretrained', None)
    freeze_text = model_config.get('freeze_text', False)
    freeze_vision = model_config.get('freeze_vision', False)
    
    # Extract experiment config
    exp_config = config.get('experiment', {})
    experiment_name = exp_config.get('experiment_name', None)
    output_dir = exp_config.get('output_dir', './checkpoints')
    use_wandb = exp_config.get('use_wandb', False)
    wandb_project = exp_config.get('wandb_project', 'biomedclip-training')
    seed = exp_config.get('seed', 42)
    
    # Set model_type for trainer
    config['model_type'] = model_config.get('model_type', 'biomedclip')
    
    # Print config
    logger.info("\n⚙️  Configuration:")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Dataset: {config['dataset']['dataset_name']}")
    logger.info(f"   Batch size: {config['dataset']['batch_size']}")
    logger.info(f"   Learning rate: {config['optimizer']['lr']}")
    logger.info(f"   Epochs: {config['training']['num_epochs']}")
    logger.info(f"   Mixed precision: {config['training'].get('use_amp', False)}")
    logger.info(f"   Freeze text: {freeze_text}")
    logger.info(f"   Freeze vision: {freeze_vision}")
    if checkpoint:
        logger.info(f"   Resume from: {checkpoint}")
    if vision_pretrained:
        logger.info(f"   Vision pretrained: {vision_pretrained}")
    
    # Initialize model
    logger.info("\n📦 Initializing BioMedCLIP model...")
    model = BioMedCLIPModel(
        model_name=model_name,
        vision_pretrained=vision_pretrained,
        checkpoint=checkpoint
    )
    logger.info(f"✅ Model initialized on device: {model.device}")
    
    # Freeze components if requested
    if freeze_text:
        logger.info("🔒 Freezing text encoder...")
        for param in model.model.text.parameters():
            param.requires_grad = False
        logger.info("   Text encoder frozen")
    
    if freeze_vision:
        logger.info("🔒 Freezing vision encoder...")
        for param in model.model.visual.parameters():
            param.requires_grad = False
        logger.info("   Vision encoder frozen")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n📊 Model parameters:")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"   Frozen: {total_params - trainable_params:,}")
    
    # Auto-generate experiment name if not provided
    if experiment_name is None:
        freeze_suffix = ""
        if freeze_text:
            freeze_suffix = "_vision_only"
        elif freeze_vision:
            freeze_suffix = "_text_only"
        experiment_name = f"biomedclip_{config['dataset']['dataset_name']}{freeze_suffix}"
    
    # Create trainer
    logger.info("\n🎓 Creating trainer...")
    trainer = VisionLanguageTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        seed=seed
    )
    logger.info(f"✅ Trainer created")
    logger.info(f"   Experiment: {experiment_name}")
    logger.info(f"   Checkpoints: {trainer.checkpoint_dir}")
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("🏋️  Starting training...")
    logger.info("=" * 80 + "\n")
    
    try:
        trainer.train()
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\n📊 Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"💾 Checkpoints saved to: {trainer.checkpoint_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        logger.info(f"💾 Last checkpoint saved to: {trainer.checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

