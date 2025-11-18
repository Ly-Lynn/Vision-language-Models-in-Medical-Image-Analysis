"""
Training script for MedCLIP model using VisionLanguageTrainer

All configurations are loaded from YAML config file.

Usage:
    python scripts/train_medclip.py --config configs/medclip_vit_finetune.yaml
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.medclip import MedCLIPModel
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
        description='Train MedCLIP model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune MedCLIP-ViT
  python scripts/train_medclip.py --config configs/medclip_vit_finetune.yaml
  
  # Train from scratch
  python scripts/train_medclip.py --config configs/medclip_vit_from_scratch.yaml
  
  # Train with pretrained vision encoder
  python scripts/train_medclip.py --config configs/medclip_vision_pretrained.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/medclip_vit_finetune.yaml',
        help='Path to config file (default: configs/medclip_vit_finetune.yaml)'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 MedCLIP Training")
    logger.info("=" * 80)
    
    # Load config
    logger.info(f"\n📝 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Extract model config
    model_config = config.get('model', {})
    text_encoder_type = model_config.get('text_encoder_type', 'bert')
    vision_encoder_type = model_config.get('vision_encoder_type', 'vit')
    checkpoint = model_config.get('checkpoint', None)
    vision_pretrained = model_config.get('vision_pretrained', None)
    text_pretrained = model_config.get('text_pretrained', None)
    vision_checkpoint = model_config.get('vision_checkpoint', None)
    logit_scale_init_value = model_config.get('logit_scale_init_value', 0.07)
    freeze_text = model_config.get('freeze_text', False)
    freeze_vision = model_config.get('freeze_vision', False)
    
    # Extract experiment config
    exp_config = config.get('experiment', {})
    experiment_name = exp_config.get('experiment_name', None)
    output_dir = exp_config.get('output_dir', './checkpoints')
    use_wandb = exp_config.get('use_wandb', False)
    wandb_project = exp_config.get('wandb_project', 'medclip-training')
    seed = exp_config.get('seed', 42)
    
    # Set model_type for trainer
    config['model_type'] = model_config.get('model_type', 'medclip')
    
    # Print config
    logger.info("\n⚙️  Configuration:")
    logger.info(f"   Text Encoder: {text_encoder_type}")
    logger.info(f"   Vision Encoder: {vision_encoder_type}")
    logger.info(f"   Dataset: {config['dataset']['dataset_name']}")
    logger.info(f"   Batch size: {config['dataset']['batch_size']}")
    logger.info(f"   Learning rate: {config['optimizer']['lr']}")
    logger.info(f"   Epochs: {config['training']['num_epochs']}")
    logger.info(f"   Mixed precision: {config['training'].get('use_amp', False)}")
    logger.info(f"   Freeze text: {freeze_text}")
    logger.info(f"   Freeze vision: {freeze_vision}")
    if checkpoint:
        logger.info(f"   Checkpoint: {checkpoint}")
    if vision_pretrained:
        logger.info(f"   Vision pretrained: {vision_pretrained}")
    if text_pretrained:
        logger.info(f"   Text pretrained: {text_pretrained}")
    
    # Initialize model
    logger.info("\n📦 Initializing MedCLIP model...")
    model = MedCLIPModel(
        text_encoder_type=text_encoder_type,
        vision_encoder_type=vision_encoder_type,
        checkpoint=checkpoint,
        vision_checkpoint=vision_checkpoint,
        vision_pretrained=vision_pretrained,
        text_pretrained=text_pretrained,
        logit_scale_init_value=logit_scale_init_value
    )
    logger.info(f"✅ Model initialized on device: {model.device}")
    
    # Print encoder info
    encoder_info = model.get_encoder_info()
    logger.info(f"\n🔍 Encoder Information:")
    logger.info(f"   Text encoder: {encoder_info['text_encoder']} ({encoder_info['text_model_type']})")
    logger.info(f"   Vision encoder: {encoder_info['vision_encoder']} ({encoder_info['vision_model_type']})")
    
    # Freeze components if requested
    if freeze_text:
        logger.info("\n🔒 Freezing text encoder...")
        model.freeze_text_encoder()
    
    if freeze_vision:
        logger.info("🔒 Freezing vision encoder...")
        model.freeze_vision_encoder()
    
    # Get parameter statistics
    param_info = model.get_trainable_parameters()
    logger.info(f"\n📊 Model Parameters:")
    logger.info(f"   Vision encoder:")
    logger.info(f"      Total: {param_info['vision_total']:,}")
    logger.info(f"      Trainable: {param_info['vision_trainable']:,}")
    logger.info(f"      Frozen: {param_info['vision_frozen']:,}")
    logger.info(f"   Text encoder:")
    logger.info(f"      Total: {param_info['text_total']:,}")
    logger.info(f"      Trainable: {param_info['text_trainable']:,}")
    logger.info(f"      Frozen: {param_info['text_frozen']:,}")
    logger.info(f"   Overall:")
    logger.info(f"      Total: {param_info['total_parameters']:,}")
    logger.info(f"      Trainable: {param_info['total_trainable']:,} ({100*param_info['total_trainable']/param_info['total_parameters']:.1f}%)")
    
    # Auto-generate experiment name if not provided
    if experiment_name is None:
        encoder_suffix = f"{vision_encoder_type}"
        freeze_suffix = ""
        if freeze_text and freeze_vision:
            freeze_suffix = "_frozen"
        elif freeze_text:
            freeze_suffix = "_vision_only"
        elif freeze_vision:
            freeze_suffix = "_text_only"
        
        dataset_name = config['dataset']['dataset_name']
        experiment_name = f"medclip_{encoder_suffix}_{dataset_name}{freeze_suffix}"
    
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
    logger.info(f"   Output directory: {output_dir}")
    
    # Log training strategy
    if vision_pretrained and not checkpoint:
        logger.info(f"\n💡 Training Strategy: Using pretrained vision encoder")
    elif checkpoint:
        logger.info(f"\n💡 Training Strategy: Fine-tuning from checkpoint")
    else:
        logger.info(f"\n💡 Training Strategy: Training from scratch")
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("🏋️  Starting training...")
    logger.info("=" * 80 + "\n")
    
    try:
        trainer.train()
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\n📊 Training Summary:")
        logger.info(f"   Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"   Total epochs: {config['training']['num_epochs']}")
        logger.info(f"\n💾 Checkpoints saved to: {trainer.checkpoint_dir}")
        logger.info(f"   Best model: {os.path.join(trainer.checkpoint_dir, 'best_model.pth')}")
        logger.info(f"   Final model: {os.path.join(trainer.checkpoint_dir, 'final_model.pth')}")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        logger.info(f"💾 Last checkpoint saved to: {trainer.checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save emergency checkpoint
        try:
            emergency_path = os.path.join(trainer.checkpoint_dir, 'emergency_checkpoint.pth')
            model.save_pretrained(emergency_path)
            logger.info(f"\n💾 Emergency checkpoint saved to: {emergency_path}")
        except:
            logger.error("Failed to save emergency checkpoint")
        
        raise


if __name__ == '__main__':
    main()

