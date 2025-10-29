"""
Script to train ENTRep model with contrastive learning
"""

import argparse
import yaml
from pathlib import Path
import torch

from modules.models.factory import create_model
from modules.trainer.entrep import ENTRepTrainer
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train ENTRep model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/entrep_contrastive.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for tracking'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='entrep-training',
        help='WandB project name'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"📋 Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.use_wandb:
        config['experiment']['use_wandb'] = True
    if args.wandb_project:
        config['experiment']['wandb_project'] = args.wandb_project
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
        
    # Create model
    logger.info("🏗️ Creating ENTRep model...")
    model_config = config['model']
    
    model = create_model(**model_config)
    
    # Print model information
    model_type = type(model).__name__
    logger.info(f"✅ Model created: {model_type}")
    
    # Print encoder info if available
    if hasattr(model, 'get_encoder_info'):
        encoder_info = model.get_encoder_info()
        for key, value in encoder_info.items():
            logger.info(f"   {key}: {value}")
    
    # Check wrapper usage
    if hasattr(model, 'vision_model'):
        vision_type = type(model.vision_model).__name__
        if vision_type in ['DinoV2Model', 'EntVitModel']:
            logger.info(f"   🎁 Vision wrapper: {vision_type}")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   📊 Total parameters: {total_params:,}")
    logger.info(f"   📊 Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("🎓 Creating trainer...")
    trainer_config = {
        **config.get('training', {}),
        **config.get('experiment', {})
    }
    
    trainer = ENTRepTrainer(
        model=model,
        config=config,
        output_dir=trainer_config.get('output_dir', './checkpoints'),
        experiment_name=args.experiment_name,
        use_wandb=trainer_config.get('use_wandb', False),
        wandb_project=trainer_config.get('wandb_project', 'entrep-training'),
        seed=trainer_config.get('seed', 42)
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("🚀 Starting training...")
    trainer.train()
    
    logger.info("🎉 Training completed!")


if __name__ == '__main__':
    main()
