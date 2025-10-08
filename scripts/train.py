#!/usr/bin/env python3
"""
Training script for medical vision-language models
"""

import argparse
import yaml
import torch
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.models.factory import create_model
from modules.dataset.factory import create_dataloader
from modules.losses import ImageTextContrastiveLoss, ImageSuperviseLoss
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_loss_function(model, loss_config: dict):
    """Create loss function based on config"""
    loss_type = loss_config.get('type', 'contrastive')
    
    if loss_type == 'contrastive':
        return ImageTextContrastiveLoss(model)
    elif loss_type == 'supervised':
        return ImageSuperviseLoss(model)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    logger.info(f"🚀 Training epoch {epoch + 1}...")
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = loss_fn(**batch)
        loss = outputs['loss_value']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"✅ Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train medical vision-language model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"📋 Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = create_model(**model_config)
    model = model.to(device)
    logger.info(f"🏗️ Created model: {model_config['model_type']}")
    
    # Create dataloaders
    dataset_config = config['dataset']
    train_dataloader = create_dataloader(
        split='train',
        **dataset_config
    )
    logger.info(f"📊 Created train dataloader with {len(train_dataloader)} batches")
    
    # Create loss function
    loss_config = config.get('loss', {'type': 'contrastive'})
    loss_fn = create_loss_function(model, loss_config)
    logger.info(f"🎯 Created loss function: {loss_config['type']}")
    
    # Create optimizer
    optimizer_config = config.get('optimizer', {'type': 'adam', 'lr': 1e-4})
    if optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config['lr'])
    elif optimizer_config['type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Training loop
    training_config = config.get('training', {'num_epochs': 10})
    num_epochs = training_config['num_epochs']
    
    logger.info(f"🎓 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, epoch)
        
        # Save checkpoint
        if (epoch + 1) % training_config.get('save_every', 5) == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    logger.info("🎉 Training completed!")


if __name__ == "__main__":
    main()
