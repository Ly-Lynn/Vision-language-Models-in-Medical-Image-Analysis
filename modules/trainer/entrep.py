"""
ENTRep trainer for contrastive learning
"""

import os
import json
from typing import Dict, Optional, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime

from ..models.entrep import ENTRepModel
from ..dataset.entrep import create_entrep_dataloader
from ..utils.logging_config import get_logger
from ..utils.helpers import setup_seed

logger = get_logger(__name__)


class ENTRepTrainer:
    """
    Trainer for ENTRep model with contrastive learning
    
    Features:
    - Contrastive learning với CLIP loss
    - Multi-GPU training support
    - Mixed precision training
    - Checkpoint saving/loading
    - WandB logging support
    - Validation tracking
    """
    
    def __init__(
        self,
        model: ENTRepModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = './checkpoints',
        experiment_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = 'entrep-training',
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize ENTRep trainer
        
        Args:
            model: ENTRepModel instance
            config: Training configuration dict
            device: Device to use (auto-detect if None)
            output_dir: Directory for checkpoints
            experiment_name: Name for experiment
            use_wandb: Whether to use WandB logging
            wandb_project: WandB project name
            seed: Random seed
        """
        self.model = model
        self.config = config
        self.seed = seed
        setup_seed(seed)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
        else:
            self.is_parallel = False
            
        # Output directory
        self.output_dir = Path(output_dir)
        if experiment_name is None:
            experiment_name = f"entrep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.checkpoint_dir = self.output_dir / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Initialize WandB if requested
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=config
            )
            wandb.watch(model)
            
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"✅ Trainer initialized")
        logger.info(f"📁 Checkpoints will be saved to: {self.checkpoint_dir}")
        logger.info(f"🔧 Device: {self.device}")
        
    def create_optimizer(self) -> Optimizer:
        """Create optimizer from config"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam').lower()
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        # Get model parameters
        if self.is_parallel:
            params = self.model.module.parameters()
        else:
            params = self.model.parameters()
            
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(
                params, 
                lr=lr, 
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
            
        logger.info(f"Created {opt_type} optimizer with lr={lr}")
        return optimizer
        
    def create_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config"""
        sched_config = self.config.get('scheduler', None)
        if sched_config is None:
            return None
            
        sched_type = sched_config.get('type', 'cosine').lower()
        
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config.get('T_max', 100),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_config.get('milestones', [30, 60, 90]),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
            
        logger.info(f"Created {sched_type} scheduler")
        return scheduler
        
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create train and validation dataloaders"""
        dataset_config = self.config.get('dataset', {})
        
        # Common config
        data_root = dataset_config.get('data_root', 'local_data/entrep')
        model_type = dataset_config.get('model_type', 'entrep')
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        # Create train dataloader
        train_loader = create_entrep_dataloader(
            data_root=data_root,
            split='train',
            model_type=model_type,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            tokenizer_name=dataset_config.get('tokenizer_name', None)
        )
        
        # Create validation dataloader
        val_loader = create_entrep_dataloader(
            data_root=data_root,
            split='val',
            model_type=model_type,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            tokenizer_name=dataset_config.get('tokenizer_name', None)
        )
        
        logger.info(f"📊 Created dataloaders:")
        logger.info(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        logger.info(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        
        return {
            'train': train_loader,
            'val': val_loader
        }
        
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        # Use mixed precision if scaler provided
        use_amp = scaler is not None
        
        with tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Forward pass with mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch.get('input_ids'),
                            pixel_values=batch['pixel_values'],
                            attention_mask=batch.get('attention_mask'),
                            return_loss=True
                        )
                        loss = outputs['loss_value']
                else:
                    outputs = self.model(
                        input_ids=batch.get('input_ids'),
                        pixel_values=batch['pixel_values'],
                        attention_mask=batch.get('attention_mask'),
                        return_loss=True
                    )
                    loss = outputs['loss_value']
                
                # Backward pass
                optimizer.zero_grad()
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to WandB
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
                    
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
        
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                    
            # Forward pass
            outputs = self.model(
                input_ids=batch.get('input_ids'),
                pixel_values=batch['pixel_values'],
                attention_mask=batch.get('attention_mask'),
                return_loss=True
            )
            loss = outputs['loss_value']
            
            total_loss += loss.item()
            
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
        
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        # Get model state dict
        if self.is_parallel:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Saved best checkpoint to {best_path}")
            
        # Save epoch checkpoint
        if self.current_epoch % self.config.get('save_every', 5) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
            torch.save(checkpoint, epoch_path)
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_parallel:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        logger.info(f"   Epoch: {self.current_epoch}, Step: {self.global_step}")
        
    def train(self):
        """Main training loop"""
        # Create optimizer and scheduler
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        
        # Create dataloaders
        dataloaders = self.create_dataloaders()
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Mixed precision training
        use_amp = self.config.get('use_amp', False) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training config
        num_epochs = self.config.get('num_epochs', 100)
        val_every = self.config.get('val_every', 1)
        
        logger.info(f"🎓 Starting training for {num_epochs} epochs")
        if use_amp:
            logger.info("⚡ Using mixed precision training")
            
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, 
                optimizer, 
                scheduler,
                scaler
            )
            logger.info(f"📈 Train loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if (epoch + 1) % val_every == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"📊 Val loss: {val_metrics['loss']:.4f}")
                
                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    
                # Save checkpoint
                metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss']
                }
                self.save_checkpoint(metrics, is_best)
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss'],
                        'val/epoch_loss': val_metrics['loss'],
                        'val/best_loss': self.best_val_loss
                    })
            else:
                # Save checkpoint without validation
                metrics = {'train_loss': train_metrics['loss']}
                self.save_checkpoint(metrics, is_best=False)
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss']
                    })
                    
        logger.info(f"\n🎉 Training completed!")
        logger.info(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
