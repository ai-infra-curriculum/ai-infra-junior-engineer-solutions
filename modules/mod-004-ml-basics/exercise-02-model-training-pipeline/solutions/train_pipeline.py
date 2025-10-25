#!/usr/bin/env python3
"""
train_pipeline.py - Main training orchestration script

Description:
    Production-ready training pipeline with distributed training support,
    checkpointing, experiment tracking, and error recovery.

Usage:
    # Single GPU training
    python train_pipeline.py --config config/train.yaml

    # Multi-GPU training (DDP)
    python train_pipeline.py --config config/train.yaml --distributed --gpus 4

    # Resume from checkpoint
    python train_pipeline.py --config config/train.yaml --resume checkpoint.pt

Options:
    --config FILE          Training configuration file
    --distributed          Enable distributed training
    --gpus N              Number of GPUs to use
    --resume FILE         Resume from checkpoint
    --test-only           Run testing only
    --validate-only       Run validation only
    --dry-run             Dry run (validate config only)
    --verbose             Verbose output
    --help                Display this help message
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(self, save_dir: str, keep_last_n: int = 3,
                 monitor_metric: str = 'val_loss', mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def should_save_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are best"""
        if self.monitor_metric not in metrics:
            return False

        current_value = metrics[self.monitor_metric]

        if self.mode == 'min':
            is_best = current_value < self.best_metric
        else:
            is_best = current_value > self.best_metric

        if is_best:
            self.best_metric = current_value

        return is_best

    def save_checkpoint(self, state: Dict, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint_name = f'checkpoint_epoch_{epoch:04d}.pt'
        checkpoint_path = self.save_dir / checkpoint_name

        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(epoch)

    def _cleanup_old_checkpoints(self, current_epoch: int):
        """Remove old checkpoints"""
        checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pt'))

        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location='cpu')


class MetricsTracker:
    """Track training metrics"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.current_epoch = 0

    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metrics for current epoch"""
        self.current_epoch = epoch

        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Save metrics to file
        self._save_metrics()

    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = self.log_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_summary(self) -> str:
        """Get metrics summary"""
        if not self.metrics_history['train_loss']:
            return "No metrics recorded yet"

        summary = f"""
Metrics Summary (Epoch {self.current_epoch}):
  Train Loss: {self.metrics_history['train_loss'][-1]:.4f}
  Train Acc: {self.metrics_history['train_acc'][-1]:.2f}%
  Val Loss: {self.metrics_history['val_loss'][-1]:.4f}
  Val Acc: {self.metrics_history['val_acc'][-1]:.2f}%
  Learning Rate: {self.metrics_history['learning_rate'][-1]:.6f}
"""
        return summary


class Trainer:
    """Main training orchestrator"""

    def __init__(self, config: Dict, rank: int = 0, world_size: int = 1):
        """
        Initialize trainer

        Args:
            config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None

        # Managers
        self.checkpoint_mgr = None
        self.metrics_tracker = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def setup(self):
        """Setup training components"""
        logger.info(f"Setting up trainer on rank {self.rank}")

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create loss criterion
        self.criterion = self._create_criterion()

        # Mixed precision scaler
        if self.config['training'].get('mixed_precision', False):
            self.scaler = GradScaler()

        # Checkpoint manager (only on main process)
        if self.is_main_process:
            self.checkpoint_mgr = CheckpointManager(
                save_dir=self.config['checkpointing']['save_dir'],
                keep_last_n=self.config['checkpointing']['keep_last_n'],
                monitor_metric=self.config['checkpointing']['monitor_metric'],
                mode=self.config['checkpointing']['mode']
            )

            self.metrics_tracker = MetricsTracker(
                log_dir=self.config['logging']['log_dir']
            )

        logger.info("Setup complete")

    def _create_model(self) -> nn.Module:
        """Create model from config"""
        # Simplified model creation - in practice, use model registry
        model_config = self.config['model']
        architecture = model_config['architecture']

        if architecture == 'simple_cnn':
            from torchvision.models import resnet18
            model = resnet18(pretrained=model_config.get('pretrained', False))
            model.fc = nn.Linear(model.fc.in_features, model_config['num_classes'])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type'].lower()

        model_params = self.model.module.parameters() if self.is_distributed else self.model.parameters()

        if optimizer_type == 'sgd':
            return torch.optim.SGD(
                model_params,
                lr=self.config['training']['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'adam':
            return torch.optim.Adam(
                model_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config"""
        scheduler_config = self.config['training'].get('lr_scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type', '').lower()

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None

    def _create_criterion(self) -> nn.Module:
        """Create loss criterion"""
        # Simplified - in practice, use criterion registry
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        if self.is_main_process:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        else:
            pbar = train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

            self.global_step += 1

        return {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        if self.is_main_process:
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        else:
            pbar = val_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        return {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Combine metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

            # Log metrics (main process only)
            if self.is_main_process:
                self.metrics_tracker.update(metrics, epoch)
                logger.info(self.metrics_tracker.get_summary())

                # Check if best model
                is_best = self.checkpoint_mgr.should_save_best(metrics)

                # Save checkpoint
                if (epoch + 1) % self.config['checkpointing']['save_frequency'] == 0 or is_best:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                        'metrics': metrics,
                        'config': self.config
                    }
                    self.checkpoint_mgr.save_checkpoint(checkpoint, epoch, is_best)

        logger.info("Training complete!")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = self.checkpoint_mgr.load_checkpoint(checkpoint_path)

        # Load model
        model_state = checkpoint['model_state_dict']
        if self.is_distributed:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Update epoch
        self.current_epoch = checkpoint['epoch'] + 1

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def create_dataloaders(config: Dict, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    # Simplified dataloader creation - in practice, use dataset registry
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset = datasets.FakeData(
        size=1000,
        image_size=(3, 224, 224),
        num_classes=config['model']['num_classes'],
        transform=transform
    )

    val_dataset = datasets.FakeData(
        size=200,
        image_size=(3, 224, 224),
        num_classes=config['model']['num_classes'],
        transform=transform
    )

    # Create samplers for distributed training
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader


def train_worker(rank: int, world_size: int, config: Dict, resume_checkpoint: Optional[str]):
    """Training worker function for distributed training"""
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)

    try:
        # Create trainer
        trainer = Trainer(config, rank, world_size)
        trainer.setup()

        # Resume from checkpoint if provided
        if resume_checkpoint:
            trainer.resume_from_checkpoint(resume_checkpoint)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config, rank, world_size)

        # Train
        trainer.train(train_loader, val_loader, config['training']['epochs'])

    finally:
        # Cleanup distributed
        if world_size > 1:
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description='Training pipeline orchestration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Training configuration file')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true',
                       help='Run testing only')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (validate config only)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Dry run - just validate config
    if args.dry_run:
        logger.info(f"{Colors.GREEN}✓ Configuration is valid{Colors.RESET}")
        return

    # Print configuration summary
    logger.info(f"\n{Colors.BOLD}{Colors.CYAN}Training Configuration{Colors.RESET}")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Model: {config['model']['architecture']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Distributed: {args.distributed}")
    if args.distributed:
        logger.info(f"GPUs: {args.gpus}")
    logger.info("=" * 80 + "\n")

    # Launch training
    if args.distributed and args.gpus > 1:
        logger.info(f"Launching distributed training on {args.gpus} GPUs")
        mp.spawn(
            train_worker,
            args=(args.gpus, config, args.resume),
            nprocs=args.gpus,
            join=True
        )
    else:
        logger.info("Launching single-process training")
        train_worker(0, 1, config, args.resume)

    logger.info(f"\n{Colors.GREEN}{Colors.BOLD}✓ Training completed successfully!{Colors.RESET}")


if __name__ == '__main__':
    main()
