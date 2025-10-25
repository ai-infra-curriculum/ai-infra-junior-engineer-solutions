# Exercise 02: Model Training Pipeline

## Overview

Build production-ready ML training pipelines that handle data management, distributed training, monitoring, checkpointing, and model versioning. Learn to create robust, reproducible training workflows that scale from single GPU to multi-node clusters.

## Learning Objectives

- ✅ Design and implement end-to-end training pipelines
- ✅ Manage training data efficiently at scale
- ✅ Configure and monitor distributed training jobs
- ✅ Implement robust checkpointing and recovery
- ✅ Track experiments and model versioning
- ✅ Optimize training performance and resource utilization
- ✅ Handle failures and implement retry logic
- ✅ Generate training reports and visualizations

## Topics Covered

### 1. Training Data Management

#### Data Pipeline Architecture

```
Raw Data → Preprocessing → Feature Engineering → Data Loading → Training
    ↓           ↓                ↓                    ↓            ↓
  Storage    Validation      Caching           Batching      Monitoring
```

#### Data Organization Best Practices

```bash
# Recommended data structure
data/
├── raw/                    # Original, immutable data
│   ├── train/
│   ├── val/
│   └── test/
├── processed/              # Preprocessed data
│   ├── train/
│   ├── val/
│   └── test/
├── features/               # Extracted features
│   └── version-1/
├── splits/                 # Train/val/test split info
│   └── split-20240101.json
└── metadata/
    ├── statistics.json
    └── schema.json
```

#### Data Validation

```python
import pandas as pd
from typing import Dict, List
import json

class DataValidator:
    """Validate training data before pipeline execution"""

    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            self.schema = json.load(f)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """Validate DataFrame against schema"""
        issues = []

        # Check required columns
        required_cols = self.schema['required_columns']
        missing = set(required_cols) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check data types
        for col, expected_type in self.schema['dtypes'].items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    issues.append(f"Column '{col}' type mismatch: "
                                f"expected {expected_type}, got {actual_type}")

        # Check for nulls
        null_counts = df.isnull().sum()
        for col in self.schema['non_nullable']:
            if null_counts[col] > 0:
                issues.append(f"Column '{col}' has {null_counts[col]} null values")

        # Check value ranges
        for col, constraints in self.schema.get('constraints', {}).items():
            if col in df.columns:
                if 'min' in constraints:
                    violations = (df[col] < constraints['min']).sum()
                    if violations > 0:
                        issues.append(f"Column '{col}' has {violations} values "
                                    f"below minimum {constraints['min']}")

                if 'max' in constraints:
                    violations = (df[col] > constraints['max']).sum()
                    if violations > 0:
                        issues.append(f"Column '{col}' has {violations} values "
                                    f"above maximum {constraints['max']}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

# Usage
validator = DataValidator('data/metadata/schema.json')
df = pd.read_csv('data/processed/train/data.csv')
result = validator.validate_dataframe(df)

if not result['valid']:
    print("Validation failed:")
    for issue in result['issues']:
        print(f"  - {issue}")
```

#### Data Preprocessing Pipeline

```python
from typing import Callable, List
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PreprocessingPipeline:
    """Composable preprocessing pipeline"""

    def __init__(self):
        self.steps: List[tuple] = []
        self.fitted = False

    def add_step(self, name: str, transform_fn: Callable):
        """Add preprocessing step"""
        self.steps.append((name, transform_fn))
        return self

    def fit(self, X, y=None):
        """Fit all preprocessing steps"""
        X_transformed = X.copy()

        for name, transform_fn in self.steps:
            if hasattr(transform_fn, 'fit'):
                transform_fn.fit(X_transformed, y)
            X_transformed = transform_fn.transform(X_transformed)

        self.fitted = True
        return self

    def transform(self, X):
        """Apply all preprocessing steps"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_transformed = X.copy()
        for name, transform_fn in self.steps:
            X_transformed = transform_fn.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# Example usage
pipeline = PreprocessingPipeline()
pipeline.add_step('scaler', StandardScaler())
pipeline.add_step('normalize', lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))

X_train_processed = pipeline.fit_transform(X_train)
X_val_processed = pipeline.transform(X_val)
```

#### Data Caching Strategies

```python
import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path

class CachedDataset(Dataset):
    """Dataset with disk caching for expensive preprocessing"""

    def __init__(self, data_dir: str, preprocess_fn: Callable,
                 cache_dir: str = None):
        self.data_dir = Path(data_dir)
        self.preprocess_fn = preprocess_fn
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

        # List all data files
        self.data_files = sorted(list(self.data_dir.glob('*.pt')))

    def __len__(self):
        return len(self.data_files)

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for index"""
        filename = self.data_files[idx].stem
        return self.cache_dir / f"{filename}_cached.pkl"

    def __getitem__(self, idx: int):
        cache_path = self._get_cache_path(idx)

        # Try to load from cache
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Load and preprocess
        data = torch.load(self.data_files[idx])
        processed = self.preprocess_fn(data)

        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(processed, f)

        return processed

# Usage
def preprocess(data):
    # Expensive preprocessing
    return data * 2  # Simplified example

dataset = CachedDataset(
    data_dir='data/processed/train',
    preprocess_fn=preprocess,
    cache_dir='data/cache'
)
```

### 2. Training Configuration Management

#### Configuration File Structure

```yaml
# config/train_config.yaml
experiment:
  name: "resnet50_imagenet"
  version: "v1.0"
  description: "ResNet50 training on ImageNet"
  tags: ["resnet", "imagenet", "baseline"]

data:
  train_dir: "data/processed/train"
  val_dir: "data/processed/val"
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 1000
  dropout: 0.2

training:
  batch_size: 256
  epochs: 90
  learning_rate: 0.1
  lr_scheduler:
    type: "cosine"
    T_max: 90
    eta_min: 0.0001

  optimizer:
    type: "sgd"
    momentum: 0.9
    weight_decay: 0.0001

  mixed_precision: true
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

distributed:
  enabled: true
  backend: "nccl"
  world_size: 4
  strategy: "ddp"  # DDP, FSDP, or DeepSpeed

checkpointing:
  save_dir: "checkpoints"
  save_frequency: 5  # epochs
  keep_last_n: 3
  save_best: true
  monitor_metric: "val_accuracy"
  mode: "max"

logging:
  log_dir: "logs"
  log_frequency: 100  # steps
  wandb:
    enabled: true
    project: "ml-training"
    entity: "my-team"

  tensorboard:
    enabled: true
    log_dir: "runs"

validation:
  frequency: 1  # epochs
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_accuracy"
    mode: "max"

resources:
  gpu_ids: [0, 1, 2, 3]
  fp16: true
  allow_tf32: true
```

#### Configuration Loading and Validation

```python
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class ExperimentConfig:
    name: str
    version: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

@dataclass
class ModelConfig:
    architecture: str
    pretrained: bool = True
    num_classes: int = 1000
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    lr_scheduler: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None

@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    strategy: str = "ddp"

@dataclass
class CheckpointingConfig:
    save_dir: str
    save_frequency: int = 1
    keep_last_n: int = 3
    save_best: bool = True
    monitor_metric: str = "val_loss"
    mode: str = "min"

@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    distributed: DistributedConfig
    checkpointing: CheckpointingConfig
    logging: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            experiment=ExperimentConfig(**config_dict['experiment']),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            distributed=DistributedConfig(**config_dict.get('distributed', {})),
            checkpointing=CheckpointingConfig(**config_dict['checkpointing']),
            logging=config_dict.get('logging', {}),
            validation=config_dict.get('validation', {}),
            resources=config_dict.get('resources', {})
        )

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []

        # Check data directories exist
        if not Path(self.data.train_dir).exists():
            errors.append(f"Training directory not found: {self.data.train_dir}")

        if not Path(self.data.val_dir).exists():
            errors.append(f"Validation directory not found: {self.data.val_dir}")

        # Check distributed config
        if self.distributed.enabled:
            if self.distributed.world_size < 2:
                errors.append("Distributed training requires world_size >= 2")

        # Check batch size is divisible by world size
        if self.training.batch_size % self.distributed.world_size != 0:
            errors.append(f"Batch size {self.training.batch_size} not divisible "
                        f"by world size {self.distributed.world_size}")

        return errors

# Usage
config = Config.from_yaml('config/train_config.yaml')
errors = config.validate()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    exit(1)
```

### 3. Training Loop Implementation

#### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

class Trainer:
    """Basic training loop implementation"""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100. * correct / total
        }

    def train(self, num_epochs: int):
        """Main training loop"""
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])

            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.2f}%")
```

#### Training Loop with Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer(Trainer):
    """Training loop with automatic mixed precision"""

    def __init__(self, *args, use_amp: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with AMP"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with autocast
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass with gradient scaling
            if self.use_amp:
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

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total
        }
```

#### Training Loop with Gradient Accumulation

```python
class GradientAccumulationTrainer(Trainer):
    """Training loop with gradient accumulation"""

    def __init__(self, *args, accumulation_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Normalize loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Metrics (use unnormalized loss for display)
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total
        }
```

### 4. Distributed Training

#### PyTorch DDP (Distributed Data Parallel)

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_ddp(rank: int, world_size: int, config: Config):
    """Distributed training function"""
    print(f"Running DDP on rank {rank}")

    # Setup DDP
    setup_ddp(rank, world_size)

    # Create model and move to GPU
    model = create_model(config.model)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dataset with distributed sampler
    train_dataset = create_dataset(config.data.train_dir)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size // world_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=0.9
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.training.epochs):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pt')

    cleanup_ddp()

# Launch distributed training
def main():
    world_size = torch.cuda.device_count()
    config = Config.from_yaml('config/train_config.yaml')

    mp.spawn(
        train_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

#### Launch Script for Multi-Node Training

```bash
#!/bin/bash
# train_distributed.sh - Launch distributed training

# Configuration
NUM_NODES=2
NUM_GPUS_PER_NODE=4
MASTER_ADDR="192.168.1.100"
MASTER_PORT=29500
CONFIG_FILE="config/train_config.yaml"

# Node rank (set this differently on each node)
NODE_RANK=${1:-0}

echo "Starting distributed training..."
echo "  Nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS_PER_NODE"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Node rank: $NODE_RANK"

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config $CONFIG_FILE \
    --distributed
```

### 5. Checkpointing and Model Saving

#### Checkpoint Manager

```python
import torch
from pathlib import Path
import json
from typing import Dict, Optional
import shutil

class CheckpointManager:
    """Manage model checkpoints with versioning"""

    def __init__(self, save_dir: str, keep_last_n: int = 3,
                 save_best: bool = True, monitor_metric: str = 'val_loss',
                 mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

        # Load existing checkpoints
        self._load_checkpoint_list()

    def _load_checkpoint_list(self):
        """Load list of existing checkpoints"""
        checkpoint_file = self.save_dir / 'checkpoints.json'
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                self.checkpoints = json.load(f)

    def _save_checkpoint_list(self):
        """Save list of checkpoints"""
        with open(self.save_dir / 'checkpoints.json', 'w') as f:
            json.dump(self.checkpoints, f, indent=2)

    def save_checkpoint(self, state_dict: Dict, epoch: int,
                       metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint_name = f'checkpoint_epoch_{epoch:04d}.pt'
        checkpoint_path = self.save_dir / checkpoint_name

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'metrics': metrics,
            'is_best': is_best
        }, checkpoint_path)

        # Update checkpoint list
        self.checkpoints.append({
            'epoch': epoch,
            'path': str(checkpoint_path),
            'metrics': metrics,
            'is_best': is_best
        })

        # Save best model separately
        if is_best and self.save_best:
            best_path = self.save_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)

        # Remove old checkpoints
        self._cleanup_old_checkpoints()

        # Save checkpoint list
        self._save_checkpoint_list()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only last N"""
        if len(self.checkpoints) > self.keep_last_n:
            # Sort by epoch
            sorted_checkpoints = sorted(self.checkpoints,
                                      key=lambda x: x['epoch'])

            # Keep last N and any marked as best
            to_keep = set()

            # Keep last N
            for ckpt in sorted_checkpoints[-self.keep_last_n:]:
                to_keep.add(ckpt['path'])

            # Keep best
            for ckpt in sorted_checkpoints:
                if ckpt.get('is_best', False):
                    to_keep.add(ckpt['path'])

            # Remove others
            for ckpt in sorted_checkpoints:
                if ckpt['path'] not in to_keep:
                    path = Path(ckpt['path'])
                    if path.exists():
                        path.unlink()
                    self.checkpoints.remove(ckpt)

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

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint"""
        return torch.load(checkpoint_path)

    def load_best_checkpoint(self) -> Optional[Dict]:
        """Load best checkpoint"""
        best_path = self.save_dir / 'best_model.pt'
        if best_path.exists():
            return self.load_checkpoint(str(best_path))
        return None

# Usage
checkpoint_mgr = CheckpointManager(
    save_dir='checkpoints',
    keep_last_n=3,
    save_best=True,
    monitor_metric='val_accuracy',
    mode='max'
)

# During training
for epoch in range(num_epochs):
    # Train and validate
    metrics = {'val_accuracy': 0.95, 'val_loss': 0.2}

    # Check if best
    is_best = checkpoint_mgr.should_save_best(metrics)

    # Save checkpoint
    checkpoint_mgr.save_checkpoint(
        state_dict=model.state_dict(),
        epoch=epoch,
        metrics=metrics,
        is_best=is_best
    )
```

#### Resume Training from Checkpoint

```python
def resume_training(checkpoint_path: str, model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """Resume training from checkpoint"""

    checkpoint = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(checkpoint['state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if available
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get epoch number
    start_epoch = checkpoint['epoch'] + 1

    # Load metrics history
    history = checkpoint.get('history', {})

    print(f"Resumed training from epoch {start_epoch}")
    print(f"Best metric so far: {checkpoint.get('best_metric', 'N/A')}")

    return start_epoch, history

# Usage
start_epoch = 0
history = {}

if resume_from_checkpoint:
    start_epoch, history = resume_training(
        checkpoint_path='checkpoints/checkpoint_epoch_0042.pt',
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )

# Continue training
for epoch in range(start_epoch, num_epochs):
    # Training code
    pass
```

### 6. Experiment Tracking and Logging

#### Weights & Biases Integration

```python
import wandb

class WandbLogger:
    """Weights & Biases logging integration"""

    def __init__(self, config: Config):
        self.config = config
        self.run = None

    def init(self):
        """Initialize W&B run"""
        wandb_config = self.config.logging.get('wandb', {})

        if not wandb_config.get('enabled', False):
            return

        self.run = wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=self.config.experiment.name,
            config=self.config.__dict__,
            tags=self.config.experiment.tags
        )

        # Log system info
        wandb.config.update({
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        })

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics"""
        if self.run:
            wandb.log(metrics, step=step)

    def log_images(self, images: Dict[str, any], step: int):
        """Log images"""
        if self.run:
            wandb_images = {
                key: wandb.Image(img) for key, img in images.items()
            }
            wandb.log(wandb_images, step=step)

    def log_model(self, model_path: str, name: str):
        """Log model artifact"""
        if self.run:
            artifact = wandb.Artifact(name, type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Finish W&B run"""
        if self.run:
            wandb.finish()

# Usage
logger = WandbLogger(config)
logger.init()

for epoch in range(num_epochs):
    # Training
    train_metrics = train_epoch()
    val_metrics = validate()

    # Log metrics
    logger.log_metrics({
        **train_metrics,
        **val_metrics,
        'learning_rate': optimizer.param_groups[0]['lr']
    }, step=epoch)

logger.finish()
```

#### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """TensorBoard logging integration"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram"""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """Log image"""
        self.writer.add_image(tag, img_tensor, step)

    def log_graph(self, model: nn.Module, input_to_model: torch.Tensor):
        """Log model graph"""
        self.writer.add_graph(model, input_to_model)

    def log_pr_curve(self, tag: str, labels: torch.Tensor,
                     predictions: torch.Tensor, step: int):
        """Log precision-recall curve"""
        self.writer.add_pr_curve(tag, labels, predictions, step)

    def close(self):
        """Close writer"""
        self.writer.close()

# Usage
tb_logger = TensorBoardLogger('runs/experiment1')

# Log model graph
dummy_input = torch.randn(1, 3, 224, 224).to(device)
tb_logger.log_graph(model, dummy_input)

for epoch in range(num_epochs):
    # Training
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Log training loss
        global_step = epoch * len(train_loader) + batch_idx
        tb_logger.log_scalar('Loss/train', loss.item(), global_step)

    # Validation
    val_loss, val_acc = validate()
    tb_logger.log_scalars('Metrics', {
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }, epoch)

    # Log histograms of model parameters
    for name, param in model.named_parameters():
        tb_logger.log_histogram(f'Parameters/{name}', param, epoch)
        if param.grad is not None:
            tb_logger.log_histogram(f'Gradients/{name}', param.grad, epoch)

tb_logger.close()
```

### 7. Training Monitoring and Alerts

#### Training Monitor

```python
import time
from collections import deque
from typing import Deque, Dict
import smtplib
from email.mime.text import MIMEText

class TrainingMonitor:
    """Monitor training progress and send alerts"""

    def __init__(self, alert_email: Optional[str] = None):
        self.alert_email = alert_email
        self.start_time = time.time()

        # Track metrics
        self.loss_history: Deque[float] = deque(maxlen=100)
        self.gpu_memory: Deque[float] = deque(maxlen=100)

        # Alert thresholds
        self.max_loss_stall_epochs = 10
        self.max_gpu_memory_pct = 95
        self.min_gpu_utilization = 20

    def update(self, metrics: Dict[str, float], epoch: int):
        """Update monitoring metrics"""
        # Track loss
        if 'train_loss' in metrics:
            self.loss_history.append(metrics['train_loss'])

        # Check for issues
        self._check_loss_stall(epoch)
        self._check_gpu_memory()
        self._check_gpu_utilization()

    def _check_loss_stall(self, epoch: int):
        """Check if loss has stalled"""
        if len(self.loss_history) < self.max_loss_stall_epochs:
            return

        recent_losses = list(self.loss_history)[-self.max_loss_stall_epochs:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])

        if loss_change < 0.001:  # Threshold for "stalled"
            message = (f"⚠️ Training may have stalled at epoch {epoch}\n"
                      f"Loss has not changed significantly in "
                      f"{self.max_loss_stall_epochs} epochs")
            self._send_alert(message)

    def _check_gpu_memory(self):
        """Check GPU memory usage"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            memory_pct = memory_used * 100

            if memory_pct > self.max_gpu_memory_pct:
                message = (f"⚠️ High GPU memory usage: {memory_pct:.1f}%\n"
                          f"Consider reducing batch size or model size")
                self._send_alert(message)

    def _check_gpu_utilization(self):
        """Check GPU utilization"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            utilization = float(result.stdout.strip())

            if utilization < self.min_gpu_utilization:
                message = (f"⚠️ Low GPU utilization: {utilization}%\n"
                          f"GPU may be underutilized - check data loading")
                self._send_alert(message)
        except Exception:
            pass

    def _send_alert(self, message: str):
        """Send alert via email"""
        print(f"\n{message}\n")

        if not self.alert_email:
            return

        try:
            msg = MIMEText(message)
            msg['Subject'] = 'Training Alert'
            msg['From'] = 'training-monitor@example.com'
            msg['To'] = self.alert_email

            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send alert email: {e}")

    def get_training_summary(self) -> str:
        """Get training summary"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        summary = f"""
Training Summary:
  Duration: {hours}h {minutes}m
  Average Loss: {sum(self.loss_history) / len(self.loss_history):.4f}
  GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB
"""
        return summary

# Usage
monitor = TrainingMonitor(alert_email='user@example.com')

for epoch in range(num_epochs):
    # Training
    metrics = train_epoch()

    # Update monitor
    monitor.update(metrics, epoch)

print(monitor.get_training_summary())
```

### 8. Error Handling and Recovery

#### Training with Automatic Recovery

```python
class RobustTrainer:
    """Trainer with automatic error recovery"""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 checkpoint_dir: str, max_retries: int = 3):
        self.model = model
        self.train_loader = train_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_retries = max_retries

        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)

    def train_with_recovery(self, num_epochs: int):
        """Train with automatic recovery on failure"""
        start_epoch = 0
        retries = 0

        while start_epoch < num_epochs:
            try:
                # Try to load latest checkpoint
                latest_checkpoint = self._get_latest_checkpoint()
                if latest_checkpoint:
                    start_epoch = self._load_checkpoint(latest_checkpoint)
                    print(f"Resumed from epoch {start_epoch}")

                # Train
                for epoch in range(start_epoch, num_epochs):
                    try:
                        self._train_epoch(epoch)

                        # Save checkpoint every epoch
                        self._save_checkpoint(epoch)

                        # Reset retry counter on successful epoch
                        retries = 0

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM error at epoch {epoch}, clearing cache...")
                            torch.cuda.empty_cache()

                            if retries < self.max_retries:
                                retries += 1
                                print(f"Retrying epoch {epoch} (attempt {retries}/{self.max_retries})")
                                continue
                            else:
                                raise
                        else:
                            raise

                # Training completed successfully
                break

            except Exception as e:
                print(f"Training failed with error: {e}")

                if retries < self.max_retries:
                    retries += 1
                    print(f"Retrying from last checkpoint (attempt {retries}/{self.max_retries})")
                    time.sleep(10)  # Wait before retry
                else:
                    print("Max retries exceeded, aborting training")
                    raise

    def _train_epoch(self, epoch: int):
        """Train single epoch"""
        self.model.train()
        for inputs, targets in self.train_loader:
            # Training code
            pass

    def _save_checkpoint(self, epoch: int):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return epoch"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'] + 1

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None

# Usage
trainer = RobustTrainer(
    model=model,
    train_loader=train_loader,
    checkpoint_dir='checkpoints',
    max_retries=3
)

trainer.train_with_recovery(num_epochs=100)
```

---

## Project: End-to-End Training Pipeline

Build a complete, production-ready training pipeline that incorporates all concepts learned.

### Requirements

**Pipeline Components:**
1. Data validation and preprocessing
2. Configuration management
3. Distributed training support
4. Checkpoint management
5. Experiment tracking
6. Training monitoring
7. Error recovery
8. Report generation

**Features:**
- CLI interface for easy configuration
- Automatic resume from checkpoint
- Multi-GPU and multi-node support
- Real-time monitoring dashboard
- Email alerts for issues
- Comprehensive logging
- Performance profiling
- Model versioning

### Implementation

See `solutions/` directory for complete implementations:

1. **`prepare_dataset.py`** - Data validation and preprocessing pipeline
2. **`train_pipeline.py`** - Main training orchestration script
3. **`monitor_training.py`** - Real-time training monitor with dashboard
4. **`generate_report.py`** - Training report generator with visualizations

---

## Practice Problems

### Problem 1: Data Pipeline Optimizer

Create a tool that:
- Analyzes data loading performance
- Identifies bottlenecks (I/O, preprocessing, GPU transfer)
- Recommends optimal DataLoader settings
- Benchmarks different configurations
- Generates performance report

### Problem 2: Training Job Manager

Create a system that:
- Manages multiple training jobs
- Monitors resource usage per job
- Handles job queuing and scheduling
- Provides dashboard for all jobs
- Automatically restarts failed jobs

### Problem 3: Experiment Tracker

Build a custom experiment tracking system that:
- Tracks all hyperparameters
- Logs metrics over time
- Compares multiple experiments
- Generates comparison reports
- Provides web interface for visualization

---

## Best Practices

### 1. Data Management

```python
# Always validate data before training
validator = DataValidator('schema.json')
result = validator.validate_dataframe(train_df)
assert result['valid'], f"Data validation failed: {result['issues']}"

# Use data versioning
data_version = "v1.2.0"
data_path = f"data/processed/{data_version}/train"

# Document data statistics
stats = {
    'num_samples': len(train_df),
    'num_features': train_df.shape[1],
    'class_distribution': train_df['label'].value_counts().to_dict(),
    'date_created': datetime.now().isoformat()
}
```

### 2. Reproducibility

```python
# Set all random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Log all configuration
config_dict = {
    'model': model_config,
    'data': data_config,
    'training': training_config,
    'environment': {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0)
    }
}
```

### 3. Checkpointing

```python
# Save complete training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict() if scaler else None,
    'random_state': torch.get_rng_state(),
    'numpy_random_state': np.random.get_state(),
    'metrics': metrics,
    'config': config,
    'best_metric': best_metric
}

torch.save(checkpoint, checkpoint_path)
```

### 4. Monitoring

```python
# Monitor training health
def check_training_health(metrics: Dict[str, float], epoch: int):
    # Check for NaN/Inf
    if not np.isfinite(metrics['train_loss']):
        raise ValueError(f"Invalid loss at epoch {epoch}: {metrics['train_loss']}")

    # Check for gradient explosion
    if metrics.get('grad_norm', 0) > 100:
        print(f"Warning: Large gradient norm at epoch {epoch}: {metrics['grad_norm']}")

    # Check GPU utilization
    gpu_util = get_gpu_utilization()
    if gpu_util < 50:
        print(f"Warning: Low GPU utilization at epoch {epoch}: {gpu_util}%")
```

---

## Validation

Test your training pipeline:

```bash
# Validate data
python solutions/prepare_dataset.py --validate --data-dir data/raw

# Test training (single epoch)
python solutions/train_pipeline.py --config config/test.yaml --epochs 1

# Test distributed training (local)
python solutions/train_pipeline.py --config config/train.yaml --distributed --gpus 2

# Monitor training
python solutions/monitor_training.py --experiment-id exp_001

# Generate report
python solutions/generate_report.py --checkpoint checkpoints/best_model.pt
```

---

## Resources

- [PyTorch Training Tips](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Weights & Biases](https://docs.wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

---

## Next Steps

1. **Exercise 03: Model Deployment** - Deploy models to production
2. Optimize training performance further
3. Implement advanced distributed strategies (FSDP, DeepSpeed)
4. Build custom training frameworks
5. Learn model compression and optimization

---

**Master production ML training pipelines! 🚀**
