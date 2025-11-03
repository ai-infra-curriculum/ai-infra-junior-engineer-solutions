#!/usr/bin/env python3
"""
Dictionary operations for ML configuration management.

Demonstrates dictionary usage for model configs and metadata.
"""

from typing import Dict, Any


def main():
    """Demonstrate dictionary operations for ML configs."""
    print("=" * 60)
    print("Dictionary Operations for ML Configuration")
    print("=" * 60)
    print()

    # Model configuration
    model_config = {
        "name": "ResNet50",
        "version": "1.0.0",
        "input_shape": (224, 224, 3),
        "num_classes": 1000,
        "pretrained": True,
        "freeze_layers": 10
    }

    # Access values
    print(f"Model: {model_config['name']}")
    print(f"Version: {model_config['version']}")
    print()

    # Safe access with get()
    optimizer = model_config.get("optimizer", "adam")  # Default to adam
    print(f"Optimizer (with default): {optimizer}")
    print()

    # Update config
    model_config["learning_rate"] = 0.001
    model_config.update({
        "optimizer": "adam",
        "weight_decay": 0.0001
    })

    print("Updated configuration:")
    print(f"  Learning rate: {model_config['learning_rate']}")
    print(f"  Optimizer: {model_config['optimizer']}")
    print(f"  Weight decay: {model_config['weight_decay']}")
    print()

    # Check key existence
    if "dropout" not in model_config:
        model_config["dropout"] = 0.5
        print("Added dropout configuration: 0.5")
    print()

    # Iterate over config
    print("Full configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print()

    # Get all keys and values
    config_keys = list(model_config.keys())
    config_values = list(model_config.values())
    print(f"Configuration keys ({len(config_keys)}): {config_keys[:5]}...")
    print()

    # Remove keys
    removed_value = model_config.pop("freeze_layers", None)
    print(f"Removed 'freeze_layers': {removed_value}")
    print(f"Keys after removal: {len(model_config)}")
    print()

    # Copy and clear
    temp_config = model_config.copy()
    temp_config.clear()
    print(f"Cleared copy: {len(temp_config)} keys")
    print(f"Original intact: {len(model_config)} keys")
    print()

    # Nested dictionaries
    print("=" * 60)
    print("Nested Configuration")
    print("=" * 60)
    print()

    training_config = {
        "model": {
            "architecture": "resnet50",
            "pretrained": True,
            "num_classes": 1000
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": {
                "type": "adam",
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
        },
        "data": {
            "train_path": "/data/train",
            "val_path": "/data/val",
            "augmentation": True
        }
    }

    print("Nested configuration structure:")
    print(f"  Model architecture: {training_config['model']['architecture']}")
    print(f"  Batch size: {training_config['training']['batch_size']}")
    print(f"  Optimizer type: {training_config['training']['optimizer']['type']}")
    print(f"  Data augmentation: {training_config['data']['augmentation']}")
    print()

    # Merge dictionaries
    defaults = {"dropout": 0.5, "weight_decay": 0.0001, "lr_schedule": "cosine"}
    user_config = {"dropout": 0.3, "learning_rate": 0.01}

    final_config = {**defaults, **user_config}  # user_config overrides defaults
    print("Merged configuration (defaults + user):")
    for key, value in final_config.items():
        print(f"  {key}: {value}")

    print()
    print("✓ Dictionary operations demonstration complete")


if __name__ == "__main__":
    main()
