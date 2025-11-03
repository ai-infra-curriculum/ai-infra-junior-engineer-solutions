#!/usr/bin/env python3
"""
ML Training Simulator with Signal Handling

This script simulates a machine learning training process with:
- Configurable epochs and checkpoint intervals
- Graceful shutdown on SIGTERM/SIGINT
- Checkpoint saving at intervals and on shutdown
- Progress tracking and metrics logging
"""

import time
import sys
import signal
import json
import os
from datetime import datetime
import argparse


class TrainingSimulator:
    """Simulates ML training process with proper signal handling"""

    def __init__(self, epochs=100, checkpoint_interval=10):
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.current_epoch = 0
        self.running = True
        self.start_time = None

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)

        print(f"[{self.timestamp()}] Training Simulator initialized")
        print(f"PID: {os.getpid()}")
        print(f"Epochs: {self.epochs}")
        print(f"Checkpoint interval: {self.checkpoint_interval}")
        print(f"Signal handlers registered (SIGTERM, SIGINT)")
        print("=" * 60)

    def timestamp(self):
        """Return current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def graceful_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")

        print(f"\n[{self.timestamp()}] Received {signal_name}")
        print(f"[{self.timestamp()}] Initiating graceful shutdown...")
        print(f"[{self.timestamp()}] Saving checkpoint at epoch {self.current_epoch}...")

        self.running = False
        self.save_checkpoint(reason="shutdown")

        print(f"[{self.timestamp()}] Checkpoint saved successfully")
        print(f"[{self.timestamp()}] Cleanup complete")
        print(f"[{self.timestamp()}] Shutdown complete")

        sys.exit(0)

    def save_checkpoint(self, reason="interval"):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'timestamp': self.timestamp(),
            'status': reason,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'progress_percent': (self.current_epoch / self.epochs) * 100,
            'metrics': {
                'loss': 1.0 / max(self.current_epoch, 1),
                'accuracy': 1 - (1.0 / max(self.current_epoch, 1))
            }
        }

        filename = f'checkpoint_epoch_{self.current_epoch:04d}.json'
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"[{self.timestamp()}] Checkpoint saved: {filename}")

    def calculate_metrics(self, epoch):
        """Calculate simulated training metrics"""
        # Simulated loss: decreases with epochs
        loss = 1.0 / max(epoch, 1)

        # Simulated accuracy: increases with epochs
        accuracy = 1 - (1.0 / max(epoch, 1))

        # Add some variation
        import random
        loss += random.uniform(-0.01, 0.01)
        accuracy += random.uniform(-0.01, 0.01)

        return loss, accuracy

    def train(self):
        """Run training simulation"""
        self.start_time = time.time()

        print(f"[{self.timestamp()}] Starting training for {self.epochs} epochs")
        print("=" * 60)

        for epoch in range(1, self.epochs + 1):
            if not self.running:
                break

            self.current_epoch = epoch

            # Simulate training epoch
            loss, accuracy = self.calculate_metrics(epoch)

            # Progress bar
            progress = int((epoch / self.epochs) * 50)
            bar = "#" * progress + "-" * (50 - progress)
            percent = (epoch / self.epochs) * 100

            # Print progress
            print(f"[{self.timestamp()}] Epoch {epoch:4d}/{self.epochs} "
                  f"[{bar}] {percent:5.1f}% "
                  f"- Loss: {loss:.4f} - Acc: {accuracy:.4f}")

            # Simulate epoch time
            time.sleep(1)

            # Periodic checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(reason="interval")

        # Training complete
        if self.running:
            elapsed = time.time() - self.start_time

            print("=" * 60)
            print(f"[{self.timestamp()}] Training complete!")
            print(f"Total epochs: {self.current_epoch}")
            print(f"Elapsed time: {elapsed:.2f} seconds")
            print(f"Average time per epoch: {elapsed/self.current_epoch:.2f} seconds")

            # Save final model
            final_model = {
                'epochs': self.epochs,
                'final_loss': 1.0 / self.epochs,
                'final_accuracy': 1 - (1.0 / self.epochs),
                'training_time': elapsed,
                'timestamp': self.timestamp(),
                'status': 'completed'
            }

            with open('final_model.json', 'w') as f:
                json.dump(final_model, f, indent=2)

            print(f"[{self.timestamp()}] Final model saved: final_model.json")
            print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ML Training Simulator with Signal Handling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.epochs < 1:
        print("Error: epochs must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.checkpoint_interval < 1:
        print("Error: checkpoint-interval must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Create and run trainer
    try:
        trainer = TrainingSimulator(
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval
        )
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
