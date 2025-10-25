#!/usr/bin/env python3
"""
generate_report.py - Training report generator with visualizations

Description:
    Generate comprehensive training reports including metrics plots,
    model analysis, performance statistics, and recommendations.

Usage:
    python generate_report.py [OPTIONS]

Options:
    --checkpoint FILE      Model checkpoint to analyze
    --metrics FILE         Metrics JSON file
    --log-dir DIR          Training log directory
    --output DIR           Output directory for report (default: reports)
    --format FORMAT        Report format: html, pdf, markdown (default: html)
    --include-plots        Include visualization plots
    --verbose              Verbose output
    --help                 Display this help message
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import io

import torch
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plots will be skipped")


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


class ReportGenerator:
    """Generate training reports"""

    def __init__(self, checkpoint_path: Optional[str] = None,
                 metrics_path: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 output_dir: str = 'reports'):
        """
        Initialize report generator

        Args:
            checkpoint_path: Path to model checkpoint
            metrics_path: Path to metrics JSON file
            log_dir: Training log directory
            output_dir: Output directory for reports
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.checkpoint = None
        self.metrics = None
        self.config = None

    def load_data(self):
        """Load checkpoint and metrics"""
        # Load checkpoint
        if self.checkpoint_path and self.checkpoint_path.exists():
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.config = self.checkpoint.get('config', {})
        else:
            print(f"{Colors.YELLOW}Warning: Checkpoint not found{Colors.RESET}")

        # Load metrics
        if self.metrics_path and self.metrics_path.exists():
            print(f"Loading metrics: {self.metrics_path}")
            with open(self.metrics_path) as f:
                self.metrics = json.load(f)
        else:
            print(f"{Colors.YELLOW}Warning: Metrics file not found{Colors.RESET}")

    def analyze_model(self) -> Dict:
        """Analyze model from checkpoint"""
        if not self.checkpoint:
            return {}

        analysis = {}

        # Model size
        if 'model_state_dict' in self.checkpoint:
            model_state = self.checkpoint['model_state_dict']

            # Count parameters
            total_params = 0
            trainable_params = 0

            for name, param in model_state.items():
                params = param.numel()
                total_params += params
                trainable_params += params  # Assume all trainable

            analysis['total_parameters'] = total_params
            analysis['trainable_parameters'] = trainable_params
            analysis['model_size_mb'] = sum(
                p.numel() * p.element_size() for p in model_state.values()
            ) / (1024 * 1024)

        # Training info
        if 'epoch' in self.checkpoint:
            analysis['trained_epochs'] = self.checkpoint['epoch'] + 1

        if 'metrics' in self.checkpoint:
            analysis['final_metrics'] = self.checkpoint['metrics']

        return analysis

    def analyze_metrics(self) -> Dict:
        """Analyze training metrics"""
        if not self.metrics:
            return {}

        analysis = {}

        # Training progress
        if 'train_loss' in self.metrics:
            train_losses = self.metrics['train_loss']
            analysis['train_loss'] = {
                'initial': train_losses[0] if train_losses else 0,
                'final': train_losses[-1] if train_losses else 0,
                'min': min(train_losses) if train_losses else 0,
                'max': max(train_losses) if train_losses else 0,
                'improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0
            }

        if 'val_loss' in self.metrics:
            val_losses = self.metrics['val_loss']
            analysis['val_loss'] = {
                'initial': val_losses[0] if val_losses else 0,
                'final': val_losses[-1] if val_losses else 0,
                'min': min(val_losses) if val_losses else 0,
                'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else 0
            }

        if 'val_acc' in self.metrics:
            val_accs = self.metrics['val_acc']
            analysis['val_acc'] = {
                'initial': val_accs[0] if val_accs else 0,
                'final': val_accs[-1] if val_accs else 0,
                'max': max(val_accs) if val_accs else 0,
                'best_epoch': val_accs.index(max(val_accs)) + 1 if val_accs else 0
            }

        # Convergence analysis
        if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > 10:
            recent_losses = self.metrics['val_loss'][-10:]
            loss_variance = np.var(recent_losses)
            analysis['convergence'] = {
                'converged': loss_variance < 0.001,
                'loss_variance': loss_variance
            }

        return analysis

    def generate_plots(self):
        """Generate visualization plots"""
        if not PLOTTING_AVAILABLE or not self.metrics:
            print(f"{Colors.YELLOW}Skipping plots (matplotlib not available or no metrics){Colors.RESET}")
            return

        print("Generating plots...")

        # Set style
        sns.set_style("whitegrid")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Report', fontsize=16, fontweight='bold')

        # Plot 1: Training and Validation Loss
        if 'train_loss' in self.metrics and 'val_loss' in self.metrics:
            ax = axes[0, 0]
            epochs = range(1, len(self.metrics['train_loss']) + 1)

            ax.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Training and Validation Accuracy
        if 'train_acc' in self.metrics and 'val_acc' in self.metrics:
            ax = axes[0, 1]
            epochs = range(1, len(self.metrics['train_acc']) + 1)

            ax.plot(epochs, self.metrics['train_acc'], 'b-', label='Train Acc', linewidth=2)
            ax.plot(epochs, self.metrics['val_acc'], 'r-', label='Val Acc', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Learning Rate Schedule
        if 'learning_rate' in self.metrics:
            ax = axes[1, 0]
            epochs = range(1, len(self.metrics['learning_rate']) + 1)

            ax.plot(epochs, self.metrics['learning_rate'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Plot 4: Overfitting Analysis
        if 'train_loss' in self.metrics and 'val_loss' in self.metrics:
            ax = axes[1, 1]
            epochs = range(1, len(self.metrics['train_loss']) + 1)

            gap = [val - train for train, val in zip(
                self.metrics['train_loss'],
                self.metrics['val_loss']
            )]

            ax.plot(epochs, gap, 'm-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val Loss - Train Loss')
            ax.set_title('Overfitting Analysis (Gap)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"{Colors.GREEN}✓ Plots saved to: {plot_path}{Colors.RESET}")

    def generate_recommendations(self, model_analysis: Dict, metrics_analysis: Dict) -> List[str]:
        """Generate training recommendations"""
        recommendations = []

        # Check overfitting
        if 'val_loss' in metrics_analysis and 'train_loss' in metrics_analysis:
            val_loss = metrics_analysis['val_loss']['final']
            train_loss = metrics_analysis['train_loss']['final']

            if val_loss > train_loss * 1.5:
                recommendations.append(
                    "⚠️  Model appears to be overfitting. Consider:\n"
                    "   - Adding regularization (dropout, weight decay)\n"
                    "   - Reducing model complexity\n"
                    "   - Collecting more training data"
                )

        # Check underfitting
        if 'train_loss' in metrics_analysis:
            if metrics_analysis['train_loss']['final'] > 1.0:
                recommendations.append(
                    "⚠️  Model may be underfitting. Consider:\n"
                    "   - Increasing model capacity\n"
                    "   - Training for more epochs\n"
                    "   - Adjusting learning rate"
                )

        # Check convergence
        if 'convergence' in metrics_analysis:
            if not metrics_analysis['convergence']['converged']:
                recommendations.append(
                    "⚠️  Model has not converged. Consider:\n"
                    "   - Training for more epochs\n"
                    "   - Adjusting learning rate schedule\n"
                    "   - Checking for training instabilities"
                )

        # Check best epoch vs final epoch
        if 'val_loss' in metrics_analysis:
            best_epoch = metrics_analysis['val_loss']['best_epoch']
            total_epochs = model_analysis.get('trained_epochs', 0)

            if total_epochs - best_epoch > 10:
                recommendations.append(
                    f"ℹ️  Best validation loss occurred at epoch {best_epoch}, "
                    f"but training continued to epoch {total_epochs}. "
                    "Consider using early stopping."
                )

        # Model size recommendations
        if 'model_size_mb' in model_analysis:
            size_mb = model_analysis['model_size_mb']

            if size_mb > 1000:
                recommendations.append(
                    f"ℹ️  Large model size ({size_mb:.1f} MB). Consider:\n"
                    "   - Model compression techniques\n"
                    "   - Quantization for deployment\n"
                    "   - Knowledge distillation"
                )

        if not recommendations:
            recommendations.append("✓ No major issues detected. Training looks healthy!")

        return recommendations

    def generate_markdown_report(self, model_analysis: Dict, metrics_analysis: Dict) -> str:
        """Generate markdown report"""
        report = io.StringIO()

        # Header
        report.write("# Training Report\n\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.write("---\n\n")

        # Configuration
        if self.config:
            report.write("## Configuration\n\n")
            report.write(f"- **Experiment**: {self.config.get('experiment', {}).get('name', 'N/A')}\n")
            report.write(f"- **Model**: {self.config.get('model', {}).get('architecture', 'N/A')}\n")
            report.write(f"- **Batch Size**: {self.config.get('training', {}).get('batch_size', 'N/A')}\n")
            report.write(f"- **Learning Rate**: {self.config.get('training', {}).get('learning_rate', 'N/A')}\n")
            report.write(f"- **Epochs**: {model_analysis.get('trained_epochs', 'N/A')}\n\n")

        # Model Analysis
        report.write("## Model Analysis\n\n")
        report.write(f"- **Total Parameters**: {model_analysis.get('total_parameters', 'N/A'):,}\n")
        report.write(f"- **Trainable Parameters**: {model_analysis.get('trainable_parameters', 'N/A'):,}\n")
        report.write(f"- **Model Size**: {model_analysis.get('model_size_mb', 0):.2f} MB\n\n")

        # Training Metrics
        report.write("## Training Metrics\n\n")

        if 'train_loss' in metrics_analysis:
            report.write("### Training Loss\n\n")
            report.write(f"- Initial: {metrics_analysis['train_loss']['initial']:.4f}\n")
            report.write(f"- Final: {metrics_analysis['train_loss']['final']:.4f}\n")
            report.write(f"- Min: {metrics_analysis['train_loss']['min']:.4f}\n")
            report.write(f"- Improvement: {metrics_analysis['train_loss']['improvement']:.4f}\n\n")

        if 'val_loss' in metrics_analysis:
            report.write("### Validation Loss\n\n")
            report.write(f"- Initial: {metrics_analysis['val_loss']['initial']:.4f}\n")
            report.write(f"- Final: {metrics_analysis['val_loss']['final']:.4f}\n")
            report.write(f"- Min: {metrics_analysis['val_loss']['min']:.4f}\n")
            report.write(f"- Best Epoch: {metrics_analysis['val_loss']['best_epoch']}\n\n")

        if 'val_acc' in metrics_analysis:
            report.write("### Validation Accuracy\n\n")
            report.write(f"- Initial: {metrics_analysis['val_acc']['initial']:.2f}%\n")
            report.write(f"- Final: {metrics_analysis['val_acc']['final']:.2f}%\n")
            report.write(f"- Max: {metrics_analysis['val_acc']['max']:.2f}%\n")
            report.write(f"- Best Epoch: {metrics_analysis['val_acc']['best_epoch']}\n\n")

        # Convergence
        if 'convergence' in metrics_analysis:
            report.write("### Convergence Analysis\n\n")
            converged = metrics_analysis['convergence']['converged']
            report.write(f"- Converged: {'✓ Yes' if converged else '✗ No'}\n")
            report.write(f"- Loss Variance (last 10 epochs): {metrics_analysis['convergence']['loss_variance']:.6f}\n\n")

        # Recommendations
        recommendations = self.generate_recommendations(model_analysis, metrics_analysis)
        report.write("## Recommendations\n\n")
        for rec in recommendations:
            report.write(f"{rec}\n\n")

        # Plots
        if PLOTTING_AVAILABLE:
            report.write("## Visualizations\n\n")
            report.write("![Training Plots](training_plots.png)\n\n")

        return report.getvalue()

    def generate_report(self, format: str = 'markdown'):
        """Generate complete report"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Generating Training Report{Colors.RESET}")
        print("=" * 80)

        # Load data
        self.load_data()

        # Analyze model
        print("\nAnalyzing model...")
        model_analysis = self.analyze_model()

        # Analyze metrics
        print("Analyzing metrics...")
        metrics_analysis = self.analyze_metrics()

        # Generate plots
        if PLOTTING_AVAILABLE:
            self.generate_plots()

        # Generate report
        print(f"\nGenerating {format} report...")

        if format == 'markdown':
            report_content = self.generate_markdown_report(model_analysis, metrics_analysis)
            report_path = self.output_dir / 'training_report.md'

            with open(report_path, 'w') as f:
                f.write(report_content)

            print(f"{Colors.GREEN}✓ Report saved to: {report_path}{Colors.RESET}")

        # Print summary to console
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")

        if model_analysis:
            print(f"  Parameters: {model_analysis.get('total_parameters', 0):,}")
            print(f"  Model Size: {model_analysis.get('model_size_mb', 0):.2f} MB")

        if 'val_loss' in metrics_analysis:
            print(f"  Best Val Loss: {metrics_analysis['val_loss']['min']:.4f} "
                  f"(epoch {metrics_analysis['val_loss']['best_epoch']})")

        if 'val_acc' in metrics_analysis:
            print(f"  Best Val Accuracy: {metrics_analysis['val_acc']['max']:.2f}% "
                  f"(epoch {metrics_analysis['val_acc']['best_epoch']})")

        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Report generation complete!{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description='Training report generator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint to analyze')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Metrics JSON file')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Training log directory')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for report')
    parser.add_argument('--format', type=str, default='markdown',
                       choices=['markdown', 'html', 'pdf'],
                       help='Report format (default: markdown)')
    parser.add_argument('--include-plots', action='store_true', default=True,
                       help='Include visualization plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine metrics path
    if args.metrics is None and args.log_dir:
        args.metrics = str(Path(args.log_dir) / 'metrics.json')

    # Validate inputs
    if args.checkpoint is None and args.metrics is None:
        print(f"{Colors.RED}Error: Must provide either --checkpoint or --metrics{Colors.RESET}")
        sys.exit(1)

    # Create report generator
    generator = ReportGenerator(
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        log_dir=args.log_dir,
        output_dir=args.output
    )

    # Generate report
    generator.generate_report(format=args.format)


if __name__ == '__main__':
    main()
