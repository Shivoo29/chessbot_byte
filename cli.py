#!/usr/bin/env python3
"""
ChessBot Byte - Command Line Interface

A comprehensive CLI for training, evaluating, and using the chess bot model.
"""

import argparse
import sys
from pathlib import Path


def train_command(args):
    """Run training."""
    print("Starting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    # Update configs
    from configs import train_config, data_config
    if args.epochs:
        train_config.num_epochs = args.epochs
    if args.lr:
        train_config.learning_rate = args.lr
    if args.data_files:
        data_config.number_of_files = args.data_files
    if args.batch_size:
        data_config.batch_size = args.batch_size

    # Import and run training
    import train
    print("\nTraining completed!")


def evaluate_command(args):
    """Run evaluation."""
    from evaluate import main as eval_main
    sys.argv = [
        'evaluate.py',
        '--checkpoint', args.checkpoint,
        '--data-files', str(args.data_files),
        '--batch-size', str(args.batch_size),
        '--output', args.output,
    ]
    eval_main()


def inference_command(args):
    """Run inference."""
    from inference import main as inference_main

    argv = [
        'inference.py',
        '--checkpoint', args.checkpoint,
        '--fen', args.fen,
        '--top-k', str(args.top_k),
    ]

    if args.move:
        argv.extend(['--move', args.move])
    if args.interactive:
        argv.append('--interactive')

    sys.argv = argv
    inference_main()


def setup_command(args):
    """Setup the project."""
    print("Setting up ChessBot Byte...")
    print("-" * 80)

    # Create necessary directories
    directories = ['data', 'data/train', 'data/val', 'checkpoints', 'logs']
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True, parents=True)
        print(f"✓ Created directory: {dir_name}")

    # Check if data exists
    from configs import data_config
    data_path = Path(data_config.filename.replace('@3', '00000-of-02148'))
    if not data_path.exists():
        print("\n⚠ Warning: Training data not found!")
        print(f"Expected data at: {data_path}")
        print("\nTo download data, run:")
        print("  bash download_data.sh")
        print("\nOr set custom data path with environment variable:")
        print("  export CHESSBOT_DATA_DIR=/path/to/your/data")

    print("\n" + "-" * 80)
    print("Setup complete!")
    print("\nNext steps:")
    print("  1. Download training data (if not already done)")
    print("  2. Run training: python cli.py train")
    print("  3. Evaluate model: python cli.py evaluate")
    print("  4. Use inference: python cli.py infer --interactive")


def info_command(args):
    """Display project information."""
    from configs import parent_config, model_config, train_config, data_config
    import torch

    print("=" * 80)
    print("ChessBot Byte - Project Information")
    print("=" * 80)

    print("\n📋 Model Configuration:")
    print(f"  Architecture:       Transformer with Mixture of Experts (MoE)")
    print(f"  Model dimension:    {model_config.d_model}")
    print(f"  Feed-forward dim:   {model_config.dim_feedforward}")
    print(f"  Attention heads:    {model_config.nhead}")
    print(f"  Encoder layers:     {model_config.encoder_layers}")
    print(f"  Number of experts:  {model_config.num_experts}")
    print(f"  Experts per token:  {model_config.num_experts_per_tok}")

    print("\n🎯 Training Configuration:")
    print(f"  Epochs:             {train_config.num_epochs}")
    print(f"  Learning rate:      {train_config.learning_rate}")
    print(f"  Max grad norm:      {train_config.max_grad_norm}")
    print(f"  Batch size:         {data_config.batch_size}")

    print("\n💾 Data Configuration:")
    print(f"  Data files:         {data_config.number_of_files}")
    print(f"  Return buckets:     {parent_config.num_return_buckets}")
    print(f"  Sequence length:    {model_config.SEQUENCE_LENGTH}")

    print("\n🖥  Hardware:")
    print(f"  Device:             {parent_config.device}")
    print(f"  CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:        {torch.cuda.get_device_name(0)}")

    # Check for checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"\n📦 Available Checkpoints:")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name:30s} ({size_mb:.1f} MB)")
        else:
            print(f"\n📦 No checkpoints found")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='ChessBot Byte - A transformer-based chess engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup the project
  python cli.py setup

  # Train the model
  python cli.py train --epochs 10 --lr 1e-4

  # Evaluate the model
  python cli.py evaluate

  # Interactive inference
  python cli.py infer --interactive

  # Evaluate a specific move
  python cli.py infer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --move e2e4

  # Show project info
  python cli.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup the project')
    setup_parser.set_defaults(func=setup_command)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=None,
                              help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=None,
                              help='Learning rate')
    train_parser.add_argument('--data-files', type=int, default=None,
                              help='Number of data files to use')
    train_parser.add_argument('--batch-size', type=int, default=None,
                              help='Batch size')
    train_parser.set_defaults(func=train_command)

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str,
                             default='checkpoints/best_checkpoint.pt',
                             help='Path to model checkpoint')
    eval_parser.add_argument('--data-files', type=int, default=3,
                             help='Number of data files to use')
    eval_parser.add_argument('--batch-size', type=int, default=256,
                             help='Batch size')
    eval_parser.add_argument('--output', type=str,
                             default='evaluation_results.json',
                             help='Output file for results')
    eval_parser.set_defaults(func=evaluate_command)

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str,
                              default='checkpoints/best_checkpoint.pt',
                              help='Path to model checkpoint')
    infer_parser.add_argument('--fen', type=str,
                              default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                              help='Position in FEN notation')
    infer_parser.add_argument('--move', type=str, default=None,
                              help='Specific move to evaluate (UCI format)')
    infer_parser.add_argument('--top-k', type=int, default=10,
                              help='Number of top moves to show')
    infer_parser.add_argument('--interactive', action='store_true',
                              help='Start interactive mode')
    infer_parser.set_defaults(func=inference_command)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show project information')
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the command
    args.func(args)


if __name__ == '__main__':
    main()
