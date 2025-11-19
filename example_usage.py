#!/usr/bin/env python3
"""
Example usage of ChessBot Byte programmatically.

This script demonstrates how to use the chess bot in your own Python code.
"""

from pathlib import Path


def example_training():
    """Example: Train a model programmatically."""
    print("=" * 80)
    print("Example 1: Training a Model")
    print("=" * 80)

    # Import training modules
    from configs import train_config, data_config

    # Configure training
    train_config.num_epochs = 5
    train_config.learning_rate = 1e-4
    data_config.number_of_files = 1  # Use small dataset for demo

    print("\nConfiguration:")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Data files: {data_config.number_of_files}")

    print("\nTo run training, uncomment the import below:")
    print("  # import train")
    print("\nOr use CLI: python cli.py train --epochs 5 --data-files 1")
    print()


def example_inference():
    """Example: Use the model for inference."""
    print("=" * 80)
    print("Example 2: Using the Model for Inference")
    print("=" * 80)

    checkpoint_path = Path("checkpoints/best_checkpoint.pt")

    if not checkpoint_path.exists():
        print("\n⚠ No trained model found!")
        print("  Please train a model first: python cli.py train")
        print()
        return

    from inference import ChessBotInference

    # Initialize the bot
    print("\nInitializing chess bot...")
    bot = ChessBotInference(str(checkpoint_path))

    # Example position (starting position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    print(f"\nAnalyzing position: {fen}")

    # Get best move
    print("\nGetting best move...")
    best = bot.get_best_move(fen)

    if best:
        print(f"  Best move: {best['move_san']} ({best['move']})")
        print(f"  Win probability: {best['win_probability']:.1%}")
        print(f"  Confidence: {best['confidence']:.1%}")

    # Evaluate top moves
    print("\nEvaluating top 5 moves...")
    top_moves = bot.evaluate_position(fen, top_k=5)

    for i, move in enumerate(top_moves, 1):
        print(f"  {i}. {move['move_san']:8s} Win%: {move['win_probability']:.1%}  "
              f"Conf: {move['confidence']:.1%}")

    # Evaluate specific move
    print("\nEvaluating specific move (e2e4)...")
    result = bot.predict_move_value(fen, "e2e4")
    print(f"  Win probability: {result['win_probability']:.1%}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print()


def example_evaluation():
    """Example: Evaluate model performance."""
    print("=" * 80)
    print("Example 3: Evaluating Model Performance")
    print("=" * 80)

    checkpoint_path = Path("checkpoints/best_checkpoint.pt")

    if not checkpoint_path.exists():
        print("\n⚠ No trained model found!")
        print("  Please train a model first: python cli.py train")
        print()
        return

    print("\nTo evaluate the model, use:")
    print("  python cli.py evaluate")
    print("\nOr with custom settings:")
    print("  python cli.py evaluate --data-files 5 --batch-size 128")
    print()


def example_batch_analysis():
    """Example: Analyze multiple positions."""
    print("=" * 80)
    print("Example 4: Batch Position Analysis")
    print("=" * 80)

    checkpoint_path = Path("checkpoints/best_checkpoint.pt")

    if not checkpoint_path.exists():
        print("\n⚠ No trained model found!")
        print("  Please train a model first: python cli.py train")
        print()
        return

    from inference import ChessBotInference

    # Initialize bot
    bot = ChessBotInference(str(checkpoint_path))

    # Multiple positions to analyze
    positions = [
        ("Starting position",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("After 1.e4 e5",
         "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
    ]

    print("\nAnalyzing multiple positions...")
    print()

    for name, fen in positions:
        print(f"{name}:")
        best = bot.get_best_move(fen)
        if best:
            print(f"  Best: {best['move_san']} ({best['win_probability']:.1%})")
        print()


def example_custom_config():
    """Example: Use custom configuration."""
    print("=" * 80)
    print("Example 5: Custom Configuration")
    print("=" * 80)

    print("\nTo use custom configuration:")
    print("1. Edit configs.py")
    print("2. Modify parameters:")
    print()
    print("   # Model configuration")
    print("   model_config.d_model = 128")
    print("   model_config.nhead = 8")
    print("   model_config.encoder_layers = 6")
    print()
    print("   # Training configuration")
    print("   train_config.num_epochs = 20")
    print("   train_config.learning_rate = 5e-5")
    print()
    print("   # Data configuration")
    print("   data_config.batch_size = 512")
    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ChessBot Byte - Example Usage" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    examples = [
        ("Training", example_training),
        ("Inference", example_inference),
        ("Evaluation", example_evaluation),
        ("Batch Analysis", example_batch_analysis),
        ("Custom Config", example_custom_config),
    ]

    for i, (name, func) in enumerate(examples, 1):
        func()
        if i < len(examples):
            input("Press Enter to continue to next example...")
            print("\n" * 2)

    print("=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print()
    print("For more information:")
    print("  - README.md: Project overview")
    print("  - USAGE.md: Detailed usage guide")
    print("  - CONTRIBUTING.md: Contribution guidelines")
    print()


if __name__ == '__main__':
    main()
