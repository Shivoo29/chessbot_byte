"""Evaluation script for chess bot model."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import argparse

from model import chessbot_model
from dataloader import ChessDataset
from train_utils import loss_fn
from configs import data_config, parent_config
import utils


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset.

    Args:
        model: The chess bot model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs, labels)

            total_loss += loss.item()
            total_batches += 1

            # Get predictions (index of max log-probability)
            predictions = outputs[:, -1, :].argmax(dim=-1)
            targets = inputs[:, -1]

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    avg_loss = total_loss / total_batches
    accuracy = (all_predictions == all_targets).mean()

    # Top-k accuracy
    top5_correct = 0
    top10_correct = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get top-k predictions for the return bucket
            probs = outputs[:, -1, :]
            top5_preds = probs.topk(5, dim=-1)[1]
            top10_preds = probs.topk(10, dim=-1)[1]

            targets = inputs[:, -1].unsqueeze(-1)
            top5_correct += (top5_preds == targets.cpu()).any(dim=-1).sum().item()
            top10_correct += (top10_preds == targets.cpu()).any(dim=-1).sum().item()

    total_samples = len(dataloader.dataset)
    top5_accuracy = top5_correct / total_samples
    top10_accuracy = top10_correct / total_samples

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
        'num_samples': total_samples,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chess Bot Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-files', type=int, default=3,
                        help='Number of data files to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')

    args = parser.parse_args()

    device = parent_config.device
    print(f"Evaluation running on device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = chessbot_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load evaluation data
    print(f"Loading evaluation data (using {args.data_files} files)...")
    # Update config for evaluation
    data_config.number_of_files = args.data_files
    data_config.batch_size = args.batch_size

    eval_dataset = ChessDataset(data_config.filename)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"Evaluation dataset size: {len(eval_dataset)}")
    print(f"Number of batches: {len(eval_loader)}")
    print("-" * 80)

    # Evaluate
    print("Running evaluation...")
    metrics = evaluate_model(model, eval_loader, device)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"Loss:             {metrics['loss']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.2%}")
    print(f"Top-5 Accuracy:   {metrics['top5_accuracy']:.2%}")
    print(f"Top-10 Accuracy:  {metrics['top10_accuracy']:.2%}")
    print(f"Samples:          {metrics['num_samples']}")
    print("-" * 80)

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'metrics': metrics,
        'config': {
            'data_files': args.data_files,
            'batch_size': args.batch_size,
        }
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
