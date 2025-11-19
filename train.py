import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import json
from datetime import datetime

from model import chessbot_model
from dataloader import dataloader_instance
from train_utils import EMA, loss_fn
from tokenizer import tokenize, SEQUENCE_LENGTH
from configs import train_config, parent_config

dtype = parent_config.dtype
device = parent_config.device

# Hyperparameters
learning_rate = train_config.learning_rate
num_epochs = train_config.num_epochs
max_grad_norm = train_config.max_grad_norm

# Setup directories
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Initialize model, optimizer, and EMA
train_loader = dataloader_instance
model = chessbot_model().to(device)
criterion = loss_fn
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize EMA for better generalization
ema = EMA(model, decay=0.999, device=device)

# Training metadata
training_stats = {
    "start_time": datetime.now().isoformat(),
    "epochs": [],
    "best_loss": float('inf'),
}

print(f"Training on device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Starting training for {num_epochs} epochs...")
print("-" * 80)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs, labels)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Update EMA after optimizer step
        ema.update()

        # Accumulate loss
        running_loss += loss.item()
        num_batches += 1

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f}")

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)

    # Log epoch stats
    epoch_stats = {
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "learning_rate": learning_rate,
    }
    training_stats["epochs"].append(epoch_stats)

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  Average Loss: {epoch_loss:.4f}")

    # Save checkpoint
    is_best = epoch_loss < training_stats["best_loss"]
    if is_best:
        training_stats["best_loss"] = epoch_loss

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.ema_values,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'training_stats': training_stats,
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)
    print(f"  Checkpoint saved: {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        print(f"  New best model saved: {best_path} (Loss: {epoch_loss:.4f})")

    print("-" * 80)

# Save final model with EMA weights
print("\nTraining completed!")
print(f"Best loss achieved: {training_stats['best_loss']:.4f}")

# Save final model with EMA weights
ema.apply_ema()
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'training_stats': training_stats,
}
final_path = checkpoint_dir / "final_model_ema.pt"
torch.save(final_checkpoint, final_path)
print(f"Final model with EMA weights saved: {final_path}")

# Save training stats to JSON
stats_path = checkpoint_dir / "training_stats.json"
with open(stats_path, 'w') as f:
    json.dump(training_stats, f, indent=2)
print(f"Training statistics saved: {stats_path}")
