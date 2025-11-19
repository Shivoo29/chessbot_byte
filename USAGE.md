# ChessBot Byte - Usage Guide

Complete guide for using ChessBot Byte from setup to inference.

## Table of Contents

1. [First-Time Setup](#first-time-setup)
2. [Training Your Model](#training-your-model)
3. [Evaluating Performance](#evaluating-performance)
4. [Using the Model](#using-the-model)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## First-Time Setup

### Step 1: Verify Installation

Check that all dependencies are installed:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import chess; print('python-chess: OK')"
```

### Step 2: Run Setup

Initialize the project structure:

```bash
python cli.py setup
```

This creates:
- `data/` - For training data
- `checkpoints/` - For model checkpoints
- `logs/` - For training logs

### Step 3: Download Training Data

Run the download script:

```bash
bash download_data.sh
```

Or set a custom data path:

```bash
export CHESSBOT_DATA_DIR=/path/to/chess/data
```

### Step 4: Verify Setup

Check project information:

```bash
python cli.py info
```

## Training Your Model

### Quick Start Training

Train with default settings (good for testing):

```bash
python cli.py train
```

This uses:
- 10 epochs
- Learning rate: 1e-4
- 3 data files
- Batch size: 256

### Recommended Training

For better results, train with more data:

```bash
python cli.py train --epochs 20 --data-files 100
```

### Full Training

Use all available data for best performance:

```bash
python cli.py train --epochs 50 --data-files 2148 --batch-size 512
```

**Note**: This requires significant compute resources and time.

### Custom Training

Fine-tune all parameters:

```bash
python cli.py train \
  --epochs 30 \
  --lr 5e-5 \
  --data-files 500 \
  --batch-size 256
```

### Monitoring Training

During training, you'll see:
- Batch-level progress every 10 batches
- Epoch-level summaries
- Automatic checkpoint saving
- Best model tracking

Example output:
```
Epoch [1/10] Batch [10/50] Loss: 2.3456
Epoch [1/10] Batch [20/50] Loss: 2.2145
...
Epoch [1/10] Summary:
  Average Loss: 2.1234
  Checkpoint saved: checkpoints/latest_checkpoint.pt
  New best model saved: checkpoints/best_checkpoint.pt
```

### Training Outputs

Training creates:
- `checkpoints/latest_checkpoint.pt` - Most recent model
- `checkpoints/best_checkpoint.pt` - Best performing model
- `checkpoints/final_model_ema.pt` - Final model with EMA weights
- `checkpoints/training_stats.json` - Training statistics

## Evaluating Performance

### Quick Evaluation

Evaluate the best model:

```bash
python cli.py evaluate
```

### Custom Evaluation

Specify checkpoint and data:

```bash
python cli.py evaluate \
  --checkpoint checkpoints/final_model_ema.pt \
  --data-files 10 \
  --batch-size 128
```

### Understanding Metrics

The evaluation provides:

- **Loss**: Lower is better (typical range: 1.5 - 3.0)
- **Accuracy**: Exact bucket prediction (typical: 10-20%)
- **Top-5 Accuracy**: Prediction in top 5 buckets (typical: 30-50%)
- **Top-10 Accuracy**: Prediction in top 10 buckets (typical: 45-65%)

Example output:
```
Evaluation Results:
--------------------------------------------------------------------------------
Loss:             1.8234
Accuracy:         15.32%
Top-5 Accuracy:   42.18%
Top-10 Accuracy:  58.67%
Samples:          76,800
```

## Using the Model

### Interactive Mode (Recommended)

Start interactive chess analysis:

```bash
python cli.py infer --interactive
```

#### Interactive Commands

```
> fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Position set to: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

> eval
Evaluating position...

Top 10 moves:
 1. e2e4     (e2e4  ) Win%: 51.2%  Conf: 23.4%
 2. d2d4     (d2d4  ) Win%: 50.8%  Conf: 21.1%
 3. Ng1f3    (g1f3  ) Win%: 50.5%  Conf: 18.7%
 ...

> best
Best move: e2e4 (e2e4)
Win probability: 51.2%
Confidence: 23.4%

> move e2e4
Move: e2e4
Win probability: 51.2%
Confidence: 23.4%

> quit
```

### Single Position Analysis

Analyze a specific position:

```bash
python cli.py infer \
  --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" \
  --top-k 5
```

### Specific Move Evaluation

Evaluate a single move:

```bash
python cli.py infer \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --move e2e4
```

### Using Different Checkpoints

Compare different models:

```bash
# Use best checkpoint
python cli.py infer --checkpoint checkpoints/best_checkpoint.pt --interactive

# Use EMA checkpoint
python cli.py infer --checkpoint checkpoints/final_model_ema.pt --interactive

# Use latest checkpoint
python cli.py infer --checkpoint checkpoints/latest_checkpoint.pt --interactive
```

## Advanced Usage

### Custom Data Path

Set custom data directory:

```bash
export CHESSBOT_DATA_DIR=/mnt/chess_data
python cli.py train
```

### GPU Training

The model automatically uses GPU if available. To force CPU:

Edit `configs.py`:
```python
class parent_config:
    device = 'cpu'  # Force CPU
```

### Batch Processing

Process multiple positions from a file:

```python
from inference import ChessBotInference

bot = ChessBotInference('checkpoints/best_checkpoint.pt')

positions = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]

for fen in positions:
    best = bot.get_best_move(fen)
    print(f"Position: {fen}")
    print(f"Best move: {best['move_san']} ({best['win_probability']:.1%})")
```

### Programmatic Usage

Use the model in your Python code:

```python
from inference import ChessBotInference

# Initialize
bot = ChessBotInference('checkpoints/best_checkpoint.pt')

# Evaluate position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
top_moves = bot.evaluate_position(fen, top_k=5)

for i, move in enumerate(top_moves, 1):
    print(f"{i}. {move['move_san']}: {move['win_probability']:.1%}")

# Get best move
best = bot.get_best_move(fen)
print(f"Best: {best['move_san']}")

# Evaluate specific move
result = bot.predict_move_value(fen, "e2e4")
print(f"e2e4 win%: {result['win_probability']:.1%}")
```

## Troubleshooting

### Data Not Found

**Error**: `Training data not found!`

**Solution**:
```bash
# Download data
bash download_data.sh

# Or set custom path
export CHESSBOT_DATA_DIR=/path/to/data
```

### Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python cli.py train --batch-size 128

# Or use CPU
# Edit configs.py and set device = 'cpu'
```

### Slow Training

**Issue**: Training is very slow

**Solution**:
1. Use GPU if available
2. Reduce data files for testing: `--data-files 3`
3. Increase batch size if GPU has memory: `--batch-size 512`

### Invalid Move

**Error**: `Invalid move: xyz`

**Solution**:
- Ensure moves are in UCI format (e.g., `e2e4`, not `e4`)
- Check move is legal in the position
- For promotions, include piece: `e7e8q`

### Model Performance Issues

**Issue**: Model predictions seem random

**Solution**:
1. Train for more epochs: `--epochs 50`
2. Use more training data: `--data-files 1000`
3. Check that training loss is decreasing
4. Use EMA checkpoint for inference

### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python cli.py info
```

## Best Practices

1. **Start Small**: Test with `--data-files 3` before full training
2. **Use EMA**: The EMA checkpoint usually performs better
3. **Monitor Loss**: Training loss should steadily decrease
4. **Save Checkpoints**: Keep multiple checkpoints for comparison
5. **Validate Often**: Run evaluation periodically during training
6. **Use GPU**: Significantly faster than CPU training
7. **Backup Models**: Copy important checkpoints to safe storage

## Performance Tips

### Training
- Use GPU for 10-20x speedup
- Increase batch size to maximize GPU utilization
- Use more data files for better model quality
- Train for at least 20 epochs for good results

### Inference
- Use EMA checkpoint for best quality
- Interactive mode is best for exploration
- Batch multiple positions for efficiency
- Cache model in memory for repeated queries

## Next Steps

After completing this guide:

1. Train a model with your desired settings
2. Evaluate to check performance
3. Use interactive mode to explore capabilities
4. Integrate into your chess application
5. Experiment with model configurations
6. Contribute improvements back to the project

## Support

- Check the main README.md for project overview
- Review code comments for implementation details
- Open GitHub issues for bugs or questions
- Share your results and improvements!

---

Happy chess bot training! 🤖♟️
