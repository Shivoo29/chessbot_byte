# ChessBot Byte 🤖♟️

A transformer-based chess engine using Mixture of Experts (MoE) architecture to predict move values from chess positions.

## 🌟 Features

- **Advanced Architecture**: Transformer encoder with Mixture of Experts (MoE) for efficient learning
- **Value Prediction**: Predicts win probabilities for chess positions and moves
- **Complete Training Pipeline**: Full training loop with EMA, checkpointing, and metrics tracking
- **Evaluation Tools**: Comprehensive evaluation scripts with chess-specific metrics
- **Interactive Inference**: Easy-to-use CLI for position evaluation and move suggestions
- **Production Ready**: Proper checkpointing, logging, and model management

## 🏗️ Architecture

- **Model**: Transformer encoder with MoE feed-forward networks
- **Input**: FEN string tokenization (77 tokens)
- **Output**: Return bucket predictions (128 buckets mapping to win probabilities)
- **Features**:
  - 4 encoder layers
  - 64 model dimensions
  - 4 attention heads
  - 8 experts with top-2 routing
  - Exponential Moving Average (EMA) for better generalization

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chessbot_byte.git
cd chessbot_byte
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup the project:
```bash
python cli.py setup
```

4. Download training data:
```bash
bash download_data.sh
```

Or set a custom data directory:
```bash
export CHESSBOT_DATA_DIR=/path/to/your/data
```

## 🚀 Quick Start

### Training

Train the model with default settings:
```bash
python cli.py train
```

Train with custom parameters:
```bash
python cli.py train --epochs 20 --lr 1e-4 --data-files 10 --batch-size 256
```

Or run training directly:
```bash
python train.py
```

### Evaluation

Evaluate a trained model:
```bash
python cli.py evaluate --checkpoint checkpoints/best_checkpoint.pt
```

### Inference

#### Interactive Mode
```bash
python cli.py infer --interactive
```

Commands in interactive mode:
- `fen <position>` - Set position (FEN notation)
- `eval` - Evaluate current position
- `move <move>` - Evaluate specific move (UCI format)
- `best` - Get best move
- `quit` - Exit

#### Single Position Evaluation
```bash
python cli.py infer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --top-k 5
```

#### Specific Move Evaluation
```bash
python cli.py infer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --move e2e4
```

### Project Info

View configuration and available checkpoints:
```bash
python cli.py info
```

## 📁 Project Structure

```
chessbot_byte/
├── cli.py              # Main CLI interface
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── inference.py        # Inference engine
├── model.py            # Model architecture
├── configs.py          # Configuration classes
├── dataloader.py       # Data loading utilities
├── tokenizer.py        # FEN tokenization
├── train_utils.py      # Training utilities (EMA, loss)
├── utils.py            # Chess utilities (moves, buckets)
├── bagz.py             # Bagz file format reader
├── download_data.sh    # Data download script
├── requirements.txt    # Python dependencies
└── checkpoints/        # Model checkpoints (created during training)
```

## 🔧 Configuration

Edit `configs.py` to customize:

### Model Configuration
- `d_model`: Model dimension (default: 64)
- `nhead`: Number of attention heads (default: 4)
- `encoder_layers`: Number of encoder layers (default: 4)
- `num_experts`: Number of experts (default: 8)
- `num_experts_per_tok`: Active experts per token (default: 2)

### Training Configuration
- `num_epochs`: Training epochs (default: 10)
- `learning_rate`: Learning rate (default: 1e-4)
- `max_grad_norm`: Gradient clipping (default: 1.0)

### Data Configuration
- `batch_size`: Batch size (default: 256)
- `number_of_files`: Data files to use (default: 3, max: 2148)

## 📊 Model Performance

The model is evaluated using:
- **Loss**: Cross-entropy loss on return bucket predictions
- **Accuracy**: Exact bucket prediction accuracy
- **Top-5/Top-10 Accuracy**: Prediction within top-k buckets

## 🎯 Use Cases

1. **Chess Move Evaluation**: Analyze positions and get move recommendations
2. **Position Analysis**: Understand win probabilities for different moves
3. **Chess AI Research**: Experiment with transformer architectures for chess
4. **Educational**: Learn about transformers and MoE architectures

## 🔬 Technical Details

### Data Format
- **Input**: FEN strings with move annotations
- **Format**: Compressed .bag files (Bagz format)
- **Processing**: Tokenized to fixed-length sequences

### Training Process
1. FEN tokenization (77 tokens)
2. Move encoding (1 token)
3. Return bucket prediction (128 buckets)
4. EMA for stable training
5. Gradient clipping for stability
6. Automatic checkpointing

### Inference Pipeline
1. Parse FEN position
2. Tokenize board state
3. Generate predictions for all legal moves
4. Convert bucket predictions to win probabilities
5. Rank moves by expected value

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add learning rate scheduling
- [ ] Implement validation set evaluation
- [ ] Add TensorBoard logging
- [ ] Support for different model sizes
- [ ] UCI protocol integration for chess GUIs
- [ ] ONNX export for deployment
- [ ] Quantization for faster inference

## 📝 License

Apache License 2.0 (for code derived from DeepMind)

## 🙏 Acknowledgments

- Based on research from DeepMind
- Uses transformer architecture with Mixture of Experts
- Inspired by modern chess engine design

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research/educational project. For competitive chess, consider using traditional engines like Stockfish or Leela Chess Zero.

## 🛠️ Helper Scripts

### Quick Start Script
Automated setup and verification:
```bash
bash quick_start.sh
```

### Verify Setup
Check installation and dependencies:
```bash
python verify_setup.py
```

### Example Usage
See programmatic usage examples:
```bash
python example_usage.py
```

## 💻 Programmatic Usage

```python
from inference import ChessBotInference

# Initialize the bot
bot = ChessBotInference('checkpoints/best_checkpoint.pt')

# Get best move
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
best = bot.get_best_move(fen)
print(f"Best move: {best['move_san']} ({best['win_probability']:.1%})")

# Evaluate top moves
top_moves = bot.evaluate_position(fen, top_k=5)
for move in top_moves:
    print(f"{move['move_san']}: {move['win_probability']:.1%}")
```

See `example_usage.py` for more examples.
