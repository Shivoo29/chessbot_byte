# Changelog

All notable changes and improvements to ChessBot Byte.

## [2.0.0] - Complete Platform Overhaul

### 🎉 Major Features Added

#### Training Infrastructure
- **Exponential Moving Average (EMA)**: Integrated EMA for better model generalization
- **Automatic Checkpointing**: Saves latest, best, and final models automatically
- **Training Statistics**: JSON export of all training metrics
- **Progress Tracking**: Batch-level and epoch-level progress reporting
- **Gradient Clipping**: Stable training with gradient norm clipping

#### Evaluation System
- **Comprehensive Metrics**: Loss, accuracy, top-5, and top-10 accuracy
- **Flexible Evaluation**: Configurable checkpoint and dataset selection
- **Results Export**: JSON export of evaluation results
- **Chess-Specific Metrics**: Metrics tailored for chess move prediction

#### Inference Engine
- **Interactive Mode**: Real-time position analysis and move evaluation
- **Batch Processing**: Evaluate multiple positions efficiently
- **Best Move Selection**: Automatic best move recommendation
- **Win Probability**: Converts predictions to interpretable win percentages
- **Legal Move Validation**: Only evaluates legal chess moves

#### Command-Line Interface
- **Unified CLI**: Single entry point for all operations (`cli.py`)
- **Setup Command**: Automatic project initialization
- **Info Command**: Display project configuration and status
- **Flexible Arguments**: Customizable parameters for all operations

### 🔧 Technical Improvements

#### Code Quality
- **Removed Debug Code**: Cleaned up print statements from model.py
- **Added Documentation**: Comprehensive docstrings for all functions
- **Type Hints**: Better code clarity and IDE support
- **Error Handling**: Robust error handling throughout

#### Configuration
- **Environment Agnostic**: No hard-coded paths
- **Environment Variables**: Support for `CHESSBOT_DATA_DIR`
- **Flexible Device**: Auto-detect GPU/CPU
- **Standardized Naming**: Consistent variable naming conventions

#### Architecture
- **Cleaned MoE Implementation**: Improved feed-forward block
- **Better Code Organization**: Modular and maintainable structure
- **Optimized Data Loading**: Efficient data pipeline

### 📚 Documentation

#### New Documentation Files
- **README.md**: Complete project overview and quick start
- **USAGE.md**: Detailed usage guide with examples
- **CONTRIBUTING.md**: Contribution guidelines
- **CHANGELOG.md**: This file, tracking all changes

#### Documentation Sections
- Installation instructions
- Quick start guide
- Configuration reference
- API documentation
- Troubleshooting guide
- Best practices
- Performance tips

### 🚀 New Scripts

1. **cli.py**: Unified command-line interface
   - Setup command
   - Train command
   - Evaluate command
   - Infer command
   - Info command

2. **evaluate.py**: Comprehensive evaluation
   - Multiple metrics
   - Configurable parameters
   - Results export

3. **inference.py**: Production-ready inference
   - ChessBotInference class
   - Interactive mode
   - Batch processing
   - Move ranking

### 🔄 Breaking Changes

- Configuration file structure updated
- Device configuration now auto-detects GPU
- Data path configuration changed to use environment variables
- Renamed `miniDataSet` to `mini_dataset` for consistency

### 🐛 Bug Fixes

- Fixed hard-coded user paths in configs.py
- Removed debug print statements causing cluttered output
- Fixed device configuration to properly use GPU when available
- Corrected variable naming inconsistencies

### 📦 Project Structure

New file organization:
```
chessbot_byte/
├── cli.py                  # Main CLI interface
├── train.py               # Enhanced training script
├── evaluate.py            # New evaluation script
├── inference.py           # New inference engine
├── model.py               # Cleaned model code
├── configs.py             # Updated configuration
├── dataloader.py          # Fixed data loader
├── README.md              # Comprehensive documentation
├── USAGE.md               # Detailed usage guide
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # This changelog
└── .gitignore            # Updated ignore patterns
```

### 🎯 User Experience Improvements

#### Onboarding
- One-command setup: `python cli.py setup`
- Clear error messages with solutions
- Helpful command examples
- Project info command

#### Training
- Real-time progress updates
- Automatic best model tracking
- Training statistics export
- Clear completion summary

#### Evaluation
- Multiple accuracy metrics
- Easy checkpoint comparison
- Results export for analysis
- Performance benchmarking

#### Inference
- Interactive exploration mode
- Simple command-line usage
- Programmatic API
- Clear output formatting

### 📊 Performance

- Proper EMA integration for better accuracy
- Efficient batch processing
- GPU auto-detection and usage
- Optimized data loading

### 🔮 Future Improvements

Planned for next releases:
- Learning rate scheduling
- Validation set support
- TensorBoard logging
- UCI protocol integration
- Model quantization
- ONNX export
- Multi-GPU training

## [1.0.0] - Initial Implementation

### Features
- Basic transformer model with MoE
- Simple training loop
- FEN tokenization
- Chess move utilities
- Data loading from .bag files

### Known Issues
- Hard-coded paths
- Debug print statements
- No evaluation script
- No inference capabilities
- Limited documentation

---

## Migration Guide

### From v1.0.0 to v2.0.0

1. **Update Configs**:
   ```python
   # Old
   device = 'cpu'
   filename = '/home/shivam/Desktop/...'

   # New
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   data_dir = os.getenv('CHESSBOT_DATA_DIR', project_root / 'data')
   ```

2. **Use New CLI**:
   ```bash
   # Old
   python train.py

   # New
   python cli.py train
   ```

3. **Run Setup**:
   ```bash
   python cli.py setup
   ```

4. **Set Data Path** (if needed):
   ```bash
   export CHESSBOT_DATA_DIR=/path/to/data
   ```

---

For detailed usage instructions, see [USAGE.md](USAGE.md).
For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
