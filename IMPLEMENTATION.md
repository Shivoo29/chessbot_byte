# IMPLEMENTATION.md - Developer Guide

**Last Updated**: 2025-11-19
**Target Audience**: Developers joining the project or doing deep development work

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Current Implementation Status](#current-implementation-status)
4. [Known Issues & Bugs](#known-issues--bugs)
5. [Potential Errors](#potential-errors)
6. [Running & Debugging](#running--debugging)
7. [Code Structure & Flow](#code-structure--flow)
8. [Design Decisions](#design-decisions)
9. [Technical Debt](#technical-debt)
10. [Performance Considerations](#performance-considerations)
11. [Testing Strategy](#testing-strategy)
12. [Development Workflow](#development-workflow)

---

## Project Overview

### What is ChessBot Byte?

ChessBot Byte is a **transformer-based chess move value predictor** that uses a Mixture of Experts (MoE) architecture to predict win probabilities for chess positions and moves.

**NOT a traditional chess engine** - It doesn't use minimax, alpha-beta pruning, or traditional chess algorithms. Instead, it's a learned model trained on chess game data.

### Core Concept

```
Input: FEN position + Move → Model → Win Probability (0-1)
```

The model predicts which return bucket (out of 128 buckets spanning 0-1) the win probability falls into.

### Technical Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch 2.6.0
- **Architecture**: Transformer Encoder with MoE FFN
- **Data Format**: Bagz (compressed binary format)
- **Dependencies**: See requirements.txt (180 packages!)

---

## Architecture Deep Dive

### Model Architecture

```
Input Sequence (79 tokens)
    ↓
[FEN Tokenization: 77 tokens] + [Move: 1 token] + [Return Bucket: 1 token]
    ↓
Embedding Layer (d_model=64)
    ↓
Positional Encoding (learned or sinusoidal)
    ↓
Transformer Encoder (4 layers)
    ├─ Multi-Head Attention (4 heads)
    └─ MoE Feed-Forward Network
        ├─ Gating Network (softmax)
        ├─ 8 Experts (top-2 selected per token)
        └─ Expert: Linear → SiLU → Linear
    ↓
Decoder (LayerNorm + Linear)
    ↓
Log Softmax (128 buckets)
```

### Key Components

#### 1. Tokenization (`tokenizer.py`)
```python
FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ↓
77-token sequence:
[side, board (65 chars expanded), castling (4), en_passant (2), halfmoves (3), fullmoves (3)]
```

**Character Mapping**: 31 unique characters → integer indices
**Padding**: Empty squares represented as '.' (30th index)

#### 2. Data Loading (`dataloader.py`, `bagz.py`)
```python
BagDataSource → ChessDataset → DataLoader
    ↓
Reads .bag files (compressed protocol buffers)
Decodes: (FEN, Move, WinProb) tuples
Converts to: (sequence[79], mask[79])
```

**Important**: Data is sharded across multiple files (max 2148 shards)

#### 3. Model (`model.py`)

**Key Classes**:
- `chessbot_model`: Main model wrapper
- `TransformerEncoderLayer`: Custom encoder with MoE
- `google_expert`: FFN expert using SiLU activation
- `GatingNetwork`: Softmax gating for expert selection
- `cust_embeddings`: Custom embedding + positional encoding
- `decoder`: Final projection to return buckets

**MoE Implementation**:
```python
# Gating selects top-2 experts per token
gating_scores = softmax(GatingNetwork(x))
top_scores, top_indices = gating_scores.topk(2)

# Sparse gating matrix
gating_results = scatter(top_scores, top_indices)

# All experts compute, then weighted by gating
expert_outputs = [expert(x) for expert in experts]
output = sum(gating_results * expert_outputs)
```

#### 4. Training (`train.py`, `train_utils.py`)

**Training Loop**:
```python
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model(inputs)  # log probabilities
        loss = criterion(outputs, inputs, labels)  # custom loss
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        ema.update()  # update EMA weights
```

**Loss Function** (`train_utils.py:loss_fn`):
```python
# Gather log probs for target tokens
true_conditionals = gather(predictions, sequences)

# Apply mask (ignore padding)
true_conditionals = where(mask, 0.0, true_conditionals)

# Average over sequence length
loss = -mean(sum(true_conditionals, dim=1) / seq_lengths)
```

**EMA** (`train_utils.py:EMA`):
```python
# Exponential Moving Average with decay=0.999
ema_value = ema_value - (1 - decay) * (ema_value - param)
```

#### 5. Inference (`inference.py`)

**ChessBotInference Class**:
```python
# Load model
model.load_state_dict(checkpoint['model_state_dict'])

# Predict move value
def predict_move_value(fen, move):
    state_tokens = tokenize(fen)  # 77 tokens
    action = MOVE_TO_ACTION[move]  # 1 token
    sequence = concat([state_tokens, action])  # 78 tokens

    outputs = model(sequence)  # (1, 79, 128)
    probs = exp(outputs[0, -1, :])  # last position

    predicted_bucket = argmax(probs)
    win_prob = bucket_values[predicted_bucket]
    return win_prob
```

---

## Current Implementation Status

### ✅ Fully Implemented

1. **Training Pipeline**
   - EMA integration ✅
   - Automatic checkpointing ✅
   - Progress tracking ✅
   - Statistics export ✅
   - Gradient clipping ✅

2. **Evaluation System**
   - Loss calculation ✅
   - Accuracy metrics ✅
   - Top-k accuracy ✅
   - Results export ✅

3. **Inference Engine**
   - Interactive mode ✅
   - Position evaluation ✅
   - Move prediction ✅
   - Best move selection ✅

4. **CLI & Tools**
   - Unified CLI ✅
   - Setup automation ✅
   - Verification script ✅
   - Example code ✅

### ⚠️ Partially Implemented / TODO

1. **Validation Split**
   - Currently uses ALL data for training
   - No separate validation set
   - **Impact**: Can't detect overfitting properly

2. **Learning Rate Scheduling**
   - Fixed LR throughout training
   - No cosine annealing or warmup
   - **Impact**: Suboptimal convergence

3. **Logging**
   - Basic print statements
   - No TensorBoard/Wandb integration
   - **Impact**: Hard to track experiments

4. **Testing**
   - No unit tests
   - No integration tests
   - Manual testing only
   - **Impact**: Regression risks

### ❌ Not Implemented

1. **UCI Protocol**
   - Can't connect to chess GUIs
   - **Impact**: Limited usability

2. **Model Export**
   - No ONNX export
   - No quantization
   - **Impact**: Deployment challenges

3. **Distributed Training**
   - Single GPU only
   - No DataParallel/DistributedDataParallel
   - **Impact**: Slow on large datasets

4. **Data Augmentation**
   - No position flipping
   - No color swapping
   - **Impact**: Model bias

---

## Known Issues & Bugs

### 🐛 Confirmed Bugs

#### 1. **Data Path Resolution Issue**
**Location**: `configs.py:19-20`
```python
data_dir = os.getenv('CHESSBOT_DATA_DIR', project_root / 'data')
filename = f'{data_dir}/train/action_value-@{number_of_files}_data.bag'
```

**Problem**: Mixing Path object with string interpolation
```python
# If CHESSBOT_DATA_DIR not set:
data_dir = PosixPath('/home/user/chessbot_byte/data')  # Path object
filename = f'{data_dir}/train/...'  # Works but inconsistent
```

**Fix**: Convert to string
```python
data_dir = str(os.getenv('CHESSBOT_DATA_DIR', project_root / 'data'))
```

**Severity**: LOW (works but type-unsafe)

#### 2. **Bagz File Pattern Mismatch**
**Location**: `bagz.py:137`
```python
filename = re.sub(r'@(\d+)', f'{idx:05d}-of-02148', filename)
```

**Problem**: Hard-coded `02148` but user may have fewer files

**Scenario**:
```python
# User has 3 files
data_config.number_of_files = 3
# But bagz.py tries to read:
# 00000-of-02148, 00001-of-02148, 00002-of-02148
# Actual files might be:
# 00000-of-00003, 00001-of-00003, 00002-of-00003
```

**Fix**: Make total shards configurable

**Severity**: HIGH (fails if file naming differs)

#### 3. **Loss Function Clamping**
**Location**: `train_utils.py:32`
```python
sequences = torch.clamp(sequences, min=0, max=127)  # jax wont require it
```

**Problem**:
- Silently clamps out-of-range values
- Comment mentions JAX but we use PyTorch
- If vocab size > 128, this will cause bugs

**Why it exists**: Comment says "NaN dedeta hai" (gives NaN)

**Severity**: MEDIUM (hidden bug if vocab changes)

#### 4. **EMA Device Mismatch Potential**
**Location**: `train_utils.py:96-98`
```python
self.ema_values[name] = param.data.clone().detach()
if device is not None:
    self.ema_values[name] = self.ema_values[name].to(device)
```

**Problem**: If model is on CUDA but EMA initialized without device param, mismatch occurs

**Scenario**:
```python
model = model.to('cuda')
ema = EMA(model)  # device=None
# ema_values stay on CPU!
```

**Fix**: Auto-detect device from model

**Severity**: MEDIUM (breaks on GPU without explicit device)

### ⚠️ Known Limitations

#### 1. **Memory Usage**
- Model loads entire dataset into memory
- No streaming for large datasets
- **Impact**: OOM on large files (>10GB)

#### 2. **Checkpoint Size**
- Full checkpoint: ~50MB
- Saves optimizer state (unnecessary for inference)
- **Impact**: Slow save/load

#### 3. **No Resume Training**
- Can't resume from checkpoint
- Optimizer state loaded but epoch not tracked
- **Impact**: Restart from scratch

#### 4. **Hard-coded Bucket Count**
- 128 buckets fixed everywhere
- Changing requires code updates
- **Impact**: Inflexible

---

## Potential Errors

### Runtime Errors You Might Encounter

#### 1. **CUDA Out of Memory**
```python
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Causes**:
- Batch size too large
- Model too big for GPU
- Gradient accumulation

**Solutions**:
```bash
# Reduce batch size
python cli.py train --batch-size 128

# Use CPU
# Edit configs.py: device = 'cpu'

# Clear CUDA cache
torch.cuda.empty_cache()
```

#### 2. **File Not Found**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'data/train/...'
```

**Causes**:
- Data not downloaded
- Wrong CHESSBOT_DATA_DIR
- File naming mismatch

**Solutions**:
```bash
# Download data
bash download_data.sh

# Set custom path
export CHESSBOT_DATA_DIR=/path/to/data

# Check file names match pattern
ls data/train/
```

#### 3. **Import Errors**
```python
ModuleNotFoundError: No module named 'torch'
```

**Causes**:
- Dependencies not installed
- Wrong Python environment

**Solutions**:
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+

# Verify imports
python verify_setup.py
```

#### 4. **Shape Mismatch**
```python
RuntimeError: shape mismatch: value tensor of shape [256, 79] but got [256, 78]
```

**Causes**:
- Sequence length inconsistency
- Wrong tokenization
- Data corruption

**Debug**:
```python
# Check sequence lengths
for inputs, labels in dataloader:
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    break
```

#### 5. **NaN Loss**
```python
Loss: nan
```

**Causes**:
- Learning rate too high
- Gradient explosion
- Division by zero
- Bad initialization

**Solutions**:
```python
# Reduce learning rate
train_config.learning_rate = 1e-5

# Check for NaN in data
torch.isnan(inputs).any()

# Add gradient clipping (already implemented)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Data-Related Errors

#### 1. **Bagz Decoding Error**
```python
ValueError: Bagz file too small
```

**Cause**: Corrupted .bag file

**Solution**: Re-download data

#### 2. **Invalid FEN**
```python
ValueError: Invalid FEN string
```

**Cause**: Malformed FEN in data

**Solution**: Add FEN validation in dataloader

#### 3. **Move Not in Action Space**
```python
KeyError: 'e8e9'
```

**Cause**: Move not in the 4,672 possible chess moves

**Solution**: This shouldn't happen with legal moves; check data

---

## Running & Debugging

### Development Setup

```bash
# 1. Clone and enter directory
git clone <repo>
cd chessbot_byte

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py

# 5. Run setup
python cli.py setup
```

### Running Training (Debug Mode)

```python
# train_debug.py
import torch
from configs import train_config, data_config, parent_config

# Use small dataset
data_config.mini_dataset = True
data_config.mini_set_count = 100

# Use CPU for debugging
parent_config.device = 'cpu'

# Short training
train_config.num_epochs = 2

# Run training
import train
```

### Debugging Tips

#### 1. **Print Model Summary**
```python
from model import chessbot_model

model = chessbot_model()
print(model)
print(f"Total params: {sum(p.numel() for p in model.parameters())}")
```

#### 2. **Inspect Data**
```python
from dataloader import dataloader_instance

for batch_idx, (inputs, labels) in enumerate(dataloader_instance):
    print(f"Batch {batch_idx}:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Input dtype: {inputs.dtype}")
    print(f"  Input range: [{inputs.min()}, {inputs.max()}]")
    print(f"  Label shape: {labels.shape}")
    print(f"  Sample input: {inputs[0]}")
    if batch_idx >= 2:
        break
```

#### 3. **Check Model Output**
```python
from model import chessbot_model
import torch

model = chessbot_model()
model.eval()

# Dummy input (batch=2, seq=79)
dummy_input = torch.randint(0, 31, (2, 79))

with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")  # Should be (2, 79, 128)
print(f"Output range: [{output.min()}, {output.max()}]")
print(f"Output sum: {torch.exp(output).sum(dim=-1)}")  # Should be ~1 (log softmax)
```

#### 4. **Profile Training**
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    # Training code here
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, inputs, labels)
        loss.backward()
        optimizer.step()
        break

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

#### 5. **Memory Debugging**
```python
import torch

# Track GPU memory
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # After each step
    torch.cuda.reset_peak_memory_stats()
```

### Interactive Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

### Logging for Debugging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
```

---

## Code Structure & Flow

### Data Flow

```
1. Data Loading
   BagDataSource(__getitem__) → bytes
   ↓
   ConvertActionValueDataToSequence(map) → decode
   ↓
   (fen, move, win_prob) → tokenize + convert
   ↓
   (sequence[79], mask[79]) as tensors

2. Model Forward
   Input: (batch, 79) int64
   ↓
   Embedding: (batch, 79, 64)
   ↓
   Encoder Layer 1-4:
      MultiHeadAttention → (batch, 79, 64)
      ↓
      MoE Feed-Forward → (batch, 79, 64)
   ↓
   Decoder: (batch, 79, 128)
   ↓
   Log Softmax: (batch, 79, 128)

3. Loss Calculation
   Predictions: (batch, 79, 128) log probs
   Targets: (batch, 79) token indices
   Mask: (batch, 79) boolean
   ↓
   Gather target log probs
   ↓
   Apply mask
   ↓
   Average over sequence
   ↓
   Scalar loss

4. Training Step
   loss.backward()
   ↓
   clip_grad_norm_()
   ↓
   optimizer.step()
   ↓
   ema.update()
```

### Call Graph (Training)

```
main (train.py)
├─ dataloader_instance (dataloader.py)
│  ├─ ChessDataset.__init__
│  │  └─ BagDataSource (bagz.py)
│  └─ DataLoader
├─ chessbot_model (model.py)
│  ├─ cust_embeddings
│  ├─ TransformerEncoderLayer (x4)
│  │  ├─ MultiheadAttention
│  │  └─ _ff_block (MoE)
│  │     ├─ GatingNetwork
│  │     └─ google_expert (x8)
│  └─ decoder
├─ loss_fn (train_utils.py)
└─ EMA (train_utils.py)
   ├─ update()
   ├─ apply_ema()
   └─ restore()
```

### File Dependencies

```
cli.py
├─ imports train, evaluate, inference
└─ uses argparse

train.py
├─ model.py (chessbot_model)
├─ dataloader.py (dataloader_instance)
├─ train_utils.py (EMA, loss_fn)
├─ configs.py (all configs)
└─ tokenizer.py (SEQUENCE_LENGTH)

model.py
├─ configs.py (model_config, parent_config)
└─ torch.nn modules

dataloader.py
├─ bagz.py (BagDataSource)
├─ tokenizer.py (tokenize)
├─ utils.py (MOVE_TO_ACTION, compute_return_buckets)
└─ configs.py (data_config)

inference.py
├─ model.py (chessbot_model)
├─ tokenizer.py (tokenize)
├─ utils.py (MOVE_TO_ACTION, get_uniform_buckets_edges_values)
├─ configs.py (parent_config)
└─ chess (python-chess library)

evaluate.py
├─ model.py (chessbot_model)
├─ dataloader.py (ChessDataset)
├─ train_utils.py (loss_fn)
└─ configs.py (all configs)
```

---

## Design Decisions

### Why Mixture of Experts?

**Decision**: Use MoE instead of standard FFN

**Rationale**:
- Sparse activation (only 2/8 experts active)
- Better capacity without 8x compute
- Different experts can specialize (opening, endgame, tactics)

**Trade-offs**:
- More complex implementation
- Harder to debug
- Training instability (gating network)

### Why 128 Return Buckets?

**Decision**: Discretize win probability into 128 buckets

**Rationale**:
- Classification easier than regression
- Bucket granularity: 1/128 ≈ 0.78%
- Matches DeepMind's approach

**Trade-offs**:
- Loss of precision
- Arbitrary bucketing
- Edge effects at boundaries

### Why EMA?

**Decision**: Use Exponential Moving Average of weights

**Rationale**:
- Smoother training
- Better generalization
- Common in large models (Stable Diffusion, etc.)

**Trade-offs**:
- 2x memory (store both current and EMA weights)
- Slightly slower training
- Extra complexity

### Why Custom Embeddings?

**Decision**: Custom embedding instead of nn.Embedding

**Rationale**:
- Need sqrt(d_model) scaling
- Position + token embedding sum
- Control initialization

**Trade-offs**:
- Reinventing the wheel
- More code to maintain

### Why Bagz Format?

**Decision**: Use Bagz compressed format

**Rationale**:
- Fast random access
- Compression (Zstandard)
- PyGrain compatibility
- Designed for large datasets

**Trade-offs**:
- Non-standard format
- Requires special reader
- Harder to inspect

---

## Technical Debt

### High Priority

1. **Hard-coded Values**
   ```python
   # bagz.py:137
   f'{idx:05d}-of-02148'  # Hard-coded shard count

   # configs.py:10
   num_return_buckets = 128  # Hard-coded everywhere

   # model.py:344
   self.sequence_length = SEQUENCE_LENGTH + 2  # Magic number
   ```

   **Fix**: Make configurable

2. **No Type Hints**
   ```python
   # Most functions lack type hints
   def loss_fn(predictions, sequences, mask):  # No types!
   ```

   **Fix**: Add typing annotations

3. **Path/String Inconsistency**
   ```python
   # configs.py mixes Path and str
   data_dir = os.getenv(..., project_root / 'data')  # Path
   filename = f'{data_dir}/train/...'  # str interpolation
   ```

   **Fix**: Use pathlib consistently

4. **No Input Validation**
   ```python
   # inference.py:67
   def predict_move_value(self, fen, move):
       # No FEN validation!
       # No move validation!
   ```

   **Fix**: Add validation

### Medium Priority

1. **Global State**
   ```python
   # dataloader.py:14
   dataloader_instance = None  # Global variable

   # Later:
   dataloader_instance = create_dataloader()  # Modified at import time
   ```

   **Fix**: Dependency injection

2. **Comment Clutter**
   ```python
   # model.py has lots of commented code
   # Lines 138-153: Old implementation
   ```

   **Fix**: Remove dead code

3. **Magic Numbers**
   ```python
   # tokenizer.py:44
   SEQUENCE_LENGTH = 77  # No explanation why

   # train.py:79
   if (batch_idx + 1) % 10 == 0:  # Why 10?
   ```

   **Fix**: Named constants with comments

4. **Error Messages**
   ```python
   # Many places just raise without context
   raise ValueError("Invalid move")  # Which move? Why invalid?
   ```

   **Fix**: Better error messages

### Low Priority

1. **Code Duplication**
   ```python
   # Similar code in train.py and evaluate.py
   # for model loading
   ```

2. **Inconsistent Naming**
   ```python
   # Some use snake_case, some camelCase
   # miniDataSet vs mini_dataset (fixed but pattern exists)
   ```

3. **Long Functions**
   ```python
   # cli.py:train_command could be broken down
   # inference.py:evaluate_position is long
   ```

---

## Performance Considerations

### Bottlenecks

1. **Data Loading**
   - Bagz decompression (Zstandard)
   - File I/O
   - **Profile**: ~30% of training time

2. **MoE Feed-Forward**
   - All 8 experts compute (not truly sparse)
   - Then weighted sum
   - **Profile**: ~40% of forward pass

3. **Attention**
   - O(n²) complexity
   - Sequence length = 79 (manageable)
   - **Profile**: ~30% of forward pass

4. **EMA Update**
   - Clone all parameters every step
   - **Profile**: ~5% overhead

### Optimization Opportunities

#### 1. **Data Loading**
```python
# Current: Single-threaded
DataLoader(dataset, batch_size=256, shuffle=True)

# Optimize: Multi-worker
DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
```

#### 2. **Mixed Precision**
```python
# Current: FP32
dtype = torch.float32

# Optimize: AMP
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, inputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. **Gradient Accumulation**
```python
# Current: Update every batch

# Optimize: Accumulate gradients
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, inputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 4. **True Sparse MoE**
```python
# Current: All experts compute

# Optimize: Only compute selected experts
# (Requires custom CUDA kernel or using libraries like tutel)
```

### Memory Optimization

```python
# 1. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.encoders:
        x = checkpoint(layer, x)  # Trade compute for memory

# 2. Empty cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# 3. Delete large tensors
del outputs
torch.cuda.empty_cache()
```

### Profiling Commands

```bash
# PyTorch profiler
python -m torch.utils.bottleneck train.py

# Line profiler
pip install line_profiler
kernprof -l -v train.py

# Memory profiler
pip install memory_profiler
python -m memory_profiler train.py

# CUDA profiler (if using GPU)
nvprof python train.py
```

---

## Testing Strategy

### Current State: ⚠️ **NO TESTS**

This is a **major gap**. Here's what should be tested:

### Unit Tests Needed

#### 1. **Tokenization Tests**
```python
# test_tokenizer.py
def test_tokenize_starting_position():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tokens = tokenize(fen)
    assert len(tokens) == 77
    assert tokens.dtype == np.uint8

def test_tokenize_empty_squares():
    fen = "8/8/8/8/8/8/8/8 w - - 0 1"  # Empty board
    tokens = tokenize(fen)
    # Should be all dots (index 30)
```

#### 2. **Model Tests**
```python
# test_model.py
def test_model_forward_shape():
    model = chessbot_model()
    x = torch.randint(0, 31, (2, 79))
    output = model(x)
    assert output.shape == (2, 79, 128)

def test_model_output_probabilities():
    model = chessbot_model()
    model.eval()
    x = torch.randint(0, 31, (1, 79))
    output = model(x)
    # Log softmax should sum to ~1 when exponentiated
    probs = torch.exp(output[0, 0, :])
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
```

#### 3. **Loss Tests**
```python
# test_loss.py
def test_loss_fn_shape():
    predictions = torch.randn(2, 79, 128)
    sequences = torch.randint(0, 128, (2, 79))
    mask = torch.zeros(2, 79, dtype=torch.bool)
    loss = loss_fn(predictions, sequences, mask)
    assert loss.ndim == 0  # Scalar

def test_loss_fn_masking():
    predictions = torch.randn(2, 79, 128)
    sequences = torch.randint(0, 128, (2, 79))
    mask = torch.ones(2, 79, dtype=torch.bool)  # Mask all
    loss = loss_fn(predictions, sequences, mask)
    # Should be 0 or very small (all masked)
```

#### 4. **Utils Tests**
```python
# test_utils.py
def test_move_to_action_bijection():
    # Every move should map to unique action
    assert len(MOVE_TO_ACTION) == len(ACTION_TO_MOVE)

def test_action_to_move_inverse():
    for move, action in MOVE_TO_ACTION.items():
        assert ACTION_TO_MOVE[action] == move
```

### Integration Tests Needed

```python
# test_integration.py
def test_train_one_batch():
    # Can we train for one batch without errors?
    from train import model, optimizer, criterion, train_loader

    inputs, labels = next(iter(train_loader))
    outputs = model(inputs)
    loss = criterion(outputs, inputs, labels)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)

def test_save_load_checkpoint():
    # Can we save and load a checkpoint?
    model = chessbot_model()
    path = "test_checkpoint.pt"
    torch.save({'model_state_dict': model.state_dict()}, path)

    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    os.remove(path)

def test_inference_on_starting_position():
    # Can we run inference on a known position?
    bot = ChessBotInference('checkpoints/best_checkpoint.pt')
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = bot.predict_move_value(fen, "e2e4")

    assert 0 <= result['win_probability'] <= 1
    assert isinstance(result['confidence'], float)
```

### How to Add Testing

```bash
# 1. Install pytest
pip install pytest pytest-cov

# 2. Create tests directory
mkdir tests
touch tests/__init__.py
touch tests/test_tokenizer.py
touch tests/test_model.py

# 3. Run tests
pytest tests/ -v

# 4. With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/chessbot_byte.git
cd chessbot_byte

# 2. Create branch
git checkout -b feature/your-feature-name

# 3. Install in development mode
pip install -e .  # If setup.py exists
# Or just:
pip install -r requirements.txt

# 4. Install dev dependencies
pip install pytest black flake8 mypy

# 5. Verify setup
python verify_setup.py
```

### Code Quality Tools

#### 1. **Black (Formatting)**
```bash
# Format all Python files
black .

# Check without changing
black --check .
```

#### 2. **Flake8 (Linting)**
```bash
# Check for style issues
flake8 . --max-line-length=100 --ignore=E501,W503

# Save config in setup.cfg:
[flake8]
max-line-length = 100
ignore = E501, W503
exclude = __pycache__, venv
```

#### 3. **MyPy (Type Checking)**
```bash
# Check types
mypy train.py --ignore-missing-imports

# Config in mypy.ini:
[mypy]
python_version = 3.8
ignore_missing_imports = True
```

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/add-validation-split

# 2. Make changes
# ... edit files ...

# 3. Run tests (when they exist)
pytest tests/

# 4. Format code
black .

# 5. Commit
git add .
git commit -m "Add validation split to training

- Split data into train/val (80/20)
- Evaluate on val set each epoch
- Track val loss separately"

# 6. Push
git push origin feature/add-validation-split

# 7. Create PR
# Go to GitHub and create pull request
```

### Pre-commit Hooks (Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
EOF

# Install hooks
pre-commit install

# Now hooks run automatically on git commit
```

### Development Checklist

Before committing:
- [ ] Code runs without errors
- [ ] Added docstrings for new functions
- [ ] Updated documentation if needed
- [ ] Ran code formatter (black)
- [ ] Ran linter (flake8)
- [ ] Added tests for new features
- [ ] All tests pass
- [ ] Checked for performance regressions

---

## Common Development Tasks

### Adding a New Model Configuration

```python
# 1. Edit configs.py
class model_config(parent_config):
    # Add new parameter
    use_layer_dropout = True
    layer_dropout_rate = 0.1

# 2. Edit model.py
class TransformerEncoderLayer(nn.Module):
    def __init__(self, ..., use_layer_dropout=model_config.use_layer_dropout):
        # Add dropout layer
        if use_layer_dropout:
            self.layer_dropout = nn.Dropout(model_config.layer_dropout_rate)

# 3. Update documentation
# README.md, USAGE.md

# 4. Test
python cli.py train --epochs 1
```

### Adding a New Evaluation Metric

```python
# 1. Edit evaluate.py
def evaluate_model(model, dataloader, device):
    # ... existing code ...

    # Add new metric
    total_perplexity = 0
    for inputs, labels in dataloader:
        # ... get outputs ...
        perplexity = torch.exp(loss)
        total_perplexity += perplexity.item()

    avg_perplexity = total_perplexity / total_batches
    metrics['perplexity'] = avg_perplexity

    return metrics

# 2. Update CLI output
# evaluate.py: Print new metric

# 3. Update docs
# README.md: Add to metrics list
```

### Adding a New CLI Command

```python
# 1. Edit cli.py
def benchmark_command(args):
    """Run benchmark."""
    print("Running benchmark...")
    # Implementation

# 2. Add parser
benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark')
benchmark_parser.add_argument('--iterations', type=int, default=100)
benchmark_parser.set_defaults(func=benchmark_command)

# 3. Test
python cli.py benchmark --iterations 50

# 4. Document
# README.md, USAGE.md
```

---

## Debugging Production Issues

### Model Predicts Random Values

**Symptoms**: Win probabilities all ~50%, no variation

**Diagnosis**:
```python
# Check if model is actually trained
checkpoint = torch.load('checkpoints/best_checkpoint.pt')
print(checkpoint.keys())
print(f"Training epochs: {checkpoint.get('epoch', 'unknown')}")
print(f"Training loss: {checkpoint.get('loss', 'unknown')}")

# Check model weights aren't random
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
    # If std is huge, weights not trained
```

**Fixes**:
- Train for more epochs
- Check training loss is decreasing
- Verify data is loading correctly

### Training Stalls / Loss Doesn't Decrease

**Symptoms**: Loss stays constant or increases

**Diagnosis**:
```python
# 1. Check learning rate
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# 2. Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")

# 3. Check data variation
for inputs, labels in dataloader:
    print(f"Input unique values: {torch.unique(inputs).numel()}")
    print(f"Label unique values: {torch.unique(labels).numel()}")
    break
```

**Fixes**:
- Increase learning rate
- Check gradients flow to all layers
- Verify data has variation
- Try different optimizer (AdamW, SGD)

### Out of Memory During Inference

**Symptoms**: OOM when running inference

**Diagnosis**:
```python
# Check model size
model_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model size: {model_size / 1e6:.2f} MB")

# Check if model is in eval mode
print(f"Model training: {model.training}")  # Should be False
```

**Fixes**:
```python
# 1. Use eval mode
model.eval()

# 2. Use no_grad
with torch.no_grad():
    outputs = model(inputs)

# 3. Move to CPU
model = model.cpu()

# 4. Reduce batch size
# Inference one position at a time
```

---

## FAQ for Developers

### Q: Why is training so slow?

A: Several reasons:
1. CPU vs GPU (10-20x difference)
2. Batch size too small
3. Data loading bottleneck (increase `num_workers`)
4. MoE implementation (all experts compute)
5. No mixed precision training

### Q: Can I change the model architecture?

A: Yes, but carefully:
- Change `model_config` parameters
- Retrain from scratch (can't load old checkpoints)
- Update documentation

### Q: How do I add a new dataset?

A: Options:
1. Convert to Bagz format (complex)
2. Create new Dataset class inheriting from torch.utils.data.Dataset
3. Modify `dataloader.py` to support your format

### Q: Why 128 buckets specifically?

A: Arbitrary choice from original research. Could be 64, 256, etc.
- More buckets = finer granularity but harder to learn
- Fewer buckets = easier to learn but less precise

### Q: Can this beat Stockfish?

A: No. This is a **learned value predictor**, not a search-based engine.
- Stockfish uses minimax + alpha-beta + evaluation
- This uses learned patterns
- For strong play, you'd need search on top of this

### Q: What's the expected accuracy?

A: Depends on training:
- Random baseline: ~0.78% (1/128)
- Undertrained: 10-15%
- Well-trained: 15-25%
- Top-5 accuracy: 40-60%

### Q: How much data do I need?

A: More is better:
- Minimum: 1M positions (testing)
- Decent: 10M positions
- Good: 100M+ positions
- Best: Billions (like AlphaZero)

### Q: Can I use this for other board games?

A: Yes, with modifications:
1. Change tokenization (chess-specific)
2. Change action space (4,672 chess moves)
3. Change return calculation (win probability)
4. Retrain from scratch

---

## Getting Help

### When You're Stuck

1. **Check Logs**: Look for error messages
2. **Read Docs**: README.md, USAGE.md
3. **Search Code**: grep for error messages
4. **Debug Print**: Add print statements
5. **Use Debugger**: pdb, IPython embed
6. **Ask Community**: GitHub issues

### Useful Debug Snippets

```python
# 1. Check tensor shapes everywhere
print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

# 2. Check for NaN/Inf
assert not torch.isnan(tensor).any(), "NaN detected!"
assert not torch.isinf(tensor).any(), "Inf detected!"

# 3. Memory tracking
import tracemalloc
tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")

# 4. Time tracking
import time
start = time.time()
# ... your code ...
print(f"Elapsed: {time.time() - start:.2f}s")
```

---

## Summary for New Developers

### Quick Start

1. **Read this file** (you're doing it!)
2. **Run setup**: `bash quick_start.sh`
3. **Run verification**: `python verify_setup.py`
4. **Train small model**: `python cli.py train --epochs 2 --data-files 1`
5. **Read code**: Start with `model.py`, then `train.py`
6. **Experiment**: Change configs, see what happens
7. **Contribute**: Fix bugs, add features, improve docs

### Key Files to Understand

1. **model.py** - Architecture (start here!)
2. **train.py** - Training loop
3. **configs.py** - All configuration
4. **dataloader.py** - Data pipeline
5. **inference.py** - How to use trained model

### Architecture in One Sentence

**Transformer encoder with Mixture of Experts that tokenizes chess positions (FEN) and predicts win probability buckets.**

### Most Important Design Choices

1. **MoE instead of standard FFN** - Sparse activation
2. **Bucketized output** - Classification not regression
3. **EMA** - Better generalization
4. **Bagz format** - Fast compressed data
5. **Custom loss** - Sequence-aware masking

---

**Good luck and happy coding! 🚀**

If you find bugs or have questions, please open an issue or contribute a fix!
