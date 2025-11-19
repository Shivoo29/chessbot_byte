# Contributing to ChessBot Byte

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/chessbot_byte.git
   cd chessbot_byte
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run setup:
   ```bash
   python cli.py setup
   ```

## Development Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Test your changes:
   ```bash
   # Test with small dataset
   python cli.py train --epochs 1 --data-files 1
   python cli.py evaluate
   python cli.py infer --interactive
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add comments for complex logic

Example:
```python
def evaluate_position(self, fen: str, top_k: int = 5) -> list:
    """Evaluate all legal moves from a position.

    Args:
        fen: Position in FEN notation
        top_k: Number of top moves to return

    Returns:
        List of dictionaries with move evaluations
    """
    # Implementation
```

## Areas for Contribution

### High Priority

- [ ] **Learning Rate Scheduling**: Add cosine annealing or step decay
- [ ] **Validation Set**: Split data and add validation loop
- [ ] **TensorBoard Logging**: Add detailed metrics tracking
- [ ] **Unit Tests**: Add comprehensive test coverage

### Medium Priority

- [ ] **UCI Protocol**: Enable integration with chess GUIs
- [ ] **Model Variants**: Support for different model sizes
- [ ] **Data Augmentation**: Board flipping, rotation
- [ ] **Better Metrics**: Add chess-specific evaluation metrics

### Advanced Features

- [ ] **ONNX Export**: For production deployment
- [ ] **Quantization**: INT8 quantization for faster inference
- [ ] **Distributed Training**: Multi-GPU support
- [ ] **Hyperparameter Search**: Automated tuning

## Testing Guidelines

Before submitting a PR:

1. **Test Training**:
   ```bash
   python cli.py train --epochs 2 --data-files 1
   ```

2. **Test Evaluation**:
   ```bash
   python cli.py evaluate
   ```

3. **Test Inference**:
   ```bash
   python cli.py infer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
   ```

4. **Test CLI**:
   ```bash
   python cli.py info
   python cli.py setup
   ```

## Documentation

When adding features:

1. Update relevant docstrings
2. Add examples to README.md if applicable
3. Update USAGE.md with usage instructions
4. Add entry to this CONTRIBUTING.md if it's a new contribution area

## Pull Request Guidelines

### PR Title Format

- `Add: new feature`
- `Fix: bug description`
- `Update: component improvement`
- `Docs: documentation changes`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed the code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Tested changes
```

## Feature Request Process

1. Open an issue with "Feature Request:" prefix
2. Describe the feature and use case
3. Discuss implementation approach
4. Get approval before starting work
5. Submit PR when ready

## Bug Report Process

1. Check if issue already exists
2. Open issue with "Bug:" prefix
3. Include:
   - Description of the bug
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - System info (OS, Python version, PyTorch version)
   - Error messages/logs

## Code Review Process

1. Maintainers will review PRs
2. Address review comments
3. Once approved, PR will be merged
4. Your contribution will be acknowledged!

## Community Guidelines

- Be respectful and constructive
- Help others in issues and discussions
- Share your results and experiments
- Contribute documentation improvements
- Report bugs and suggest features

## Development Tips

### Quick Testing

Use mini dataset for fast iteration:
```python
# In configs.py
data_config.mini_dataset = True
data_config.mini_set_count = 100
```

### Debugging

Add verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Profile training:
```python
import torch.profiler as profiler

with profiler.profile() as prof:
    # Your code
    pass

print(prof.key_averages().table())
```

## Questions?

- Open an issue for questions
- Tag with "question" label
- Maintainers will respond

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in README.md

Thank you for contributing to ChessBot Byte! 🎉
