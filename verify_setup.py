#!/usr/bin/env python3
"""
Verify ChessBot Byte installation and setup.

This script checks that all dependencies are installed and
the project is ready to use.
"""

import sys
from pathlib import Path


def check_imports():
    """Check that all required packages can be imported."""
    print("Checking dependencies...")
    errors = []

    packages = [
        ('torch', 'PyTorch'),
        ('chess', 'python-chess'),
        ('numpy', 'NumPy'),
        ('apache_beam', 'Apache Beam'),
        ('zstandard', 'Zstandard'),
    ]

    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - MISSING")
            errors.append(name)

    return errors


def check_project_files():
    """Check that all project files exist."""
    print("\nChecking project files...")
    errors = []

    files = [
        'cli.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'model.py',
        'configs.py',
        'dataloader.py',
        'tokenizer.py',
        'train_utils.py',
        'utils.py',
        'bagz.py',
        'requirements.txt',
        'README.md',
        'USAGE.md',
    ]

    for file in files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            errors.append(file)

    return errors


def check_directories():
    """Check that required directories exist or can be created."""
    print("\nChecking directories...")

    directories = ['checkpoints', 'data', 'logs']

    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ (exists)")
        else:
            print(f"  ⚠ {dir_name}/ (will be created)")

    return []


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {device_name}")
        else:
            print(f"  ⚠ No GPU available (will use CPU)")
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")

    return []


def check_data():
    """Check if training data exists."""
    print("\nChecking data...")

    from configs import data_config

    # Check if data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"  ⚠ Data directory not found")
        print(f"    Run: python cli.py setup")
        return ['data']

    # Check for training data files
    data_pattern = str(data_config.filename).replace('@3', '00000-of-02148')
    data_file = Path(data_pattern)

    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Training data found ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠ Training data not found")
        print(f"    Expected: {data_file}")
        print(f"    Run: bash download_data.sh")
        print(f"    Or set: export CHESSBOT_DATA_DIR=/path/to/data")
        return ['training_data']

    return []


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("ChessBot Byte - Setup Verification")
    print("=" * 80)
    print()

    all_errors = []

    # Run all checks
    all_errors.extend(check_imports())
    all_errors.extend(check_project_files())
    all_errors.extend(check_directories())
    all_errors.extend(check_gpu())
    all_errors.extend(check_data())

    # Summary
    print("\n" + "=" * 80)
    if all_errors:
        print("❌ Setup verification FAILED")
        print(f"\nIssues found: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")
        print("\nPlease fix the issues above and run this script again.")
        print("\nFor help, see:")
        print("  - README.md for installation instructions")
        print("  - USAGE.md for detailed setup guide")
        sys.exit(1)
    else:
        print("✅ Setup verification PASSED")
        print("\nAll checks completed successfully!")
        print("\nNext steps:")
        print("  1. Train a model:    python cli.py train")
        print("  2. Evaluate model:   python cli.py evaluate")
        print("  3. Use inference:    python cli.py infer --interactive")
        print("\nFor detailed usage, see USAGE.md")
        sys.exit(0)


if __name__ == '__main__':
    main()
