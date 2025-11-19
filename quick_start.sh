#!/bin/bash
#
# ChessBot Byte - Quick Start Script
#
# This script helps you get started with ChessBot Byte quickly.
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ChessBot Byte - Quick Start                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}➜${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Step 1: Check Python
print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo ""

# Step 2: Check/Install dependencies
print_step "Checking dependencies..."
if python3 -c "import torch" 2>/dev/null; then
    print_success "PyTorch is installed"
else
    print_warning "PyTorch not found. Installing dependencies..."
    pip install -r requirements.txt
fi
echo ""

# Step 3: Verify setup
print_step "Verifying setup..."
if [ -f "verify_setup.py" ]; then
    python3 verify_setup.py
    if [ $? -ne 0 ]; then
        print_error "Setup verification failed. Please fix issues above."
        exit 1
    fi
else
    print_warning "verify_setup.py not found, skipping verification"
fi
echo ""

# Step 4: Setup project
print_step "Setting up project directories..."
python3 cli.py setup
echo ""

# Step 5: Check for data
print_step "Checking for training data..."
if [ -d "data/train" ] && [ "$(ls -A data/train)" ]; then
    print_success "Training data found"
else
    print_warning "Training data not found"
    echo ""
    echo "To download training data, run:"
    echo "  bash download_data.sh"
    echo ""
    echo "Or set a custom data directory:"
    echo "  export CHESSBOT_DATA_DIR=/path/to/your/data"
    echo ""
fi

# Step 6: Show next steps
echo "════════════════════════════════════════════════════════════════════════════"
echo "Setup Complete! 🎉"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next Steps:"
echo ""
echo "  1. View project info:"
echo "     python3 cli.py info"
echo ""
echo "  2. Train a model (quick test):"
echo "     python3 cli.py train --epochs 2 --data-files 1"
echo ""
echo "  3. Train a model (full):"
echo "     python3 cli.py train --epochs 20 --data-files 100"
echo ""
echo "  4. Evaluate the model:"
echo "     python3 cli.py evaluate"
echo ""
echo "  5. Use interactive mode:"
echo "     python3 cli.py infer --interactive"
echo ""
echo "For detailed usage instructions, see:"
echo "  - README.md: Project overview"
echo "  - USAGE.md: Detailed usage guide"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
