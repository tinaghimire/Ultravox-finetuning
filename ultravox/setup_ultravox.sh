#!/bin/bash

# Ultravox Finetuning Setup Script
# This script automates the setup process for Ultravox finetuning environment
#
# Usage: ./setup_ultravox.sh [OPTIONS]
#   -c, --config CONFIG_PATH    Path to training config file (default: ultravox/training/configs/hausa_stage1_projector.yaml)
#   -g, --gpus NUM_GPUS        Number of GPUs to use (default: 4)
#   -h, --help                 Show this help message

set -e  # Exit on error

# Default values
DEFAULT_CONFIG="ultravox/training/configs/hausa_stage1_projector.yaml"
DEFAULT_GPUS=4

# Parse command line arguments
CONFIG_PATH="$DEFAULT_CONFIG"
NUM_GPUS="$DEFAULT_GPUS"

show_help() {
    echo "Ultravox Finetuning Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config CONFIG_PATH    Path to training config file"
    echo "                              (default: $DEFAULT_CONFIG)"
    echo "  -g, --gpus NUM_GPUS        Number of GPUs to use (default: $DEFAULT_GPUS)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --config ultravox/training/configs/my_config.yaml --gpus 8"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Validate NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    print_error "Number of GPUs must be a positive integer, got: $NUM_GPUS"
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root for sudo commands
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        print_warning "This script requires sudo access. You may be prompted for your password."
    fi
}

# Main setup process
main() {
    print_status "Starting Ultravox Finetuning setup..."
    echo ""
    echo "Configuration:"
    echo "  Config file: $CONFIG_PATH"
    echo "  Number of GPUs: $NUM_GPUS"
    echo ""
    
    # Check sudo access
    check_sudo
    
    # Update system packages
    print_status "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    
    # Clone repository
    if [ ! -d "Ultravox-finetuning" ]; then
        print_status "Cloning Ultravox-finetuning repository..."
        git clone https://github.com/tinaghimire/Ultravox-finetuning.git
    else
        print_warning "Ultravox-finetuning directory already exists, skipping clone..."
    fi
    
    cd Ultravox-finetuning/
    
    # Install just
    print_status "Installing 'just' command runner..."
    if ! command -v just &> /dev/null; then
        if sudo apt install just -y; then
            print_status "'just' installed successfully via apt"
        else
            print_warning "apt installation failed, trying alternative method..."
            curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
            export PATH="$HOME/bin:$PATH"
            
            # Add to bashrc if not already present
            if ! grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc; then
                echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
                source ~/.bashrc
            fi
        fi
    else
        print_status "'just' is already installed"
    fi
    
    # Install xz-utils
    print_status "Installing xz-utils..."
    sudo apt install xz-utils -y
    
    # Install pyenv dependencies
    print_status "Installing pyenv dependencies..."
    sudo apt install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        tk-dev
    
    # Install pyenv
    if [ ! -d "$HOME/.pyenv" ]; then
        print_status "Installing pyenv..."
        curl https://pyenv.run | bash
        
        # Add pyenv to bashrc if not already present
        if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
            print_status "Adding pyenv configuration to ~/.bashrc..."
            cat >> ~/.bashrc << 'EOF'

# Pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
EOF
        fi
        
        # Reload bashrc to pick up pyenv configuration
        print_status "Reloading shell configuration..."
        source ~/.bashrc
        
        # Load pyenv in current shell (backup in case source didn't work)
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    else
        print_status "pyenv is already installed"
        # Reload bashrc to ensure pyenv is loaded
        source ~/.bashrc
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
    
    # Install Python 3.11
    if ! pyenv versions | grep -q "3.11"; then
        print_status "Installing Python 3.11 via pyenv..."
        pyenv install 3.11
    else
        print_status "Python 3.11 is already installed"
    fi
    
    print_status "Setting Python 3.11 as global version..."
    pyenv global 3.11
    
    # Setup virtual environment and install poetry
    print_status "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing poetry..."
    pip install poetry
    
    # Configure poetry
    export POETRY_VIRTUALENVS_USE_SYSTEM_GIT_CLIENT=true
    
    print_status "Installing project dependencies with poetry..."
    poetry add wandb
    
    if poetry install; then
        print_status "Poetry dependencies installed successfully"
    else
        print_warning "Poetry install encountered issues, attempting to install evals separately..."
        poetry run pip install git+https://github.com/openai/evals.git || {
            print_warning "Evals installation may have issues, but continuing..."
        }
    fi
    
    # Setup HuggingFace
    print_status "Setting up HuggingFace..."
    echo ""
    echo "================================================"
    echo "Please login to HuggingFace"
    echo "You'll need your HuggingFace access token"
    echo "================================================"
    huggingface-cli login
    
    # Setup Weights & Biases
    print_status "Setting up Weights & Biases..."
    export WANDB_PROJECT="ultravox-hausa"
    
    echo ""
    echo "================================================"
    echo "Please enter your Weights & Biases API key:"
    echo "================================================"
    read -r WANDB_API_KEY
    export WANDB_API_KEY
    
    print_status "Logging into wandb..."
    wandb login
    
    # Add environment variables to a local config file
    print_status "Creating environment configuration file..."
    cat > .env << EOF
export WANDB_PROJECT="ultravox-hausa"
export WANDB_API_KEY="${WANDB_API_KEY}"
EOF
    
    # Optional: Prefetch weights and start training
    echo ""
    echo "================================================"
    echo "Would you like to start training now? (y/n)"
    echo "Note: This will prefetch the model weights and start training"
    echo "Config file: $CONFIG_PATH"
    echo "Number of GPUs: $NUM_GPUS"
    echo "================================================"
    read -r START_TRAINING
    
    if [[ "$START_TRAINING" =~ ^[Yy]$ ]]; then
        print_status "Starting training process..."
        
        # Verify config file exists
        if [ ! -f "$CONFIG_PATH" ]; then
            print_error "Config file not found: $CONFIG_PATH"
            exit 1
        fi
        
        # Step 1: Prefetch weights
        print_status "Prefetching model weights..."
        TRAIN_ARGS="--config_path $CONFIG_PATH"
        echo "Running: poetry run python -m ultravox.training.helpers.prefetch_weights $TRAIN_ARGS"
        poetry run python -m ultravox.training.helpers.prefetch_weights $TRAIN_ARGS
        
        # Step 2: Run training
        print_status "Starting training with $NUM_GPUS GPUs..."
        echo "Running: poetry run torchrun --nproc_per_node=$NUM_GPUS -m ultravox.training.train --config_path $CONFIG_PATH"
        poetry run torchrun --nproc_per_node=$NUM_GPUS \
            -m ultravox.training.train \
            --config_path $CONFIG_PATH
    else
        print_status "Skipping training. You can run it later with:"
        echo ""
        echo "  cd Ultravox-finetuning"
        echo "  source venv/bin/activate"
        echo "  source .env"
        echo "  TRAIN_ARGS=\"--config_path $CONFIG_PATH\""
        echo "  poetry run python -m ultravox.training.helpers.prefetch_weights \$TRAIN_ARGS"
        echo "  poetry run torchrun --nproc_per_node=$NUM_GPUS -m ultravox.training.train --config_path $CONFIG_PATH"
        echo ""
    fi
    
    print_status "Setup complete!"
    echo ""
    echo "================================================"
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo "================================================"
    echo ""
    echo "To activate the environment in the future, run:"
    echo "  cd Ultravox-finetuning"
    echo "  source venv/bin/activate"
    echo "  source .env"
    echo ""
    echo "To run this script with custom settings:"
    echo "  ./setup_ultravox.sh --config path/to/config.yaml --gpus 8"
    echo "  ./setup_ultravox.sh --help    # Show all options"
    echo ""
    echo "Note: If pyenv or just commands are not found in new terminals,"
    echo "      restart your shell or run 'source ~/.bashrc'"
    echo "================================================"
    
}

# Run main function
main "$@"
