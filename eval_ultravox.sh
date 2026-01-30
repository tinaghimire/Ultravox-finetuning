#!/bin/bash

# Ultravox Evaluation Setup Script
# This script automates the setup process for Ultravox evaluation environment
#
# Prerequisites:
#   - Clone the repository first: git clone https://github.com/tinaghimire/Ultravox-finetuning.git
#   - Run this script from the parent directory or inside Ultravox-finetuning/
#
# Usage: ./eval_ultravox.sh [OPTIONS]
#   -c, --config CONFIG_PATH    Path to evaluation config file (default: ultravox/evaluation/configs/eval_config_hausa.yaml)
#   -g, --gpus NUM_GPUS        Number of GPUs to use (default: 1)
#   -h, --help                 Show this help message

set -e  # Exit on error

# Default values
DEFAULT_CONFIG="ultravox/evaluation/configs/eval_config_hausa.yaml"
DEFAULT_GPUS=1

# Parse command line arguments
CONFIG_PATH="$DEFAULT_CONFIG"
NUM_GPUS="$DEFAULT_GPUS"

show_help() {
    echo "Ultravox Evaluation Setup Script"
    echo ""
    echo "Prerequisites:"
    echo "  Clone the repository first:"
    echo "    git clone https://github.com/tinaghimire/Ultravox-finetuning.git"
    echo "  Then run this script from the parent directory or inside Ultravox-finetuning/"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config CONFIG_PATH    Path to evaluation config file"
    echo "                              (default: $DEFAULT_CONFIG)"
    echo "  -g, --gpus NUM_GPUS        Number of GPUs to use (default: $DEFAULT_GPUS)"
    echo "                              Use 2+ GPUs for tensor parallelism with large models (70B)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --config ultravox/evaluation/configs/eval_config_hausa.yaml --gpus 1"
    echo "  $0 --config ultravox/evaluation/configs/eval_config_hausa.yaml --gpus 2"
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
    print_status "Starting Ultravox Evaluation setup..."
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
    
    # Navigate to Ultravox-finetuning directory
    # Check if we're already in the directory
    if [[ "$(basename "$PWD")" == "Ultravox-finetuning" ]]; then
        print_status "Already in Ultravox-finetuning directory"
    elif [ -d "Ultravox-finetuning" ]; then
        print_status "Found Ultravox-finetuning directory, navigating to it..."
        cd Ultravox-finetuning/
    else
        print_error "Ultravox-finetuning directory not found!"
        print_error "Please clone the repository first:"
        print_error "  git clone https://github.com/tinaghimire/Ultravox-finetuning.git"
        print_error "Or run this script from inside the Ultravox-finetuning directory"
        exit 1
    fi
    
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
    
    # Setup environment variables from .env file
    print_status "Setting up environment variables..."
    
    # Create .env from .env.example if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_status "Creating .env file from .env.example..."
            cp .env.example .env
            print_warning "Please edit .env file and add your HuggingFace token and WANDB API key"
            print_warning "  - HF_TOKEN: Get from https://huggingface.co/settings/tokens"
            print_warning "  - WANDB_API_KEY: Get from https://wandb.ai/authorize"
            echo ""
            echo "Press Enter after you've edited .env file..."
            read -r
        else
            print_error ".env.example file not found!"
            exit 1
        fi
    else
        print_status ".env file already exists, using existing values"
    fi
    
    # Source .env file to load variables
    if [ -f ".env" ]; then
        # Read .env file and export variables (handles both export and non-export formats)
        set -a
        source .env
        set +a
    else
        print_error ".env file not found!"
        exit 1
    fi
    
    # Validate required environment variables
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "your_huggingface_token_here" ]; then
        print_error "HF_TOKEN is not set in .env file!"
        print_error "Please edit .env and add your HuggingFace token"
        exit 1
    fi
    
    if [ -z "$WANDB_API_KEY" ] || [ "$WANDB_API_KEY" == "your_wandb_api_key_here" ]; then
        print_error "WANDB_API_KEY is not set in .env file!"
        print_error "Please edit .env and add your WANDB API key"
        exit 1
    fi
    
    # Setup HuggingFace
    print_status "Setting up HuggingFace..."
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    # Login using the token
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    
    # Setup Weights & Biases
    print_status "Setting up Weights & Biases..."
    if [ -z "$WANDB_PROJECT" ]; then
        export WANDB_PROJECT="ultravox-hausa"
    fi
    export WANDB_API_KEY
    
    print_status "Logging into wandb..."
    echo "$WANDB_API_KEY" | wandb login --relogin
    
    # Optional: Run evaluation
    echo ""
    echo "================================================"
    echo "Would you like to start evaluation now? (y/n)"
    echo "Note: This will run evaluation on the test set"
    echo "Config file: $CONFIG_PATH"
    echo "Number of GPUs: $NUM_GPUS"
    echo "================================================"
    read -r START_EVAL
    
    if [[ "$START_EVAL" =~ ^[Yy]$ ]]; then
        print_status "Starting evaluation process..."
        
        # Verify config file exists
        if [ ! -f "$CONFIG_PATH" ]; then
            print_error "Config file not found: $CONFIG_PATH"
            exit 1
        fi
        
        # Run evaluation - use torchrun for multi-GPU, regular python for single GPU
        if [ "$NUM_GPUS" -gt 1 ]; then
            print_status "Running evaluation with $NUM_GPUS GPUs (tensor parallelism)..."
            echo "Running: poetry run torchrun --nproc_per_node=$NUM_GPUS -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
            poetry run torchrun --nproc_per_node=$NUM_GPUS \
                -m ultravox.evaluation.eval \
                --config_path $CONFIG_PATH
        else
            print_status "Running evaluation with single GPU..."
            echo "Running: poetry run python -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
            poetry run python -m ultravox.evaluation.eval --config_path $CONFIG_PATH
        fi
    else
        print_status "Skipping evaluation. You can run it later with:"
        echo ""
        echo "  cd Ultravox-finetuning"
        echo "  source venv/bin/activate"
        echo "  set -a && source .env && set +a"
        if [ "$NUM_GPUS" -gt 1 ]; then
            echo "  poetry run torchrun --nproc_per_node=$NUM_GPUS -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
        else
            echo "  poetry run python -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
        fi
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
    echo "  set -a && source .env && set +a"
    echo ""
    echo "To run evaluation manually:"
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "  # Multi-GPU with tensor parallelism:"
        echo "  poetry run torchrun --nproc_per_node=$NUM_GPUS -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
    else
        echo "  # Single GPU:"
        echo "  poetry run python -m ultravox.evaluation.eval --config_path $CONFIG_PATH"
    fi
    echo ""
    echo "To run this script with custom settings:"
    echo "  ./eval_ultravox.sh --config path/to/eval_config.yaml --gpus 2"
    echo "  ./eval_ultravox.sh --help    # Show all options"
    echo ""
    echo "Note: If pyenv or just commands are not found in new terminals,"
    echo "      restart your shell or run 'source ~/.bashrc'"
    echo "================================================"
    
}

# Run main function
main "$@"
