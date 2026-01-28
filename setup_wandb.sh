#!/bin/bash
# WandB Setup Script for Ultravox Hausa Training

# Set WandB API key (get from https://wandb.ai/authorize)
# Or use: export WANDB_API_KEY="your-key-here"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set. Please export it first:"
    echo "  export WANDB_API_KEY='your-api-key-here'"
    echo ""
    echo "Or run: wandb login"
    exit 1
fi

# Set project name
export WANDB_PROJECT="ultravox-hausa"

# Login to WandB (if not already logged in)
wandb login $WANDB_API_KEY

# Verify setup
echo ""
echo "=========================================="
echo "WandB Setup Complete"
echo "=========================================="
echo "Project: $WANDB_PROJECT"
echo "API Key: ${WANDB_API_KEY:0:10}..."
echo ""
echo "To use in training, run:"
echo "  export WANDB_PROJECT='ultravox-hausa'"
echo "  export WANDB_API_KEY='your-key-here'"
echo "  torchrun --nproc_per_node=4 python -m ultravox.training.train --config_path ultravox/training/configs/hausa_stage1_projector.yaml"
echo "=========================================="
