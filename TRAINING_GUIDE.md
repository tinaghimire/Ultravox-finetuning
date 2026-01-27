# Training Guide for Hausa Dataset

## Prerequisites Setup

### 1. Clone Repository
```bash
git clone https://github.com/fixie-ai/ultravox.git
cd ultravox/
```

### 2. Install System Dependencies

```bash
sudo apt upgrade
sudo apt update

# Install just
sudo apt install just
# OR
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
export PATH="$HOME/bin:$PATH"

# Install pyenv dependencies
sudo apt install xz-utils
curl https://pyenv.run | bash

# Add to ~/.bashrc
nano ~/.bashrc
# Add these lines:
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

source ~/.bashrc

# Install Python build dependencies
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

# Install Python 3.11
pyenv install 3.11
pyenv global 3.11

# Verify Python version
python --version  # Should show 3.11.x
```

### 3. Install Python Dependencies

```bash
# Install poetry
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry

# Install Ultravox dependencies
just install
```

## Dataset Setup

### 1. Verify Dataset Files

Your Hausa dataset JSON files should already be in place:
```bash
ls -la ultravox/data/datasets/hausa/*.json
```

Expected files:
- `conversation_pairs_completed_batch_0.json`
- `conversation_pairs_completed_batch_1.json`
- `conversation_pairs_completed_batch_2.json`

### 2. Login to Hugging Face

```bash
huggingface-cli login
# Enter your token when prompted
```

**Important**: This is required to:
- Download the Ultravox checkpoint (`fixie-ai/ultravox-v0_7-glm-4_6`)
- Access the audio dataset (`naijavoices/naijavoices-dataset`)

## Training Configuration

### Verify Config File

Check `ultravox/training/configs/example_config.yaml`:

```yaml
text_model: "zai-org/GLM-4.6"
audio_model: "openai/whisper-large-v3-turbo"
model_load_dir: "fixie-ai/ultravox-v0_7-glm-4_6"  # Pre-trained checkpoint

train_sets:
  - name: hausa-train
val_sets:
  - name: hausa-val
test_sets:
  - name: hausa-test

batch_size: 10
max_steps: 100

# Validation configuration
val_steps: 1.0  # Validate every epoch (when using num_epochs) or every N steps
val_batch_size: 8
do_eval: true  # Enable evaluation during training
```

### Validation Configuration

**Validate after every epoch:**

When using `num_epochs` (epoch-based training):
```yaml
num_epochs: 3
val_steps: 1.0  # Evaluates once per epoch (100% of steps = 1 epoch)
val_batch_size: 8
do_eval: true
```

When using `max_steps` (step-based training):
```yaml
max_steps: 1000
val_steps: 100  # Evaluates every 100 steps (absolute number)
# OR
val_steps: 0.1  # Evaluates every 10% of steps (0.1 * 1000 = every 100 steps)
val_batch_size: 8
do_eval: true
```

**Calculate steps per epoch** (for precise epoch-based validation):
```
steps_per_epoch = len(train_dataset) / (batch_size * grad_accum_steps * num_gpus)
val_steps: <calculated_steps_per_epoch>  # Evaluates once per epoch
```

**Validation runs on:**
- `val_sets` during training (e.g., `hausa-val`)
- `test_sets` at the end of training (e.g., `hausa-test`)

### Using Different Models

**Yes, you can use different models!** Ultravox supports:

**Text Models** (any causal LM from Hugging Face):
- Llama: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`
- GLM: `zai-org/GLM-4.6`, `THUDM/glm-4-9b-chat`
- Mistral: `mistralai/Mistral-7B-Instruct-v0.2`
- Qwen: `Qwen/Qwen2.5-7B-Instruct`
- Any model compatible with `AutoModelForCausalLM`

**Audio Models** (any audio encoder from Hugging Face):
- Whisper: `openai/whisper-large-v3-turbo`, `openai/whisper-base`
- Wav2Vec2: `facebook/wav2vec2-base-960h`
- Any model compatible with `AutoModel` that outputs embeddings

**Important Considerations:**
1. **Checkpoint compatibility**: If loading a checkpoint (`model_load_dir`), it should match your model architecture
2. **Projector dimensions**: The projector adapts to your model's embedding dimensions automatically
3. **LoRA target_modules**: Must match your model's layer names (see staged training section)
4. **From scratch**: If training from scratch (no checkpoint), you can use any compatible models

**Example with different models:**
```yaml
text_model: "meta-llama/Llama-3.1-8B-Instruct"  # Different LLM
audio_model: "openai/whisper-base"                # Smaller Whisper
# model_load_dir: null  # Train from scratch, or use compatible checkpoint
```

## Training

### Single GPU Training

```bash
poetry run python -m ultravox.training.train \
  --config_path ultravox/training/configs/example_config.yaml
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=8 \
  -m ultravox.training.train \
  --config_path ultravox/training/configs/example_config.yaml
```

Replace `8` with your number of GPUs.

## What Happens During Training

1. **Model Loading**:
   - Downloads `zai-org/GLM-4.6` (text model)
   - Downloads `openai/whisper-large-v3-turbo` (audio encoder)
   - Downloads `fixie-ai/ultravox-v0_7-glm-4_6` (checkpoint)

2. **Dataset Loading**:
   - Loads JSON files from `ultravox/data/datasets/hausa/`
   - Downloads audio files from Hugging Face (`naijavoices/naijavoices-dataset`)
   - Applies 80/10/10 train/val/test split
   - Filters to only use audio files referenced in JSON

3. **Training**:
   - Fine-tunes the projector on Hausa data
   - Audio + transcript → response
   - Saves checkpoints periodically

## Monitoring

- **Wandb**: Training metrics logged automatically (if `WANDB_API_KEY` is set)
- **Console**: Progress bars and loss values
- **Checkpoints**: Saved to `outputs/` directory

## Troubleshooting

### Dataset Not Found
```bash
# Verify dataset files exist
ls -la ultravox/data/datasets/hausa/*.json

# Check dataset registration
poetry run python -c "from ultravox.data.registry import DATASET_MAP; print('hausa-train' in DATASET_MAP)"
```

### Hugging Face Authentication Error
```bash
# Verify token is set
echo $HF_TOKEN

# Re-login
huggingface-cli login
```

### Audio Files Not Loading
- Ensure `HF_TOKEN` is set (required for `naijavoices/naijavoices-dataset`)
- Check internet connection for HF dataset access
- Audio files are downloaded on-demand, first run may be slow

### Out of Memory
- Reduce `batch_size` in config
- Reduce `max_audio_duration_secs` in config
- Use gradient accumulation

## Staged Training (Optional)

You can train components separately for better control:

### Stage 1: Projector Only (Default)
```bash
poetry run python -m ultravox.training.train \
  --config_path ultravox/training/configs/hausa_stage1_projector.yaml
```
- Trains: Multi-modal projector only
- Frozen: Audio tower + LLM
- Checkpoint: `outputs/hausa_stage1_projector/checkpoint-{step}/`

### Stage 2: Add Audio LoRA
Update checkpoint path in `hausa_stage2_audio_lora.yaml`, then:
```bash
poetry run python -m ultravox.training.train \
  --config_path ultravox/training/configs/hausa_stage2_audio_lora.yaml
```
- Trains: Projector + Audio tower (via LoRA)
- Frozen: LLM
- LoRA config: `r: 8`, `lora_alpha: 16`, target: `["k_proj", "q_proj", "v_proj", "out_proj"]`

### Stage 3: Add LLM LoRA
Update checkpoint path in `hausa_stage3_llm_lora.yaml`, then:
```bash
poetry run python -m ultravox.training.train \
  --config_path ultravox/training/configs/hausa_stage3_llm_lora.yaml
```
- Trains: All components (via LoRA)
- LoRA config: `r: 16`, `lora_alpha: 32`, target: `["q_proj", "k_proj", "v_proj", "o_proj"]`

**Note**: `target_modules` are actual layer names from Hugging Face models:
- **LLM** (GLM/Llama): `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Audio** (Whisper): `q_proj`, `k_proj`, `v_proj`, `out_proj` (note: `out_proj` not `o_proj`)

## Evaluation

After training, evaluate your model on the test set:

### Evaluation Metrics Available

For Hausa conversation dataset, you can use:

1. **BLEU Score** (default): Measures response quality
   ```yaml
   eval_config:
     metric: "bleu"
     args: {"tokenize": "none"}  # Or appropriate tokenizer for Hausa
   ```

2. **WER (Word Error Rate)**: Measures transcription/response accuracy
   ```yaml
   eval_config:
     metric: "wer"
     args: {"lang_id": "ha"}  # Hausa language code
   ```

3. **Conversation Quality** (GPT-based): Evaluates conversation flow
   ```yaml
   eval_config:
     metric: "conversation"
   ```

4. **Partial Match**: Checks if expected response is in generated response
   ```yaml
   eval_config:
     metric: "partial_match"
   ```

### Running Evaluation

**Option 1: Using eval config file** (evaluates both val and test)
```bash
# Update model path in eval_config_hausa.yaml first!
poetry run python -m ultravox.evaluation.eval \
  --config_path ultravox/evaluation/configs/eval_config_hausa.yaml
```
This evaluates on both `hausa-val` and `hausa-test` datasets.

**Option 2: Evaluation during training**
- **Validation**: Runs on `val_sets` (e.g., `hausa-val`) based on `val_steps` configuration
- **Test**: Runs on `test_sets` (e.g., `hausa-test`) at the end of training
- Both happen automatically if `do_eval: true` (default)

**Option 3: Standalone evaluation**
```bash
# Evaluate on validation set
poetry run python -m ultravox.evaluation.eval \
  --model "outputs/hausa_stage1_projector/checkpoint-1000" \
  --eval_sets hausa-val \
  --eval_batch_size 8

# Evaluate on test set
poetry run python -m ultravox.evaluation.eval \
  --model "outputs/hausa_stage1_projector/checkpoint-1000" \
  --eval_sets hausa-test \
  --eval_batch_size 8
```

### Evaluation Config

Created `ultravox/evaluation/configs/eval_config_hausa.yaml`:
- Evaluates on `hausa-test` dataset
- Uses BLEU metric (can be changed)
- Update `model` path to your trained checkpoint

**Note**: The Hausa dataset config already includes `eval_config` with BLEU metric. You can modify it in `ultravox/data/configs/hausa.py` to use different metrics.

## Next Steps After Training

1. **Evaluate**: Test on `hausa-test` dataset (see above)
2. **Inference**: Use trained model for Hausa voice interactions
3. **Save**: Model checkpoints saved in `outputs/` directory

