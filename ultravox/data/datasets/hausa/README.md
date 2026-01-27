# Hausa Dataset Configuration

This dataset combines:
- **Transcript/Response pairs** from local JSON files (`conversation_pairs_completed_batch_*.json`)
- **Audio files** from Hugging Face dataset (`naijavoices/naijavoices-dataset`)

## How It Works

### Data Sources

1. **JSON Files** (local): Contains transcript and response pairs
   - Located in: `ultravox/data/datasets/hausa/`
   - Files: `conversation_pairs_completed_batch_0.json`, `conversation_pairs_completed_batch_1.json`, `conversation_pairs_completed_batch_2.json`
   - **Fields**:
     - `transcript`: User input text (used for user message)
     - `response`: Assistant output text (**used as training target**)
     - `audio_path`: Audio filename (used to match with HF dataset)

2. **Audio Files** (Hugging Face): Loaded on-demand from HF
   - Dataset: `naijavoices/naijavoices-dataset`
   - Configs: `hausa-batch-0`, `hausa-batch-1`, `hausa-batch-2`
   - **Storage Format**: HF datasets are stored as **parquet files** (columnar, compressed)
   - **Audio Decoding**: Audio is automatically decoded from parquet when accessed via `Audio` feature
   - **On-Demand Loading**: Audio files are downloaded/cached and decoded only when needed

### Parquet Format Handling

Hugging Face datasets stored as parquet files work seamlessly:

```python
# Parquet files are automatically handled - no special code needed
hf_ds = load_dataset("naijavoices/naijavoices-dataset", "hausa-batch-0")

# Audio is decoded automatically from parquet
audio_obj = hf_ds[0]["audio"]  # {"path": "...", "array": [...], "sampling_rate": 16000}
```

The `HausaHFDataset` class handles parquet-backed datasets transparently - you don't need to worry about the underlying storage format.

### Matching & Filtering Process

3. **Matching & Filtering**: The system automatically:
   - Builds an index of available audio files from HF datasets (lazy loading, works with parquet)
   - **Filters during iteration/training**: Only processes JSON entries whose audio exists in HF dataset
   - Creates a mapping between JSON `audio_path` and HF dataset audio entries
   - Loads audio on-demand from parquet when samples are accessed
   - Skips samples with missing audio automatically during training/batching

## Requirements

- Hugging Face CLI login (if dataset is gated):
  ```bash
  huggingface-cli login
  ```

- The HF dataset will be automatically downloaded/cached on first use

## Usage

The dataset is automatically registered and can be used in training configs:

```yaml
train_sets:
  - name: hausa-train  # Uses 80% of combined batches

val_sets:
  - name: hausa-val     # Uses 10% of combined batches
```

## Benefits

- **No local storage needed**: Audio files are loaded from HF on-demand
- **Automatic filtering**: Only uses audio files referenced in JSON
- **Efficient**: Audio is cached by Hugging Face datasets library
- **Dynamic**: Sample counts calculated automatically

## Configuration

The dataset configuration is set in `ultravox/data/configs/hausa.py`:

### Template Configuration
- `transcript_template="{{transcript}}"` → Maps to user message
- `assistant_template="{{response}}"` → **Maps to assistant output (training target)**
- `audio_field="audio_path"` → Field containing audio filename

### Hugging Face Dataset
- `HAUSA_HF_DATASET_NAME = "naijavoices/naijavoices-dataset"`
- `HAUSA_HF_BATCH_CONFIGS = ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2"]`

## How Response Field Works

The `response` field from your JSON files is automatically used as the **assistant output** (the text the model should generate). See `HOW_RESPONSE_WORKS.md` for detailed explanation.

**Example:**
- **Input**: Audio + transcript "Na san shugaban..."
- **Target Output**: response "Lallai, waye sunansa..." ← This is what the model learns to generate

