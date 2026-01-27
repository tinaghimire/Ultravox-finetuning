# Hausa Dataset Configuration

This dataset combines:
- **Transcript/Response pairs** from JSON files in HuggingFace dataset (`conversation_pairs_completed_batch_*.json`)
- **Audio files** from Hugging Face dataset (`naijavoices/naijavoices-dataset`)

## How It Works

### Data Sources

1. **JSON Files** (HuggingFace): Contains transcript and response pairs
   - Loaded from HuggingFace dataset
   - Files: `conversation_pairs_completed_batch_0.json`, `conversation_pairs_completed_batch_1.json`, `conversation_pairs_completed_batch_2.json`
   - **Fields**:
     - `transcript`: User input text (used for user message)
     - `response`: Assistant output text (**used as training target**)
     - `audio_path`: Audio filename (used to match with HF dataset)

2. **Audio Files** (Hugging Face): Loaded on-demand from HF
   - Dataset: `naijavoices/naijavoices-dataset` (or custom via `HAUSA_HF_DATASET_NAME`)
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

**Matching & Filtering**: The system automatically:
- Loads JSON metadata files from HuggingFace dataset
- Builds an index of available audio files from HF datasets (lazy loading, works with parquet)
- **Filters during iteration/training**: Only processes JSON entries whose audio exists in HF dataset
- Creates a mapping between JSON `audio_path` and HF dataset audio entries
- Loads audio on-demand from parquet when samples are accessed
- Skips samples with missing audio automatically during training/batching

## Requirements

- **For public datasets**: No authentication needed
- **For private datasets**: Hugging Face authentication required (choose one):
  1. **CLI login** (recommended):
     ```bash
     huggingface-cli login
     ```
  2. **Environment variable**:
     ```bash
     export HF_TOKEN="your_huggingface_token"
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

- **No local storage needed**: Both JSON metadata and audio files are loaded from HF on-demand
- **Automatic filtering**: Only uses audio files referenced in JSON
- **Efficient**: Data is cached by Hugging Face datasets library
- **Dynamic**: Sample counts calculated automatically
- **Centralized**: All data in one place (HuggingFace)

## Configuration

The dataset configuration is set in `ultravox/data/configs/hausa.py`:

### Template Configuration
- `transcript_template="{{transcript}}"` → Maps to user message
- `assistant_template="{{response}}"` → **Maps to assistant output (training target)**
- `audio_field="audio_path"` → Field containing audio filename

### Hugging Face Dataset
- `HAUSA_HF_DATASET_NAME = "naijavoices/naijavoices-dataset"` (default)
- `HAUSA_HF_BATCH_CONFIGS = ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2"]`

### Using Private HuggingFace Datasets

To use your own private HuggingFace dataset:

1. **Set the dataset name via environment variable**:
   ```bash
   export HAUSA_HF_DATASET_NAME="your-username/your-private-dataset"
   ```

2. **Authenticate with HuggingFace** (if not already done):
   ```bash
   huggingface-cli login
   # OR
   export HF_TOKEN="your_huggingface_token"
   ```

3. **Ensure your HF dataset has the correct structure**:
   - Three configs: `hausa-batch-0`, `hausa-batch-1`, `hausa-batch-2`
   - Each config should contain audio files with filenames matching the `audio_path` field in your JSON metadata files
   - Audio should be stored with the `Audio` feature type for automatic decoding

4. **JSON metadata files** - Must be uploaded to your HuggingFace dataset:
   - JSON files should be uploaded as data files to your HF dataset:
     - `conversation_pairs_completed_batch_0.json`
     - `conversation_pairs_completed_batch_1.json`
     - `conversation_pairs_completed_batch_2.json`
   - Alternatively, the dataset configs can contain `transcript` and `response` columns directly (in which case JSON files are not needed)

## How Response Field Works

The `response` field from your JSON files is automatically used as the **assistant output** (the text the model should generate). See `HOW_RESPONSE_WORKS.md` for detailed explanation.

**Example:**
- **Input**: Audio + transcript "Na san shugaban..."
- **Target Output**: response "Lallai, waye sunansa..." ← This is what the model learns to generate

