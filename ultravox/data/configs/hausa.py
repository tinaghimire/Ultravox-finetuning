import os
from ultravox.data import types

# Get the directory where this config file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to dataset files within ultravox directory
_ULTRAVOX_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(_CONFIG_DIR))
)

# Hugging Face dataset configuration for audio files
# Can be overridden via environment variable HAUSA_HF_DATASET_NAME
# Example: export HAUSA_HF_DATASET_NAME="your-username/your-dataset"
HAUSA_HF_DATASET_NAME = "bataju/hausa-audio-dataset-large-v0"

# Batch configs - set to None or empty list if dataset has no batch structure
# If None, audio files are loaded directly from dataset root
HAUSA_HF_BATCH_CONFIGS = None  # No batch structure - load from dataset root
# Alternative with batches:
# HAUSA_HF_BATCH_CONFIGS = [
#     "hausa-batch-0", "hausa-batch-1", "hausa-batch-2"
# ]

# JSON metadata file paths - loaded from HuggingFace with local fallback
# Paths are set to special markers that trigger HF loading in HausaHFDataset
# If not found in HF, falls back to local files in
# ultravox/data/datasets/hausa/
HAUSA_DATASET_DIR = os.path.join(
    _ULTRAVOX_ROOT, "ultravox", "data", "datasets", "hausa"
)
HAUSA_BATCH_0_PATH = "hf://conversation_pairs_completed_batch_0.json"
HAUSA_BATCH_1_PATH = "hf://conversation_pairs_completed_batch_1.json"
HAUSA_BATCH_2_PATH = "hf://conversation_pairs_completed_batch_2.json"

# Local fallback paths (used if JSON not found in HuggingFace)
HAUSA_LOCAL_BATCH_0_PATH = os.path.join(
    HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_0.json"
)
HAUSA_LOCAL_BATCH_1_PATH = os.path.join(
    HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_1.json"
)
HAUSA_LOCAL_BATCH_2_PATH = os.path.join(
    HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_2.json"
)

# Combined path for all batches (comma-separated)
HAUSA_ALL_BATCHES_PATH = (
    f"{HAUSA_BATCH_0_PATH},{HAUSA_BATCH_1_PATH},{HAUSA_BATCH_2_PATH}"
)

# Maximum dataset size to use (applied before splitting)
# Set to -1 to use all available data (default), or specify a number to limit
# This limit is applied via VoiceDatasetArgs.max_samples when creating
# the dataset. If max_samples is -1 (default), uses all available data.
# TODO-HAUSA: Example: To use only 10000 samples total, set HAUSA_MAX_DATASET_SIZE=10000
HAUSA_MAX_DATASET_SIZE = -1  # Use all available data by default

# Base config for Hausa conversation dataset
# Note: Dataset size limiting is controlled via VoiceDatasetArgs.max_samples
# The dataset will be limited to max_samples (if > 0) before splitting
# into train/val/test
HAUSA_BASE_CONFIG = types.DatasetConfig(
    name="hausa",
    path=HAUSA_ALL_BATCHES_PATH,  # All batches combined
    transcript_template="{{transcript}}",
    assistant_template="{{response}}",
    user_template_args={"transcript_language": "Hausa"},
    audio_field="audio_path",  # Field name containing audio file path
    # Evaluation config: Use BLEU and WER for Hausa conversation
    # responses
    eval_config=types.EvalConfig(
        metric="bleu",  # BLEU score for response quality
        # No tokenization for Hausa (or use appropriate tokenizer)
        args={"tokenize": "none"},
    ),
)

# Combined config with 90/5/5 split across all batches
# Train gets 90% of data, val and test are 5% each
# Dataset loads completely first (or limited by max_samples if set),
# then splits are applied
HAUSA_TRAIN_CONFIG = types.DatasetConfig(
    name="hausa-train",
    base="hausa",
    path=HAUSA_ALL_BATCHES_PATH,
    splits=[
        types.DatasetSplitConfig(
            name="train",
            num_samples=-1,  # Fixed: extract exactly 2000 samples
            split=types.DatasetSplit.TRAIN
        ),
    ],
)

HAUSA_VAL_CONFIG = types.DatasetConfig(
    name="hausa-val",
    base="hausa",
    path=HAUSA_ALL_BATCHES_PATH,
    splits=[
        types.DatasetSplitConfig(
            name="validation",
            num_samples=-1,  # Will be calculated dynamically (5% of total)
            split=types.DatasetSplit.VALIDATION
        ),
    ],
)

HAUSA_TEST_CONFIG = types.DatasetConfig(
    name="hausa-test",
    base="hausa",
    path=HAUSA_ALL_BATCHES_PATH,
    splits=[
        types.DatasetSplitConfig(
            name="test",
            num_samples=-1,  # Will be calculated dynamically (5% of total)
            split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    HAUSA_BASE_CONFIG,
    HAUSA_TRAIN_CONFIG,
    HAUSA_VAL_CONFIG,
    HAUSA_TEST_CONFIG,
]
