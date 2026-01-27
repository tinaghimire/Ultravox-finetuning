import os
from ultravox.data import types

# Get the directory where this config file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to dataset files within ultravox directory
_ULTRAVOX_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CONFIG_DIR)))
HAUSA_DATASET_DIR = os.path.join(_ULTRAVOX_ROOT, "ultravox", "data", "datasets", "hausa")

# Hugging Face dataset configuration for audio files
HAUSA_HF_DATASET_NAME = "naijavoices/naijavoices-dataset"
HAUSA_HF_BATCH_CONFIGS = ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2"]

# Individual batch file paths (now in ultravox directory)
HAUSA_BATCH_0_PATH = os.path.join(HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_0.json")
HAUSA_BATCH_1_PATH = os.path.join(HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_1.json")
HAUSA_BATCH_2_PATH = os.path.join(HAUSA_DATASET_DIR, "conversation_pairs_completed_batch_2.json")

# Combined path for all batches (comma-separated)
HAUSA_ALL_BATCHES_PATH = f"{HAUSA_BATCH_0_PATH},{HAUSA_BATCH_1_PATH},{HAUSA_BATCH_2_PATH}"

# Base config for Hausa conversation dataset
HAUSA_BASE_CONFIG = types.DatasetConfig(
    name="hausa",
    path=HAUSA_ALL_BATCHES_PATH,  # All batches combined
    transcript_template="{{transcript}}",
    assistant_template="{{response}}",
    user_template_args={"transcript_language": "Hausa"},
    audio_field="audio_path",  # Field name containing audio file path
    # Evaluation config: Use BLEU and WER for Hausa conversation responses
    eval_config=types.EvalConfig(
        metric="bleu",  # BLEU score for response quality
        args={"tokenize": "none"},  # No tokenization for Hausa (or use appropriate tokenizer)
    ),
)

# Combined config with 80/10/10 split across all batches
# num_samples will be calculated dynamically based on actual dataset size
HAUSA_TRAIN_CONFIG = types.DatasetConfig(
    name="hausa-train",
    base="hausa",
    path=HAUSA_ALL_BATCHES_PATH,
    splits=[
        types.DatasetSplitConfig(
            name="train", 
            num_samples=50,  # Will be calculated dynamically (80% of total)
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
            num_samples=10,  # Will be calculated dynamically (10% of total)
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
            num_samples=10,  # Will be calculated dynamically (10% of total)
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

