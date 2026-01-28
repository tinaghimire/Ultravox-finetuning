import os
from ultravox.data import types

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_ULTRAVOX_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(_CONFIG_DIR))
)

# ============================================================================
# HuggingFace Dataset Configuration
# ============================================================================
HAUSA_HF_DATASET_NAME = "vaghawan/hausa-audio-resampled"
HAUSA_HF_BATCH_CONFIGS = [
    "hausa-batch-0",
    "hausa-batch-1",
    "hausa-batch-2",
]

# Dataset size limit (-1 = use all data)
HAUSA_MAX_DATASET_SIZE = -1

# HuggingFace dataset configs
HAUSA_HF_BASE_CONFIG = types.DatasetConfig(
    name="hausa-huggingface",
    path=HAUSA_HF_DATASET_NAME,
    transcript_template="{{transcript}}",
    assistant_template="{{completion}}",
    user_template_args={"transcript_language": "Hausa"},
    audio_field="audio",
    eval_config=types.EvalConfig(
        metric="wer",  # Primary metric: WER for word-level accuracy
        args={"lang_id": "ha"},  # Hausa language ID for proper normalization
        additional_metrics=[
            # Secondary metric: BLEU for n-gram overlap
            {"metric": "bleu", "args": {"tokenize": "none"}},
        ],
    ),
)

HAUSA_HF_TRAIN_CONFIG = types.DatasetConfig(
    name="hausa-huggingface-train",
    base="hausa-huggingface",
    path=HAUSA_HF_DATASET_NAME,
    splits=[
        types.DatasetSplitConfig(
            name="train",
            num_samples=-1,
            split=types.DatasetSplit.TRAIN
        ),
    ],
)

HAUSA_HF_VAL_CONFIG = types.DatasetConfig(
    name="hausa-huggingface-val",
    base="hausa-huggingface",
    path=HAUSA_HF_DATASET_NAME,
    splits=[
        types.DatasetSplitConfig(
            name="validation",
            num_samples=-1,
            split=types.DatasetSplit.VALIDATION
        ),
    ],
)

HAUSA_HF_TEST_CONFIG = types.DatasetConfig(
    name="hausa-huggingface-test",
    base="hausa-huggingface",
    path=HAUSA_HF_DATASET_NAME,
    splits=[
        types.DatasetSplitConfig(
            name="test",
            num_samples=-1,
            split=types.DatasetSplit.TEST
        ),
    ],
)

# ============================================================================
# Offline Dataset Configuration
# ============================================================================
HAUSA_OFFLINE_DATASET_DIR = os.path.join(
    _ULTRAVOX_ROOT, "ultravox", "data", "dataset"
)
HAUSA_OFFLINE_AUDIO_DIR = os.path.join(HAUSA_OFFLINE_DATASET_DIR, "audio")

# Offline dataset configs
HAUSA_OFFLINE_BASE_CONFIG = types.DatasetConfig(
    name="hausa-offline",
    path=os.path.join(HAUSA_OFFLINE_DATASET_DIR, "train.json"),
    transcript_template="{{transcript}}",
    assistant_template="{{completion}}",
    user_template_args={"transcript_language": "Hausa"},
    audio_field="audio_path",
    eval_config=types.EvalConfig(
        metric="wer",  # Primary metric: WER for word-level accuracy
        args={"lang_id": "ha"},  # Hausa language ID for proper normalization
        additional_metrics=[
            # Secondary metric: BLEU for n-gram overlap
            {"metric": "bleu", "args": {"tokenize": "none"}},
        ],
    ),
)

HAUSA_OFFLINE_TRAIN_CONFIG = types.DatasetConfig(
    name="hausa-offline-train",
    base="hausa-offline",
    path=os.path.join(HAUSA_OFFLINE_DATASET_DIR, "train.json"),
    splits=[
        types.DatasetSplitConfig(
            name="train",
            num_samples=-1,
            split=types.DatasetSplit.TRAIN
        ),
    ],
)

HAUSA_OFFLINE_VAL_CONFIG = types.DatasetConfig(
    name="hausa-offline-val",
    base="hausa-offline",
    path=os.path.join(HAUSA_OFFLINE_DATASET_DIR, "validation.json"),
    splits=[
        types.DatasetSplitConfig(
            name="validation",
            num_samples=-1,
            split=types.DatasetSplit.VALIDATION
        ),
    ],
)

HAUSA_OFFLINE_TEST_CONFIG = types.DatasetConfig(
    name="hausa-offline-test",
    base="hausa-offline",
    path=os.path.join(HAUSA_OFFLINE_DATASET_DIR, "test.json"),
    splits=[
        types.DatasetSplitConfig(
            name="test",
            num_samples=-1,
            split=types.DatasetSplit.TEST
        ),
    ],
)

configs = [
    # HuggingFace configs
    HAUSA_HF_BASE_CONFIG,
    HAUSA_HF_TRAIN_CONFIG,
    HAUSA_HF_VAL_CONFIG,
    HAUSA_HF_TEST_CONFIG,
    # Offline configs
    HAUSA_OFFLINE_BASE_CONFIG,
    HAUSA_OFFLINE_TRAIN_CONFIG,
    HAUSA_OFFLINE_VAL_CONFIG,
    HAUSA_OFFLINE_TEST_CONFIG,
]
