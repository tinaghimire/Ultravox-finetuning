import abc
import logging
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import jinja2
import numpy as np
import streaming as mds
import transformers
from torch.utils import data

from ultravox.data import data_sample
from ultravox.data import text_proc
from ultravox.data import types

# TODO(juberti): set these in the environment so they don't need to be hard-coded here.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "fixie-training"

# Silence the spurious warnings coming from the MosaicML streaming library.
logging.getLogger("streaming.base.dataset").setLevel(logging.ERROR)


def _get_messages(
    user_message: str,
    assistant_message: str,
    message_history: Optional[List[Dict[str, str]]] = None,
    sys_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert a user message, assistant message, and optional message history into a list of messages.
    If `sys_prompt` is set, it is prepended as a system message.
    """
    messages = []

    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})

    if message_history is not None:
        # For now, we only support chat history with user and assistant messages.
        assert all("role" in msg and "content" in msg for msg in message_history)
        assert all(msg["role"] in ["user", "assistant"] for msg in message_history)
        messages.extend(message_history)

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})

    return messages


def _get_worker_info(length: int):
    """
    Calculate number of samples for this worker, accounting for max workers limit.
    Returns 0 if worker_id exceeds max allowed workers.
    """
    worker_id = 0
    num_workers = 1
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    # Calculate samples for this worker
    worker_samples = length // num_workers
    extra_samples = length % num_workers

    # Workers with id < extra_samples get one extra sample
    if worker_id < extra_samples:
        worker_samples += 1

    return num_workers, worker_id, worker_samples


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An interface for an IterableDataset that provides a length method.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._name = "[unset]"
        self._length = -1

    # num_samples is the total number of samples in the dataset
    def _init_dataset(
        self,
        dataset: data.Dataset,
        name: str,
        num_samples: int,
    ) -> None:
        self._dataset = dataset
        self._name = name
        self._length = num_samples

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
    ) -> data.Dataset:
        # HF datasets sometimes fails to download due to network issues, so retry a few times.
        dataset = hf_datasets.load_dataset(
            path,
            name,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
        if audio_field is not None:
            dataset = dataset.cast_column(
                audio_field, hf_datasets.Audio(sampling_rate=data_sample.SAMPLE_RATE)
            )
        if self._args.shuffle:
            if streaming:
                dataset = dataset.shuffle(
                    seed=self._args.shuffle_seed,
                    buffer_size=self._args.shuffle_buffer_size,
                )
            else:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset

    def _load_mds_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        batch_size: int = 1,
    ) -> data.Dataset:
        gcs_path = path.replace("/", "_")
        if name:
            gcs_path += f"/{name}"
        if split:
            gcs_path += f"/{split}"
        url = f"gs://fixie-datasets/mds/{gcs_path}"
        temp_dir = os.path.join(
            tempfile.gettempdir(), f"mds_{gcs_path.replace('/', '_')}"
        )
        return mds.StreamingDataset(
            remote=url,
            local=temp_dir,
            batch_size=batch_size,
            shuffle=self._args.shuffle,
            shuffle_seed=self._args.shuffle_seed,
        )

    def __iter__(self):
        num_workers, _, _ = _get_worker_info(self._length)
        if num_workers > 1:
            assert hasattr(
                self._dataset, "n_shards"
            ), f"{self._name} does not have n_shards attribute, which is required when num_workers ({num_workers}) > 1"
            assert (
                self._dataset.n_shards >= num_workers
            ), f"{self._name} has {self._dataset.n_shards} shards, which is less than the number of workers ({num_workers})."

        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self.name} for row {row}")
                bad_samples += 1
                continue

            input_characters = sum(len(msg["content"]) for msg in sample.messages)
            if (
                self._args.max_input_characters is not None
                and input_characters > self._args.max_input_characters
            ):
                print(
                    f"Sample has input characters longer than {self._args.max_input_characters} in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            elif len(sample.messages[-1]["content"]) == 0:
                print(
                    f"Sample has empty assistant message in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            if self._args.include_audio:
                if sample.audio is None:
                    print(f"Audio is None for sample {sample}")
                    bad_samples += 1
                    continue
                if sample.audio.shape[-1] == 0:
                    print(f"Audio length is 0 for sample {sample}")
                    bad_samples += 1
                    continue
                if (
                    self._args.max_audio_duration_secs > 0
                    and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    skipped_samples += 1
                    continue

            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), removed {bad_samples} bad samples, and skipped {skipped_samples} samples for exceeding max audio duration ({self._args.max_audio_duration_secs}s)."
        )

    @abc.abstractmethod
    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = "audio"
    ) -> np.ndarray:
        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.
        # For MDS, this object is flattened into audio_array and audio_sampling_rate fields.
        if column_name in row:
            audio = row[column_name]["array"]
            sampling_rate = row[column_name]["sampling_rate"]
        elif f"{column_name}_array" in row:
            audio = row[f"{column_name}_array"]
            sampling_rate = row[f"{column_name}_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == data_sample.SAMPLE_RATE
        return audio

    def _make_messages(
        self, user_content: str, assistant_content: str
    ) -> List[Dict[str, str]]:
        return _get_messages(user_content, assistant_content)

    def _make_sample(
        self,
        messages: List[Dict[str, str]],
        audio: Optional[np.ndarray] = None,
        audio_transcript: Optional[str] = None,
        label: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> data_sample.VoiceSample:
        if not self._args.include_audio:
            return data_sample.VoiceSample(
                messages,
                label=label,
                extra_kwargs=extra_kwargs,
            )
        return data_sample.VoiceSample(
            messages,
            audio,
            audio_transcript=audio_transcript,
            label=label,
            extra_kwargs=extra_kwargs,
        )


class GenericDataset(VoiceDataset):
    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
    ) -> None:
        assert config.splits is not None
        assert config.path is not None
        assert config.mds_batch_size is not None
        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split == self._args.split:
                if not config.use_mds:
                    ds = self._load_hf_dataset(
                        config.path,
                        config.subset,
                        split=split.name,
                        audio_field=config.audio_field,
                    )
                else:
                    ds = self._load_mds_dataset(
                        config.path,
                        name=config.subset,
                        split=split.name,
                        batch_size=config.mds_batch_size,
                    )
                dsets.append(ds)
                total_samples += split.num_samples
        assert (
            len(dsets) > 0
        ), f"The {config.name} dataset has no {self._args.split} splits."
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)

        dataset_name = f"{config.name}.{self._args.split.value}"

        if self._config.messages_direct_column is None:
            assert self._config.transcript_template is not None
            assert self._config.user_template is not None
            assert self._config.user_template_args is not None
            assert (
                self._config.message_history_roles is not None
                if self._config.message_history_column is not None
                else True
            ), "message_history_roles must be provided if message_history_column is provided"
            assert self._config.assistant_template is not None

        super()._init_dataset(dataset, dataset_name, total_samples)

    def __str__(self):
        return f"GenericDataset({self._config})"

    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:

        # Setting up extra_kwargs for datasets like Voicebench
        extra_kwargs = None
        if (
            self._config.eval_config is not None
            and self._config.eval_config.extra_kwargs_map is not None
        ):
            extra_kwargs = {
                key: row.get(value)
                for key, value in self._config.eval_config.extra_kwargs_map.items()
            }

        # If the messages_direct_column is provided, we use it directly to create the messages and transcript.
        if self._config.messages_direct_column is not None:
            messages = row[self._config.messages_direct_column]
            if len(messages) == 0:
                raise ValueError("messages_direct_column is empty")

            label = (
                row[self._config.label_column]
                if self._config.label_column is not None
                else None
            )

            if not self._args.include_audio:
                return self._make_sample(
                    messages,
                    label=label,
                    extra_kwargs=extra_kwargs,
                )

            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            audio = self._get_audio(row, self._config.audio_field)
            return self._make_sample(
                messages,
                audio,
                audio_transcript=transcript,
                extra_kwargs=extra_kwargs,
            )

        # Convert the dataset's message_history_column into a list of messages
        message_history = (
            text_proc.format_message_history(
                row[self._config.message_history_column],
                self._config.message_history_roles,
            )
            if self._config.message_history_column is not None
            and self._config.message_history_roles is not None
            and not self._args.ignore_message_history
            else None
        )

        try:
            user_content = jinja2.Template(
                self._config.user_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(
                **row,
                text_proc=text_proc,
                **self._config.user_template_args,  # type: ignore[arg-type]
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            system_prompt = (
                jinja2.Template(
                    self._config.system_prompt_template,  # type: ignore[arg-type]
                    undefined=jinja2.StrictUndefined,
                ).render(**row, text_proc=text_proc)
                if self._config.system_prompt_template is not None
                and not self._args.ignore_system_prompt
                else None
            )

        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"system_prompt_template: {self._config.system_prompt_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e
        if not self._args.include_audio:
            user_content = user_content.replace(
                types.AUDIO_PLACEHOLDER, f'"{transcript}"'
            )

        messages = _get_messages(
            user_content,
            assistant_content,
            message_history=message_history,
            sys_prompt=system_prompt,
        )
        audio: Optional[np.ndarray] = (  # type: ignore[no-redef]
            self._get_audio(row, self._config.audio_field)
            if self._args.include_audio
            else None
        )
        return self._make_sample(
            messages,
            audio,
            audio_transcript=transcript,
            extra_kwargs=extra_kwargs,
        )

    def get_config(self):
        return self._config


class HausaHFDataset(VoiceDataset):
    """
    Dataset class for Hausa dataset that loads:
    - Transcript/response pairs from local JSON files
    - Audio files from Hugging Face dataset (naijavoices/naijavoices-dataset)
    
    Only includes audio files whose audio_path matches entries in the JSON files.
    """
    
    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
        json_file_path: str,
        hf_dataset_name: str,
        hf_batch_configs: List[str],
    ) -> None:
        """
        Args:
            args: VoiceDatasetArgs for dataset configuration
            config: DatasetConfig with templates and settings
            json_file_path: Path to the local JSON file(s) (comma-separated)
            hf_dataset_name: Hugging Face dataset name (e.g., "naijavoices/naijavoices-dataset")
            hf_batch_configs: List of HF dataset configs (e.g., ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2"])
        """
        super().__init__(args)
        self._config = config
        self._hf_dataset_name = hf_dataset_name
        self._hf_batch_configs = hf_batch_configs
        
        # Handle multiple JSON files
        if isinstance(json_file_path, str):
            if ',' in json_file_path:
                json_file_paths = [p.strip() for p in json_file_path.split(',')]
            else:
                json_file_paths = [json_file_path]
        else:
            json_file_paths = json_file_path
        
        # Load JSON files for transcript/response pairs
        json_datasets = []
        for json_path in json_file_paths:
            ds = self._load_json_dataset(json_path)
            json_datasets.append(ds)
        
        if len(json_datasets) > 1:
            json_ds = hf_datasets.concatenate_datasets(json_datasets)
        else:
            json_ds = json_datasets[0]
        
        # Store HF dataset configs for lazy loading during iteration
        # We'll load audio on-demand and filter during iteration instead of upfront
        self._hf_dataset_name = hf_dataset_name
        self._hf_batch_configs = hf_batch_configs
        
        # Store JSON dataset (no upfront filtering - will filter during iteration)
        self._json_dataset = json_ds
        
        # Cache for HF dataset lookups (lazy loading)
        self._hf_audio_cache = {}  # audio_filename -> HF dataset row
        self._hf_datasets_loaded = False
        self._hf_audio_indices = {}  # audio_filename -> (batch_idx, hf_idx)
        
        # We'll build the mapping lazily during first iteration or when needed
        
        # Split the dataset if needed
        matching_split = None
        for split in config.splits:
            if split.split == self._args.split:
                matching_split = split
                break
        
        if matching_split is None:
            raise ValueError(f"No split config found for {self._args.split}")
        
        # Apply 80/10/10 split if multiple batches
        if len(json_file_paths) > 1:
            split_seed = 42
            train_pct = 0.8
            
            if self._args.shuffle:
                json_ds = json_ds.shuffle(seed=split_seed)
            
            train_val_test = json_ds.train_test_split(
                train_size=train_pct,
                seed=split_seed,
                shuffle=False
            )
            
            val_test_split = train_val_test["test"].train_test_split(
                train_size=0.5,
                seed=split_seed,
                shuffle=False
            )
            
            if self._args.split == types.DatasetSplit.TRAIN:
                self._json_dataset = train_val_test["train"]
            elif self._args.split == types.DatasetSplit.VALIDATION:
                self._json_dataset = val_test_split["train"]
            elif self._args.split == types.DatasetSplit.TEST:
                self._json_dataset = val_test_split["test"]
        
        # Calculate actual number of samples (will be adjusted during iteration as we filter)
        try:
            actual_num_samples = len(self._json_dataset)
        except (TypeError, NotImplementedError, AttributeError):
            actual_num_samples = matching_split.num_samples if matching_split.num_samples != -1 else 0
        
        dataset_name = f"{config.name}.{self._args.split.value}"
        super()._init_dataset(self._json_dataset, dataset_name, actual_num_samples)
    
    def _load_hf_datasets_if_needed(self):
        """Lazy load HF datasets and build audio mapping when first needed.
        This happens during the first iteration, filtering happens on-the-fly.
        
        Works with both parquet-backed and regular HF datasets. Parquet files are
        automatically handled by Hugging Face datasets library.
        """
        if self._hf_datasets_loaded:
            return
        
        # Load Hugging Face datasets for audio
        # HF datasets can be stored as parquet files - load_dataset handles this automatically
        # We use streaming=False to build index, but parquet is efficient for this
        for batch_idx, batch_config in enumerate(self._hf_batch_configs):
            hf_ds = hf_datasets.load_dataset(
                self._hf_dataset_name,
                batch_config,
                trust_remote_code=True,
                streaming=False,  # Need non-streaming to build index efficiently
                # Parquet files are automatically handled - no special config needed
                download_config=hf_datasets.DownloadConfig(max_retries=10),
            )
            # Get the default split (usually "train")
            split_name = list(hf_ds.keys())[0]
            hf_ds_split = hf_ds[split_name]
            
            # Build mapping from audio filename to (batch_idx, hf_idx)
            # For parquet-backed datasets, this iterates efficiently through the parquet file
            for hf_idx in range(len(hf_ds_split)):
                row = hf_ds_split[hf_idx]
                audio_path = None
                
                # Try to get audio path from various possible fields
                # For parquet datasets with Audio feature, audio is typically in "audio" field
                if "audio" in row:
                    audio_obj = row["audio"]
                    if isinstance(audio_obj, dict):
                        # Audio feature stores path in dict: {"path": "...", "array": ..., "sampling_rate": ...}
                        audio_path = audio_obj.get("path", "")
                    elif isinstance(audio_obj, str):
                        audio_path = audio_obj
                elif "path" in row:
                    audio_path = row["path"]
                elif "file" in row:
                    audio_path = row["file"]
                elif "filename" in row:
                    audio_path = row["filename"]
                elif "audio_path" in row:
                    audio_path = row["audio_path"]
                
                # Extract just the filename
                if audio_path:
                    audio_filename = os.path.basename(audio_path)
                    if audio_filename:
                        # Store first occurrence (don't cache row yet - load on demand)
                        if audio_filename not in self._hf_audio_indices:
                            self._hf_audio_indices[audio_filename] = (batch_idx, hf_idx)
        
        self._hf_datasets_loaded = True
    
    def _get_hf_audio_row(self, audio_filename: str):
        """Get HF audio row for a given audio filename, loading on-demand."""
        self._load_hf_datasets_if_needed()
        
        # Check cache first
        if audio_filename in self._hf_audio_cache:
            return self._hf_audio_cache[audio_filename]
        
        # If not in cache but in indices, load it
        if audio_filename in self._hf_audio_indices:
            batch_idx, hf_idx = self._hf_audio_indices[audio_filename]
            # Load the specific dataset and row (cache the dataset to avoid reloading)
            if not hasattr(self, '_hf_dataset_cache'):
                self._hf_dataset_cache = {}
            
            cache_key = batch_idx
            if cache_key not in self._hf_dataset_cache:
                hf_ds = hf_datasets.load_dataset(
                    self._hf_dataset_name,
                    self._hf_batch_configs[batch_idx],
                    trust_remote_code=True,
                    streaming=False,
                    download_config=hf_datasets.DownloadConfig(max_retries=10),
                )
                split_name = list(hf_ds.keys())[0]
                self._hf_dataset_cache[cache_key] = hf_ds[split_name]
            
            row = self._hf_dataset_cache[cache_key][hf_idx]
            self._hf_audio_cache[audio_filename] = row
            return row
        
        return None
    
    def _load_json_dataset(self, json_file_path: str):
        """Load JSON file as Hugging Face dataset."""
        return hf_datasets.load_dataset(
            "json",
            data_files=json_file_path,
            split="train",
            trust_remote_code=True,
            streaming=False,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
    
    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = None
    ) -> np.ndarray:
        """Load audio from Hugging Face dataset (lazy loading)."""
        audio_field = self._config.audio_field or "audio_path"
        audio_path = row.get(audio_field, "")
        audio_filename = os.path.basename(audio_path) if audio_path else ""
        
        if not audio_filename:
            raise ValueError(f"Invalid audio_path in row: {audio_path}")
        
        # Get HF dataset row (lazy loading)
        hf_row = self._get_hf_audio_row(audio_filename)
        
        if hf_row is None:
            raise ValueError(f"Audio file not found in HF dataset: {audio_filename}")
        
        # Extract audio from HF dataset
        # For parquet-backed datasets with Audio feature, audio is decoded automatically
        if "audio" in hf_row:
            audio_obj = hf_row["audio"]
            if isinstance(audio_obj, dict):
                if "array" in audio_obj:
                    # Audio feature provides decoded array
                    audio_array = audio_obj["array"]
                    sampling_rate = audio_obj.get("sampling_rate", data_sample.SAMPLE_RATE)
                    # Resample if needed
                    if sampling_rate != data_sample.SAMPLE_RATE:
                        import librosa
                        audio_array = librosa.resample(
                            audio_array.astype(np.float32),
                            orig_sr=sampling_rate,
                            target_sr=data_sample.SAMPLE_RATE
                        )
                    return audio_array.astype(np.float32)
                elif "path" in audio_obj:
                    # If only path is provided (not decoded), load it
                    audio_path = audio_obj["path"]
                    return data_sample.audio_from_file(audio_path)
            elif isinstance(audio_obj, str):
                # Direct path string
                return data_sample.audio_from_file(audio_obj)
        
        raise ValueError(f"Could not extract audio from HF dataset row for {audio_filename}. Available keys: {list(hf_row.keys())}")
    
    def __iter__(self):
        """Override iteration to filter samples with missing audio during training."""
        num_workers, _, _ = _get_worker_info(self._length)
        if num_workers > 1:
            assert hasattr(
                self._dataset, "n_shards"
            ), f"{self._name} does not have n_shards attribute, which is required when num_workers ({num_workers}) > 1"
            assert (
                self._dataset.n_shards >= num_workers
            ), f"{self._name} has {self._dataset.n_shards} shards, which is less than the number of workers ({num_workers})."

        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        missing_audio_samples = 0
        dataset_iter = iter(self._dataset)
        
        for row in dataset_iter:
            actual_length += 1
            
            # Check if audio exists in HF dataset before processing
            if self._args.include_audio:
                audio_field = self._config.audio_field or "audio_path"
                audio_path = row.get(audio_field, "")
                audio_filename = os.path.basename(audio_path) if audio_path else ""
                
                if audio_filename:
                    # Lazy load HF datasets if needed
                    self._load_hf_datasets_if_needed()
                    
                    # Check if audio exists in HF dataset
                    if audio_filename not in self._hf_audio_indices:
                        missing_audio_samples += 1
                        continue  # Skip this sample - audio not found in HF dataset
            
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self.name} for row {row}")
                bad_samples += 1
                continue

            input_characters = sum(len(msg["content"]) for msg in sample.messages)
            if (
                self._args.max_input_characters is not None
                and input_characters > self._args.max_input_characters
            ):
                print(
                    f"Sample has input characters longer than {self._args.max_input_characters} in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            elif len(sample.messages[-1]["content"]) == 0:
                print(
                    f"Sample has empty assistant message in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            if self._args.include_audio:
                if sample.audio is None:
                    print(f"Audio is None for sample {sample}")
                    bad_samples += 1
                    continue
                if sample.audio.shape[-1] == 0:
                    print(f"Audio length is 0 for sample {sample}")
                    bad_samples += 1
                    continue
                if (
                    self._args.max_audio_duration_secs > 0
                    and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    skipped_samples += 1
                    continue

            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), "
            f"removed {bad_samples} bad samples, skipped {skipped_samples} samples for exceeding max audio duration, "
            f"and filtered {missing_audio_samples} samples with missing audio in HF dataset."
        )
    
    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:
        """Convert row to VoiceSample using templates."""
        # Use the same logic as GenericDataset._get_sample
        message_history = (
            text_proc.format_message_history(
                row[self._config.message_history_column],
                self._config.message_history_roles,
            )
            if self._config.message_history_column is not None
            and self._config.message_history_roles is not None
            and not self._args.ignore_message_history
            else None
        )

        try:
            user_content = jinja2.Template(
                self._config.user_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(
                **row,
                text_proc=text_proc,
                **self._config.user_template_args,  # type: ignore[arg-type]
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            system_prompt = (
                jinja2.Template(
                    self._config.system_prompt_template,  # type: ignore[arg-type]
                    undefined=jinja2.StrictUndefined,
                ).render(**row, text_proc=text_proc)
                if self._config.system_prompt_template is not None
                and not self._args.ignore_system_prompt
                else None
            )

        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"system_prompt_template: {self._config.system_prompt_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e
        if not self._args.include_audio:
            user_content = user_content.replace(
                types.AUDIO_PLACEHOLDER, f'"{transcript}"'
            )

        messages = _get_messages(
            user_content,
            assistant_content,
            message_history=message_history,
            sys_prompt=system_prompt,
        )
        audio: Optional[np.ndarray] = (  # type: ignore[no-redef]
            self._get_audio(row, self._config.audio_field)
            if self._args.include_audio
            else None
        )
        return self._make_sample(
            messages,
            audio,
            audio_transcript=transcript,
        )
    
    def __str__(self):
        return f"HausaHFDataset({self._config.name})"
    
    def get_config(self):
        return self._config


class LocalJsonDataset(VoiceDataset):
    """
    Dataset class for loading local JSON files with audio_path fields.
    Handles loading audio from file paths instead of embedded audio data.
    Supports multiple JSON files (can be a list of paths or a single path).
    """
    
    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
        json_file_path: str,
        audio_base_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            args: VoiceDatasetArgs for dataset configuration
            config: DatasetConfig with templates and settings
            json_file_path: Path to the local JSON file(s). Can be:
                - A single file path (string)
                - A list of file paths (list of strings)
                - A comma-separated string of paths
            audio_base_dir: Base directory for audio files (if audio_path is relative)
        """
        super().__init__(args)
        self._config = config
        self._audio_base_dir = audio_base_dir
        
        # Handle multiple JSON files
        if isinstance(json_file_path, str):
            # Check if it's a comma-separated list of paths
            if ',' in json_file_path:
                json_file_paths = [p.strip() for p in json_file_path.split(',')]
            else:
                json_file_paths = [json_file_path]
        else:
            json_file_paths = json_file_path
        
        # Set audio_base_dir from first file if not provided
        if self._audio_base_dir is None:
            self._audio_base_dir = os.path.dirname(json_file_paths[0])
        
        # Load dataset from JSON file(s)
        assert config.splits is not None, "Dataset config must have splits defined"
        
        # Load all JSON files and concatenate them first
        all_datasets = []
        for json_path in json_file_paths:
            ds = self._load_hf_dataset(
                json_path,
                split="train",  # Load as train, we'll split later
            )
            all_datasets.append(ds)
        
        # Concatenate all datasets
        if len(all_datasets) > 1:
            combined_ds = hf_datasets.concatenate_datasets(all_datasets)
        else:
            combined_ds = all_datasets[0]
        
        # Now split the combined dataset based on the requested split
        # Find the split config that matches the requested split type
        matching_split = None
        for split in config.splits:
            if split.split == self._args.split:
                matching_split = split
                break
        
        if matching_split is None:
            raise ValueError(f"No split config found for {self._args.split}")
        
        # If we have multiple JSON files, we need to split the combined data
        # Use 80/10/10 split (train/val/test) with a fixed seed for consistency
        if len(json_file_paths) > 1:
            # Fixed seed for deterministic splitting across all configs
            split_seed = 42
            
            # 80/10/10 split proportions
            train_pct = 0.8
            val_pct = 0.1
            test_pct = 0.1
            
            # Shuffle the combined dataset first (if not already shuffled)
            if self._args.shuffle:
                combined_ds = combined_ds.shuffle(seed=split_seed)
            
            # First split: train (80%) vs (val+test) (20%)
            train_val_test = combined_ds.train_test_split(
                train_size=train_pct,
                seed=split_seed,
                shuffle=False  # Already shuffled above
            )
            
            # Second split: val (10%) vs test (10%) from the remaining 20%
            # val should be 50% of the remaining (10% of total)
            val_test_split = train_val_test["test"].train_test_split(
                train_size=0.5,  # 50% of 20% = 10% of total
                seed=split_seed,
                shuffle=False
            )
            
            # Select the appropriate split
            if self._args.split == types.DatasetSplit.TRAIN:
                dataset = train_val_test["train"]
            elif self._args.split == types.DatasetSplit.VALIDATION:
                dataset = val_test_split["train"]
            elif self._args.split == types.DatasetSplit.TEST:
                dataset = val_test_split["test"]
            else:
                # Fallback: use combined dataset
                dataset = combined_ds
        else:
            # Single file, no splitting needed, use the combined dataset
            dataset = combined_ds
        
        # Calculate actual number of samples dynamically
        # If num_samples is -1 or not set, calculate from actual dataset
        if matching_split.num_samples == -1 or matching_split.num_samples is None:
            # Try to get the actual length from the split dataset
            try:
                actual_num_samples = len(dataset)
            except (TypeError, NotImplementedError, AttributeError):
                # Fallback: calculate from combined dataset if available
                try:
                    total_len = len(combined_ds)
                    if len(json_file_paths) > 1:
                        # Use 80/10/10 split proportions
                        if self._args.split == types.DatasetSplit.TRAIN:
                            actual_num_samples = int(total_len * 0.8)
                        elif self._args.split == types.DatasetSplit.VALIDATION:
                            actual_num_samples = int(total_len * 0.1)
                        else:  # TEST
                            # Test gets the remainder to account for rounding
                            actual_num_samples = total_len - int(total_len * 0.8) - int(total_len * 0.1)
                    else:
                        actual_num_samples = total_len
                except (TypeError, NotImplementedError, AttributeError):
                    # Can't determine length - use 0 as placeholder
                    # Length will be determined during iteration
                    actual_num_samples = 0
        else:
            actual_num_samples = matching_split.num_samples
        
        dataset_name = f"{config.name}.{self._args.split.value}"
        super()._init_dataset(dataset, dataset_name, actual_num_samples)
    
    def __str__(self):
        return f"LocalJsonDataset({self._config.name})"
    
    def _load_hf_dataset(
        self,
        json_file_path: str,
        split: Optional[str] = None,
        streaming: bool = False,  # Use non-streaming to get actual length
    ) -> data.Dataset:
        """Load JSON file using Hugging Face datasets."""
        # Load JSON file using Hugging Face datasets
        # Use non-streaming mode so we can get the actual dataset length
        dataset = hf_datasets.load_dataset(
            "json",
            data_files=json_file_path,
            split=split if split else "train",
            trust_remote_code=True,
            streaming=streaming,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
        
        # Note: We don't cast audio_field here because audio_path contains paths, not audio data
        # Audio will be loaded in _get_audio method
        
        if self._args.shuffle:
            if streaming:
                dataset = dataset.shuffle(
                    seed=self._args.shuffle_seed,
                    buffer_size=self._args.shuffle_buffer_size,
                )
            else:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset
    
    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = None
    ) -> np.ndarray:
        """Override to load audio from file path."""
        # Use audio_field from config, or default to "audio_path"
        audio_field = self._config.audio_field or "audio_path"
        
        # Get audio path from row
        if audio_field not in row:
            # Try alternative names
            if "audio_path" in row:
                audio_path = row["audio_path"]
            elif "audio" in row:
                # If it's already loaded as Audio object
                audio = row["audio"]
                if isinstance(audio, dict) and "array" in audio:
                    sampling_rate = audio.get("sampling_rate", data_sample.SAMPLE_RATE)
                    assert sampling_rate == data_sample.SAMPLE_RATE
                    return audio["array"]
                else:
                    raise ValueError(f"Unexpected audio format in row: {audio}")
            else:
                raise ValueError(f"No audio field found in row. Available keys: {list(row.keys())}")
        else:
            audio_path = row[audio_field]
        
        # Handle relative paths
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(self._audio_base_dir, audio_path)
        
        # Load audio using librosa (via data_sample utility)
        if not os.path.exists(audio_path):
            # Try looking in subdirectories
            audio_filename = os.path.basename(audio_path)
            # Check common subdirectories
            for subdir in ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2", ""]:
                potential_path = os.path.join(self._audio_base_dir, subdir, audio_filename)
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio file
        audio = data_sample.audio_from_file(audio_path)
        return audio
    
    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:
        """Convert row to VoiceSample using templates."""
        # Use the same logic as GenericDataset._get_sample
        # Convert the dataset's message_history_column into a list of messages
        message_history = (
            text_proc.format_message_history(
                row[self._config.message_history_column],
                self._config.message_history_roles,
            )
            if self._config.message_history_column is not None
            and self._config.message_history_roles is not None
            and not self._args.ignore_message_history
            else None
        )

        try:
            user_content = jinja2.Template(
                self._config.user_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(
                **row,
                text_proc=text_proc,
                **self._config.user_template_args,  # type: ignore[arg-type]
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            transcript = jinja2.Template(
                self._config.transcript_template,  # type: ignore[arg-type]
                undefined=jinja2.StrictUndefined,
            ).render(**row, text_proc=text_proc)
            system_prompt = (
                jinja2.Template(
                    self._config.system_prompt_template,  # type: ignore[arg-type]
                    undefined=jinja2.StrictUndefined,
                ).render(**row, text_proc=text_proc)
                if self._config.system_prompt_template is not None
                and not self._args.ignore_system_prompt
                else None
            )

        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"system_prompt_template: {self._config.system_prompt_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e
        if not self._args.include_audio:
            user_content = user_content.replace(
                types.AUDIO_PLACEHOLDER, f'"{transcript}"'
            )

        messages = _get_messages(
            user_content,
            assistant_content,
            message_history=message_history,
            sys_prompt=system_prompt,
        )
        audio: Optional[np.ndarray] = (  # type: ignore[no-redef]
            self._get_audio(row, self._config.audio_field)
            if self._args.include_audio
            else None
        )
        return self._make_sample(
            messages,
            audio,
            audio_transcript=transcript,
        )
    
    def get_config(self):
        return self._config


class LibriSpeechDummyDataset(GenericDataset):
    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        VoiceDataset.__init__(self, args)
        # This dataset doesn't support streaming.
        dataset = self._load_hf_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,
        )
        self._init_dataset(dataset, "dummy", 73)

    def __str__(self):
        return "LibriSpeechDummyDataset"

    @property
    def name(self):
        return "dummy"

    def get_config(self):
        return types.DatasetConfig(
            name="dummy",
            path="hf-internal-testing/librispeech_asr_dummy",
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        text = text_proc.format_asr_text(row["text"])
        user_content = "Transcribe\n"
        user_content += (
            types.AUDIO_PLACEHOLDER if self._args.include_audio else f'"{text}"'
        )
        return self._make_sample(
            self._make_messages(user_content, text),
            # some of our test models that use this dataset can only handle up to 4 seconds of audio
            self._get_audio(row, "audio")[: 4 * data_sample.SAMPLE_RATE],
            audio_transcript=text,
        )


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length

    def __str__(self):
        return f"EmptyDataset(length={self._length})"

    @property
    def name(self):
        return "empty"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple SizedIterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: An optional list of dataset weights, i.e., the number of times it should be repeated.
            seed: Optional seed for reproducibility.
        """
        self._datasets = datasets
        if weights is not None:
            assert len(weights) == len(datasets)
        else:
            weights = [1.0] * len(datasets)
        self._weights = weights
        self._weighted_samples = [int(w * len(d)) for w, d in zip(weights, datasets)]
        self._total_samples = sum(self._weighted_samples)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]
        ds_pos = [0] * len(ds_iters)
        num_workers, worker_id, worker_samples = _get_worker_info(self._total_samples)
        # Find the iterator that is least far along and vend from it.
        for i in range(worker_samples):
            min_fraction = 1.0
            for j in range(len(ds_iters)):
                iter_fraction = ds_pos[j] / self._weighted_samples[j]
                if iter_fraction < min_fraction:
                    min_fraction = iter_fraction
                    iter_index = j
            try:
                yield next(ds_iters[iter_index])
            except StopIteration:
                ds_iters[iter_index] = iter(self._datasets[iter_index])
                try:
                    yield next(ds_iters[iter_index])
                except StopIteration:
                    warnings.warn(
                        f"Dataset {iter_index} is empty for worker {worker_id}/{num_workers}. num_workers is likely too high. Stopping iteration."
                    )
                    break
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples

    def __str__(self):
        return "+".join([f"{d}:{w:.2f}" for w, d in zip(self._weights, self._datasets)])

    @property
    def name(self):
        return "+".join([ds.name for ds in self._datasets])


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: data_sample.VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        # Replace generator expression with a regular function that yields items
        for sample in self._dataset:
            yield self._process(sample)

    def __len__(self):
        return len(self._dataset)

    def __str__(self):
        return f"Dataproc({self._dataset})"

    @property
    def name(self):
        return self._dataset.name


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self,
        dataset: SizedIterableDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            warnings.warn(
                f"num_samples ({self._length}) exceeds dataset length ({len(dataset)}). Truncating to {len(dataset)}."
            )
            self._length = len(dataset)
        self._name = f"{dataset.name}.{self._length}"

    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        if worker_samples == 0:
            return iter([])
        yielded_samples = 0
        try:
            for sample in self._dataset:
                yielded_samples += 1
                yield sample
                if yielded_samples == worker_samples:
                    break
        except Exception as e:
            logging.error(
                f"Worker {worker_id}/{num_workers} failed after yielding {yielded_samples}/{worker_samples} samples, out of {self._length} total samples with error: {e}"
            )
            raise e
        if yielded_samples < worker_samples:
            logging.warn(
                f"Worker {worker_id}/{num_workers} only yielded {yielded_samples} (expected {worker_samples}) samples, out of {self._length} total samples"
            )

    def __str__(self):
        return f"Range({self._dataset}%{len(self)})"

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def get_config(self):
        if isinstance(self._dataset, GenericDataset):
            return self._dataset.get_config()
        else:
            raise ValueError("Cannot get config for non-GenericDataset")
