import dataclasses
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import simple_parsing
import transformers
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

from ultravox.model import file_utils
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.utils import device_helpers

# This script is used to upload a model to the HuggingFace Hub, for either internal or external consumption.
# Ex: python -m ultravox.tools.push_to_hub -m wandb://fixie/ultravox/<model_path> -u fixie-ai/ultravox-vXYZ
@dataclasses.dataclass
class UploadToHubArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(alias="-m")
    # HuggingFace Hub model_id to push to
    hf_upload_model: str = simple_parsing.field(alias="-u")
    # Only the llm for finetuned models
    text_only: bool = simple_parsing.field(default=False, alias="-t")
    # Device to use for the model
    device: Optional[str] = simple_parsing.field(
        default=device_helpers.default_device(), alias="-D"
    )
    # Data type to use for the model
    data_type: Optional[str] = None
    # Public or private (default)
    private: bool = True
    # Verify the model after uploading
    verify: bool = False
    chat_template: Optional[str] = simple_parsing.field(default=None, alias="-c")
    # Upload all files from checkpoint directory (for checkpoint-* directories)
    # If False, only uploads required model files (for best model)
    upload_checkpoint_files: bool = simple_parsing.field(default=True)
    # Best model mode: only upload required model files, skip checkpoint files
    best_model_only: bool = simple_parsing.field(default=False, alias="-b")

    def __post_init__(self):
        if self.chat_template and self.chat_template.startswith("file://"):
            file_path = self.chat_template[7:].strip()  # Remove "file://" prefix
            try:
                with open(file_path, "r") as f:
                    self.chat_template = f.read()
            except Exception as e:
                raise ValueError(
                    f"Failed to load chat template from file {file_path}: {e}"
                )


def find_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the last checkpoint directory in output/checkpoint-* format.
    Returns the path to the last checkpoint directory, or None if not found.
    """
    output_path = Path(output_dir)
    
    # If the path itself is a checkpoint directory, check if there's a parent output dir
    if "checkpoint-" in output_path.name:
        parent_dir = output_path.parent
        checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
        match = checkpoint_pattern.match(output_path.name)
        if match:
            # This is already a checkpoint, check if there are others in parent
            checkpoints = []
            for item in parent_dir.iterdir():
                if item.is_dir() and checkpoint_pattern.match(item.name):
                    checkpoints.append(item)
            
            if checkpoints:
                # Sort by checkpoint number and return the last one
                checkpoints.sort(key=lambda x: int(checkpoint_pattern.match(x.name).group(1)))
                return str(checkpoints[-1])
            else:
                return str(output_path)
    
    # Look for checkpoint directories in the output directory
    if not output_path.exists():
        return None
    
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    checkpoints = []
    
    for item in output_path.iterdir():
        if item.is_dir() and checkpoint_pattern.match(item.name):
            checkpoints.append(item)
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number and return the last one
    checkpoints.sort(key=lambda x: int(checkpoint_pattern.match(x.name).group(1)))
    last_checkpoint = checkpoints[-1]
    
    logging.info(f"Found {len(checkpoints)} checkpoints, using last: {last_checkpoint.name}")
    return str(last_checkpoint)


def upload_checkpoint_files(checkpoint_dir: str, repo_id: str, private: bool = True, include_training_artifacts: bool = True):
    """
    Upload files from a checkpoint directory to HuggingFace Hub.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: HuggingFace Hub repository ID
        private: Whether the repo is private
        include_training_artifacts: If True, upload all files including training artifacts.
                                   If False, only upload files needed for inference.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    api = HfApi()
    
    # Create the repository if it doesn't exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    
    # Training artifacts to exclude when include_training_artifacts=False
    training_artifacts = {
        "optimizer.bin",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    # RNG state files (rng_state_*.pth)
    rng_state_pattern = re.compile(r"rng_state_\d+\.pth")
    
    # Get all files in the checkpoint directory
    files_to_upload = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path from checkpoint directory
            rel_path = os.path.relpath(file_path, checkpoint_dir)
            
            # Filter out training artifacts if not including them
            if not include_training_artifacts:
                if file in training_artifacts or rng_state_pattern.match(file):
                    logging.debug(f"Skipping training artifact: {rel_path}")
                    continue
            
            files_to_upload.append((file_path, rel_path))
    
    checkpoint_name = checkpoint_path.name
    upload_type = "all files" if include_training_artifacts else "inference files only"
    logging.info(f"Found {len(files_to_upload)} files to upload ({upload_type}) from checkpoint: {checkpoint_name}")
    logging.info(f"Uploading {upload_type} from {checkpoint_name} to {repo_id}...")
    
    # Upload all files in a single commit using upload_folder
    # This is more efficient than uploading files individually
    try:
        if include_training_artifacts:
            # Upload entire folder (all files)
            api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=repo_id,
                commit_message=f"Upload all files from last checkpoint: {checkpoint_name}",
            )
        else:
            # Upload only selected files (inference files only)
            # We need to upload files individually to exclude training artifacts
            for file_path, rel_path in files_to_upload:
                try:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=rel_path,
                        repo_id=repo_id,
                        commit_message=f"Upload inference file: {rel_path}",
                    )
                    logging.debug(f"Uploaded: {rel_path}")
                except Exception as file_error:
                    logging.warning(f"Failed to upload {rel_path}: {file_error}")
        
        logging.info(f"Successfully uploaded {len(files_to_upload)} files from {checkpoint_name}")
    except Exception as e:
        logging.warning(f"Failed to upload folder, trying individual files: {e}")
        # Fallback to individual file uploads
        for file_path, rel_path in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    commit_message=f"Upload checkpoint file: {rel_path}",
                )
                logging.debug(f"Uploaded: {rel_path}")
            except Exception as file_error:
                logging.warning(f"Failed to upload {rel_path}: {file_error}")
    
    logging.info(f"Finished uploading checkpoint files to {repo_id}")


def main(args: UploadToHubArgs):
    """
    Main function to upload model to HuggingFace Hub.
    
    Upload behavior:
    - Last checkpoint (best_model_only=False): 
      * Uploads ALL files from checkpoint directory (including training artifacts)
      * Training artifacts: optimizer.bin, scheduler.pt, rng_state_*.pth, trainer_state.json, training_args.bin
      * Then uploads model via pipeline
    
    - Best model (best_model_only=True):
      * Uploads checkpoint files BUT excludes training artifacts (only inference files)
      * Inference files: model weights, config, tokenizer, processor files
      * Then uploads model via pipeline
    """
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load the model and tokenizer, then merge LoRA weights if they exist
    model_path = file_utils.download_dir_if_needed(args.model)
    
    # Determine if we should include training artifacts in checkpoint upload
    # Last checkpoint: include_training_artifacts=True (upload everything)
    # Best model: include_training_artifacts=False (exclude training artifacts)
    include_training_artifacts = not args.best_model_only
    
    if args.best_model_only:
        logging.info("="*80)
        logging.info("BEST MODEL MODE")
        logging.info("="*80)
        logging.info("Uploading checkpoint files (excluding training artifacts) + model files")
        logging.info("Excluded: optimizer.bin, scheduler.pt, rng_state_*.pth, trainer_state.json, training_args.bin")
    else:
        logging.info("="*80)
        logging.info("LAST CHECKPOINT MODE")
        logging.info("="*80)
        logging.info("Uploading ALL checkpoint files (including training artifacts) + model files")
    
    if args.upload_checkpoint_files:
        model_path_obj = Path(model_path).resolve()
        
        # Try to find the last checkpoint directory
        last_checkpoint = None
        
        # Case 1: model_path is already a checkpoint directory
        if "checkpoint-" in model_path_obj.name:
            # Check if there are other checkpoints in the parent directory
            parent_dir = model_path_obj.parent
            last_checkpoint = find_last_checkpoint(str(parent_dir))
            if not last_checkpoint:
                # Use the current checkpoint if it's the only one
                last_checkpoint = str(model_path_obj)
        
        # Case 2: model_path is an output directory or contains checkpoints
        elif "output" in str(model_path_obj):
            last_checkpoint = find_last_checkpoint(str(model_path_obj))
            if not last_checkpoint:
                # Check parent directory
                last_checkpoint = find_last_checkpoint(str(model_path_obj.parent))
        
        # Case 3: Check parent directories for output/checkpoint-* structure
        else:
            current = model_path_obj.parent
            max_depth = 5  # Limit search depth
            depth = 0
            while current != current.parent and depth < max_depth:
                if "output" in str(current):
                    last_checkpoint = find_last_checkpoint(str(current))
                    if last_checkpoint:
                        break
                current = current.parent
                depth += 1
        
        if last_checkpoint:
            logging.info(f"Found last checkpoint: {last_checkpoint}")
            # Upload checkpoint files with appropriate filtering
            # include_training_artifacts is True for last checkpoint, False for best model
            upload_checkpoint_files(last_checkpoint, args.hf_upload_model, args.private, include_training_artifacts=include_training_artifacts)
        else:
            logging.info("No checkpoint directories found, proceeding with regular model upload")
    
    dtype = device_helpers.get_dtype(args.data_type)
    model = ultravox_model.UltravoxModel.from_pretrained(model_path, torch_dtype=dtype)
    model.merge_and_unload()

    if args.text_only:
        text_llm = model.language_model
        tokenizer_repo = model.config.text_model_id

        print("Preparing text language model with tokenizer for upload...")
        with tempfile.TemporaryDirectory() as temp_dir:
            text_llm.save_pretrained(temp_dir)

            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            for file in tokenizer_files:
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=tokenizer_repo, filename=file
                    )
                    target_path = os.path.join(temp_dir, file)
                    shutil.copy2(downloaded_file, target_path)
                except Exception as e:
                    print(
                        f"Warning: Could not download {file} from {tokenizer_repo}: {e}"
                    )

            # Upload the combined model
            print("Uploading text model with tokenizer to HuggingFace Hub...")
            api = HfApi()
            # Create the repository if it doesn't exist
            api.create_repo(
                repo_id=args.hf_upload_model, private=args.private, exist_ok=True
            )
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=args.hf_upload_model,
                commit_message=f"Upload text model with tokenizer from {tokenizer_repo}",
            )
        return

    # Upload model using pipeline (handles model files, config, tokenizer properly)
    pipe = ultravox_pipeline.UltravoxPipeline(
        model=model,
        device=args.device,
        chat_template=args.chat_template,
    )

    if args.best_model_only:
        logging.info("Best model mode: Uploading only required model files (skipping checkpoint training artifacts)...")
    else:
        logging.info("Uploading model to HuggingFace Hub...")
    
    pipe.push_to_hub(args.hf_upload_model, private=args.private)

    if args.verify:
        from ultravox import data as datasets

        print("Model uploaded. Testing model...")
        loaded_pipe = transformers.pipeline(
            model=args.hf_upload_model, trust_remote_code=True
        )
        ds = datasets.create_dataset("boolq", datasets.VoiceDatasetArgs())
        sample = next(iter(ds))
        generated = loaded_pipe(
            {"audio": sample.audio, "turns": sample.messages[:-1]}, max_new_tokens=10
        )
        print(f"Generated (max 10 tokens): {generated}")


if __name__ == "__main__":
    main(simple_parsing.parse(UploadToHubArgs))
