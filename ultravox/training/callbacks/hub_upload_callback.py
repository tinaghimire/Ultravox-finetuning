"""
Callback to automatically upload best and last checkpoints to HuggingFace Hub
after each save_steps checkpoint.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HubUploadCallback(transformers.TrainerCallback):
    """
    Callback that uploads best and last checkpoints to HuggingFace Hub
    after each checkpoint save.

    Args:
        hub_repo: Base HuggingFace Hub repository name
            (e.g., "vaghawan/hausa-ultravox-stage1")
        best_tag: Tag/suffix for best model repo (default: "best")
        last_tag: Tag/suffix for last model repo (default: "last")
        device: Device to use for upload (default: "cuda")
        upload_best: Whether to upload best checkpoint (default: True)
        upload_last: Whether to upload last checkpoint (default: True)
    """

    def __init__(
        self,
        hub_repo: str,
        best_tag: str = "best",
        last_tag: str = "last",
        device: str = "cuda",
        upload_best: bool = True,
        upload_last: bool = True,
    ):
        self.hub_repo = hub_repo
        self.best_tag = best_tag
        self.last_tag = last_tag
        self.device = device
        self.upload_best = upload_best
        self.upload_last = upload_last
        self.best_checkpoint_path: Optional[Path] = None
        self.failed_upload_path: Optional[Path] = None

    def _upload_checkpoint(
        self,
        checkpoint_path: Path,
        repo_name: str,
        is_global_master: bool,
        upload_all_files: bool = True,
    ) -> bool:
        """
        Upload a checkpoint to HuggingFace Hub.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            repo_name: HuggingFace Hub repository name
            is_global_master: Whether this is the main process
            upload_all_files: If True, upload all checkpoint files (for last checkpoint).
                            If False, only upload required model files (for best model).
        
        Returns:
            True if upload was successful, False otherwise
        """
        if not is_global_master:
            return False

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
            return False

        upload_type = "all files from" if upload_all_files else "required files from"
        logger.info(f"Uploading {upload_type} {checkpoint_path.name} to {repo_name}...")

        cmd = [
            sys.executable,
            "-m",
            "ultravox.tools.push_to_hub",
            "-m",
            str(checkpoint_path),
            "-u",
            repo_name,
            "-D",
            self.device,
        ]
        
        # Control checkpoint file upload based on upload_all_files flag
        # For last checkpoint (upload_all_files=True): upload ALL files including training artifacts
        # For best model (upload_all_files=False): upload checkpoint files BUT exclude training artifacts
        # The -b flag (best_model_only) filters out training artifacts from checkpoint upload
        if not upload_all_files:
            # Add -b flag to exclude training artifacts (optimizer, scheduler, rng_state, trainer_state, etc.)
            cmd.append("-b")

        try:
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise on error, log instead
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            if result.returncode == 0:
                logger.info(
                    f"✓ Successfully uploaded {checkpoint_path.name} to "
                    f"{repo_name}"
                )
                return True
            else:
                logger.error(
                    f"Failed to upload {checkpoint_path.name} to {repo_name}. "
                    f"Return code: {result.returncode}, "
                    f"stderr: {result.stderr[:500]}"
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error(
                f"Upload timeout (1h) for {checkpoint_path.name} to "
                f"{repo_name}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Exception during upload of {checkpoint_path.name} to "
                f"{repo_name}: {e}"
            )
            return False

    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        model=None,
        **kwargs,
    ):
        """
        Called after a checkpoint is saved.
        Uploads both best and last checkpoints.
        """
        # Only upload on the main process (rank 0)
        is_global_master = (
            args.local_rank == -1 or args.local_rank == 0
        )

        if not is_global_master:
            return

        output_dir = Path(args.output_dir)

        # Upload last checkpoint (current checkpoint)
        # For last checkpoint: upload ALL files from checkpoint directory
        if self.upload_last:
            current_checkpoint = output_dir / f"checkpoint-{state.global_step}"
            if current_checkpoint.exists():
                repo_last = f"{self.hub_repo}-{self.last_tag}"
                # Upload all files from last checkpoint
                self._upload_checkpoint(
                    current_checkpoint,
                    repo_last, 
                    is_global_master,
                    upload_all_files=True
                )

        # Upload best checkpoint if it exists
        if self.upload_best and state.best_metric is not None:
            # Find the best checkpoint directory
            # HuggingFace Trainer saves best model info in trainer_state.json
            # The best checkpoint is the one with best_metric
            import json

            trainer_state_file = output_dir / "trainer_state.json"
            best_checkpoint = None

            if trainer_state_file.exists():
                try:
                    with open(trainer_state_file) as f:
                        state_data = json.load(f)
                    # Get the best model checkpoint step
                    if "best_model_checkpoint" in state_data:
                        best_model_path = state_data["best_model_checkpoint"]
                        if best_model_path:
                            best_checkpoint = Path(best_model_path)
                except Exception as e:
                    logger.warning(
                        f"Could not read trainer_state.json: {e}"
                    )

            # Fallback: find checkpoint with best eval_loss from all checkpoints
            if best_checkpoint is None or not best_checkpoint.exists():
                checkpoints = list(output_dir.glob("checkpoint-*"))
                if checkpoints:
                    best_step = None
                    best_loss = float("inf")
                    for checkpoint in checkpoints:
                        cp_state_file = checkpoint / "trainer_state.json"
                        if cp_state_file.exists():
                            try:
                                with open(cp_state_file) as f:
                                    cp_state_data = json.load(f)
                                    if (
                                        "best_metric" in cp_state_data
                                        and cp_state_data[
                                            "best_metric"
                                        ] < best_loss
                                    ):
                                        best_loss = cp_state_data["best_metric"]
                                        best_step = checkpoint
                            except Exception:
                                pass

                    if best_step:
                        best_checkpoint = best_step

            if best_checkpoint and best_checkpoint.exists():
                # Normalize paths for comparison (resolve to absolute paths)
                best_checkpoint_resolved = best_checkpoint.resolve()
                previous_best_resolved = (
                    self.best_checkpoint_path.resolve()
                    if self.best_checkpoint_path and self.best_checkpoint_path.exists()
                    else None
                )
                failed_upload_resolved = (
                    self.failed_upload_path.resolve()
                    if self.failed_upload_path and self.failed_upload_path.exists()
                    else None
                )
                
                # Skip upload if this is the same checkpoint as the previous best (already uploaded)
                if best_checkpoint_resolved == previous_best_resolved:
                    logger.info(
                        f"Skipping upload: Best model checkpoint {best_checkpoint.name} "
                        f"is the same as previously uploaded best model"
                    )
                # Skip upload if this is the same checkpoint that previously failed
                elif best_checkpoint_resolved == failed_upload_resolved:
                    logger.info(
                        f"Skipping upload: Best model checkpoint {best_checkpoint.name} "
                        f"previously failed to upload, skipping retry"
                    )
                else:
                    repo_best = f"{self.hub_repo}-{self.best_tag}"
                    # For best model: only upload required model files, not all checkpoint files
                    upload_success = self._upload_checkpoint(
                        best_checkpoint,
                        repo_best,
                        is_global_master,
                        upload_all_files=False
                    )
                    # Update tracking based on upload result
                    if upload_success:
                        # Clear failed upload tracking and update successful upload
                        self.best_checkpoint_path = best_checkpoint
                        self.failed_upload_path = None
                    else:
                        # Track failed upload to avoid repeated retries
                        self.failed_upload_path = best_checkpoint

        return control
