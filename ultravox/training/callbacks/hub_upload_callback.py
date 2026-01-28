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

    def _upload_checkpoint(
        self, checkpoint_path: Path, repo_name: str, is_global_master: bool
    ):
        """Upload a checkpoint to HuggingFace Hub."""
        if not is_global_master:
            return

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
            return

        logger.info(f"Uploading {checkpoint_path.name} to {repo_name}...")

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
            else:
                logger.error(
                    f"Failed to upload {checkpoint_path.name} to {repo_name}. "
                    f"Return code: {result.returncode}, "
                    f"stderr: {result.stderr[:500]}"
                )
        except subprocess.TimeoutExpired:
            logger.error(
                f"Upload timeout (1h) for {checkpoint_path.name} to "
                f"{repo_name}"
            )
        except Exception as e:
            logger.error(
                f"Exception during upload of {checkpoint_path.name} to "
                f"{repo_name}: {e}"
            )

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
        if self.upload_last:
            current_checkpoint = output_dir / f"checkpoint-{state.global_step}"
            if current_checkpoint.exists():
                repo_last = f"{self.hub_repo}-{self.last_tag}"
                self._upload_checkpoint(
                    current_checkpoint, repo_last, is_global_master
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
                repo_best = f"{self.hub_repo}-{self.best_tag}"
                self._upload_checkpoint(
                    best_checkpoint, repo_best, is_global_master
                )

        return control
