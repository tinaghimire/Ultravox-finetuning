#!/usr/bin/env python3
"""
Test script to verify HuggingFace Hub upload override functionality.

This script:
1. Creates a dummy JSON file
2. Uploads it to HuggingFace Hub
3. Modifies the JSON file
4. Uploads it again to the same repo
5. Verifies that the second upload replaced the first one

Usage:
    poetry run python -m ultravox.tools.test_hub_override \
        --repo test-user/test-repo-override \
        --file test_override.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_json(file_path: Path, version: int) -> dict:
    """Create a test JSON file with version info."""
    data = {
        "version": version,
        "timestamp": time.time(),
        "message": f"This is version {version} of the test file",
        "test_data": {
            "upload_count": version,
            "random_value": version * 42,
        },
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Created test JSON file: {file_path} (version {version})")
    return data


def upload_file_to_hub(
    file_path: Path, repo_id: str, filename: str, private: bool = True
):
    """Upload a file to HuggingFace Hub."""
    api = HfApi()
    logger.info(f"Uploading {file_path.name} to {repo_id}/{filename}...")

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info(f"Repository {repo_id} ready")
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Upload the file
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=f"Upload test file version {file_path.stem}",
        )
        logger.info(f"✓ Successfully uploaded {filename} to {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")
        return False


def download_file_from_hub(repo_id: str, filename: str) -> dict:
    """Download a file from HuggingFace Hub and return its content."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="model"
        )
        with open(local_path, "r") as f:
            data = json.load(f)
        logger.info(f"✓ Downloaded {filename} from {repo_id}")
        return data
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return None


def verify_override(original_data: dict, downloaded_data: dict) -> bool:
    """Verify that the downloaded file is different from original."""
    if downloaded_data is None:
        return False

    if downloaded_data.get("version") == original_data.get("version"):
        logger.warning(
            "Downloaded file has same version - override may not have worked"
        )
        return False

    logger.info(
        f"Original version: {original_data.get('version')}, "
        f"Downloaded version: "
        f"{downloaded_data.get('version')}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test HuggingFace Hub upload override functionality"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace Hub repository (e.g., test-user/test-repo)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="test_override.json",
        help="Filename to use in the repo (default: test_override.json)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default: True)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public (overrides --private)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the test file after completion",
    )

    args = parser.parse_args()

    private = args.private and not args.public
    test_file = Path("test_upload_temp.json")

    try:
        # Step 1: Create and upload first version
        logger.info("=" * 60)
        logger.info("STEP 1: Creating and uploading first version")
        logger.info("=" * 60)
        data_v1 = create_test_json(test_file, version=1)
        success_v1 = upload_file_to_hub(
            test_file, args.repo, args.file, private=private
        )

        if not success_v1:
            logger.error("Failed to upload first version. Exiting.")
            return

        # Wait a bit to ensure upload completes
        logger.info("Waiting 3 seconds for upload to complete...")
        time.sleep(3)

        # Step 2: Download and verify first version
        logger.info("=" * 60)
        logger.info("STEP 2: Downloading and verifying first version")
        logger.info("=" * 60)
        downloaded_v1 = download_file_from_hub(args.repo, args.file)
        if downloaded_v1:
            logger.info(
                f"First version content: "
                f"{json.dumps(downloaded_v1, indent=2)}"
            )

        # Step 3: Modify and upload second version
        logger.info("=" * 60)
        logger.info("STEP 3: Creating and uploading second version (override)")
        logger.info("=" * 60)
        data_v2 = create_test_json(test_file, version=2)
        success_v2 = upload_file_to_hub(
            test_file, args.repo, args.file, private=private
        )

        if not success_v2:
            logger.error("Failed to upload second version. Exiting.")
            return

        # Wait a bit to ensure upload completes
        logger.info("Waiting 3 seconds for upload to complete...")
        time.sleep(3)

        # Step 4: Download and verify override worked
        logger.info("=" * 60)
        logger.info("STEP 4: Downloading and verifying override")
        logger.info("=" * 60)
        downloaded_v2 = download_file_from_hub(args.repo, args.file)
        if downloaded_v2:
            content_str = json.dumps(downloaded_v2, indent=2)
            logger.info(f"Second version content: {content_str}")

        # Step 5: Verify override
        logger.info("=" * 60)
        logger.info("STEP 5: Verifying override worked")
        logger.info("=" * 60)
        if verify_override(data_v1, downloaded_v2):
            logger.info(
                "✓ SUCCESS: Override worked! Second upload replaced first."
            )
        else:
            logger.error("✗ FAILED: Override did not work as expected.")

        # Summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Repository: {args.repo}")
        logger.info(f"File: {args.file}")
        logger.info(f"First upload version: {data_v1.get('version')}")
        logger.info(f"Second upload version: {data_v2.get('version')}")
        if downloaded_v2:
            logger.info(
                f"Downloaded version: {downloaded_v2.get('version')}"
            )
            if downloaded_v2.get("version") == 2:
                logger.info("✓ Override test PASSED")
            else:
                logger.error("✗ Override test FAILED")
        else:
            logger.error("✗ Could not verify override")

    finally:
        # Cleanup
        if args.cleanup and test_file.exists():
            test_file.unlink()
            logger.info(f"Cleaned up temporary file: {test_file}")


if __name__ == "__main__":
    main()
