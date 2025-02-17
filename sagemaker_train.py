#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path
from sagemaker_training import environment # type: ignore
import yaml

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('training')

# Constants
CONFIG_PATH = Path("/opt/ml/code/config.yaml")
TRAINING_SCRIPT_PATH = Path("/opt/ml/code/mistral-finetune/src/train.py")

def _print_environment_info() -> None:
    """Print relevant environment information for debugging"""

    logger.info(f"==== TRAIN.PY {os.environ.get('IMAGE_VERSION')} STARTED ====")
    logger.info(f"Current Directory: {os.getcwd()}")
    logger.info(f"Files in Current Directory: {os.listdir('.')}")
    logger.info("Environment Variables:")
    for k, v in os.environ.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Path: {sys.executable}")

def _check_gpu_availability() -> int:
    """
    Check if GPUs are available and return count
    
    Returns:
        int: Number of available GPUs
    """
    try:
        gpu_count_result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True
        )
        num_gpus = len(gpu_count_result.stdout.strip().split('\n'))
        logger.info(f"Detected {num_gpus} GPUs")
        return num_gpus
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.info(f"No GPUs detected or nvidia-smi not available: {e}")
        return 0


def _verify_model_directory() -> str:
    """
    Verify the model directory
    """
    model_dir = os.environ.get('SM_CHANNEL_MODEL')
    if not model_dir:
        raise ValueError("SM_CHANNEL_MODEL environment variable must be set")
    model_files = list(Path(model_dir).glob("*.safetensors"))
    if not model_files:
        model_files = list(Path(model_dir).glob("*.tar.gz"))
    if not model_files:
        raise ValueError("No model files found in SM_CHANNEL_MODEL")
    logger.info(f"Found model files: {model_files}")

def _verify_input_directories() -> str:
    """
    Setup the expected directory structure for the data
    
    Returns:
        str: Path to the base data directory
    """
    logger.info("Setting up data directory structure")
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    eval_path = os.environ.get('SM_CHANNEL_TEST')
    if not train_path or not eval_path:
        raise ValueError("SM_CHANNEL_TRAIN and SM_CHANNEL_TEST environment variables must be set")
    logger.info(f"Training path: {train_path}")
    logger.info(f"Evaluation path: {eval_path}")
    train_files = list(Path(train_path).glob("*.jsonl"))
    eval_files = list(Path(eval_path).glob("*.jsonl"))
    if not train_files or not eval_files:
        raise ValueError("No .jsonl files found in data directories")
    logger.info(f"Found training files: {train_files}")
    logger.info(f"Found evaluation files: {eval_files}")


def _process_config() -> None:
    config_path = Path("/opt/ml/code/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["run_dir"] = os.environ.get('RUN_DIR')
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def _process_constructor(num_gpus: int) -> list[str]:
    training_script = Path("/opt/ml/code/mistral-finetune/src/train.py")
    if not training_script.exists():
        raise FileNotFoundError(f"Training script not found at {training_script}")
    logger.info(f"Training script: {training_script} found")
    if num_gpus > 0:
        logger.info(f"Using torchrun with {num_gpus} GPUs")
        cmd = [
            "torchrun",
            "--nproc-per-node", str(num_gpus),
            "-m",
            "mistral_finetune.train",
            "config.yaml",
        ]
        # [
        #     "torchrun",
        #     "--nproc-per-node", str(num_gpus),
        #     str(training_script),
        #     "--config", "config.yaml"
        # ]
    else:
        logger.info("No GPUs detected, using regular Python execution")
        cmd = [
            "python",
            "-m",
            "mistral_finetune.train",
            "config.yaml",
        ]
        # [
        #     sys.executable,
        #     str(training_script),
        #     "--config", "config.yaml"
        # ]
    return cmd

def train() -> None:
    """Main training function that orchestrates the training process"""
    try:
        _print_environment_info()
        env = environment.Environment()  # noqa: F841
        _verify_input_directories()
        _verify_model_directory()
        _process_config()
        num_gpus = _check_gpu_availability()
        cmd = _process_constructor(num_gpus)
        logger.info(f"Launching training with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ)
        )
        logger.info("Training script output:")
        logger.info(result.stdout)
        if result.stderr:
            logger.warning("Training script stderr:")
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error("Training script failed with error:")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    train()