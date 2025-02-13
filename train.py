#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from sagemaker_training import environment # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('training')

def print_environment_info() -> None:
    """Print relevant environment information for debugging"""

    logger.info("==== TRAIN.PY 0.3 STARTED ====")
    logger.info(f"Current Directory: {os.getcwd()}")
    logger.info(f"Files in Current Directory: {os.listdir('.')}")
    logger.info("Environment Variables:")
    for k, v in os.environ.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Path: {sys.executable}")

def check_gpu_availability() -> int:
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


def setup_data_directory() -> str:
    """
    Setup the expected directory structure for the data
    
    Returns:
        str: Path to the base data directory
    """
    logger.info("Setting up data directory structure")
    base_dir = Path("/opt/ml/code/datasets/test_gen-00000-of-00001-3d4cd8309148a71f")
    base_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    try:
        shutil.copy2(
            train_files[0],
            base_dir / 'train.jsonl'
        )
        shutil.copy2(
            eval_files[0],
            base_dir / 'eval.jsonl'
        )
        logger.info("Data files copied successfully")
    except Exception as e:
        logger.error(f"Failed to copy data files: {str(e)}", exc_info=True)
        raise
    
    return str(base_dir)


def train() -> None:
    """Main training function that orchestrates the training process"""
    try:
        print_environment_info()
        env = environment.Environment()  # noqa: F841
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        logger.info(f"Model directory: {model_dir}")
        data_dir = setup_data_directory()
        logger.info(f"Data directory setup complete: {data_dir}")
        model_path = os.environ.get('SM_CHANNEL_MODEL')
        if model_path:
            logger.info(f"Using model from: {model_path}")
        else:
            logger.info("No model provided in SM_CHANNEL_MODEL, using HuggingFace model")
        training_script = Path("/opt/ml/code/mistral-finetune/train.py")
        if not training_script.exists():
            raise FileNotFoundError(f"Training script not found at {training_script}")
        num_gpus = check_gpu_availability()
        if num_gpus > 0:
            logger.info(f"Using torchrun with {num_gpus} GPUs")
            cmd = [
                "torchrun",
                "--nproc-per-node", str(num_gpus),
                str(training_script),
                "--config", "config.yaml"
            ]
        else:
            logger.info("No GPUs detected, using regular Python execution")
            cmd = [
                sys.executable,
                str(training_script),
                "--config", "config.yaml"
            ]
        
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