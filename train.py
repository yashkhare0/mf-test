#!/usr/bin/env python3

import os
import sys
import subprocess
import yaml
import tarfile
import shutil
import logging
import gzip
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

    logger.info("==== TRAIN.PY 0.2 STARTED ====")
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

def extract_model(model_path: str, extract_path: str) -> str:
    """
    Extract tar.gz model file
    
    Args:
        model_path: Path to the model archive
        extract_path: Path where model should be extracted
        
    Returns:
        str: Path to the extracted model
    """
    logger.info(f"Extracting model from {model_path} to {extract_path}")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        
        os.makedirs(extract_path, exist_ok=True)
        
        with open(model_path, 'rb') as f:
            magic_bytes = f.read(4)
        logger.info(f"File magic bytes: {magic_bytes!r}")

        try:
            with tarfile.open(model_path, 'r:gz') as tar:
                def is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar: tarfile.TarFile, path: str) -> None:
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted path traversal in tar file")
                    tar.extractall(path=path)

                safe_extract(tar, extract_path)
        except gzip.BadGzipFile:
            logger.info("Not a gzip file, trying regular tar...")
            with tarfile.open(model_path, 'r:') as tar:
                safe_extract(tar, extract_path)
        except Exception as e:
            logger.info("Not a tar file, trying to copy directly...")
            logger.info(f"Error: {e}")
            dest_path = os.path.join(extract_path, 'model.bin')
            shutil.copy2(model_path, dest_path)
            return extract_path
        extracted_dirs = os.listdir(extract_path)
        logger.info(f"Extracted contents: {extracted_dirs}")
        if not extracted_dirs:
            raise Exception("No files were extracted from the archive")
        if len(extracted_dirs) == 1:
            final_path = os.path.join(extract_path, extracted_dirs[0])
            logger.info(f"Model extracted successfully to {final_path}")
            return final_path    
        logger.info(f"Model extracted successfully to {extract_path}")
        return extract_path
        
    except Exception as e:
        logger.error(f"Failed to extract model: {str(e)}", exc_info=True)
        if os.path.exists(model_path):
            logger.info(f"Contents of model directory {os.path.dirname(model_path)}:")
            logger.info(os.listdir(os.path.dirname(model_path)))
        raise

def setup_data_directory() -> str:
    """
    Setup the expected directory structure for the data
    
    Returns:
        str: Path to the base data directory
    """
    logger.info("Setting up data directory structure")
    base_dir = Path("/opt/ml/code/datasets/test_gen-00000-of-00001-3d4cd8309148a71f")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the SageMaker paths
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    eval_path = os.environ.get('SM_CHANNEL_TEST')
    
    if not train_path or not eval_path:
        raise ValueError("SM_CHANNEL_TRAIN and SM_CHANNEL_TEST environment variables must be set")
        
    logger.info(f"Training path: {train_path}")
    logger.info(f"Evaluation path: {eval_path}")
    
    # Get all jsonl files from the directories
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

def load_and_validate_config(config_path: str) -> dict[str, any]:
    """
    Load and validate the configuration file
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dict[str, Any]: Loaded configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {config}")
    if not isinstance(config, dict):
        raise ValueError("Invalid config format - must be a YAML dictionary")
    required_keys = ['model_id_or_path', 'data'] # TODO: optimize this better later.
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    return config

def train() -> None:
    """Main training function that orchestrates the training process"""
    try:
        print_environment_info()
        env = environment.Environment()  # noqa: F841
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        logger.info(f"Model directory: {model_dir}")
        data_dir = setup_data_directory()
        logger.info(f"Data directory setup complete: {data_dir}")
        config = load_and_validate_config('config.yaml')
        model_path = os.environ.get('SM_CHANNEL_MODEL')
        if model_path:
            # logger.info(f"Processing model from: {model_path}")
            # model_files = list(Path(model_path).glob("*.tar.gz"))
            # if not model_files:
            #     raise ValueError("No .tar.gz file found in model directory")
            # model_archive = str(model_files[0])
            # local_model_path = Path('/opt/ml/code/models/Mistral-7B')
            # local_model_path.mkdir(parents=True, exist_ok=True)
            # extracted_path = extract_model(model_archive, str(local_model_path))
            logger.info(f"Using model from: {model_path}")
            config['model_id_or_path'] = model_path
        else:
            logger.info("No model provided in SM_CHANNEL_MODEL, using HuggingFace model")
            config['model_id_or_path'] = "mistralai/Mistral-7B-v0.3"
        config['run_dir'] = model_dir
        with open('training_config.yaml', 'w') as f:
            yaml.dump(config, f)
        logger.info("Current configuration:")
        logger.info(yaml.dump(config))
        
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
                "--config", "training_config.yaml"
            ]
        else:
            logger.info("No GPUs detected, using regular Python execution")
            cmd = [
                sys.executable,
                str(training_script),
                "--config", "training_config.yaml"
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