#!/usr/bin/env python3

import os
import sys
import subprocess
import yaml
import tarfile
from sagemaker_training import environment
import shutil
import logging
import gzip

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('training')

logger.info("==== TRAIN.PY STARTED ====")
logger.info(f"Current Directory: {os.getcwd()}")
logger.info(f"Files in Current Directory: {os.listdir('.')}")
logger.info("Environment Variables:")
for k, v in os.environ.items():
    logger.info(f"{k}: {v}")

logger.info(f"Python Version: {sys.version}")
logger.info(f"Python Path: {sys.executable}")

def extract_model(model_path, extract_path):
    """Extract tar.gz model file"""
    logger.info(f"Extracting model from {model_path} to {extract_path}")
    try:
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        with open(model_path, 'rb') as f:
            magic_bytes = f.read(4)
        logger.info(f"File magic bytes: {magic_bytes!r}")

        try:
            with tarfile.open(model_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
        except gzip.BadGzipFile:
            logger.info("Not a gzip file, trying regular tar...")
            with tarfile.open(model_path, 'r:') as tar:
                tar.extractall(path=extract_path)
        except Exception as e:
            logger.info("Not a tar file, trying to copy directly...")
            logger.info(f"Error: {e}")
            shutil.copy2(model_path, os.path.join(extract_path, 'model.bin'))
            return extract_path
        extracted_dirs = os.listdir(extract_path)
        logger.info(f"Extracted contents: {extracted_dirs}")
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

def setup_data_directory():
    """Setup the expected directory structure for the data"""
    logger.info("Setting up data directory structure")
    base_dir = "/opt/ml/code/datasets/test_gen-00000-of-00001-3d4cd8309148a71f"
    os.makedirs(base_dir, exist_ok=True)
    
    # Get the SageMaker paths
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    eval_path = os.environ.get('SM_CHANNEL_TEST')
    model_path = os.environ.get('SM_CHANNEL_MODEL')
    logger.info(f"Training path: {train_path}")
    logger.info(f"Evaluation path: {eval_path}")
    logger.info(f"Model path: {model_path}")
    try:
        shutil.copy(train_path, os.path.join(base_dir, 'train.jsonl'))
        shutil.copy(eval_path, os.path.join(base_dir, 'eval.jsonl'))
        logger.info("Data files copied successfully")
    except Exception as e:
        logger.error(f"Failed to copy data files: {str(e)}", exc_info=True)
        raise
    
    return base_dir

def train():
    logger.info("Starting training process")
    env = environment.Environment()
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    logger.info(f"Model directory: {model_dir}")
    
    try:
        data_dir = setup_data_directory()
        logger.info(f"Data directory setup complete: {data_dir}")
        logger.info("Loading configuration")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        model_path = os.environ.get('SM_CHANNEL_MODEL')
        if model_path:
            logger.info(f"Processing model from: {model_path}")
            logger.info(f"Contents of model path directory: {os.listdir(model_path)}")
            model_files = [f for f in os.listdir(model_path) if f.endswith('.tar.gz')]
            logger.info(f"Found model files: {model_files}")
            if not model_files:
                logger.error("No .tar.gz file found in model directory")
                raise ValueError("No .tar.gz file found in model directory")
            model_archive = os.path.join(model_path, model_files[0])
            logger.info(f"Full path to model archive: {model_archive}")
            local_model_path = '/opt/ml/code/models/Mistral-7B'
            os.makedirs(local_model_path, exist_ok=True)
            extracted_path = extract_model(model_archive, local_model_path)
            config['model_id_or_path'] = extracted_path
        else:
            logger.info("No model provided in SM_CHANNEL_MODEL, using HuggingFace model")
            config['model_id_or_path'] = "mistralai/Mistral-7B-v0.3"
        config['run_dir'] = model_dir
        logger.info("Saving updated configuration")
        with open('training_config.yaml', 'w') as f:
            yaml.dump(config, f)
        logger.info("Starting training with config:")
        logger.info(yaml.dump(config))
        
        logger.info("Launching Mistral training script")
        result = subprocess.run([
            "python", 
            "/opt/ml/code/mistral-finetune/train.py",
            "--config", "training_config.yaml"
        ], check=True, capture_output=True, text=True)
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