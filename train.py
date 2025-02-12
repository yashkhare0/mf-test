import os
import sys
import subprocess
import yaml
import tarfile
from sagemaker_training import environment
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('training')

def extract_model(model_path, extract_path):
    """Extract tar.gz model file"""
    logger.info(f"Extracting model from {model_path} to {extract_path}")
    try:
        with tarfile.open(model_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        extracted_dirs = os.listdir(extract_path)
        if len(extracted_dirs) == 1:
            final_path = os.path.join(extract_path, extracted_dirs[0])
            logger.info(f"Model extracted successfully to {final_path}")
            return final_path
        logger.info(f"Model extracted successfully to {extract_path}")
        return extract_path
    except Exception as e:
        logger.error(f"Failed to extract model: {str(e)}", exc_info=True)
        raise

def setup_data_directory():
    """Setup the expected directory structure for the data"""
    logger.info("Setting up data directory structure")
    base_dir = "/opt/ml/code/datasets/test_gen-00000-of-00001-3d4cd8309148a71f"
    os.makedirs(base_dir, exist_ok=True)
    
    # Get the SageMaker paths
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    eval_path = os.environ.get('SM_CHANNEL_TEST')
    logger.info(f"Training path: {train_path}")
    logger.info(f"Evaluation path: {eval_path}")
    
    # Get all jsonl files from the directories
    train_files = [f for f in os.listdir(train_path) if f.endswith('.jsonl')]
    eval_files = [f for f in os.listdir(eval_path) if f.endswith('.jsonl')]
    
    if not train_files or not eval_files:
        logger.error("No .jsonl files found in data directories")
        raise ValueError("No .jsonl files found in data directories")
    
    logger.info(f"Found training files: {train_files}")
    logger.info(f"Found evaluation files: {eval_files}")
    
    # Copy files to the expected location
    try:
        shutil.copy(
            os.path.join(train_path, train_files[0]),
            os.path.join(base_dir, 'train.jsonl')
        )
        shutil.copy(
            os.path.join(eval_path, eval_files[0]),
            os.path.join(base_dir, 'eval.jsonl')
        )
        logger.info("Data files copied successfully")
    except Exception as e:
        logger.error(f"Failed to copy data files: {str(e)}", exc_info=True)
        raise
    
    return base_dir

def train():
    logger.info("Starting training process")
    # env = environment.Environment()
    
    # SageMaker specific directory structure
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Setup data directory structure
        data_dir = setup_data_directory()
        logger.info(f"Data directory setup complete: {data_dir}")
        
        # Load the config file
        logger.info("Loading configuration")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
        # Handle model loading
        model_path = os.environ.get('SM_CHANNEL_MODEL')
        if model_path:
            logger.info(f"Processing model from: {model_path}")
            model_files = [f for f in os.listdir(model_path) if f.endswith('.tar.gz')]
            if not model_files:
                logger.error("No .tar.gz file found in model directory")
                raise ValueError("No .tar.gz file found in model directory")
            
            model_archive = os.path.join(model_path, model_files[0])
            local_model_path = '/opt/ml/code/models/Mistral-7B'
            os.makedirs(local_model_path, exist_ok=True)
            extracted_path = extract_model(model_archive, local_model_path)
            config['model_id_or_path'] = extracted_path
        else:
            logger.info("No model provided in SM_CHANNEL_MODEL, using HuggingFace model")
            config['model_id_or_path'] = "mistralai/Mistral-7B-v0.3"
        
        # Update run directory in config
        config['run_dir'] = model_dir
        
        # Save updated config
        logger.info("Saving updated configuration")
        with open('training_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        logger.info("Starting training with config:")
        logger.info(yaml.dump(config))
        
        # Run the Mistral training script
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