import sagemaker
from sagemaker.estimator import Estimator
import logging
import sys
from datetime import datetime
from sagemaker.huggingface import HuggingFace
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'sagemaker_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('sagemaker_training')

session = sagemaker.Session()

try:
    logger.info("Initializing SageMaker training job")
    logger.info("Creating SageMaker estimator...")
    

    huggingface_estimator = HuggingFace(
        image_uri='954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest',
        entry_point='train.py',  # Your training script
        source_dir='./',  # Directory where train.py is located
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role='arn:aws:iam::954976316440:role/MistralFineTuneRole',
        transformers_version='4.6.1',
        pytorch_version='1.7.1',
        py_version='py36',
        hyperparameters={
            'epochs': 3,
            'batch_size': 8,
            'max_length': 128,
        },
        environment={
            'MODEL_DATA_URI': 's3://sagemaker-ap-south-1-954976316440/sagemaker/models/mistral-7b-v0.3'
        }
    )

    # estimator = HuggingFace(
    #     image_uri='954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest',
    #     role='arn:aws:iam::954976316440:role/MistralFineTuneRole',
    #     instance_count=1,
    #     instance_type='ml.m5.xlarge', # 'local' for local testing
    #     output_path='s3://sagemaker-ap-south-1-954976316440/sagemaker/output',
    #     volume_size=100,
    #     max_run=24*60*60,
    #     environment={
    #         'WANDB_PROJECT': 'mf-test',
    #         'WANDB_MODE': 'online'
    #     }
    # )
    #logger.info(f"Estimator created with output path: {estimator.output_path}")

    data_channels = {
        'train': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test_gen-00000-of-00001-3d4cd8309148a71f/training',
        'test': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test_gen-00000-of-00001-3d4cd8309148a71f/evaluation',
        'model': 's3://sagemaker-ap-south-1-954976316440/sagemaker/models/mistral-7b-v0.3'
    }

    logger.info("Data channels configured:")
    for channel, path in data_channels.items():
        logger.info(f"  {channel}: {path}")

    logger.info("Starting training job...")

    # estimator.fit(inputs=data_channels, wait=True)
    # estimator.fit('s3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test_gen-00000-of-00001-3d4cd8309148a71f/training/train.jsonl', wait=True)
    huggingface_estimator.fit(data_channels)

    logger.info("Training job completed successfully")

    logger.info(f"Training job name: {huggingface_estimator.latest_training_job.job_name}")
    logger.info(f"Training job status: {huggingface_estimator.latest_training_job.describe()['TrainingJobStatus']}")
    logger.info(f"CloudWatch logs: {huggingface_estimator.latest_training_job.describe()['LogUrl']}")

except Exception as e:
    logger.error(f"Training failed with error: {str(e)}", exc_info=True)
    raise