import sagemaker
from sagemaker.estimator import Estimator
import logging
import sys
from datetime import datetime
import boto3
import time
from sagemaker.local import LocalSession

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/sagemaker_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('sagemaker_training')

local_session = sagemaker.Session()

# Add CloudWatch client
cloudwatch = boto3.client('cloudwatch')

class CloudWatchCallback(object):
    def __init__(self, job_name, namespace="SageMaker/Training"):
        self.job_name = job_name
        self.namespace = namespace
        
    def put_metric(self, metric_name, value, unit='None'):
        try:
            cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Dimensions': [
                            {
                                'Name': 'TrainingJobName',
                                'Value': self.job_name
                            }
                        ]
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Failed to put CloudWatch metric: {str(e)}")

try:
    local_session = LocalSession()
except Exception as e:
    logger.error(f"Failed to create LocalSession: {str(e)}")
    raise

try:
    logger.info("Initializing SageMaker training job")
    logger.info("Creating SageMaker estimator...")
    
    estimator = Estimator(
        base_job_name='mf-test-job',
        image_uri='954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest',
        role='arn:aws:iam::954976316440:role/MistralFineTuneRole',
        instance_count=1,
        instance_type= 'local_gpu', #'ml.c4.2xlarge',	#'ml.g4dn.2xlarge',#'ml.m5.xlarge', # 'local' for local testing # ml.g5.xlarge
        output_path='s3://sagemaker-ap-south-1-954976316440/sagemaker/output',
        volume_size=100,
        max_run=24*60*60,
        environment={
            'WANDB_PROJECT': 'mf-test',
            'WANDB_MODE': 'online'
        },
        # sagemaker_session = local_session
        entry_point='train.py',
        source_dir='.'
    )
    logger.info(f"Estimator created with output path: {estimator.output_path}")

    data_channels = {
        'train': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test_gen-00000-of-00001-3d4cd8309148a71f/training',
        'test': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test_gen-00000-of-00001-3d4cd8309148a71f/evaluation',
        'model': 's3://sagemaker-ap-south-1-954976316440/sagemaker/models/mistral-7b-v0.3/model.tar.gz'
    }

    logger.info("Data channels configured:")
    for channel, path in data_channels.items():
        logger.info(f"  {channel}: {path}")

    logger.info("Starting training job...")
    
    job_name = f"mf-test-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    estimator.fit(
        inputs=data_channels, 
        wait=True, 
        logs=True, 
        job_name=job_name
    )
    logger.info()
    
    job_name = estimator.latest_training_job.job_name
    cloudwatch_callback = CloudWatchCallback(job_name)
    
    status = None
    while status not in ['Completed', 'Failed', 'Stopped']:
        description = estimator.latest_training_job.describe()
        new_status = description['TrainingJobStatus']
        
        if new_status != status:
            status = new_status
            logger.info(f"Training job status: {status}")
            
            # Log metrics to CloudWatch
            if 'ResourceConfig' in description:
                cloudwatch_callback.put_metric(
                    'InstanceCount',
                    description['ResourceConfig']['InstanceCount'],
                    'Count'
                )
            
            if 'BillableTimeInSeconds' in description:
                cloudwatch_callback.put_metric(
                    'BillableTime',
                    description['BillableTimeInSeconds'],
                    'Seconds'
                )
                
            if 'MetricData' in description:
                for metric in description.get('MetricData', []):
                    cloudwatch_callback.put_metric(
                        metric['MetricName'],
                        metric['Value'],
                        metric.get('Unit', 'None')
                    )
        
        time.sleep(30)  # Check status every 30 seconds
    
    if status == 'Completed':
        logger.info("Training job completed successfully")
    else:
        logger.error(f"Training job ended with status: {status}")

    logger.info(f"Training job name: {job_name}")
    logger.info(f"CloudWatch logs: {description}")

except Exception as e:
    logger.error(f"Training failed with error: {str(e)}", exc_info=True)
    raise