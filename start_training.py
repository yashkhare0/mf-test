from typing import Union
import sagemaker
from sagemaker.estimator import Estimator
import logging
import sys
from datetime import datetime
import boto3
import botocore
import time
from sagemaker.local import LocalSession
import warnings

# Add this after imports but before any code
warnings.filterwarnings(
    "ignore",
    message="Field name .* shadows an attribute in parent .*",
    category=UserWarning,
    module="pydantic.*"
)

def configure_logging():
    """Configure and return a logger."""
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mf-test")
    logger.setLevel(logging.DEBUG)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    # File handler
    log_filename = f"logs/estimator/sagemaker_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = configure_logging()

# Initialize the CloudWatch client
cloudwatch = boto3.client('cloudwatch')

class CloudWatchCallback:
    def __init__(self, job_name, namespace="SageMaker/Training"):
        self.job_name = job_name
        self.namespace = namespace

    def put_metric(self, metric_name, value, unit='None'):
        try:
            cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Dimensions': [{
                        'Name': 'TrainingJobName',
                        'Value': self.job_name
                    }]
                }]
            )
        except Exception as e:
            logger.error(f"Failed to put CloudWatch metric: {e}")

def create_local_session(local:bool=True):
    """Create and return a local SageMaker session."""
    logger.info(f"Creating {'local' if local else 'SageMaker'} session...")
    try:
        if local:
            session = LocalSession()
            logger.info("LocalSession created successfully")
        else:
            session = sagemaker.Session()
            logger.info("SageMaker Session created successfully")
        logger.debug(f"Session details: {session}")
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}", exc_info=True)
        raise

def create_estimator(instance_type:str='local_gpu', session:Union[sagemaker.Session, LocalSession]=None):
    """Create and return a SageMaker estimator for training."""
    logger.info(f"Creating SageMaker estimator with instance type: {instance_type}")
    logger.info(f"Using session type: {type(session)}")
    try:
        estimator = Estimator(
            base_job_name='mf-test-job',
            image_uri='954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest',
            role='arn:aws:iam::954976316440:role/MistralFineTuneRole',
            instance_count=1,
            instance_type=instance_type, 
            output_path='s3://sagemaker-ap-south-1-954976316440/sagemaker/output',
            volume_size=100,
            max_run=24*60*60,
            environment={
                'WANDB_PROJECT': 'mf-test',
                'WANDB_MODE': 'online'
            },
        )
        logger.info("Estimator created successfully")
        return estimator
    except Exception as e:
        logger.error(f"Failed to create estimator: {str(e)}", exc_info=True)
        raise

def get_data_channels():
    """Return a dictionary of S3 data channels."""
    data_channels = {
        'train': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test-data/training/train.jsonl',
        'test': 's3://sagemaker-ap-south-1-954976316440/sagemaker/datasets/test-data/evaluation/eval.jsonl',
        'model': 's3://sagemaker-ap-south-1-954976316440/sagemaker/models/mistral-7b-v0.3/model.tar.gz'
    }
    logger.info("Data channels configured:")
    for channel, path in data_channels.items():
        logger.info(f"  {channel}: {path}")
    return data_channels

def verify_s3_files(data_channels):
    """Verify that the S3 files exist for each channel."""
    logger.info("Starting S3 file verification...")
    s3 = boto3.client('s3')
    for channel_name, s3_uri in data_channels.items():
        if not s3_uri.startswith('s3://'):
            logger.warning(f"Skipping verification for non-S3 URI: {s3_uri}")
            continue
        parts = s3_uri.split('/')
        bucket = parts[2]
        key = '/'.join(parts[3:])
        logger.info(f"Verifying {channel_name} channel:")
        try:
            response = s3.head_object(Bucket=bucket, Key=key)
            logger.info(f"File verification successful for {channel_name}:")
            try:
                logger.info(f"  Size: {float(response['ContentLength'])/1000/1000} MB")
            except Exception:
                logger.info(f"  Size: {response['ContentLength']} bytes")
            logger.info(f"  Content Type: {response.get('ContentType', 'N/A')}")
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            if error_code == "404":
                logger.error(f"File not found for {channel_name}:")
                logger.error(f"  URI: s3://{bucket}/{key}")
                logger.error(f"  Error: {error_msg}")
                raise
            elif error_code == "403":
                logger.error(f"Access denied for {channel_name}:")
                logger.error(f"  URI: s3://{bucket}/{key}")
                logger.error(f"  Error: {error_msg}")
                logger.error("Please check AWS credentials and bucket permissions")
                raise
            else:
                logger.error(f"Unexpected error for {channel_name}:")
                logger.error(f"  URI: s3://{bucket}/{key}")
                logger.error(f"  Error Code: {error_code}")
                logger.error(f"  Error Message: {error_msg}")
                raise

def monitor_training_job(estimator, job_name, cloudwatch_callback):
    """Poll and log the training job status until completion."""
    logger.info(f"Starting training job monitoring for job: {job_name}")
    status = None
    last_description = None
    
    while status not in ['Completed', 'Failed', 'Stopped']:
        try:
            description = estimator.latest_training_job.describe()
            new_status = description['TrainingJobStatus']
            if new_status != status:
                status = new_status
                logger.info(f"Job status changed to: {status}")
                logger.info(f"Secondary status: {description.get('SecondaryStatus', 'N/A')}")
                if 'FailureReason' in description:
                    logger.error(f"Failure reason: {description['FailureReason']}")
                if 'ResourceConfig' in description:
                    instance_count = description['ResourceConfig']['InstanceCount']
                    instance_type = description['ResourceConfig']['InstanceType']
                    logger.info(f"Running on {instance_count} x {instance_type}")
                    cloudwatch_callback.put_metric('InstanceCount', instance_count, 'Count')
                if 'BillableTimeInSeconds' in description:
                    billable_time = description['BillableTimeInSeconds']
                    logger.info(f"Billable time: {billable_time} seconds")
                    cloudwatch_callback.put_metric('BillableTime', billable_time, 'Seconds')
                if description != last_description:
                    logger.debug(f"Full job description: {description}")
                    last_description = description
            time.sleep(30)
        except Exception as e:
            logger.error(f"Error monitoring training job: {str(e)}", exc_info=True)
            raise
    return status, description

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Run in local mode')
    args = parser.parse_args()
    logger.info("=== Starting Training Process ===")
    logger.info(f"Running in {'local' if args.local else 'SageMaker'} mode")
    
    try:
        session = create_local_session(local=args.local)
        logger.info(f"Session created: {type(session)}")

        instance_type = 'local_gpu' if args.local else 'ml.m5.xlarge'
        logger.info(f"Using instance type: {instance_type}")
        
        estimator = create_estimator(instance_type=instance_type, session=session)
        logger.info("Estimator created successfully")
        
        data_channels = get_data_channels()
        logger.info("Data channels configured")
        
        verify_s3_files(data_channels)
        logger.info("S3 files verified successfully")
        
        job_name = f"mf-test-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting training job: {job_name}")
        
        estimator.fit(
            inputs=data_channels,
            wait=True,
            logs=True,
            job_name=job_name
        )
        job_name = estimator.latest_training_job.job_name
        logger.info(f"Training job started with name: {job_name}")
        cloudwatch_callback = CloudWatchCallback(job_name)
        status, description = monitor_training_job(estimator, job_name, cloudwatch_callback)
        if status == 'Completed':
            logger.info("=== Training Job Completed Successfully ===")
            logger.info(f"Model artifacts location: {estimator.model_data}")
        else:
            logger.error(f"=== Training Job Failed with Status: {status} ===")
        
        logger.info("=== Final Job Details ===")
        logger.info(f"Job Name: {job_name}")
        logger.info(f"Final Status: {status}")
        logger.debug(f"Full job description: {description}")
        
    except Exception as e:
        logger.error("=== Training Failed ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
