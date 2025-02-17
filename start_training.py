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
import argparse

# Add this after imports but before any code
BASE_JOB_NAME = 'mf-test-job'
ROLE = 'arn:aws:iam::954976316440:role/MistralFineTuneRole'
OUTPUT_PATH = 's3://sagemaker-ap-south-1-954976316440/sagemaker/output'


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
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    log_filename = f"logs/estimator/sagemaker_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return logger

logger = configure_logging()

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

def create_estimator(local:bool=True,
                     session:Union[sagemaker.Session,
                                  LocalSession]=None,
                     image_uri:str=None,
                     version:str='unknown'):
    """Create and return a SageMaker estimator for training."""
    logger.info(f"Using session type: {type(session)}")
    job_name = f"{BASE_JOB_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    instance_type = 'local_gpu' if local else 'ml.g5.xlarge' #'ml.m5.xlarge'
    logger.info(f"Creating SageMaker estimator with instance type: {instance_type}")
    logger.info(f"Using container image: {image_uri}")
    logger.info(f"Running in {'local' if local else 'SageMaker'} mode")
    try:
        estimator = Estimator(
            base_job_name=BASE_JOB_NAME,
            image_uri=image_uri,
            role=ROLE,
            instance_count=1,
            instance_type=instance_type, 
            output_path=OUTPUT_PATH,
            volume_size=100,
            max_run=24*60*60,
            environment={
                'WANDB_PROJECT': 'mf-test',
                'WANDB_MODE': 'offline' if local else 'online',
                'RUN_DIR': f'{OUTPUT_PATH}/{job_name}',
                'IMAGE_VERSION': version
            },
        )
        logger.info("Estimator created successfully")
        return estimator, job_name
    except Exception as e:
        logger.error(f"Failed to create estimator: {str(e)}", exc_info=True)
        raise

def get_and_verify_data_channels(data_channels_path):
    import yaml
    """Return a dictionary of S3 data channels."""
    try:
        data_channels = yaml.safe_load(open(data_channels_path))
        logger.info("Data channels configured")
        logger.info(f"{channel}: {path}" for channel, path in data_channels.items())
        verify_s3_files(data_channels)
        return data_channels
    except Exception as e:
        logger.error(f"Failed to get and verify data channels: {str(e)}", exc_info=True)
        raise

def verify_s3_files(data_channels):
    """Verify that the S3 files exist for each channel."""
    def _logger_info(channel_name, s3_uri, bucket, key, error_msg, error_code):
        logger.info(f"Verifying {channel_name} channel:")
        logger.info(f"  URI: s3://{bucket}/{key}")
        logger.info(f"  Error: {error_msg}")
        logger.info(f"  Error Code: {error_code}")
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
            logger.info(f"{channel_name} verified: {float(response['ContentLength'])/1000/1000} MB,{response.get('ContentType', 'N/A')}")
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            if error_code == "404":
                _logger_info(channel_name, s3_uri, bucket, key, error_msg, error_code)
                raise
            elif error_code == "403":
                _logger_info(channel_name, s3_uri, bucket, key, error_msg, error_code)
                raise
            else:
                _logger_info(channel_name, s3_uri, bucket, key, error_msg, error_code)
                raise
    return True

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Run in local mode')
    parser.add_argument('--data-channels', type=str, help='Path to data-channels.yaml file', default='config/data-channels.yaml')
    parser.add_argument('--version', type=str, default='1.0.14', help='Container version tag')
    args = parser.parse_args()
    
    image_uri = f'954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:{args.version}'
    logger.info(f"Using container image: {image_uri}")
    logger.info("=== Starting Training Process ===")
    try:
        session = create_local_session(local=args.local)
        data_channels = get_and_verify_data_channels(args.data_channels)
        estimator, job_name = create_estimator(args.local, session=session, image_uri=image_uri, version=args.version)
        estimator.fit(
            inputs=data_channels,
            wait=True,
            logs=True,
            job_name=job_name
        )
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
