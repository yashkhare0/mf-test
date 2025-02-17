# MF-Test: Docker & SageMaker Training Setup

This repository provides everything you need to build a Docker image, push it to AWS ECR, set up your Python environment, and launch a training job using SageMaker. Detailed instructions for each step, including environment setup, Docker commands, and SageMaker SDK installation, are provided below.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Image Build & Push](#docker-image-build--push)
3. [Python Virtual Environment Setup](#python-virtual-environment-setup)
4. [Launching Training](#launching-training)
5. [SageMaker SDK Installation](#sagemaker-sdk-installation)
6. [Repository Details](#repository-details)
7. [Running Docker with Mounted Volumes](#running-docker-with-mounted-volumes)
8. [Additional Resources](#additional-resources)

---

## Prerequisites

- **Docker:** Installed and properly configured.
- **AWS CLI:** Installed and configured with the necessary permissions.
- **Python 3.8+** (or later) installed, with support for virtual environments.
- **AWS ECR Access:** Ensure you have access to create repositories and push images in the `ap-south-1` region.

---

## Docker Image Build & Push

### 1. Create an ECR Repository

Create an ECR repository named `mf-test` in the `ap-south-1` region:

```bash
aws ecr create-repository --repository-name mf-test --region ap-south-1
```

### 2. Build the Docker Image

Build your Docker image locally with the tag `mf-test`:

```bash
docker build -t mf-test .
```

### 3. Login to ECR

Login to your AWS ECR account. Make sure to use the correct account ID (the example below uses `763104351884` for login):

```bash
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ap-south-1.amazonaws.com
```

### 4. Tag and Push the Docker Image

Tag your image for the target repository and push it. (Note the use of a different account ID `954976316440` in this example.)

```bash
docker tag mf-test:latest 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest
docker push 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest
```

---

## Python Virtual Environment Setup

Set up your Python virtual environment and install the necessary package:

```bash
python -m venv .venv --prompt mf-test
source .venv/bin/activate
pip install boto3
```

This creates an isolated environment named `.venv` with `boto3` installed for AWS interactions.

---

## Launching Training

With your environment set up, launch the training job by executing:

```bash
python launch_training.py
```

This script is responsible for initiating your training workflow.

---

## SageMaker SDK Installation

Install the patched version of the SageMaker SDK (which includes the estimator fix) directly from the GitHub repository:

```bash
pip install git+https://github.com/yashkhare0/sagemaker-python-sdk-estimator-fix.git
```

---

## Repository Details

For reference, here is an example of the repository details obtained from AWS ECR:

```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:ap-south-1:954976316440:repository/mf-test",
        "registryId": "954976316440",
        "repositoryName": "mf-test",
        "repositoryUri": "954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test",
        "createdAt": 1739356900.054,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```

This JSON provides details such as the repository ARN, URI, and configurations like tag mutability and encryption.

---

## Running Docker with Mounted Volumes

To run your Docker container while mounting local directories for code, models, datasets, and output, use the command below. This command sets the necessary environment variables for SageMaker:

```bash
docker run -v $(pwd):/opt/ml/code \
-v $(pwd)/data/models:/opt/ml/model \
-v $(pwd)/data/datasets/train:/opt/ml/input/data/train \
-v $(pwd)/data/datasets/test:/opt/ml/input/data/test \
-v $(pwd)/data/runs:/opt/ml/output \
-e SM_MODEL_DIR=/opt/ml/model \
-e SM_INPUT_DIR=/opt/ml/input/data \
-e SM_OUTPUT_DIR=/opt/ml/output \
-e SM_LOG_DIR=/opt/ml/output \
-e SM_CHANNEL_TRAIN=/opt/ml/input/data/train \
-e SM_CHANNEL_TEST=/opt/ml/input/data/test \
mf-test:latest
```

### Volume Mapping Details

- **Code Directory:** `$(pwd)` → `/opt/ml/code`
- **Model Directory:** `$(pwd)/data/models` → `/opt/ml/model`
- **Training Data:** `$(pwd)/data/datasets/train` → `/opt/ml/input/data/train`
- **Test Data:** `$(pwd)/data/datasets/test` → `/opt/ml/input/data/test`
- **Output Directory:** `$(pwd)/data/runs` → `/opt/ml/output`

Environment variables such as `SM_MODEL_DIR`, `SM_INPUT_DIR`, and `SM_OUTPUT_DIR` are configured to inform the container where to read and write data.

---

## Additional Resources

- [AWS ECR Documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html)
- [Docker Documentation](https://docs.docker.com/)
- [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
- [SageMaker Python SDK Repository](https://github.com/yashkhare0/sagemaker-python-sdk-estimator-fix)

---

Feel free to modify these instructions as needed for your specific setup. If you encounter any issues or have questions, please open an issue in the repository or reach out for support.

Happy Training!

## Challenges

1. The training job is not starting

    - Resolved with the following patch: sagemaker-python-sdk-estimator-fix[https://github.com/aws/sagemaker-python-sdk/pull/4970]
    - Install the patched version of the SageMaker SDK using the following command:

```bash
pip install git+https://github.com/yashkhare0/sagemaker-python-sdk-estimator-fix.git
```

## Docker Run

```bash
docker run -it 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:1.0.12
```
