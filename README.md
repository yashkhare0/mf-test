# Readme

1. Build and push the Docker image:

```bash
# Create ECR repository
aws ecr create-repository --repository-name mf-test --region ap-south-1

# Build the container
docker build -t mf-test .

# Login to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 954976316440.dkr.ecr.ap-south-1.amazonaws.com

# Tag and push
docker tag mf-test:latest 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest
docker push 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest
```

```bash
python -m venv .venv --prompt mf-test
source .venv/bin/activate
pip install boto3
```

2. Launch the training:

```bash
python launch_training.py
```

## Others

```bash
repository : {
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