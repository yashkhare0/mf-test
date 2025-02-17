#!/bin/bash
set -e

TAG=$1
REGION="ap-south-1"
ACCOUNT_ID="954976316440"
REPO_NAME="mf-test"

# Authenticate with your ECR repository
echo "Authenticating with ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Authenticate with AWS Deep Learning Container repository
echo "Authenticating with AWS DLC repository..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

echo "Building and pushing with tag: ${TAG}"
docker build -t ${REPO_NAME}:${TAG} .
docker tag ${REPO_NAME}:${TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}

if [ $? -ne 0 ]; then
    echo "Docker push failed"
    exit 1
fi

echo "Successfully built and pushed ${REPO_NAME}:${TAG}" 