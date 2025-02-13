#!/bin/bash

# Default tag is 'latest' if no argument is provided
TAG=${1:-latest}
REGION="ap-south-1"
ACCOUNT="954976316440"
ECR_REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/mf-test:${TAG}"

echo "Authenticating with ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

if [ $? -ne 0 ]; then
    echo "ECR authentication failed"
    exit 1
fi

echo "Building and pushing with tag: ${TAG}"

# Build the image
docker build -t $ECR_REPO .

if [ $? -ne 0 ]; then
    echo "Docker build failed"
    exit 1
fi

# Push to ECR
docker push $ECR_REPO

if [ $? -ne 0 ]; then
    echo "Docker push failed"
    exit 1
fi

echo "Successfully built and pushed ${ECR_REPO}" 