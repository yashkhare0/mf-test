#!/bin/bash

# Default tag is 'latest' if no argument is provided
TAG=${1:-latest}
ECR_REPO="954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:${TAG}"

echo "Building and pushing with tag: ${TAG}"

# Build the image
docker build -t $ECR_REPO .

# Push to ECR
docker push $ECR_REPO 