ARG REGION=ap-south-1
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2
# FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-ec2
# FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#FROM 954976316440.dkr.ecr.ap-south-1.amazonaws.com/mf-test:latest


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH=/opt/conda/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

    
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install sagemaker-training
RUN pip install wandb

# Clone the Mistral finetuning repository
RUN git clone https://github.com/mistralai/mistral-finetune.git mistral-finetune

# Create necessary directories
RUN mkdir -p models
RUN mkdir -p datasets/test_gen-00000-of-00001-3d4cd8309148a71f

COPY train.py .
COPY config.yaml .

# Make the training script executable
RUN chmod +x train.py

# Set the correct program name for SageMaker
ENV SAGEMAKER_PROGRAM=./train.py

CMD ["python", "train.py"]