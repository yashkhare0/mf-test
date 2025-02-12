FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh


RUN git clone https://github.com/mistralai/mistral-finetune.git mistral-finetune

RUN mkdir -p models
RUN mkdir -p datasets/test_gen-00000-of-00001-3d4cd8309148a71f

COPY train.py .
COPY config.yaml .


ENV SAGEMAKER_PROGRAM=train.py
