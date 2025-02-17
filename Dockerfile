ARG REGION=ap-south-1
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2

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

RUN pip install git+https://github.com/yashkhare0/sagemaker-python-sdk-estimator-fix.git

RUN pip install sagemaker-training

RUN git clone https://github.com/mistralai/mistral-finetune.git mistral-finetune

WORKDIR /opt/ml/code/mistral-finetune

RUN cd /opt/ml/code/mistral-finetune && \
    mkdir -p src/mistral_finetune && \
    mv finetune model utils src/mistral_finetune/ && \
    touch src/mistral_finetune/__init__.py && \
    mv train.py src/mistral_finetune/
    
RUN cd /opt/ml/code/mistral-finetune/src/mistral_finetune && \
    ln -s $(pwd)/finetune /opt/ml/code/mistral-finetune/src/finetune && \
    ln -s $(pwd)/utils /opt/ml/code/mistral-finetune/src/utils && \
    ln -s $(pwd)/model /opt/ml/code/mistral-finetune/src/model

COPY scripts/mft-setup.py /opt/ml/code/mistral-finetune/setup.py
RUN pip install -e .

ENV PYTHONPATH=""
ENV PYTHONPATH="/opt/ml/code/mistral-finetune/src:${PYTHONPATH}"

WORKDIR /opt/ml/code

COPY sagemaker_train.py .
COPY config/config.yaml .

RUN chmod +x sagemaker_train.py

ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV CUDA_VISIBLE_DEVICES="0"

ENV SAGEMAKER_PROGRAM=./sagemaker_train.py

# CMD ["python", "train.py"]