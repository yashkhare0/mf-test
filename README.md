# mf-test

Fine-tuning pipeline for **Mistral 7B v0.3** using [mistral-finetune](https://github.com/mistralai/mistral-finetune), orchestrated via **AWS SageMaker** with Docker.

## What It Does

This project packages a Mistral 7B LoRA fine-tuning job into a Docker container that runs on SageMaker. The pipeline:

1. Builds a Docker image based on the PyTorch 2.5.1 GPU training image from AWS ECR
2. Clones Mistral's official `mistral-finetune` repository into the container
3. Copies training/evaluation JSONL data from S3 into the expected directory layout
4. Runs distributed training via `torchrun` (multi-GPU) or falls back to single-process execution
5. Saves LoRA adapters to the SageMaker model output directory

Training is tracked with **Weights & Biases** and monitored via **CloudWatch** custom metrics.

## Tech Stack

- **Model**: Mistral 7B v0.3 (LoRA fine-tuning, rank 8)
- **Framework**: PyTorch 2.2, xformers 0.0.24, triton 2.2
- **Infrastructure**: AWS SageMaker, ECR, S3
- **Monitoring**: Weights & Biases, CloudWatch
- **Container**: Docker (based on `pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2`)

## Project Structure

```
├── train.py              # SageMaker entry point — sets up data, launches mistral-finetune
├── start_training.py     # SageMaker Estimator launcher (local or cloud mode)
├── config.yaml           # Mistral fine-tuning config (LoRA rank, seq_len, batch size, etc.)
├── data-channels.yaml    # S3 paths for train/eval data and base model
├── Dockerfile            # Container definition
├── build_and_push.sh     # ECR build and push script
└── requirements.txt      # Python dependencies
```

## Training Configuration

Key parameters from `config.yaml`:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| Sequence length | 128 |
| Batch size | 4 |
| Max steps | 1,000 |
| Learning rate | 6e-5 |
| Weight decay | 0.1 |
| Eval frequency | Every 100 steps |
| Checkpoint frequency | Every 100 steps |

## Usage

### Prerequisites

- AWS CLI configured with appropriate IAM permissions
- Docker installed
- Python 3.11+ with a virtual environment

### Build and Push the Container

```bash
docker build -t mf-test .
# Tag and push to your ECR repository
```

### Launch Training

```bash
python -m venv .venv --prompt mf-test
source .venv/bin/activate
pip install boto3 sagemaker sagemaker-training

# Run on SageMaker
python start_training.py

# Run locally (requires GPU + Docker)
python start_training.py --local
```

### Data Format

Training and evaluation data should be JSONL files stored in S3. The paths are configured in `data-channels.yaml`.

## Notes

- The project uses a [custom fork of the SageMaker Python SDK](https://github.com/yashkhare0/sagemaker-python-sdk-estimator-fix) to fix an estimator issue.
- LoRA adapters (not full model weights) are saved to reduce storage and training cost.

## License

MIT
