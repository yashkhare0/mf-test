# Model Fine-tuning Report

## Model Details

- **Base Model**: Mistral-7B-v0.3
- **Training Type**: Fine-tuning with LoRA
- **LoRA Configuration**:
  - Rank: 8

## Training Configuration

### Data Settings

- Training data path: `/opt/ml/input/data/train`
- Evaluation data path: `/opt/ml/input/data/test`
- Sequence Length: 128 tokens
- Batch Size: 4

### Training Parameters

- Maximum Steps: 1,000
- Learning Rate: 6e-5
- Weight Decay: 0.1
- Learning Rate Warmup: 5% of total steps
- Random Seed: 0

### Optimization

- Learning Rate Schedule:
  - Percentage Start: 5% (warmup period)
- Weight Decay: 0.1

### Checkpointing and Evaluation

- Checkpoint Frequency: Every 100 steps
- Evaluation Frequency: Every 100 steps
- Logging Frequency: Every step
- Saving Method: Adapter weights only

### Infrastructure

- Output Directory: `/opt/ml/output/data`
- Model Directory: `/opt/ml/model`

### Monitoring

- **Weights & Biases Integration**:

  - Project Name: Configured
  - Run Name: "mistral-7b-sagemaker"
  - Mode: Offline tracking
  
## Notes

- The training setup uses SageMaker infrastructure
- LoRA fine-tuning is employed to efficiently adapt the model while maintaining reasonable memory usage
- Adapter weights are saved separately from the base model

## Results

Please check the initial version of the model training metrics in the wandb report: WANDB_REPORT[https://api.wandb.ai/links/yash-khare010-zof/5t5a5pk7] for the results.
