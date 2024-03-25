## Set Up
create a conda environment for this homework
```bash
module load anaconda
conda create --name ssm_hw7 python=3.9.18
```
and install the required packages
```bash
cd hw7
conda activate ssm_hw7
pip install -r requirements.txt
```

Create [wandb.ai](https://wandb.ai/home) account and use the given command to login in terminal
```bash
wandb login
```

## Experiment
The trained models and predictions will be saved under `./checkpoints`.