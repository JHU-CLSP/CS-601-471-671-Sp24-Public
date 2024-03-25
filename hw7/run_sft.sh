#!/bin/bash

#SBATCH -A cs601471
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="hw7 test"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate ssm_hw7 # activate the Python environment

# initialize the policy model
python ./run_sft.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --bf16 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file ./data/train_1k.json \
    --validation_file ./data/dev.json \
    --test_file ./data/test.json \
    --output_dir ./checkpoints/sft \
    --cache_dir ./checkpoints \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=128 \
    --predict_with_generate \
    --generation_max_length 200 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --report_to wandb \
    --metric_for_best_model rougeLsum
