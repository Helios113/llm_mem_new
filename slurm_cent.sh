#!/bin/bash
#SBATCH -J LLM_finetune 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:1657

#! specify node
#SBATCH -w mauao
#SBATCH --output=slurm_out/%x.%j.ans
#SBATCH --error=slurm_out/err_%x.%j.ans


#WANDB_MODE=disabled 
HYDRA_FULL_ERROR=1 poetry run python centralised_training.py
# HYDRA_FULL_ERROR=1 poetry run python centralised_training.py

