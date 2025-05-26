#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00                                     # maximum time limit is 12h. We need to rerun the jobs every 12 hours.
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

cd $HOME/VLM-R1/                    # change this to where your repo is located
source ./.env

RUN_NAME="softprompt_eval"

# pixi run python -u notebooks/prompt_tunning/claude_eval.py \
pixi run python -u notebooks/prompt_tunning/evaluate_with_probs.py \
    --softprompt_model /scratch/izar/vanousek/vlm_r1/output/softprompt_golden_seq/final/ \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /scratch/izar/vanousek/vlm_r1/results/prompt_tunning/
