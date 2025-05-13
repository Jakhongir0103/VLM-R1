#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH --time 2:00:00                                     # maximum time limit is 12h. We need to rerun the jobs every 12 hours.
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

cd $HOME/VLM-R1/                    # change this to where your repo is located
source ./.env
pixi run jupyter nbconvert --to notebook --execute --inplace --allow-errors notebooks/prompt_tunning/dataset_generation.ipynb