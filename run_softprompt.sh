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

RUN_NAME="softprompt_with_logits"

pixi run python -u notebooks/prompt_tunning/prompt_tunning_on_output.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataset_name "hovno" \
    --logging_steps 1 \
    --fp16 \
    --torch_dtype float16 \
    --seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 0.05 \
    --save_total_limit 3 \
    --save_only_model false \
    --push_to_hub=false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \