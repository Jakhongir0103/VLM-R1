# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="Qwen2.5-VL-3B-GRPO-lora"

export DEBUG_MODE="true"
export LOG_PATH="$BASE_LOG_PATH/debug_log_$RUN_NAME.txt"

python -u grpo.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name $DATA_PATH/images/vsr \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --fp16 \
    --torch_dtype float16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 0.05 \
    --save_only_model true \
    --push_to_hub=false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


