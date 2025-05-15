# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="SmolVLM-500M-Instruct-SFT"

python -u sft_SmolVLM.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path HuggingFaceTB/SmolVLM-500M-Instruct \
    --dataset_name $DATA_PATH/vsr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --fp16 \
    --torch_dtype bfloat16 \
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
    --lora_task_type CAUSAL_LM 