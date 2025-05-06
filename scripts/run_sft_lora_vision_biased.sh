# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

DATA_BIAS="acc_0.574"

RUN_NAME="SFT-lora-$DATA_BIAS-Vision-Biased"

python -u sft.py \
    --output_dir $VISION_BIASED_OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name $VISION_BIASED_DATA_PATH/$DATA_BIAS/vsr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --lora_task_type CAUSAL_LM