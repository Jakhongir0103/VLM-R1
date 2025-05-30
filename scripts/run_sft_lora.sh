# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="Qwen2.5-VL-3B-SFT-drivingvqa-5_epoch"

python -u sft.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name $DATA_PATH/drivingvqa \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --save_only_model false \
    --push_to_hub=false \
    --learning_rate 1e-7 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM

    # --fp16 \
    # --torch_dtype float16 \
