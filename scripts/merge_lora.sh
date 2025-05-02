# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="Qwen2.5-VL-3B-GRPO-lora"

python -u merge_lora.py \
    --baseline_model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter_model_path $OUT_PATH/$RUN_NAME/final \
    --output_path $OUT_PATH/$RUN_NAME/merged