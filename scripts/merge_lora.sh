# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="SmolVLM-2B-Instruct-SFT"

python -u merge_lora_SmolVLM.py \
    --baseline_model_path HuggingFaceTB/SmolVLM-Instruct \
    --adapter_model_path $OUT_PATH/$RUN_NAME/final \
    --output_path $OUT_PATH/$RUN_NAME/merged