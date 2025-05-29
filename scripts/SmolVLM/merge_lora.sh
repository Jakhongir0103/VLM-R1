# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="SmolVLM-2B-Instruct-SFT-Text-Biased-997"

python -u merge_lora_SmolVLM.py \
    --baseline_model_path HuggingFaceTB/SmolVLM-Instruct \
    --adapter_model_path $TEXT_BIASED_OUT_PATH/$RUN_NAME/final \
    --output_path $TEXT_BIASED_OUT_PATH/$RUN_NAME/merged