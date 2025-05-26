# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

RUN_NAME="SFT-lora-acc_0.997-Text-Biased"

python -u merge_lora.py \
    --baseline_model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter_model_path $TEXT_BIASED_OUT_PATH/$RUN_NAME/final \
    --output_path $TEXT_BIASED_OUT_PATH/$RUN_NAME/merged