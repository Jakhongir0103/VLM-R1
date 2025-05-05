# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

MODEL_DIR=/scratch/izar/saydalie/vlm-r1/output

# python -u evaluate.py \
#     --model_path $MODEL_DIR/Qwen2.5-VL-3B-GRPO-lora/merged \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/saydalie/project/VLM-R1/results/vsr/grpo

python -u evaluate.py \
    --model_path $MODEL_DIR/Qwen2.5-VL-3B-SFT-lora/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/saydalie/project/VLM-R1/results/vsr/sft

# python -u evaluate.py \
#     --model_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/saydalie/project/VLM-R1/results/vsr/base