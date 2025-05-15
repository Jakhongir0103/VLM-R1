# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

MODEL_DIR=/scratch/izar/delsad/vlm_r1/output

python -u evaluate_SmolVLM.py \
    --model_path $MODEL_DIR/SmolVLM-3B-SFT-lora/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/SmolVLM/sft

