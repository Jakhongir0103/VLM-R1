# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

# MODEL_DIR=/scratch/izar/delsad/vlm_r1_text_bias/output
MODEL_DIR=/scratch/izar/delsad/vlm_r1_text_bias/output

python -u evaluate.py \
    --model_path $MODEL_DIR/SFT-lora-acc_extreme_small-Text-Biased/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/sft-acc_extreme_small-Text-Biased

