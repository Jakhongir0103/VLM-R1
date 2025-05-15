# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

MODEL_DIR=/scratch/izar/delsad/vlm_r1_text_bias/output

python -u evaluate.py \
    --model_path $MODEL_DIR/GRPO-lora-acc_0.627-Text-Biased/merged \
    --reasoning \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/test-grpo-acc_0.627-Text-Biased

