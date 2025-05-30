# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

MODEL_DIR=/scratch/izar/delsad/vlm_r1/output
MODEL_DIR2=/scratch/izar/delsad/vlm_r1_text_bias/output

python -u evaluate_SmolVLM.py \
    --model_path $MODEL_DIR2/SmolVLM-2B-Instruct-SFT/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/SmolVLM/sft-strongtextbias-all

python -u evaluate_SmolVLM.py \
    --model_path $MODEL_DIR/SmolVLM-2B-Instruct-SFT/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/SmolVLM/sft-all
