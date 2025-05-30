# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

MODEL_DIR=/scratch/izar/delsad/vlm_r1_text_bias/output

# python -u evaluate.py \
#     --model_path Jakh0103/Qwen2.5-VL-3B-GRPO-VSR \
#     --reasoning \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/grpo-test-set
#     --use_test_set

# python -u evaluate.py \
#     --model_path Jakh0103/Qwen2.5-VL-3B-SFT-VSR \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/sft-test-set
#     --use_test_set

# python -u evaluate.py \
#     --model_path $MODEL_DIR/Qwen2.5-VL-3B-GRPO-lora/merged \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/saydalie/project/VLM-R1/results/vsr/grpo

# python -u evaluate.py \
#     --model_path $MODEL_DIR/GRPO-lora-acc_0.627-Text-Biased/merged \
#     --reasoning \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/grpo-acc_0.627-Text-Biased

python -u evaluate.py \
    --model_path $MODEL_DIR/SFT-lora-acc_0.69-Text-Biased/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/All-Set_sft-acc_0.69-Text-Biased
    # --use_test_set

python -u evaluate.py \
    --model_path $MODEL_DIR/GRPO-lora-acc_0.69-Text-Biased/merged \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/All-Set_grpo-acc_0.69-Text-Biased \
    --reasoning
    # --use_test_set \
    

# python -u evaluate.py \
#     --model_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --input_data_dir $DATA_PATH/vsr \
#     --output_data_dir /home/saydalie/project/VLM-R1/results/vsr/base