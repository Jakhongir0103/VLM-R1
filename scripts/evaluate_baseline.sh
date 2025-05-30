# replace with your WandB details
cd $REPO/src/open-r1-multimodal/src/open_r1/

echo $PWD
echo "Jakh0103/Qwen2.5-VL-3B-GRPO-VSR"

python -u evaluate.py \
    --model_path Jakh0103/Qwen2.5-VL-3B-GRPO-VSR \
    --reasoning \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/All-Set_grpo-baseline
    # --use_test_set

python -u evaluate.py \
    --model_path Jakh0103/Qwen2.5-VL-3B-SFT-VSR \
    --input_data_dir $DATA_PATH/vsr \
    --output_data_dir /home/delsad/VI/VLM-R1/results/vsr/All-Set_sft-baseline
    # --use_test_set
