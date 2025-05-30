export DATA_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/data/"
export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export PYTHONPATH="$PYTHONPATH:$REPO/src/"

python -u $REPO/src/open_r1/prepare_sft_data.py \
    --data_dir $DATA_PATH \
    --explanation_type bbox \
    --output_name driving_bbox_full \
    --seed 42