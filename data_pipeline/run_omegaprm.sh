#!/bin/bash

SPLIT_DIR="formatted_data"
LOG_DIR="logs"
OUTPUT_DIR="outputs"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

MAX_NEW_TOKENS=4096
TEMPERATURE=1.0
TOP_K=50
TOP_P=0.9

C_PUCT=0.125
ALPHA=0.5
BETA=0.9
LENGTH_SCALE=2000
NUM_ROLLOUTS=16
MAX_SEARCH_COUNT=200
ROLLOUT_BUDGET=1000

API_ENDPOINT="http://localhost:8000/v1/"


mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

START=$((NODE_RANK * 8 + 1))

for i in $(seq 0 7)
do
    j=$((START + i))
    GPU_ID=$i
    INPUT_FILE="$SPLIT_DIR/${j}_formatted.json"
    LOG_FILE="$LOG_DIR/omegaprm_part_${j}.log"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python run_omegaprm.py \
        --input_file $INPUT_FILE \
        --log_file $LOG_FILE \
        --output_dir $OUTPUT_DIR \
        --model $MODEL \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --c_puct $C_PUCT \
        --alpha $ALPHA \
        --beta $BETA \
        --length_scale $LENGTH_SCALE \
        --num_rollouts $NUM_ROLLOUTS \
        --max_search_count $MAX_SEARCH_COUNT \
        --rollout_budget $ROLLOUT_BUDGET \
        --api_endpoint $API_ENDPOINT &
done