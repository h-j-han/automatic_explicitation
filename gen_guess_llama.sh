#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/autoexpl:$(pwd)/llama_repo
LANG=en
DATASET=esqbv1htall
MP=1
model_size=7B
START=0
END=-1
MAX_LEN=660
PORT=29509

TARGET_FOLDER=/path/to/llama_weights
SCRIPT=autoexpl/xqb/gen_guess_llama_split.py
torchrun --nnodes=1 --master_port $PORT --nproc_per_node $MP $SCRIPT --lang $LANG --dataset-name $DATASET --ckpt-dir $TARGET_FOLDER/$model_size --start $START --end $END --max-seq-len $MAX_LEN
