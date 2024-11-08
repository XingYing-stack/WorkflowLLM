#!/bin/bash

MODEL_PATH=$1


if [ -z "$MODEL_PATH" ]; then
  echo "Error: No model path provided."
  echo "Usage: $0 /path/to/model"
  exit 1
fi

LOG_FILE="./logs/eval_$(basename $MODEL_PATH).log"

nohup accelerate launch ./code/train_bf16.py --load_path $MODEL_PATH --do_infer > $LOG_FILE 2>&1 &