#!/bin/bash
# sh train_OOD.sh {model_path} {data_path}

MODEL_PATH=$1

TRAIN_FILE=$2

if [ -z "$MODEL_PATH" ]; then
  echo "Error: No model path provided."
  echo "Usage: $0 /path/to/model /path/to/train_file [infer]"
  exit 1
fi

if [ -z "$TRAIN_FILE" ]; then
  echo "Error: No train file provided."
  echo "Usage: $0 /path/to/model /path/to/train_file [infer]"
  exit 1
fi

DO_INFER=$3

MODEL_BASENAME=$(basename $MODEL_PATH)
TRAIN_BASENAME=$(basename $TRAIN_FILE .json)

LOG_FILE="./logs/train_${MODEL_BASENAME}_${TRAIN_BASENAME}.log"

if [ "$DO_INFER" = "true" ]; then
  INFER_FLAG="--do_infer"
  echo "Inference mode enabled."
else
  INFER_FLAG=""
  echo "Training mode enabled."
fi

nohup accelerate launch ./src/train_bf16.py --load_path $MODEL_PATH --train_file $TRAIN_FILE --OOD --do_train $INFER_FLAG > $LOG_FILE 2>&1 &