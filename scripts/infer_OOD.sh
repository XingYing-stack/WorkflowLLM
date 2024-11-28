#!/bin/bash

# 设置模型路径为第一个环境变量
MODEL_PATH=$1

# 检查是否传入了路径
if [ -z "$MODEL_PATH" ]; then
  echo "Error: No model path provided."
  echo "Usage: $0 /path/to/model"
  exit 1
fi

# 获取路径的最后一节并设置日志文件名
LOG_FILE="./logs/eval_OOD_$(basename $MODEL_PATH).log"

# 运行训练脚本并将输出保存到指定的日志文件中
nohup accelerate launch ./src/train_bf16.py --load_path $MODEL_PATH --OOD --do_infer > $LOG_FILE 2>&1 &