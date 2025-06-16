#!/bin/bash

# 设置错误处理
set -e  # 遇到错误立即退出
set -o pipefail  # 管道中任何命令失败都会导致整个管道失败

# 使用vLLM生成中文训练数据的示例脚本
echo "开始生成中文训练数据..."

if python generate_data_vllm.py \
    --model_path "/mnt/project/EAGLE/Qwen3-0.6B" \
    --input_file "example_prompts.jsonl" \
    --output_file "generated_training_data_chinese.jsonl" \
    --max_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_samples 1 \
    --batch_size 16 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --language chinese; then
    echo "中文训练数据生成完成！"
    echo "输出文件: generated_training_data_chinese.jsonl"
    
    # 检查输出文件是否存在且不为空
    if [[ -f "generated_training_data_chinese.jsonl" && -s "generated_training_data_chinese.jsonl" ]]; then
        echo "生成的样本数量: $(wc -l < generated_training_data_chinese.jsonl)"
    else
        echo "警告: 输出文件不存在或为空"
        exit 1
    fi
else
    echo "错误: Python脚本执行失败"
    exit 1
fi
