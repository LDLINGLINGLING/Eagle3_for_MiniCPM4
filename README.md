# EAGLE Training Implementation (TrainEAGLE3)

一个高效的EAGLE（推测解码）模型分布式训练框架，支持大语言模型的加速推理优化。

## 🎯 项目简介

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是一种创新的推测解码方法，通过训练轻量级的草稿模型来预测目标模型的下一个token，从而显著提升大语言模型的推理速度。
本项目为MiniCPM4适配EAGLE3的投机解码模型。

### 训练效果
以下表格是EAGLE3在MiniCPM4上的使用14000条alpaca数据训练后的测试结果
| 预测位置 | 准确率 (%) | 准确率标准差 | 损失 | 损失标准差 |
|---------|-----------|-------------|------|-----------|
| 位置 0  | 48.05     | ±7.58       | 0.8946 | ±0.3383 |
| 位置 1  | 48.72     | ±7.64       | 0.8844 | ±0.3379 |
| 位置 2  | 48.68     | ±8.03       | 0.8839 | ±0.3390 |
| 位置 3  | 48.40     | ±8.15       | 0.8884 | ±0.3411 |
| 位置 4  | 48.09     | ±8.18       | 0.8935 | ±0.3434 |
| 位置 5  | 47.34     | ±8.15       | 0.9006 | ±0.3467 |
| 位置 6  | 46.79     | ±7.89       | 0.9093 | ±0.3490 |
### 核心思想
- 🎯 **推测解码**：使用小模型预测大模型的输出，减少推理延迟
- 🔄 **多步预测**：一次性预测多个token，提高并行度
- 📊 **词汇表压缩**：基于频率统计压缩输出词汇表，减少计算量
- 🤝 **知识蒸馏**：让草稿模型学习目标模型的输出分布

## ✨ 主要特性

- ⚡ **分布式训练**：基于DeepSpeed的多卡/多机训练支持
- 🔧 **混合精度**：支持FP16/BF16，降低显存使用
- 💾 **断点续训**：自动检查点保存和恢复
- 📈 **实时监控**：集成Wandb训练监控
- 🎛️ **灵活配置**：支持多种模型架构和训练策略
- 🧪 **完整测试**：包含单卡测试脚本

## 📦 环境要求

### 硬件要求
- **GPU**: NVIDIA V100/A100/H100等，建议8卡或以上
- **显存**: 每卡至少24GB（取决于模型大小）
- **内存**: 建议256GB以上

### 软件依赖
```bash
# Python 3.8+
pip install torch>=1.13.0
pip install transformers>=4.21.0
pip install deepspeed>=0.9.0
pip install datasets>=2.0.0
pip install wandb
pip install numpy
pip install tqdm
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/LDLINGLINGLING/Eagle3_for_MiniCPM4.git
cd Eagle3_for_MiniCPM4/eagle/traineagle3

# 安装依赖
pip install -r requirements.txt

# 设置Wandb（可选）
export WANDB_API_KEY="your_wandb_api_key"
```

### 2. 数据准备
数据格式为JSONL，每行包含对话数据：
```json
{
  "id": "unique_id",
  "conversations": [
    {"from": "human", "value": "用户问题1"},
    {"from": "gpt", "value": "助手回答2"},
    {"from": "human", "value": "用户问题1"},
    {"from": "gpt", "value": "助手回答2"}
  ]
}
```

### 3. 配置文件
确保以下配置文件存在：
- `ds_config.json`: DeepSpeed配置
- `config.json`: EAGLE模型配置

### 4. 开始训练
```bash
# 单机多卡训练
deepspeed --num_gpus=8 main.py \
    --basepath /path/to/base/model \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir ./checkpoints \
    --deepspeed_config ds_config.json

# 多机训练
deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    main.py [参数同上]
```

## 📋 详细使用说明

### 训练参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--basepath` | str | `/data1/minicpm4/` | 预训练目标模型路径 |
| `--trainpath` | str | - | 训练数据路径 |
| `--testpath` | str | - | 测试数据路径 |
| `--savedir` | str | `'0'` | 模型保存目录 |
| `--deepspeed_config` | str | - | DeepSpeed配置文件 |
| `--local_rank` | int | `-1` | 分布式训练本地rank |

### 配置文件详解

#### DeepSpeed配置 (ds_config.json)
```json
{
  "fp16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2,
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 0.0001}
  }
}
```

#### EAGLE模型配置 (config.json)
```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "draft_vocab_size": 16000,
  "length": 8
}
```

### 训练流程

1. **数据扫描**：分析训练数据，构建压缩词汇表
2. **模型初始化**：加载目标模型和EAGLE模型
3. **分布式设置**：配置DeepSpeed分布式训练
4. **训练循环**：
   - 数据预处理和批次构建
   - 目标模型前向传播（获取监督信号）
   - EAGLE模型训练（多步预测）
   - 损失计算和反向传播
   - 参数更新和检查点保存

### 测试和评估

```bash
# 单卡测试
python test.py \
    --model_path ./checkpoints/state_39/pytorch_model.bin \
    --basepath /path/to/base/model \
    --testpath /path/to/test.jsonl \
    --batch_size 2 \
    --save_results ./test_results
```

## 📁 项目结构

```
traineagle3/
├── main.py              # 主训练脚本
├── test.py              # 单卡测试脚本
├── cnets.py             # EAGLE模型实现
├── configs.py           # 配置类定义
├── ds_config.json       # DeepSpeed配置
├── config.json          # EAGLE模型配置
├── requirements.txt     # 依赖列表
└── README.md           # 项目说明
```

## 🔧 核心模块说明

### cnets.py - EAGLE模型实现
- `Model`类：主要的EAGLE模型实现
- `scandata`方法：数据统计和词汇表压缩
- `forward`方法：多步预测训练逻辑
- `dataprepare`方法：数据预处理

### configs.py - 配置管理
- `EConfig`类：EAGLE模型配置
- 支持从JSON文件加载配置
- 参数验证和默认值设置

### main.py - 训练主流程
- DeepSpeed分布式训练设置
- 数据加载和预处理
- 训练循环和检查点管理
- Wandb监控集成

## 📊 性能优化

### 内存优化
- **ZeRO Stage 2**：分片优化器状态
- **梯度累积**：减少通信频率
- **混合精度**：FP16/BF16训练
- **词汇表压缩**：减少输出维度

### 计算优化
- **FlashAttention**：高效注意力计算
- **多步预测**：并行token生成
- **批次处理**：提高GPU利用率

### 通信优化
- **重叠通信**：计算和通信并行
- **梯度压缩**：减少传输数据量
- **分层通信**：优化多机通信

## 🐛 常见问题

### 1. CUDA内存不足
```bash
# 减少批次大小
"train_micro_batch_size_per_gpu": 1

# 增加梯度累积
"gradient_accumulation_steps": 8

# 启用CPU卸载
"offload_optimizer": {"device": "cpu"}
```

### 2. FlashAttention错误
```bash
# 确保数据类型正确
export CUDA_VISIBLE_DEVICES=0
# 模型会自动选择合适的数据类型
```

### 3. 检查点加载失败
```bash
# 检查文件完整性
ls checkpoints/state_*/

# 清理损坏的检查点
rm -rf checkpoints/state_broken/
```

### 4. 分布式训练问题
```bash
# 检查网络连接
ping master_node_ip

# 确保端口可用
netstat -an | grep 29500

# 检查NCCL版本
python -c "import torch; print(torch.cuda.nccl.version())"
```

## 📈 监控和日志

### Wandb监控
- 训练损失和准确率
- GPU利用率和内存使用
- 学习率变化
- 各层预测精度

### 本地日志
- 控制台输出：实时训练信息
- 检查点：自动保存训练状态
- 错误日志：异常和警告信息

## 🤝 贡献指南

欢迎贡献代码和提出改进建议！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

**Happy Training! 🚀**
