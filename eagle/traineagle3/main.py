"""
EAGLE模型训练脚本
本脚本实现了EAGLE（推测解码）模型的分布式训练流程

EAGLE的核心思想：
1. 使用预训练的大模型（target model）提取隐藏状态作为监督信号
2. 训练一个轻量级的草稿模型（draft model）来预测下一个token
3. 通过多步迭代预测实现推测解码，加速大模型推理

训练流程：
1. 从目标模型提取多层隐藏状态
2. 使用EAGLE解码器处理连接的embedding和隐藏状态
3. 多步预测：每一步预测下一个token，并将结果用于下一步
4. 使用KL散度损失训练模型匹配目标模型的输出分布
"""

import argparse
import deepspeed

# 命令行参数解析
parser = argparse.ArgumentParser(description='EAGLE模型训练')
parser.add_argument('--basepath', type=str, default='/data1/minicpm4/',
                    help='预训练目标模型的路径')
parser.add_argument('--trainpath', type=str,
                    default="/data2/lizhao21/LLaMA-Factory/cpm/EAGLE/eagle/traineagle3/train/chatml_05_26.jsonl",
                    help='训练数据路径')
parser.add_argument('--testpath', type=str,
                    default="/data2/lizhao21/LLaMA-Factory/cpm/EAGLE/eagle/traineagle3/train/chatml_05_26.jsonl",
                    help='测试数据路径')
parser.add_argument('--savedir', type=str, default='0',
                    help='模型保存目录')
parser.add_argument("--local_rank", type=int, default=-1, 
                    help="分布式训练的本地rank")

parser = deepspeed.add_config_arguments(parser)  # 添加DeepSpeed配置参数
args = parser.parse_args()
import json
import re

# 读取DeepSpeed配置
deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)

# 训练配置
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],  # 批次大小
    "num_epochs": 40,        # 训练轮数
    "num_workers": 2,        # 数据加载器工作进程数
    "max_len": 2048,         # 最大序列长度
    "config_path": "/data2/lizhao21/LLaMA-Factory/cpm/EAGLE/eagle/traineagle3/config.json",  # EAGLE模型配置路径
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32以提高性能
from accelerate.utils import set_seed

set_seed(0)  # 设置随机种子确保可重现性
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup


def build_dataset_rank(tokenizer, datapath):
    """
    构建训练/测试数据集
    
    核心处理：
    1. 加载对话数据并格式化为chat template
    2. 生成loss mask：只对assistant回复部分计算损失
    3. 返回包含input_ids、attention_mask、loss_mask的数据集
    """
    # 加载JSON格式的对话数据
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)  # 随机打乱数据
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8  # 并行处理进程数

    def preprocess_function(examples):
        """预处理函数：将对话数据转换为模型输入格式"""
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        
        for i in range(len(examples['id'])):
            # 构建系统提示词
            messages = [
                # {"role": "system",
                #  "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            
            # 对话角色映射
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            
            if not source:
                continue
                
            # 确保对话从用户开始
            if roles[source[0]["from"]] != "user":
                source = source[1:]
                
            # 构建对话消息列表
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            
            # 应用chat template格式化对话
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            # 分词并生成input_ids
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            
            # 初始化loss mask（全部设为1）
            loss_mask = torch.ones_like(input_ids)

            # 重要：构建loss mask，只对assistant的回复计算损失
            sep = "<|im_end|>\n<|im_start|>assistant\n"  # assistant开始标记
            sep2 = "<|im_end|>\n<|im_start|>user\n"          # user开始标记
            
            total_len = len(input_ids)
            turns = conversation.split(sep2)  # 按用户消息分割对话

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0  # 开始部分不计算损失
            
            # 为每个对话轮次设置loss mask
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)  # 分割用户输入和assistant回复
                if len(parts) != 2:
                    break
                parts[0] += sep
                
                # 计算指令部分长度（用户输入）
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # 对用户指令部分不计算损失（设为0）
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                    
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0  # 结尾部分不计算损失
            attention_mask = torch.ones_like(loss_mask)

            # 添加到批次数据中
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    # 并行处理数据集
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")  # 设置为PyTorch tensor格式
    return ds1


class DataCollatorWithPadding:
    """
    数据整理器：将不同长度的序列填充到批次中的最大长度
    确保批次内所有序列长度一致
    """

    def paddingtensor(self, intensors, N):
        """3D tensor填充"""
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        """2D tensor填充"""
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批次数据整理主函数"""
        # 找到批次中的最大序列长度
        max_length = max(item['input_ids'].shape[1] for item in features)
        
        # 将所有序列填充到最大长度
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


# 初始化tokenizer和数据集
tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

# 初始化EAGLE模型
config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True, load_head=True)
# 扫描数据构建压缩词汇表（EAGLE的重要优化）
model.scandata(args.trainpath, args.basepath)

# 损失函数（这里定义了但实际使用的是模型内部的KL散度损失）
criterion = nn.SmoothL1Loss(reduction="none")
num_epochs = train_config["num_epochs"]

# 初始化DeepSpeed分布式训练
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

# 获取分布式训练相关信息
global_rank = deepspeed.comm.get_rank()        # 全局rank
rank = deepspeed.comm.get_local_rank()         # 本地rank
world_size = deepspeed.comm.get_world_size()   # 总进程数

# 只在主进程初始化wandb日志记录
if global_rank == 0:
    import wandb
    wandb.login(key="")  # 需要填入实际的wandb API key
    wandb.init(project="l382", entity="yuhui-li", config=ds_config)

# 创建保存目录
os.makedirs(args.savedir, exist_ok=True)

# 设置分布式数据采样器和数据加载器
sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, 
                        num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, 
                         num_workers=4, pin_memory=True, collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    """
    查找最新的检查点文件
    用于断点续训：找到最大的state_X目录作为恢复点
    """
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


# 检查是否存在检查点，支持断点续训
checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"从检查点加载: {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)

# 主训练循环
for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)  # 确保每个epoch的数据打乱不同
    print(f"开始训练第 {epoch} 轮")

    model.train()  # 设置为训练模式
    
    # 初始化每个epoch的统计指标
    epoch_acces = [[] for _ in range(model.length)]    # 每一步的准确率
    epoch_plosses = [[] for _ in range(model.length)]  # 每一步的预测损失

    # 训练阶段
    for batch_idx, data in enumerate(tqdm(train_loader)):
        model.zero_grad()  # 清零梯度

        # 前向传播：获取多步预测的损失和准确率
        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        # 计算加权总损失：较早的预测步骤权重更高
        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        
        # 反向传播和参数更新
        model_engine.backward(loss)
        model_engine.step()

        # 记录训练指标（只在主进程）
        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            # 记录每一步的预测损失
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            # 记录每一步的准确率
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)
            
        # 累积epoch统计
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # 计算并记录epoch级别的训练准确率
    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)  # 跨进程平均
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(f"训练 Epoch [{epoch + 1}/{num_epochs}], 位置 {i}, 准确率: {acc_i:.2f}")

    # 计算并记录epoch级别的训练损失
    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)  # 跨进程平均
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(f"训练 Epoch [{epoch + 1}/{num_epochs}], 位置 {i}, 损失: {loss_i:.2f}")

    # 验证阶段
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():  # 验证时不计算梯度
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # 计算并记录验证准确率
    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochacc_{i}": acc_i})
            print(f"测试 Epoch [{epoch + 1}/{num_epochs}], 位置 {i}, 准确率: {acc_i:.2f}")

    # 计算并记录验证损失
    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochploss_{i}": loss_i})
            print(f"测试 Epoch [{epoch + 1}/{num_epochs}], 位置 {i}, 损失: {loss_i:.2f}")

    # 保存模型检查点
    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    # 每10个epoch保存完整的DeepSpeed检查点
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
