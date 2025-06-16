"""
EAGLE模型单卡测试脚本
用于加载训练好的.bin文件进行模型推理测试
"""

import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# 导入模型相关模块
from cnets import Model
from configs import EConfig
from torch.utils.data import DataLoader


def build_test_dataset(tokenizer, datapath, max_samples=None):
    """
    构建测试数据集（与训练时相同的预处理逻辑）
    """
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": [],
            "conversations": []  # 保留原始对话用于分析
        }

        for i in range(len(examples['id'])):
            # 构建系统提示词
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]

            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]

            if not source:
                continue

            if roles[source[0]["from"]] != "user":
                source = source[1:]

            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )

            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]

            loss_mask = torch.ones_like(input_ids)

            # 构建loss mask
            sep = "<|im_end|>\n<|im_start|>assistant\n"
            sep2 = "<|im_end|>\n<|im_start|>user\n"

            total_len = len(input_ids)
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0

            for j, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                if j == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0

                cur_len += turn_len
                if j != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])
            new_examples["conversations"].append(examples['conversations'][i])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=ds.column_names,
        load_from_cache_file=False
    )

    ds.set_format(type="torch")
    return ds


class DataCollatorWithPadding:
    """数据整理器"""

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features):
        max_length = max(item['input_ids'].shape[1] for item in features)

        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }


def load_model_from_bin(model, bin_path, dtype=torch.float16):
    """从.bin文件加载模型权重，并设置正确的数据类型"""
    print(f"正在加载模型权重: {bin_path}")

    try:
        # 加载权重文件
        state_dict = torch.load(bin_path, map_location='cpu')

        # 如果是嵌套的字典结构，提取模型权重
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 加载权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys}")

        # 设置模型数据类型
        print(f"转换模型到 {dtype}...")
        model = model.to(dtype)

        print("模型权重加载成功!")
        return True, model

    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return False, model


def get_optimal_dtype():
    """根据GPU能力选择最佳数据类型"""
    if not torch.cuda.is_available():
        return torch.float32

    # 获取GPU计算能力
    major, minor = torch.cuda.get_device_capability()

    # Ampere架构(A100, RTX30系列)及以上支持bf16
    if major >= 8:
        print("检测到支持bf16的GPU，使用bf16")
        return torch.bfloat16
    # Turing架构(RTX20系列)及以上支持fp16
    elif major >= 7:
        print("检测到支持fp16的GPU，使用fp16")
        return torch.float16
    else:
        print("GPU不支持混合精度，使用fp32")
        return torch.float32


def test_model(args):
    """主测试函数"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 选择最佳数据类型
    optimal_dtype = get_optimal_dtype()

    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.basepath)

    # 加载测试数据
    print("加载测试数据...")
    test_dataset = build_test_dataset(tokenizer, args.testpath, max_samples=args.max_samples)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=2
    )

    # 初始化模型
    print("初始化EAGLE模型...")
    config = EConfig.from_pretrained(args.config_path)
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)

    # 扫描数据构建词汇表映射
    print("构建压缩词汇表...")
    model.scandata(args.testpath, args.basepath)

    # 加载训练好的权重并设置数据类型
    success, model = load_model_from_bin(model, args.model_path, dtype=optimal_dtype)
    if not success:
        print("模型加载失败，退出测试")
        return

    # 移动到GPU
    model = model.to(device)
    model.eval()

    # 测试循环
    print("开始测试...")
    all_losses = [[] for _ in range(model.length)]
    all_accs = [[] for _ in range(model.length)]

    total_samples = 0
    successful_batches = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="测试进度")):
            try:
                # 移动数据到GPU并设置正确的数据类型
                input_ids = data["input_ids"].to(device)

                # attention_mask和loss_mask需要匹配模型的数据类型
                if optimal_dtype == torch.float32:
                    attention_mask = data["attention_mask"].to(device).float()
                    loss_mask = data["loss_mask"].to(device).float()
                elif optimal_dtype == torch.float16:
                    attention_mask = data["attention_mask"].to(device).half()
                    loss_mask = data["loss_mask"].to(device).half()
                else:  # bfloat16
                    attention_mask = data["attention_mask"].to(device).to(torch.bfloat16)
                    loss_mask = data["loss_mask"].to(device).to(torch.bfloat16)

                # 前向传播
                plosses, vlosses, acces = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask
                )

                # 收集指标
                for i in range(len(plosses)):
                    all_losses[i].append(plosses[i].item())
                for i in range(len(acces)):
                    all_accs[i].append(acces[i])

                total_samples += input_ids.size(0)
                successful_batches += 1

                # 每处理一定数量的batch打印中间结果
                if (batch_idx + 1) % args.print_every == 0:
                    print(f"\n批次 {batch_idx + 1}, 已处理样本: {total_samples}, 成功批次: {successful_batches}")
                    for i in range(len(all_accs)):
                        if all_accs[i]:
                            avg_acc = np.mean(all_accs[i])
                            avg_loss = np.mean(all_losses[i])
                            print(f"  位置 {i}: 准确率 {avg_acc:.4f}, 损失 {avg_loss:.4f}")

            except torch.cuda.OutOfMemoryError:
                print(f"批次 {batch_idx} 显存不足，跳过该批次")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {e}")
                continue

    # 计算最终结果
    print("\n" + "="*50)
    print("测试完成！最终结果:")
    print("="*50)

    if successful_batches == 0:
        print("警告: 没有成功处理的批次!")
        return

    for i in range(len(all_accs)):
        if all_accs[i]:
            avg_acc = np.mean(all_accs[i])
            avg_loss = np.mean(all_losses[i])
            std_acc = np.std(all_accs[i])
            std_loss = np.std(all_losses[i])

            print(f"位置 {i}:")
            print(f"  准确率: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"  损失:   {avg_loss:.4f} ± {std_loss:.4f}")

    print(f"\n总共测试样本数: {total_samples}")
    print(f"成功处理批次数: {successful_batches}/{len(test_loader)}")

    # 保存结果
    if args.save_results:
        results = {
            "total_samples": total_samples,
            "successful_batches": successful_batches,
            "total_batches": len(test_loader),
            "data_type": str(optimal_dtype),
            "results": []
        }

        for i in range(len(all_accs)):
            if all_accs[i]:
                results["results"].append({
                    "position": i,
                    "accuracy": {
                        "mean": float(np.mean(all_accs[i])),
                        "std": float(np.std(all_accs[i]))
                    },
                    "loss": {
                        "mean": float(np.mean(all_losses[i])),
                        "std": float(np.std(all_losses[i]))
                    }
                })

        result_path = f"{args.save_results}/test_results.json"
        os.makedirs(args.save_results, exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EAGLE模型测试脚本')

    # 模型和数据路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的.bin模型文件路径')
    parser.add_argument('--basepath', type=str, default='/data1/minicpm4/',
                        help='预训练目标模型的路径')
    parser.add_argument('--testpath', type=str, required=True,
                        help='测试数据路径')
    parser.add_argument('--config_path', type=str, 
                        default="/data2/lizhao21/LLaMA-Factory/cpm/EAGLE/eagle/traineagle3/config.json",
                        help='EAGLE模型配置文件路径')

    # 测试参数
    parser.add_argument('--batch_size', type=int, default=2,
                        help='测试批次大小（建议设置较小避免显存问题）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数（None表示测试全部）')
    parser.add_argument('--print_every', type=int, default=10,
                        help='每多少个batch打印一次进度')
    parser.add_argument('--save_results', type=str, default=None,
                        help='保存测试结果的目录路径')

    args = parser.parse_args()

    # 运行测试
    test_model(args)