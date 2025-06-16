import argparse
import json
import os
import sys
from typing import List, Dict, Any
import random
from tqdm import tqdm

# 延迟导入vLLM以避免初始化问题
def import_vllm():
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except Exception as e:
        print(f"vLLM导入失败: {e}")
        print("请确保正确安装了vLLM: pip install vllm")
        sys.exit(1)

def import_transformers():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except Exception as e:
        print(f"transformers导入失败: {e}")
        print("请确保正确安装了transformers: pip install transformers")
        sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用vLLM生成训练数据')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--input_file', type=str, required=True, help='输入提示文件 (JSONL格式)')
    parser.add_argument('--output_file', type=str, required=True, help='输出训练数据文件 (JSONL格式)')
    parser.add_argument('--max_tokens', type=int, default=1024, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样')
    parser.add_argument('--num_samples', type=int, default=1, help='每个提示生成的样本数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='张量并行大小')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU内存使用率')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'], help='生成语言')
    return parser.parse_args()

def load_prompts(input_file: str) -> List[Dict[str, Any]]:
    """从输入文件加载提示"""
    print(f"正在加载提示文件: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    prompts.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
    
    print(f"成功加载 {len(prompts)} 个提示")
    return prompts

def format_prompt_for_generation(prompt_data: Dict[str, Any], language: str = 'chinese') -> List[Dict]:
    """将提示数据格式化为对话格式用于生成"""
    if language == 'chinese':
        system_message = "你是一个有用、尊重和诚实的助手。请始终尽可能有帮助地回答，同时保持安全。你的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法内容。请确保你的回答在社会上是公正和积极的。\n\n如果一个问题没有任何意义，或者在事实上不连贯，请解释为什么，而不是回答不正确的内容。如果你不知道问题的答案，请不要分享虚假信息。"
    else:
        system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
    messages = [{"role": "system", "content": system_message}]
    
    # 添加用户消息
    if 'prompt' in prompt_data:
        messages.append({"role": "user", "content": prompt_data['prompt']})
    elif 'instruction' in prompt_data:
        messages.append({"role": "user", "content": prompt_data['instruction']})
    elif 'question' in prompt_data:
        messages.append({"role": "user", "content": prompt_data['question']})
    else:
        raise ValueError("在输入数据中没有找到有效的提示字段")
    
    return messages

def format_output_conversation(messages: List[Dict], response: str, original_id: str) -> Dict[str, Any]:
    """将生成的回应格式化为训练数据格式"""
    conversations = []
    
    # 跳过系统消息，用于对话格式
    for msg in messages[1:]:  # 跳过系统消息
        if msg['role'] == 'user':
            conversations.append({"from": "human", "value": msg['content']})
    
    # 添加生成的回应
    conversations.append({"from": "gpt", "value": response})
    
    return {
        "id": f"{original_id}_{random.randint(1000, 9999)}",
        "conversations": conversations
    }

def generate_training_data(args):
    """生成训练数据的主函数"""
    print("开始生成训练数据...")
    
    # 验证输入
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
    
    # 导入所需的库
    print("正在导入必要的库...")
    LLM, SamplingParams = import_vllm()
    AutoTokenizer = import_transformers()
    
    # 加载提示
    prompts_data = load_prompts(args.input_file)
    if not prompts_data:
        print("错误: 没有找到有效的提示数据")
        return
    
    # 初始化vLLM
    print("正在加载模型...")
    try:
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 加载分词器用于聊天模板
    print("正在加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("分词器加载成功")
    except Exception as e:
        print(f"分词器加载失败: {e}")
        return
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples
    )
    
    # 准备用于生成的对话
    conversations = []
    original_ids = []
    
    print("正在准备对话数据...")
    for i, prompt_data in enumerate(prompts_data):
        try:
            messages = format_prompt_for_generation(prompt_data, args.language)
            
            # 应用聊天模板
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            conversations.append(conversation)
            original_ids.append(prompt_data.get('id', f"prompt_{i}"))
        except Exception as e:
            print(f"警告: 处理第{i+1}个提示时出错: {e}")
            continue
    
    if not conversations:
        print("错误: 没有成功准备任何对话数据")
        return
    
    print(f"成功准备 {len(conversations)} 个对话")
    
    # 生成回应
    print(f"正在为 {len(conversations)} 个提示生成回应...")
    training_data = []
    
    # 批量处理
    for i in tqdm(range(0, len(conversations), args.batch_size), desc="生成进度"):
        batch_conversations = conversations[i:i+args.batch_size]
        batch_ids = original_ids[i:i+args.batch_size]
        
        try:
            # 生成
            outputs = llm.generate(batch_conversations, sampling_params)
            
            # 处理输出
            for j, output in enumerate(outputs):
                original_id = batch_ids[j]
                prompt_data = prompts_data[i + j]
                messages = format_prompt_for_generation(prompt_data, args.language)
                
                # 处理多个样本
                for k, completion in enumerate(output.outputs):
                    response = completion.text.strip()
                    
                    # 跳过空回应
                    if not response:
                        continue
                    
                    # 格式化为训练数据
                    training_sample = format_output_conversation(
                        messages, 
                        response, 
                        f"{original_id}_{k}" if args.num_samples > 1 else original_id
                    )
                    training_data.append(training_sample)
        except Exception as e:
            print(f"警告: 批次 {i//args.batch_size + 1} 生成失败: {e}")
            continue
    
    if not training_data:
        print("错误: 没有生成任何训练数据")
        return
    
    # 保存训练数据
    print(f"正在将 {len(training_data)} 个训练样本保存到 {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ 训练数据生成完成。生成了 {len(training_data)} 个样本。")
        print(f"✓ 输出文件: {args.output_file}")
        
        # 验证输出文件
        if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
            print(f"✓ 输出文件验证成功，文件大小: {os.path.getsize(args.output_file)} 字节")
        else:
            print("⚠ 警告: 输出文件为空或不存在")
            
    except Exception as e:
        print(f"保存文件失败: {e}")

def main():
    try:
        args = parse_args()
        print("参数解析成功")
        print(f"模型路径: {args.model_path}")
        print(f"输入文件: {args.input_file}")
        print(f"输出文件: {args.output_file}")
        print(f"批次大小: {args.batch_size}")
        print(f"最大tokens: {args.max_tokens}")
        
        generate_training_data(args)
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
