import json
json_path = "/data2/lizhao21/LLaMA-Factory/data/sharegpt_dialogue_zh_27k.json"
jsonl_path = "/data2/lizhao21/LLaMA-Factory/cpm/EAGLE/eagle/traineagle3/train/chatml_05_26.jsonl"
# 示例：将其他格式转换为所需格式
import json
def convert_to_jsonl(input_data,index=0):
    output_lines = []
    for idx, item in enumerate(input_data, start=index):  # 从88开始编号
        # 构建conversations列表
        conversations = [
            {
                "from": "human",
                "value": f"{item['instruction']}\n{item['input']}"
            },
            {
                "from": "gpt",
                "value": item['output']
            }
        ]
        
        # 创建输出字典
        output_dict = {
            "id": str(idx),
            "conversations": conversations
        }
        
        # 转换为JSON字符串并添加到输出列表
        output_lines.append(output_dict)
    
    # 返回JSONL格式的字符串（每行一个JSON对象）
    return output_lines
def convert_to_required_format(input_data):
    converted_data = []
    for i, item in enumerate(input_data):
        formatted_item = {
            "id": str(i),
            "conversations": []
        }

        messages = item["messages"]

        # 跳过system消息，因为代码中已经硬编码了system prompt
        if messages[0]['role']=='system':
            messages[1]['content']=messages[0]['content']+messages[1]['content']
        filtered_messages = messages
        # 确保从human开始
        if filtered_messages and filtered_messages[0]["role"] != "user":
            filtered_messages = filtered_messages[1:]

        # 转换为conversations格式
        for msg in filtered_messages:
            conversation = {
                "from": "human" if msg["role"] == "user" else "gpt",
                "value": msg["content"]
            }
            formatted_item["conversations"].append(conversation)

        # 确保对话是human-gpt交替的
        if len(formatted_item["conversations"]) > 0:
            converted_data.append(formatted_item)

    return converted_data
with open(json_path,'r',encoding='utf-8') as f:
    datas = json.load(f)
with open('/data2/lizhao21/LLaMA-Factory/data/alpaca_10000.json','r',encoding='utf-8') as f:
    datas_new = json.load(f)

convert_data = convert_to_required_format(datas)
convert_data_new = convert_to_jsonl(datas_new,len(datas))
convert_data+=convert_data_new
with open(jsonl_path,'w',encoding='utf-8') as f:
    for l in convert_data:
        line=json.dumps(l,ensure_ascii=False)  # 写入单个对象
        f.write(line+'\n')       # 添加换行符分隔[1,3](@ref)