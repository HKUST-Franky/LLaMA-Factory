import json

# 读取原始 JSON 文件
with open("/root/LLaMA-Factory/wxh_scripts/selected_profiles.json", "r") as f:
    profile = json.load(f)

# 读取 general instructions 文件
general_instructions = {}
with open("/root/LLaMA-Factory/data/RoleBench/instructions-eng/instructions-general.jsonl", "r") as f:
    for line in f:
        line = line.strip()  # 去除行尾的换行符
        if line:
            data = json.loads(line)  # 将每行解析为 JSON 对象
            general_instructions[data["instruction"]] = data["target"]

# 读取 role instructions 文件
role_instructions = []
with open("/root/LLaMA-Factory/data/RoleBench/rolebench-eng/instruction-generalization/general/train.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            role_name = data["role"]
            if role_name in profile:
                final_format = {
                    "role": role_name,
                    "question": data["question"],
                    "answer": data["generated"][0]
                }
                role_instructions.append(final_format)

# 生成 dpo_data
dpo_data = []
for ele1 in role_instructions:
    if ele1['question'] in general_instructions:
        ele2 = general_instructions[ele1['question']]
        tmp = {
            "conversations": [
                {
                    "from": "system",
                    "value": f"You are {ele1['role']}. Your personality is described as: {profile[ele1['role']]}"
                },
                {
                    "from": "human",
                    "value": ele1['question']
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": ele1['answer']
            },
            "rejected": {
                "from": "gpt",
                "value": ele2
            }
        }
        dpo_data.append(tmp)

# 写入 JSON 文件
with open("/root/LLaMA-Factory/data/rolellm_dpo.json", "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(len(dpo_data))