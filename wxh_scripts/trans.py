import json
import os
import re

# 读取原始 JSON 文件
with open("/root/LLaMA-Factory/wxh_scripts/selected_profiles.json", "r") as f:
    profile = json.load(f)

# 获取文件夹路径
file_folder = "/root/LLaMA-Factory/data/RoleBench/instructions-eng"

# 获取文件夹中的所有文件名
all_files = os.listdir(file_folder)

# 定义正则表达式模式
pattern = re.compile(r'role-specific-(.*)\.jsonl')

# 提取角色名称并检查是否在 profile 中
filenames = []
for file in all_files:
    # 使用正则表达式提取角色名称
    match = pattern.match(file)
    if match:
        role_name = match.group(1)
        if role_name in profile:
            # print(role_name)
            filenames.append(file)

# print(filenames)

final_format = []
# 读取 JSONL 文件
for filename in filenames:
    match = pattern.match(filename)
    if match:
        role = match.group(1)
        with open(f"{file_folder}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()  # 去除行尾的换行符
                if line:  # 确保行不为空
                    data = json.loads(line)  # 将每行解析为 JSON 对象
                    tmp = {}
                    tmp["instruction"] = f"You are {role}. Your personality is described as: {profile[role]}"
                    tmp["input"] = data["instruction"]
                    tmp["output"] = data["answer"]
                    final_format.append(tmp)

# # 写入 JSON 文件
# with open("/root/LLaMA-Factory/data/rolellm_instructions.json", "w", encoding="utf-8") as f:
#     json.dump(final_format, f, ensure_ascii=False, indent=4)

print(len(final_format))