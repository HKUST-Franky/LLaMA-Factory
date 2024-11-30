import json
import os
import re

# 读取原始 JSON 文件
with open("/root/LLaMA-Factory/wxh_scripts/selected_profiles.json", "r") as f:
    profile = json.load(f)

final_format = []
with open("/root/LLaMA-Factory/data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            role = data["role"]  # 定义 role 变量
            if role in profile:
                tmp = {}
                tmp["instruction"] = f"You are {role}. Your personality is described as: {profile[role]}"
                tmp["input"] = data["question"]
                if "generated" in data and len(data["generated"]) > 0:
                    tmp["output"] = data["generated"][0]
                else:
                    tmp["output"] = "" 
                final_format.append(tmp)



with open("/root/LLaMA-Factory/data/RoleBench/rolebench-eng/role-generalization/role_specific/test.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            role = data["role"]  # 定义 role 变量
            if role in profile:
                tmp = {}
                tmp["instruction"] = f"You are {role}. Your personality is described as: {profile[role]}"
                tmp["input"] = data["question"]
                if "generated" in data and len(data["generated"]) > 0:
                    tmp["output"] = data["generated"][0]
                else:
                    tmp["output"] = "" 
                final_format.append(tmp)


# 写入 JSON 文件
with open("/root/LLaMA-Factory/data/rolellm_test.json", "w", encoding="utf-8") as f:
    json.dump(final_format, f, ensure_ascii=False, indent=4)
