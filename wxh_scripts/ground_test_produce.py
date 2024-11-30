import json
import os
import re


final_format = []
with open("/root/LLaMA-Factory/data/RoleBench/instructions-eng/instructions-general.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            if line:
                tmp = {}
                tmp["instruction"] = data["instruction"]
                tmp["input"] = ""
                tmp["output"] = data["target"]
                final_format.append(tmp)


# 写入 JSON 文件
with open("/root/LLaMA-Factory/data/rolellm_ground_test.json", "w", encoding="utf-8") as f:
    json.dump(final_format, f, ensure_ascii=False, indent=4)