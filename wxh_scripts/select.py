import json
import random

# 读取原始 JSON 文件
with open("/root/LLaMA-Factory/data/RoleBench/profiles-eng/desc.json", "r") as f:
    data = json.load(f)

# 获取所有的键（人名）
names = list(data.keys())

# 随机选择 20 个人名
# random_names = random.sample(names, 20)

# 创建一个新的字典，包含随机选择的 20 个人名和他们的描述
selected_data = {name: data[name] for name in names}

# 将新的字典保存到新的 JSON 文件中
with open("/root/LLaMA-Factory/wxh_scripts/selected_profiles1.json", "w") as f:
    json.dump(selected_data, f, indent=4)

# 打印随机选择的 20 个人名
for name in names:
    print(name)