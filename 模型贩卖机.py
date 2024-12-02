import subprocess
import os
import shutil

def get_free_space_gb(directory):
    """获取指定目录的剩余空间（单位为GB）"""
    total, used, free = shutil.disk_usage(directory)
    return free // (2**30)  # 从字节转换为GB

def clear_directory(directory):
    """清空指定目录"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def move_model_files(download_path, nested_path, model_name):
    """将模型文件从嵌套文件夹移动到目标文件夹"""
    target_dir = os.path.join(download_path, model_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 移动所有文件到目标文件夹
    for item in os.listdir(nested_path):
        src_path = os.path.join(nested_path, item)
        dst_path = os.path.join(target_dir, item)
        shutil.move(src_path, dst_path)
    
    # 删除多余的文件夹层级
    hub_path = os.path.join(download_path, "hub")
    if os.path.exists(hub_path):
        shutil.rmtree(hub_path)

def update_model_config(model_path, update_package_path):
    """更新模型的配置文件"""
    print("检测到需要更新的模型，开始更新配置文件...")
    
    # 保留safetensors文件，删除其他文件
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if not item.endswith('.safetensors'):
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    # 复制新的配置文件
    if os.path.exists(update_package_path):
        for item in os.listdir(update_package_path):
            src_path = os.path.join(update_package_path, item)
            dst_path = os.path.join(model_path, item)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
        print("配置文件更新完成！")
    else:
        print("警告：更新包目录不存在！")

def download_model():
    download_path = "/root/autodl-tmp"
    # 确保下载目录存在
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # 检查剩余空间
    free_space_gb = get_free_space_gb(download_path)
    if free_space_gb < 30:
        print(f"剩余空间不足（仅剩 {free_space_gb} GB），正在清空目录...")
        clear_directory(download_path)
    
    model_name = input("请输入模型名称（如 glm-4-9b-chat）：").strip()
    
    # 特殊模型名称处理
    if model_name == "Qwen2-VL-7B-Instruct":
        print("检测到 Qwen2-VL-7B-Instruct 模型，使用 modelscope 下载...")
        os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp"
        command = "modelscope download --model Qwen/Qwen2-VL-7B-Instruct"
        subprocess.run(command, shell=True)
        nested_path = os.path.join(download_path, "hub", "Qwen", "Qwen2-VL-7B-Instruct")
        if os.path.exists(nested_path):
            move_model_files(download_path, nested_path, model_name)
        print(f"模型已整理到 {download_path}/{model_name}")
        return
    elif model_name == "internlm2_5-7b-chat":
        print("检测到 internlm2_5-7b-chat 模型，使用 modelscope 下载...")
        os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp"
        command = "modelscope download --model Shanghai_AI_Laboratory/internlm2_5-7b-chat"
        subprocess.run(command, shell=True)
        nested_path = os.path.join(download_path, "hub", "Shanghai_AI_Laboratory", "internlm2_5-7b-chat")
        if os.path.exists(nested_path):
            move_model_files(download_path, nested_path, model_name)
        print(f"模型已整理到 {download_path}/{model_name}")
        return
    elif model_name == "Qwen-7B-Chat":
        model_name = "Qwen-7B-Chat-new"
    elif model_name == "Llama3-8B-chat":
        model_name = "SmartFlowAI/Meta-Llama3-8B-Instruct"
    
    os.chdir(download_path)
    command = f"cg down {model_name}"
    print(f"将在 {download_path} 下执行下载命令...")
    subprocess.run(command, shell=True)
    print(f"模型已下载到 {download_path}")
    
    model_path = os.path.join(download_path, model_name)
    
    # 根据不同模型选择不同的更新包
    if model_name == "glm-4-9b-chat":
        update_package_path = "/root/LLaMA-Factory/chuli/glm4更新包"
        if os.path.exists(model_path):
            update_model_config(model_path, update_package_path)
        else:
            print("警告：未找到模型目录！")
    elif model_name == "glm-4-9b-chat-1m":
        update_package_path = "/root/LLaMA-Factory/chuli/glm4-1M更新包"
        if os.path.exists(model_path):
            update_model_config(model_path, update_package_path)
        else:
            print("警告：未找到模型目录！")
    elif model_name == "glm-4-9b":
        update_package_path = "/root/LLaMA-Factory/chuli/glm-base更新包"
        if os.path.exists(model_path):
            update_model_config(model_path, update_package_path)
        else:
            print("警告：未找到模型目录！")

if __name__ == "__main__":
    download_model()