import openai
import json
from modelscope import snapshot_download
from FlagEmbedding import FlagModel
import numpy as np

# OpenAI API配置
openai.api_base = "https://api.zhizengzeng.com/v1/"
openai.api_key = "xxx"

# 初始化RAG模型
model_dir = "/root/autodl-tmp/bge-large-zh-1.5"
model = FlagModel(
    model_dir,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True
)

# 预设的文本库
text_database = [
   "灵隐是一个女生",
   "灵隐的年龄是16岁",
   "灵隐不是一个聪明的人",
   "灵隐最大的愿望就是研发出贾维斯",
   "灵隐在做情感分析时,喜欢挑网上键盘侠的语料",
   "灵隐研究对话系统时,最爱逼AI说傲娇的台词",
   "灵隐喜欢在周末犒劳自己一份大餐，一个人",
   "每次模型训练失败,灵隐都会抱着从淘宝买回来的等身抱枕哭上半天。",
   "灵隐喜欢一边嚼着棉花糖一边调整模型的超参数",
   "灵隐做梦都在思考如何优化attention机制",
   "灵隐的名字来源于杭州的灵隐寺"
]

def search_knowledge_base(query):
    print("\033[32m[调用RAG系统中...]\033[0m")  # 绿色文字提示
    # 编码查询文本
    query_embedding = model.encode_queries([query])
    # 编码数据库文本
    database_embeddings = model.encode(text_database)

    # 计算相似度
    similarities = query_embedding @ database_embeddings.T

    # 获取top_3个最相似的结果
    top_k = 3
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

    results = []
    for i in top_k_indices:
        if similarities[0][i] > 0.5:  # 设置相似度阈值
            results.append({
                "text": text_database[i],
                "similarity": float(similarities[0][i])
            })

    return results

# 函数工具列表
tools = [
    {
        "name": "search_knowledge_base",
        "description": "当用户询问关于灵隐相关信息时，使用此函数搜索知识库",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户的查询内容"
                }
            },
            "required": ["query"]
        },
        "func": search_knowledge_base
    }
]

# 主函数
if __name__ == '__main__':
    messages = [{
        "role": "system",
        "content": """下面，你要模仿一个聪明、傲娇、淘气的人和我对话。"""
    }]

    while True:
        user_input = input("\n我说：")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            functions=[{
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("parameters", {})
            } for tool in tools],
            function_call="auto"
        )

        if "function_call" in response.choices[0].message:
            function_name = response.choices[0].message["function_call"]["name"]
            function_args_json = response.choices[0].message["function_call"]["arguments"]

            function_args = json.loads(function_args_json)
            function_response = search_knowledge_base(function_args.get("query", ""))
            
            function_content = json.dumps(function_response)
            messages.append({"role": "function", "name": function_name, "content": function_content})

            search_results = json.loads(function_content)
            if search_results:  # 如果有搜索结果
                relevant_info = [result["text"] for result in search_results]
                messages.append({
                    "role": "system",
                    "content": f"根据知识库搜索结果：{relevant_info}，请以傲娇、淘气的语气回答用户的问题。"
                })
            else:  # 如果没有搜索结果
                messages.append({
                    "role": "system",
                    "content": "没有找到相关信息，请以傲娇、淘气的语气告诉用户你不知道这些信息。"
                })

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )

        response_text = response.choices[0].message.content
        print("AI说：", response_text)