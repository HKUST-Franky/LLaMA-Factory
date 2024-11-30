import json
from modelscope import snapshot_download
from FlagEmbedding import FlagModel
import numpy as np
from openai import OpenAI
import os

# OpenAI客户端配置
client = OpenAI(
   api_key="EMPTY",
   base_url="http://localhost:6006/v1"
)

# 初始化RAG模型部分保持不变
model_dir = "/root/autodl-tmp/bge-large-zh-1.5"
model = FlagModel(
   model_dir,
   query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
   use_fp16=True
)

# 文本库部分保持不变
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
   print("\033[32m[调用RAG系统中...]\033[0m")
   query_embedding = model.encode_queries([query])
   database_embeddings = model.encode(text_database)
   similarities = query_embedding @ database_embeddings.T
   top_k = 3
   top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
   results = []
   for i in top_k_indices:
       if similarities[0][i] > 0.5:
           results.append({
               "text": text_database[i],
               "similarity": float(similarities[0][i])
           })
   return results

# 工具定义改为新版格式
tools = [
   {
       "type": "function",
       "function": {
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
           }
       }
   }
]

def process_conversation(user_input, messages):
   try:
       messages.append({"role": "user", "content": user_input})

       try:
           # 第一次调用，检查是否需要使用函数
           response = client.chat.completions.create(
               model="gpt-4",
               messages=messages,
               tools=tools,
               temperature=0.7,  # 控制随机性，0-2之间，越大越随机
               top_p=0.9,       # 控制输出多样性，0-1之间
               presence_penalty=0.6,  # 控制话题重复度，-2到2之间
               frequency_penalty=0.5  # 控制用词重复度，-2到2之间
           )

           # 如果需要调用函数
           if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls is not None:
               messages.append(response.choices[0].message)
               
               tool_call = response.choices[0].message.tool_calls[0].function
               function_args = json.loads(tool_call.arguments)
               function_response = search_knowledge_base(function_args.get("query", ""))
               
               messages.append({
                   "role": "tool",
                   "name": "search_knowledge_base",
                   "content": json.dumps(function_response)
               })

               # 第二次调用，获取最终回复
               response = client.chat.completions.create(
                   model="gpt-4",
                   messages=messages,
                   temperature=0.7,
                   top_p=0.6,
                   presence_penalty=0.6,
                   frequency_penalty=0.5
               )

           ai_response = response.choices[0].message.content
           messages.append({"role": "assistant", "content": ai_response})
           return ai_response, messages

       except Exception as e:
           print(f"对话处理出错：{str(e)}")
           return f"抱歉，我遇到了一些问题：{str(e)}", messages

   except Exception as e:
       print(f"发生错误：{str(e)}")
       return "抱歉，发生了一些错误，请稍后再试。", messages

def main():
   messages = [{
       "role": "system",
       "content": """下面，你要模仿一个聪明、傲娇、淘气的人和我对话。"""
   }]

   print("开始对话，输入 'quit' 结束对话")

   while True:
       user_input = input("\n我说：")
       if user_input.lower() == 'quit':
           break

       response_text, messages = process_conversation(user_input, messages)
       print("AI说：", response_text)

if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\n对话已结束")
   except Exception as e:
       print(f"程序发生错误：{str(e)}")