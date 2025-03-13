import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
file_object = client.files.create(file=Path("file/money.xlsx"), purpose="file-extract")
completion = client.chat.completions.create(
    model="qwen-long",
    messages=[
        {'role': 'system', 'content': f'fileid://{file_object.id}'},
        {'role': 'user', 'content': '无形资产的年初余额和年末余额相差多少？上升了还是下降了？'}
    ]
)
print(completion.model_dump_json())