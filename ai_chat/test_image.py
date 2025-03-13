import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=[{"role": "user","content": [
            {"type": "text","text": "图中有几个人？"},
            {"type": "image_url",
             "image_url": {"url": "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fsafe-img.xhscdn.com%2Fbw1%2Fc72f6a45-fd95-48a4-95a7-631f09de5796%3FimageView2%2F2%2Fw%2F1080%2Fformat%2Fjpg&refer=http%3A%2F%2Fsafe-img.xhscdn.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1741768464&t=9bf4a009691d56d677fec6a3598c1b19"}}
            ]}]
    )
print(completion.model_dump_json())