import os
from llama_index.readers.dashscope.base import DashScopeParse
from llama_index.readers.dashscope.utils import ResultType
from llama_index.indices.managed.dashscope import DashScopeCloudIndex, DashScopeCloudRetriever
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from typing import cast
from enum import Enum

def ingest_data(file_folder: str, name: str, category_id="default"):
    # 本地文档上传，并完成文档解析，可以指定百炼数据中心的目录
    parse = DashScopeParse(result_type=ResultType.DASHSCOPE_DOCMIND, category_id=category_id)
    file_extractor = {".pdf": parse, '.doc': parse, '.docx': parse}
    documents = SimpleDirectoryReader(
        file_folder, file_extractor=file_extractor
    ).load_data(num_workers=4)

    # 构建知识索引，完成文档切分、向量化和入库操作
    _ = DashScopeCloudIndex.from_documents(documents, name, verbose=True)
    return documents