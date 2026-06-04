# app/rag/embeddings.py
"""
向量化模块
把文本转成数学向量，语义相近的文本向量也相近
这样用户提问"怎么退票"能匹配到文档里的"退款流程"
"""
from langchain_community.embeddings import DashScopeEmbeddings
from app.core.config import settings


def get_embeddings():
    """
    获取嵌入模型实例
    使用通义千问 tongyi-embedding-vision-plus-2026-03-06
    """
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
    )