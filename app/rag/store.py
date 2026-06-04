# app/rag/store.py
"""
向量数据库存储模块
把向量化后的文档块存入 ChromaDB，并支持持久化到磁盘
ChromaDB 是轻量级向量数据库，适合单机部署
"""
import os
import shutil
from langchain_community.vectorstores import Chroma
from app.core.config import settings
from app.rag.embeddings import get_embeddings


def create_vectorstore(chunks: list, collection_name: str = "docmind"):
    """
    创建向量数据库，将文档块向量化后存入
    参数: chunks - 切片后的 Document 列表
          collection_name - ChromaDB 集合名称
    返回: Chroma 向量数据库实例
    """
    # 删除旧数据，避免重复追加
    persist_dir = os.path.join(settings.CHROMA_PATH, collection_name)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vectorstore


def load_vectorstore(collection_name: str = "docmind"):
    """
    从磁盘加载已有的向量数据库
    参数: collection_name - ChromaDB 集合名称
    返回: Chroma 向量数据库实例
    """
    embeddings = get_embeddings()
    persist_dir = os.path.join(settings.CHROMA_PATH, collection_name)

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vectorstore