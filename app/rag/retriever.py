# app/rag/retriever.py
"""
混合检索模块
结合 BM25 关键词检索 + 语义向量检索，两者互补提升召回率
"""

from langchain_classic.retrievers import EnsembleRetriever
from app.rag.store import load_vectorstore


def get_retriever(collection_name: str = "docmind", k: int = 5):
    """
    创建混合检索器
    参数: collection_name - ChromaDB 集合名称
          k - 最终返回的文档块数量
    返回: EnsembleRetriever 混合检索器
    """
    # 加载向量库
    vectorstore = load_vectorstore(collection_name)

    # 语义检索：基于向量相似度
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    # 关键词检索：ChromaDB 内置的全文检索，不额外加载内存
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance，兼顾相关性和多样性
        search_kwargs={
            "k": k * 2,
            "fetch_k": k * 4,
            "lambda_mult": 0.7,  # 0=纯多样性，1=纯相关性
        }
    )


    # 混合检索 = 语义（相关性）+ MMR（多样性）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever,mmr_retriever],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever