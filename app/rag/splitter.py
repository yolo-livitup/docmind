# app/rag/splitter.py
"""
智能文本切片器
把长文档切成小块，方便向量检索和 LLM 处理
切片策略直接影响检索质量——太大检索不准，太小丢失上下文
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: list):
    """
    对文档列表进行语义切片
    参数: docs - Document 对象列表
    返回: 切片后的 Document 对象列表
    """
    # chunk_size=500: 每块最多 500 个字符
    # chunk_overlap=50: 相邻两块重叠 50 个字符，防止关键信息被切断
    # separators: 按优先级依次尝试切割——先按标题切，再按段落，最后按句子
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n", "。", ".", "；", ";", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    return chunks