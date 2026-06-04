"""
文档加载器
负责读取用户上传的文档，支持 PDF、Word、Markdown、TXT 格式
返回 LangChain 的 Document 对象列表
"""
from langchain_community.document_loaders import (
    PyPDFLoader,       # PDF 文件
    Docx2txtLoader,    # Word 文件
    TextLoader,        # TXT 文件
    UnstructuredMarkdownLoader,  # Markdown 文件
)
from pathlib import Path


def load_document(file_path: str):
    """
    根据文件后缀自动选择对应的加载器
    参数: file_path - 文件路径
    返回: Document 对象列表，每个 Document 包含 page_content 和 metadata
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    docs = loader.load()
    return docs