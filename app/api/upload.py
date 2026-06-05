# app/api/upload.py
"""
文档上传接口
接收用户上传的文件，执行完整 RAG 建库流程：
加载 → 切片 → 向量化 → 存入 ChromaDB
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.rag.loader import load_document
from app.rag.splitter import split_documents
from app.rag.store import create_vectorstore

router = APIRouter(prefix="/api", tags=["文档管理"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文档接口
    参数: file - 用户上传的文件 (PDF/Word/MD/TXT)
    返回: 切片数量和存储状态
    """
    # 1. 检查文件格式
    allowed_ext = {".pdf", ".docx", ".md", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}，仅支持 {allowed_ext}")

    # 2. 保存上传的文件到本地
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 3. 文档加载
        docs = load_document(file_path)

        # 4. 切片
        chunks = split_documents(docs)

        # 5. 向量化 + 存储
        vectorstore = create_vectorstore(chunks)
        collection_count = vectorstore._collection.count()

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "vectors": collection_count,
            "message": f"文档已成功处理，共生成 {len(chunks)} 个切片，{collection_count} 条向量"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")