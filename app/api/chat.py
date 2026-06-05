# app/api/chat.py
"""
对话问答接口
接收用户问题，调 RAG 链检索 + 生成回答，附带来源溯源
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.chain import create_qa_chain

router = APIRouter(prefix="/api", tags=["对话问答"])

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        _chain = create_qa_chain()
    return _chain


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        chain = get_chain()
        result = chain.invoke(req.question)

        # 从链返回结果中取来源文档
        source_docs = result.get("source_documents", [])
        sources = list(set(
            doc.metadata.get("source", "未知来源") for doc in source_docs
        ))

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")