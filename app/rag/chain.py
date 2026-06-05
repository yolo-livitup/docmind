# app/rag/chain.py
"""
问答链模块 - LCEL 管道写法
检索 → 拼 Prompt → LLM 生成，每一步独立可控
"""
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings
from app.rag.retriever import get_retriever
from langchain_core.runnables import RunnableParallel
# 自定义 Prompt 模板
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个企业知识库助手。严格根据以下文档内容回答问题。如果文档中没有答案，明确说'知识库中未找到相关信息'。"),
    ("user", "文档内容：\n{context}\n\n用户问题：{question}\n\n回答时请引用原文来源。"),
])



def create_qa_chain(collection_name: str = "docmind"):
    llm = ChatDeepSeek(model=settings.LLM_MODEL, api_key=settings.DASHSCOPE_API_KEY)
    retriever = get_retriever(collection_name, k=5)

    def format_docs(docs):
        return "\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # 同时返回回答和来源文档
    chain = RunnableParallel(
        answer=rag_chain,
        source_documents=retriever,
    )
    return chain