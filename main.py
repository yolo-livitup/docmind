# main.py
"""
DocMind 启动入口
FastAPI 服务，挂载上传和问答接口
"""
from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(title="DocMind", description="企业智能知识库问答系统")

# 注册路由
app.include_router(upload_router)
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)