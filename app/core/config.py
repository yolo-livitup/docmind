# app/core/config.py
"""
统一配置中心
使用 pydantic-settings 读取 .env 文件和环境变量，集中管理所有配置项
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 通义千问 API Key（从 .env 读取 DASHSCOPE_API_KEY）
    DASHSCOPE_API_KEY: str = ""

    # 向量化模型，把文本转成数字向量
    EMBEDDING_MODEL: str = "tongyi-embedding-vision-plus-2026-03-06"

    # 大语言模型，生成回答
    LLM_MODEL: str = "deepseek-v4-flash"

    # 向量数据库存储路径，ChromaDB 数据存在这个目录
    CHROMA_PATH: str = "./chroma_db"

    # 用户上传的文档存放目录
    UPLOAD_DIR: str = "./uploads"

    class Config:
        # 指定从项目根目录的 .env 文件加载配置
        env_file = ".env"


# 全局配置实例，其他文件直接 from app.core.config import settings 使用
settings = Settings()