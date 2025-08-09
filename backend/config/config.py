from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 5001

    # 数据库配置
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///gmg.db"
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # 文件上传配置
    UPLOAD_FOLDER: str = "uploads"
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set = None

    # AI API 配置
    AI_PROVIDER: str = os.getenv('AI_PROVIDER', 'deepseek')  # 'azure' 或 'deepseek'

    # Azure OpenAI 配置
    AZURE_OPENAI_ENDPOINT: str = "https://gennn-assisted-learn.openai.azure.com/"
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4"

    # Deepseek 配置
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # 模型存储配置
    MODEL_STORAGE_PATH: str = "models"

    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {'csv'}

        # 确保目录存在
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.MODEL_STORAGE_PATH, exist_ok=True)