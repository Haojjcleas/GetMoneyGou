import os
from dotenv import load_dotenv
load_dotenv()
from config.config import Config
from config.config_dev import DevConfig
from config.config_prod import ProdConfig

env_name = os.getenv("APP_ENV")
if env_name == "prod":
    config = ProdConfig()
else:
    config = DevConfig()

__all__ = ["config"]
