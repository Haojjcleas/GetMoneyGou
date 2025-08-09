from blueprints.api import api_bp
from blueprints.web import web_bp

all_bp = [api_bp, web_bp]

__all__ = ['all_bp']
