"""
Web页面路由
提供静态HTML页面
"""

from flask import Blueprint, send_from_directory, current_app
import os

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """主页"""
    return send_from_directory(current_app.static_folder, 'index.html')

@web_bp.route('/health')
def health_check():
    """健康检查"""
    return {'status': 'ok', 'message': 'GetMoneyGou API is running'}
