#!/usr/bin/env python3
"""
数据库初始化脚本
创建数据库表并插入初始数据
"""

from gmg import app
from models import db, SystemConfig

def init_database():
    """初始化数据库"""
    with app.app_context():
        # 删除所有表（如果存在）
        db.drop_all()
        
        # 创建所有表
        db.create_all()
        
        # 插入初始配置数据
        initial_configs = [
            {
                'key': 'azure_openai_api_key',
                'value': '',
                'description': 'Azure OpenAI API密钥'
            },
            {
                'key': 'azure_openai_endpoint',
                'value': 'https://gennn-assisted-learn.openai.azure.com/',
                'description': 'Azure OpenAI服务端点'
            },
            {
                'key': 'azure_openai_api_version',
                'value': '2024-02-01',
                'description': 'Azure OpenAI API版本'
            },
            {
                'key': 'azure_openai_deployment_name',
                'value': 'gpt-4',
                'description': 'Azure OpenAI部署名称'
            },
            {
                'key': 'max_file_size_mb',
                'value': '100',
                'description': '最大文件上传大小（MB）'
            },
            {
                'key': 'supported_algorithms',
                'value': '["linear_regression", "random_forest", "svm", "gradient_boosting", "lstm"]',
                'description': '支持的机器学习算法列表'
            }
        ]
        
        for config_data in initial_configs:
            config = SystemConfig(
                key=config_data['key'],
                value=config_data['value'],
                description=config_data['description']
            )
            db.session.add(config)
        
        # 提交更改
        db.session.commit()
        
        print("数据库初始化完成！")
        print("创建的表:")
        print("- datasets (数据集)")
        print("- market_data (市场数据)")
        print("- ml_models (机器学习模型)")
        print("- model_explanations (模型解释)")
        print("- generated_codes (生成代码)")
        print("- system_configs (系统配置)")

if __name__ == '__main__':
    init_database()
