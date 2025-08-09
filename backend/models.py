"""
数据库模型定义
包含所有数据表的SQLAlchemy模型
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Dataset(db.Model):
    """数据集表 - 存储上传的CSV文件信息"""
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    total_records = db.Column(db.Integer, default=0)
    date_range_start = db.Column(db.DateTime)
    date_range_end = db.Column(db.DateTime)
    description = db.Column(db.Text)
    
    # 关联关系
    market_data = db.relationship('MarketData', backref='dataset', lazy=True, cascade='all, delete-orphan')
    models = db.relationship('MLModel', backref='dataset', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'total_records': self.total_records,
            'date_range_start': self.date_range_start.isoformat() if self.date_range_start else None,
            'date_range_end': self.date_range_end.isoformat() if self.date_range_end else None,
            'description': self.description
        }

class MarketData(db.Model):
    """市场数据表 - 存储OHLCV和买卖盘数据"""
    __tablename__ = 'market_data'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    buy_amount = db.Column(db.Float, nullable=False)
    sell_amount = db.Column(db.Float, nullable=False)
    
    # 索引
    __table_args__ = (
        db.Index('idx_dataset_timestamp', 'dataset_id', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'buy_amount': self.buy_amount,
            'sell_amount': self.sell_amount
        }

class MLModel(db.Model):
    """机器学习模型表 - 存储模型信息和结果"""
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    algorithm = db.Column(db.String(100), nullable=False)  # linear_regression, random_forest, svm, gradient_boosting, lstm
    status = db.Column(db.String(50), default='pending')  # pending, training, completed, failed
    created_time = db.Column(db.DateTime, default=datetime.utcnow)
    training_start_time = db.Column(db.DateTime)
    training_end_time = db.Column(db.DateTime)
    
    # 模型参数 (JSON格式存储)
    parameters = db.Column(db.Text)  # JSON string
    
    # 性能指标
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    mse = db.Column(db.Float)  # Mean Squared Error
    mae = db.Column(db.Float)  # Mean Absolute Error
    r2_score = db.Column(db.Float)  # R-squared
    
    # 模型文件路径
    model_file_path = db.Column(db.String(500))
    
    # 预测结果
    predictions = db.Column(db.Text)  # JSON string
    
    # 关联关系
    explanations = db.relationship('ModelExplanation', backref='model', lazy=True, cascade='all, delete-orphan')
    generated_codes = db.relationship('GeneratedCode', backref='model', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'name': self.name,
            'algorithm': self.algorithm,
            'status': self.status,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
            'training_end_time': self.training_end_time.isoformat() if self.training_end_time else None,
            'parameters': json.loads(self.parameters) if self.parameters else None,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'predictions': json.loads(self.predictions) if self.predictions else None
        }
    
    def set_parameters(self, params_dict):
        """设置模型参数"""
        self.parameters = json.dumps(params_dict)
    
    def get_parameters(self):
        """获取模型参数"""
        return json.loads(self.parameters) if self.parameters else {}
    
    def set_predictions(self, predictions_list):
        """设置预测结果"""
        self.predictions = json.dumps(predictions_list)
    
    def get_predictions(self):
        """获取预测结果"""
        return json.loads(self.predictions) if self.predictions else []

class ModelExplanation(db.Model):
    """模型解释表 - 存储AI解释和对话记录"""
    __tablename__ = 'model_explanations'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)  # 对话会话ID
    user_question = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    created_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'session_id': self.session_id,
            'user_question': self.user_question,
            'ai_response': self.ai_response,
            'created_time': self.created_time.isoformat() if self.created_time else None
        }

class GeneratedCode(db.Model):
    """生成代码表 - 存储生成的Python和C++代码"""
    __tablename__ = 'generated_codes'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)
    language = db.Column(db.String(20), nullable=False)  # python, cpp
    code_content = db.Column(db.Text, nullable=False)
    generated_time = db.Column(db.DateTime, default=datetime.utcnow)
    is_tested = db.Column(db.Boolean, default=False)
    test_results = db.Column(db.Text)  # JSON string
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'language': self.language,
            'code_content': self.code_content,
            'generated_time': self.generated_time.isoformat() if self.generated_time else None,
            'is_tested': self.is_tested,
            'test_results': json.loads(self.test_results) if self.test_results else None
        }
    
    def set_test_results(self, results_dict):
        """设置测试结果"""
        self.test_results = json.dumps(results_dict)
    
    def get_test_results(self):
        """获取测试结果"""
        return json.loads(self.test_results) if self.test_results else {}

class SystemConfig(db.Model):
    """系统配置表 - 存储系统配置信息"""
    __tablename__ = 'system_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    updated_time = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'updated_time': self.updated_time.isoformat() if self.updated_time else None
        }
