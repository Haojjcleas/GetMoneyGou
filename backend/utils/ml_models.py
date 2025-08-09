"""
机器学习模型管理器
实现多种机器学习算法的训练和预测
"""

import numpy as np
import pandas as pd
import logging
import multiprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Callable
import json

class MLModelManager:
    """机器学习模型管理器"""
    
    def __init__(self, model_storage_path: str = "models", progress_callback: Optional[Callable] = None):
        self.model_storage_path = model_storage_path
        self.progress_callback = progress_callback
        self.n_cores = min(multiprocessing.cpu_count(), 4)  # 限制最大核心数
        os.makedirs(model_storage_path, exist_ok=True)

        # 支持的算法配置
        self.algorithms = {
            'linear_regression': {
                'type': 'regression',
                'model_class': LinearRegression,
                'default_params': {}
            },
            'logistic_regression': {
                'type': 'classification',
                'model_class': LogisticRegression,
                'default_params': {'random_state': 42, 'max_iter': 1000}
            },
            'random_forest': {
                'type': 'both',
                'classification_class': RandomForestClassifier,
                'regression_class': RandomForestRegressor,
                'default_params': {'n_estimators': 100, 'random_state': 42}
            },
            'svm': {
                'type': 'both',
                'classification_class': SVC,
                'regression_class': SVR,
                'default_params': {'random_state': 42}
            },
            'gradient_boosting': {
                'type': 'both',
                'classification_class': GradientBoostingClassifier,
                'regression_class': GradientBoostingRegressor,
                'default_params': {'n_estimators': 100, 'random_state': 42}
            }
        }

    def _report_progress(self, message: str, progress: float = None):
        """报告训练进度"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            logging.info(f"[训练进度] {message}")

    def get_model_instance(self, algorithm: str, task_type: str = 'classification',
                          custom_params: Optional[Dict] = None) -> Any:
        """获取模型实例"""
        if algorithm not in self.algorithms:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        algo_config = self.algorithms[algorithm]
        params = algo_config['default_params'].copy()

        # 为支持多核的算法添加n_jobs参数
        if algorithm in ['random_forest', 'gradient_boosting']:
            params['n_jobs'] = self.n_cores

        if custom_params:
            params.update(custom_params)

        # 根据任务类型选择模型类
        if algo_config['type'] == 'both':
            if task_type == 'classification':
                model_class = algo_config['classification_class']
            else:
                model_class = algo_config['regression_class']
        else:
            model_class = algo_config['model_class']

        return model_class(**params)
    
    def train_model(self, algorithm: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   task_type: str = 'classification',
                   custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """训练模型并返回结果（优化版本）"""

        try:
            self._report_progress("开始模型训练...", 0.0)

            # 获取模型实例（添加多核支持）
            model = self.get_model_instance(algorithm, task_type, custom_params)

            # 为支持多核的算法添加n_jobs参数
            if hasattr(model, 'n_jobs') and algorithm in ['random_forest', 'gradient_boosting']:
                if hasattr(model, 'set_params'):
                    model.set_params(n_jobs=self.n_cores)
                    self._report_progress(f"使用 {self.n_cores} 个CPU核心进行训练", 0.1)

            # 记录训练开始时间
            train_start_time = datetime.now()
            self._report_progress("模型训练中...", 0.2)

            # 训练模型
            model.fit(X_train, y_train)
            self._report_progress("模型训练完成", 0.6)
            
            # 记录训练结束时间
            train_end_time = datetime.now()
            training_duration = (train_end_time - train_start_time).total_seconds()
            self._report_progress(f"训练耗时: {training_duration:.2f}秒", 0.7)

            # 预测
            self._report_progress("生成预测结果...", 0.75)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # 计算性能指标
            self._report_progress("计算性能指标...", 0.8)
            metrics = self._calculate_metrics(
                y_train, y_train_pred, y_test, y_test_pred, task_type
            )

            # 交叉验证（使用多核）
            self._report_progress("执行交叉验证...", 0.85)
            cv_scores = self._perform_cross_validation(model, X_train, y_train, task_type)

            # 特征重要性（如果模型支持）
            self._report_progress("分析特征重要性...", 0.9)
            feature_importance = self._get_feature_importance(model)

            self._report_progress("模型训练完成！", 1.0)
            
            return {
                'model': model,
                'metrics': metrics,
                'cross_validation': cv_scores,
                'feature_importance': feature_importance,
                'training_time': training_duration,
                'predictions': {
                    'train': y_train_pred.tolist(),
                    'test': y_test_pred.tolist()
                },
                'model_params': model.get_params()
            }
            
        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")
    
    def _calculate_metrics(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                          y_test_true: np.ndarray, y_test_pred: np.ndarray,
                          task_type: str) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {}
        
        if task_type == 'classification':
            # 分类指标
            metrics.update({
                'train_accuracy': accuracy_score(y_train_true, y_train_pred),
                'test_accuracy': accuracy_score(y_test_true, y_test_pred),
                'train_precision': precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0),
                'train_f1': f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
                'test_f1': f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
            })
        else:
            # 回归指标
            metrics.update({
                'train_mse': mean_squared_error(y_train_true, y_train_pred),
                'test_mse': mean_squared_error(y_test_true, y_test_pred),
                'train_mae': mean_absolute_error(y_train_true, y_train_pred),
                'test_mae': mean_absolute_error(y_test_true, y_test_pred),
                'train_r2': r2_score(y_train_true, y_train_pred),
                'test_r2': r2_score(y_test_true, y_test_pred)
            })
        
        return metrics
    
    def _perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 task_type: str, cv: int = 5) -> Dict[str, float]:
        """执行交叉验证（多核优化）"""
        try:
            if task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'r2'

            # 使用多核进行交叉验证
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_cores  # 使用多核
            )

            return {
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            logging.error(f"交叉验证失败: {e}")
            return {'cv_mean': 0.0, 'cv_std': 0.0, 'cv_scores': []}
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict]:
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                return {
                    'importances': model.feature_importances_.tolist(),
                    'type': 'feature_importances'
                }
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # 对于多类分类，取第一个类的系数
                return {
                    'importances': np.abs(coef).tolist(),
                    'type': 'coefficients'
                }
            else:
                return None
        except Exception as e:
            print(f"获取特征重要性失败: {e}")
            return None
    
    def save_model(self, model: Any, model_id: int, algorithm: str) -> str:
        """保存模型到文件"""
        try:
            filename = f"model_{model_id}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            filepath = os.path.join(self.model_storage_path, filename)
            
            joblib.dump(model, filepath)
            return filepath
        except Exception as e:
            raise Exception(f"保存模型失败: {str(e)}")
    
    def load_model(self, filepath: str) -> Any:
        """从文件加载模型"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")
            
            return joblib.load(filepath)
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测"""
        try:
            return model.predict(X)
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """获取预测概率（仅适用于分类模型）"""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return None
        except Exception as e:
            print(f"获取预测概率失败: {e}")
            return None
    
    def get_supported_algorithms(self) -> Dict[str, Dict]:
        """获取支持的算法列表"""
        return {
            algo: {
                'type': config['type'],
                'default_params': config['default_params']
            }
            for algo, config in self.algorithms.items()
        }
    
    def validate_algorithm_params(self, algorithm: str, params: Dict) -> Tuple[bool, str]:
        """验证算法参数"""
        if algorithm not in self.algorithms:
            return False, f"不支持的算法: {algorithm}"
        
        try:
            # 尝试创建模型实例来验证参数
            self.get_model_instance(algorithm, 'classification', params)
            return True, "参数验证通过"
        except Exception as e:
            return False, f"参数验证失败: {str(e)}"
