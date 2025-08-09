"""
机器学习训练API
提供模型训练、管理和预测的接口
"""

from flask import Blueprint, request, jsonify, current_app
import traceback
import json
from datetime import datetime
import threading
import os
import numpy as np

from utils.ml_models import MLModelManager
from utils.data_preprocessing import DataPreprocessor
from models import db, MLModel, Dataset

ml_training_bp = Blueprint('ml_training', __name__, url_prefix='/ml')

# 全局模型管理器
model_manager = MLModelManager()

@ml_training_bp.route('/algorithms', methods=['GET'])
def get_supported_algorithms():
    """获取支持的算法列表"""
    try:
        algorithms = model_manager.get_supported_algorithms()
        return jsonify({
            'algorithms': algorithms,
            'message': '获取算法列表成功'
        }), 200
    except Exception as e:
        return jsonify({'error': f'获取算法列表失败: {str(e)}'}), 500

@ml_training_bp.route('/datasets/<int:dataset_id>/models', methods=['POST'])
def create_model(dataset_id):
    """创建并训练新模型"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取请求参数
        data = request.get_json()
        algorithm = data.get('algorithm')
        model_name = data.get('name', f'{algorithm}_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        custom_params = data.get('parameters', {})
        task_type = data.get('task_type', 'classification')
        test_size = data.get('test_size', 0.2)
        
        # 验证参数
        if not algorithm:
            return jsonify({'error': '必须指定算法'}), 400
        
        if algorithm not in model_manager.get_supported_algorithms():
            return jsonify({'error': f'不支持的算法: {algorithm}'}), 400
        
        if task_type not in ['classification', 'regression']:
            return jsonify({'error': '任务类型必须是 classification 或 regression'}), 400
        
        # 验证算法参数
        is_valid, validation_message = model_manager.validate_algorithm_params(algorithm, custom_params)
        if not is_valid:
            return jsonify({'error': validation_message}), 400
        
        # 创建模型记录
        ml_model = MLModel(
            dataset_id=dataset_id,
            name=model_name,
            algorithm=algorithm,
            status='pending',
            created_time=datetime.utcnow()
        )
        ml_model.set_parameters({
            'algorithm_params': custom_params,
            'task_type': task_type,
            'test_size': test_size,
            'prediction_horizon': 1  # 固定为1个时间点
        })
        
        db.session.add(ml_model)
        db.session.commit()
        
        # 异步训练模型
        training_thread = threading.Thread(
            target=train_model_async,
            args=(current_app._get_current_object(), ml_model.id, dataset_id, algorithm, custom_params, task_type, test_size)
        )
        training_thread.start()
        
        return jsonify({
            'message': '模型创建成功，正在后台训练',
            'model': ml_model.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        error_msg = f'创建模型失败: {str(e)}'
        print(f"Error in create_model: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

def train_model_async(app, model_id, dataset_id, algorithm, custom_params, task_type, test_size):
    """异步训练模型"""
    try:
        with app.app_context():
            # 获取模型记录
            ml_model = MLModel.query.get(model_id)
            if not ml_model:
                return
            
            # 更新状态为训练中
            ml_model.status = 'training'
            ml_model.training_start_time = datetime.utcnow()
            db.session.commit()
            
            # 定义进度回调函数
            def progress_callback(message, progress=None):
                print(f"[模型 {model_id}] {message}")
                if progress is not None:
                    print(f"[模型 {model_id}] 进度: {progress*100:.1f}%")

            # 准备数据（带进度回调）
            preprocessor = DataPreprocessor(progress_callback=progress_callback)
            data_dict = preprocessor.prepare_data_for_ml(
                dataset_id=dataset_id,
                test_size=test_size
            )
            
            # 选择目标变量
            if task_type == 'classification':
                y_train = data_dict['y_train_binary']
                y_test = data_dict['y_test_binary']
            else:
                y_train = data_dict['y_train_regression']
                y_test = data_dict['y_test_regression']
            
            # 创建带进度回调的模型管理器
            training_model_manager = MLModelManager(progress_callback=progress_callback)

            # 训练模型
            training_result = training_model_manager.train_model(
                algorithm=algorithm,
                X_train=data_dict['X_train'],
                y_train=y_train,
                X_test=data_dict['X_test'],
                y_test=y_test,
                task_type=task_type,
                custom_params=custom_params,
            )
            
            # 保存模型文件
            model_file_path = model_manager.save_model(
                training_result['model'], model_id, algorithm
            )
            
            # 更新模型记录
            ml_model.status = 'completed'
            ml_model.training_end_time = datetime.utcnow()
            ml_model.model_file_path = model_file_path
            
            # 保存性能指标
            metrics = training_result['metrics']
            if task_type == 'classification':
                ml_model.accuracy = metrics.get('test_accuracy')
                ml_model.precision = metrics.get('test_precision')
                ml_model.recall = metrics.get('test_recall')
                ml_model.f1_score = metrics.get('test_f1')
            else:
                ml_model.mse = metrics.get('test_mse')
                ml_model.mae = metrics.get('test_mae')
                ml_model.r2_score = metrics.get('test_r2')
            
            # 保存预测结果
            ml_model.set_predictions(training_result['predictions']['test'])
            
            db.session.commit()
            
    except Exception as e:
        try:
            with app.app_context():
                ml_model = MLModel.query.get(model_id)
                if ml_model:
                    ml_model.status = 'failed'
                    ml_model.training_end_time = datetime.utcnow()
                    db.session.commit()
        except:
            pass
        
        print(f"异步训练失败: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

@ml_training_bp.route('/datasets/<int:dataset_id>/models', methods=['GET'])
def get_models(dataset_id):
    """获取数据集的所有模型"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取模型列表
        models = MLModel.query.filter_by(dataset_id=dataset_id)\
                            .order_by(MLModel.created_time.desc()).all()
        
        return jsonify({
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name
            },
            'models': [model.to_dict() for model in models]
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}'}), 500

@ml_training_bp.route('/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """获取特定模型的详细信息"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        model_info = model.to_dict()
        
        # 添加额外信息
        if model.training_start_time and model.training_end_time:
            training_duration = (model.training_end_time - model.training_start_time).total_seconds()
            model_info['training_duration_seconds'] = training_duration
        
        return jsonify(model_info), 200
        
    except Exception as e:
        return jsonify({'error': f'获取模型详情失败: {str(e)}'}), 500

@ml_training_bp.route('/models/<int:model_id>/predict', methods=['POST'])
def predict_with_model(model_id):
    """使用模型进行预测"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        if model.status != 'completed':
            return jsonify({'error': '模型尚未训练完成'}), 400
        
        if not model.model_file_path or not os.path.exists(model.model_file_path):
            return jsonify({'error': '模型文件不存在'}), 404
        
        # 获取预测数据
        data = request.get_json()
        prediction_data = data.get('data')
        
        if not prediction_data:
            return jsonify({'error': '缺少预测数据'}), 400
        
        # 加载模型
        trained_model = model_manager.load_model(model.model_file_path)
        
        # 进行预测
        predictions = model_manager.predict(trained_model, np.array(prediction_data))
        
        # 获取预测概率（如果是分类模型）
        probabilities = model_manager.predict_proba(trained_model, np.array(prediction_data))
        
        result = {
            'model_id': model_id,
            'predictions': predictions.tolist(),
            'prediction_count': len(predictions)
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities.tolist()
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f'预测失败: {str(e)}'
        print(f"Error in predict_with_model: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@ml_training_bp.route('/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        # 删除模型文件
        if model.model_file_path and os.path.exists(model.model_file_path):
            os.remove(model.model_file_path)
        
        # 删除数据库记录
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({'message': '模型删除成功'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除模型失败: {str(e)}'}), 500
