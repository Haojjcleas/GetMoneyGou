"""
系统状态API
提供系统健康检查和状态监控
"""

from flask import Blueprint, jsonify
import os
import psutil
from datetime import datetime
import traceback

from models import db, Dataset, MLModel, ModelExplanation, GeneratedCode
from utils.azure_openai_client import test_azure_openai_connection

system_status_bp = Blueprint('system_status', __name__, url_prefix='/system')

@system_status_bp.route('/health', methods=['GET'])
def health_check():
    """系统健康检查"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'components': {}
        }
        
        # 数据库连接检查
        try:
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            health_status['components']['database'] = {
                'status': 'healthy',
                'message': '数据库连接正常'
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'message': f'数据库连接失败: {str(e)}'
            }
            health_status['status'] = 'degraded'
        
        # Azure OpenAI连接检查
        try:
            ai_test = test_azure_openai_connection()
            health_status['components']['azure_openai'] = {
                'status': 'healthy' if ai_test['success'] else 'degraded',
                'message': ai_test['message']
            }
            if not ai_test['success']:
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['azure_openai'] = {
                'status': 'unhealthy',
                'message': f'AI服务检查失败: {str(e)}'
            }
            health_status['status'] = 'degraded'
        
        # 文件系统检查
        try:
            upload_folder = 'uploads'
            model_folder = 'models'
            
            upload_exists = os.path.exists(upload_folder)
            model_exists = os.path.exists(model_folder)
            
            if upload_exists and model_exists:
                health_status['components']['filesystem'] = {
                    'status': 'healthy',
                    'message': '文件系统正常'
                }
            else:
                health_status['components']['filesystem'] = {
                    'status': 'degraded',
                    'message': f'文件夹状态 - uploads: {upload_exists}, models: {model_exists}'
                }
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['components']['filesystem'] = {
                'status': 'unhealthy',
                'message': f'文件系统检查失败: {str(e)}'
            }
            health_status['status'] = 'degraded'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': f'健康检查失败: {str(e)}'
        }), 500

@system_status_bp.route('/stats', methods=['GET'])
def system_stats():
    """系统统计信息"""
    try:
        # 数据库统计
        dataset_count = Dataset.query.count()
        model_count = MLModel.query.count()
        explanation_count = ModelExplanation.query.count()
        code_count = GeneratedCode.query.count()
        
        # 模型状态统计
        model_status_stats = db.session.query(
            MLModel.status,
            db.func.count(MLModel.id).label('count')
        ).group_by(MLModel.status).all()
        
        model_status_dict = {status: count for status, count in model_status_stats}
        
        # 算法统计
        algorithm_stats = db.session.query(
            MLModel.algorithm,
            db.func.count(MLModel.id).label('count')
        ).group_by(MLModel.algorithm).all()
        
        algorithm_dict = {algorithm: count for algorithm, count in algorithm_stats}
        
        # 系统资源统计
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_resources = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'disk_usage_percent': disk.percent,
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'disk_used_gb': round(disk.used / (1024**3), 2)
            }
        except Exception as e:
            system_resources = {
                'error': f'无法获取系统资源信息: {str(e)}'
            }
        
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_stats': {
                'total_datasets': dataset_count,
                'total_models': model_count,
                'total_explanations': explanation_count,
                'total_generated_codes': code_count
            },
            'model_stats': {
                'by_status': model_status_dict,
                'by_algorithm': algorithm_dict
            },
            'system_resources': system_resources
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        error_msg = f'获取系统统计失败: {str(e)}'
        print(f"Error in system_stats: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@system_status_bp.route('/performance', methods=['GET'])
def performance_metrics():
    """性能指标"""
    try:
        # 模型性能统计
        completed_models = MLModel.query.filter_by(status='completed').all()
        
        if completed_models:
            accuracies = [m.accuracy for m in completed_models if m.accuracy is not None]
            precisions = [m.precision for m in completed_models if m.precision is not None]
            recalls = [m.recall for m in completed_models if m.recall is not None]
            f1_scores = [m.f1_score for m in completed_models if m.f1_score is not None]
            
            performance_stats = {
                'total_completed_models': len(completed_models),
                'accuracy_stats': {
                    'avg': round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
                    'min': round(min(accuracies), 4) if accuracies else 0,
                    'max': round(max(accuracies), 4) if accuracies else 0
                },
                'precision_stats': {
                    'avg': round(sum(precisions) / len(precisions), 4) if precisions else 0,
                    'min': round(min(precisions), 4) if precisions else 0,
                    'max': round(max(precisions), 4) if precisions else 0
                },
                'recall_stats': {
                    'avg': round(sum(recalls) / len(recalls), 4) if recalls else 0,
                    'min': round(min(recalls), 4) if recalls else 0,
                    'max': round(max(recalls), 4) if recalls else 0
                },
                'f1_stats': {
                    'avg': round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
                    'min': round(min(f1_scores), 4) if f1_scores else 0,
                    'max': round(max(f1_scores), 4) if f1_scores else 0
                }
            }
        else:
            performance_stats = {
                'total_completed_models': 0,
                'message': '暂无已完成的模型'
            }
        
        # 训练时间统计
        models_with_time = MLModel.query.filter(
            MLModel.training_start_time.isnot(None),
            MLModel.training_end_time.isnot(None)
        ).all()
        
        if models_with_time:
            training_times = []
            for model in models_with_time:
                duration = (model.training_end_time - model.training_start_time).total_seconds()
                training_times.append(duration)
            
            time_stats = {
                'avg_training_time_seconds': round(sum(training_times) / len(training_times), 2),
                'min_training_time_seconds': round(min(training_times), 2),
                'max_training_time_seconds': round(max(training_times), 2)
            }
        else:
            time_stats = {
                'message': '暂无训练时间数据'
            }
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_performance': performance_stats,
            'training_time_stats': time_stats
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        error_msg = f'获取性能指标失败: {str(e)}'
        print(f"Error in performance_metrics: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@system_status_bp.route('/cleanup', methods=['POST'])
def cleanup_system():
    """系统清理"""
    try:
        cleanup_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'actions': []
        }
        
        # 清理失败的模型
        failed_models = MLModel.query.filter_by(status='failed').all()
        if failed_models:
            for model in failed_models:
                # 删除模型文件
                if model.model_file_path and os.path.exists(model.model_file_path):
                    os.remove(model.model_file_path)
                
                # 删除数据库记录
                db.session.delete(model)
            
            db.session.commit()
            cleanup_results['actions'].append({
                'action': 'delete_failed_models',
                'count': len(failed_models),
                'message': f'删除了 {len(failed_models)} 个失败的模型'
            })
        
        # 清理孤立的代码记录
        orphaned_codes = GeneratedCode.query.filter(
            ~GeneratedCode.model_id.in_(
                db.session.query(MLModel.id).filter_by(status='completed')
            )
        ).all()
        
        if orphaned_codes:
            for code in orphaned_codes:
                db.session.delete(code)
            
            db.session.commit()
            cleanup_results['actions'].append({
                'action': 'delete_orphaned_codes',
                'count': len(orphaned_codes),
                'message': f'删除了 {len(orphaned_codes)} 个孤立的代码记录'
            })
        
        if not cleanup_results['actions']:
            cleanup_results['actions'].append({
                'action': 'no_cleanup_needed',
                'message': '系统无需清理'
            })
        
        return jsonify(cleanup_results), 200
        
    except Exception as e:
        db.session.rollback()
        error_msg = f'系统清理失败: {str(e)}'
        print(f"Error in cleanup_system: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@system_status_bp.route('/info', methods=['GET'])
def system_info():
    """系统信息"""
    try:
        import platform
        import sys
        
        info = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'architecture': platform.architecture()[0]
            },
            'application': {
                'name': 'GetMoneyGou',
                'version': '1.0.0',
                'description': '智能金融数据分析与预测平台'
            },
            'features': {
                'data_upload': True,
                'data_preprocessing': True,
                'machine_learning': True,
                'ai_explanation': True,
                'code_generation': True,
                'online_testing': True
            },
            'supported_algorithms': [
                'logistic_regression',
                'random_forest',
                'svm',
                'gradient_boosting'
            ],
            'supported_languages': [
                'python',
                'cpp'
            ]
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': f'获取系统信息失败: {str(e)}'}), 500
