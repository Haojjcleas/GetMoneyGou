"""
数据预处理API
提供数据预处理和特征工程的接口
"""

from flask import Blueprint, request, jsonify
import traceback
import pandas as pd

from utils.data_preprocessing import DataPreprocessor
from models import Dataset

data_preprocessing_bp = Blueprint('data_preprocessing', __name__, url_prefix='/preprocessing')

@data_preprocessing_bp.route('/datasets/<int:dataset_id>/prepare', methods=['POST'])
def prepare_dataset(dataset_id):
    """准备数据集进行机器学习"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取参数
        test_size = request.json.get('test_size', 0.2)
        prediction_horizon = request.json.get('prediction_horizon', 1)
        
        # 验证参数
        if not 0.1 <= test_size <= 0.5:
            return jsonify({'error': '测试集比例必须在0.1到0.5之间'}), 400
        
        if not 1 <= prediction_horizon <= 10:
            return jsonify({'error': '预测时间范围必须在1到10之间'}), 400
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor()
        
        # 准备数据
        data_dict = preprocessor.prepare_data_for_ml(
            dataset_id=dataset_id,
            test_size=test_size,
            prediction_horizon=prediction_horizon
        )
        
        # 返回数据统计信息
        result = {
            'message': '数据预处理完成',
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name,
                'total_records': dataset.total_records
            },
            'preprocessing_info': {
                'train_samples': len(data_dict['X_train']),
                'test_samples': len(data_dict['X_test']),
                'feature_count': len(data_dict['feature_columns']),
                'test_size': test_size,
                'prediction_horizon': prediction_horizon
            },
            'feature_columns': data_dict['feature_columns'],
            'target_distribution': {
                'train_positive_ratio': float(data_dict['y_train_binary'].mean()),
                'test_positive_ratio': float(data_dict['y_test_binary'].mean())
            },
            'data_quality': {
                'train_date_range': {
                    'start': data_dict['train_dates'].min().isoformat(),
                    'end': data_dict['train_dates'].max().isoformat()
                },
                'test_date_range': {
                    'start': data_dict['test_dates'].min().isoformat(),
                    'end': data_dict['test_dates'].max().isoformat()
                }
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f'数据预处理失败: {str(e)}'
        print(f"Error in prepare_dataset: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@data_preprocessing_bp.route('/datasets/<int:dataset_id>/features', methods=['GET'])
def get_feature_info(dataset_id):
    """获取数据集的特征信息"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor()
        
        # 加载和清洗数据
        df = preprocessor.load_data_from_db(dataset_id)
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_technical_indicators(df_clean)
        
        # 获取特征统计信息
        feature_stats = {}
        numeric_columns = df_features.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_columns:
            if not df_features[col].isnull().all():
                feature_stats[col] = {
                    'mean': float(df_features[col].mean()),
                    'std': float(df_features[col].std()),
                    'min': float(df_features[col].min()),
                    'max': float(df_features[col].max()),
                    'null_count': int(df_features[col].isnull().sum()),
                    'unique_count': int(df_features[col].nunique())
                }
        
        # 数据质量报告
        quality_report = {
            'total_rows': len(df_features),
            'total_features': len(numeric_columns),
            'missing_data_percentage': float(df_features.isnull().sum().sum() / (len(df_features) * len(df_features.columns)) * 100),
            'date_range': {
                'start': df_features.index.min().isoformat(),
                'end': df_features.index.max().isoformat(),
                'duration_days': (df_features.index.max() - df_features.index.min()).days
            }
        }
        
        return jsonify({
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name
            },
            'feature_statistics': feature_stats,
            'quality_report': quality_report
        }), 200
        
    except Exception as e:
        error_msg = f'获取特征信息失败: {str(e)}'
        print(f"Error in get_feature_info: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@data_preprocessing_bp.route('/datasets/<int:dataset_id>/sample', methods=['GET'])
def get_processed_sample(dataset_id):
    """获取预处理后的数据样本"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取参数
        sample_size = request.args.get('sample_size', 100, type=int)
        sample_size = min(sample_size, 1000)  # 限制最大样本数
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor()
        
        # 加载和处理数据
        df = preprocessor.load_data_from_db(dataset_id)
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_technical_indicators(df_clean)
        df_target = preprocessor.create_target_variable(df_features)
        
        # 移除NaN行
        df_final = df_target.dropna()
        
        # 获取样本
        if len(df_final) > sample_size:
            # 均匀采样
            step = len(df_final) // sample_size
            sample_df = df_final.iloc[::step][:sample_size]
        else:
            sample_df = df_final
        
        # 转换为JSON格式
        sample_data = []
        for idx, row in sample_df.iterrows():
            sample_data.append({
                'timestamp': idx.isoformat(),
                'original_data': {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                },
                'features': {col: float(row[col]) for col in ['price_change_pct', 'rsi', 'macd', 'bb_position', 'volume_ratio'] if col in row and not pd.isna(row[col])},
                'targets': {
                    'price_direction': int(row['price_direction']) if 'price_direction' in row and not pd.isna(row['price_direction']) else None,
                    'price_change_target': float(row['price_change_target']) if 'price_change_target' in row and not pd.isna(row['price_change_target']) else None
                }
            })
        
        return jsonify({
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name
            },
            'sample_info': {
                'requested_size': sample_size,
                'actual_size': len(sample_data),
                'total_available': len(df_final)
            },
            'sample_data': sample_data
        }), 200
        
    except Exception as e:
        error_msg = f'获取数据样本失败: {str(e)}'
        print(f"Error in get_processed_sample: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500
