"""
数据上传API
处理CSV文件上传、验证和解析
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

from models import db, Dataset, MarketData

data_upload_bp = Blueprint('data_upload', __name__, url_prefix='/data')

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config.get('ALLOWED_EXTENSIONS', {'csv'})

def validate_csv_format(df):
    """验证CSV文件格式"""
    required_columns = [
        'timestamp', 'open', 'high', 'low', 'close', 
        'volume', 'buy_amount', 'sell_amount'
    ]
    
    # 检查必需列是否存在
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"缺少必需的列: {', '.join(missing_columns)}"
    
    # 检查数据类型
    try:
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 转换数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'buy_amount', 'sell_amount']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 检查是否有无效数据
        if df[numeric_columns].isnull().any().any():
            return False, "数据中包含无效的数值"
        
        # 检查价格逻辑
        invalid_price_rows = df[(df['high'] < df['low']) | 
                               (df['high'] < df['open']) | 
                               (df['high'] < df['close']) |
                               (df['low'] > df['open']) | 
                               (df['low'] > df['close'])]
        
        if len(invalid_price_rows) > 0:
            return False, f"发现 {len(invalid_price_rows)} 行价格数据不合理 (high < low 或价格超出范围)"
        
        return True, "数据格式验证通过"
        
    except Exception as e:
        return False, f"数据格式错误: {str(e)}"

@data_upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """上传CSV文件"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型，请上传CSV文件'}), 400
        
        # 获取其他参数
        dataset_name = request.form.get('name', file.filename)
        description = request.form.get('description', '')
        
        # 生成安全的文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 保存文件
        file.save(file_path)
        
        # 读取和验证CSV文件
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            os.remove(file_path)  # 删除无效文件
            return jsonify({'error': f'无法读取CSV文件: {str(e)}'}), 400
        
        # 验证数据格式
        is_valid, message = validate_csv_format(df)
        if not is_valid:
            os.remove(file_path)  # 删除无效文件
            return jsonify({'error': message}), 400
        
        # 创建数据集记录
        dataset = Dataset(
            name=dataset_name,
            filename=filename,
            file_path=file_path,
            total_records=len(df),
            date_range_start=df['timestamp'].min(),
            date_range_end=df['timestamp'].max(),
            description=description
        )
        
        db.session.add(dataset)
        db.session.flush()  # 获取dataset.id
        
        # 批量插入市场数据
        market_data_records = []
        for _, row in df.iterrows():
            market_data = MarketData(
                dataset_id=dataset.id,
                timestamp=row['timestamp'],
                open_price=row['open'],
                high_price=row['high'],
                low_price=row['low'],
                close_price=row['close'],
                volume=row['volume'],
                buy_amount=row['buy_amount'],
                sell_amount=row['sell_amount']
            )
            market_data_records.append(market_data)
        
        # 批量插入（分批处理以避免内存问题）
        batch_size = 1000
        for i in range(0, len(market_data_records), batch_size):
            batch = market_data_records[i:i + batch_size]
            db.session.bulk_save_objects(batch)
        
        db.session.commit()
        
        return jsonify({
            'message': '文件上传成功',
            'dataset': dataset.to_dict(),
            'validation_message': message
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@data_upload_bp.route('/datasets', methods=['GET'])
def get_datasets():
    """获取所有数据集列表"""
    try:
        datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
        return jsonify({
            'datasets': [dataset.to_dict() for dataset in datasets]
        }), 200
    except Exception as e:
        return jsonify({'error': f'获取数据集失败: {str(e)}'}), 500

@data_upload_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """获取特定数据集的详细信息"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取数据统计信息
        market_data_query = MarketData.query.filter_by(dataset_id=dataset_id)
        
        # 基本统计
        total_records = market_data_query.count()
        
        # 价格统计
        price_stats = db.session.query(
            db.func.min(MarketData.close_price).label('min_price'),
            db.func.max(MarketData.close_price).label('max_price'),
            db.func.avg(MarketData.close_price).label('avg_price'),
            db.func.avg(MarketData.volume).label('avg_volume')
        ).filter_by(dataset_id=dataset_id).first()
        
        dataset_info = dataset.to_dict()
        dataset_info.update({
            'statistics': {
                'total_records': total_records,
                'min_price': float(price_stats.min_price) if price_stats.min_price else 0,
                'max_price': float(price_stats.max_price) if price_stats.max_price else 0,
                'avg_price': float(price_stats.avg_price) if price_stats.avg_price else 0,
                'avg_volume': float(price_stats.avg_volume) if price_stats.avg_volume else 0
            }
        })
        
        return jsonify(dataset_info), 200
        
    except Exception as e:
        return jsonify({'error': f'获取数据集详情失败: {str(e)}'}), 500

@data_upload_bp.route('/datasets/<int:dataset_id>/data', methods=['GET'])
def get_dataset_data(dataset_id):
    """获取数据集的市场数据（支持分页）"""
    try:
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)
        per_page = min(per_page, 1000)  # 限制每页最大数量
        
        # 查询数据
        market_data_query = MarketData.query.filter_by(dataset_id=dataset_id)\
                                          .order_by(MarketData.timestamp)
        
        pagination = market_data_query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return jsonify({
            'data': [data.to_dict() for data in pagination.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'获取数据失败: {str(e)}'}), 500

@data_upload_bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """删除数据集"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 删除文件
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # 删除数据库记录（级联删除相关数据）
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({'message': '数据集删除成功'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除数据集失败: {str(e)}'}), 500
