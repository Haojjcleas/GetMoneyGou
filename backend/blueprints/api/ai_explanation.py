"""
AI解释API
提供模型解释和对话功能
"""
import os

from flask import Blueprint, request, jsonify
import traceback
import uuid
from datetime import datetime

from utils.ai_client import get_ai_client, test_azure_openai_connection
from models import db, MLModel, ModelExplanation

ai_explanation_bp = Blueprint('ai_explanation', __name__, url_prefix='/ai')

@ai_explanation_bp.route('/test', methods=['GET'])
def test_connection():
    """测试Azure OpenAI连接"""
    try:
        result = test_azure_openai_connection()
        return jsonify(result), 200 if result['success'] else 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'测试失败: {str(e)}',
            'error': str(e)
        }), 500

@ai_explanation_bp.route('/models/<int:model_id>/explain', methods=['POST'])
def explain_model(model_id):
    """解释模型"""
    try:
        # 获取模型信息
        model = MLModel.query.get_or_404(model_id)
        
        if model.status != 'completed':
            return jsonify({'error': '模型尚未训练完成，无法进行解释'}), 400
        
        # 获取用户问题（可选）
        data = request.get_json() or {}
        user_question = data.get('question')
        
        # 获取AI客户端
        client = get_ai_client()
        
        # 准备模型信息
        model_info = model.to_dict()

        # 获取特征重要性（如果模型有保存的话）
        feature_importance = None
        if model.model_file_path and os.path.exists(model.model_file_path):
            try:
                from utils.ml_models import MLModelManager
                model_manager = MLModelManager()
                trained_model = model_manager.load_model(model.model_file_path)
                feature_importance_data = model_manager._get_feature_importance(trained_model)
                if feature_importance_data and 'importances' in feature_importance_data:
                    # 创建特征名称到重要性的映射
                    feature_names = [
                        'price_change_pct', 'rsi', 'macd', 'volume_ratio', 'bb_position',
                        'ma_5_ratio', 'buy_sell_ratio', 'close_lag_1', 'ma_10_ratio', 'volume_lag_1',
                        'ma_20_ratio', 'atr', 'stoch_k', 'stoch_d', 'williams_r',
                        'cci', 'adx', 'momentum', 'roc', 'trix'
                    ]
                    importances = feature_importance_data['importances']
                    if len(importances) >= len(feature_names):
                        feature_importance = dict(zip(feature_names, importances[:len(feature_names)]))
            except Exception as e:
                print(f"获取特征重要性失败: {e}")

        # 生成解释
        explanation = client.explain_model(model_info, feature_importance, user_question)
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 保存解释记录
        explanation_record = ModelExplanation(
            model_id=model_id,
            session_id=session_id,
            user_question=user_question or "请解释这个模型",
            ai_response=explanation,
            created_time=datetime.utcnow()
        )
        
        db.session.add(explanation_record)
        db.session.commit()
        
        return jsonify({
            'explanation': explanation,
            'session_id': session_id,
            'model_info': {
                'id': model.id,
                'name': model.name,
                'algorithm': model.algorithm,
                'status': model.status
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        error_msg = f'模型解释失败: {str(e)}'
        print(f"Error in explain_model: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@ai_explanation_bp.route('/models/<int:model_id>/chat', methods=['POST'])
def chat_with_model(model_id):
    """与模型进行对话"""
    try:
        # 获取模型信息
        model = MLModel.query.get_or_404(model_id)
        
        if model.status != 'completed':
            return jsonify({'error': '模型尚未训练完成，无法进行对话'}), 400
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '缺少请求数据'}), 400
        
        session_id = data.get('session_id')
        new_question = data.get('question')
        
        if not session_id or not new_question:
            return jsonify({'error': '缺少会话ID或问题'}), 400
        
        # 获取对话历史
        conversation_history = ModelExplanation.query.filter_by(
            model_id=model_id,
            session_id=session_id
        ).order_by(ModelExplanation.created_time).all()
        
        # 构建对话历史格式
        history = []
        for record in conversation_history:
            history.append({
                'role': 'user',
                'content': record.user_question
            })
            history.append({
                'role': 'assistant',
                'content': record.ai_response
            })
        
        # 获取Azure OpenAI客户端
        client = get_azure_openai_client()
        
        # 准备模型信息
        model_info = model.to_dict()
        
        # 继续对话
        response = client.continue_conversation(model_info, history, new_question)
        
        # 保存新的对话记录
        explanation_record = ModelExplanation(
            model_id=model_id,
            session_id=session_id,
            user_question=new_question,
            ai_response=response,
            created_time=datetime.utcnow()
        )
        
        db.session.add(explanation_record)
        db.session.commit()
        
        return jsonify({
            'response': response,
            'session_id': session_id
        }), 200
        
    except Exception as e:
        db.session.rollback()
        error_msg = f'对话失败: {str(e)}'
        print(f"Error in chat_with_model: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@ai_explanation_bp.route('/models/<int:model_id>/conversations', methods=['GET'])
def get_conversations(model_id):
    """获取模型的对话历史"""
    try:
        # 验证模型是否存在
        model = MLModel.query.get_or_404(model_id)
        
        # 获取会话ID参数
        session_id = request.args.get('session_id')
        
        if session_id:
            # 获取特定会话的对话历史
            conversations = ModelExplanation.query.filter_by(
                model_id=model_id,
                session_id=session_id
            ).order_by(ModelExplanation.created_time).all()
        else:
            # 获取所有对话，按会话分组
            conversations = ModelExplanation.query.filter_by(
                model_id=model_id
            ).order_by(ModelExplanation.created_time.desc()).limit(50).all()
        
        # 格式化对话数据
        if session_id:
            # 返回特定会话的完整对话
            conversation_data = []
            for conv in conversations:
                conversation_data.append({
                    'id': conv.id,
                    'user_question': conv.user_question,
                    'ai_response': conv.ai_response,
                    'created_time': conv.created_time.isoformat()
                })
            
            return jsonify({
                'session_id': session_id,
                'conversations': conversation_data
            }), 200
        else:
            # 返回会话列表
            sessions = {}
            for conv in conversations:
                if conv.session_id not in sessions:
                    sessions[conv.session_id] = {
                        'session_id': conv.session_id,
                        'first_question': conv.user_question,
                        'last_activity': conv.created_time.isoformat(),
                        'message_count': 0
                    }
                sessions[conv.session_id]['message_count'] += 1
            
            return jsonify({
                'model_id': model_id,
                'sessions': list(sessions.values())
            }), 200
        
    except Exception as e:
        error_msg = f'获取对话历史失败: {str(e)}'
        print(f"Error in get_conversations: {error_msg}")
        return jsonify({'error': error_msg}), 500

@ai_explanation_bp.route('/datasets/<int:dataset_id>/insights', methods=['POST'])
def generate_market_insights(dataset_id):
    """生成市场洞察"""
    try:
        from models import Dataset, MarketData
        
        # 验证数据集是否存在
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # 获取数据统计信息
        market_data_query = MarketData.query.filter_by(dataset_id=dataset_id)
        
        # 基本统计
        total_records = market_data_query.count()
        
        if total_records == 0:
            return jsonify({'error': '数据集为空，无法生成洞察'}), 400
        
        # 价格统计
        price_stats = db.session.query(
            db.func.min(MarketData.close_price).label('min_price'),
            db.func.max(MarketData.close_price).label('max_price'),
            db.func.avg(MarketData.close_price).label('avg_price'),
            db.func.avg(MarketData.volume).label('avg_volume'),
            db.func.avg(MarketData.buy_amount).label('avg_buy_amount'),
            db.func.avg(MarketData.sell_amount).label('avg_sell_amount')
        ).filter_by(dataset_id=dataset_id).first()
        
        # 构建数据摘要
        data_summary = {
            'dataset_name': dataset.name,
            'total_records': total_records,
            'date_range': {
                'start': dataset.date_range_start.isoformat() if dataset.date_range_start else None,
                'end': dataset.date_range_end.isoformat() if dataset.date_range_end else None
            },
            'price_statistics': {
                'min_price': float(price_stats.min_price) if price_stats.min_price else 0,
                'max_price': float(price_stats.max_price) if price_stats.max_price else 0,
                'avg_price': float(price_stats.avg_price) if price_stats.avg_price else 0,
                'price_volatility': float(price_stats.max_price - price_stats.min_price) if price_stats.max_price and price_stats.min_price else 0
            },
            'trading_statistics': {
                'avg_volume': float(price_stats.avg_volume) if price_stats.avg_volume else 0,
                'avg_buy_amount': float(price_stats.avg_buy_amount) if price_stats.avg_buy_amount else 0,
                'avg_sell_amount': float(price_stats.avg_sell_amount) if price_stats.avg_sell_amount else 0
            }
        }
        
        # 获取Azure OpenAI客户端
        client = get_azure_openai_client()
        
        # 生成市场洞察
        insights = client.generate_market_insights(data_summary)
        
        return jsonify({
            'insights': insights,
            'data_summary': data_summary,
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name
            }
        }), 200
        
    except Exception as e:
        error_msg = f'生成市场洞察失败: {str(e)}'
        print(f"Error in generate_market_insights: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@ai_explanation_bp.route('/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """删除对话记录"""
    try:
        conversation = ModelExplanation.query.get_or_404(conversation_id)
        
        db.session.delete(conversation)
        db.session.commit()
        
        return jsonify({'message': '对话记录删除成功'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除对话记录失败: {str(e)}'}), 500
