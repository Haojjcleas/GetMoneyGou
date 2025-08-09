"""
AI功能测试API
"""

from flask import Blueprint, request, jsonify
import traceback

from utils.ai_client import get_ai_client, test_ai_connection

test_ai_bp = Blueprint('test_ai', __name__)

@test_ai_bp.route('/test-connection', methods=['GET'])
def test_connection():
    """测试AI连接"""
    try:
        result = test_ai_connection()
        return jsonify(result)
    except Exception as e:
        error_msg = f"连接测试失败: {str(e)}"
        print(f"Error in test_connection: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': error_msg,
            'error': str(e)
        }), 500

@test_ai_bp.route('/test-chat', methods=['POST'])
def test_chat():
    """测试AI对话"""
    try:
        data = request.get_json()
        message = data.get('message', '你好，请回复测试成功')
        
        client = get_ai_client()
        
        if client.provider == 'deepseek':
            messages = [{"role": "user", "content": message}]
            response = client._call_deepseek_api(messages)
        else:
            response = "当前不是Deepseek提供商"
        
        return jsonify({
            'success': True,
            'provider': client.provider,
            'message': message,
            'response': response
        })
        
    except Exception as e:
        error_msg = f"对话测试失败: {str(e)}"
        print(f"Error in test_chat: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': error_msg,
            'error': str(e)
        }), 500

@test_ai_bp.route('/test-explain', methods=['POST'])
def test_explain():
    """测试模型解释"""
    try:
        data = request.get_json()
        question = data.get('question', '这个模型的准确率如何？')
        
        client = get_ai_client()
        
        # 模拟模型信息
        model_info = {
            'name': '测试模型',
            'algorithm': 'random_forest',
            'status': 'completed',
            'accuracy': 0.85
        }
        
        response = client.explain_model(model_info, user_question=question)
        
        return jsonify({
            'success': True,
            'provider': client.provider,
            'question': question,
            'explanation': response
        })
        
    except Exception as e:
        error_msg = f"解释测试失败: {str(e)}"
        print(f"Error in test_explain: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': error_msg,
            'error': str(e)
        }), 500
