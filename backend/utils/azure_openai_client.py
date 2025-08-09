"""
Azure OpenAI客户端 - 清理版本
"""

import os
import json
import traceback
from typing import Dict, List, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from config.config import Config


class AzureOpenAIClient:
    """Azure OpenAI客户端"""
    
    def __init__(self):
        """初始化Azure OpenAI客户端"""
        config = Config()
        self.api_key = config.AZURE_OPENAI_API_KEY
        self.endpoint = config.AZURE_OPENAI_ENDPOINT
        self.deployment_name = config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.api_version = config.AZURE_OPENAI_API_VERSION
        
        if not self.api_key or not self.endpoint:
            print("警告: Azure OpenAI配置不完整，将无法使用AI功能")
            self.langchain_client = None
        else:
            try:
                self.langchain_client = AzureChatOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    deployment_name=self.deployment_name,
                    temperature=0.7,
                    max_tokens=2000
                )
            except Exception as e:
                print(f"初始化Azure OpenAI客户端失败: {e}")
                self.langchain_client = None

    def create_model_explanation_prompt(self, model_info: Dict, feature_importance: Dict = None, user_question: str = None) -> str:
        """创建模型解释的提示词，专注于预测逻辑和特征重要性"""
        
        # 基础模型信息
        algorithm_names = {
            'logistic_regression': '逻辑回归',
            'random_forest': '随机森林',
            'svm': '支持向量机',
            'gradient_boosting': '梯度提升',
            'linear_regression': '线性回归'
        }
        
        algorithm_name = algorithm_names.get(model_info.get('algorithm', ''), model_info.get('algorithm', ''))
        
        system_prompt = f"""你是一个专业的机器学习模型解释专家，专门分析模型的预测逻辑和决策依据。

你的任务是：
1. 解释模型如何做出预测决策
2. 分析哪些特征对预测结果影响最大
3. 说明特征变化如何影响预测结果
4. 提供基于特征重要性的实用建议
5. 用通俗易懂的语言解释技术概念

模型信息：
- 算法类型：{algorithm_name}
- 模型名称：{model_info.get('name', '未知')}
- 训练状态：{model_info.get('status', '未知')}
"""

        if model_info.get('status') == 'completed':
            metrics_info = ""
            if model_info.get('accuracy'):
                metrics_info += f"- 准确率：{model_info['accuracy']*100:.1f}%\n"
            if model_info.get('precision'):
                metrics_info += f"- 精确率：{model_info['precision']*100:.1f}%\n"
            if model_info.get('recall'):
                metrics_info += f"- 召回率：{model_info['recall']*100:.1f}%\n"
            if model_info.get('f1_score'):
                metrics_info += f"- F1分数：{model_info['f1_score']*100:.1f}%\n"
            
            if metrics_info:
                system_prompt += f"\n性能指标：\n{metrics_info}"
        
        # 添加特征重要性信息
        if feature_importance:
            system_prompt += f"\n特征重要性分析：\n"
            # 获取前10个最重要的特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_features, 1):
                system_prompt += f"{i}. {feature}: {importance:.4f}\n"
        
        # 如果有用户问题，添加到提示词中
        if user_question:
            human_prompt = f"用户问题：{user_question}\n\n请基于模型的预测逻辑和特征重要性来回答用户的问题。"
        else:
            human_prompt = """请详细解释这个数据预测模型的逻辑，包括：

1. **预测决策机制**：模型是如何做出预测的？
2. **关键特征分析**：哪些数据特征对预测结果影响最大？为什么？
3. **特征影响方式**：这些重要特征是如何影响预测结果的？
4. **决策边界**：什么样的特征组合会导致不同的预测结果？
5. **实用建议**：基于特征重要性，在实际应用中应该重点关注哪些数据指标？

请用通俗易懂的语言解释，避免过于技术性的术语。重点说明模型的"思考过程"。"""
        
        return system_prompt, human_prompt

    def explain_model(self, model_info: Dict, feature_importance: Dict = None, user_question: str = None) -> str:
        """使用LangChain解释模型的预测逻辑"""
        try:
            if not self.langchain_client:
                return "AI解释服务未配置，请检查Azure OpenAI设置。"
            
            # 尝试调用真实的Azure OpenAI API
            system_prompt, human_prompt = self.create_model_explanation_prompt(model_info, feature_importance, user_question)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            try:
                response = self.langchain_client(messages)
                return response.content
            except Exception as api_error:
                error_str = str(api_error)
                print(f"Azure OpenAI API调用失败: {api_error}")
                
                # 检查是否是403错误
                if "403" in error_str or "Forbidden" in error_str:
                    return f"API访问被拒绝 (403 Forbidden): {error_str}\n\n请检查您的API密钥权限或配额限制。"
                elif "401" in error_str or "Unauthorized" in error_str:
                    return f"API认证失败 (401 Unauthorized): {error_str}\n\n请检查您的API密钥是否正确。"
                elif "429" in error_str or "rate limit" in error_str.lower():
                    return f"API请求频率限制 (429): {error_str}\n\n请稍后再试。"
                else:
                    return f"API调用失败: {error_str}\n\n请检查网络连接和API配置。"
                    
        except Exception as e:
            error_msg = f"模型解释失败: {str(e)}"
            print(f"Error in explain_model: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"模型解释服务不可用: {error_msg}"

    def continue_conversation(self, model_info: Dict, conversation_history: List[Dict], 
                            new_question: str) -> str:
        """继续对话"""
        try:
            if not self.langchain_client:
                return "AI对话服务未配置，请检查Azure OpenAI设置。"
            
            # 尝试调用真实的Azure OpenAI API
            messages = []
            system_prompt, _ = self.create_model_explanation_prompt(model_info)
            messages.append(SystemMessage(content=system_prompt))
            
            for msg in conversation_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            messages.append(HumanMessage(content=new_question))
            
            try:
                response = self.langchain_client(messages)
                return response.content
            except Exception as api_error:
                error_str = str(api_error)
                print(f"Azure OpenAI API调用失败: {api_error}")
                
                # 检查是否是403错误
                if "403" in error_str or "Forbidden" in error_str:
                    return f"API访问被拒绝 (403 Forbidden): {error_str}"
                elif "401" in error_str or "Unauthorized" in error_str:
                    return f"API认证失败 (401 Unauthorized): {error_str}"
                elif "429" in error_str or "rate limit" in error_str.lower():
                    return f"API请求频率限制 (429): {error_str}"
                else:
                    return f"API调用失败: {error_str}"
                    
        except Exception as e:
            error_msg = f"对话失败: {str(e)}"
            print(f"Error in continue_conversation: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"对话服务不可用: {error_msg}"

    def generate_market_insights(self, data_summary: Dict) -> str:
        """生成市场洞察"""
        try:
            if not self.langchain_client:
                return "市场洞察服务未配置，请检查Azure OpenAI设置。"
            
            system_prompt = """你是一个资深的数据分析师。基于提供的数据统计信息，生成专业的数据分析报告。

请包含以下内容：
1. 数据概览和关键统计
2. 趋势分析
3. 数据波动特征
4. 交易量分析
5. 风险评估
6. 应用建议

请使用专业但易懂的语言，避免过于技术性的术语。"""

            human_prompt = f"""请分析以下数据：

数据统计信息：
{json.dumps(data_summary, indent=2, ensure_ascii=False)}

请生成一份专业的数据分析报告。"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            try:
                response = self.langchain_client(messages)
                return response.content
            except Exception as api_error:
                error_str = str(api_error)
                print(f"Azure OpenAI API调用失败: {api_error}")
                
                if "403" in error_str or "Forbidden" in error_str:
                    return f"API访问被拒绝 (403 Forbidden): {error_str}"
                elif "401" in error_str or "Unauthorized" in error_str:
                    return f"API认证失败 (401 Unauthorized): {error_str}"
                elif "429" in error_str or "rate limit" in error_str.lower():
                    return f"API请求频率限制 (429): {error_str}"
                else:
                    return f"API调用失败: {error_str}"
            
        except Exception as e:
            error_msg = f"市场洞察生成失败: {str(e)}"
            print(f"Error in generate_market_insights: {error_msg}")
            return f"市场洞察服务不可用: {error_msg}"
    
    def test_connection(self) -> Dict[str, Any]:
        """测试Azure OpenAI连接"""
        try:
            if not self.langchain_client:
                return {
                    'success': False,
                    'message': 'Azure OpenAI客户端未初始化',
                    'error': '请检查API配置'
                }
            
            # 尝试真实的连接测试
            messages = [
                SystemMessage(content="你是一个AI助手。"),
                HumanMessage(content="请回复'连接测试成功'")
            ]
            
            try:
                response = self.langchain_client(messages)
                return {
                    'success': True,
                    'message': '连接测试成功',
                    'response': response.content
                }
            except Exception as api_error:
                error_str = str(api_error)
                print(f"Azure OpenAI连接测试失败: {api_error}")
                return {
                    'success': False,
                    'message': f'连接测试失败: {error_str}',
                    'error': error_str
                }
                
        except Exception as e:
            error_msg = f"连接测试失败: {str(e)}"
            print(f"Error in test_connection: {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'error': str(e)
            }


# 全局客户端实例
_azure_openai_client = None

def get_azure_openai_client():
    """获取Azure OpenAI客户端实例"""
    global _azure_openai_client
    if _azure_openai_client is None:
        _azure_openai_client = AzureOpenAIClient()
    return _azure_openai_client

def test_azure_openai_connection():
    """测试Azure OpenAI连接"""
    client = get_azure_openai_client()
    return client.test_connection()
