"""
AI客户端 - 支持Azure OpenAI和Deepseek API
"""

import os
import json
import traceback
import requests
from typing import Dict, List, Any, Optional
from config.config import Config


class AIClient:
    """AI客户端，支持多种AI服务提供商"""
    
    def __init__(self):
        """初始化AI客户端"""
        self.config = Config()
        self.provider = self.config.AI_PROVIDER
        
        if self.provider == 'deepseek':
            self._init_deepseek()
        elif self.provider == 'azure':
            self._init_azure()
        else:
            print(f"不支持的AI提供商: {self.provider}")
            self.client = None
    
    def _init_deepseek(self):
        """初始化Deepseek客户端"""
        self.api_key = self.config.DEEPSEEK_API_KEY
        self.base_url = self.config.DEEPSEEK_BASE_URL
        self.model = self.config.DEEPSEEK_MODEL
        
        if not self.api_key:
            print("警告: Deepseek API密钥未配置")
            self.client = None
        else:
            self.client = "deepseek"
            print(f"使用Deepseek API: {self.model}")
    
    def _init_azure(self):
        """初始化Azure OpenAI客户端"""
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage, AIMessage
            
            self.api_key = self.config.AZURE_OPENAI_API_KEY
            self.endpoint = self.config.AZURE_OPENAI_ENDPOINT
            self.deployment_name = self.config.AZURE_OPENAI_DEPLOYMENT_NAME
            self.api_version = self.config.AZURE_OPENAI_API_VERSION
            
            if not self.api_key or not self.endpoint:
                print("警告: Azure OpenAI配置不完整")
                self.client = None
            else:
                self.client = AzureChatOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    deployment_name=self.deployment_name,
                    temperature=0.7,
                    max_tokens=2000
                )
                print(f"使用Azure OpenAI: {self.deployment_name}")
        except ImportError:
            print("Azure OpenAI依赖未安装")
            self.client = None
        except Exception as e:
            print(f"初始化Azure OpenAI客户端失败: {e}")
            self.client = None

    def _call_deepseek_api(self, messages: List[Dict], max_retries: int = 3) -> str:
        """调用Deepseek API（带重试机制）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        for attempt in range(max_retries):
            try:
                print(f"[Deepseek API] 尝试第 {attempt + 1} 次调用...")

                # 根据尝试次数调整超时时间
                timeout = 30 + (attempt * 15)  # 30s, 45s, 60s

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )

                print(f"[Deepseek API] 响应状态码: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"[Deepseek API] 调用成功，返回内容长度: {len(content)}")
                    return content
                else:
                    error_msg = f"Deepseek API错误 {response.status_code}: {response.text}"
                    print(f"[Deepseek API] {error_msg}")

                    # 如果是客户端错误（4xx），不重试
                    if 400 <= response.status_code < 500:
                        return f"API调用失败: {error_msg}"

                    # 服务器错误（5xx），继续重试
                    if attempt == max_retries - 1:
                        return f"API调用失败: {error_msg}"

            except requests.exceptions.Timeout as e:
                print(f"[Deepseek API] 第 {attempt + 1} 次尝试超时: {str(e)}")
                if attempt == max_retries - 1:
                    return f"API调用超时（已重试{max_retries}次），请检查网络连接或稍后再试"

            except requests.exceptions.ConnectionError as e:
                print(f"[Deepseek API] 第 {attempt + 1} 次连接错误: {str(e)}")
                if attempt == max_retries - 1:
                    return f"网络连接失败: 无法连接到Deepseek服务器，请检查网络设置"

            except requests.exceptions.RequestException as e:
                print(f"[Deepseek API] 第 {attempt + 1} 次请求异常: {str(e)}")
                if attempt == max_retries - 1:
                    return f"网络请求失败: {str(e)}"

            except Exception as e:
                print(f"[Deepseek API] 第 {attempt + 1} 次未知异常: {str(e)}")
                if attempt == max_retries - 1:
                    return f"API调用异常: {str(e)}"

            # 等待后重试
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                print(f"[Deepseek API] 等待 {wait_time} 秒后重试...")
                import time
                time.sleep(wait_time)

        return "API调用失败: 已达到最大重试次数"

    def _call_azure_api(self, messages: List) -> str:
        """调用Azure OpenAI API"""
        try:
            response = self.client(messages)
            return response.content
        except Exception as e:
            error_str = str(e)
            print(f"Azure OpenAI API调用失败: {e}")
            
            if "403" in error_str or "Forbidden" in error_str:
                return f"API访问被拒绝 (403 Forbidden): {error_str}\n\n请检查您的API密钥权限或配额限制。"
            elif "401" in error_str or "Unauthorized" in error_str:
                return f"API认证失败 (401 Unauthorized): {error_str}\n\n请检查您的API密钥是否正确。"
            elif "429" in error_str or "rate limit" in error_str.lower():
                return f"API请求频率限制 (429): {error_str}\n\n请稍后再试。"
            else:
                return f"API调用失败: {error_str}"

    def create_model_explanation_prompt(self, model_info: Dict, feature_importance: Dict = None, user_question: str = None) -> tuple:
        """创建模型解释的提示词"""
        
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
        """解释模型的预测逻辑"""
        try:
            if not self.client:
                return "AI解释服务未配置，请检查API设置。"
            
            system_prompt, human_prompt = self.create_model_explanation_prompt(model_info, feature_importance, user_question)
            
            if self.provider == 'deepseek':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ]
                return self._call_deepseek_api(messages)
            
            elif self.provider == 'azure':
                from langchain.schema import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]
                return self._call_azure_api(messages)
                
        except Exception as e:
            error_msg = f"模型解释失败: {str(e)}"
            print(f"Error in explain_model: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"模型解释服务不可用: {error_msg}"

    def continue_conversation(self, model_info: Dict, conversation_history: List[Dict], 
                            new_question: str) -> str:
        """继续对话"""
        try:
            if not self.client:
                return "AI对话服务未配置，请检查API设置。"
            
            system_prompt, _ = self.create_model_explanation_prompt(model_info)
            
            if self.provider == 'deepseek':
                messages = [{"role": "system", "content": system_prompt}]
                
                for msg in conversation_history:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
                
                messages.append({"role": "user", "content": new_question})
                return self._call_deepseek_api(messages)
            
            elif self.provider == 'azure':
                from langchain.schema import SystemMessage, HumanMessage, AIMessage
                messages = [SystemMessage(content=system_prompt)]
                
                for msg in conversation_history:
                    if msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        messages.append(AIMessage(content=msg['content']))
                
                messages.append(HumanMessage(content=new_question))
                return self._call_azure_api(messages)
                    
        except Exception as e:
            error_msg = f"对话失败: {str(e)}"
            print(f"Error in continue_conversation: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"对话服务不可用: {error_msg}"

    def generate_market_insights(self, data_summary: Dict) -> str:
        """生成市场洞察"""
        try:
            if not self.client:
                return "市场洞察服务未配置，请检查API设置。"
            
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

            if self.provider == 'deepseek':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ]
                return self._call_deepseek_api(messages)
            
            elif self.provider == 'azure':
                from langchain.schema import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]
                return self._call_azure_api(messages)
            
        except Exception as e:
            error_msg = f"市场洞察生成失败: {str(e)}"
            print(f"Error in generate_market_insights: {error_msg}")
            return f"市场洞察服务不可用: {error_msg}"
    
    def test_connection(self) -> Dict[str, Any]:
        """测试AI连接"""
        try:
            if not self.client:
                return {
                    'success': False,
                    'message': 'AI客户端未初始化',
                    'error': '请检查API配置'
                }

            if self.provider == 'deepseek':
                print(f"[测试] 开始测试Deepseek连接...")
                print(f"[测试] API密钥: {self.api_key[:10]}...")
                print(f"[测试] 基础URL: {self.base_url}")
                print(f"[测试] 模型: {self.model}")

                messages = [{"role": "user", "content": "请回复'连接测试成功'"}]
                response = self._call_deepseek_api(messages)

                print(f"[测试] 响应内容: {response}")

                if "API调用失败" in response or "网络请求失败" in response or "API调用超时" in response:
                    return {
                        'success': False,
                        'message': f'Deepseek连接测试失败',
                        'error': response
                    }
                else:
                    return {
                        'success': True,
                        'message': 'Deepseek连接测试成功',
                        'response': response
                    }
            
            elif self.provider == 'azure':
                from langchain.schema import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content="你是一个AI助手。"),
                    HumanMessage(content="请回复'连接测试成功'")
                ]
                
                response = self._call_azure_api(messages)
                
                if "API调用失败" in response:
                    return {
                        'success': False,
                        'message': f'Azure OpenAI连接测试失败',
                        'error': response
                    }
                else:
                    return {
                        'success': True,
                        'message': 'Azure OpenAI连接测试成功',
                        'response': response
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
_ai_client = None

def get_ai_client():
    """获取AI客户端实例"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client

def test_ai_connection():
    """测试AI连接"""
    client = get_ai_client()
    return client.test_connection()

# 为了兼容性，保留原有的函数名
def get_azure_openai_client():
    """获取AI客户端实例（兼容性函数）"""
    return get_ai_client()

def test_azure_openai_connection():
    """测试AI连接（兼容性函数）"""
    return test_ai_connection()
