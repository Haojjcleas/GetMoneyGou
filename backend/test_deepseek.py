#!/usr/bin/env python3
"""
Deepseek API连接测试脚本
"""

import os
import sys
import requests
import time
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ai_client import get_ai_client

def test_basic_connection():
    """测试基本网络连接"""
    print("=" * 50)
    print("1. 测试基本网络连接")
    print("=" * 50)
    
    try:
        # 测试DNS解析
        import socket
        ip = socket.gethostbyname('api.deepseek.com')
        print(f"✅ DNS解析成功: api.deepseek.com -> {ip}")
        
        # 测试基本HTTP连接
        response = requests.get('https://api.deepseek.com', timeout=10)
        print(f"✅ 基本连接成功，状态码: {response.status_code}")
        
    except socket.gaierror as e:
        print(f"❌ DNS解析失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False
    
    return True

def test_api_key():
    """测试API密钥"""
    print("\n" + "=" * 50)
    print("2. 测试API密钥")
    print("=" * 50)
    
    api_key = ""
    
    if not api_key or api_key == "your_deepseek_api_key":
        print("❌ API密钥未设置")
        return False
    
    print(f"✅ API密钥已设置: {api_key[:10]}...{api_key[-4:]}")
    return True

def test_direct_api_call():
    """直接测试API调用"""
    print("\n" + "=" * 50)
    print("3. 直接测试API调用")
    print("=" * 50)
    
    api_key = ""
    base_url = "https://api.deepseek.com"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "请回复'测试成功'"}
        ],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        print("发送请求...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60  # 增加超时时间
        )
        
        end_time = time.time()
        print(f"请求耗时: {end_time - start_time:.2f}秒")
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ API调用成功!")
            print(f"响应内容: {content}")
            return True
        else:
            print(f"❌ API调用失败")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_ai_client():
    """测试AI客户端"""
    print("\n" + "=" * 50)
    print("4. 测试AI客户端")
    print("=" * 50)
    
    try:
        client = get_ai_client()
        print(f"AI客户端提供商: {client.provider}")
        print(f"API密钥: {client.api_key[:10]}...")
        
        result = client.test_connection()
        
        if result['success']:
            print("✅ AI客户端测试成功!")
            print(f"响应: {result['response']}")
            return True
        else:
            print("❌ AI客户端测试失败")
            print(f"错误: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ AI客户端测试异常: {e}")
        return False

def test_model_explanation():
    """测试模型解释功能"""
    print("\n" + "=" * 50)
    print("5. 测试模型解释功能")
    print("=" * 50)
    
    try:
        client = get_ai_client()
        
        # 模拟模型信息
        model_info = {
            'name': '测试模型',
            'algorithm': 'random_forest',
            'status': 'completed',
            'accuracy': 0.85
        }
        
        print("开始模型解释测试...")
        response = client.explain_model(model_info, user_question="这个模型的准确率如何？")
        
        if "API调用失败" not in response and "网络请求失败" not in response:
            print("✅ 模型解释功能测试成功!")
            print(f"解释内容长度: {len(response)} 字符")
            print(f"解释内容预览: {response[:200]}...")
            return True
        else:
            print("❌ 模型解释功能测试失败")
            print(f"错误: {response}")
            return False
            
    except Exception as e:
        print(f"❌ 模型解释功能测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始Deepseek API连接诊断")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("基本网络连接", test_basic_connection),
        ("API密钥检查", test_api_key),
        ("直接API调用", test_direct_api_call),
        ("AI客户端测试", test_ai_client),
        ("模型解释功能", test_model_explanation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Deepseek API连接正常。")
    else:
        print("⚠️  部分测试失败，请检查网络连接和API配置。")

if __name__ == "__main__":
    main()
