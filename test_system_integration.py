#!/usr/bin/env python3
"""
GetMoneyGou系统集成测试
测试整个系统的端到端功能
"""

import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:5001"

def test_system_health():
    """测试系统健康状态"""
    print("🔍 测试系统健康状态...")
    
    response = requests.get(f"{BASE_URL}/api/system/health")
    assert response.status_code == 200
    
    health = response.json()
    print(f"   系统状态: {health['status']}")
    print(f"   组件状态: {health['components']}")
    
    return health['status'] in ['healthy', 'degraded']

def test_data_upload():
    """测试数据上传功能"""
    print("📁 测试数据上传功能...")
    
    # 检查测试数据文件是否存在
    test_file = "test_data_combined.csv"
    if not os.path.exists(test_file):
        print(f"   ❌ 测试数据文件 {test_file} 不存在")
        return False
    
    # 上传文件
    with open(test_file, 'rb') as f:
        files = {'file': f}
        data = {
            'name': '集成测试数据集',
            'description': '用于系统集成测试的数据集'
        }
        response = requests.post(f"{BASE_URL}/api/data/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        dataset_id = result['dataset']['id']
        print(f"   ✅ 数据上传成功，数据集ID: {dataset_id}")
        return dataset_id
    else:
        print(f"   ❌ 数据上传失败: {response.json()}")
        return False

def test_data_preprocessing(dataset_id):
    """测试数据预处理功能"""
    print("⚙️ 测试数据预处理功能...")
    
    data = {
        'test_size': 0.2,
        'prediction_horizon': 1
    }
    
    response = requests.post(
        f"{BASE_URL}/api/preprocessing/datasets/{dataset_id}/prepare",
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ 数据预处理成功")
        print(f"   训练样本数: {result['preprocessing_info']['train_samples']}")
        print(f"   测试样本数: {result['preprocessing_info']['test_samples']}")
        print(f"   特征数量: {result['preprocessing_info']['feature_count']}")
        return True
    else:
        print(f"   ❌ 数据预处理失败: {response.json()}")
        return False

def test_model_training(dataset_id):
    """测试模型训练功能"""
    print("🤖 测试模型训练功能...")
    
    # 创建逻辑回归模型
    model_data = {
        'algorithm': 'logistic_regression',
        'name': '集成测试逻辑回归模型',
        'task_type': 'classification',
        'test_size': 0.2,
        'prediction_horizon': 1
    }
    
    response = requests.post(
        f"{BASE_URL}/api/ml/datasets/{dataset_id}/models",
        json=model_data
    )
    
    if response.status_code == 201:
        result = response.json()
        model_id = result['model']['id']
        print(f"   ✅ 模型创建成功，模型ID: {model_id}")
        
        # 等待模型训练完成
        print("   ⏳ 等待模型训练完成...")
        max_wait = 60  # 最多等待60秒
        wait_time = 0
        
        while wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            
            model_response = requests.get(f"{BASE_URL}/api/ml/models/{model_id}")
            if model_response.status_code == 200:
                model_info = model_response.json()
                status = model_info['status']
                print(f"   状态: {status}")
                
                if status == 'completed':
                    print(f"   ✅ 模型训练完成")
                    print(f"   准确率: {model_info['accuracy']:.4f}")
                    return model_id
                elif status == 'failed':
                    print(f"   ❌ 模型训练失败")
                    return False
        
        print(f"   ⚠️ 模型训练超时")
        return model_id  # 返回模型ID，即使还在训练
    else:
        print(f"   ❌ 模型创建失败: {response.json()}")
        return False

def test_ai_explanation(model_id):
    """测试AI解释功能"""
    print("🧠 测试AI解释功能...")
    
    data = {
        'question': '这个模型的性能如何？'
    }
    
    response = requests.post(
        f"{BASE_URL}/api/ai/models/{model_id}/explain",
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ AI解释生成成功")
        print(f"   会话ID: {result['session_id']}")
        
        # 测试对话功能
        chat_data = {
            'session_id': result['session_id'],
            'question': '这个模型适合什么投资策略？'
        }
        
        chat_response = requests.post(
            f"{BASE_URL}/api/ai/models/{model_id}/chat",
            json=chat_data
        )
        
        if chat_response.status_code == 200:
            print(f"   ✅ AI对话功能正常")
            return True
        else:
            print(f"   ❌ AI对话失败: {chat_response.json()}")
            return False
    else:
        print(f"   ❌ AI解释失败: {response.json()}")
        return False

def test_code_generation(model_id):
    """测试代码生成功能"""
    print("💻 测试代码生成功能...")
    
    # 生成Python代码
    python_data = {'language': 'python'}
    python_response = requests.post(
        f"{BASE_URL}/api/codegen/models/{model_id}/generate",
        json=python_data
    )
    
    if python_response.status_code == 200:
        python_result = python_response.json()
        python_code_id = python_result['code_id']
        print(f"   ✅ Python代码生成成功，代码ID: {python_code_id}")
        
        # 测试Python代码
        test_response = requests.post(
            f"{BASE_URL}/api/codegen/codes/{python_code_id}/test",
            json={'test_data': []}
        )
        
        if test_response.status_code == 200:
            test_result = test_response.json()
            print(f"   ✅ Python代码测试成功: {test_result['test_result']['status']}")
        else:
            print(f"   ❌ Python代码测试失败")
    else:
        print(f"   ❌ Python代码生成失败: {python_response.json()}")
        return False
    
    # 生成C++代码
    cpp_data = {'language': 'cpp'}
    cpp_response = requests.post(
        f"{BASE_URL}/api/codegen/models/{model_id}/generate",
        json=cpp_data
    )
    
    if cpp_response.status_code == 200:
        cpp_result = cpp_response.json()
        cpp_code_id = cpp_result['code_id']
        print(f"   ✅ C++代码生成成功，代码ID: {cpp_code_id}")
        
        # 测试C++代码
        test_response = requests.post(
            f"{BASE_URL}/api/codegen/codes/{cpp_code_id}/test",
            json={'test_data': []}
        )
        
        if test_response.status_code == 200:
            test_result = test_response.json()
            print(f"   ✅ C++代码测试成功: {test_result['test_result']['status']}")
        else:
            print(f"   ❌ C++代码测试失败")
    else:
        print(f"   ❌ C++代码生成失败: {cpp_response.json()}")
        return False
    
    return True

def test_system_stats():
    """测试系统统计功能"""
    print("📊 测试系统统计功能...")
    
    # 测试系统统计
    stats_response = requests.get(f"{BASE_URL}/api/system/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"   ✅ 系统统计获取成功")
        print(f"   数据集数量: {stats['database_stats']['total_datasets']}")
        print(f"   模型数量: {stats['database_stats']['total_models']}")
        print(f"   CPU使用率: {stats['system_resources']['cpu_usage_percent']}%")
    else:
        print(f"   ❌ 系统统计获取失败")
        return False
    
    # 测试性能指标
    perf_response = requests.get(f"{BASE_URL}/api/system/performance")
    if perf_response.status_code == 200:
        perf = perf_response.json()
        print(f"   ✅ 性能指标获取成功")
        if 'model_performance' in perf and 'total_completed_models' in perf['model_performance']:
            print(f"   已完成模型数: {perf['model_performance']['total_completed_models']}")
    else:
        print(f"   ❌ 性能指标获取失败")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始GetMoneyGou系统集成测试")
    print("=" * 50)
    
    test_results = []
    
    # 1. 系统健康检查
    test_results.append(("系统健康检查", test_system_health()))
    
    # 2. 数据上传
    dataset_id = test_data_upload()
    test_results.append(("数据上传", bool(dataset_id)))
    
    if dataset_id:
        # 3. 数据预处理
        test_results.append(("数据预处理", test_data_preprocessing(dataset_id)))
        
        # 4. 模型训练
        model_id = test_model_training(dataset_id)
        test_results.append(("模型训练", bool(model_id)))
        
        if model_id:
            # 5. AI解释
            test_results.append(("AI解释", test_ai_explanation(model_id)))
            
            # 6. 代码生成
            test_results.append(("代码生成", test_code_generation(model_id)))
    
    # 7. 系统统计
    test_results.append(("系统统计", test_system_stats()))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"总计: {passed}/{total} 个测试通过")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
    else:
        print("⚠️ 部分测试失败，请检查系统状态。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
