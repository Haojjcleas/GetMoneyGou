"""
代码生成器
基于训练好的模型生成等效的Python和C++代码
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
import joblib

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self):
        self.algorithm_templates = {
            'logistic_regression': {
                'python': self._generate_logistic_regression_python,
                'cpp': self._generate_logistic_regression_cpp
            },
            'random_forest': {
                'python': self._generate_random_forest_python,
                'cpp': self._generate_random_forest_cpp
            },
            'svm': {
                'python': self._generate_svm_python,
                'cpp': self._generate_svm_cpp
            },
            'gradient_boosting': {
                'python': self._generate_gradient_boosting_python,
                'cpp': self._generate_gradient_boosting_cpp
            }
        }
    
    def generate_code(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None, language: str = 'python') -> str:
        """基于AI解释生成代码"""
        algorithm = model_info.get('algorithm')

        if algorithm not in self.algorithm_templates:
            raise ValueError(f"不支持的算法: {algorithm}")

        if language not in ['python', 'cpp']:
            raise ValueError(f"不支持的语言: {language}")

        generator_func = self.algorithm_templates[algorithm][language]
        return generator_func(model_info, feature_importance, ai_explanation)
    
    def _generate_logistic_regression_python(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """基于AI解释生成逻辑回归Python代码"""
        model_name = model_info.get('name', 'LogisticRegressionModel')
        accuracy = model_info.get('accuracy', 0.5)

        # 从AI解释中提取关键特征
        key_features = []
        if feature_importance:
            # 获取前5个最重要的特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            key_features = [f[0] for f in sorted_features]

        # 生成特征重要性注释
        feature_comments = ""
        if feature_importance:
            feature_comments = "\n# 基于AI分析的关键特征重要性:\n"
            for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                feature_comments += f"# {i}. {feature}: {importance:.1%} - "
                if feature == 'price_change_pct':
                    feature_comments += "价格变化百分比，最重要的动量指标\n"
                elif feature == 'rsi':
                    feature_comments += "RSI相对强弱指标，判断超买超卖\n"
                elif feature == 'macd':
                    feature_comments += "MACD指标，反映趋势变化\n"
                elif feature == 'volume_ratio':
                    feature_comments += "成交量比率，衡量市场活跃度\n"
                else:
                    feature_comments += "重要的技术指标特征\n"
        
        code = f'''#!/usr/bin/env python3
"""
{model_name} - 基于AI解释的逻辑回归金融预测模型
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
模型准确率: {accuracy*100:.2f}%

AI模型解释摘要:
- 该模型通过分析技术指标特征来预测价格走势
- 关键决策特征: {', '.join(key_features[:3]) if key_features else '价格动量、RSI、MACD'}
- 预测逻辑: 基于特征组合的历史模式识别
{feature_comments}
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinancialPredictor:
    """金融市场预测器"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def create_features(self, df):
        """创建特征"""
        df_features = df.copy()
        
        # 基础价格特征
        df_features['price_change'] = df_features['close'] - df_features['open']
        df_features['price_change_pct'] = df_features['price_change'] / df_features['open']
        df_features['price_range'] = df_features['high'] - df_features['low']
        
        # 移动平均线
        for window in [5, 10, 20]:
            df_features[f'ma_{{window}}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ma_{{window}}_ratio'] = df_features['close'] / df_features[f'ma_{{window}}']
        
        # 成交量特征
        df_features['volume_ma_5'] = df_features['volume'].rolling(window=5).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_5']
        
        # 买卖比例
        df_features['buy_sell_ratio'] = df_features['buy_amount'] / (df_features['sell_amount'] + 1e-8)
        df_features['buy_pressure'] = df_features['buy_amount'] / (df_features['buy_amount'] + df_features['sell_amount'])
        
        # 滞后特征
        for lag in [1, 2, 3]:
            df_features[f'close_lag_{{lag}}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{{lag}}'] = df_features['volume'].shift(lag)
        
        return df_features
    
    def create_target(self, df, prediction_horizon=1):
        """创建目标变量"""
        future_close = df['close'].shift(-prediction_horizon)
        price_change = future_close - df['close']
        return (price_change > 0).astype(int)
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
        # 创建特征
        df_features = self.create_features(df)
        
        # 创建目标变量
        target = self.create_target(df_features)
        
        # 选择特征列
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'buy_amount', 'sell_amount']
        feature_columns = [col for col in df_features.columns 
                          if col not in exclude_columns and df_features[col].dtype in ['int64', 'float64']]
        
        self.feature_columns = feature_columns
        
        # 移除NaN行
        df_final = df_features[feature_columns + ['close']].copy()
        df_final['target'] = target
        df_final = df_final.dropna()
        
        # 分割数据
        split_index = int(len(df_final) * (1 - test_size))
        
        X_train = df_final[feature_columns].iloc[:split_index]
        X_test = df_final[feature_columns].iloc[split_index:]
        y_train = df_final['target'].iloc[:split_index]
        y_test = df_final['target'].iloc[split_index:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, df, test_size=0.2):
        """训练模型"""
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        
        # 特征缩放
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # 评估模型
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"训练集准确率: {{train_accuracy:.4f}}")
        print(f"测试集准确率: {{test_accuracy:.4f}}")
        print("\\n分类报告:")
        print(classification_report(y_test, test_pred))
        
        return {{
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': dict(zip(self.feature_columns, np.abs(self.model.coef_[0])))
        }}
    
    def predict(self, df):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 创建特征
        df_features = self.create_features(df)
        
        # 选择特征
        X = df_features[self.feature_columns].fillna(0)
        
        # 特征缩放
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return {{
            'predictions': predictions,
            'probabilities': probabilities,
            'prediction_labels': ['下跌', '上涨']
        }}
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {{
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }}
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {{filepath}}")
    
    def load_model(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"模型已从 {{filepath}} 加载")

# 使用示例
if __name__ == "__main__":
    # 创建预测器
    predictor = FinancialPredictor()
    
    # 加载数据 (请替换为您的数据文件路径)
    # df = pd.read_csv('your_data.csv')
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df = df.set_index('timestamp')
    
    # 训练模型
    # results = predictor.train(df)
    
    # 进行预测
    # predictions = predictor.predict(df.tail(10))
    # print("预测结果:", predictions)
    
    # 保存模型
    # predictor.save_model('financial_model.joblib')
    
    print("金融预测模型代码生成完成！")
    print("请根据您的数据格式调整代码中的数据加载部分。")
'''
        
        return code
    
    def _generate_logistic_regression_cpp(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """基于AI解释生成逻辑回归C++代码"""
        model_name = model_info.get('name', 'LogisticRegressionModel')
        accuracy = model_info.get('accuracy', 0.5)

        # 从AI解释中提取关键特征
        key_features = []
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            key_features = [f[0] for f in sorted_features]

        code = f'''/*
{model_name} - 基于AI解释的逻辑回归金融预测模型 (C++)
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
模型准确率: {accuracy*100:.2f}%

AI模型解释摘要:
- 该模型通过分析技术指标特征来预测价格走势
- 关键决策特征: {', '.join(key_features[:3]) if key_features else '价格动量、RSI、MACD'}
- 预测逻辑: 基于特征组合的历史模式识别

编译命令: g++ -std=c++17 -O2 financial_predictor.cpp -o financial_predictor
*/

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <map>

class FinancialPredictor {{
private:
    std::vector<double> weights;
    double bias;
    std::vector<std::string> feature_names;
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    bool is_trained;

public:
    FinancialPredictor() : bias(0.0), is_trained(false) {{}}
    
    // Sigmoid激活函数
    double sigmoid(double x) {{
        return 1.0 / (1.0 + std::exp(-x));
    }}
    
    // 标准化特征
    std::vector<double> standardize_features(const std::vector<double>& features) {{
        std::vector<double> standardized(features.size());
        for (size_t i = 0; i < features.size(); ++i) {{
            if (feature_stds[i] > 0) {{
                standardized[i] = (features[i] - feature_means[i]) / feature_stds[i];
            }} else {{
                standardized[i] = 0.0;
            }}
        }}
        return standardized;
    }}
    
    // 创建技术指标特征
    std::vector<double> create_features(const std::vector<double>& prices, 
                                       const std::vector<double>& volumes,
                                       const std::vector<double>& buy_amounts,
                                       const std::vector<double>& sell_amounts) {{
        std::vector<double> features;
        
        if (prices.size() < 20) {{
            throw std::runtime_error("需要至少20个数据点来计算特征");
        }}
        
        int idx = prices.size() - 1; // 使用最新的数据点
        
        // 价格变化特征
        double price_change_pct = (prices[idx] - prices[idx-1]) / prices[idx-1];
        features.push_back(price_change_pct);
        
        // 移动平均线特征
        for (int window : {{5, 10, 20}}) {{
            double ma = 0.0;
            for (int i = 0; i < window; ++i) {{
                ma += prices[idx - i];
            }}
            ma /= window;
            double ma_ratio = prices[idx] / ma;
            features.push_back(ma_ratio);
        }}
        
        // 成交量特征
        double volume_ma = 0.0;
        for (int i = 0; i < 5; ++i) {{
            volume_ma += volumes[idx - i];
        }}
        volume_ma /= 5.0;
        double volume_ratio = volumes[idx] / volume_ma;
        features.push_back(volume_ratio);
        
        // 买卖比例
        double buy_sell_ratio = buy_amounts[idx] / (sell_amounts[idx] + 1e-8);
        double buy_pressure = buy_amounts[idx] / (buy_amounts[idx] + sell_amounts[idx]);
        features.push_back(buy_sell_ratio);
        features.push_back(buy_pressure);
        
        // 滞后特征
        for (int lag : {{1, 2, 3}}) {{
            if (idx >= lag) {{
                features.push_back(prices[idx - lag]);
                features.push_back(volumes[idx - lag]);
            }} else {{
                features.push_back(0.0);
                features.push_back(0.0);
            }}
        }}
        
        return features;
    }}
    
    // 预测函数
    double predict_probability(const std::vector<double>& features) {{
        if (!is_trained) {{
            throw std::runtime_error("模型尚未训练");
        }}
        
        if (features.size() != weights.size()) {{
            throw std::runtime_error("特征维度不匹配");
        }}
        
        // 标准化特征
        std::vector<double> std_features = standardize_features(features);
        
        // 计算线性组合
        double linear_combination = bias;
        for (size_t i = 0; i < weights.size(); ++i) {{
            linear_combination += weights[i] * std_features[i];
        }}
        
        // 应用sigmoid函数
        return sigmoid(linear_combination);
    }}
    
    // 预测类别
    int predict_class(const std::vector<double>& features) {{
        double prob = predict_probability(features);
        return (prob > 0.5) ? 1 : 0;
    }}
    
    // 从文件加载模型参数
    bool load_model(const std::string& filename) {{
        std::ifstream file(filename);
        if (!file.is_open()) {{
            std::cerr << "无法打开模型文件: " << filename << std::endl;
            return false;
        }}
        
        std::string line;
        
        // 读取偏置
        if (std::getline(file, line)) {{
            bias = std::stod(line);
        }}
        
        // 读取权重
        if (std::getline(file, line)) {{
            std::istringstream iss(line);
            std::string weight_str;
            weights.clear();
            while (std::getline(iss, weight_str, ',')) {{
                weights.push_back(std::stod(weight_str));
            }}
        }}
        
        // 读取特征均值
        if (std::getline(file, line)) {{
            std::istringstream iss(line);
            std::string mean_str;
            feature_means.clear();
            while (std::getline(iss, mean_str, ',')) {{
                feature_means.push_back(std::stod(mean_str));
            }}
        }}
        
        // 读取特征标准差
        if (std::getline(file, line)) {{
            std::istringstream iss(line);
            std::string std_str;
            feature_stds.clear();
            while (std::getline(iss, std_str, ',')) {{
                feature_stds.push_back(std::stod(std_str));
            }}
        }}
        
        is_trained = true;
        file.close();
        return true;
    }}
    
    // 保存模型参数到文件
    bool save_model(const std::string& filename) {{
        std::ofstream file(filename);
        if (!file.is_open()) {{
            std::cerr << "无法创建模型文件: " << filename << std::endl;
            return false;
        }}
        
        // 保存偏置
        file << bias << std::endl;
        
        // 保存权重
        for (size_t i = 0; i < weights.size(); ++i) {{
            file << weights[i];
            if (i < weights.size() - 1) file << ",";
        }}
        file << std::endl;
        
        // 保存特征均值
        for (size_t i = 0; i < feature_means.size(); ++i) {{
            file << feature_means[i];
            if (i < feature_means.size() - 1) file << ",";
        }}
        file << std::endl;
        
        // 保存特征标准差
        for (size_t i = 0; i < feature_stds.size(); ++i) {{
            file << feature_stds[i];
            if (i < feature_stds.size() - 1) file << ",";
        }}
        file << std::endl;
        
        file.close();
        return true;
    }}
}};

// 使用示例
int main() {{
    try {{
        FinancialPredictor predictor;
        
        // 示例数据 (请替换为实际数据)
        std::vector<double> prices = {{50000, 50100, 50200, 50150, 50300, 50250, 50400, 50350, 50500, 50450,
                                     50600, 50550, 50700, 50650, 50800, 50750, 50900, 50850, 51000, 50950}};
        std::vector<double> volumes = {{1000, 1100, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450,
                                      1600, 1550, 1700, 1650, 1800, 1750, 1900, 1850, 2000, 1950}};
        std::vector<double> buy_amounts = {{500000, 550000, 600000, 575000, 650000, 625000, 700000, 675000, 750000, 725000,
                                          800000, 775000, 850000, 825000, 900000, 875000, 950000, 925000, 1000000, 975000}};
        std::vector<double> sell_amounts = {{500000, 550000, 600000, 575000, 650000, 625000, 700000, 675000, 750000, 725000,
                                           800000, 775000, 850000, 825000, 900000, 875000, 950000, 925000, 1000000, 975000}};
        
        // 创建特征
        std::vector<double> features = predictor.create_features(prices, volumes, buy_amounts, sell_amounts);
        
        std::cout << "特征向量大小: " << features.size() << std::endl;
        std::cout << "特征值: ";
        for (double f : features) {{
            std::cout << f << " ";
        }}
        std::cout << std::endl;
        
        // 注意: 在实际使用中，您需要先训练模型或加载预训练的模型
        // predictor.load_model("model_params.txt");
        
        std::cout << "金融预测模型C++代码生成完成！" << std::endl;
        std::cout << "请先训练模型并保存参数，然后使用load_model()加载。" << std::endl;
        
    }} catch (const std::exception& e) {{
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }}
    
    return 0;
}}
'''
        
        return code
    
    def _generate_random_forest_python(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成随机森林Python代码"""
        model_name = model_info.get('name', 'RandomForestModel')
        accuracy = model_info.get('accuracy', 0.5)

        key_features = []
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            key_features = [f[0] for f in sorted_features]

        return f'''# {model_name} - 基于AI解释的随机森林金融预测模型
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 模型准确率: {accuracy*100:.2f}%
# 关键特征: {', '.join(key_features[:3]) if key_features else '价格动量、RSI、MACD'}

# 随机森林Python代码生成功能开发中...
# 该模型将基于AI解释的特征重要性进行优化
'''

    def _generate_random_forest_cpp(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成随机森林C++代码"""
        return "// 随机森林C++代码生成功能开发中..."

    def _generate_svm_python(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成SVM Python代码"""
        return "# SVM Python代码生成功能开发中..."

    def _generate_svm_cpp(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成SVM C++代码"""
        return "// SVM C++代码生成功能开发中..."

    def _generate_gradient_boosting_python(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成梯度提升Python代码"""
        return "# 梯度提升Python代码生成功能开发中..."

    def _generate_gradient_boosting_cpp(self, model_info: Dict, feature_importance: Dict = None, ai_explanation: str = None) -> str:
        """生成梯度提升C++代码"""
        return "// 梯度提升C++代码生成功能开发中..."
