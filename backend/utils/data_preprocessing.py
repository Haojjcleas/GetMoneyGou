"""
数据预处理和特征工程模块
包含数据清洗、特征提取和预处理功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import ta  # 技术分析库
from typing import Tuple, Dict, List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import time

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.scaler = None
        self.feature_columns = []
        self.target_column = 'price_direction'  # 预测价格方向（上涨/下跌）
        self.progress_callback = progress_callback
        self.n_cores = min(multiprocessing.cpu_count(), 4)  # 限制最大核心数

    def _report_progress(self, message: str, progress: float = None):
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def load_data_from_db(self, dataset_id: int) -> pd.DataFrame:
        """从数据库加载数据"""
        from models import MarketData
        
        # 查询数据
        market_data = MarketData.query.filter_by(dataset_id=dataset_id)\
                                    .order_by(MarketData.timestamp).all()
        
        # 转换为DataFrame
        data = []
        for record in market_data:
            data.append({
                'timestamp': record.timestamp,
                'open': record.open_price,
                'high': record.high_price,
                'low': record.low_price,
                'close': record.close_price,
                'volume': record.volume,
                'buy_amount': record.buy_amount,
                'sell_amount': record.sell_amount
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        df_clean = df.copy()
        
        # 1. 处理缺失值
        # 使用前向填充处理价格数据的缺失值
        price_columns = ['open', 'high', 'low', 'close']
        df_clean[price_columns] = df_clean[price_columns].ffill()
        
        # 使用中位数填充成交量数据的缺失值
        volume_columns = ['volume', 'buy_amount', 'sell_amount']
        for col in volume_columns:
            if df_clean[col].isnull().any():
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
        
        # 2. 处理异常值
        # 检查价格逻辑错误
        invalid_rows = (df_clean['high'] < df_clean['low']) | \
                      (df_clean['high'] < df_clean['open']) | \
                      (df_clean['high'] < df_clean['close']) | \
                      (df_clean['low'] > df_clean['open']) | \
                      (df_clean['low'] > df_clean['close'])
        
        if invalid_rows.any():
            print(f"发现 {invalid_rows.sum()} 行价格逻辑错误，将进行修正")
            # 修正逻辑错误：使用相邻行的平均值
            for idx in df_clean[invalid_rows].index:
                if idx > 0:
                    prev_close = df_clean.loc[df_clean.index[df_clean.index.get_loc(idx)-1], 'close']
                    df_clean.loc[idx, ['open', 'high', 'low', 'close']] = prev_close
        
        # 3. 处理极端异常值（使用IQR方法）
        for col in price_columns + volume_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 使用3倍IQR作为极端异常值的阈值
            upper_bound = Q3 + 3 * IQR
            
            # 将极端异常值替换为边界值
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征（优化版本）"""
        df_features = df.copy()

        self._report_progress("开始计算技术指标...", 0.0)

        # 1. 基础价格特征
        self._report_progress("计算基础价格特征...", 0.1)
        df_features['price_range'] = df_features['high'] - df_features['low']
        df_features['price_change'] = df_features['close'] - df_features['open']
        df_features['price_change_pct'] = df_features['price_change'] / df_features['open']

        # 2. 移动平均线（并行计算）
        self._report_progress("计算移动平均线...", 0.2)
        windows = [5, 10, 20, 50]

        # 使用向量化操作而不是循环
        for i, window in enumerate(windows):
            df_features[f'ma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ma_{window}_ratio'] = df_features['close'] / df_features[f'ma_{window}']
            self._report_progress(f"完成 MA{window}...", 0.2 + (i + 1) * 0.1 / len(windows))
        
        # 3. 技术指标（使用ta库，优化计算）
        self._report_progress("计算技术指标...", 0.3)
        try:
            # RSI (相对强弱指数)
            self._report_progress("计算RSI指标...", 0.35)
            df_features['rsi'] = ta.momentum.RSIIndicator(df_features['close']).rsi()

            # MACD
            self._report_progress("计算MACD指标...", 0.4)
            macd = ta.trend.MACD(df_features['close'])
            df_features['macd'] = macd.macd()
            df_features['macd_signal'] = macd.macd_signal()
            df_features['macd_histogram'] = macd.macd_diff()

            # 布林带
            self._report_progress("计算布林带指标...", 0.45)
            bollinger = ta.volatility.BollingerBands(df_features['close'])
            df_features['bb_upper'] = bollinger.bollinger_hband()
            df_features['bb_lower'] = bollinger.bollinger_lband()
            df_features['bb_middle'] = bollinger.bollinger_mavg()
            df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
            df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / df_features['bb_width']

            # 随机指标
            self._report_progress("计算随机指标...", 0.5)
            stoch = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close'])
            df_features['stoch_k'] = stoch.stoch()
            df_features['stoch_d'] = stoch.stoch_signal()

            # 威廉指标
            self._report_progress("计算威廉指标...", 0.55)
            df_features['williams_r'] = ta.momentum.WilliamsRIndicator(df_features['high'], df_features['low'], df_features['close']).williams_r()

            # 添加更多技术指标
            self._report_progress("计算其他技术指标...", 0.6)
            df_features['cci'] = ta.trend.CCIIndicator(df_features['high'], df_features['low'], df_features['close']).cci()
            df_features['atr'] = ta.volatility.AverageTrueRange(df_features['high'], df_features['low'], df_features['close']).average_true_range()
            df_features['adx'] = ta.trend.ADXIndicator(df_features['high'], df_features['low'], df_features['close']).adx()

        except Exception as e:
            self._report_progress(f"计算技术指标时出错: {e}", None)
        
        # 4. 成交量指标
        self._report_progress("计算成交量指标...", 0.65)
        df_features['volume_ma_5'] = df_features['volume'].rolling(window=5).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_5']

        # 买卖比例
        df_features['buy_sell_ratio'] = df_features['buy_amount'] / (df_features['sell_amount'] + 1e-8)
        df_features['buy_pressure'] = df_features['buy_amount'] / (df_features['buy_amount'] + df_features['sell_amount'])

        # 5. 时间特征
        self._report_progress("计算时间特征...", 0.7)
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month

        # 6. 滞后特征（向量化计算）
        self._report_progress("计算滞后特征...", 0.75)
        lags = [1, 2, 3, 5]
        for i, lag in enumerate(lags):
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            df_features[f'price_change_lag_{lag}'] = df_features['price_change'].shift(lag)
            self._report_progress(f"完成滞后特征 lag-{lag}...", 0.75 + (i + 1) * 0.05 / len(lags))

        self._report_progress("技术指标计算完成", 0.8)
        
        return df_features
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建目标变量（使用下一个时间点）"""
        df_target = df.copy()

        self._report_progress("创建目标变量...", 0.85)

        # 预测下一个时间点的价格方向（固定为1个时间点）
        future_close = df_target['close'].shift(-1)
        df_target['future_price_change'] = future_close - df_target['close']
        df_target['future_price_change_pct'] = df_target['future_price_change'] / df_target['close']

        # 二分类目标：1表示上涨，0表示下跌
        df_target[self.target_column] = (df_target['future_price_change_pct'] > 0).astype(int)

        # 多分类目标：0=大跌，1=小跌，2=小涨，3=大涨
        df_target['price_direction_multi'] = pd.cut(
            df_target['future_price_change_pct'],
            bins=[-np.inf, -0.02, 0, 0.02, np.inf],
            labels=[0, 1, 2, 3]
        )
        # 处理NaN值并转换为整数
        df_target['price_direction_multi'] = df_target['price_direction_multi'].fillna(1).astype(int)

        # 回归目标：未来价格变化百分比
        df_target['price_change_target'] = df_target['future_price_change_pct']

        return df_target
    
    def select_features(self, df: pd.DataFrame, max_features: int = 25) -> List[str]:
        """优化的特征选择（快速版本）"""
        self._report_progress("开始特征选择...", 0.9)

        # 排除不用于训练的列
        exclude_columns = [
            'open', 'high', 'low', 'close', 'volume', 'buy_amount', 'sell_amount',  # 原始数据
            'future_price_change', 'future_price_change_pct',  # 目标变量相关
            self.target_column, 'price_direction_multi', 'price_change_target'
        ]

        # 选择数值型特征
        feature_candidates = []
        for col in df.columns:
            if col not in exclude_columns and df[col].dtype in ['int64', 'float64']:
                # 检查特征是否有足够的变化（不是常数）且缺失值不太多
                if (df[col].nunique() > 1 and
                    not df[col].isnull().all() and
                    df[col].isnull().sum() / len(df) < 0.3):  # 缺失值少于30%
                    feature_candidates.append(col)

        self._report_progress(f"发现 {len(feature_candidates)} 个候选特征", 0.92)

        # 如果特征数量合适，直接返回
        if len(feature_candidates) <= max_features:
            self.feature_columns = feature_candidates
            self._report_progress("特征数量合适，无需进一步筛选", 0.95)
            return feature_candidates

        # 使用简单的方差筛选（快速）
        try:
            self._report_progress("执行快速特征筛选...", 0.93)

            # 计算特征方差，选择方差最大的特征
            feature_vars = []
            for col in feature_candidates:
                var = df[col].fillna(0).var()
                if not np.isnan(var) and not np.isinf(var):
                    feature_vars.append((col, var))

            # 按方差排序，选择前max_features个
            feature_vars.sort(key=lambda x: x[1], reverse=True)
            selected_features = [col for col, _ in feature_vars[:max_features]]

            self.feature_columns = selected_features
            self._report_progress(f"选择了 {len(selected_features)} 个高方差特征", 0.95)
            return selected_features

        except Exception as e:
            self._report_progress(f"特征选择出错: {e}，使用前{max_features}个特征", None)
            # 如果出错，返回前max_features个特征
            selected = feature_candidates[:max_features]
            self.feature_columns = selected
            return selected
    
    def prepare_data_for_ml(self, dataset_id: int, test_size: float = 0.2) -> Dict:
        """准备机器学习数据（优化版本）"""

        # 1. 加载数据
        self._report_progress("加载数据...", 0.0)
        df = self.load_data_from_db(dataset_id)

        # 2. 数据清洗
        self._report_progress("数据清洗...", 0.1)
        df_clean = self.clean_data(df)

        # 3. 特征工程
        df_features = self.create_technical_indicators(df_clean)

        # 4. 创建目标变量（固定使用下一个时间点）
        df_target = self.create_target_variable(df_features)

        # 5. 特征选择
        feature_columns = self.select_features(df_target)
        
        # 6. 移除包含NaN的行
        self._report_progress("清理最终数据...", 0.96)
        df_final = df_target.dropna()

        if len(df_final) == 0:
            raise ValueError("处理后没有有效数据")

        # 7. 分割数据
        self._report_progress("分割训练和测试数据...", 0.97)
        split_index = int(len(df_final) * (1 - test_size))

        train_data = df_final.iloc[:split_index]
        test_data = df_final.iloc[split_index:]

        # 8. 准备特征和目标
        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]

        y_train_binary = train_data[self.target_column]
        y_test_binary = test_data[self.target_column]

        y_train_multi = train_data['price_direction_multi']
        y_test_multi = test_data['price_direction_multi']

        y_train_regression = train_data['price_change_target']
        y_test_regression = test_data['price_change_target']

        # 9. 特征缩放
        self._report_progress("特征标准化...", 0.98)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self._report_progress("数据预处理完成！", 1.0)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train_binary': y_train_binary,
            'y_test_binary': y_test_binary,
            'y_train_multi': y_train_multi,
            'y_test_multi': y_test_multi,
            'y_train_regression': y_train_regression,
            'y_test_regression': y_test_regression,
            'feature_columns': feature_columns,
            'train_dates': train_data.index,
            'test_dates': test_data.index,
            'scaler': self.scaler,
            'original_data': df_final
        }
    
    def get_feature_importance_names(self) -> List[str]:
        """获取特征名称（用于特征重要性分析）"""
        return self.feature_columns
