#!/usr/bin/env python3
"""
生成测试用的金融数据CSV文件
包含OHLCV数据和买卖盘数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_financial_data(start_date, end_date, initial_price=50000):
    """
    生成模拟的金融市场数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_price: 初始价格
    
    Returns:
        DataFrame: 包含OHLCV和买卖盘数据
    """
    
    # 生成时间序列（每分钟）
    date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    data = []
    current_price = initial_price
    
    for timestamp in date_range:
        # 模拟价格波动（随机游走 + 趋势）
        # 添加一些周期性和趋势性
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # 模拟市场开盘时间的高波动性
        if 9 <= hour <= 16:  # 交易时间
            volatility = 0.002
        else:
            volatility = 0.001
            
        # 周末较低波动性
        if day_of_week >= 5:
            volatility *= 0.5
            
        # 随机价格变化
        price_change = np.random.normal(0, volatility) * current_price
        current_price += price_change
        
        # 确保价格不会变成负数
        current_price = max(current_price, 1000)
        
        # 生成OHLC数据
        # Open价格基于前一个收盘价
        open_price = current_price
        
        # 生成High和Low
        daily_range = abs(np.random.normal(0, 0.01)) * current_price
        high_price = open_price + random.uniform(0, daily_range)
        low_price = open_price - random.uniform(0, daily_range)
        
        # Close价格在High和Low之间
        close_price = random.uniform(low_price, high_price)
        current_price = close_price
        
        # 生成成交量（与价格波动相关）
        base_volume = 1000
        volume_multiplier = 1 + abs(price_change / current_price) * 10
        volume = int(base_volume * volume_multiplier * random.uniform(0.5, 2.0))
        
        # 生成买卖盘数据
        total_amount = volume * close_price
        
        # 买卖比例基于价格趋势
        if close_price > open_price:  # 上涨
            buy_ratio = random.uniform(0.55, 0.75)
        else:  # 下跌
            buy_ratio = random.uniform(0.25, 0.45)
            
        buy_amount = total_amount * buy_ratio
        sell_amount = total_amount * (1 - buy_ratio)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'buy_amount': round(buy_amount, 2),
            'sell_amount': round(sell_amount, 2)
        })
    
    return pd.DataFrame(data)

def generate_multiple_scenarios():
    """生成多种市场情况的测试数据"""
    
    # 场景1: 牛市数据（上涨趋势）
    print("生成牛市数据...")
    bull_start = datetime(2023, 1, 1)
    bull_end = datetime(2023, 4, 1)
    bull_data = generate_financial_data(bull_start, bull_end, initial_price=30000)
    
    # 为牛市添加上涨趋势
    trend_factor = np.linspace(1.0, 2.0, len(bull_data))
    for col in ['open', 'high', 'low', 'close']:
        bull_data[col] = bull_data[col] * trend_factor
    
    bull_data.to_csv('test_data_bull_market.csv', index=False)
    print(f"牛市数据已保存: {len(bull_data)} 条记录")
    
    # 场景2: 熊市数据（下跌趋势）
    print("生成熊市数据...")
    bear_start = datetime(2023, 5, 1)
    bear_end = datetime(2023, 8, 1)
    bear_data = generate_financial_data(bear_start, bear_end, initial_price=60000)
    
    # 为熊市添加下跌趋势
    trend_factor = np.linspace(1.0, 0.6, len(bear_data))
    for col in ['open', 'high', 'low', 'close']:
        bear_data[col] = bear_data[col] * trend_factor
    
    bear_data.to_csv('test_data_bear_market.csv', index=False)
    print(f"熊市数据已保存: {len(bear_data)} 条记录")
    
    # 场景3: 震荡市数据（横盘整理）
    print("生成震荡市数据...")
    sideways_start = datetime(2023, 9, 1)
    sideways_end = datetime(2023, 12, 1)
    sideways_data = generate_financial_data(sideways_start, sideways_end, initial_price=45000)
    
    # 为震荡市添加周期性波动
    cycle_length = len(sideways_data) // 10
    for i in range(len(sideways_data)):
        cycle_factor = 1 + 0.1 * np.sin(2 * np.pi * i / cycle_length)
        for col in ['open', 'high', 'low', 'close']:
            sideways_data.loc[i, col] *= cycle_factor
    
    sideways_data.to_csv('test_data_sideways_market.csv', index=False)
    print(f"震荡市数据已保存: {len(sideways_data)} 条记录")
    
    # 场景4: 综合数据（包含所有情况）
    print("生成综合数据...")
    combined_data = pd.concat([bull_data, bear_data, sideways_data], ignore_index=True)
    combined_data.to_csv('test_data_combined.csv', index=False)
    print(f"综合数据已保存: {len(combined_data)} 条记录")
    
    # 生成数据统计信息
    print("\n数据统计信息:")
    print(f"总记录数: {len(combined_data)}")
    print(f"时间范围: {combined_data['timestamp'].min()} 到 {combined_data['timestamp'].max()}")
    print(f"价格范围: {combined_data['close'].min():.2f} 到 {combined_data['close'].max():.2f}")
    print(f"平均成交量: {combined_data['volume'].mean():.0f}")
    
    return combined_data

if __name__ == "__main__":
    print("开始生成测试数据...")
    
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    random.seed(42)
    
    # 生成测试数据
    data = generate_multiple_scenarios()
    
    print("\n测试数据生成完成！")
    print("生成的文件:")
    print("- test_data_bull_market.csv (牛市数据)")
    print("- test_data_bear_market.csv (熊市数据)")
    print("- test_data_sideways_market.csv (震荡市数据)")
    print("- test_data_combined.csv (综合数据)")
    
    # 显示数据样本
    print("\n数据样本:")
    print(data.head(10).to_string(index=False))
