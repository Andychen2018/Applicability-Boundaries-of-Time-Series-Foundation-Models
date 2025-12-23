#!/usr/bin/env python3
"""
简化版Chronos微调代码 - 使用ZhenDong数据
基于AutoGluon TimeSeriesPredictor的简单接口
"""

import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def prepare_zhendong_data(data_path, domain="ZhenDong", max_sequences=100):
    """
    准备ZhenDong域的数据用于微调
    """
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 筛选指定域的数据
    domain_data = df[df['item_id'].str.startswith(domain)]
    print(f"原始 {domain} 域数据: {len(domain_data)} 条记录, {domain_data['item_id'].nunique()} 个序列")
    
    # 限制序列数量以加快训练
    unique_items = domain_data['item_id'].unique()[:max_sequences]
    domain_data = domain_data[domain_data['item_id'].isin(unique_items)]
    
    print(f"使用 {domain} 域数据: {len(domain_data)} 条记录, {len(unique_items)} 个序列")
    print(f"标签分布: {domain_data['label'].value_counts().to_dict()}")
    
    # 转换为TimeSeriesDataFrame格式
    # 重命名列以符合AutoGluon要求
    ts_data = domain_data.rename(columns={'item_id': 'item_id', 'target': 'target'}).copy()
    
    # 确保时间戳列存在
    if 'timestamp' not in ts_data.columns:
        # 为每个item_id创建连续的时间戳
        ts_data['timestamp'] = ts_data.groupby('item_id').cumcount()
    
    # 转换为TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame(ts_data)
    
    return ts_df

def main():
    # ================== 1. 加载本地数据 ==================
    data_path = "/home/deep/TimeSeries/Zhendong/output/train_data.csv"
    
    # 准备数据 - 可以调整max_sequences来控制训练数据量
    data = prepare_zhendong_data(data_path, domain="ZhenDong", max_sequences=50)
    
    # 设置预测步长 - 可以根据需要调整
    prediction_length = 1024  # 与之前实验保持一致
    
    # 划分训练集 / 测试集
    train_data, test_data = data.train_test_split(prediction_length)
    
    print(f"训练集: {len(train_data)} 条记录")
    print(f"测试集: {len(test_data)} 条记录")
    
    # ================== 2. 创建并微调 Chronos 本地模型 ==================
    
    # 设置保存路径 - 与之前实验保持一致的目录结构
    save_path = "/home/deep/TimeSeries/Zhendong/output/simple_finetune/ZhenDong/predictor"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    predictor = TimeSeriesPredictor(
        path=save_path,
        prediction_length=prediction_length,
        target="target",
        eval_metric="WQL",
        verbosity=2
    ).fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": [
                {
                    "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                    "ag_args": {"name_suffix": "ZeroShot"}   # 零样本
                },
                {
                    "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                    "fine_tune": True,                       # 开启微调
                    "fine_tune_lr": 5e-5,                    # 学习率 - 可调整
                    "fine_tune_steps": 5000,                 # 微调步数 - 可调整
                    "context_length": 2048,                 # 模型输入的历史窗口长度 - 可调整
                    "prediction_length": prediction_length,
                    "dropout": 0.1,                         # dropout防过拟合 - 可调整
                    "ag_args": {"name_suffix": "FineTuned"}  # 区分模型
                },
            ]
        },
        time_limit=3600,          # 训练+微调时间上限（秒）- 可调整
        enable_ensemble=False,    # 关闭集成，单模型对比
    )
    
    print(f"模型已保存到: {save_path}")
    
    # ================== 3. 预测 & 可视化 ==================
    # 在测试集上预测
    print("开始预测...")
    predictions = predictor.predict(test_data)
    print("预测结果:")
    print(predictions.head())
    
    # 可视化前两个序列的预测效果
    try:
        print("生成可视化图表...")
        predictor.plot(
            data=test_data,
            predictions=predictions,
            item_ids=test_data.item_ids[:2],   # 取前两个序列
            max_history_length=200,
        )
        print("可视化图表已生成")
    except Exception as e:
        print(f"可视化生成失败: {str(e)}")
    
    # ================== 4. 查看模型表现 ==================
    print("模型性能评估:")
    leaderboard = predictor.leaderboard(test_data, silent=False)
    print(leaderboard)
    
    # 保存评估结果
    results_path = os.path.join(os.path.dirname(save_path), "evaluation_results.csv")
    leaderboard.to_csv(results_path, index=False)
    print(f"评估结果已保存到: {results_path}")
    
    # ================== 5. 保存模型信息 ==================
    model_info = {
        "model_path": save_path,
        "prediction_length": prediction_length,
        "context_length": 2048,
        "fine_tune_lr": 5e-5,
        "fine_tune_steps": 5000,
        "domain": "ZhenDong",
        "training_sequences": len(train_data.item_ids),
        "test_sequences": len(test_data.item_ids) if test_data else 0
    }
    
    info_path = os.path.join(os.path.dirname(save_path), "model_info.json")
    import json
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"模型信息已保存到: {info_path}")
    print("微调完成！")

if __name__ == "__main__":
    main()

# ================== 参数调整说明 ==================
"""
可调整的关键参数：

1. 数据相关：
   - max_sequences: 控制训练数据量（默认50，可增加到100+）
   - domain: 选择域（"ZhenDong" 或 "ShengYing"）

2. 模型相关：
   - prediction_length: 预测长度（默认1024）
   - context_length: 输入历史长度（默认2048，可调整为512, 1024, 4096）
   - fine_tune_lr: 学习率（默认5e-5，可尝试1e-5, 3e-5, 1e-4）
   - fine_tune_steps: 微调步数（默认5000，可调整为2000, 10000）
   - dropout: 防过拟合（默认0.1，可调整为0.05, 0.2）

3. 训练相关：
   - time_limit: 训练时间限制（默认3600秒=1小时）

使用方法：
1. 直接运行: python simple_finetune.py
2. 修改参数后重新运行
3. 模型会自动保存到指定目录
4. 可以通过修改save_path来保存到不同位置
"""
