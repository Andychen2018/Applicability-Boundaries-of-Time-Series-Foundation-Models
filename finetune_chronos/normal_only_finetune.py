#!/usr/bin/env python3
"""
Normal-only微调Chronos模型
只使用normal数据进行微调，然后用于异常检测特征提取
"""

import pandas as pd
import numpy as np
import os
import json
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def main():
    # ================== 1. 加载并筛选normal数据 ==================
    print("加载数据...")
    df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    
    # 选择域（可修改为'ShengYing'）
    domain = 'ZhenDong'
    domain_data = df[df['item_id'].str.startswith(domain)]
    
    # *** 关键：只使用normal数据进行微调 ***
    normal_data = domain_data[domain_data['label'] == 'normal']
    
    print(f"原始{domain}域数据: {domain_data['item_id'].nunique()} 个序列")
    print(f"Normal数据: {normal_data['item_id'].nunique()} 个序列")
    print(f"原始标签分布: {domain_data['label'].value_counts().to_dict()}")
    
    # 限制序列数量（可调整）
    max_sequences = 200  # 只用normal数据，可以用更多序列
    unique_items = normal_data['item_id'].unique()[:max_sequences]
    train_normal_data = normal_data[normal_data['item_id'].isin(unique_items)]
    
    print(f"用于微调的normal数据: {len(train_normal_data)} 条记录, {len(unique_items)} 个序列")
    
    # 转换为TimeSeriesDataFrame格式
    train_normal_data['timestamp'] = train_normal_data.groupby('item_id').cumcount()
    data = TimeSeriesDataFrame(train_normal_data)
    
    # 设置预测参数
    prediction_length = 1024  # 预测长度
    context_length = 2048     # 输入长度（受模型限制）
    
    # 划分训练集/测试集
    train_data, test_data = data.train_test_split(prediction_length)
    
    print(f"训练集: {len(train_data)} 条记录")
    print(f"测试集: {len(test_data)} 条记录")
    
    # ================== 2. Normal-only微调 ==================
    save_path = f"/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/{domain}/predictor"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("开始Normal-only微调...")
    predictor = TimeSeriesPredictor(
        path=save_path,
        prediction_length=prediction_length,
        target="target",
        eval_metric="WQL",
        verbosity=2
    ).fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": {
                "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                "fine_tune": True,
                "fine_tune_lr": 5e-5,
                "fine_tune_steps": 10000,
                "context_length": context_length,
                "prediction_length": prediction_length,
                "dropout": 0.1,
            }
        },
        time_limit=7200,  # 2小时
        enable_ensemble=False,
    )
    
    print(f"Normal-only微调完成！模型保存到: {save_path}")
    
    # ================== 3. 评估微调效果 ==================
    print("评估微调效果...")
    predictions = predictor.predict(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=False)
    
    # 保存评估结果
    results_path = os.path.join(os.path.dirname(save_path), "normal_only_evaluation.csv")
    leaderboard.to_csv(results_path, index=False)
    
    # ================== 4. 保存模型信息 ==================
    model_info = {
        "model_type": "normal_only_finetune",
        "domain": domain,
        "training_data": "normal_only",
        "training_sequences": len(unique_items),
        "prediction_length": prediction_length,
        "context_length": context_length,
        "fine_tune_lr": 5e-5,
        "fine_tune_steps": 10000,
        "model_path": save_path,
        "evaluation_score": float(leaderboard.iloc[0]['score_test']) if len(leaderboard) > 0 else None
    }
    
    info_path = os.path.join(os.path.dirname(save_path), "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"模型信息保存到: {info_path}")
    print("Normal-only微调完成！")
    
    # ================== 5. 下一步说明 ==================
    print("\n" + "="*50)
    print("下一步：使用微调后的模型进行异常检测")
    print("="*50)
    print("1. 加载所有数据（normal, spark, vibrate）")
    print("2. 使用微调后的模型进行预测")
    print("3. 计算预测残差")
    print("4. 提取残差特征")
    print("5. 训练分类器进行三分类")
    print(f"6. 微调后的模型路径: {save_path}")

if __name__ == "__main__":
    main()

# ================== 使用说明 ==================
"""
这个脚本实现了Normal-only微调策略：

1. 数据选择：
   - 只使用normal标签的数据进行微调
   - 避免异常数据影响模型的正常模式学习

2. 微调目标：
   - 让模型更好地学习正常数据的模式
   - 提高对正常数据的预测精度

3. 后续使用：
   - 用微调后的模型对所有数据（包括异常数据）进行预测
   - 正常数据的预测残差应该较小
   - 异常数据的预测残差应该较大
   - 基于残差特征进行异常分类

4. 参数调整：
   - max_sequences: 控制训练数据量
   - prediction_length: 预测长度
   - context_length: 输入长度
   - fine_tune_lr: 学习率
   - fine_tune_steps: 微调步数

运行方法：
python normal_only_finetune.py
"""
