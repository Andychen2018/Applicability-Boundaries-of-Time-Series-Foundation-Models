#!/usr/bin/env python3
"""
双模型Chronos微调：
1. Normal-only微调模型 - 只用normal数据
2. All-class微调模型 - 用所有三类数据
"""

import pandas as pd
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def main():
    # ================== 1. 加载和准备数据 ==================
    print("加载所有数据集...")

    # 加载train、val、test三个数据集
    train_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    val_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/val_data.csv")
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")

    # 合并所有数据集
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"合并后总数据: {len(df)} 条记录, {df['item_id'].nunique()} 个序列")

    # 筛选ZhenDong域数据
    domain_data = df[df['item_id'].str.startswith('ZhenDong')]
    normal_data = domain_data[domain_data['label'] == 'normal']
    
    print(f"ZhenDong域总数据: {domain_data['item_id'].nunique()} 个序列")
    print(f"Normal数据: {normal_data['item_id'].nunique()} 个序列")
    print(f"ZhenDong域标签分布: {domain_data['label'].value_counts().to_dict()}")
    print(f"使用所有train+val+test数据进行微调（用于特征提取）")
    
    # 准备两套数据
    # 1. Normal-only数据
    normal_train_data = normal_data.copy()
    normal_train_data['timestamp'] = normal_train_data.groupby('item_id').cumcount()
    normal_data_ts = TimeSeriesDataFrame(normal_train_data)
    
    # 2. All-class数据
    all_class_data = domain_data.copy()
    all_class_data['timestamp'] = all_class_data.groupby('item_id').cumcount()
    all_class_data_ts = TimeSeriesDataFrame(all_class_data)
    
    # 设置预测参数
    prediction_length = 48  # 预测长度
    
    # 划分训练集/测试集
    normal_train, normal_test = normal_data_ts.train_test_split(prediction_length)
    all_class_train, all_class_test = all_class_data_ts.train_test_split(prediction_length)
    
    print(f"Normal训练集: {len(normal_train)} 条记录 (来自train+val+test)")
    print(f"All-class训练集: {len(all_class_train)} 条记录 (来自train+val+test)")
    print("注意: 使用所有数据进行微调，因为目标是特征提取而非预测性能评估")
    
    # ================== 2. 训练模型1: Normal-only微调 ==================
    print("\n" + "="*60)
    print("开始训练模型1: Normal-only微调")
    print("="*60)
    
    save_path_normal = "/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/ZhenDong/predictor"
    os.makedirs(os.path.dirname(save_path_normal), exist_ok=True)
    
    predictor_normal = TimeSeriesPredictor(
        path=save_path_normal,
        prediction_length=prediction_length,
        target="target"
    ).fit(
        train_data=normal_train,
        hyperparameters={
            "Chronos": {
                "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                "fine_tune": True,
                "fine_tune_lr": 3e-5,
                "fine_tune_steps": 5000,
                "context_length": 2048,
                "prediction_length": prediction_length,
                "patch_length": 32,  # 关键：65536 ÷ 32 = 2048
                "dropout": 0.1,
            }
        },
        time_limit=20000,  # 约5.5小时
        enable_ensemble=False,
    )
    
    print(f"Normal-only模型保存到: {save_path_normal}")
    
    # 评估Normal-only模型
    print("评估Normal-only模型...")
    normal_predictions = predictor_normal.predict(normal_test)
    normal_leaderboard = predictor_normal.leaderboard(normal_test, silent=False)
    
    # ================== 3. 训练模型2: All-class微调 ==================
    print("\n" + "="*60)
    print("开始训练模型2: All-class微调")
    print("="*60)
    
    save_path_all = "/home/deep/TimeSeries/Zhendong/output/all_class_finetune/ZhenDong/predictor"
    os.makedirs(os.path.dirname(save_path_all), exist_ok=True)
    
    predictor_all = TimeSeriesPredictor(
        path=save_path_all,
        prediction_length=prediction_length,
        target="target"
    ).fit(
        train_data=all_class_train,
        hyperparameters={
            "Chronos": {
                "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                "fine_tune": True,
                "fine_tune_lr": 3e-5,
                "fine_tune_steps": 5000,
                "context_length": 2048,
                "prediction_length": prediction_length,
                "patch_length": 32,  # 关键：65536 ÷ 32 = 2048
                "dropout": 0.1,
            }
        },
        time_limit=20000,  # 约5.5小时
        enable_ensemble=False,
    )
    
    print(f"All-class模型保存到: {save_path_all}")
    
    # 评估All-class模型
    print("评估All-class模型...")
    all_predictions = predictor_all.predict(all_class_test)
    all_leaderboard = predictor_all.leaderboard(all_class_test, silent=False)
    
    # ================== 4. 保存结果 ==================
    print("\n" + "="*60)
    print("保存评估结果")
    print("="*60)
    
    # 保存Normal-only结果
    normal_results_path = os.path.join(os.path.dirname(save_path_normal), "evaluation_results.csv")
    normal_leaderboard.to_csv(normal_results_path, index=False)
    
    # 保存All-class结果
    all_results_path = os.path.join(os.path.dirname(save_path_all), "evaluation_results.csv")
    all_leaderboard.to_csv(all_results_path, index=False)
    
    print(f"Normal-only结果保存到: {normal_results_path}")
    print(f"All-class结果保存到: {all_results_path}")
    
    # ================== 5. 总结 ==================
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print("1. Normal-only微调模型:")
    print(f"   - 路径: {save_path_normal}")
    print(f"   - 训练数据: 所有normal标签数据 (train+val+test)")
    print(f"   - 用途: 异常检测（normal数据残差小，异常数据残差大）")
    print()
    print("2. All-class微调模型:")
    print(f"   - 路径: {save_path_all}")
    print(f"   - 训练数据: 所有三类数据 (train+val+test)")
    print(f"   - 用途: 通用时序预测或embedding提取")
    print()
    print("3. 关键参数:")
    print("   - patch_length: 32 (将65536点分成2048个patches)")
    print("   - context_length: 2048 (模型最大输入长度)")
    print("   - prediction_length: 48 (预测长度)")
    print()
    print("4. 下游使用:")
    print("   - 特征提取时需要保持相同的patch_length=32")
    print("   - 这个参数已经保存在模型中")
    print("   - 直接加载模型即可使用")

if __name__ == "__main__":
    main()

# ================== 使用说明 ==================
"""
关于patch_length的重要说明：

1. patch_length=32的作用：
   - 将65536个时间点分成 65536÷32=2048 个patches
   - 每个patch包含32个连续时间点
   - 模型处理2048个patches，获得全局时序信息

2. 模型保存和加载：
   - patch_length参数已经保存在微调后的模型中
   - 加载模型时会自动使用正确的patch_length
   - 不需要单独保存这个参数

3. 下游特征提取：
   - 直接加载微调后的模型
   - 模型会自动应用正确的patch处理
   - 无需手动设置patch_length

4. 两个模型的用途：
   - Normal-only: 用于异常检测（残差分析）
   - All-class: 用于通用预测或embedding提取

运行方法：
python dual_model_finetune.py
"""
