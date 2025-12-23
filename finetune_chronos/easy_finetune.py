#!/usr/bin/env python3
"""
双模型Chronos微调：
1. Normal-only微调模型
2. All-class微调模型
"""

import pandas as pd
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def train_models():
    # ================== 1. 加载本地数据 ==================
    print("加载数据...")
    df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")

    # 筛选ZhenDong域数据（可改为ShengYing）
    domain_data = df[df['item_id'].str.startswith('ZhenDong')]
    normal_data = domain_data[domain_data['label'] == 'normal']

    print(f"原始ZhenDong域数据: {domain_data['item_id'].nunique()} 个序列")
    print(f"Normal数据: {normal_data['item_id'].nunique()} 个序列")
    print(f"所有数据标签分布: {domain_data['label'].value_counts().to_dict()}")

    # *** 使用所有normal数据 ***
    all_normal_items = normal_data['item_id'].unique()
    normal_train_data = normal_data.copy()

    # *** 使用所有三类数据 ***
    all_class_data = domain_data.copy()

    print(f"Normal数据: {len(normal_train_data)} 条记录, {len(all_normal_items)} 个序列")
    print(f"All-class数据: {len(all_class_data)} 条记录, {domain_data['item_id'].nunique()} 个序列")

    # 转换为TimeSeriesDataFrame格式
    normal_train_data = normal_train_data.copy()
    normal_train_data['timestamp'] = normal_train_data.groupby('item_id').cumcount()
    normal_data_ts = TimeSeriesDataFrame(normal_train_data)

    all_class_data = all_class_data.copy()
    all_class_data['timestamp'] = all_class_data.groupby('item_id').cumcount()
    all_class_data_ts = TimeSeriesDataFrame(all_class_data)

    # 设置预测步长
    prediction_length = 256  # 预测长度，主要用于训练

    # 划分训练集 / 测试集
    normal_train_data_split, normal_test_data = normal_data_ts.train_test_split(prediction_length)
    all_class_train_data, all_class_test_data = all_class_data_ts.train_test_split(prediction_length)

    return (normal_train_data_split, normal_test_data,
            all_class_train_data, all_class_test_data, prediction_length)

# ================== 主程序 ==================
normal_train, normal_test, all_class_train, all_class_test, pred_len = train_models()

# ================== 2. 训练两个模型 ==================

print("\n" + "="*60)
print("开始训练模型1: Normal-only微调")
print("="*60)

# 模型1: Normal-only微调
save_path_normal = "/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/ZhenDong/predictor"
os.makedirs(os.path.dirname(save_path_normal), exist_ok=True)

predictor = TimeSeriesPredictor(
            path=save_path,
            prediction_length=prediction_length,   # 这里保持和上面一致
            target="target"
        ).fit(
            train_data=train_data,
            hyperparameters={
                "Chronos": [
                    {
                        "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                        "ag_args": {"name_suffix": "ZeroShot"}   # 零样本，不训练
                    },
                    {
                        "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                        "fine_tune": True,
                        "fine_tune_lr": 3e-5,             # 学习率
                        "fine_tune_steps": 5000,          # 微调步数
                        "context_length": 2048,           # 输入长度
                        "prediction_length": 256,         # 预测长度（这里主要是训练信号，不用太长）
                        "patch_length": 32,               # <<< 关键：65536 ÷ 32 = 2048
                        "dropout": 0.1,
                        "ag_args": {"name_suffix": "FineTuned"}
                    }
                ]
            },
            time_limit=40000,          # 训练时间上限（秒，可调整）
            enable_ensemble=False,     # 关闭集成
)


print(f"模型已保存到: {save_path}")

# ================== 3. 预测 & 可视化 ==================
# 在测试集上预测
predictions = predictor.predict(test_data)
print(predictions.head())

# 可视化前两个序列的预测效果
try:
    predictor.plot(
        data=test_data,
        predictions=predictions,
        item_ids=test_data.item_ids[:2],   # 取前两个序列
        max_history_length=200,
    )
except Exception as e:
    print(f"可视化失败: {e}")

# ================== 4. 查看模型表现 ==================
print(predictor.leaderboard(test_data, silent=False))

# 保存评估结果
leaderboard = predictor.leaderboard(test_data, silent=True)
results_path = os.path.join(os.path.dirname(save_path), "evaluation_results.csv")
leaderboard.to_csv(results_path, index=False)
print(f"结果已保存到: {results_path}")

print("\n" + "="*60)
print("Normal-only微调完成！")
print("="*60)
print("训练策略说明:")
print("1. 只使用normal标签的数据进行微调")
print("2. 每个序列包含65536个数据点（1秒@65536Hz）")
print("3. 模型学习正常数据的模式")
print("4. 后续可用于异常检测：")
print("   - 对所有数据（normal/spark/vibrate）进行预测")
print("   - 计算预测残差")
print("   - 正常数据残差小，异常数据残差大")
print("   - 基于残差特征进行分类")

# ================== 可调整参数说明 ==================
"""
主要可调整参数：

1. 第15行: max_sequences = 50        # 训练序列数量
2. 第12行: 'ZhenDong' -> 'ShengYing' # 切换域
3. 第24行: prediction_length = 1024  # 预测长度
4. 第42行: fine_tune_lr = 5e-5       # 学习率
5. 第43行: fine_tune_steps = 5000    # 微调步数  
6. 第44行: context_length = 2048     # 输入长度
7. 第46行: dropout = 0.1             # Dropout率
8. 第51行: time_limit = 3600         # 训练时间限制

运行方法：
python easy_finetune.py
"""
