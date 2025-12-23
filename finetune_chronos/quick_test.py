#!/usr/bin/env python3
"""
快速测试脚本 - 使用预训练Chronos模型验证流程
不进行微调，直接使用预训练模型进行预测和特征提取
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_chronos_data(df, prediction_length=1024, context_length=2048):
    """
    准备Chronos数据格式
    """
    chronos_data = []
    
    for item_id in df['item_id'].unique()[:10]:  # 只取前10个序列进行快速测试
        item_data = df[df['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= context_length + prediction_length:
            selected_data = item_data.iloc[:context_length + prediction_length].copy()
            chronos_data.append(selected_data)
    
    if chronos_data:
        result_df = pd.concat(chronos_data, ignore_index=True)
        return TimeSeriesDataFrame(result_df)
    else:
        return None

def extract_residual_features(y_true, y_pred, prediction_length=1024):
    """
    从预测残差中提取特征
    """
    residual = y_true - y_pred
    
    features = {}
    
    # 基本残差特征
    features['mae'] = np.mean(np.abs(residual))
    features['mse'] = np.mean(residual ** 2)
    features['rmse'] = np.sqrt(features['mse'])
    features['std'] = np.std(residual)
    features['mean'] = np.mean(residual)
    
    # 分位数特征
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        features[f'residual_q{int(q*100)}'] = np.percentile(np.abs(residual), q * 100)
    
    # 简单的自相关特征
    for lag in range(1, 6):
        if len(residual) > lag:
            corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
            features[f'acf_lag{lag}'] = corr if not np.isnan(corr) else 0
    
    return features

def quick_test_method_a():
    """
    快速测试方案A的流程
    """
    print("=== 快速测试方案A流程 ===")
    
    # 读取数据
    train_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    
    # 只取ShengYing域的少量数据进行测试
    train_domain = train_df[train_df['item_id'].str.startswith('ShengYing')]
    test_domain = test_df[test_df['item_id'].str.startswith('ShengYing')]
    
    print(f"测试数据: 训练集 {train_domain['item_id'].nunique()} 个序列, 测试集 {test_domain['item_id'].nunique()} 个序列")
    
    # 使用预训练模型
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=1024,
        eval_metric="WQL",
        verbosity=1
    )
    
    # 使用预训练Chronos模型
    hyperparameters = {
        "Chronos": {
            "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
            "context_length": 2048,
            "prediction_length": 1024,
            "fine_tune": False,  # 不进行微调
        }
    }
    
    # 准备少量训练数据
    train_normal = train_domain[train_domain['label'] == 'normal']
    chronos_train_data = prepare_chronos_data(train_normal, 1024, 2048)
    
    if chronos_train_data is None:
        print("没有足够的训练数据")
        return
    
    print("开始训练预测器...")
    predictor.fit(
        train_data=chronos_train_data,
        hyperparameters=hyperparameters,
        time_limit=300  # 5分钟时间限制
    )
    
    print("开始提取特征...")
    
    # 提取特征
    features_list = []
    labels_list = []
    item_ids_list = []
    
    # 合并训练和测试数据
    all_data = pd.concat([train_domain, test_domain], ignore_index=True)

    # 获取每种标签的一些样本
    normal_items = all_data[all_data['label'] == 'normal']['item_id'].unique()[:20]
    spark_items = all_data[all_data['label'] == 'spark']['item_id'].unique()[:15]
    vibrate_items = all_data[all_data['label'] == 'vibrate']['item_id'].unique()[:15]

    unique_items = np.concatenate([normal_items, spark_items, vibrate_items])
    print(f"准备处理 {len(unique_items)} 个序列 (normal: {len(normal_items)}, spark: {len(spark_items)}, vibrate: {len(vibrate_items)})")

    for i, item_id in enumerate(unique_items):
        if i % 10 == 0:
            print(f"处理进度: {i}/{len(unique_items)}")

        try:
            item_data = all_data[all_data['item_id'] == item_id].sort_values('timestamp')
        except Exception as e:
            print(f"获取 {item_id} 数据时出错: {str(e)}")
            continue
        
        if len(item_data) >= 2048 + 1024:
            context_data = item_data.iloc[:2048].copy()
            true_future = item_data.iloc[2048:2048 + 1024]['target'].values
            
            chronos_data = TimeSeriesDataFrame(context_data)
            
            try:
                predictions = predictor.predict(chronos_data)

                # 修复预测结果索引问题
                if isinstance(predictions, pd.DataFrame):
                    # 如果predictions是DataFrame，取第一列的值
                    pred_values = predictions.iloc[:, 0].values
                elif hasattr(predictions, 'values'):
                    pred_values = predictions.values.flatten()
                else:
                    # 尝试直接访问item_id
                    try:
                        pred_values = predictions[item_id].values
                    except:
                        # 如果都失败了，取第一个可用的预测值
                        pred_values = list(predictions.values())[0].values if hasattr(predictions, 'values') else predictions

                # 确保pred_values是numpy数组且长度正确
                if not isinstance(pred_values, np.ndarray):
                    pred_values = np.array(pred_values)

                # 确保长度匹配
                min_length = min(len(true_future), len(pred_values))
                if min_length < 100:  # 如果预测长度太短，跳过
                    print(f"跳过 {item_id}: 预测长度太短 ({min_length})")
                    continue

                true_future_trimmed = true_future[:min_length]
                pred_values_trimmed = pred_values[:min_length]

                features = extract_residual_features(true_future_trimmed, pred_values_trimmed, min_length)
                features_list.append(features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)

            except Exception as e:
                print(f"预测 {item_id} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(features_list) == 0:
        print("没有成功提取到特征")
        return
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的特征")
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # 简单的分类测试
    if len(features_df['label'].unique()) > 1:
        feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['label']
        
        # 简单的训练测试划分
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        
        # 训练分类器
        clf = RandomForestClassifier(random_state=123, n_estimators=50)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"分类准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 保存测试结果
        output_dir = "/home/deep/TimeSeries/Zhendong/output/quick_test"
        os.makedirs(output_dir, exist_ok=True)
        
        test_results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(feature_cols, clf.feature_importances_))
        }
        
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        
        features_df.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
        
        print(f"测试结果保存到: {output_dir}")
    
    print("快速测试完成！")

def main():
    try:
        quick_test_method_a()
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
