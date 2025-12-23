#!/usr/bin/env python3
"""
方案A：Normal-only微调 + 残差特征多分类
仅用normal数据微调Chronos，然后用预测残差构造特征进行三分类
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_chronos_data(df, prediction_length=1024, context_length=4096):
    """
    准备Chronos训练数据格式
    """
    # 按item_id分组，每个序列截取context_length + prediction_length长度
    chronos_data = []
    
    for item_id in df['item_id'].unique():
        item_data = df[df['item_id'] == item_id].sort_values('timestamp')
        
        # 确保序列长度足够
        if len(item_data) >= context_length + prediction_length:
            # 取前context_length + prediction_length个点
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
    
    # 整体残差特征
    features['mae'] = np.mean(np.abs(residual))
    features['mse'] = np.mean(residual ** 2)
    features['rmse'] = np.sqrt(features['mse'])
    features['mape'] = np.mean(np.abs(residual / (y_true + 1e-8))) * 100
    features['mdae'] = np.median(np.abs(residual))
    
    # 分段残差特征 (4段，每段256点)
    segment_size = prediction_length // 4
    for i in range(4):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment_residual = residual[start_idx:end_idx]
        
        features[f'seg{i}_mae'] = np.mean(np.abs(segment_residual))
        features[f'seg{i}_mse'] = np.mean(segment_residual ** 2)
    
    # 分位数残差特征
    for q in [0.1, 0.5, 0.9]:
        features[f'residual_q{int(q*100)}'] = np.percentile(np.abs(residual), q * 100)
    
    # 自相关特征 (前10阶)
    try:
        from statsmodels.tsa.stattools import acf
        acf_values = acf(residual, nlags=10, fft=True)
        for i, acf_val in enumerate(acf_values[1:], 1):  # 跳过lag=0
            features[f'acf_lag{i}'] = acf_val
    except:
        # 如果statsmodels不可用，用简单的自相关计算
        for lag in range(1, 11):
            if len(residual) > lag:
                corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
                features[f'acf_lag{lag}'] = corr if not np.isnan(corr) else 0
    
    # 频域特征
    try:
        fft_residual = np.fft.fft(residual)
        freqs = np.fft.fftfreq(len(residual))
        
        # 高频/低频能量比
        low_freq_mask = np.abs(freqs) < 0.2
        high_freq_mask = np.abs(freqs) >= 0.2
        
        low_freq_energy = np.sum(np.abs(fft_residual[low_freq_mask]) ** 2)
        high_freq_energy = np.sum(np.abs(fft_residual[high_freq_mask]) ** 2)
        
        features['freq_ratio'] = high_freq_energy / (low_freq_energy + 1e-8)
    except:
        features['freq_ratio'] = 0
    
    return features

def train_method_a_single_domain(domain, output_dir, prediction_length=1024, context_length=4096):
    """
    训练单个域的方案A模型
    """
    print(f"\n=== 开始训练 {domain} 域的方案A模型 ===")
    
    # 创建输出目录
    domain_output_dir = os.path.join(output_dir, f"methodA_residual_normal_ft/{domain}")
    os.makedirs(domain_output_dir, exist_ok=True)
    os.makedirs(os.path.join(domain_output_dir, "figures"), exist_ok=True)
    
    # 读取数据
    train_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    val_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/val_data.csv")
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    
    # 筛选域数据
    train_domain = train_df[train_df['item_id'].str.startswith(domain)]
    val_domain = val_df[val_df['item_id'].str.startswith(domain)]
    test_domain = test_df[test_df['item_id'].str.startswith(domain)]
    
    print(f"{domain} 域数据统计:")
    print(f"  训练集: {len(train_domain)} 条记录, {train_domain['item_id'].nunique()} 个序列")
    print(f"  验证集: {len(val_domain)} 条记录, {val_domain['item_id'].nunique()} 个序列")
    print(f"  测试集: {len(test_domain)} 条记录, {test_domain['item_id'].nunique()} 个序列")
    
    # 1. 准备normal-only训练数据
    train_normal = train_domain[train_domain['label'] == 'normal']
    print(f"Normal训练数据: {len(train_normal)} 条记录, {train_normal['item_id'].nunique()} 个序列")
    
    # 准备Chronos数据格式
    print("准备Chronos训练数据...")
    chronos_train_data = prepare_chronos_data(train_normal, prediction_length, context_length)
    
    if chronos_train_data is None:
        print(f"警告: {domain} 域没有足够长度的normal数据进行训练")
        return
    
    # 2. 微调Chronos模型
    print("开始微调Chronos模型...")
    predictor_path = os.path.join(domain_output_dir, "predictor")
    
    predictor = TimeSeriesPredictor(
        path=predictor_path,
        target="target",
        prediction_length=prediction_length,
        eval_metric="WQL",
        verbosity=2
    )
    
    # 微调参数
    hyperparameters = {
        "Chronos": {
            "model_path": "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
            "context_length": context_length,
            "prediction_length": prediction_length,
            "fine_tune": True,
            "fine_tune_lr": 5e-5,
            "fine_tune_steps": 10000,
            "dropout": 0.1,
        }
    }
    
    predictor.fit(
        train_data=chronos_train_data,
        hyperparameters=hyperparameters,
        time_limit=3600  # 1小时时间限制
    )
    
    print("Chronos模型微调完成！")

    # 3. 对所有数据进行预测，提取残差特征
    print("开始提取残差特征...")

    # 合并所有数据用于特征提取
    all_data = pd.concat([train_domain, val_domain, test_domain], ignore_index=True)

    # 为每个序列提取特征
    features_list = []
    labels_list = []
    item_ids_list = []

    for item_id in all_data['item_id'].unique():
        item_data = all_data[all_data['item_id'] == item_id].sort_values('timestamp')

        if len(item_data) >= context_length + prediction_length:
            # 准备预测数据
            context_data = item_data.iloc[:context_length].copy()
            true_future = item_data.iloc[context_length:context_length + prediction_length]['target'].values

            # 转换为Chronos格式
            chronos_data = TimeSeriesDataFrame(context_data)

            try:
                # 进行预测
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

                # 提取残差特征
                features = extract_residual_features(true_future_trimmed, pred_values_trimmed, min_length)
                features_list.append(features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)

            except Exception as e:
                print(f"预测 {item_id} 时出错: {str(e)}")
                continue

    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list

    print(f"成功提取 {len(features_df)} 个序列的残差特征")

    # 4. 划分特征数据
    train_features = features_df[features_df['item_id'].isin(train_domain['item_id'].unique())]
    val_features = features_df[features_df['item_id'].isin(val_domain['item_id'].unique())]
    test_features = features_df[features_df['item_id'].isin(test_domain['item_id'].unique())]

    # 保存特征数据
    train_features.to_csv(os.path.join(domain_output_dir, "residual_features_train.csv"), index=False)
    val_features.to_csv(os.path.join(domain_output_dir, "residual_features_val.csv"), index=False)
    test_features.to_csv(os.path.join(domain_output_dir, "residual_features_test.csv"), index=False)

    # 5. 训练分类器
    print("开始训练分类器...")

    # 准备特征和标签
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]

    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['label']
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features['label']
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features['label']

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 训练多个分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
        'SVM': SVC(random_state=123, probability=True),
        'MLP': MLPClassifier(random_state=123, max_iter=1000)
    }

    best_classifier = None
    best_score = 0
    results = {}

    for clf_name, clf in classifiers.items():
        print(f"训练 {clf_name}...")

        try:
            if clf_name == 'LightGBM':
                clf.fit(X_train, y_train)
                val_pred = clf.predict(X_val)
            else:
                clf.fit(X_train_scaled, y_train)
                val_pred = clf.predict(X_val_scaled)

            val_accuracy = accuracy_score(y_val, val_pred)
            results[clf_name] = {
                'val_accuracy': val_accuracy,
                'classifier': clf
            }

            print(f"{clf_name} 验证集准确率: {val_accuracy:.4f}")

            if val_accuracy > best_score:
                best_score = val_accuracy
                best_classifier = clf
                best_clf_name = clf_name

        except Exception as e:
            print(f"训练 {clf_name} 时出错: {str(e)}")
            continue

    # 6. 使用最佳分类器进行最终评估
    print(f"\n最佳分类器: {best_clf_name} (验证集准确率: {best_score:.4f})")

    if best_classifier is not None:
        # 在测试集上评估
        if best_clf_name == 'LightGBM':
            test_pred = best_classifier.predict(X_test)
            test_pred_proba = best_classifier.predict_proba(X_test)
        else:
            test_pred = best_classifier.predict(X_test_scaled)
            test_pred_proba = best_classifier.predict_proba(X_test_scaled)

        test_accuracy = accuracy_score(y_test, test_pred)

        # 保存分类器
        with open(os.path.join(domain_output_dir, "clf_model.pkl"), "wb") as f:
            pickle.dump({
                'classifier': best_classifier,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'classifier_name': best_clf_name
            }, f)

        # 保存评估结果
        metrics = {
            'domain': domain,
            'method': 'A_residual_normal_ft',
            'best_classifier': best_clf_name,
            'val_accuracy': best_score,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }

        with open(os.path.join(domain_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 生成可视化图表
        plt.figure(figsize=(12, 4))

        # 混淆矩阵
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{domain} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 特征重要性 (仅对树模型)
        if best_clf_name in ['LightGBM', 'RandomForest']:
            plt.subplot(1, 3, 2)
            importance = best_classifier.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)

            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.title(f'{domain} - Top 10 Feature Importance')
            plt.xlabel('Importance')

        # 类别分布
        plt.subplot(1, 3, 3)
        test_features['label'].value_counts().plot(kind='bar')
        plt.title(f'{domain} - Test Set Label Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(domain_output_dir, "figures", "evaluation.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, test_pred)}")

    return predictor

def main():
    # 设置参数
    prediction_length = 1024
    context_length = 4096
    output_dir = "/home/deep/TimeSeries/Zhendong/output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练各个域的模型
    domains = ["ZhenDong"]  # 只训练ZhenDong域，ShengYing已完成
    
    for domain in domains:
        try:
            predictor = train_method_a_single_domain(
                domain, output_dir, prediction_length, context_length
            )
            print(f"{domain} 域方案A训练完成")
        except Exception as e:
            print(f"训练 {domain} 域时出错: {str(e)}")
            continue

if __name__ == "__main__":
    main()
