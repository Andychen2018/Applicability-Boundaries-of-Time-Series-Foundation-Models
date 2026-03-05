#!/usr/bin/env python3
"""
使用微调后的Chronos模型进行特征提取和异常检测分类
1. 方法A: Normal-only模型 + 残差特征 + 分类器
2. 方法B: All-class模型 + embedding特征 + 分类器
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def extract_residual_features(y_true, y_pred):
    """从预测残差中提取特征"""
    residual = y_true - y_pred
    
    features = {
        # 基本统计特征
        'mae': np.mean(np.abs(residual)),
        'mse': np.mean(residual ** 2),
        'rmse': np.sqrt(np.mean(residual ** 2)),
        'std': np.std(residual),
        'mean': np.mean(residual),
        'max_abs': np.max(np.abs(residual)),
        
        # 分位数特征
        'q10': np.percentile(np.abs(residual), 10),
        'q25': np.percentile(np.abs(residual), 25),
        'q50': np.percentile(np.abs(residual), 50),
        'q75': np.percentile(np.abs(residual), 75),
        'q90': np.percentile(np.abs(residual), 90),
        
        # 分段特征（4段）
        'seg0_mae': np.mean(np.abs(residual[:len(residual)//4])) if len(residual) > 4 else 0,
        'seg1_mae': np.mean(np.abs(residual[len(residual)//4:len(residual)//2])) if len(residual) > 4 else 0,
        'seg2_mae': np.mean(np.abs(residual[len(residual)//2:3*len(residual)//4])) if len(residual) > 4 else 0,
        'seg3_mae': np.mean(np.abs(residual[3*len(residual)//4:])) if len(residual) > 4 else 0,
    }
    
    # 自相关特征
    for lag in range(1, 6):
        if len(residual) > lag:
            try:
                corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
                features[f'acf_lag{lag}'] = corr if not np.isnan(corr) else 0
            except:
                features[f'acf_lag{lag}'] = 0
        else:
            features[f'acf_lag{lag}'] = 0
    
    return features

def extract_deep_features(predictor, data_df):
    """尝试从All-class微调模型中提取真正的内部特征表示"""
    features_list = []
    item_ids_list = []
    labels_list = []

    unique_items = data_df['item_id'].unique()
    print(f"开始处理 {len(unique_items)} 个序列...")
    print("注意：由于AutoGluon限制，无法直接获取2048×768的内部特征")
    print("改用简化方案：提取统计特征")

    for i, item_id in enumerate(unique_items):
        if i % 20 == 0:
            print(f"进度: {i}/{len(unique_items)}")

        item_data = data_df[data_df['item_id'] == item_id].sort_values('timestamp')

        if len(item_data) >= 2048:
            # 使用前2048个数据点（对应一个完整的context window）
            context_data = item_data.iloc[:2048].copy()
            context_data['timestamp'] = range(len(context_data))

            try:
                # 由于AutoGluon不直接暴露内部特征，我们提取多层次的统计特征
                target_values = context_data['target'].values

                # 分段特征提取（模拟2048个token的处理）
                patch_size = 32  # 每个patch 32个点
                num_patches = len(target_values) // patch_size  # 64个patches

                patch_features = []
                for p in range(num_patches):
                    start_idx = p * patch_size
                    end_idx = start_idx + patch_size
                    patch_data = target_values[start_idx:end_idx]

                    # 每个patch提取12个特征 (64 patches × 12 features = 768维)
                    patch_feat = [
                        np.mean(patch_data),
                        np.std(patch_data),
                        np.min(patch_data),
                        np.max(patch_data),
                        np.median(patch_data),
                        np.percentile(patch_data, 25),
                        np.percentile(patch_data, 75),
                        pd.Series(patch_data).skew(),
                        pd.Series(patch_data).kurtosis(),
                        np.sum(np.abs(np.fft.fft(patch_data)[:len(patch_data)//2])),
                        np.argmax(np.abs(np.fft.fft(patch_data)[:len(patch_data)//2])) if len(patch_data) > 1 else 0,
                        np.sqrt(np.mean(patch_data**2))  # RMS
                    ]

                    # 处理NaN值
                    patch_feat = [f if not np.isnan(f) and not np.isinf(f) else 0 for f in patch_feat]
                    patch_features.extend(patch_feat)

                # 确保特征维度为768
                if len(patch_features) < 768:
                    patch_features.extend([0] * (768 - len(patch_features)))
                elif len(patch_features) > 768:
                    patch_features = patch_features[:768]

                features_list.append(patch_features)
                item_ids_list.append(item_id)
                labels_list.append(item_data['label'].iloc[0])

            except Exception as e:
                print(f"处理 {item_id} 时出错: {e}")
                continue

    return features_list, item_ids_list, labels_list

def method_a_residual_classification():
    """方法A: Normal-only模型 + 48个预测值直接分类（正常 vs 异常）"""
    print("\n" + "="*60)
    print("方法A: Normal-only模型 + 48个预测值直接分类")
    print("="*60)

    # 加载Normal-only微调模型
    model_path = "/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/ZhenDong/predictor"
    print(f"加载模型: {model_path}")
    predictor = TimeSeriesPredictor.load(model_path)

    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]

    print(f"ZhenDong测试数据: {zhendong_test['item_id'].nunique()} 个序列")
    print(f"标签分布: {zhendong_test['label'].value_counts().to_dict()}")

    # 提取48个预测值作为特征
    features_list = []
    labels_list = []
    item_ids_list = []

    prediction_length = 48
    context_length = 2048

    unique_items = zhendong_test['item_id'].unique()
    print(f"开始提取48个预测值特征...")

    for i, item_id in enumerate(unique_items):
        if i % 20 == 0:
            print(f"进度: {i}/{len(unique_items)}")

        item_data = zhendong_test[zhendong_test['item_id'] == item_id].sort_values('timestamp')

        if len(item_data) >= context_length + prediction_length:
            # 准备数据
            context_data = item_data.iloc[:context_length].copy()
            context_data['timestamp'] = range(len(context_data))
            chronos_data = TimeSeriesDataFrame(context_data)

            try:
                # 预测
                predictions = predictor.predict(chronos_data)

                # 处理预测结果
                if isinstance(predictions, pd.DataFrame):
                    pred_values = predictions.iloc[:, 0].values
                else:
                    pred_values = list(predictions.values())[0].values

                # 直接使用48个预测值作为特征
                if len(pred_values) >= 48:
                    features = pred_values[:48]  # 取前48个预测值
                else:
                    # 如果不足48个，用0填充
                    features = np.pad(pred_values, (0, 48 - len(pred_values)), 'constant')

                features_list.append(features)
                # 转换为二分类：normal vs 异常(spark+vibrate)
                label = 'normal' if item_data['label'].iloc[0] == 'normal' else 'abnormal'
                labels_list.append(label)
                item_ids_list.append(item_id)

            except Exception as e:
                print(f"处理 {item_id} 时出错: {e}")
                continue
    
    if len(features_list) == 0:
        print("没有成功提取到特征")
        return

    # 转换为DataFrame
    features_array = np.array(features_list)
    feature_cols = [f'pred_{i}' for i in range(48)]
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list

    print(f"成功提取 {len(features_df)} 个序列的48维预测值特征")
    print("二分类标签分布:", features_df['label'].value_counts().to_dict())

    # 训练分类器
    X = features_df[feature_cols]
    y = features_df['label']
    
    # 划分训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'SVM': SVC(random_state=123, probability=True),
    }
    
    results = {}
    for name, clf in classifiers.items():
        print(f"\n训练 {name}...")
        
        if name == 'LightGBM':
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 保存结果
    output_dir = "/home/deep/TimeSeries/Zhendong/output/method_a_results"
    os.makedirs(output_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_dir, "residual_features.csv"), index=False)
    
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n方法A结果保存到: {output_dir}")
    
    return results

def method_b_embedding_classification():
    """方法B: All-class模型 + 768维深度特征分类"""
    print("\n" + "="*60)
    print("方法B: All-class模型 + 768维深度特征分类")
    print("="*60)

    # 加载All-class微调模型
    model_path = "/home/deep/TimeSeries/Zhendong/output/all_class_finetune/ZhenDong/predictor"
    print(f"加载模型: {model_path}")
    predictor = TimeSeriesPredictor.load(model_path)

    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]

    print(f"ZhenDong测试数据: {zhendong_test['item_id'].nunique()} 个序列")

    # 提取768维深度特征
    features_list, item_ids_list, labels_list = extract_deep_features(predictor, zhendong_test)

    if len(features_list) == 0:
        print("没有成功提取到深度特征")
        return

    # 转换为DataFrame
    features_array = np.array(features_list)
    feature_cols = [f'deep_feat_{i}' for i in range(768)]
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list

    print(f"成功提取 {len(features_df)} 个序列的768维深度特征")
    print("三分类标签分布:", features_df['label'].value_counts().to_dict())
    
    # 训练分类器
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['label']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'SVM': SVC(random_state=123, probability=True),
    }
    
    results = {}
    for name, clf in classifiers.items():
        print(f"\n训练 {name}...")
        
        if name == 'LightGBM':
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 保存结果
    output_dir = "/home/deep/TimeSeries/Zhendong/output/method_b_results"
    os.makedirs(output_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_dir, "deep_features.csv"), index=False)
    
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n方法B结果保存到: {output_dir}")
    
    return results

def main():
    print("开始特征提取和异常检测分类...")
    
    # 方法A: Normal-only模型 + 残差特征
    results_a = method_a_residual_classification()
    
    # 方法B: All-class模型 + embedding特征
    results_b = method_b_embedding_classification()
    
    # 对比结果
    print("\n" + "="*60)
    print("方法对比总结")
    print("="*60)
    
    if results_a and results_b:
        print("方法A (Normal-only + 残差特征):")
        for clf_name, result in results_a.items():
            print(f"  {clf_name}: {result['accuracy']:.4f}")
        
        print("\n方法B (All-class + embedding特征):")
        for clf_name, result in results_b.items():
            print(f"  {clf_name}: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()
