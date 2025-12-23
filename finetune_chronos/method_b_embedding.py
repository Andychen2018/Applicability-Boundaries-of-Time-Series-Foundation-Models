#!/usr/bin/env python3
"""
方案B：Normal-only微调 + embedding特征多分类
仅用normal数据微调Chronos，然后抽取embedding特征进行三分类
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
from sklearn.manifold import TSNE
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_chronos_data(df, prediction_length=1024, context_length=2048):
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

def extract_embeddings(predictor, data_df, context_length=2048):
    """
    从Chronos模型中提取embedding特征
    """
    embeddings_list = []
    item_ids_list = []
    labels_list = []
    
    for item_id in data_df['item_id'].unique():
        item_data = data_df[data_df['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= context_length:
            # 取前context_length个点作为输入
            context_data = item_data.iloc[:context_length].copy()
            
            # 转换为Chronos格式
            chronos_data = TimeSeriesDataFrame(context_data)
            
            try:
                # 这里需要访问模型内部来获取embedding
                # 由于AutoGluon的限制，我们先用预测结果作为特征
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

                # 确保pred_values是numpy数组
                if not isinstance(pred_values, np.ndarray):
                    pred_values = np.array(pred_values)
                
                # 使用预测值的统计特征作为embedding的替代
                embedding_features = {
                    'pred_mean': np.mean(pred_values),
                    'pred_std': np.std(pred_values),
                    'pred_min': np.min(pred_values),
                    'pred_max': np.max(pred_values),
                    'pred_median': np.median(pred_values),
                    'pred_q25': np.percentile(pred_values, 25),
                    'pred_q75': np.percentile(pred_values, 75),
                    'pred_skew': pd.Series(pred_values).skew(),
                    'pred_kurt': pd.Series(pred_values).kurtosis()
                }
                
                # 添加原始数据的统计特征
                original_values = context_data['target'].values
                embedding_features.update({
                    'orig_mean': np.mean(original_values),
                    'orig_std': np.std(original_values),
                    'orig_min': np.min(original_values),
                    'orig_max': np.max(original_values),
                    'orig_median': np.median(original_values),
                    'orig_q25': np.percentile(original_values, 25),
                    'orig_q75': np.percentile(original_values, 75),
                    'orig_skew': pd.Series(original_values).skew(),
                    'orig_kurt': pd.Series(original_values).kurtosis()
                })
                
                # 添加频域特征
                try:
                    fft_values = np.fft.fft(original_values)
                    freqs = np.fft.fftfreq(len(original_values))
                    
                    # 低频和高频能量
                    low_freq_mask = np.abs(freqs) < 0.1
                    high_freq_mask = np.abs(freqs) >= 0.1
                    
                    low_freq_energy = np.sum(np.abs(fft_values[low_freq_mask]) ** 2)
                    high_freq_energy = np.sum(np.abs(fft_values[high_freq_mask]) ** 2)
                    
                    embedding_features.update({
                        'low_freq_energy': low_freq_energy,
                        'high_freq_energy': high_freq_energy,
                        'freq_ratio': high_freq_energy / (low_freq_energy + 1e-8)
                    })
                except:
                    embedding_features.update({
                        'low_freq_energy': 0,
                        'high_freq_energy': 0,
                        'freq_ratio': 0
                    })
                
                embeddings_list.append(embedding_features)
                item_ids_list.append(item_id)
                labels_list.append(item_data['label'].iloc[0])
                
            except Exception as e:
                print(f"提取 {item_id} 的embedding时出错: {str(e)}")
                continue
    
    return embeddings_list, item_ids_list, labels_list

def train_method_b_single_domain(domain, output_dir, prediction_length=1024, context_length=2048):
    """
    训练单个域的方案B模型
    """
    print(f"\n=== 开始训练 {domain} 域的方案B模型 ===")
    
    # 创建输出目录
    domain_output_dir = os.path.join(output_dir, f"methodB_embed_normal_ft/{domain}")
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
    
    # 3. 对所有数据提取embedding特征
    print("开始提取embedding特征...")
    
    # 合并所有数据用于特征提取
    all_data = pd.concat([train_domain, val_domain, test_domain], ignore_index=True)
    
    embeddings_list, item_ids_list, labels_list = extract_embeddings(predictor, all_data, context_length)
    
    # 转换为DataFrame
    embeddings_df = pd.DataFrame(embeddings_list)
    embeddings_df['item_id'] = item_ids_list
    embeddings_df['label'] = labels_list
    
    print(f"成功提取 {len(embeddings_df)} 个序列的embedding特征")
    
    # 4. 划分特征数据
    train_embeddings = embeddings_df[embeddings_df['item_id'].isin(train_domain['item_id'].unique())]
    val_embeddings = embeddings_df[embeddings_df['item_id'].isin(val_domain['item_id'].unique())]
    test_embeddings = embeddings_df[embeddings_df['item_id'].isin(test_domain['item_id'].unique())]
    
    # 保存embedding数据
    train_embeddings.to_csv(os.path.join(domain_output_dir, "embeddings_train.csv"), index=False)
    val_embeddings.to_csv(os.path.join(domain_output_dir, "embeddings_val.csv"), index=False)
    test_embeddings.to_csv(os.path.join(domain_output_dir, "embeddings_test.csv"), index=False)
    
    # 5. 训练分类器
    print("开始训练分类器...")

    # 准备特征和标签
    feature_cols = [col for col in embeddings_df.columns if col not in ['item_id', 'label']]

    X_train = train_embeddings[feature_cols].fillna(0)
    y_train = train_embeddings['label']
    X_val = val_embeddings[feature_cols].fillna(0)
    y_val = val_embeddings['label']
    X_test = test_embeddings[feature_cols].fillna(0)
    y_test = test_embeddings['label']

    print(f"训练集特征形状: {X_train.shape}")
    print(f"验证集特征形状: {X_val.shape}")
    print(f"测试集特征形状: {X_test.shape}")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 训练多个分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
    }

    best_classifier = None
    best_score = 0
    best_clf_name = None
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
    if best_classifier is not None:
        print(f"\n最佳分类器: {best_clf_name} (验证集准确率: {best_score:.4f})")

        # 在测试集上评估
        if best_clf_name == 'LightGBM':
            test_pred = best_classifier.predict(X_test)
        else:
            test_pred = best_classifier.predict(X_test_scaled)

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
            'method': 'B_embed_normal_ft',
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
        test_embeddings['label'].value_counts().plot(kind='bar')
        plt.title(f'{domain} - Test Set Label Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(domain_output_dir, "figures", "evaluation.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, test_pred)}")

        return True
    else:
        print("没有成功训练任何分类器")
        return False

def main():
    # 设置参数
    prediction_length = 1024
    context_length = 2048  # 使用2048避免超出限制
    output_dir = "/home/deep/TimeSeries/Zhendong/output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练各个域的模型
    domains = ["ZhenDong"]  # 只训练ZhenDong域，ShengYing已完成
    
    for domain in domains:
        try:
            predictor, embeddings_df = train_method_b_single_domain(
                domain, output_dir, prediction_length, context_length
            )
            print(f"{domain} 域方案B训练完成")
        except Exception as e:
            print(f"训练 {domain} 域时出错: {str(e)}")
            continue

if __name__ == "__main__":
    main()
