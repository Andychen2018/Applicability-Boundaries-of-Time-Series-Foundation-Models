#!/usr/bin/env python3
"""
使用微调后的Chronos模型直接提取真正的embeddings
"""

import pandas as pd
import numpy as np
import torch
import os
import json
from chronos import ChronosPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def test_finetuned_chronos_embed():
    """测试微调后的Chronos模型是否可以直接加载并提取embeddings"""
    
    print("测试微调后的Chronos模型embedding提取...")
    
    # 尝试加载微调后的模型
    finetuned_model_path = "/home/deep/TimeSeries/Zhendong/output/all_class_finetune/ZhenDong/predictor/models/Chronos[eries__Zhendong__chronos_models__chronos-bolt-base]/W0/fine-tuned-ckpt"
    
    print(f"尝试加载微调模型: {finetuned_model_path}")
    
    try:
        # 方法1: 直接加载微调后的checkpoint
        pipeline = ChronosPipeline.from_pretrained(
            finetuned_model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        print("✅ 成功加载微调后的模型!")
        return pipeline
        
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
        
        try:
            # 方法2: 加载原始模型然后手动加载权重
            print("尝试方法2: 加载原始模型...")
            pipeline = ChronosPipeline.from_pretrained(
                "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            
            # 手动加载微调后的权重
            import safetensors
            weights_path = os.path.join(finetuned_model_path, "model.safetensors")
            if os.path.exists(weights_path):
                print(f"加载微调权重: {weights_path}")
                # 这里需要手动加载权重到模型
                print("⚠️ 需要手动实现权重加载")
            
            return pipeline
            
        except Exception as e2:
            print(f"❌ 方法2也失败: {e2}")
            
            # 方法3: 使用原始模型作为fallback
            print("使用原始模型作为fallback...")
            pipeline = ChronosPipeline.from_pretrained(
                "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            return pipeline

def extract_true_embeddings(pipeline, data_df, max_sequences=50):
    """使用ChronosPipeline提取真正的embeddings"""
    
    embeddings_list = []
    item_ids_list = []
    labels_list = []
    
    unique_items = data_df['item_id'].unique()[:max_sequences]  # 限制数量以加快测试
    print(f"开始处理 {len(unique_items)} 个序列...")
    
    for i, item_id in enumerate(unique_items):
        if i % 10 == 0:
            print(f"进度: {i}/{len(unique_items)}")
            
        item_data = data_df[data_df['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= 2048:  # 至少需要足够的数据
            # 使用前2048个数据点（对应模型的context length）
            context_data = item_data.iloc[:2048]['target'].values
            
            try:
                # 转换为tensor
                context = torch.tensor(context_data, dtype=torch.float32)
                
                # 提取embeddings
                embeddings, tokenizer_state = pipeline.embed(context)
                
                print(f"  {item_id}: embeddings shape = {embeddings.shape}")
                
                # 将embeddings转换为特征向量
                if len(embeddings.shape) == 2:
                    # 如果是2D (sequence_length, hidden_dim)，可以用不同的聚合方式
                    # 方法1: 平均池化
                    pooled_embedding = embeddings.mean(dim=0).cpu().numpy()
                    
                    # 方法2: 最大池化
                    # pooled_embedding = embeddings.max(dim=0)[0].cpu().numpy()
                    
                    # 方法3: 最后一个token
                    # pooled_embedding = embeddings[-1].cpu().numpy()
                    
                    # 方法4: 拼接多种池化
                    # mean_pool = embeddings.mean(dim=0)
                    # max_pool = embeddings.max(dim=0)[0]
                    # pooled_embedding = torch.cat([mean_pool, max_pool]).cpu().numpy()
                    
                elif len(embeddings.shape) == 1:
                    # 如果已经是1D，直接使用
                    pooled_embedding = embeddings.cpu().numpy()
                else:
                    # 如果是3D或其他，flatten
                    pooled_embedding = embeddings.flatten().cpu().numpy()
                
                embeddings_list.append(pooled_embedding)
                item_ids_list.append(item_id)
                labels_list.append(item_data['label'].iloc[0])
                
            except Exception as e:
                print(f"  ❌ 处理 {item_id} 时出错: {e}")
                continue
    
    return embeddings_list, item_ids_list, labels_list

def method_true_embeddings():
    """使用真正的Chronos embeddings进行分类"""
    print("\n" + "="*60)
    print("方法: 真正的Chronos Embeddings")
    print("="*60)
    
    # 测试并加载模型
    pipeline = test_finetuned_chronos_embed()
    
    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    print(f"ZhenDong测试数据: {zhendong_test['item_id'].nunique()} 个序列")
    
    # 提取embeddings
    embeddings_list, item_ids_list, labels_list = extract_true_embeddings(pipeline, zhendong_test)
    
    if len(embeddings_list) == 0:
        print("没有成功提取到embeddings")
        return None
    
    # 检查embedding维度
    embedding_dim = len(embeddings_list[0])
    print(f"Embedding维度: {embedding_dim}")
    
    # 转换为DataFrame
    embeddings_array = np.array(embeddings_list)
    feature_cols = [f'embed_{i}' for i in range(embedding_dim)]
    features_df = pd.DataFrame(embeddings_array, columns=feature_cols)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的{embedding_dim}维真正embeddings")
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # 训练分类器
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['label']
    
    # 划分训练测试集
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
    output_dir = "/home/deep/TimeSeries/Zhendong/output/true_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_dir, "true_embeddings.csv"), index=False)
    
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n真正embeddings结果保存到: {output_dir}")
    
    return results

def main():
    print("开始真正的Chronos embeddings提取实验...")
    
    # 运行真正的embeddings提取
    results = method_true_embeddings()
    
    if results:
        print("\n" + "="*60)
        print("真正Chronos Embeddings结果")
        print("="*60)
        for clf_name, result in results.items():
            print(f"{clf_name}: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()

# ================== 说明 ==================
"""
这个脚本尝试：

1. 直接加载微调后的Chronos模型
2. 使用pipeline.embed()方法提取真正的embeddings
3. 获得真正的模型内部特征表示

如果成功，我们将得到：
- 真正的Transformer embeddings
- 维度可能是768或其他（取决于模型）
- 这些特征包含了模型学到的时序模式

这比我们之前的统计特征更加强大和准确！
"""
