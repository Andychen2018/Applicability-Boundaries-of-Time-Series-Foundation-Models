#!/usr/bin/env python3
"""
测试Chronos原生的embed方法，了解embedding维度和提取方式
"""

import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline

def test_chronos_embed():
    """测试Chronos的embed方法"""
    
    print("测试Chronos原生embed方法...")
    
    # 加载预训练模型
    pipeline = ChronosPipeline.from_pretrained(
        "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    # 测试不同长度的输入
    test_lengths = [512, 1024, 2048, 4096, 8192]
    
    # 选择一个测试序列
    sample_item = zhendong_test['item_id'].iloc[0]
    sample_data = zhendong_test[zhendong_test['item_id'] == sample_item]['target'].values
    
    print(f"测试序列: {sample_item}")
    print(f"序列长度: {len(sample_data)}")
    print(f"标签: {zhendong_test[zhendong_test['item_id'] == sample_item]['label'].iloc[0]}")
    
    for length in test_lengths:
        if length <= len(sample_data):
            print(f"\n测试输入长度: {length}")
            
            # 准备输入数据
            context = torch.tensor(sample_data[:length], dtype=torch.float32)
            
            try:
                # 提取embeddings
                embeddings, tokenizer_state = pipeline.embed(context)
                
                print(f"  Embeddings shape: {embeddings.shape}")
                print(f"  Embeddings dtype: {embeddings.dtype}")
                print(f"  Tokenizer state keys: {list(tokenizer_state.keys()) if tokenizer_state else 'None'}")
                
                # 检查embeddings的统计信息
                print(f"  Embeddings mean: {embeddings.mean():.6f}")
                print(f"  Embeddings std: {embeddings.std():.6f}")
                print(f"  Embeddings min: {embeddings.min():.6f}")
                print(f"  Embeddings max: {embeddings.max():.6f}")
                
            except Exception as e:
                print(f"  错误: {e}")
    
    return pipeline

def test_multiple_sequences(pipeline, max_sequences=5):
    """测试多个序列的embedding提取"""
    
    print(f"\n测试多个序列的embedding提取...")
    
    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    unique_items = zhendong_test['item_id'].unique()[:max_sequences]
    
    embeddings_list = []
    labels_list = []
    
    for item_id in unique_items:
        item_data = zhendong_test[zhendong_test['item_id'] == item_id]
        target_values = item_data['target'].values
        label = item_data['label'].iloc[0]
        
        # 使用固定长度
        context_length = 2048
        if len(target_values) >= context_length:
            context = torch.tensor(target_values[:context_length], dtype=torch.float32)
            
            try:
                embeddings, _ = pipeline.embed(context)
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(label)
                
                print(f"{item_id}: {embeddings.shape}, label: {label}")
                
            except Exception as e:
                print(f"{item_id}: 错误 - {e}")
    
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        print(f"\n所有embeddings shape: {embeddings_array.shape}")
        print(f"标签: {labels_list}")
        
        # 检查不同标签的embedding差异
        labels_array = np.array(labels_list)
        for label in np.unique(labels_array):
            label_embeddings = embeddings_array[labels_array == label]
            print(f"{label} embeddings mean: {label_embeddings.mean(axis=0)[:5]}...")  # 只显示前5维
    
    return embeddings_list, labels_list

def compare_context_lengths():
    """比较不同context长度对embedding的影响"""
    
    print(f"\n比较不同context长度的影响...")
    
    # 加载模型
    pipeline = ChronosPipeline.from_pretrained(
        "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # 加载一个测试序列
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    sample_item = zhendong_test['item_id'].iloc[0]
    sample_data = zhendong_test[zhendong_test['item_id'] == sample_item]['target'].values
    
    # 测试不同的context长度
    context_lengths = [512, 1024, 2048]
    
    for length in context_lengths:
        if length <= len(sample_data):
            context = torch.tensor(sample_data[:length], dtype=torch.float32)
            
            try:
                embeddings, _ = pipeline.embed(context)
                print(f"Context length {length}: embedding shape {embeddings.shape}")
                
                # 如果embedding是2D的，计算每个位置的统计
                if len(embeddings.shape) == 2:
                    print(f"  每个token的embedding维度: {embeddings.shape[1]}")
                    print(f"  Token数量: {embeddings.shape[0]}")
                    
                    # 可以尝试不同的聚合方式
                    mean_embedding = embeddings.mean(dim=0)
                    max_embedding = embeddings.max(dim=0)[0]
                    last_embedding = embeddings[-1]
                    
                    print(f"  Mean pooling shape: {mean_embedding.shape}")
                    print(f"  Max pooling shape: {max_embedding.shape}")
                    print(f"  Last token shape: {last_embedding.shape}")
                
            except Exception as e:
                print(f"Context length {length}: 错误 - {e}")

if __name__ == "__main__":
    # 测试基本的embed功能
    pipeline = test_chronos_embed()
    
    # 测试多个序列
    embeddings_list, labels_list = test_multiple_sequences(pipeline)
    
    # 比较不同context长度
    compare_context_lengths()
