#!/usr/bin/env python3
"""
简单测试embed功能，使用更小的数据
"""

import pandas as pd
import numpy as np
import torch

def test_simple_embed():
    """使用简单数据测试embed"""
    
    print("测试简单embed功能...")
    
    try:
        from chronos import ChronosPipeline
        print("✅ 成功导入ChronosPipeline")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 尝试加载原始模型
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        print("✅ 成功加载模型!")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    # 测试不同的数据
    test_cases = [
        ("小数据", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("中等数据", torch.randn(50)),
        ("正常数据", torch.randn(100)),
        ("长数据", torch.randn(512)),
    ]
    
    for name, data in test_cases:
        print(f"\n测试 {name} (长度: {len(data)})...")
        
        try:
            embeddings, tokenizer_state = pipeline.embed(data)
            print(f"  ✅ 成功! Embeddings shape: {embeddings.shape}")
            print(f"  Embeddings dtype: {embeddings.dtype}")
            print(f"  Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            
            if len(embeddings.shape) == 2:
                print(f"  序列长度: {embeddings.shape[0]}, 特征维度: {embeddings.shape[1]}")
            
            return pipeline, embeddings  # 返回成功的结果
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue
    
    return None, None

def test_real_data_simple(pipeline):
    """使用真实数据测试"""
    
    if pipeline is None:
        print("没有可用的pipeline")
        return
    
    print("\n测试真实数据...")
    
    # 加载真实数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    # 选择第一个序列
    first_item = zhendong_test['item_id'].iloc[0]
    item_data = zhendong_test[zhendong_test['item_id'] == first_item]
    
    print(f"测试序列: {first_item}")
    print(f"标签: {item_data['label'].iloc[0]}")
    
    # 测试不同长度
    test_lengths = [32, 64, 128, 256]
    
    for length in test_lengths:
        if length <= len(item_data):
            print(f"\n测试长度: {length}")
            
            # 准备数据
            context_data = item_data.iloc[:length]['target'].values
            
            # 标准化数据（重要！）
            context_data = (context_data - context_data.mean()) / (context_data.std() + 1e-8)
            context = torch.tensor(context_data, dtype=torch.float32)
            
            try:
                embeddings, tokenizer_state = pipeline.embed(context)
                print(f"  ✅ 成功! Embeddings shape: {embeddings.shape}")
                
                if len(embeddings.shape) == 2:
                    seq_len, hidden_dim = embeddings.shape
                    print(f"  序列长度: {seq_len}, 隐藏维度: {hidden_dim}")
                    
                    # 尝试不同的池化方式
                    mean_pool = embeddings.mean(dim=0)
                    print(f"  Mean pooling shape: {mean_pool.shape}")
                    print(f"  这就是我们要的特征向量!")
                    
                    return embeddings, mean_pool
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
    
    return None, None

def main():
    print("开始简单embed测试...")
    
    # 测试基本功能
    pipeline, sample_embeddings = test_simple_embed()
    
    # 测试真实数据
    if pipeline:
        real_embeddings, pooled_features = test_real_data_simple(pipeline)
        
        if pooled_features is not None:
            print(f"\n🎉 成功提取特征!")
            print(f"特征维度: {pooled_features.shape}")
            print(f"这就是我们想要的真正embeddings!")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
