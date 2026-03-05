#!/usr/bin/env python3
"""
简单测试Chronos的embed功能
"""

import pandas as pd
import numpy as np
import torch

def test_chronos_versions():
    """测试不同的Chronos加载方式"""
    
    print("测试Chronos库...")
    
    try:
        from chronos import ChronosPipeline
        print("✅ 成功导入ChronosPipeline")
    except Exception as e:
        print(f"❌ 导入ChronosPipeline失败: {e}")
        return
    
    # 测试1: 尝试加载原始模型
    print("\n测试1: 加载原始chronos-bolt-base模型...")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base",
            device_map="cpu",  # 先用CPU测试
            torch_dtype=torch.float32,
        )
        print("✅ 成功加载原始模型!")
        
        # 测试embed功能
        print("测试embed功能...")
        test_data = torch.randn(100)  # 100个随机数据点

        try:
            embeddings, tokenizer_state = pipeline.embed(test_data)
            print(f"✅ Embed成功! Embeddings shape: {embeddings.shape}")
            print(f"   Tokenizer state keys: {list(tokenizer_state.keys()) if tokenizer_state else 'None'}")
            return pipeline

        except Exception as e:
            print(f"❌ Embed失败: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"❌ 加载原始模型失败: {e}")
        
        # 测试2: 尝试从HuggingFace加载
        print("\n测试2: 从HuggingFace加载...")
        try:
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            print("✅ 成功加载HuggingFace模型!")
            
            # 测试embed
            test_data = torch.randn(100)
            embeddings, tokenizer_state = pipeline.embed(test_data)
            print(f"✅ Embed成功! Embeddings shape: {embeddings.shape}")
            return pipeline
            
        except Exception as e:
            print(f"❌ 加载HuggingFace模型也失败: {e}")
            return None

def test_real_data_embed(pipeline):
    """使用真实数据测试embed"""
    
    if pipeline is None:
        print("没有可用的pipeline")
        return
    
    print("\n测试真实数据embedding...")
    
    # 加载一个真实的时序数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    # 选择第一个序列
    first_item = zhendong_test['item_id'].iloc[0]
    item_data = zhendong_test[zhendong_test['item_id'] == first_item]
    
    print(f"测试序列: {first_item}")
    print(f"序列长度: {len(item_data)}")
    print(f"标签: {item_data['label'].iloc[0]}")
    
    # 测试不同长度的输入
    test_lengths = [64, 128, 256, 512, 1024, 2048]
    
    for length in test_lengths:
        if length <= len(item_data):
            print(f"\n测试长度: {length}")
            
            # 准备数据
            context_data = item_data.iloc[:length]['target'].values
            context = torch.tensor(context_data, dtype=torch.float32)
            
            try:
                # 提取embeddings
                embeddings, tokenizer_state = pipeline.embed(context)
                
                print(f"  ✅ 成功! Embeddings shape: {embeddings.shape}")
                print(f"  Embeddings dtype: {embeddings.dtype}")
                print(f"  Embeddings mean: {embeddings.mean():.6f}")
                print(f"  Embeddings std: {embeddings.std():.6f}")
                
                # 如果是2D，显示更多信息
                if len(embeddings.shape) == 2:
                    seq_len, hidden_dim = embeddings.shape
                    print(f"  序列长度: {seq_len}, 隐藏维度: {hidden_dim}")
                    
                    # 计算不同的池化结果
                    mean_pool = embeddings.mean(dim=0)
                    max_pool = embeddings.max(dim=0)[0]
                    last_token = embeddings[-1]
                    
                    print(f"  Mean pooling shape: {mean_pool.shape}")
                    print(f"  Max pooling shape: {max_pool.shape}")
                    print(f"  Last token shape: {last_token.shape}")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")

def main():
    print("开始Chronos embed功能测试...")
    
    # 测试基本功能
    pipeline = test_chronos_versions()
    
    # 测试真实数据
    test_real_data_embed(pipeline)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
