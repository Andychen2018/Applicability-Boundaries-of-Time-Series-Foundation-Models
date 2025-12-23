#!/usr/bin/env python3
"""
数据划分脚本 - 按motor_id分组进行train/val/test划分
确保同一motor_id不会同时出现在训练和测试中
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from pathlib import Path

def extract_motor_id(item_id):
    """从item_id中提取motor_id"""
    # item_id格式: ShengYing_normal_a_normal_16_09
    # 提取最后的数字部分作为motor_id
    parts = item_id.split('_')
    if len(parts) >= 2:
        # 取最后两部分作为motor_id
        return '_'.join(parts[-2:])
    return item_id

def split_data_by_motor_id(df, test_size=0.2, val_size=0.1, random_state=123):
    """
    按motor_id分组进行数据划分
    """
    # 提取motor_id
    df['motor_id'] = df['item_id'].apply(extract_motor_id)
    
    # 获取唯一的motor_id和对应的域信息
    motor_info = df.groupby('motor_id').agg({
        'item_id': 'first',  # 取第一个item_id来判断域
        'label': lambda x: list(x.unique())  # 获取该motor_id的所有标签
    }).reset_index()
    
    # 提取域信息
    motor_info['domain'] = motor_info['item_id'].apply(lambda x: x.split('_')[0])
    
    print(f"总共有 {len(motor_info)} 个唯一的motor_id")
    print(f"域分布: {motor_info['domain'].value_counts().to_dict()}")
    
    # 分别对每个域进行划分，确保每个域在train/val/test中都有代表
    train_motors, val_motors, test_motors = [], [], []
    
    for domain in motor_info['domain'].unique():
        domain_motors = motor_info[motor_info['domain'] == domain]['motor_id'].tolist()
        print(f"\n{domain} 域有 {len(domain_motors)} 个motor_id")
        
        # 先分出test集
        train_val_motors, test_motors_domain = train_test_split(
            domain_motors, test_size=test_size, random_state=random_state
        )
        
        # 再从train_val中分出val集
        train_motors_domain, val_motors_domain = train_test_split(
            train_val_motors, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        train_motors.extend(train_motors_domain)
        val_motors.extend(val_motors_domain)
        test_motors.extend(test_motors_domain)
        
        print(f"  训练集: {len(train_motors_domain)} 个motor_id")
        print(f"  验证集: {len(val_motors_domain)} 个motor_id") 
        print(f"  测试集: {len(test_motors_domain)} 个motor_id")
    
    # 根据motor_id划分数据
    train_df = df[df['motor_id'].isin(train_motors)].copy()
    val_df = df[df['motor_id'].isin(val_motors)].copy()
    test_df = df[df['motor_id'].isin(test_motors)].copy()
    
    print(f"\n最终数据划分:")
    print(f"训练集: {len(train_df)} 条记录, {train_df['item_id'].nunique()} 个序列")
    print(f"验证集: {len(val_df)} 条记录, {val_df['item_id'].nunique()} 个序列")
    print(f"测试集: {len(test_df)} 条记录, {test_df['item_id'].nunique()} 个序列")
    
    # 检查标签分布
    for split_name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        print(f"\n{split_name}标签分布:")
        print(split_df['label'].value_counts())
        print(f"{split_name}域分布:")
        split_df['domain'] = split_df['item_id'].apply(lambda x: x.split('_')[0])
        print(split_df['domain'].value_counts())
    
    return train_df, val_df, test_df

def main():
    # 设置路径
    data_path = "/home/deep/TimeSeries/Zhendong/data3/processed_motor_data.csv"
    output_dir = "/home/deep/TimeSeries/Zhendong/output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("读取数据...")
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 数据划分
    print("\n开始数据划分...")
    train_df, val_df, test_df = split_data_by_motor_id(df)
    
    # 保存划分后的数据
    print("\n保存划分后的数据...")
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    
    # 保存划分信息
    split_info = {
        "train_motor_ids": train_df['motor_id'].unique().tolist(),
        "val_motor_ids": val_df['motor_id'].unique().tolist(), 
        "test_motor_ids": test_df['motor_id'].unique().tolist(),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_sequences": train_df['item_id'].nunique(),
        "val_sequences": val_df['item_id'].nunique(),
        "test_sequences": test_df['item_id'].nunique()
    }
    
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n数据划分完成！文件保存在: {output_dir}")
    print("- train_data.csv: 训练集")
    print("- val_data.csv: 验证集") 
    print("- test_data.csv: 测试集")
    print("- split_info.json: 划分信息")

if __name__ == "__main__":
    main()
