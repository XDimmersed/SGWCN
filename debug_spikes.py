#!/usr/bin/env python3

import torch
import numpy as np
from src.models import SGWCNClassifier
from configs.sgwcn_config import get_debug_stable_config

def debug_spike_behavior():
    """详细调试脉冲行为"""
    print("🔍 调试脉冲神经元行为")
    print("=" * 50)
    
    config = get_debug_stable_config()
    model = SGWCNClassifier(
        hidden_dims=[32],  # 只用一层便于调试
        num_time_steps=config.num_time_steps,
        tau_mem=config.tau_mem,
        theta_pos=config.theta_pos,
        theta_neg=config.theta_neg,
        num_points=64  # 更少的点
    )
    
    # 创建简单输入
    batch_size = 2
    input_data = torch.randn(batch_size, 64, 3) * 0.5  # 较小的输入
    
    print(f"输入范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # 逐步分析
    model.eval()
    with torch.no_grad():
        analysis = model.forward_with_analysis(input_data)
        
        print(f"\n📊 脉冲统计:")
        for i, stats in enumerate(analysis['spike_statistics']):
            print(f"  层 {i+1}:")
            print(f"    正脉冲率: {stats['pos_firing_rate_mean']:.4f}")
            print(f"    负脉冲率: {stats['neg_firing_rate_mean']:.4f}")
            print(f"    总脉冲数: {stats['total_pos_spikes'] + stats['total_neg_spikes']}")
            print(f"    稀疏度: {stats['sparsity']:.4f}")
        
        # 检查最终输出
        logits = analysis['logits']
        probs = torch.softmax(logits, dim=1)
        
        print(f"\n📈 最终输出:")
        print(f"  Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"  最大概率: {probs.max():.4f}")
        print(f"  最小概率: {probs.min():.4f}")
        print(f"  预测类别: {logits.argmax(dim=1).tolist()}")
        
        # 检查是否所有输出都相似（说明没有学习）
        logit_std = logits.std(dim=1).mean()
        print(f"  输出标准差: {logit_std:.4f}")
        
        if logit_std < 0.1:
            print("⚠️  警告: 输出变化很小，模型可能没有有效学习")

def test_threshold_sensitivity():
    """测试阈值敏感性"""
    print("\n🎯 测试阈值敏感性")
    print("=" * 50)
    
    thresholds = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    
    for theta in thresholds:
        config = get_debug_stable_config()
        config.theta_pos = theta
        config.theta_neg = -theta
        
        model = SGWCNClassifier(
            hidden_dims=[32],
            num_time_steps=4,
            tau_mem=config.tau_mem,
            theta_pos=theta,
            theta_neg=-theta,
            num_points=64
        )
        
        input_data = torch.randn(2, 64, 3) * 0.5
        
        with torch.no_grad():
            analysis = model.forward_with_analysis(input_data)
            
            pos_rate = analysis['spike_statistics'][0]['pos_firing_rate_mean']
            neg_rate = analysis['spike_statistics'][0]['neg_firing_rate_mean']
            total_spikes = analysis['spike_statistics'][0]['total_pos_spikes'] + \
                          analysis['spike_statistics'][0]['total_neg_spikes']
            
            print(f"  阈值 ±{theta}: 正脉冲率={pos_rate:.3f}, 负脉冲率={neg_rate:.3f}, 总脉冲={total_spikes}")

def test_time_steps():
    """测试时间步数的影响"""
    print("\n⏰ 测试时间步数影响")
    print("=" * 50)
    
    time_steps = [2, 4, 8, 16, 32]
    
    for T in time_steps:
        config = get_debug_stable_config()
        
        model = SGWCNClassifier(
            hidden_dims=[32],
            num_time_steps=T,
            tau_mem=config.tau_mem,
            theta_pos=config.theta_pos,
            theta_neg=config.theta_neg,
            num_points=64
        )
        
        input_data = torch.randn(2, 64, 3) * 0.5
        
        with torch.no_grad():
            analysis = model.forward_with_analysis(input_data)
            
            logits = analysis['logits']
            logit_std = logits.std(dim=1).mean()
            energy = analysis['energy_consumption']
            
            print(f"  T={T:2d}: 输出标准差={logit_std:.4f}, 能耗={energy:.2e}J")

if __name__ == "__main__":
    debug_spike_behavior()
    test_threshold_sensitivity()
    test_time_steps() 