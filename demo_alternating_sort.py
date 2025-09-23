#!/usr/bin/env python3
"""
Demo script to show the alternating sort strategy for q_proj and k_proj
"""

import torch
import numpy as np
from reorder_algorithms import qk_proj_quadruple_reorder_index

def demo_alternating_sort():
    """Demonstrate the alternating sort strategy"""
    print("=== QK Proj Alternating Sort Demo ===\n")
    
    # 创建测试数据，故意让q和k有不同的分布
    D = 64
    block_size = 8
    K = D // block_size
    
    # 生成有明显差异的q和k数据
    torch.manual_seed(42)  # 固定随机种子以便复现
    q_xmax = torch.randn(D) * 4.0 + 2.0  # q: 范围更大，均值更高
    q_xmin = torch.randn(D) * 3.0 + 1.0
    k_xmax = torch.randn(D) * 1.0 + 0.5  # k: 范围较小，均值较低
    k_xmin = torch.randn(D) * 0.8 + 0.2
    
    print(f"Data shape: {D}, Block size: {block_size}, Number of blocks: {K}")
    print(f"Q range: [{q_xmin.min():.3f}, {q_xmax.max():.3f}]")
    print(f"K range: [{k_xmin.min():.3f}, {k_xmax.max():.3f}]")
    
    # 计算绝对最大值
    q_abs_max = torch.max(torch.abs(q_xmax), torch.abs(q_xmin))
    k_abs_max = torch.max(torch.abs(k_xmax), torch.abs(k_xmin))
    
    print(f"\nQ absolute max: {q_abs_max.max():.3f}")
    print(f"K absolute max: {k_abs_max.max():.3f}")
    
    # 测试四元数重排序
    print("\n=== Running Quadruple Reordering with Alternating Sort ===")
    idx, counts = qk_proj_quadruple_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=0.25)
    
    print(f"Generated index shape: {idx.shape}")
    print(f"Top {int(K * 0.25)} blocks will be kept for special treatment")
    
    # 分析每个block的内容
    print("\n=== Block Analysis ===")
    for block_idx in range(K):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size
        block_indices = idx[start_idx:end_idx]
        
        # 计算这个block中q和k的最大值
        block_q_max = q_abs_max[block_indices].max()
        block_k_max = k_abs_max[block_indices].max()
        
        # 判断这个block是按q还是k排序的
        sort_type = "K-sorted" if block_idx % 2 == 0 else "Q-sorted"
        
        print(f"Block {block_idx:2d} ({sort_type:9s}): "
              f"Q_max={block_q_max:.3f}, K_max={block_k_max:.3f}, "
              f"indices={block_indices.tolist()}")
    
    # 验证交替排序的效果
    print("\n=== Verification ===")
    q_max_found_in_q_blocks = False
    k_max_found_in_k_blocks = False
    
    for block_idx in range(K):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size
        block_indices = idx[start_idx:end_idx]
        
        block_q_max = q_abs_max[block_indices].max()
        block_k_max = k_abs_max[block_indices].max()
        
        if block_idx % 2 == 0:  # K-sorted block
            if block_k_max == k_abs_max.max():
                k_max_found_in_k_blocks = True
        else:  # Q-sorted block
            if block_q_max == q_abs_max.max():
                q_max_found_in_q_blocks = True
    
    print(f"Q maximum found in Q-sorted blocks: {q_max_found_in_q_blocks}")
    print(f"K maximum found in K-sorted blocks: {k_max_found_in_k_blocks}")
    
    if q_max_found_in_q_blocks and k_max_found_in_k_blocks:
        print("✓ Alternating sort strategy working correctly!")
    else:
        print("✗ Alternating sort strategy may need adjustment")
    
    # 显示前几个block的详细内容
    print("\n=== Top Blocks Detail ===")
    top_blocks = min(4, K)
    for block_idx in range(top_blocks):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size
        block_indices = idx[start_idx:end_idx]
        
        sort_type = "K-sorted" if block_idx % 2 == 0 else "Q-sorted"
        print(f"\nBlock {block_idx} ({sort_type}):")
        print(f"  Indices: {block_indices.tolist()}")
        print(f"  Q values: {q_abs_max[block_indices].tolist()}")
        print(f"  K values: {k_abs_max[block_indices].tolist()}")

if __name__ == '__main__':
    demo_alternating_sort()
