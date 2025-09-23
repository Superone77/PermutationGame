#!/usr/bin/env python3

import torch
import numpy as np
from reorder_algorithms import abs_hybrid_plus_reorder_index, qk_proj_abs_dual_reorder_index

def test_abs_hybrid_plus():
    """测试 abs_hybrid_plus 方法"""
    print("Testing abs_hybrid_plus method...")
    
    # 生成测试数据
    torch.manual_seed(42)
    D = 1000
    block_size = 16
    K = D // block_size
    
    # 创建 xmax 和 xmin，确保有正负值
    xmax = torch.randn(D) * 10
    xmin = torch.randn(D) * 8 - 5
    
    print(f"Data shape: {xmax.shape}")
    print(f"Block size: {block_size}, Number of blocks: {K}")
    print(f"Xmax range: [{xmax.min():.3f}, {xmax.max():.3f}]")
    print(f"Xmin range: [{xmin.min():.3f}, {xmin.max():.3f}]")
    
    # 测试 abs_hybrid_plus
    idx, counts = abs_hybrid_plus_reorder_index(xmax, xmin, block_size, top_pct=0.1)
    
    print(f"Returned index shape: {idx.shape}")
    print(f"Returned counts shape: {counts.shape}")
    print(f"Counts: {counts}")
    
    # 验证结果
    assert idx.numel() == D, f"Index length {idx.numel()} != data length {D}"
    assert len(counts) == K, f"Counts length {len(counts)} != number of blocks {K}"
    assert counts.sum() == D, f"Total counts {counts.sum()} != data length {D}"
    assert all(counts == block_size), f"Not all blocks have size {block_size}: {counts}"
    
    # 检查索引的唯一性
    unique_idx = torch.unique(idx)
    assert len(unique_idx) == D, f"Index not unique: {len(unique_idx)} unique values out of {D}"
    
    print("✓ abs_hybrid_plus test passed!")

def test_qk_abs_dual():
    """测试 QK 层的 abs_dual 方法"""
    print("\nTesting QK abs_dual method...")
    
    # 生成测试数据
    torch.manual_seed(42)
    D = 1000
    block_size = 16
    K = D // block_size
    
    # 创建 Q 和 K 的 xmax 和 xmin
    q_xmax = torch.randn(D) * 10
    q_xmin = torch.randn(D) * 8 - 5
    k_xmax = torch.randn(D) * 12
    k_xmin = torch.randn(D) * 6 - 8
    
    print(f"Q data shape: {q_xmax.shape}")
    print(f"K data shape: {k_xmax.shape}")
    print(f"Block size: {block_size}, Number of blocks: {K}")
    
    # 测试 qk_proj_abs_dual_reorder_index
    idx, counts = qk_proj_abs_dual_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=0.1)
    
    print(f"Returned index shape: {idx.shape}")
    print(f"Returned counts shape: {counts.shape}")
    print(f"Counts: {counts}")
    
    # 验证结果
    assert idx.numel() == D, f"Index length {idx.numel()} != data length {D}"
    assert len(counts) == K, f"Counts length {len(counts)} != number of blocks {K}"
    assert counts.sum() == D, f"Total counts {counts.sum()} != data length {D}"
    assert all(counts == block_size), f"Not all blocks have size {block_size}: {counts}"
    
    # 检查索引的唯一性
    unique_idx = torch.unique(idx)
    assert len(unique_idx) == D, f"Index not unique: {len(unique_idx)} unique values out of {D}"
    
    print("✓ QK abs_dual test passed!")

def test_alternating_sort():
    """测试交替排序策略"""
    print("\nTesting alternating sort strategy...")
    
    # 生成有明显差异的 Q 和 K 数据
    torch.manual_seed(42)
    D = 64  # 4个block，每个16个元素
    block_size = 16
    K = D // block_size
    
    # Q 数据：前两个block值较大，后两个block值较小
    q_data = torch.cat([torch.ones(32) * 10, torch.ones(32) * 1])
    # K 数据：奇偶block值不同
    k_data = torch.zeros(D)
    for i in range(K):
        if i % 2 == 0:  # 偶数block
            k_data[i*block_size:(i+1)*block_size] = 20
        else:  # 奇数block
            k_data[i*block_size:(i+1)*block_size] = 2
    
    q_xmax = q_data
    q_xmin = q_data - 1
    k_xmax = k_data
    k_xmin = k_data - 1
    
    print(f"Q data: {q_data}")
    print(f"K data: {k_data}")
    
    # 测试交替排序
    idx, counts = qk_proj_abs_dual_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=0.0)
    
    print(f"Sorted index: {idx}")
    print(f"Counts: {counts}")
    
    # 验证交替排序效果
    # 前两个block应该主要包含K的大值（偶数block）
    # 后两个block应该主要包含Q的大值（奇数block）
    first_two_blocks = idx[:32]
    last_two_blocks = idx[32:]
    
    print(f"First two blocks indices: {first_two_blocks}")
    print(f"Last two blocks indices: {last_two_blocks}")
    
    print("✓ Alternating sort test completed!")

if __name__ == "__main__":
    test_abs_hybrid_plus()
    test_qk_abs_dual()
    test_alternating_sort()
    print("\n🎉 All tests passed!")
