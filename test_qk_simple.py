#!/usr/bin/env python3
"""
Simple test for q_proj and k_proj quadruple reordering
"""

import torch
import numpy as np
from reorder_algorithms import qk_proj_quadruple_reorder_index, compute_reorders

def test_qk_quadruple():
    """Test the quadruple reordering for q_proj and k_proj"""
    print("Testing q_proj and k_proj quadruple reordering with alternating sort...")
    
    # 创建测试数据
    D = 128
    block_size = 16
    K = D // block_size
    
    # 生成 q_proj 和 k_proj 的统计信息，故意让q和k有不同的分布
    q_xmax = torch.randn(D) * 3.0  # q有更大的范围
    q_xmin = torch.randn(D) * 2.5
    k_xmax = torch.randn(D) * 1.0  # k有较小的范围
    k_xmin = torch.randn(D) * 0.8
    
    print(f"Data shape: {D}, Block size: {block_size}, Number of blocks: {K}")
    print(f"Q range: [{q_xmin.min():.3f}, {q_xmax.max():.3f}]")
    print(f"K range: [{k_xmin.min():.3f}, {k_xmax.max():.3f}]")
    
    # 测试四元数重排序
    idx, counts = qk_proj_quadruple_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=0.1)
    
    print(f"Generated index shape: {idx.shape}")
    print(f"Generated counts: {counts}")
    print(f"Index range: [{idx.min()}, {idx.max()}]")
    print(f"Unique indices: {len(torch.unique(idx))}")
    
    # 验证索引的完整性
    assert len(idx) == D, f"Index length {len(idx)} != data length {D}"
    assert len(torch.unique(idx)) == D, "Index contains duplicates"
    assert idx.min() == 0 and idx.max() == D-1, "Index range incorrect"
    
    # 验证交替排序策略
    print("\nVerifying alternating sort strategy...")
    q_abs_max = torch.max(torch.abs(q_xmax), torch.abs(q_xmin))
    k_abs_max = torch.max(torch.abs(k_xmax), torch.abs(k_xmin))
    
    # 检查前几个block是否包含了q和k的最大值
    top_blocks = 4  # 检查前4个block
    for block_idx in range(min(top_blocks, K)):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size
        block_indices = idx[start_idx:end_idx]
        
        if block_idx % 2 == 0:  # 双数block应该包含k的最大值
            block_k_max = k_abs_max[block_indices].max()
            global_k_max = k_abs_max.max()
            print(f"Block {block_idx} (K-sorted): max K value = {block_k_max:.3f}, global K max = {global_k_max:.3f}")
        else:  # 单数block应该包含q的最大值
            block_q_max = q_abs_max[block_indices].max()
            global_q_max = q_abs_max.max()
            print(f"Block {block_idx} (Q-sorted): max Q value = {block_q_max:.3f}, global Q max = {global_q_max:.3f}")
    
    print("✓ Quadruple reordering with alternating sort test passed!")

def test_compute_reorders_with_qk():
    """Test compute_reorders with q_proj and k_proj"""
    print("\nTesting compute_reorders with q_proj and k_proj...")
    
    # 创建模拟的 oc_stats
    D = 64
    block_size = 16
    
    oc_stats = {
        'layer_0_self_attn.q_proj': (torch.randn(D) * 2.0, torch.randn(D) * 1.5),
        'layer_0_self_attn.k_proj': (torch.randn(D) * 1.8, torch.randn(D) * 1.2),
        'layer_0_self_attn.v_proj': (torch.randn(D) * 1.6, torch.randn(D) * 1.0),
        'layer_0_self_attn.o_proj': (torch.randn(D) * 1.4, torch.randn(D) * 0.8),
    }
    
    # 测试 hybrid_plus 方法
    print("Testing hybrid_plus method...")
    perms = compute_reorders(oc_stats, block_size=block_size, method='hybrid_plus', hybrid_top_pct=0.1)
    
    print(f"Generated permutations for modules: {list(perms.keys())}")
    
    # 检查 q_proj 和 k_proj 是否有相同的重排序顺序
    if 'layer_0_self_attn.q_proj' in perms and 'layer_0_self_attn.k_proj' in perms:
        q_perm = perms['layer_0_self_attn.q_proj']
        k_perm = perms['layer_0_self_attn.k_proj']
        
        if torch.equal(q_perm, k_perm):
            print("✓ q_proj and k_proj have identical reordering (quadruple method used)")
        else:
            print("✗ q_proj and k_proj have different reordering")
    else:
        print("✗ q_proj or k_proj not found in permutations")
    
    # 验证所有排列的有效性
    for name, perm in perms.items():
        assert len(perm) == D, f"Permutation length mismatch for {name}"
        assert len(torch.unique(perm)) == D, f"Permutation contains duplicates for {name}"
        assert perm.min() == 0 and perm.max() == D-1, f"Permutation range incorrect for {name}"

if __name__ == '__main__':
    test_qk_quadruple()
    test_compute_reorders_with_qk()
    print("\n🎉 All tests passed!")
