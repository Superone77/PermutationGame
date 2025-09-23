#!/usr/bin/env python3
"""
Simple test for q_proj and k_proj quadruple reordering
"""

import torch
import numpy as np
from reorder_algorithms import qk_proj_quadruple_reorder_index, compute_reorders

def test_qk_quadruple():
    """Test the quadruple reordering for q_proj and k_proj"""
    print("Testing q_proj and k_proj quadruple reordering...")
    
    # 创建测试数据
    D = 128
    block_size = 16
    K = D // block_size
    
    # 生成 q_proj 和 k_proj 的统计信息
    q_xmax = torch.randn(D) * 2.0
    q_xmin = torch.randn(D) * 1.5
    k_xmax = torch.randn(D) * 1.8
    k_xmin = torch.randn(D) * 1.2
    
    print(f"Data shape: {D}, Block size: {block_size}, Number of blocks: {K}")
    
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
    
    print("✓ Quadruple reordering test passed!")

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
