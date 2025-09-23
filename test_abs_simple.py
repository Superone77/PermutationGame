#!/usr/bin/env python3

import torch
import numpy as np

def test_abs_hybrid_plus_simple():
    """简单测试 abs_hybrid_plus 方法"""
    print("Testing abs_hybrid_plus method...")
    
    # 生成小规模测试数据
    torch.manual_seed(42)
    D = 32  # 2个block，每个16个元素
    block_size = 16
    K = D // block_size
    
    # 创建 xmax 和 xmin
    xmax = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.1, 
                         -0.1, -0.5, -1.0, -2.0, -4.0, -6.0, -8.0, -10.0,
                         15.0, 12.0, 9.0, 6.0, 3.0, 1.5, 0.8, 0.2,
                         -0.2, -0.8, -1.5, -3.0, -6.0, -9.0, -12.0, -15.0])
    xmin = xmax - 1.0
    
    print(f"Data shape: {xmax.shape}")
    print(f"Block size: {block_size}, Number of blocks: {K}")
    print(f"Xmax: {xmax}")
    print(f"Xmin: {xmin}")
    
    # 计算绝对最大值
    abs_max = torch.max(torch.abs(xmax), torch.abs(xmin))
    print(f"Abs max: {abs_max}")
    
    # 测试排序
    idx_abs_sorted = torch.argsort(abs_max, descending=True, stable=True)
    print(f"Sorted indices: {idx_abs_sorted}")
    print(f"Sorted abs_max: {abs_max[idx_abs_sorted]}")
    
    print("✓ Simple test completed!")

if __name__ == "__main__":
    test_abs_hybrid_plus_simple()
