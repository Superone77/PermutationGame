#!/usr/bin/env python3

import torch
import numpy as np

def test_abs_logic():
    """测试绝对值逻辑"""
    print("Testing absolute value logic...")
    
    # 创建测试数据
    xmax = torch.tensor([5.0, -3.0, 8.0, -1.0])
    xmin = torch.tensor([2.0, -6.0, 4.0, -4.0])
    
    print(f"xmax: {xmax}")
    print(f"xmin: {xmin}")
    
    # 计算绝对最大值
    abs_max = torch.max(torch.abs(xmax), torch.abs(xmin))
    print(f"abs_max: {abs_max}")
    
    # 验证逻辑
    expected = torch.tensor([5.0, 6.0, 8.0, 4.0])  # max(5,2), max(3,6), max(8,4), max(1,4)
    print(f"expected: {expected}")
    print(f"Match: {torch.allclose(abs_max, expected)}")
    
    # 测试排序
    idx_sorted = torch.argsort(abs_max, descending=True)
    print(f"Sorted indices: {idx_sorted}")
    print(f"Sorted values: {abs_max[idx_sorted]}")
    
    print("✓ Logic test passed!")

if __name__ == "__main__":
    test_abs_logic()
