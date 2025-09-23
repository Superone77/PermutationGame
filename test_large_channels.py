#!/usr/bin/env python3
"""
Test script for K-means optimization with large channel counts
"""

import torch
import numpy as np
import time
from reorder_algorithms import balanced_kmeans2d_calc_reorder_index, qk_proj_quadruple_reorder_index

def test_large_channels():
    """Test K-means performance with different channel counts"""
    print("=== Testing K-means with Large Channel Counts ===\n")
    
    # 测试不同的通道数量
    channel_counts = [256, 512, 1024, 2048, 4096]
    block_size = 16
    
    for D in channel_counts:
        print(f"\n--- Testing with {D} channels ---")
        K = D // block_size
        
        # 生成测试数据
        torch.manual_seed(42)
        xmax = torch.randn(D) * 3.0
        xmin = torch.randn(D) * 2.0
        
        # 测试2D K-means
        print("Testing 2D K-means...")
        start_time = time.time()
        try:
            idx_2d, counts_2d = balanced_kmeans2d_calc_reorder_index(xmax, xmin, K, block_size)
            time_2d = time.time() - start_time
            print(f"  2D K-means: {time_2d:.3f}s, success: {len(idx_2d) == D}")
        except Exception as e:
            print(f"  2D K-means failed: {e}")
            time_2d = float('inf')
        
        # 测试四元数K-means
        print("Testing quadruple K-means...")
        start_time = time.time()
        try:
            # 生成q和k的数据
            q_xmax = torch.randn(D) * 2.5
            q_xmin = torch.randn(D) * 1.8
            k_xmax = torch.randn(D) * 1.5
            k_xmin = torch.randn(D) * 1.2
            
            idx_quad, counts_quad = qk_proj_quadruple_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size)
            time_quad = time.time() - start_time
            print(f"  Quadruple K-means: {time_quad:.3f}s, success: {len(idx_quad) == D}")
        except Exception as e:
            print(f"  Quadruple K-means failed: {e}")
            time_quad = float('inf')
        
        # 性能总结
        print(f"  Performance summary:")
        print(f"    2D K-means: {time_2d:.3f}s")
        print(f"    Quadruple K-means: {time_quad:.3f}s")

def test_clustering_quality():
    """Test clustering quality with different parameters"""
    print("\n=== Testing Clustering Quality ===\n")
    
    D = 2048
    block_size = 16
    K = D // block_size
    
    # 生成有明显聚类结构的数据
    torch.manual_seed(42)
    
    # 创建4个明显的聚类
    cluster_centers = torch.tensor([[3.0, 2.0], [1.0, 4.0], [4.0, 1.0], [2.0, 3.0]])
    cluster_size = D // 4
    
    xmax_list = []
    xmin_list = []
    
    for i, center in enumerate(cluster_centers):
        # 为每个聚类生成数据
        cluster_xmax = torch.randn(cluster_size) * 0.5 + center[0]
        cluster_xmin = torch.randn(cluster_size) * 0.5 + center[1]
        xmax_list.append(cluster_xmax)
        xmin_list.append(cluster_xmin)
    
    xmax = torch.cat(xmax_list)
    xmin = torch.cat(xmin_list)
    
    print(f"Generated data with {D} channels and {K} blocks")
    print(f"Data range: xmax=[{xmax.min():.3f}, {xmax.max():.3f}], xmin=[{xmin.min():.3f}, {xmin.max():.3f}]")
    
    # 测试聚类质量
    print("\nTesting clustering quality...")
    idx, counts = balanced_kmeans2d_calc_reorder_index(xmax, xmin, K, block_size)
    
    # 分析聚类结果
    print(f"Generated {len(counts)} clusters with sizes: {counts}")
    print(f"Index range: [{idx.min()}, {idx.max()}]")
    print(f"Unique indices: {len(torch.unique(idx))}")
    
    # 检查每个block的数据分布
    print("\nBlock analysis:")
    for i in range(min(8, K)):  # 显示前8个block
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        block_indices = idx[start_idx:end_idx]
        block_xmax = xmax[block_indices]
        block_xmin = xmin[block_indices]
        
        print(f"  Block {i}: xmax=[{block_xmax.min():.3f}, {block_xmax.max():.3f}], "
              f"xmin=[{block_xmin.min():.3f}, {block_xmin.max():.3f}]")

if __name__ == '__main__':
    test_large_channels()
    test_clustering_quality()
    print("\n🎉 Large channel testing completed!")
