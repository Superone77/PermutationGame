"""
FP4-Grid Greedy Search 算法

实现基于列均值的网格贪心搜索置换算法：
1. 计算每个列的均值
2. 按均值大小排序
3. 为每个block做贪心搜索，选择相似均值的列
4. 返回重排后的tensor和置换矩阵
"""

import torch
import torch.nn.functional as F
from . import utils

def fp4_grid_greedy_search(tensor: torch.Tensor, 
                           device: torch.device, 
                           block_size: int = 32,
                           scale_format: str = 'e4m3',
                           group_size: int = 16):
    """
    FP4-Grid Greedy Search 置换算法
    
    参数:
        tensor: 输入张量 [R, C]
        device: 计算设备
        block_size: 每个block的大小（默认32）
        scale_format: 量化格式
        group_size: 量化分组大小
    
    返回:
        new_tensor: 重排后的张量
        permute_matrix: 置换矩阵
        original_mse: 原始量化MSE
        new_mse: 重排后量化MSE
        blocks: 分组信息
    """
    R, C = tensor.shape
    num_blocks = (C + block_size - 1) // block_size  # 向上取整
    
    # 计算原始量化MSE
    ori_quantized = utils.fp4_quantize(tensor, group_size=group_size, scale_format=scale_format)
    original_mse = utils.calculate_quantization_mse(tensor, ori_quantized)
    
    # 计算每个列的均值
    col_means = tensor.mean(dim=0)  # [C]
    col_means_abs = torch.abs(col_means)  # 绝对值
    
    # 创建列索引和均值的配对列表
    col_data = [(i, col_means[i], col_means_abs[i]) for i in range(C)]
    
    # 按绝对值均值从大到小排序
    col_data.sort(key=lambda x: x[2], reverse=True)
    
    # 初始化blocks
    blocks = [[] for _ in range(num_blocks)]
    remaining_cols = col_data.copy()
    
    print(f"开始FP4-Grid Greedy Search，共{num_blocks}个block，每个block最多{block_size}列")
    
    # 为每个block做贪心搜索
    for block_idx in range(num_blocks):
        if not remaining_cols:
            break
            
        print(f"处理block {block_idx + 1}/{num_blocks}")
        
        # 选择当前最大的均值列作为该block的种子
        seed_idx, seed_mean, seed_mean_abs = remaining_cols[0]
        blocks[block_idx].append(seed_idx)
        remaining_cols.pop(0)
        
        print(f"  Block {block_idx + 1} 种子列: {seed_idx}, 均值: {seed_mean:.6f}")
        
        # 定义相似度阈值（相对于种子列均值的比例）
        similarity_thresholds = [0.0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3]
        
        # 遍历剩余列，寻找相似均值的列
        i = 0
        while i < len(remaining_cols) and len(blocks[block_idx]) < block_size:
            col_idx, col_mean, col_mean_abs = remaining_cols[i]
            
            # 检查是否满足任一相似度阈值
            added = False
            for threshold in similarity_thresholds:
                if abs(col_mean_abs - seed_mean_abs * threshold) < 1e-6:  # 约等于
                    blocks[block_idx].append(col_idx)
                    remaining_cols.pop(i)
                    print(f"    添加列 {col_idx} (均值: {col_mean:.6f}, 阈值: {threshold})")
                    added = True
                    break
            
            if not added:
                i += 1
        
        print(f"  Block {block_idx + 1} 完成，包含 {len(blocks[block_idx])} 列")
    
    # 重新构建tensor
    new_tensor = torch.zeros_like(tensor)
    permute_matrix = torch.zeros(C, dtype=torch.long, device=device)
    
    col_idx = 0
    for block_idx, block_cols in enumerate(blocks):
        for i, orig_col_idx in enumerate(block_cols):
            new_tensor[:, col_idx] = tensor[:, orig_col_idx]
            permute_matrix[col_idx] = orig_col_idx
            col_idx += 1
    
    # 计算重排后的量化MSE
    new_quantized = utils.fp4_quantize(new_tensor, group_size=group_size, scale_format=scale_format)
    new_mse = utils.calculate_quantization_mse(new_tensor, new_quantized)
    
    return new_tensor, permute_matrix, original_mse, new_mse, blocks

def run_fp4_grid_experiment(tensor: torch.Tensor, 
                           device: torch.device, 
                           block_size: int = 32,
                           scale_format: str = 'e4m3',
                           group_size: int = 16):
    """
    运行FP4-Grid Greedy Search实验
    
    参数:
        tensor: 输入张量
        device: 计算设备
        block_size: block大小
        scale_format: 量化格式
        group_size: 量化分组大小
    
    返回:
        result: 实验结果字典
    """
    print(f"运行FP4-Grid Greedy Search实验")
    print(f"张量形状: {tensor.shape}")
    print(f"Block大小: {block_size}")
    print(f"量化格式: {scale_format}")
    print(f"量化分组: {group_size}")
    print("-" * 50)
    
    # 运行算法
    new_tensor, permute_matrix, original_mse, new_mse, blocks = fp4_grid_greedy_search(
        tensor, device, block_size, scale_format, group_size
    )
    
    # 验证置换矩阵
    unique_indices = torch.unique(permute_matrix)
    print(f"置换矩阵验证: 包含 {unique_indices.numel()} 个唯一索引")
    print(f"索引范围: [{unique_indices.min()}, {unique_indices.max()}]")
    
    # 输出结果
    print(f"原始量化MSE: {original_mse.item():.6f}")
    print(f"重排后量化MSE: {new_mse.item():.6f}")
    print(f"MSE差异: {new_mse.item() - original_mse.item():.6f}")
    
    # 输出block信息
    print(f"\nBlock分组信息:")
    for i, block in enumerate(blocks):
        print(f"  Block {i+1}: {len(block)} 列")
    
    result = {
        'original_mse': original_mse.item(),
        'new_mse': new_mse.item(),
        'permute_matrix': permute_matrix,
        'new_tensor': new_tensor,
        'blocks': blocks,
        'block_size': block_size
    }
    
    return result

def main():
    """主函数：运行FP4-Grid Greedy Search实验"""
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 生成测试数据（使用与其他实验一致的随机种子）
    num_vectors = 512
    vec_size = 512
    seed = 42  # 保持一致的随机种子
    
    # 使用utils中的函数生成tensor
    tensor, means, stds = utils.generate_random_tensor_for_permutation(
        num_vectors, vec_size, device, seed
    )
    
    print(f"张量形状: {tensor.shape}")
    print(f"均值范围: [{means.min():.2f}, {means.max():.2f}]")
    print(f"标准差范围: [{stds.min():.2f}, {stds.max():.2f}]")
    print(f"值范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print("=" * 60)
    
    # 运行FP4-Grid Greedy Search实验
    result = run_fp4_grid_experiment(
        tensor, 
        device, 
        block_size=32,  # 每个block 32列
        scale_format='e4m3',
        group_size=16
    )
    
    print("=" * 60)
    print("实验完成！")
    
    return result

if __name__ == "__main__":
    main()
