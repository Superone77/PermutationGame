

## zigzag permutation
import torch
import torch.nn.functional as F
import numpy as np
from . import utils

def zigzag_permutation(tensor: torch.Tensor, device: torch.device, num_groups=32, group_size=16):
    """
    Zigzag置换算法
    
    参数:
        tensor: 输入张量
        device: 计算设备
        num_groups: 分组数量
        group_size: 每组大小
    
    返回:
        new_tensor: 重排后的张量
        permute_matrix: 置换矩阵
        original_mse: 原始量化MSE
        new_mse: 重排后量化MSE
    """
    # 计算原始量化MSE
    ori_quantized = utils.fp4_quantize(tensor, group_size=group_size, scale_format='e4m3')
    original_mse = utils.calculate_quantization_mse(tensor, ori_quantized)
    
    # 初始分组
    initial_groups = [tensor[:, i*group_size : (i+1)*group_size] 
                     for i in range(num_groups)]
    
    # 收集所有子向量及其原始索引
    all_vectors = []
    all_indices = []
    
    for group_idx in range(num_groups):
        # 获取当前组的子向量
        group = initial_groups[group_idx]
        # 计算当前组的原始索引范围
        start_idx = group_idx * group_size
        end_idx = (group_idx + 1) * group_size
        original_indices = torch.arange(start_idx, end_idx, device=device)
        
        # 收集所有子向量及其索引
        all_vectors.append(group.T)  # 转置为 (16, 512)
        all_indices.append(original_indices)
    
    # 将所有向量和其索引合并为一个列表
    all_vectors = torch.cat(all_vectors, dim=0)  # 形状: (512, 512)
    all_indices = torch.cat(all_indices, dim=0)  # 形状: (512,)
    
    # 按照每个子向量的最大值进行从大到小排序（保持索引对应）
    vec_maxes = all_vectors.max(dim=1)[0]  # 计算每个子向量的最大值
    sorted_indices = torch.argsort(vec_maxes, descending=True)
    sorted_vectors = all_vectors[sorted_indices]
    sorted_indices = all_indices[sorted_indices]  # 保持索引与向量对应
    
    # 使用zigzag算法分配子向量
    # 初始化组和组索引
    groups = [torch.empty((0, tensor.shape[0]), device=device) for _ in range(num_groups)]
    group_indices = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_groups)]
    
    # 实现zigzag分配模式：1→2→...→k→k→k-1→...→1→1→2→...
    for i, (vec, idx) in enumerate(zip(sorted_vectors, sorted_indices)):
        # 计算当前应该分配到的组索引
        cycle = 2 * (num_groups - 1)
        pos_in_cycle = i % cycle
        
        if pos_in_cycle < num_groups:
            group_idx = pos_in_cycle
        else:
            group_idx = 2 * (num_groups - 1) - pos_in_cycle
        
        # 将向量和索引加入对应组
        vec = vec.unsqueeze(0)  # 增加一个维度
        idx = idx.unsqueeze(0)
        
        groups[group_idx] = torch.cat([groups[group_idx], vec], dim=0)
        group_indices[group_idx] = torch.cat([group_indices[group_idx], idx], dim=0)
    
    # 重新构建tensor并构建置换矩阵
    # 转置回来以匹配原始形状 (512, 512)
    new_tensor = torch.cat([group.T for group in groups], dim=1)
    permute_matrix = torch.cat(group_indices, dim=0)  # 置换矩阵：新张量中每个位置对应的原始索引
    
    # 计算重排后的量化MSE
    new_quantized = utils.fp4_quantize(new_tensor, group_size=group_size, scale_format='e4m3')
    new_mse = utils.calculate_quantization_mse(new_tensor, new_quantized)
    
    return new_tensor, permute_matrix, original_mse, new_mse, groups,ori_quantized,new_quantized

def main():
    """主函数：运行zigzag置换实验"""
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 生成测试数据
    num_vectors = 512  # 子向量总数（行数）
    vec_size = 512     # 每个子向量的维度（列数）
    
    # 使用utils中的函数生成tensor
    tensor, means, stds = utils.generate_random_tensor_for_permutation(
        num_vectors, vec_size, device, seed=0
    )

    print(f"张量形状: {tensor.shape}")
    print(f"均值范围: [{means.min():.2f}, {means.max():.2f}]")
    print(f"标准差范围: [{stds.min():.2f}, {stds.max():.2f}]")
    print(f"子块值范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
    
    num_groups = 32
    group_size = 16
    
    # 运行zigzag置换
    new_tensor, permute_matrix, original_mse, new_mse, groups,ori_quantized,new_quantized = zigzag_permutation(
        tensor, device, num_groups, group_size
    )
    
    # 输出结果
    print(f"所有子向量数量: {tensor.shape[1]}")
    print(f"所有子向量已按最大值从大到小排序")
    
    # 验证每组的大小
    for i in range(num_groups):
        print(f"组 {i+1} 大小: {groups[i].shape[0]}")
    
    print(f"重新分组后的张量形状: {new_tensor.shape}")
    print(f"置换矩阵形状: {permute_matrix.shape}")
    
    # 验证置换矩阵的正确性
    unique_indices = torch.unique(permute_matrix)
    print(f"置换矩阵验证: 包含 {unique_indices.numel()} 个唯一索引，范围 [{unique_indices.min()}, {unique_indices.max()}]")
    
    
    
    
    print(f"原始张量量化MSE: {original_mse.item()}")
    print(f"重新排列后的张量量化MSE: {new_mse.item()}")
    print(f"MSE差异: {new_mse.item() - original_mse.item()}")

    # 计算逆置换：将量化后的打乱张量恢复到原始顺序
    inv_indices = torch.argsort(permute_matrix)  # 逆置换
    restored_quantized = new_quantized[:, inv_indices]  # 应用逆置换恢复顺序

    # 对比原始量化结果和恢复后的量化结果
    diff = ori_quantized - restored_quantized
    mse_between_quantized = F.mse_loss(ori_quantized, restored_quantized)
    max_abs_diff = torch.max(torch.abs(diff))
    num_different = torch.sum(torch.abs(diff) > 1e-9).item()
    total_elements = diff.numel()
    percent_different = (num_different / total_elements) * 100
    
    # 输出对比结果
    print(f"\n量化后恢复与原始量化的对比:")
    print(f"  元素差异MSE: {mse_between_quantized.item():.10f}")
    print(f"  最大绝对差异: {max_abs_diff.item():.10f}")
    print(f"  不同元素数量: {num_different}/{total_elements} ({percent_different:.2f}%)")
    
    return new_tensor, permute_matrix, original_mse, new_mse

if __name__ == "__main__":
    # 运行多次实验
    for i in range(10):
        print(f"\n=== 实验 {i+1} ===")
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(i)
        main()
