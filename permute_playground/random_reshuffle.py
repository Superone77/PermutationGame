## random reshuffle
import torch
import torch.nn.functional as F
import numpy as np
from . import utils

def random_reshuffle_permutation(tensor: torch.Tensor, device: torch.device, seed: int = 0):
    """
    随机重排置换算法
    
    参数:
        tensor: 输入张量
        device: 计算设备
        seed: 随机种子
    
    返回:
        new_tensor: 重排后的张量
        permute_matrix: 置换矩阵
        original_mse: 原始量化MSE
        new_mse: 重排后量化MSE
    """
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 计算原始量化MSE
    ori_quantized = utils.fp4_quantize(tensor, group_size=16, scale_format='e4m3')
    original_mse = utils.calculate_quantization_mse(tensor, ori_quantized)
    
    # 生成随机置换索引
    shuffled_indices = torch.randperm(tensor.shape[1], device=device)
    
    # 按照随机索引重新排列子向量
    new_tensor = tensor[:, shuffled_indices]
    
    # 置换矩阵记录了新张量中每个位置对应的原始索引
    permute_matrix = shuffled_indices
    
    # 计算重排后的量化MSE
    new_quantized = utils.fp4_quantize(new_tensor, group_size=16, scale_format='e4m3')
    new_mse = utils.calculate_quantization_mse(new_tensor, new_quantized)
    
    return new_tensor, permute_matrix, original_mse, new_mse

def main():
    """主函数：运行随机重排置换实验"""
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
    
    # 运行随机重排置换
    new_tensor, permute_matrix, original_mse, new_mse = random_reshuffle_permutation(
        tensor, device, seed=0
    )
    
    # 输出结果
    print(f"重新排列后的张量形状: {new_tensor.shape}")
    print(f"置换矩阵形状: {permute_matrix.shape}")
    
    # 验证置换矩阵的正确性
    unique_indices = torch.unique(permute_matrix)
    print(f"置换矩阵验证: 包含 {unique_indices.numel()} 个唯一索引，范围 [{unique_indices.min()}, {unique_indices.max()}]")
    
    print(f"原始张量量化MSE: {original_mse.item()}")
    print(f"重新排列后的张量量化MSE: {new_mse.item()}")
    print(f"MSE差异: {new_mse.item() - original_mse.item()}")
    
    return new_tensor, permute_matrix, original_mse, new_mse

if __name__ == "__main__":
    # 运行多次实验
    for i in range(10):
        print(f"\n=== 实验 {i+1} ===")
        main()
