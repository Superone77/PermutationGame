"""
FP4-Grid Greedy Search 算法测试脚本

这个脚本专门用于测试和验证FP4-Grid Greedy Search算法的正确性
"""

import torch
import torch.nn.functional as F
from . import utils
from . import fp4_grid_greedy

def test_fp4_grid_algorithm():
    """测试FP4-Grid Greedy Search算法的基本功能"""
    print("=" * 60)
    print("FP4-Grid Greedy Search 算法测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用固定的随机种子确保结果可重现
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 生成测试数据
    num_vectors = 512
    vec_size = 512
    
    print(f"生成测试数据: {num_vectors}x{vec_size}")
    tensor, means, stds = utils.generate_random_tensor_for_permutation(
        num_vectors, vec_size, device, seed
    )
    
    print(f"张量形状: {tensor.shape}")
    print(f"均值范围: [{means.min():.2f}, {means.max():.2f}]")
    print(f"标准差范围: [{stds.min():.2f}, {stds.max():.2f}]")
    print(f"值范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print("-" * 60)
    
    # 测试不同的block大小
    block_sizes = [16, 32, 64]
    
    for block_size in block_sizes:
        print(f"\n测试 block_size = {block_size}")
        print("-" * 40)
        
        # 运行算法
        result = fp4_grid_greedy.run_fp4_grid_experiment(
            tensor, 
            device, 
            block_size=block_size,
            scale_format='e4m3',
            group_size=16
        )
        
        # 验证结果
        print(f"\n验证结果:")
        print(f"  置换矩阵形状: {result['permute_matrix'].shape}")
        print(f"  新张量形状: {result['new_tensor'].shape}")
        
        # 验证置换矩阵的唯一性
        unique_indices = torch.unique(result['permute_matrix'])
        print(f"  唯一索引数量: {unique_indices.numel()}")
        print(f"  索引范围: [{unique_indices.min()}, {unique_indices.max()}]")
        
        # 验证置换的正确性
        if unique_indices.numel() == vec_size and unique_indices.min() == 0 and unique_indices.max() == vec_size - 1:
            print("  ✓ 置换矩阵验证通过")
        else:
            print("  ✗ 置换矩阵验证失败")
        
        # 验证block分组
        total_cols = sum(len(block) for block in result['blocks'])
        print(f"  总列数: {total_cols}")
        if total_cols == vec_size:
            print("  ✓ 列数验证通过")
        else:
            print("  ✗ 列数验证失败")
        
        # 输出MSE结果
        print(f"  原始MSE: {result['original_mse']:.6f}")
        print(f"  重排后MSE: {result['new_mse']:.6f}")
        print(f"  MSE改善: {result['original_mse'] - result['new_mse']:.6f}")
        
        # 输出block信息
        print(f"  Block信息:")
        for i, block in enumerate(result['blocks']):
            print(f"    Block {i+1}: {len(block)} 列")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

def test_algorithm_correctness():
    """测试算法的数学正确性"""
    print("\n测试算法数学正确性...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 123
    
    # 生成小规模测试数据便于验证
    num_vectors = 8
    vec_size = 16
    block_size = 4
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    tensor, _, _ = utils.generate_random_tensor_for_permutation(
        num_vectors, vec_size, device, seed
    )
    
    print(f"小规模测试: {num_vectors}x{vec_size}, block_size={block_size}")
    
    # 运行算法
    new_tensor, permute_matrix, _, _, blocks = fp4_grid_greedy.fp4_grid_greedy_search(
        tensor, device, block_size, 'e4m3', 4
    )
    
    # 验证置换的正确性
    print(f"验证置换正确性...")
    
    # 1. 检查置换矩阵是否包含所有索引
    unique_indices = torch.unique(permute_matrix)
    assert unique_indices.numel() == vec_size, f"索引数量不匹配: {unique_indices.numel()} != {vec_size}"
    assert unique_indices.min() == 0, f"最小索引不是0: {unique_indices.min()}"
    assert unique_indices.max() == vec_size - 1, f"最大索引不正确: {unique_indices.max()}"
    
    # 2. 检查新张量是否正确重构
    for i in range(vec_size):
        orig_col = permute_matrix[i]
        assert torch.allclose(new_tensor[:, i], tensor[:, orig_col]), f"列 {i} 重构错误"
    
    # 3. 检查block分组
    total_cols = sum(len(block) for block in blocks)
    assert total_cols == vec_size, f"总列数不匹配: {total_cols} != {vec_size}"
    
    print("✓ 算法数学正确性验证通过")

def main():
    """主函数"""
    print("FP4-Grid Greedy Search 算法测试")
    print("=" * 60)
    
    try:
        # 运行基本功能测试
        test_fp4_grid_algorithm()
        
        # 运行数学正确性测试
        test_algorithm_correctness()
        
        print("\n所有测试通过！🎉")
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
