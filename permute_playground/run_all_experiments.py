"""
运行所有置换算法的实验脚本

这个脚本展示了如何使用重构后的代码来运行不同的置换算法实验
"""

import torch
from . import utils
from . import greedy_search
from . import random_reshuffle
from . import zigzag

def run_all_permutation_experiments(num_vectors=512, vec_size=512, group_size=16, scale_format='e4m3', seed=0):
    """
    运行所有置换算法的对比实验
    
    参数:
        num_vectors: 向量数量
        vec_size: 向量维度
        group_size: 分组大小
        scale_format: 量化格式
        seed: 随机种子
    
    返回:
        results: 包含所有算法结果的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"实验配置: {num_vectors}x{vec_size}, group_size={group_size}, scale={scale_format}")
    print("=" * 60)
    
    # 生成测试数据
    tensor, means, stds = utils.generate_random_tensor_for_permutation(
        num_vectors, vec_size, device, seed
    )
    
    print(f"张量形状: {tensor.shape}")
    print(f"均值范围: [{means.min():.2f}, {means.max():.2f}]")
    print(f"标准差范围: [{stds.min():.2f}, {stds.max():.2f}]")
    print(f"值范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print("-" * 60)
    
    results = {}
    
    # 1. 运行贪心搜索实验
    print("1. 运行贪心搜索算法...")
    greedy_result = greedy_search.run_greedy_experiment(
        tensor, device, group_size, scale_format
    )
    results['greedy'] = greedy_result
    print(f"   原始MSE: {greedy_result['original_mse']:.6f}")
    print(f"   贪心MSE: {greedy_result['greedy_mse']:.6f}")
    print(f"   Naive贪心MSE: {greedy_result['naive_greedy_mse']:.6f}")
    print()
    
    # 2. 运行随机重排实验
    print("2. 运行随机重排算法...")
    rand_tensor, rand_perm, rand_orig_mse, rand_new_mse = random_reshuffle.random_reshuffle_permutation(
        tensor, device, seed
    )
    results['random'] = {
        'original_mse': rand_orig_mse.item(),
        'new_mse': rand_new_mse.item(),
        'permute_matrix': rand_perm,
        'new_tensor': rand_tensor
    }
    print(f"   原始MSE: {rand_orig_mse.item():.6f}")
    print(f"   随机重排MSE: {rand_new_mse.item():.6f}")
    print(f"   MSE差异: {rand_new_mse.item() - rand_orig_mse.item():.6f}")
    print()
    
    # 3. 运行Zigzag实验
    print("3. 运行Zigzag算法...")
    zig_tensor, zig_perm, zig_orig_mse, zig_new_mse, zig_groups = zigzag.zigzag_permutation(
        tensor, device, num_groups=32, group_size=group_size
    )
    results['zigzag'] = {
        'original_mse': zig_orig_mse.item(),
        'new_mse': zig_new_mse.item(),
        'permute_matrix': zig_perm,
        'new_tensor': zig_tensor,
        'groups': zig_groups
    }
    print(f"   原始MSE: {zig_orig_mse.item():.6f}")
    print(f"   Zigzag MSE: {zig_new_mse.item():.6f}")
    print(f"   MSE差异: {zig_new_mse.item() - zig_orig_mse.item():.6f}")
    print()
    
    # 4. 运行综合实验（使用utils中的run_experiment）
    print("4. 运行综合实验（贪心+SLS优化）...")
    comprehensive_result = utils.run_experiment(
        tensor, 
        mode='greedy_sls', 
        group_size=group_size, 
        scale_format=scale_format,
        sls_steps=200,  # 减少步数以加快速度
        seed=seed
    )
    results['comprehensive'] = comprehensive_result
    print(f"   原始MSE: {comprehensive_result['base_mse']:.6f}")
    print(f"   随机MSE: {comprehensive_result['rand_mse']:.6f}")
    print(f"   贪心+SLS MSE: {comprehensive_result['perm_mse']:.6f}")
    print()
    
    # 5. 结果总结
    print("=" * 60)
    print("实验结果总结:")
    print(f"{'算法':<15} {'原始MSE':<12} {'置换后MSE':<12} {'改善':<12}")
    print("-" * 60)
    
    algorithms = [
        ('贪心搜索', results['greedy']['original_mse'], results['greedy']['greedy_mse']),
        ('Naive贪心', results['greedy']['original_mse'], results['greedy']['naive_greedy_mse']),
        ('随机重排', results['random']['original_mse'], results['random']['new_mse']),
        ('Zigzag', results['zigzag']['original_mse'], results['zigzag']['new_mse']),
        ('贪心+SLS', results['comprehensive']['base_mse'], results['comprehensive']['perm_mse'])
    ]
    
    for name, orig, new in algorithms:
        improvement = orig - new
        print(f"{name:<15} {orig:<12.6f} {new:<12.6f} {improvement:<12.6f}")
    
    print("=" * 60)
    
    return results

def main():
    """主函数"""
    print("Permutation Game - 置换算法对比实验")
    print("=" * 60)
    
    # 运行实验
    results = run_all_permutation_experiments(
        num_vectors=512,
        vec_size=512,
        group_size=16,
        scale_format='e4m3',
        seed=42
    )
    
    print("实验完成！")
    return results

if __name__ == "__main__":
    main()
