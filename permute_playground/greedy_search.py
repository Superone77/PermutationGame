import torch
import torch.nn.functional as F
import numpy as np
from . import utils

# =========================
# Greedy：按列方差装桶 + 交错
# =========================

def greedy_interlaced_permutation_columns(tensor: torch.Tensor, d: int) -> torch.Tensor:
    """
    Greedy 装桶目标：平衡每个桶的"列方差之和"，最后交错各桶得到顺序。
    注意：我们做的是列置换，但 grouping 按行每 d 列分组，与 reshape(-1,d) 对齐。
    """
    device = tensor.device
    R, C = tensor.shape
    # 列方差（沿行方向的方差）——"row variance of W_t" 的列版
    col_var = tensor.var(dim=0, unbiased=False)  # [C]
    # 从大到小放入桶：谁大先放，尽量均衡各桶的总方差
    order_by_var = torch.argsort(col_var, descending=True)

    capacities = utils._bucket_capacities(C, d)
    buckets = [[] for _ in range(d)]
    bucket_load = [0.0 for _ in range(d)]  # 当前桶的方差和（Python float 即可）
    bucket_sizes = [0 for _ in range(d)]

    for j in order_by_var.tolist():
        # 选择"未满且当前方差和最小"的桶
        candidates = [(bucket_load[b], b) for b in range(d) if bucket_sizes[b] < capacities[b]]
        _, bmin = min(candidates, key=lambda t: t[0])
        buckets[bmin].append(j)
        bucket_load[bmin] += float(col_var[j].item())
        bucket_sizes[bmin] += 1

    # 交错（interlace）：按位置 0..m_hat 依次从 0..d-1 桶中取
    interlaced = []
    m_hat = max(len(b) for b in buckets)  # ~ ceil(C/d)
    for t in range(m_hat):
        for b in range(d):
            if t < len(buckets[b]):
                interlaced.append(buckets[b][t])

    perm = torch.tensor(interlaced, device=device, dtype=torch.long)
    assert perm.numel() == C
    return perm

# =========================
# SLS：随机交换、若降误差则接受
# =========================
@torch.no_grad()
def sls_refine_columns(tensor: torch.Tensor,
                       init_order: torch.Tensor,
                       group_size=16,
                       scale_format='e4m3',
                       max_steps=500,
                       rng=None) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator(device=tensor.device)
    order = init_order.clone()
    best_mse = utils.eval_mse_for_order(tensor, order, group_size, scale_format)
    C = order.numel()
    for _ in range(max_steps):
        i = int(torch.randint(0, C, (1,), generator=rng, device=order.device).item())
        j = int(torch.randint(0, C, (1,), generator=rng, device=order.device).item())
        if i == j:
            continue
        # 交换
        order[i], order[j] = order[j], order[i]
        mse = utils.eval_mse_for_order(tensor, order, group_size, scale_format)
        if mse + 1e-12 < best_mse:  # 接受改进
            best_mse = mse
            
        else:
            # 撤销
            order[i], order[j] = order[j], order[i]
    print(best_mse)
    print("----------------------")
    return order

# ====== 新增：naive greedy permutation ======
def greedy_perm_by_mean(tensor: torch.Tensor, group_size: int = 16, scale_format: str = 'e4m3'):
    """
    实现"naive greedy search"列置换:
    1) 先按原始顺序把列分成若干组（每组 group_size 列）
    2) 每组选出列均值(沿行)最大的那一列保留在该组，其余列放入待分配列表 pool
    3) 依次从 pool 中取列，尝试加入尚未满的每个组，计算该组在"组内共享缩放"的量化 MSE，
       把该列分配到 MSE 最小的组
    4) 每组最多容纳 group_size 列，满了就不再考虑该组
    返回:
        order: 最终列置换（按组拼接后的列顺序，形状 [C]）
        groups: 列索引的列表列表，每个子列表是一个组
    """
    device = tensor.device
    R, C = tensor.shape
    assert C % group_size == 0, "列数必须能被 group_size 整除"
    num_groups = C // group_size

    # Step 1+2: 初始分组 + 选每组均值最大的列，剩余入 pool
    groups = [[] for _ in range(num_groups)]
    pool = []
    for g in range(num_groups):
        cols = list(range(g * group_size, (g + 1) * group_size))
        block = tensor[:, cols]                             # [R, group_size]
        col_means = block.mean(dim=0)                       # [group_size]
        keep_local = int(torch.argmax(col_means).item())    # 该组中均值最大的列的"组内索引"
        keep_col = cols[keep_local]
        groups[g].append(keep_col)
        # 其它列按出现顺序进入 pool（满足"顺序移除"的要求）
        for j, c in enumerate(cols):
            if j != keep_local:
                pool.append(c)

    # Step 3: 依次从 pool 中取列，贪心放入使 MSE 最小的组（仅在该组内做量化评估）
    for col_idx in pool:
        best_group, best_mse = None, None
        for g in range(num_groups):
            if len(groups[g]) >= group_size:
                continue  # 该组已满
            cand_cols = groups[g] + [col_idx]
            sub = tensor[:, cand_cols]                      # [R, k+1]
            # 组内共享缩放 => 让 quant 函数的 group_size 等于当前组列数
            q = utils.fp4_quantize(sub, group_size=len(cand_cols), scale_format=scale_format)
            mse = F.mse_loss(sub, q)
            mse_val = float(mse)  # 取标量比较
            if (best_mse is None) or (mse_val < best_mse):
                best_mse = mse_val
                best_group = g
        groups[best_group].append(col_idx)

    # Step 4: 生成最终列顺序（把每个组的列按加入顺序拼在一起）
    final_order = torch.tensor([c for g in range(num_groups) for c in groups[g]], device=device, dtype=torch.long)

    # 一些健壮性检查
    uniq = torch.unique(final_order)
    assert uniq.numel() == C and int(uniq.min()) == 0 and int(uniq.max()) == C - 1, "置换必须覆盖全部列且无重复"
    for g in range(num_groups):
        assert len(groups[g]) == group_size, f"第 {g} 组列数不是 {group_size}"

    return final_order, groups

def run_with_greedy(tensor: torch.Tensor, scale_format: str = 'e4m3', group_size: int = 16):
    """
    用贪心置换构造新张量并报告量化误差
    """
    # 原始顺序下（按连续 16 列分组）的量化误差
    ori_q = utils.fp4_quantize(tensor, group_size=group_size, scale_format=scale_format)
    ori_mse = F.mse_loss(tensor, ori_q).item()

    # 贪心置换
    order, groups = greedy_perm_by_mean(tensor, group_size=group_size, scale_format=scale_format)
    tensor_greedy = tensor[:, order]

    # 按新顺序量化（分组仍是每行连续 16 列）
    greedy_q = utils.fp4_quantize(tensor_greedy, group_size=group_size, scale_format=scale_format)
    greedy_mse = F.mse_loss(tensor_greedy, greedy_q).item()

    print(f"[Greedy] 原始MSE: {ori_mse:.6f} | 贪心后MSE: {greedy_mse:.6f} | 改善: {ori_mse - greedy_mse:.6f}")
    return order

def run_greedy_experiment(tensor: torch.Tensor, device: torch.device, group_size: int = 16, scale_format: str = 'e4m3'):
    """
    运行完整的贪心搜索实验
    
    参数:
        tensor: 输入张量
        device: 计算设备
        group_size: 分组大小
        scale_format: 量化格式
    
    返回:
        result: 包含各种置换方法结果的字典
    """
    # 计算原始量化MSE
    ori_quantized = utils.fp4_quantize(tensor, group_size=group_size, scale_format=scale_format)
    original_mse = utils.calculate_quantization_mse(tensor, ori_quantized)
    
    # 运行贪心置换
    order = greedy_interlaced_permutation_columns(tensor, group_size)
    tensor_greedy = utils.apply_permutation_columns(tensor, order)
    greedy_mse = utils.eval_mse_for_order(tensor, order, group_size, scale_format)
    
    # 运行naive greedy
    naive_order, _ = greedy_perm_by_mean(tensor, group_size, scale_format)
    naive_mse = utils.eval_mse_for_order(tensor, naive_order, group_size, scale_format)
    
    result = {
        'original_mse': original_mse.item(),
        'greedy_mse': greedy_mse,
        'naive_greedy_mse': naive_mse,
        'greedy_order': order,
        'naive_order': naive_order,
        'greedy_tensor': tensor_greedy
    }
    
    return result

# =========================
# 数据/实验主流程
# =========================

def main():
    # ============ 配置 ============
    CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_vectors': 512,
        'vec_size': 512,
        'group_size': 16,       # d：每个子向量元素数
        'scale_format': 'e4m3', # 也可试 'e8m0'
        'mode': 'naive_greedy_SLS',   # 'greedy' | 'sls' | 'greedy_sls'
        'sls_steps': 600,       # SLS 限制步数
        'seeds': list(range(10)) # 多次试验
    }
    device = torch.device(CONFIG['device'])
    print(f"使用设备: {device}, 模式: {CONFIG['mode']}, group_size={CONFIG['group_size']}, scale={CONFIG['scale_format']}")

    for seed in CONFIG['seeds']:
        tensor = utils.make_synthetic_tensor(CONFIG['num_vectors'], CONFIG['vec_size'], device, seed)
        res = utils.run_experiment(
            tensor,
            mode=CONFIG['mode'],
            group_size=CONFIG['group_size'],
            scale_format=CONFIG['scale_format'],
            sls_steps=CONFIG['sls_steps'],
            seed=seed
        )
        print(f"[seed={seed}] base_mse={res['base_mse']:.6f}, rand_mse={res['rand_mse']:.6f}, perm_mse={res['perm_mse']:.6f}")

if __name__ == "__main__":
    main()
