import math, torch, torch.nn as nn
import utils

torch.manual_seed(0)

# ---------- SoftSort（连续置换） ----------
def softsort_matrix(w, tau: float):
    """
    w: (N,) trainable weights
    return: P_soft (N,N) row-stochastic, using SoftSortD_tau(w) = softmax(-|w_sort - w|/tau)
    """
    w = w.view(-1)
    N = w.numel()
    w_sorted, _ = torch.sort(w)                       # (N,)
    # L1 distance matrix D_ij = |w_sorted[j] - w[i]|
    D = torch.abs(w.view(N, 1) - w_sorted.view(1, N)) # (N,N)
    P = torch.softmax(-D / tau, dim=1)                # row-wise stochastic
    return P

# ---------- ShuffleSoftSort 外循环 + 两个正则 ----------
def stochasticity_loss(P):
    # 列和接近 1 的约束（列双随机的一半；行方向由 softmax 保证）
    col_sum = P.sum(dim=0)            # (N,)
    return torch.mean((col_sum - 1.0) ** 2)

def std_preserve_loss(X, X_perm):
    # 排序前后标准差一致
    s1 = X.std()
    s2 = X_perm.std()
    return torch.abs(s1 - s2) / (s1 + 1e-12)

# ---------- 前面MXFP4的简化实现（与之前demo一致） ----------
# (为节省篇幅，这里省略了注释；若你没有上一条消息的 MXFP4 类，请把那段 MXFP4 代码粘过来)
import math
def fp4_e2m1_quantize(x):
    x_abs = x.abs()
    sign = (x < 0).to(torch.uint8)
    is_zero = (x_abs == 0)
    eps = torch.finfo(torch.float32).tiny
    exp_real = torch.floor(torch.log2(torch.clamp(x_abs, min=eps)))
    frac = x_abs / (2.0 ** exp_real)
    mant = (frac >= 1.25).to(torch.uint8)
    exp_real = torch.clamp(exp_real, -1, 2)
    significand = torch.where(mant.bool(), torch.tensor(1.5, device=x.device),
                              torch.tensor(1.0, device=x.device))
    x_q = torch.where(is_zero, torch.zeros_like(x),
                      torch.sign(x) * (2.0 ** exp_real) * significand)
    exp_field = (exp_real + 1).to(torch.uint8)
    code = (sign << 3) | (exp_field << 1) | mant
    code = torch.where(is_zero, torch.zeros_like(code), code)
    x_q = torch.where(is_zero, torch.zeros_like(x_q), x_q)
    return code, x_q

def fp4_e2m1_dequantize(code):
    sign = (code >> 3) & 0x1
    exp_field = (code >> 1) & 0x3
    mant = code & 0x1
    exp_real = exp_field.to(torch.int16) - 1
    significand = torch.where(mant.bool(), torch.tensor(1.5, device=code.device),
                              torch.tensor(1.0, device=code.device))
    val = (2.0 ** exp_real.float()) * significand
    val = torch.where(sign.bool(), -val, val)
    val = torch.where(code == 0, torch.zeros_like(val), val)
    return val

def choose_block_scale_pow2(max_abs, fp4_max=6.0):
    target = torch.clamp(max_abs / fp4_max, min=torch.finfo(torch.float32).tiny)
    e = torch.round(torch.log2(target)).to(torch.int32)
    e = torch.clamp(e, -127, 127).to(torch.int8)
    scale = torch.pow(2.0, e.float())
    return scale, e

# ---------- 用 ShuffleSoftSort 学列置换，最小化 MXFP4 误差 ----------
if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    N = 64
    OC, IC = N, N
    # W = (torch.randn(OC, IC, device=device, dtype=torch.float32) * 0.05).requires_grad_(False)
    num_vectors = N  # 子向量总数（行数）
    vec_size = N     # 每个子向量的维度（列数）
    for layer_idx in [3,4,5,6,7,8,9]:
        print(f"---------{layer_idx}-----------")
        tensor = utils.load_olmoe_q_proj_layer(layer_idx = layer_idx)
        W = utils.extract_NxN_subblock(tensor,sub_size = num_vectors).to(device)
        ori_W = W.clone()
        # 生成512x512的tensor，每行是不同高斯分布的采样

        # 可学习权重（SoftSort仅需N参数）
        w = nn.Parameter(torch.arange(IC, device=device, dtype=torch.float32))  # 线性初始化
        ori_quantized = utils.fp4_quantize(W, group_size=16, scale_format="e4m3")
        # print(ori_W)
        # print(ori_quantized)
        original_mse = utils.calculate_quantization_mse(ori_W, ori_quantized)
        print(f"[MXFP4+ShuffleSoftSort] 初始量化 MSE = {original_mse:.10e}")
        ori_quantized = utils.fp4_quantize(W, group_size=16, scale_format="e4m3",e4m3_scale = 1.2)
        # print(ori_W)
        # print(ori_quantized)
        original_mse = utils.calculate_quantization_mse(ori_W, ori_quantized)
        print(f"[MXFP4+ShuffleSoftSort] 初始量化 MSE (Double) = {original_mse:.10e}")
        opt = torch.optim.AdamW([w], lr=5e-6)

        R = 16                  # 外层 shuffle 轮数（ShuffleSoftSort）
        I = 16                  # 每轮内的 SoftSort 小步
        tau_start, tau_end = 1.0, 0.1
        lam_s, lam_sigma = 1.0, 2.0       # 两个正则的权重【Ls, Lσ】

        for r in range(1, R+1):
            # 退火 tau
            tau = tau_start * ((tau_end / tau_start) ** (r / R))

            # 随机打乱
            shuf_idx = torch.randperm(IC, device=device)
            S = torch.eye(IC, device=device)[shuf_idx]       # (N,N) shuffle 矩阵
            W_shuf = W @ S                                   # 列打乱

            # SoftSort 内循环
            for _ in range(I):
                opt.zero_grad()
                # 在打乱后的索引上做 SoftSort
                P_soft = softsort_matrix(w[shuf_idx], tau)   # (N,N), 行随机
                # 应用软置换到列：W_sorted_soft = W_shuf @ P_soft
                W_soft = W_shuf @ P_soft

                # MXFP4 量化/反量化（按最后一维分块）
                W_rec = utils.fp4_quantize(W, group_size=16, scale_format="e4m3")

                # 量化重构损失（主）
                loss_q = torch.mean((W_soft - W_rec) ** 2)
                # ShuffleSoftSort 的两个正则（列和约束、方差保持）【论文中的 Ls 与 Lσ】
                loss_s = stochasticity_loss(P_soft)
                loss_sigma = std_preserve_loss(W_shuf, W_soft)

                loss = loss_q + lam_s * loss_s + lam_sigma * loss_sigma
                loss.backward(retain_graph=True)
                opt.step()

            print(f"[Round {r}/{R}] tau={tau:.3f}  loss={loss.item():.3e}")

        # 生成“硬置换”：对行取 argmax 得到置换索引（shuffled->sorted 的映射）
        with torch.no_grad():
            P_last = softsort_matrix(w[shuf_idx], tau_end)
            sort_idx = torch.argmax(P_last, dim=1)             # (N,)
            # 组合得到原空间的最终置换：先打乱 S，再按 sort_idx 放回
            final_idx = shuf_idx[sort_idx]                      # 原列索引的最终顺序
            # 应用硬置换
            # print(final_idx)
            W_perm_hard = W[:, final_idx]
            # 量化-反量化评估
            new_quantized = utils.fp4_quantize(W_perm_hard, group_size=16, scale_format="e4m3")
            # mse_perm = utils.calculate_quantization_mse(ori_W, new_quantized)
            # print(f"[MXFP4+ShuffleSoftSort] 硬置换后量化 MSE = {mse_perm:.10e}")
            mse_perm = utils.calculate_quantization_mse(W_perm_hard, new_quantized)
            print(f"[MXFP4+ShuffleSoftSort] 硬置换后量化 MSE = {mse_perm:.10e}")