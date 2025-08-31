import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# FP4量化相关函数
# =========================

def fp4_121_positive(x: torch.Tensor, stochastic_rounding: bool = False) -> torch.Tensor:
    """FP4 1-2-1 正数量化函数"""
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    
    mask1 = x < 2.0
    mask2 = x < 4.0
    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)

FP8_E4M3_MAX = 448.0

def fp4_quantize(x: torch.Tensor, 
                 stochastic_rounding: bool = False, 
                 group_size: int = 16,
                 scale_format: str = 'e4m3') -> torch.Tensor:
    """FP4量化函数，支持多种缩放格式"""
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    ori_shape = x.shape
    x_abs = x_abs.reshape(-1, group_size)
    
    if scale_format == 'e8m0':
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
    elif scale_format == 'e4m3':
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t
        
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        input_tensor = fp4_121_max / scale_per_b
        down_cast = input_tensor.to(torch.float8_e4m3fn)
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
        
        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b
        x_fp4_abs = x_fp4_abs.reshape(ori_shape)
        return sign * x_fp4_abs * scale_per_t
    else:  # bf16风格
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    x_fp4_abs = x_fp4_abs.reshape(ori_shape)
    return sign * x_fp4_abs

def calculate_quantization_mse(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """计算原始张量和量化后张量之间的MSE"""
    return F.mse_loss(original, quantized)

# =========================
# 数据生成函数
# =========================

def make_synthetic_tensor(num_vectors=512, vec_size=512, device='cpu', seed=0):
    """生成合成张量：行独立高斯分布"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    means = torch.normal(mean=3.0, std=5.0, size=(num_vectors,), generator=g, device=device)
    stds = torch.rand((num_vectors,), generator=g, device=device) * 4.5 + 0.5
    base = torch.randn((num_vectors, vec_size), generator=g, device=device)
    return base * stds.unsqueeze(1) + means.unsqueeze(1)

def generate_random_tensor_for_permutation(num_vectors=512, vec_size=512, device='cpu', seed=0):
    """生成用于置换实验的随机tensor（与原始代码保持一致）"""
    # 随机生成均值：使用正态分布，均值为3，标准差为5
    means = torch.normal(mean=3.0, std=5.0, size=(num_vectors,), device=device)
    
    # 随机生成标准差：使用0.5到5.0之间的均匀分布
    stds = torch.rand(num_vectors, device=device) * 4.5 + 0.5
    
    # 生成基础tensor
    base_tensor = torch.randn(num_vectors, vec_size, device=device)
    
    # 按行应用对应的均值和标准差
    tensor = base_tensor * stds.unsqueeze(1) + means.unsqueeze(1)
    
    return tensor, means, stds

# =========================
# 模型加载函数
# =========================

def load_olmoe_q_proj_layer(model_name="allenai/OLMoE-1B-7B-0125-Instruct", layer_idx=3):
    """加载OLMoE模型并提取指定层的q_proj权重"""
    try:
        print(f"正在加载模型: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        if layer_idx >= len(model.model.layers):
            raise ValueError(f"层索引 {layer_idx} 超出范围，模型共有 {len(model.model.layers)} 层")
        
        layer = model.model.layers[layer_idx]
        q_proj_weight = layer.self_attn.q_proj.weight
        
        print(f"成功提取第{layer_idx+1}层q_proj权重，形状: {q_proj_weight.shape}")
        return q_proj_weight
    
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def extract_512x512_subblock(weight_matrix, start_row=0, start_col=0):
    """从权重矩阵中提取512x512的子块"""
    if weight_matrix.shape[0] < 512 or weight_matrix.shape[1] < 512:
        raise ValueError(f"权重矩阵形状 {weight_matrix.shape} 小于512x512，无法提取子块")
    
    end_row = min(start_row + 512, weight_matrix.shape[0])
    end_col = min(start_row + 512, weight_matrix.shape[1])
    
    if end_row - start_row < 512:
        start_row = weight_matrix.shape[0] - 512
        end_row = weight_matrix.shape[0]
    
    if end_col - start_col < 512:
        start_col = weight_matrix.shape[1] - 512
        end_col = weight_matrix.shape[1]
    
    subblock = weight_matrix[start_row:end_row, start_col:end_col]
    print(f"提取512x512子块，范围: 行[{start_row}:{end_row}], 列[{start_col}:{end_col}]")
    return subblock

# =========================
# 置换相关工具函数
# =========================

def apply_permutation_columns(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """列置换：对所有行使用同一列顺序"""
    return tensor[:, order]

def eval_mse_for_order(tensor: torch.Tensor, order: torch.Tensor, group_size=16, scale_format='e4m3') -> float:
    """评估特定列置换顺序下的量化MSE"""
    x = apply_permutation_columns(tensor, order)
    xq = fp4_quantize(x, group_size=group_size, scale_format=scale_format)
    return float(calculate_quantization_mse(x, xq).item())

def _bucket_capacities(m: int, d: int):
    """计算桶容量：前 r = m % d 个桶容量 = ceil(m/d)，其余为 floor(m/d)"""
    base = m // d
    r = m % d
    caps = [base + 1 if i < r else base for i in range(d)]
    return caps

# =========================
# 实验运行函数
# =========================

def run_experiment(tensor: torch.Tensor,
                   mode='greedy',
                   group_size=16,
                   scale_format='e4m3',
                   sls_steps=500,
                   seed=0):
    """运行置换实验的主函数"""
    device = tensor.device
    R, C = tensor.shape
    assert group_size >= 1 and group_size <= C

    # baseline（不置换）
    base_order = torch.arange(C, device=device)
    base_mse = eval_mse_for_order(tensor, base_order, group_size, scale_format)

    # 随机置换（参考）
    g = torch.Generator(device=device).manual_seed(seed)
    rand_order = torch.randperm(C, generator=g, device=device)
    rand_mse = eval_mse_for_order(tensor, rand_order, group_size, scale_format)

    if mode == 'greedy':
        from .greedy_search import greedy_interlaced_permutation_columns
        order = greedy_interlaced_permutation_columns(tensor, group_size)
    elif mode == 'sls':
        from .greedy_search import sls_refine_columns
        order = sls_refine_columns(tensor, base_order, group_size, scale_format, sls_steps, rng=g)
    elif mode == 'greedy_sls':
        from .greedy_search import greedy_interlaced_permutation_columns, sls_refine_columns
        init = greedy_interlaced_permutation_columns(tensor, group_size)
        order = sls_refine_columns(tensor, init, group_size, scale_format, sls_steps, rng=g)
    elif mode == "naive_greedy_SLS":
        from .greedy_search import run_with_greedy, sls_refine_columns
        init = run_with_greedy(tensor)
        order = sls_refine_columns(tensor, init, group_size, scale_format, sls_steps, rng=g)
    else:
        raise ValueError(f"unknown mode {mode}")

    perm_mse = eval_mse_for_order(tensor, order, group_size, scale_format)

    return {
        'base_mse': base_mse,
        'rand_mse': rand_mse,
        'perm_mse': perm_mse,
        'order': order
    }
