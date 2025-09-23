# abs_hybrid_plus 重排序方法说明

## 概述

`abs_hybrid_plus` 是基于 `hybrid_plus` 方法的新变体，专门针对绝对值特征进行优化。该方法只关注每个通道的 `xmax` 和 `xmin` 中绝对值的较大值，简化了聚类特征。

## 核心思想

### 1. **特征简化**
- 传统方法：使用 `(xmax, xmin)` 二维特征
- abs_hybrid_plus：只使用 `max(|xmax|, |xmin|)` 一维特征

### 2. **QK层特殊处理**
- 对于 `q_proj` 和 `k_proj` 层，使用二元数特征 `(q_abs_max, k_abs_max)`
- 保持交替排序策略，确保Q和K的极值都出现在前几个block中

## 方法对比

| 方法 | 普通层特征 | QK层特征 | 聚类维度 |
|------|------------|----------|----------|
| hybrid_plus | (xmax, xmin) | (q_xmax, q_xmin, k_xmax, k_xmin) | 2D / 4D |
| abs_hybrid_plus | max(\|xmax\|, \|xmin\|) | (q_abs_max, k_abs_max) | 1D / 2D |

## 实现细节

### 1. **普通层处理**
```python
def abs_hybrid_plus_reorder_index(xmax, xmin, block_size, top_pct=0.10):
    # 计算绝对最大值
    abs_max = torch.max(torch.abs(xmax), torch.abs(xmin))
    
    # 第一阶段：按绝对最大值排序
    idx_abs_sorted = torch.argsort(abs_max, descending=True, stable=True)
    
    # 第二阶段：对中间部分进行1D K-means聚类
    # 使用 abs_kmeans_calc_reorder_index
```

### 2. **QK层处理**
```python
def qk_proj_abs_dual_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=0.10):
    # 计算Q和K的绝对最大值
    q_abs_max = torch.max(torch.abs(q_xmax), torch.abs(q_xmin))
    k_abs_max = torch.max(torch.abs(k_xmax), torch.abs(k_xmin))
    
    # 二元数特征
    dual_features = torch.stack([q_abs_max, k_abs_max], dim=1)
    
    # 交替排序策略
    # 偶数block按k排序，奇数block按q排序
```

### 3. **K-means优化**
- **1D聚类**：针对普通层的单维特征
- **2D聚类**：针对QK层的双维特征
- **自适应参数**：根据通道数量调整K-means参数
- **数据标准化**：对大量通道进行标准化处理

## 优势

### 1. **计算效率**
- 特征维度降低：2D→1D，4D→2D
- K-means计算更快
- 内存使用更少

### 2. **聚类质量**
- 简化特征减少噪声
- 绝对值特征更稳定
- 避免正负值混合的复杂性

### 3. **QK层优化**
- 保持Q和K的联合优化
- 交替排序确保极值分布
- 二元数特征平衡Q和K的贡献

## 使用示例

### 命令行使用
```bash
python main.py --reorder-method abs_hybrid_plus --hybrid-top-pct 0.10
```

### 代码调用
```python
from reorder_algorithms import abs_hybrid_plus_reorder_index

# 普通层
idx, counts = abs_hybrid_plus_reorder_index(xmax, xmin, block_size=16, top_pct=0.10)

# QK层
idx, counts = qk_proj_abs_dual_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size=16, top_pct=0.10)
```

## 参数说明

- `block_size`: 每个聚类块的大小（默认16）
- `top_pct`: 保留的top块百分比（默认0.10，即10%）
- `n_clusters`: 聚类数量（自动计算为 `D // block_size`）

## 测试验证

### 1. **单元测试**
```bash
python test_abs_hybrid_plus.py
```

### 2. **完整测试**
```bash
./run_abs_hybrid_plus.sh
```

### 3. **逻辑验证**
```bash
python test_abs_logic.py
```

## 性能特点

### 1. **内存使用**
- 特征存储：减少50%（2D→1D）或75%（4D→2D）
- 聚类计算：显著减少

### 2. **计算速度**
- K-means迭代：更快收敛
- 特征计算：简化操作

### 3. **聚类质量**
- 稳定性：绝对值特征更稳定
- 可解释性：特征含义更清晰

## 适用场景

### 1. **推荐使用**
- 通道数较多的层
- 计算资源受限的环境
- 需要快速重排序的场景

### 2. **不推荐使用**
- 需要保留正负值信息的场景
- 对聚类精度要求极高的应用

## 注意事项

1. **特征丢失**：只保留绝对值信息，丢失正负值信息
2. **QK层依赖**：Q和K层必须同时存在才能使用联合优化
3. **参数调优**：可能需要根据具体模型调整 `top_pct` 参数

## 未来改进

1. **自适应特征选择**：根据数据分布自动选择特征维度
2. **混合策略**：结合多种特征进行聚类
3. **质量评估**：添加聚类质量评估指标
