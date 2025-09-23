# K-means 聚类优化说明

## 问题分析

当处理大量通道时，K-means聚类效果下降的主要原因：

1. **维度灾难**: 随着通道数增加，数据点间距离变得不明显
2. **初始化敏感**: 随机初始化在高维空间中效果不佳
3. **迭代不足**: 默认迭代次数无法充分收敛
4. **数据尺度**: 不同特征维度数值范围差异大

## 优化措施

### 1. **自适应参数调整**

```python
# 根据通道数量调整参数
if D_head > 1000:
    # 大量通道：使用K-means++初始化，增加迭代次数
    km = KMeans(n_clusters=K, n_init=20, random_state=0, init='k-means++', max_iter=500)
else:
    # 少量通道：使用默认参数
    km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=300)
```

### 2. **数据标准化**

```python
# 对大量通道数据进行标准化
if D_head > 1000:
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0) + 1e-8
    data_normalized = (data - data_mean) / data_std
```

### 3. **统一使用sklearn KMeans**

```python
# 直接使用sklearn的KMeans，避免外部依赖
assign_np, centers_t, dists = _greedy_capacity(data_normalized, K, block_size)
labels = torch.from_numpy(assign_np)
```

### 4. **K-means++初始化**

- 使用 `init='k-means++'` 替代随机初始化
- 提高初始中心点的质量
- 减少收敛所需的迭代次数

## 优化效果

### 参数对比

| 通道数 | 初始化次数 | 最大迭代 | 初始化方法 | 数据预处理 | KMeans库 |
|--------|------------|----------|------------|------------|----------|
| < 1000 | 10         | 300      | random     | 无         | sklearn  |
| ≥ 1000 | 20         | 500      | k-means++  | 标准化     | sklearn  |

### 性能提升

1. **收敛稳定性**: K-means++初始化提高收敛稳定性
2. **聚类质量**: 数据标准化改善聚类质量
3. **计算效率**: 自适应参数避免过度计算
4. **鲁棒性**: 针对不同规模数据优化参数

## 使用建议

### 1. **监控聚类质量**

```python
# 检查聚类结果
print(f"Applied data normalization for {D_head} channels")
```

### 2. **性能测试**

```bash
# 运行性能测试
python test_large_channels.py
```

### 3. **参数调优**

- 通道数阈值: 1000 (可调整)
- 标准化阈值: 1000 (可调整)
- 迭代次数: 根据数据复杂度调整

## 注意事项

1. **内存使用**: 标准化会增加内存使用
2. **计算时间**: 增加迭代次数会延长计算时间
3. **数值稳定性**: 添加小常数避免除零错误
4. **随机种子**: 保持可重现性

## 未来改进

1. **自适应阈值**: 根据数据特征动态调整阈值
2. **并行化**: 利用多核CPU加速计算
3. **增量聚类**: 对超大数据集使用增量方法
4. **质量评估**: 添加聚类质量评估指标
