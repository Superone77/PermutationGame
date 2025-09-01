# Permutation Game

A Play Ground for permutation matrix

## 重构说明

代码已经重构，消除了重复代码，提高了可维护性：

### 重构前的问题
- 三个主要文件都有重复的量化函数定义
- 重复的MSE计算函数
- 重复的数据生成逻辑
- 重复的模型加载函数

### 重构后的结构
```
permute_playground/
├── __init__.py              # 包初始化文件
├── utils.py                 # 通用工具函数（量化、MSE计算、数据生成等）
├── greedy_search.py         # 贪心搜索置换算法
├── random_reshuffle.py      # 随机重排置换算法
├── zigzag.py               # Zigzag置换算法
├── fp4_grid_greedy.py      # FP4-Grid Greedy Search算法
├── run_all_experiments.py  # 综合实验运行脚本
└── test_fp4_grid.py        # FP4-Grid算法测试脚本
```

## 主要功能

### 1. FP4量化函数 (`utils.py`)
- `fp4_121_positive()`: FP4 1-2-1 正数量化
- `fp4_quantize()`: 支持多种缩放格式的FP4量化
- `calculate_quantization_mse()`: 量化误差计算

### 2. 置换算法
- **贪心搜索** (`greedy_search.py`): 按列方差装桶 + 交错
- **SLS优化** (`greedy_search.py`): 随机交换优化
- **随机重排** (`random_reshuffle.py`): 随机置换对比
- **Zigzag** (`zigzag.py`): Zigzag模式分配
- **FP4-Grid Greedy Search** (`fp4_grid_greedy.py`): 基于列均值的网格贪心搜索

### 3. 工具函数 (`utils.py`)
- `make_synthetic_tensor()`: 生成合成测试数据
- `generate_random_tensor_for_permutation()`: 生成置换实验用的随机tensor
- `load_olmoe_q_proj_layer()`: 加载OLMoE模型权重
- `extract_512x512_subblock()`: 提取权重子块
- `run_experiment()`: 运行置换实验

### 4. 独立算法函数
每个算法文件都提供了独立的函数：
- `greedy_search.greedy_interlaced_permutation_columns()`: 贪心装桶+交错
- `greedy_search.sls_refine_columns()`: SLS优化
- `greedy_search.greedy_perm_by_mean()`: Naive贪心搜索
- `random_reshuffle.random_reshuffle_permutation()`: 随机重排
- `zigzag.zigzag_permutation()`: Zigzag分配
- `fp4_grid_greedy.fp4_grid_greedy_search()`: FP4-Grid贪心搜索

## FP4-Grid Greedy Search 算法详解

这是一个基于列均值的网格贪心搜索算法：

### 算法步骤
1. **计算列均值**: 计算tensor每个列的均值
2. **排序**: 按绝对值均值从大到小排序
3. **Block分组**: 为每个block选择种子列（最大均值）
4. **相似度匹配**: 根据相似度阈值选择相似均值的列
5. **重构**: 将分组后的列重新组合成新的tensor

### 相似度阈值
算法使用以下阈值来匹配相似均值的列：
- 0.0倍（完全匹配）
- 1/12倍
- 1/6倍
- 1/4倍
- 1/3倍
- 1/2倍
- 2/3倍

### 参数配置
- `block_size`: 每个block的最大列数（默认32）
- `scale_format`: 量化格式（e4m3, e8m0, bf16）
- `group_size`: 量化分组大小

## 使用方法

### 基本使用
```python
from permute_playground import utils

# 生成合成数据
tensor = utils.make_synthetic_tensor(512, 512, device='cuda')

# 运行实验
result = utils.run_experiment(
    tensor, 
    mode='greedy_sls', 
    group_size=16, 
    scale_format='e4m3'
)
```

### 单独使用置换算法
```python
from permute_playground import greedy_search, random_reshuffle, zigzag, fp4_grid_greedy

# 贪心置换
order = greedy_search.greedy_interlaced_permutation_columns(tensor, d=16)

# 随机重排
new_tensor, perm_matrix, orig_mse, new_mse = random_reshuffle.random_reshuffle_permutation(tensor, device)

# Zigzag置换
new_tensor, perm_matrix, orig_mse, new_mse, groups = zigzag.zigzag_permutation(tensor, device)

# FP4-Grid Greedy Search
new_tensor, perm_matrix, orig_mse, new_mse, blocks = fp4_grid_greedy.fp4_grid_greedy_search(tensor, device)
```

### 运行所有算法对比
```python
from permute_playground import run_all_experiments

# 运行所有置换算法的对比实验
results = run_all_experiments.run_all_permutation_experiments(
    num_vectors=512,
    vec_size=512,
    group_size=16,
    scale_format='e4m3'
)
```

### 测试FP4-Grid算法
```python
from permute_playground import test_fp4_grid

# 运行FP4-Grid算法的测试
test_fp4_grid.main()
```

## 支持的量化格式

- **e4m3**: NVIDIA FP4格式
- **e8m0**: MXFP4格式  
- **bf16**: Brain Float 16风格

## 实验配置

可以通过修改各文件中的配置参数来调整实验：

```python
CONFIG = {
    'device': 'cuda',
    'num_vectors': 512,
    'vec_size': 512,
    'group_size': 16,
    'scale_format': 'e4m3',
    'mode': 'greedy_sls',
    'sls_steps': 600,
    'seeds': list(range(10))
}
```

## 依赖要求

- PyTorch >= 1.8.0
- Transformers >= 4.0.0
- NumPy >= 1.19.0

## 重构优势

1. **消除重复代码**: 所有重复的函数都移到了 `utils.py` 中
2. **模块化设计**: 每个算法文件专注于自己的置换逻辑
3. **易于维护**: 修改量化函数只需要改一个地方
4. **功能独立**: 每个置换算法都可以独立使用
5. **统一接口**: 所有算法都提供一致的函数接口
6. **新增算法**: 实现了FP4-Grid Greedy Search算法

## 注意事项

1. 重构后的代码保持了原有的功能不变
2. 所有重复的函数都移到了 `utils.py` 中
3. 各算法文件通过 `from . import utils` 导入通用函数
4. 确保不会引入新的问题或改变现有行为
5. 新增了独立的算法函数，便于单独使用和测试
6. FP4-Grid算法使用固定的随机种子确保结果可重现
