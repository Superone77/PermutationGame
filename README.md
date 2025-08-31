# OLMoE-1B-7B-0125-Instruct Permutation验证项目

这个项目专门用于验证OLMoE-1B-7B-0125-Instruct模型中的permutation操作，帮助理解模型权重的重要性分布和结构。

## 项目结构

```
PermuteQuant/
├── permutation_validation.ipynb  # 主要的验证notebook
├── requirements.txt              # 依赖包列表
├── README.md                    # 项目说明文档
└── weights/                     # 权重保存目录（运行后自动创建）
```

## 环境设置

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 启动Jupyter：
```bash
jupyter notebook
```

## 使用方法

1. 打开 `permutation_validation.ipynb`
2. 按照notebook中的步骤执行：
   - 第一步：加载OLMoE-1B-7B-0125-Instruct模型并提取权重
   - 后续步骤：进行permutation验证（待实现）

## 注意事项

- 确保有足够的内存来加载模型
- 如果使用GPU，确保CUDA环境正确配置
- 模型名称可能需要根据实际情况调整
- 首次运行可能需要下载模型，请确保网络连接正常

## 功能特性

- 自动检测和加载OLMoE模型
- 智能识别MoE、注意力、MLP等不同类型的权重
- 权重统计分析和可视化
- 权重数据保存和加载
- 为后续permutation验证做准备

## 下一步计划

- 实现不同的permutation策略
- 分析permutation对模型性能的影响
- 可视化permutation前后的权重变化
- 量化permutation的效果 # PermutationGame
