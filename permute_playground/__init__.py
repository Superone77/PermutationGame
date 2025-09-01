"""
Permutation Game - 用于对比不同置换方法对MXFP4/NVFP4量化影响的工具包

主要功能：
- FP4量化函数（支持e4m3、e8m0、bf16等格式）
- 多种置换算法（贪心、SLS、zigzag、FP4-Grid等）
- 量化误差评估和对比
- 合成数据生成和模型权重提取

使用方法：
from permute_playground import utils
from permute_playground import greedy_search
from permute_playground import random_reshuffle
from permute_playground import zigzag
from permute_playground import fp4_grid_greedy
from permute_playground import run_all_experiments
"""

from . import utils
from . import greedy_search
from . import random_reshuffle
from . import zigzag
from . import fp4_grid_greedy
from . import run_all_experiments

__version__ = "1.0.0"
__author__ = "Permutation Game Team"
