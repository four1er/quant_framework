"""
评估模块

提供量化模型的评估工具:
- 困惑度(PPL)评估
- 精度对比
- 速度测试
"""

from .evaluator import Evaluator, PPLEvaluator, AccuracyEvaluator, LayerwiseEvaluator, SpeedEvaluator
from .metrics import compute_ppl, compute_accuracy, compute_mse, compute_cosine_similarity, compute_snr

__all__ = [
    'Evaluator',
    'PPLEvaluator',
    'AccuracyEvaluator',
    'LayerwiseEvaluator',
    'SpeedEvaluator',
    'compute_ppl',
    'compute_accuracy',
    'compute_mse',
    'compute_cosine_similarity',
    'compute_snr',
]
