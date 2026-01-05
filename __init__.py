"""
LightQuant: 工业级大模型量化框架

支持特性:
- 多种量化算法: RTN, AWQ, GPTQ, SmoothQuant等
- 多种模型: LLaMA, Qwen, Gemma等
- 多种后端导出: vLLM, HuggingFace, GGUF等
- 多种量化粒度: per_tensor, per_channel, per_group, per_head
- Fake/Real双路径一致性保证

七层架构:
1. Entry层: 配置解析、分布式初始化
2. Config/Registry层: 配置管理、插件注册
3. Model抽象层: 统一模型接口
4. Blockwise调度层: 按Block调度优化
5. Quantizer层: 量化核心逻辑
6. Algorithm层: 算法插件(RTN/AWQ/GPTQ/SmoothQuant)
7. Eval/Export层: 评估与导出
"""

__version__ = "0.1.0"
__author__ = "LightQuant Team"

# 核心组件导出
from .utils import (
    load_config,
    check_config,
    get_logger,
    setup_logger,
    ALGO_REGISTRY,
    MODEL_REGISTRY,
)

from .quantization import (
    IntegerQuantizer,
    FloatQuantizer,
    FakeQuantLinear,
    RealQuantLinear,
)

from .algorithms import (
    BlockwiseOpt,
    BaseBlockwiseQuantization,
    RTNQuantization,
    AWQQuantization,
    GPTQQuantization,
    SmoothQuantQuantization,
)

from .eval import (
    Evaluator,
    PPLEvaluator,
    AccuracyEvaluator,
)

from .export import (
    BaseExporter,
    VLLMExporter,
    HuggingFaceExporter,
    GGUFExporter,
)

__all__ = [
    # Version
    '__version__',
    # Config
    'load_config',
    'check_config',
    'get_logger',
    'setup_logger',
    # Registry
    'ALGO_REGISTRY',
    'MODEL_REGISTRY',
    # Quantizer
    'IntegerQuantizer',
    'FloatQuantizer',
    # Modules
    'FakeQuantLinear',
    'RealQuantLinear',
    # Algorithms
    'BlockwiseOpt',
    'BaseBlockwiseQuantization',
    'RTNQuantization',
    'AWQQuantization',
    'GPTQQuantization',
    'SmoothQuantQuantization',
    # Eval
    'Evaluator',
    'PPLEvaluator',
    'AccuracyEvaluator',
    # Export
    'BaseExporter',
    'VLLMExporter',
    'HuggingFaceExporter',
    'GGUFExporter',
]
