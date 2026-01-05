"""
RTN (Round-to-Nearest) 量化算法

最简单的量化算法，直接用MinMax或MSE标定，不做额外优化
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .base import BaseBlockwiseQuantization
from ..utils import ALGO_REGISTRY, get_logger
from ..quantization.modules import FakeQuantLinear, EfficientFakeQuantLinear

logger = get_logger(__name__)


@ALGO_REGISTRY
class RTN(BaseBlockwiseQuantization):
    """
    RTN量化算法
    
    特点:
    - 最简单的PTQ方法
    - 直接对权重做量化，不需要校准数据
    - 支持data-free模式
    - 速度快，但精度相对较低
    
    使用方式:
    ```yaml
    quant:
      language:
        method: RTN
        weight:
          bit: 4
          symmetric: true
          granularity: per_group
          group_size: 128
          calib_algo: minmax  # 或 mse
    ```
    """
    
    def __init__(
        self,
        model,
        quant_config,
        input,
        padding_mask,
        config,
    ):
        super().__init__(model, quant_config, input, padding_mask, config)
        
        # RTN可以在data-free模式下工作
        logger.info("RTN initialized")
    
    def block_opt(self, block: nn.Module) -> None:
        """
        RTN的Block优化逻辑
        
        只需要替换模块，不做权重变换
        """
        # 获取Block内的线性层
        linears = self.model.get_block_linears(block)
        
        if not linears:
            logger.warning(f"Block {self.block_idx} has no linear layers, skip")
            return
        
        # 使用EfficientFakeQuantLinear（构造时就计算好量化权重）
        params = self.get_replacement_params('efficient_fake_quant')
        
        self.model.replace_module_subset(
            module_cls=EfficientFakeQuantLinear,
            block=block,
            subset=linears,
            block_idx=self.block_idx,
            params_dict=params,
        )
        
        logger.debug(f"Block {self.block_idx}: Replaced {len(linears)} layers with RTN")
