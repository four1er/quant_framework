"""
SmoothQuant 算法

通过平滑激活分布，将量化难度从激活转移到权重
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .base import BaseBlockwiseQuantization
from ..utils import ALGO_REGISTRY, get_logger
from ..quantization.modules import FakeQuantLinear

logger = get_logger(__name__)


@ALGO_REGISTRY
class SmoothQuant(BaseBlockwiseQuantization):
    """
    SmoothQuant量化算法
    
    特点:
    - 平滑激活：减少激活的离群值
    - 等效变换：把平滑因子吸收到权重
    - 适合W8A8量化
    - 需要校准数据
    
    使用方式:
    ```yaml
    quant:
      language:
        method: SmoothQuant
        weight:
          bit: 8
          symmetric: true
          granularity: per_channel
        act:
          bit: 8
          symmetric: true
          granularity: per_tensor
        smoothquant:
          alpha: 0.5
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
        
        # SmoothQuant特定参数
        sq_config = quant_config.get('smoothquant', {})
        self.alpha = sq_config.get('alpha', 0.5)
        
        # 激活统计缓存
        self.act_scales = {}
        
        if self.data_free:
            logger.warning("SmoothQuant works better with calibration data")
        
        logger.info(f"SmoothQuant initialized: alpha={self.alpha}")
    
    def block_opt(self, block: nn.Module) -> None:
        """
        SmoothQuant的Block优化逻辑
        
        1. 收集激活统计
        2. 计算平滑因子
        3. 应用平滑变换
        4. 替换模块
        """
        linears = self.model.get_block_linears(block)
        
        if not linears:
            logger.warning(f"Block {self.block_idx} has no linear layers, skip")
            return
        
        # 清空缓存
        self.act_scales.clear()
        
        if not self.data_free:
            # 收集激活统计
            self._collect_act_scales(block, linears)
            
            # 应用平滑变换
            self._apply_smoothing(block, linears)
        
        # 替换模块
        params = self.get_replacement_params('fake_quant')
        self.model.replace_module_subset(
            module_cls=FakeQuantLinear,
            block=block,
            subset=linears,
            block_idx=self.block_idx,
            params_dict=params,
        )
        
        logger.debug(f"Block {self.block_idx}: SmoothQuant completed")
    
    def _collect_act_scales(
        self,
        block: nn.Module,
        linears: Dict[str, nn.Module],
    ) -> None:
        """收集激活的通道最大值"""
        handles = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                
                # 计算通道最大值
                if inp.dim() == 3:
                    act_scale = inp.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    act_scale = inp.abs().max(dim=0)[0]
                
                if name not in self.act_scales:
                    self.act_scales[name] = act_scale.cpu()
                else:
                    self.act_scales[name] = torch.max(
                        self.act_scales[name],
                        act_scale.cpu()
                    )
            
            return hook
        
        for name, module in linears.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        
        # 跑校准数据
        if self.input and 'data' in self.input:
            self.block_forward(block, self.input['data'])
        
        for h in handles:
            h.remove()
    
    def _apply_smoothing(
        self,
        block: nn.Module,
        linears: Dict[str, nn.Module],
    ) -> None:
        """应用平滑变换"""
        # 获取归一化层
        norms = self.model.get_norm_layers(block) if hasattr(self.model, 'get_norm_layers') else {}
        
        for name, module in linears.items():
            if name not in self.act_scales:
                continue
            
            act_scale = self.act_scales[name].to(module.weight.device)
            weight_scale = module.weight.abs().max(dim=0)[0]
            
            # 计算平滑因子: s = act_scale^alpha / weight_scale^(1-alpha)
            smooth_scale = (act_scale.pow(self.alpha) / 
                          weight_scale.pow(1 - self.alpha).clamp(min=1e-5))
            smooth_scale = smooth_scale.clamp(min=1e-5)
            
            # 应用到权重: W = W * s
            module.weight.data = module.weight.data * smooth_scale.view(1, -1)
            
            # 如果有对应的LayerNorm，也需要调整
            # 这里简化处理，实际需要根据模型结构找到对应的LN
