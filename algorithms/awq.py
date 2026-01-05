"""
AWQ (Activation-aware Weight Quantization) 算法

通过搜索通道缩放因子，保护重要通道，减少量化误差
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from .base import BaseBlockwiseQuantization
from ..utils import ALGO_REGISTRY, get_logger
from ..quantization.modules import FakeQuantLinear

logger = get_logger(__name__)


@ALGO_REGISTRY
class AWQ(BaseBlockwiseQuantization):
    """
    AWQ量化算法
    
    特点:
    - 激活感知：根据激活分布搜索最优缩放因子
    - 等效变换：把缩放吸收到权重中
    - 需要校准数据
    - 精度优于RTN
    
    使用方式:
    ```yaml
    quant:
      language:
        method: AWQ
        weight:
          bit: 4
          symmetric: false
          granularity: per_group
          group_size: 128
        awq:
          n_grid: 20
          max_shrink: 0.5
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
        
        # AWQ特定参数
        awq_config = quant_config.get('awq', {})
        self.n_grid = awq_config.get('n_grid', 20)
        self.max_shrink = awq_config.get('max_shrink', 0.5)
        
        if self.data_free:
            logger.warning("AWQ works better with calibration data")
        
        logger.info(f"AWQ initialized: n_grid={self.n_grid}, max_shrink={self.max_shrink}")
    
    def block_opt(self, block: nn.Module) -> None:
        """
        AWQ的Block优化逻辑
        
        1. 收集输入激活
        2. 搜索最优缩放因子
        3. 应用缩放变换
        4. 替换模块
        """
        linears = self.model.get_block_linears(block)
        
        if not linears:
            logger.warning(f"Block {self.block_idx} has no linear layers, skip")
            return
        
        # 按subset分组处理
        subsets = self.get_subsets_in_block(block)
        
        for subset_name, subset in subsets.items():
            if not subset:
                continue
            
            if not self.data_free:
                # 收集输入激活
                input_samples = self._collect_inputs(block, subset)
                
                # 搜索并应用缩放
                self._search_and_apply_scales(subset, input_samples)
            
        # 替换模块
        params = self.get_replacement_params('fake_quant')
        self.model.replace_module_subset(
            module_cls=FakeQuantLinear,
            block=block,
            subset=linears,
            block_idx=self.block_idx,
            params_dict=params,
        )
        
        logger.debug(f"Block {self.block_idx}: AWQ completed")
    
    def _collect_inputs(
        self,
        block: nn.Module,
        subset: Dict[str, nn.Module],
    ) -> Dict[str, List[torch.Tensor]]:
        """收集线性层的输入激活"""
        input_cache = {name: [] for name in subset.keys()}
        handles = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                input_cache[name].append(inp.detach().cpu())
            return hook
        
        # 注册hooks
        for name, module in subset.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        
        # 跑校准数据
        if self.input and 'data' in self.input:
            self.block_forward(block, self.input['data'])
        
        # 清理hooks
        for h in handles:
            h.remove()
        
        return input_cache
    
    def _search_and_apply_scales(
        self,
        subset: Dict[str, nn.Module],
        input_samples: Dict[str, List[torch.Tensor]],
    ) -> None:
        """搜索最优缩放因子并应用"""
        for name, module in subset.items():
            if name not in input_samples or not input_samples[name]:
                continue
            
            # 合并输入样本
            inputs = torch.cat([x.to(module.weight.device) for x in input_samples[name]], dim=0)
            
            # 计算输入的通道重要性
            importance = self._compute_importance(inputs)
            
            # 搜索最优缩放
            best_scale = self._grid_search_scale(module, inputs, importance)
            
            # 应用缩放到权重
            self._apply_scale(module, best_scale)
    
    def _compute_importance(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算通道重要性
        
        使用输入激活的绝对值均值作为重要性指标
        """
        # inputs: [batch, seq, hidden]
        if inputs.dim() == 3:
            importance = inputs.abs().mean(dim=(0, 1))
        else:
            importance = inputs.abs().mean(dim=0)
        
        return importance
    
    def _grid_search_scale(
        self,
        module: nn.Module,
        inputs: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """网格搜索最优缩放因子"""
        weight = module.weight.data
        device = weight.device
        
        best_error = float('inf')
        best_scale = torch.ones(weight.shape[1], device=device)
        
        # 归一化重要性
        importance = importance.to(device)
        importance = importance / importance.max()
        
        # 网格搜索
        for ratio in torch.linspace(0, 1, self.n_grid):
            # 计算缩放因子: scale = importance^ratio
            scale = importance.pow(ratio).clamp(min=self.max_shrink)
            
            # 应用缩放后的权重
            scaled_weight = weight * scale.view(1, -1)
            
            # 量化误差
            q_weight = self.wquantizer.fake_quant_weight_dynamic(scaled_weight)
            
            # 计算输出误差
            if inputs.dim() == 3:
                flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            else:
                flat_inputs = inputs
            
            orig_out = flat_inputs @ weight.T
            quant_out = flat_inputs @ q_weight.T
            
            error = (orig_out - quant_out).pow(2).mean().item()
            
            if error < best_error:
                best_error = error
                best_scale = scale.clone()
        
        return best_scale
    
    def _apply_scale(self, module: nn.Module, scale: torch.Tensor) -> None:
        """应用缩放到权重"""
        # 权重乘以scale
        module.weight.data = module.weight.data * scale.view(1, -1)
