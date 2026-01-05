"""
GPTQ (Gradient-based Post-Training Quantization) 算法

使用二阶优化（Hessian矩阵）进行权重量化
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import math

from .base import BaseBlockwiseQuantization
from ..utils import ALGO_REGISTRY, get_logger
from ..quantization.modules import FakeQuantLinear

logger = get_logger(__name__)


@ALGO_REGISTRY
class GPTQ(BaseBlockwiseQuantization):
    """
    GPTQ量化算法
    
    特点:
    - 二阶优化：使用Hessian矩阵指导量化
    - 列级更新：逐列量化并补偿误差
    - 需要校准数据
    - 精度高，但速度较慢
    
    使用方式:
    ```yaml
    quant:
      language:
        method: GPTQ
        weight:
          bit: 4
          symmetric: true
          granularity: per_group
          group_size: 128
        gptq:
          block_size: 128
          percdamp: 0.01
          actorder: true
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
        
        # GPTQ特定参数
        gptq_config = quant_config.get('gptq', {})
        self.block_size = gptq_config.get('block_size', 128)
        self.percdamp = gptq_config.get('percdamp', 0.01)
        self.actorder = gptq_config.get('actorder', True)
        
        # Hessian缓存
        self.H = {}
        self.nsamples = {}
        
        if self.data_free:
            raise ValueError("GPTQ requires calibration data")
        
        logger.info(f"GPTQ initialized: block_size={self.block_size}, percdamp={self.percdamp}")
    
    def block_opt(self, block: nn.Module) -> None:
        """
        GPTQ的Block优化逻辑
        
        1. 收集输入构建Hessian
        2. 按列进行量化和误差补偿
        3. 替换模块
        """
        linears = self.model.get_block_linears(block)
        
        if not linears:
            logger.warning(f"Block {self.block_idx} has no linear layers, skip")
            return
        
        # 清空Hessian缓存
        self.H.clear()
        self.nsamples.clear()
        
        # 注册hooks收集输入
        handles = self._register_hessian_hooks(linears)
        
        # 跑校准数据，收集Hessian
        if self.input and 'data' in self.input:
            self.block_forward(block, self.input['data'])
        
        # 清理hooks
        for h in handles:
            h.remove()
        
        # 对每个线性层进行GPTQ优化
        for name, module in linears.items():
            if name in self.H:
                self._gptq_layer(name, module)
        
        # 替换模块
        params = self.get_replacement_params('fake_quant')
        self.model.replace_module_subset(
            module_cls=FakeQuantLinear,
            block=block,
            subset=linears,
            block_idx=self.block_idx,
            params_dict=params,
        )
        
        logger.debug(f"Block {self.block_idx}: GPTQ completed")
    
    def _register_hessian_hooks(
        self,
        linears: Dict[str, nn.Module],
    ) -> List:
        """注册hooks收集Hessian"""
        handles = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                
                # inp: [batch, seq, hidden]
                if inp.dim() == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                
                inp = inp.float()
                
                if name not in self.H:
                    self.H[name] = torch.zeros(
                        inp.shape[1], inp.shape[1],
                        device=inp.device, dtype=torch.float32
                    )
                    self.nsamples[name] = 0
                
                # 累积 H = X^T @ X
                self.H[name] += inp.T @ inp
                self.nsamples[name] += inp.shape[0]
            
            return hook
        
        for name, module in linears.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        
        return handles
    
    def _gptq_layer(self, name: str, module: nn.Module) -> None:
        """对单个层进行GPTQ优化"""
        W = module.weight.data.clone().float()
        H = self.H[name]
        
        # 归一化Hessian
        H = H / self.nsamples[name]
        
        # 添加阻尼
        damp = self.percdamp * torch.diag(H).mean()
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        
        # Cholesky分解
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except Exception as e:
            logger.warning(f"Cholesky failed for {name}, using pseudo-inverse: {e}")
            H = torch.linalg.pinv(H)
            H = torch.linalg.cholesky(H + 1e-6 * torch.eye(H.shape[0], device=H.device), upper=True)
        
        Hinv = H
        
        # 确定列处理顺序
        if self.actorder:
            perm = torch.argsort(torch.diag(self.H[name]), descending=True)
            W = W[:, perm]
            Hinv = Hinv[perm][:, perm]
        else:
            perm = torch.arange(W.shape[1], device=W.device)
        
        # 逐块量化
        Losses = torch.zeros(W.shape[0], device=W.device)
        
        for i1 in range(0, W.shape[1], self.block_size):
            i2 = min(i1 + self.block_size, W.shape[1])
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                # 量化
                q = self._quantize_column(w)
                Q1[:, i] = q
                
                # 计算误差
                err = (w - q) / d
                Err1[:, i] = err
                
                # 更新后续列
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            
            # 更新权重
            W[:, i1:i2] = Q1
            
            # 更新后续块
            if i2 < W.shape[1]:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        
        # 恢复顺序
        if self.actorder:
            invperm = torch.argsort(perm)
            W = W[:, invperm]
        
        # 更新模块权重
        module.weight.data = W.to(module.weight.dtype)
    
    def _quantize_column(self, w: torch.Tensor) -> torch.Tensor:
        """量化单列权重"""
        # 使用quantizer的逻辑
        qparams = self.wquantizer.get_tensor_qparams(w.unsqueeze(1))
        
        scale = qparams.scale.squeeze()
        zp = qparams.zero_point.squeeze() if qparams.zero_point is not None else 0
        
        q = torch.round(w / scale + zp)
        q = torch.clamp(q, qparams.qmin, qparams.qmax)
        
        # 反量化
        return (q - zp) * scale
