"""
模块包装层：FakeQuantLinear、RealQuantLinear等

提供统一的模块替换接口，支持:
- FakeQuantLinear: 假量化，用于评测
- EfficientFakeQuantLinear: 优化版假量化
- RealQuantLinear: 真量化，用于导出
- OriginFloatLinear: 保持浮点
- RotateLinear: 带旋转的线性层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, Tuple

from ..utils import get_logger
from .quantizer import QParams

logger = get_logger(__name__)


class FakeQuantLinear(nn.Module):
    """
    假量化线性层
    
    在forward中插入QDQ操作，模拟低比特数值效果
    权重仍然是浮点存储
    """
    
    def __init__(
        self,
        ori_module: nn.Linear,
        w_qdq: Callable,
        a_qdq: Optional[Callable] = None,
    ):
        super().__init__()
        
        # 保存原始权重和bias
        self.register_buffer('weight', ori_module.weight.data.clone())
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        self.w_qdq = w_qdq  # 权重量化函数
        self.a_qdq = a_qdq  # 激活量化函数
        
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
        
        # 校准阶段标志
        self.calib = True
        
        # 缓存量化后的权重
        self._tmp_weight: Optional[torch.Tensor] = None
    
    @classmethod
    def new(
        cls,
        ori_module: nn.Linear,
        w_qdq: Callable = None,
        a_qdq: Callable = None,
        **kwargs,
    ) -> 'FakeQuantLinear':
        """工厂方法"""
        return cls(ori_module, w_qdq, a_qdq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 激活量化（校准阶段跳过）
        if self.a_qdq is not None and not self.calib:
            x = self.a_qdq(x, self)
        
        # 权重量化
        if self._tmp_weight is None and self.w_qdq is not None:
            self._tmp_weight = self.w_qdq(self)
        
        weight = self._tmp_weight if self._tmp_weight is not None else self.weight
        
        return F.linear(x, weight, self.bias)
    
    def clear_cache(self) -> None:
        """清除缓存的量化权重"""
        self._tmp_weight = None
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, calib={self.calib}'


class EfficientFakeQuantLinear(nn.Module):
    """
    优化版假量化线性层
    
    在构造时就计算好量化权重，forward更快
    适合RTN/AWQ等不需要动态更新权重的算法
    """
    
    def __init__(
        self,
        ori_module: nn.Linear,
        w_qdq: Callable,
        a_qdq: Optional[Callable] = None,
    ):
        super().__init__()
        
        # 构造时就计算量化权重
        if w_qdq is not None:
            # w_qdq期望接收module，返回量化后的权重
            quantized_weight = w_qdq(ori_module)
            self.register_buffer('weight', quantized_weight)
        else:
            self.register_buffer('weight', ori_module.weight.data.clone())
        
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        self.a_qdq = a_qdq
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
        self.calib = True
    
    @classmethod
    def new(
        cls,
        ori_module: nn.Linear,
        w_qdq: Callable = None,
        a_qdq: Callable = None,
        **kwargs,
    ) -> 'EfficientFakeQuantLinear':
        return cls(ori_module, w_qdq, a_qdq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a_qdq is not None and not self.calib:
            x = self.a_qdq(x, self)
        return F.linear(x, self.weight, self.bias)


class RealQuantLinear(nn.Module):
    """
    真量化线性层基类
    
    存储整数权重，用于导出到推理后端
    """
    
    def __init__(
        self,
        ori_module: nn.Linear,
        w_q: Callable,
        quant_config: Dict[str, Any],
    ):
        super().__init__()
        
        # 调用w_q做真量化
        quant_result = w_q(ori_module)
        
        # 存储量化权重和参数
        self.register_buffer('q_weight', quant_result['q_weight'])
        self.register_buffer('weight_scale', quant_result['scale'])
        self.register_buffer('weight_zero_point', quant_result.get('zero_point'))
        
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        self.quant_config = quant_config
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
        self.bit = quant_result.get('bit', 4)
        self.granularity = quant_result.get('granularity', 'per_group')
        self.group_size = quant_result.get('group_size')
    
    @classmethod
    def new(
        cls,
        ori_module: nn.Linear,
        w_q: Callable = None,
        quant_config: Dict[str, Any] = None,
        **kwargs,
    ) -> 'RealQuantLinear':
        return cls(ori_module, w_q, quant_config or {})
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化权重（实际部署时会用专用kernel）
        weight = self._dequant_weight()
        output = F.linear(x, weight, self.bias)
        return output
    
    def _dequant_weight(self) -> torch.Tensor:
        """反量化权重"""
        q_weight = self.q_weight.float()
        scale = self.weight_scale
        zp = self.weight_zero_point if self.weight_zero_point is not None else 0
        
        # 根据granularity恢复形状
        if self.granularity == 'per_group' and self.group_size:
            # [out, n_groups, group_size] -> [out, in]
            out_dim = self.out_features
            in_dim = self.in_features
            
            # 扩展scale到正确形状
            if scale.dim() == 2:
                scale = scale.unsqueeze(-1)  # [out, n_groups, 1]
            if isinstance(zp, torch.Tensor) and zp.dim() == 2:
                zp = zp.unsqueeze(-1)
            
            weight = (q_weight - zp) * scale
            weight = weight.reshape(out_dim, -1)[:, :in_dim]
        else:
            weight = (q_weight - zp) * scale
            weight = weight.reshape(self.out_features, self.in_features)
        
        return weight.to(self.q_weight.device)


class VllmRealQuantLinear(RealQuantLinear):
    """vLLM后端专用的量化线性层"""
    
    def __init__(
        self,
        ori_module: nn.Linear,
        w_q: Callable,
        quant_config: Dict[str, Any],
    ):
        super().__init__(ori_module, w_q, quant_config)
        
        # vLLM特定的权重打包
        self._pack_weight()
    
    def _pack_weight(self) -> None:
        """打包权重为vLLM格式"""
        # 4bit打包：两个4bit值打包成一个int8
        if self.bit == 4:
            q_weight = self.q_weight
            # 简化的打包逻辑
            # 实际实现需要根据vLLM的具体格式
            pass


class OriginFloatLinear(nn.Module):
    """
    保持浮点的线性层
    
    用于等效变换后保持浮点精度
    """
    
    def __init__(self, ori_module: nn.Linear):
        super().__init__()
        self.register_buffer('weight', ori_module.weight.data.clone())
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
    
    @classmethod
    def new(cls, ori_module: nn.Linear, **kwargs) -> 'OriginFloatLinear':
        return cls(ori_module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RotateLinear(nn.Module):
    """
    带旋转的线性层
    
    在forward前做Hadamard变换，用于QuaRot等算法
    """
    
    def __init__(
        self,
        ori_module: nn.Linear,
        had_K: torch.Tensor,
        K: int,
        online_full_had: bool = False,
        online_partial_had: bool = False,
    ):
        super().__init__()
        
        self.register_buffer('weight', ori_module.weight.data.clone())
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        
        self.register_buffer('had_K', had_K)
        self.K = K
        self.online_full_had = online_full_had
        self.online_partial_had = online_partial_had
        
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
    
    @classmethod
    def new(
        cls,
        ori_module: nn.Linear,
        had_K: torch.Tensor = None,
        K: int = 0,
        online_full_had: bool = False,
        online_partial_had: bool = False,
        **kwargs,
    ) -> 'RotateLinear':
        if had_K is None:
            # 生成Hadamard矩阵
            had_K = cls._generate_hadamard(K)
        return cls(ori_module, had_K, K, online_full_had, online_partial_had)
    
    @staticmethod
    def _generate_hadamard(n: int) -> torch.Tensor:
        """生成Hadamard矩阵"""
        if n == 0:
            return torch.tensor([[1.0]])
        
        # 递归生成
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        
        return H[:n, :n] / (n ** 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用旋转
        if self.online_full_had or self.online_partial_had:
            x = self._rotate(x)
        
        return F.linear(x, self.weight, self.bias)
    
    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """应用Hadamard旋转"""
        # x: [..., in_features]
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        # 应用旋转
        if self.K > 0 and self.had_K is not None:
            x = x @ self.had_K.to(x.device).to(x.dtype)
        
        return x.reshape(original_shape)


# 模块类型映射表
_FAKEQUANT_LINEAR_MAP_ = {
    'fake_quant': FakeQuantLinear,
    'efficient_fake_quant': EfficientFakeQuantLinear,
}

_REALQUANT_LINEAR_MAP_ = {
    'vllm_quant': VllmRealQuantLinear,
    'real_quant': RealQuantLinear,
}


def get_quant_linear_cls(mode: str):
    """根据mode获取量化线性层类"""
    if mode in _FAKEQUANT_LINEAR_MAP_:
        return _FAKEQUANT_LINEAR_MAP_[mode]
    elif mode in _REALQUANT_LINEAR_MAP_:
        return _REALQUANT_LINEAR_MAP_[mode]
    elif mode == 'origin_float':
        return OriginFloatLinear
    elif mode == 'online_rotate':
        return RotateLinear
    else:
        raise ValueError(f"Unknown mode: {mode}")
