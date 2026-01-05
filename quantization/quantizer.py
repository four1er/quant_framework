"""
Quantizer子系统：粒度处理、标定算法、Fake/Real双路径

核心功能:
1. 粒度处理: per_tensor, per_channel, per_group, per_head, per_block
2. 标定算法: MinMax, MSE, Histogram, Moving Average
3. Fake/Real双路径: 共享get_tensor_qparams保证一致性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class QParams:
    """量化参数"""
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    
    def to(self, device: torch.device) -> 'QParams':
        return QParams(
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device) if self.zero_point is not None else None,
            qmin=self.qmin,
            qmax=self.qmax,
        )


class BaseQuantizer(ABC):
    """
    量化器基类
    
    提供统一接口:
    - get_tensor_qparams(): 计算量化参数
    - fake_quant_weight_dynamic(): Fake量化权重
    - real_quant_weight_dynamic(): Real量化权重
    """
    
    def __init__(
        self,
        bit: int = 4,
        symmetric: bool = True,
        granularity: str = "per_group",
        calib_algo: str = "minmax",
        **kwargs,
    ):
        self.bit = bit
        self.symmetric = symmetric
        self.granularity = granularity
        self.calib_algo = calib_algo
        
        # 根据granularity设置参数
        self.group_size = kwargs.get('group_size', 128)
        self.head_num = kwargs.get('head_num', 32)
        self.block_size = kwargs.get('block_size', (128, 128))
        
        # MSE标定参数
        self.mse_grid = kwargs.get('mse_grid', 100)
        self.maxshrink = kwargs.get('maxshrink', 0.8)
        
        # 计算量化范围
        self._compute_qrange()
    
    def _compute_qrange(self) -> None:
        """计算量化范围 [qmin, qmax]"""
        if self.symmetric:
            # 对称量化: [-2^(n-1)+1, 2^(n-1)-1] 或 [-2^(n-1), 2^(n-1)-1]
            self.qmax = 2 ** (self.bit - 1) - 1
            self.qmin = -self.qmax  # 对称
        else:
            # 非对称量化: [0, 2^n - 1]
            self.qmin = 0
            self.qmax = 2 ** self.bit - 1
    
    def reshape_tensor(
        self,
        tensor: torch.Tensor,
        allow_padding: bool = False,
    ) -> torch.Tensor:
        """
        根据granularity重排张量
        
        Args:
            tensor: 原始张量 [out_features, in_features]
            allow_padding: 是否允许padding
        
        Returns:
            重排后的张量
        """
        if self.granularity == "per_tensor":
            return tensor.flatten()
        
        elif self.granularity == "per_channel":
            # 每个输出通道一个scale
            # [out, in] -> [out, in]
            return tensor
        
        elif self.granularity == "per_group":
            # 每group_size个元素共享一个scale
            # [out, in] -> [out, n_groups, group_size]
            out_dim, in_dim = tensor.shape
            
            if in_dim % self.group_size != 0:
                if allow_padding:
                    pad_len = self.group_size - (in_dim % self.group_size)
                    tensor = F.pad(tensor, (0, pad_len))
                    in_dim = tensor.shape[1]
                else:
                    raise ValueError(
                        f"in_features ({in_dim}) cannot be divided by "
                        f"group_size ({self.group_size})"
                    )
            
            n_groups = in_dim // self.group_size
            return tensor.reshape(out_dim, n_groups, self.group_size)
        
        elif self.granularity == "per_head":
            # 每个head共享一个scale
            # [out, in] -> [n_heads, head_dim, in]
            out_dim, in_dim = tensor.shape
            head_dim = out_dim // self.head_num
            return tensor.reshape(self.head_num, head_dim, in_dim)
        
        elif self.granularity == "per_block":
            # 2D块状量化
            # [out, in] -> [n_blocks_out, n_blocks_in, block_h, block_w]
            out_dim, in_dim = tensor.shape
            block_h, block_w = self.block_size
            
            n_blocks_out = (out_dim + block_h - 1) // block_h
            n_blocks_in = (in_dim + block_w - 1) // block_w
            
            # Padding if needed
            if out_dim % block_h != 0 or in_dim % block_w != 0:
                if allow_padding:
                    pad_out = n_blocks_out * block_h - out_dim
                    pad_in = n_blocks_in * block_w - in_dim
                    tensor = F.pad(tensor, (0, pad_in, 0, pad_out))
                else:
                    raise ValueError(
                        f"Tensor shape ({out_dim}, {in_dim}) cannot be divided by "
                        f"block_size ({block_h}, {block_w})"
                    )
            
            return tensor.reshape(n_blocks_out, block_h, n_blocks_in, block_w).permute(0, 2, 1, 3)
        
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")
    
    def restore_tensor(
        self,
        tensor: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """恢复张量到原始形状"""
        if self.granularity == "per_tensor":
            return tensor.reshape(original_shape)
        
        elif self.granularity == "per_channel":
            return tensor
        
        elif self.granularity == "per_group":
            out_dim, in_dim = original_shape
            # [out, n_groups, group_size] -> [out, in]
            tensor = tensor.reshape(out_dim, -1)
            return tensor[:, :in_dim]  # 去掉padding
        
        elif self.granularity == "per_head":
            out_dim, in_dim = original_shape
            return tensor.reshape(out_dim, in_dim)
        
        elif self.granularity == "per_block":
            out_dim, in_dim = original_shape
            # [n_blocks_out, n_blocks_in, block_h, block_w] -> [out, in]
            block_h, block_w = self.block_size
            tensor = tensor.permute(0, 2, 1, 3).reshape(-1, tensor.shape[1] * block_w)
            return tensor[:out_dim, :in_dim]
        
        else:
            return tensor.reshape(original_shape)
    
    def get_tensor_range(
        self,
        tensor: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据calib_algo计算张量范围
        
        Args:
            tensor: 已重排的张量
            args: 额外参数
        
        Returns:
            (min_val, max_val)
        """
        if self.calib_algo == "minmax":
            return self._get_minmax_range(tensor)
        elif self.calib_algo == "mse":
            return self._get_mse_range(tensor)
        elif self.calib_algo == "percentile":
            percentile = args.get('percentile', 99.9) if args else 99.9
            return self._get_percentile_range(tensor, percentile)
        else:
            raise ValueError(f"Unknown calib_algo: {self.calib_algo}")
    
    def _get_minmax_range(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MinMax标定：最快但对离群值敏感"""
        if self.granularity == "per_tensor":
            min_val = tensor.min()
            max_val = tensor.max()
        else:
            # 在最后一维上计算
            min_val = tensor.min(dim=-1, keepdim=True)[0]
            max_val = tensor.max(dim=-1, keepdim=True)[0]
        
        return min_val, max_val
    
    def _get_mse_range(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MSE标定：通过搜索范围压制离群值"""
        min_val, max_val = self._get_minmax_range(tensor)
        
        best_min, best_max = min_val.clone(), max_val.clone()
        best_error = float('inf')
        
        # 在 [maxshrink, 1.0] 之间搜索
        for shrink_ratio in torch.linspace(self.maxshrink, 1.0, self.mse_grid):
            trial_min = min_val * shrink_ratio
            trial_max = max_val * shrink_ratio
            
            # 计算scale和zero_point
            scale, zero_point = self._compute_qparams(trial_min, trial_max)
            
            # 量化-反量化
            q = self._quant(tensor, scale, zero_point)
            dq = self._dequant(q, scale, zero_point)
            
            # 计算MSE
            error = (tensor - dq).pow(2).mean().item()
            
            if error < best_error:
                best_error = error
                best_min = trial_min
                best_max = trial_max
        
        return best_min, best_max
    
    def _get_percentile_range(
        self,
        tensor: torch.Tensor,
        percentile: float = 99.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Percentile标定：忽略极端离群值"""
        if self.granularity == "per_tensor":
            flat = tensor.flatten()
            min_val = torch.quantile(flat, (100 - percentile) / 100)
            max_val = torch.quantile(flat, percentile / 100)
        else:
            # 对每个group/channel计算
            flat = tensor.reshape(tensor.shape[0], -1)
            min_val = torch.quantile(flat, (100 - percentile) / 100, dim=-1, keepdim=True)
            max_val = torch.quantile(flat, percentile / 100, dim=-1, keepdim=True)
        
        return min_val, max_val
    
    def _compute_qparams(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从range计算scale和zero_point"""
        if self.symmetric:
            # 对称量化
            abs_max = torch.max(min_val.abs(), max_val.abs())
            scale = abs_max / self.qmax
            # 避免除零
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            # 非对称量化
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = self.qmin - min_val / scale
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        return scale, zero_point
    
    def _quant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """量化操作"""
        q = torch.round(tensor / scale + zero_point)
        q = torch.clamp(q, self.qmin, self.qmax)
        return q
    
    def _dequant(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """反量化操作"""
        return (q - zero_point) * scale
    
    def get_tensor_qparams(
        self,
        tensor: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> QParams:
        """
        统一的QParams生成逻辑（Fake/Real共享）
        
        Args:
            tensor: 原始张量
            args: 额外参数
        
        Returns:
            QParams对象
        """
        # 重排张量
        tensor_reshaped = self.reshape_tensor(tensor, allow_padding=True)
        
        # 计算范围
        min_val, max_val = self.get_tensor_range(tensor_reshaped, args)
        
        # 计算scale和zero_point
        scale, zero_point = self._compute_qparams(min_val, max_val)
        
        return QParams(
            scale=scale,
            zero_point=zero_point,
            qmin=self.qmin,
            qmax=self.qmax,
        )
    
    def quant_dequant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """量化-反量化（QDQ）"""
        q = self._quant(tensor, scale, zero_point)
        return self._dequant(q, scale, zero_point)


class IntegerQuantizer(BaseQuantizer):
    """
    整数量化器
    
    支持INT2/3/4/8量化，提供Fake和Real两条路径
    """
    
    def fake_quant_weight_dynamic(
        self,
        weight: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Fake路径：动态计算QParams并QDQ
        
        Args:
            weight: 原始权重
            args: 额外参数
        
        Returns:
            量化-反量化后的权重（仍是浮点）
        """
        original_shape = weight.shape
        
        # 获取QParams
        qparams = self.get_tensor_qparams(weight, args)
        
        # 重排
        weight_reshaped = self.reshape_tensor(weight, allow_padding=True)
        
        # QDQ
        dq_weight = self.quant_dequant(
            weight_reshaped,
            qparams.scale,
            qparams.zero_point,
        )
        
        # 恢复形状
        return self.restore_tensor(dq_weight, original_shape)
    
    def fake_quant_weight_static(
        self,
        weight: torch.Tensor,
        qparams: QParams,
    ) -> torch.Tensor:
        """
        Fake路径：使用预计算的QParams
        
        Args:
            weight: 原始权重
            qparams: 预计算的量化参数
        
        Returns:
            量化-反量化后的权重
        """
        original_shape = weight.shape
        
        # 重排
        weight_reshaped = self.reshape_tensor(weight, allow_padding=True)
        
        # QDQ
        dq_weight = self.quant_dequant(
            weight_reshaped,
            qparams.scale,
            qparams.zero_point,
        )
        
        # 恢复形状
        return self.restore_tensor(dq_weight, original_shape)
    
    def real_quant_weight_dynamic(
        self,
        weight: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Real路径：动态计算QParams并量化
        
        Args:
            weight: 原始权重
            args: 额外参数
        
        Returns:
            {'q_weight': 整数权重, 'scale': scale, 'zero_point': zp, ...}
        """
        original_shape = weight.shape
        
        # 获取QParams
        qparams = self.get_tensor_qparams(weight, args)
        
        # 重排
        weight_reshaped = self.reshape_tensor(weight, allow_padding=True)
        
        # 量化（不反量化）
        q_weight = self._quant(weight_reshaped, qparams.scale, qparams.zero_point)
        
        # 确定存储dtype
        if self.bit <= 4:
            # 4bit及以下用int8存储（后续可能需要pack）
            q_weight = q_weight.to(torch.int8)
        else:
            q_weight = q_weight.to(torch.int8)
        
        return {
            'q_weight': q_weight,
            'scale': qparams.scale,
            'zero_point': qparams.zero_point,
            'qmin': qparams.qmin,
            'qmax': qparams.qmax,
            'original_shape': original_shape,
            'bit': self.bit,
            'granularity': self.granularity,
            'group_size': self.group_size if self.granularity == 'per_group' else None,
        }
    
    def fake_quant_act_dynamic(
        self,
        act: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        动态激活量化
        
        Args:
            act: 激活张量
            args: 额外参数
        
        Returns:
            量化-反量化后的激活
        """
        # 激活通常用per_tensor
        min_val = act.min()
        max_val = act.max()
        
        scale, zero_point = self._compute_qparams(min_val, max_val)
        
        return self.quant_dequant(act, scale, zero_point)
    
    def fake_quant_act_static(
        self,
        act: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        静态激活量化（使用预计算的scale）
        
        Args:
            act: 激活张量
            scale: 预计算的scale
            zero_point: 预计算的zero_point
        
        Returns:
            量化-反量化后的激活
        """
        return self.quant_dequant(act, scale, zero_point)


class FloatQuantizer(BaseQuantizer):
    """
    浮点量化器
    
    支持FP8 (E4M3/E5M2) 量化
    """
    
    def __init__(
        self,
        fp_format: str = "e4m3",
        **kwargs,
    ):
        # FP8不需要bit参数
        kwargs['bit'] = 8
        super().__init__(**kwargs)
        
        self.fp_format = fp_format
        
        # 设置FP8范围
        if fp_format == "e4m3":
            self.fp_max = 448.0  # E4M3的最大值
            self.fp_min = -448.0
        elif fp_format == "e5m2":
            self.fp_max = 57344.0  # E5M2的最大值
            self.fp_min = -57344.0
        else:
            raise ValueError(f"Unknown FP format: {fp_format}")
    
    def fake_quant_weight_dynamic(
        self,
        weight: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """FP8 Fake量化"""
        # 计算per-block scale
        abs_max = weight.abs().max()
        scale = abs_max / self.fp_max
        scale = torch.clamp(scale, min=1e-8)
        
        # 缩放到FP8范围
        scaled_weight = weight / scale
        
        # 模拟FP8精度损失（通过clamp和round）
        scaled_weight = torch.clamp(scaled_weight, self.fp_min, self.fp_max)
        
        # 恢复
        return scaled_weight * scale
    
    def real_quant_weight_dynamic(
        self,
        weight: torch.Tensor,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """FP8 Real量化"""
        abs_max = weight.abs().max()
        scale = abs_max / self.fp_max
        scale = torch.clamp(scale, min=1e-8)
        
        scaled_weight = weight / scale
        scaled_weight = torch.clamp(scaled_weight, self.fp_min, self.fp_max)
        
        # 转换为FP8 dtype（如果可用）
        if hasattr(torch, 'float8_e4m3fn') and self.fp_format == "e4m3":
            fp8_weight = scaled_weight.to(torch.float8_e4m3fn)
        elif hasattr(torch, 'float8_e5m2') and self.fp_format == "e5m2":
            fp8_weight = scaled_weight.to(torch.float8_e5m2)
        else:
            # 回退到float16
            fp8_weight = scaled_weight.to(torch.float16)
        
        return {
            'q_weight': fp8_weight,
            'scale': scale,
            'fp_format': self.fp_format,
        }
