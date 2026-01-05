"""
导出器基类
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
import json

from ..utils import get_logger
from ..utils.config import EasyDict
from ..utils.registry import Registry

logger = get_logger(__name__)

# 导出器注册表
EXPORTER_REGISTRY = Registry('EXPORTER_REGISTRY')


class BaseExporter(ABC):
    """
    导出器基类
    
    负责将量化后的模型导出为特定格式
    """
    
    def __init__(
        self,
        model: nn.Module,
        quant_config: EasyDict,
        save_dir: str,
    ):
        """
        Args:
            model: 量化后的模型
            quant_config: 量化配置
            save_dir: 保存目录
        """
        self.model = model
        self.quant_config = quant_config
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def export(self, **kwargs) -> str:
        """
        执行导出
        
        Returns:
            导出文件/目录路径
        """
        pass
    
    def save_quant_config(self) -> None:
        """保存量化配置"""
        config_path = os.path.join(self.save_dir, 'quant_config.json')
        
        # 转换为可序列化格式
        config_dict = self._to_serializable(self.quant_config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Quant config saved to {config_path}")
    
    def _to_serializable(self, obj: Any) -> Any:
        """转换为可JSON序列化的格式"""
        if isinstance(obj, EasyDict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._to_serializable(obj.__dict__)
        else:
            return obj
    
    def collect_quantized_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        收集量化后的权重
        
        Returns:
            {layer_name: {'qweight': Tensor, 'scales': Tensor, 'zeros': Tensor}}
        """
        quantized_weights = {}
        
        for name, module in self.model.named_modules():
            # 检查是否是量化模块
            if hasattr(module, 'qweight'):
                quantized_weights[name] = {
                    'qweight': module.qweight,
                    'scales': module.scales if hasattr(module, 'scales') else None,
                    'zeros': module.zeros if hasattr(module, 'zeros') else None,
                    'g_idx': module.g_idx if hasattr(module, 'g_idx') else None,
                }
            elif hasattr(module, 'buf_scales'):
                # FakeQuant模块
                quantized_weights[name] = {
                    'weight': module.weight,
                    'scales': module.buf_scales,
                    'zeros': module.buf_zeros if hasattr(module, 'buf_zeros') else None,
                }
        
        return quantized_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 统计量化层
        num_quant_layers = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'qweight') or hasattr(module, 'buf_scales'):
                num_quant_layers += 1
        
        return {
            'total_params': total_params,
            'num_quant_layers': num_quant_layers,
            'quant_bits': self.quant_config.weight.get('bit', 4),
            'group_size': self.quant_config.weight.get('group_size', 128),
        }
