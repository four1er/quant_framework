"""
vLLM格式导出器

导出为vLLM兼容的量化模型格式
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os
import json
from safetensors.torch import save_file

from .base import BaseExporter, EXPORTER_REGISTRY
from ..utils import get_logger
from ..utils.config import EasyDict
from ..quantization.modules import RealQuantLinear

logger = get_logger(__name__)


@EXPORTER_REGISTRY.register('vllm')
class VLLMExporter(BaseExporter):
    """
    vLLM格式导出器
    
    支持AWQ和GPTQ格式，兼容vLLM推理引擎
    """
    
    def __init__(
        self,
        model: nn.Module,
        quant_config: EasyDict,
        save_dir: str,
        quant_method: str = 'awq',  # 'awq' or 'gptq'
    ):
        super().__init__(model, quant_config, save_dir)
        self.quant_method = quant_method
    
    def export(
        self,
        tokenizer=None,
        use_safetensors: bool = True,
        **kwargs,
    ) -> str:
        """
        导出为vLLM格式
        
        Args:
            tokenizer: tokenizer对象，用于保存tokenizer配置
            use_safetensors: 是否使用safetensors格式
        
        Returns:
            导出目录路径
        """
        logger.info(f"Exporting model to vLLM format ({self.quant_method})...")
        
        # 1. 转换为RealQuant格式
        self._convert_to_real_quant()
        
        # 2. 收集权重
        state_dict = self._collect_vllm_weights()
        
        # 3. 保存权重
        if use_safetensors:
            self._save_safetensors(state_dict)
        else:
            self._save_pytorch(state_dict)
        
        # 4. 保存配置
        self._save_vllm_config()
        
        # 5. 保存tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(self.save_dir)
        
        # 6. 保存量化配置
        self.save_quant_config()
        
        logger.info(f"Model exported to {self.save_dir}")
        return self.save_dir
    
    def _convert_to_real_quant(self) -> None:
        """将FakeQuant模块转换为RealQuant"""
        from ..quantization.modules import FakeQuantLinear, EfficientFakeQuantLinear
        
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (FakeQuantLinear, EfficientFakeQuantLinear)):
                # 获取量化参数
                weight = module.weight.data
                
                # 执行真量化
                if hasattr(module, 'buf_scales'):
                    scales = module.buf_scales
                    zeros = module.buf_zeros if hasattr(module, 'buf_zeros') else None
                else:
                    # 动态计算
                    from ..quantization.quantizer import IntegerQuantizer
                    quantizer = IntegerQuantizer(
                        bit=self.quant_config.weight.get('bit', 4),
                        symmetric=self.quant_config.weight.get('symmetric', True),
                        granularity=self.quant_config.weight.get('granularity', 'per_group'),
                        group_size=self.quant_config.weight.get('group_size', 128),
                    )
                    result = quantizer.real_quant_weight_dynamic(weight, args={})
                    scales = result['scales']
                    zeros = result['zeros']
                
                # 创建RealQuantLinear
                real_quant = RealQuantLinear.new(
                    module,
                    w_q=lambda m: {
                        'qweight': self._pack_weight(weight, scales, zeros),
                        'scales': scales,
                        'zeros': zeros,
                    },
                    quant_config={
                        'bits': self.quant_config.weight.get('bit', 4),
                        'group_size': self.quant_config.weight.get('group_size', 128),
                    },
                )
                
                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model
                
                setattr(parent, child_name, real_quant)
        
        logger.info("Converted FakeQuant modules to RealQuant")
    
    def _pack_weight(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        打包权重为整数格式
        
        Args:
            weight: FP权重
            scales: 缩放因子
            zeros: 零点
        
        Returns:
            打包后的整数权重
        """
        bits = self.quant_config.weight.get('bit', 4)
        group_size = self.quant_config.weight.get('group_size', 128)
        
        # 量化
        out_features, in_features = weight.shape
        
        # Reshape for group quantization
        weight_reshaped = weight.reshape(out_features, -1, group_size)
        scales_reshaped = scales.reshape(out_features, -1, 1)
        
        if zeros is not None:
            zeros_reshaped = zeros.reshape(out_features, -1, 1)
            qweight = torch.round(weight_reshaped / scales_reshaped + zeros_reshaped)
        else:
            qweight = torch.round(weight_reshaped / scales_reshaped)
        
        # Clamp
        qmin = 0 if zeros is not None else -(2 ** (bits - 1))
        qmax = 2 ** bits - 1 if zeros is not None else 2 ** (bits - 1) - 1
        qweight = qweight.clamp(qmin, qmax).to(torch.int32)
        
        # Reshape back
        qweight = qweight.reshape(out_features, in_features)
        
        # Pack (for 4-bit, pack 8 values into one int32)
        if bits == 4:
            qweight = self._pack_int4(qweight)
        
        return qweight
    
    def _pack_int4(self, qweight: torch.Tensor) -> torch.Tensor:
        """将4-bit权重打包到int32"""
        out_features, in_features = qweight.shape
        
        # 确保in_features是8的倍数
        if in_features % 8 != 0:
            pad_size = 8 - (in_features % 8)
            qweight = torch.nn.functional.pad(qweight, (0, pad_size))
            in_features = qweight.shape[1]
        
        # Reshape to pack 8 values
        qweight = qweight.reshape(out_features, -1, 8)
        
        # Pack
        packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=qweight.device)
        for i in range(8):
            packed |= (qweight[:, :, i].to(torch.int32) & 0xF) << (i * 4)
        
        return packed
    
    def _collect_vllm_weights(self) -> Dict[str, torch.Tensor]:
        """收集vLLM格式的权重"""
        state_dict = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'qweight'):
                # 量化权重
                state_dict[f"{name}.qweight"] = module.qweight
                state_dict[f"{name}.scales"] = module.scales
                if hasattr(module, 'zeros') and module.zeros is not None:
                    state_dict[f"{name}.qzeros"] = module.zeros
                if hasattr(module, 'g_idx') and module.g_idx is not None:
                    state_dict[f"{name}.g_idx"] = module.g_idx
                if hasattr(module, 'bias') and module.bias is not None:
                    state_dict[f"{name}.bias"] = module.bias
            elif hasattr(module, 'weight') and not any(
                hasattr(child, 'qweight') for child in module.children()
            ):
                # 非量化权重（如embedding, lm_head等）
                if len(list(module.children())) == 0:
                    state_dict[f"{name}.weight"] = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        state_dict[f"{name}.bias"] = module.bias
        
        return state_dict
    
    def _save_safetensors(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """保存为safetensors格式"""
        # 分片保存（如果太大）
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
        
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        
        if total_size <= max_shard_size:
            # 单文件保存
            save_path = os.path.join(self.save_dir, 'model.safetensors')
            save_file(state_dict, save_path)
            
            # 保存索引
            index = {
                'metadata': {'total_size': total_size},
                'weight_map': {k: 'model.safetensors' for k in state_dict.keys()},
            }
        else:
            # 分片保存
            shards = self._split_state_dict(state_dict, max_shard_size)
            weight_map = {}
            
            for i, shard in enumerate(shards):
                shard_name = f'model-{i+1:05d}-of-{len(shards):05d}.safetensors'
                save_path = os.path.join(self.save_dir, shard_name)
                save_file(shard, save_path)
                
                for k in shard.keys():
                    weight_map[k] = shard_name
            
            index = {
                'metadata': {'total_size': total_size},
                'weight_map': weight_map,
            }
        
        # 保存索引
        index_path = os.path.join(self.save_dir, 'model.safetensors.index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Saved {len(state_dict)} tensors in safetensors format")
    
    def _save_pytorch(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """保存为PyTorch格式"""
        save_path = os.path.join(self.save_dir, 'pytorch_model.bin')
        torch.save(state_dict, save_path)
        logger.info(f"Saved model to {save_path}")
    
    def _split_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        max_size: int,
    ) -> list:
        """分片state_dict"""
        shards = []
        current_shard = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        if current_shard:
            shards.append(current_shard)
        
        return shards
    
    def _save_vllm_config(self) -> None:
        """保存vLLM配置"""
        # 量化配置
        quantization_config = {
            'quant_method': self.quant_method,
            'bits': self.quant_config.weight.get('bit', 4),
            'group_size': self.quant_config.weight.get('group_size', 128),
            'desc_act': False,  # AWQ默认
            'sym': self.quant_config.weight.get('symmetric', True),
        }
        
        config_path = os.path.join(self.save_dir, 'quantize_config.json')
        with open(config_path, 'w') as f:
            json.dump(quantization_config, f, indent=2)
        
        logger.info(f"Saved vLLM config to {config_path}")
