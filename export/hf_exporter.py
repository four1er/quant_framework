"""
HuggingFace格式导出器

导出为HuggingFace Transformers兼容格式
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

logger = get_logger(__name__)


@EXPORTER_REGISTRY.register('huggingface')
class HuggingFaceExporter(BaseExporter):
    """
    HuggingFace格式导出器
    
    导出为标准HuggingFace模型格式，支持AutoGPTQ和BitsAndBytes
    """
    
    def __init__(
        self,
        model: nn.Module,
        quant_config: EasyDict,
        save_dir: str,
        original_config: Optional[Dict] = None,
    ):
        super().__init__(model, quant_config, save_dir)
        self.original_config = original_config or {}
    
    def export(
        self,
        tokenizer=None,
        model_config=None,
        use_safetensors: bool = True,
        **kwargs,
    ) -> str:
        """
        导出为HuggingFace格式
        
        Args:
            tokenizer: tokenizer对象
            model_config: 模型配置
            use_safetensors: 是否使用safetensors格式
        
        Returns:
            导出目录路径
        """
        logger.info("Exporting model to HuggingFace format...")
        
        # 1. 保存模型权重
        state_dict = self._prepare_state_dict()
        
        if use_safetensors:
            self._save_safetensors(state_dict)
        else:
            self._save_pytorch(state_dict)
        
        # 2. 保存模型配置
        self._save_config(model_config)
        
        # 3. 保存tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(self.save_dir)
            logger.info("Saved tokenizer")
        
        # 4. 保存量化配置
        self._save_quantization_config()
        
        logger.info(f"Model exported to {self.save_dir}")
        return self.save_dir
    
    def _prepare_state_dict(self) -> Dict[str, torch.Tensor]:
        """准备state_dict"""
        state_dict = {}
        
        for name, param in self.model.named_parameters():
            state_dict[name] = param.data
        
        for name, buffer in self.model.named_buffers():
            # 包含量化相关的buffer
            if 'scale' in name or 'zero' in name or 'qweight' in name:
                state_dict[name] = buffer
        
        return state_dict
    
    def _save_safetensors(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """保存为safetensors格式"""
        # 分片保存
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        
        if total_size <= max_shard_size:
            save_path = os.path.join(self.save_dir, 'model.safetensors')
            save_file(state_dict, save_path)
            
            index = {
                'metadata': {'total_size': total_size},
                'weight_map': {k: 'model.safetensors' for k in state_dict.keys()},
            }
        else:
            # 分片
            shards = []
            current_shard = {}
            current_size = 0
            
            for key, tensor in state_dict.items():
                tensor_size = tensor.numel() * tensor.element_size()
                
                if current_size + tensor_size > max_shard_size and current_shard:
                    shards.append(current_shard)
                    current_shard = {}
                    current_size = 0
                
                current_shard[key] = tensor
                current_size += tensor_size
            
            if current_shard:
                shards.append(current_shard)
            
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
        
        logger.info(f"Saved {len(state_dict)} tensors")
    
    def _save_pytorch(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """保存为PyTorch格式"""
        save_path = os.path.join(self.save_dir, 'pytorch_model.bin')
        torch.save(state_dict, save_path)
        logger.info(f"Saved model to {save_path}")
    
    def _save_config(self, model_config: Optional[Dict]) -> None:
        """保存模型配置"""
        if model_config is None:
            # 尝试从模型获取
            if hasattr(self.model, 'config'):
                config = self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else {}
            else:
                config = self.original_config.copy()
        else:
            config = model_config.copy() if isinstance(model_config, dict) else model_config.to_dict()
        
        # 添加量化信息
        config['quantization_config'] = {
            'quant_method': 'gptq',
            'bits': self.quant_config.weight.get('bit', 4),
            'group_size': self.quant_config.weight.get('group_size', 128),
            'damp_percent': 0.01,
            'desc_act': False,
            'sym': self.quant_config.weight.get('symmetric', True),
            'true_sequential': True,
        }
        
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    def _save_quantization_config(self) -> None:
        """保存量化配置"""
        quant_config = {
            'bits': self.quant_config.weight.get('bit', 4),
            'group_size': self.quant_config.weight.get('group_size', 128),
            'damp_percent': 0.01,
            'desc_act': False,
            'static_groups': False,
            'sym': self.quant_config.weight.get('symmetric', True),
            'true_sequential': True,
            'model_name_or_path': None,
            'model_file_base_name': 'model',
        }
        
        config_path = os.path.join(self.save_dir, 'quantize_config.json')
        with open(config_path, 'w') as f:
            json.dump(quant_config, f, indent=2)
        
        logger.info(f"Saved quantization config")


@EXPORTER_REGISTRY.register('autogptq')
class AutoGPTQExporter(HuggingFaceExporter):
    """
    AutoGPTQ格式导出器
    
    专门针对AutoGPTQ格式的导出
    """
    
    def export(
        self,
        tokenizer=None,
        model_config=None,
        use_safetensors: bool = True,
        **kwargs,
    ) -> str:
        """导出为AutoGPTQ格式"""
        logger.info("Exporting model to AutoGPTQ format...")
        
        # 调用父类导出
        result = super().export(
            tokenizer=tokenizer,
            model_config=model_config,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        
        # 添加AutoGPTQ特定配置
        self._save_autogptq_config()
        
        return result
    
    def _save_autogptq_config(self) -> None:
        """保存AutoGPTQ特定配置"""
        config = {
            'bits': self.quant_config.weight.get('bit', 4),
            'group_size': self.quant_config.weight.get('group_size', 128),
            'damp_percent': 0.01,
            'desc_act': False,
            'static_groups': False,
            'sym': self.quant_config.weight.get('symmetric', True),
            'true_sequential': True,
            'quant_method': 'gptq',
        }
        
        config_path = os.path.join(self.save_dir, 'quantize_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
