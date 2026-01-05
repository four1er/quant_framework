"""
Qwen模型适配
"""
import torch.nn as nn
from typing import Dict

from .base_model import BaseModel
from ..utils import MODEL_REGISTRY, get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("qwen")
@MODEL_REGISTRY.register("qwen2")
class QwenModel(BaseModel):
    """
    Qwen系列模型适配
    
    支持: Qwen, Qwen2
    Block路径: model.model.layers
    """
    
    def find_blocks(self) -> None:
        """定位Qwen的Transformer Block"""
        # Qwen的layers在 model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.blocks = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # Qwen1的结构
            self.blocks = self.model.transformer.h
        else:
            raise RuntimeError("Cannot find layers in Qwen model structure")
        
        logger.info(f"Found {len(self.blocks)} transformer blocks")
    
    def get_block_linears(self, block: nn.Module) -> Dict[str, nn.Module]:
        """
        获取Qwen Block内的线性层
        
        Qwen2 Block结构:
        - self_attn.q_proj
        - self_attn.k_proj
        - self_attn.v_proj
        - self_attn.o_proj
        - mlp.gate_proj
        - mlp.up_proj
        - mlp.down_proj
        """
        linears = {}
        
        # Attention层
        if hasattr(block, 'self_attn'):
            attn = block.self_attn
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, name):
                    linears[f'self_attn.{name}'] = getattr(attn, name)
        elif hasattr(block, 'attn'):
            # Qwen1结构
            attn = block.attn
            if hasattr(attn, 'c_attn'):
                linears['attn.c_attn'] = attn.c_attn
            if hasattr(attn, 'c_proj'):
                linears['attn.c_proj'] = attn.c_proj
        
        # MLP层
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, name):
                    linears[f'mlp.{name}'] = getattr(mlp, name)
            # Qwen1结构
            for name in ['w1', 'w2', 'c_proj']:
                if hasattr(mlp, name):
                    linears[f'mlp.{name}'] = getattr(mlp, name)
        
        return linears
    
    def get_attn_in_block(self, block: nn.Module) -> Dict[str, nn.Module]:
        """获取Qwen Block内的注意力模块"""
        if hasattr(block, 'self_attn'):
            return {'self_attn': block.self_attn}
        elif hasattr(block, 'attn'):
            return {'attn': block.attn}
        return {}
