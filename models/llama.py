"""
LLaMA模型适配
"""
import torch.nn as nn
from typing import Dict

from .base_model import BaseModel
from ..utils import MODEL_REGISTRY, get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("llama")
@MODEL_REGISTRY.register("llama2")
@MODEL_REGISTRY.register("llama3")
class LlamaModel(BaseModel):
    """
    LLaMA系列模型适配
    
    支持: LLaMA, LLaMA-2, LLaMA-3
    Block路径: model.model.layers
    """
    
    def find_blocks(self) -> None:
        """定位LLaMA的Transformer Block"""
        # LLaMA的layers在 model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.blocks = self.model.model.layers
        else:
            raise RuntimeError("Cannot find layers in LLaMA model structure")
        
        logger.info(f"Found {len(self.blocks)} transformer blocks")
    
    def get_block_linears(self, block: nn.Module) -> Dict[str, nn.Module]:
        """
        获取LLaMA Block内的线性层
        
        LLaMA Block结构:
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
        
        # MLP层
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, name):
                    linears[f'mlp.{name}'] = getattr(mlp, name)
        
        return linears
    
    def get_attn_in_block(self, block: nn.Module) -> Dict[str, nn.Module]:
        """获取LLaMA Block内的注意力模块"""
        if hasattr(block, 'self_attn'):
            return {'self_attn': block.self_attn}
        return {}
    
    def get_embed_layers(self) -> Dict[str, nn.Module]:
        """获取嵌入层"""
        layers = {}
        if hasattr(self.model.model, 'embed_tokens'):
            layers['embed_tokens'] = self.model.model.embed_tokens
        return layers
    
    def get_head_layers(self) -> Dict[str, nn.Module]:
        """获取输出头"""
        layers = {}
        if hasattr(self.model, 'lm_head'):
            layers['lm_head'] = self.model.lm_head
        return layers
    
    def get_norm_layers(self, block: nn.Module) -> Dict[str, nn.Module]:
        """获取Block内的归一化层"""
        norms = {}
        if hasattr(block, 'input_layernorm'):
            norms['input_layernorm'] = block.input_layernorm
        if hasattr(block, 'post_attention_layernorm'):
            norms['post_attention_layernorm'] = block.post_attention_layernorm
        return norms
