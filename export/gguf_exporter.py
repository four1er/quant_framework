"""
GGUF格式导出器

导出为llama.cpp兼容的GGUF格式
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import os
import struct
import numpy as np

from .base import BaseExporter, EXPORTER_REGISTRY
from ..utils import get_logger
from ..utils.config import EasyDict

logger = get_logger(__name__)


# GGUF常量
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# 数据类型映射
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15


@EXPORTER_REGISTRY.register('gguf')
class GGUFExporter(BaseExporter):
    """
    GGUF格式导出器
    
    导出为llama.cpp兼容的GGUF格式
    """
    
    def __init__(
        self,
        model: nn.Module,
        quant_config: EasyDict,
        save_dir: str,
        model_arch: str = 'llama',
    ):
        super().__init__(model, quant_config, save_dir)
        self.model_arch = model_arch
        
        # GGUF元数据
        self.metadata = {}
        self.tensors = {}
    
    def export(
        self,
        tokenizer=None,
        output_name: str = 'model.gguf',
        quantization_type: str = 'q4_k',
        **kwargs,
    ) -> str:
        """
        导出为GGUF格式
        
        Args:
            tokenizer: tokenizer对象
            output_name: 输出文件名
            quantization_type: 量化类型 (q4_0, q4_1, q4_k, q5_k, q8_0等)
        
        Returns:
            导出文件路径
        """
        logger.info(f"Exporting model to GGUF format ({quantization_type})...")
        
        # 1. 收集元数据
        self._collect_metadata(tokenizer)
        
        # 2. 收集并量化张量
        self._collect_tensors(quantization_type)
        
        # 3. 写入GGUF文件
        output_path = os.path.join(self.save_dir, output_name)
        self._write_gguf(output_path)
        
        logger.info(f"Model exported to {output_path}")
        return output_path
    
    def _collect_metadata(self, tokenizer=None) -> None:
        """收集GGUF元数据"""
        # 通用元数据
        self.metadata['general.architecture'] = self.model_arch
        self.metadata['general.name'] = 'QuantizedModel'
        self.metadata['general.quantization_version'] = 2
        
        # 模型配置
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # LLaMA架构参数
            if hasattr(config, 'hidden_size'):
                self.metadata[f'{self.model_arch}.embedding_length'] = config.hidden_size
            if hasattr(config, 'num_hidden_layers'):
                self.metadata[f'{self.model_arch}.block_count'] = config.num_hidden_layers
            if hasattr(config, 'num_attention_heads'):
                self.metadata[f'{self.model_arch}.attention.head_count'] = config.num_attention_heads
            if hasattr(config, 'num_key_value_heads'):
                self.metadata[f'{self.model_arch}.attention.head_count_kv'] = config.num_key_value_heads
            if hasattr(config, 'intermediate_size'):
                self.metadata[f'{self.model_arch}.feed_forward_length'] = config.intermediate_size
            if hasattr(config, 'rms_norm_eps'):
                self.metadata[f'{self.model_arch}.attention.layer_norm_rms_epsilon'] = config.rms_norm_eps
            if hasattr(config, 'rope_theta'):
                self.metadata[f'{self.model_arch}.rope.freq_base'] = config.rope_theta
            if hasattr(config, 'max_position_embeddings'):
                self.metadata[f'{self.model_arch}.context_length'] = config.max_position_embeddings
        
        # Tokenizer元数据
        if tokenizer is not None:
            self.metadata['tokenizer.ggml.model'] = 'llama'
            
            if hasattr(tokenizer, 'vocab_size'):
                self.metadata['tokenizer.ggml.vocab_size'] = tokenizer.vocab_size
            
            # BOS/EOS tokens
            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                self.metadata['tokenizer.ggml.bos_token_id'] = tokenizer.bos_token_id
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                self.metadata['tokenizer.ggml.eos_token_id'] = tokenizer.eos_token_id
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                self.metadata['tokenizer.ggml.padding_token_id'] = tokenizer.pad_token_id
    
    def _collect_tensors(self, quantization_type: str) -> None:
        """收集并量化张量"""
        ggml_type = self._get_ggml_type(quantization_type)
        
        # 名称映射 (HuggingFace -> GGUF)
        name_map = self._get_name_mapping()
        
        for name, param in self.model.named_parameters():
            # 转换名称
            gguf_name = self._convert_name(name, name_map)
            
            # 获取数据
            data = param.data.cpu()
            
            # 决定是否量化
            if self._should_quantize(name, param):
                quantized_data, data_type = self._quantize_tensor(data, ggml_type)
            else:
                # 保持FP16/FP32
                if data.dtype == torch.float32:
                    quantized_data = data.numpy().astype(np.float32)
                    data_type = GGML_TYPE_F32
                else:
                    quantized_data = data.half().numpy()
                    data_type = GGML_TYPE_F16
            
            self.tensors[gguf_name] = {
                'data': quantized_data,
                'type': data_type,
                'shape': list(param.shape),
            }
        
        logger.info(f"Collected {len(self.tensors)} tensors")
    
    def _get_ggml_type(self, quantization_type: str) -> int:
        """获取GGML量化类型"""
        type_map = {
            'f32': GGML_TYPE_F32,
            'f16': GGML_TYPE_F16,
            'q4_0': GGML_TYPE_Q4_0,
            'q4_1': GGML_TYPE_Q4_1,
            'q5_0': GGML_TYPE_Q5_0,
            'q5_1': GGML_TYPE_Q5_1,
            'q8_0': GGML_TYPE_Q8_0,
            'q4_k': GGML_TYPE_Q4_K,
            'q5_k': GGML_TYPE_Q5_K,
            'q6_k': GGML_TYPE_Q6_K,
            'q8_k': GGML_TYPE_Q8_K,
        }
        return type_map.get(quantization_type.lower(), GGML_TYPE_Q4_K)
    
    def _get_name_mapping(self) -> Dict[str, str]:
        """获取名称映射"""
        # LLaMA名称映射
        return {
            'model.embed_tokens': 'token_embd',
            'model.norm': 'output_norm',
            'lm_head': 'output',
            'self_attn.q_proj': 'attn_q',
            'self_attn.k_proj': 'attn_k',
            'self_attn.v_proj': 'attn_v',
            'self_attn.o_proj': 'attn_output',
            'mlp.gate_proj': 'ffn_gate',
            'mlp.up_proj': 'ffn_up',
            'mlp.down_proj': 'ffn_down',
            'input_layernorm': 'attn_norm',
            'post_attention_layernorm': 'ffn_norm',
        }
    
    def _convert_name(self, name: str, name_map: Dict[str, str]) -> str:
        """转换名称为GGUF格式"""
        # 提取层号
        parts = name.split('.')
        layer_idx = None
        
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        
        # 替换名称
        gguf_name = name
        for hf_name, gguf_suffix in name_map.items():
            if hf_name in name:
                if layer_idx is not None:
                    gguf_name = f'blk.{layer_idx}.{gguf_suffix}'
                else:
                    gguf_name = gguf_suffix
                break
        
        # 添加.weight后缀
        if not gguf_name.endswith('.weight') and not gguf_name.endswith('.bias'):
            gguf_name += '.weight'
        
        return gguf_name
    
    def _should_quantize(self, name: str, param: torch.Tensor) -> bool:
        """判断是否应该量化"""
        # 不量化的层
        skip_patterns = ['embed', 'norm', 'lm_head', 'output']
        
        for pattern in skip_patterns:
            if pattern in name.lower():
                return False
        
        # 只量化2D权重
        if param.dim() != 2:
            return False
        
        return True
    
    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        ggml_type: int,
    ) -> tuple:
        """
        量化张量
        
        简化实现，实际应使用llama.cpp的量化函数
        """
        # 简化实现：使用Q4_0格式
        if ggml_type == GGML_TYPE_Q4_0:
            return self._quantize_q4_0(tensor)
        elif ggml_type == GGML_TYPE_Q8_0:
            return self._quantize_q8_0(tensor)
        else:
            # 默认返回FP16
            return tensor.half().numpy(), GGML_TYPE_F16
    
    def _quantize_q4_0(self, tensor: torch.Tensor) -> tuple:
        """
        Q4_0量化
        
        每32个值共享一个scale
        """
        data = tensor.float().numpy()
        shape = data.shape
        
        # Flatten
        flat = data.flatten()
        n = len(flat)
        
        # Pad to multiple of 32
        block_size = 32
        if n % block_size != 0:
            pad_size = block_size - (n % block_size)
            flat = np.pad(flat, (0, pad_size))
            n = len(flat)
        
        # Reshape to blocks
        blocks = flat.reshape(-1, block_size)
        num_blocks = len(blocks)
        
        # 量化
        # Q4_0格式: scale (fp16) + 16 bytes (32 x 4-bit)
        result = bytearray()
        
        for block in blocks:
            # 计算scale
            max_val = np.max(np.abs(block))
            scale = max_val / 7.0 if max_val > 0 else 1.0
            
            # 量化到4-bit (-8 to 7)
            quantized = np.round(block / scale).astype(np.int8)
            quantized = np.clip(quantized, -8, 7)
            
            # 打包
            # Scale as FP16
            scale_bytes = np.float16(scale).tobytes()
            result.extend(scale_bytes)
            
            # 打包4-bit值 (两个值打包到一个字节)
            for i in range(0, block_size, 2):
                low = (quantized[i] + 8) & 0x0F
                high = (quantized[i + 1] + 8) & 0x0F
                result.append(low | (high << 4))
        
        return bytes(result), GGML_TYPE_Q4_0
    
    def _quantize_q8_0(self, tensor: torch.Tensor) -> tuple:
        """
        Q8_0量化
        
        每32个值共享一个scale
        """
        data = tensor.float().numpy()
        
        # Flatten
        flat = data.flatten()
        n = len(flat)
        
        # Pad to multiple of 32
        block_size = 32
        if n % block_size != 0:
            pad_size = block_size - (n % block_size)
            flat = np.pad(flat, (0, pad_size))
            n = len(flat)
        
        # Reshape to blocks
        blocks = flat.reshape(-1, block_size)
        
        # 量化
        result = bytearray()
        
        for block in blocks:
            max_val = np.max(np.abs(block))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            
            quantized = np.round(block / scale).astype(np.int8)
            quantized = np.clip(quantized, -128, 127)
            
            # Scale as FP16
            scale_bytes = np.float16(scale).tobytes()
            result.extend(scale_bytes)
            
            # 8-bit值
            result.extend(quantized.tobytes())
        
        return bytes(result), GGML_TYPE_Q8_0
    
    def _write_gguf(self, output_path: str) -> None:
        """写入GGUF文件"""
        with open(output_path, 'wb') as f:
            # 1. 写入头部
            self._write_header(f)
            
            # 2. 写入元数据
            self._write_metadata(f)
            
            # 3. 写入张量信息
            self._write_tensor_info(f)
            
            # 4. 对齐
            self._align_to(f, 32)
            
            # 5. 写入张量数据
            self._write_tensor_data(f)
        
        logger.info(f"GGUF file written: {output_path}")
    
    def _write_header(self, f) -> None:
        """写入GGUF头部"""
        # Magic
        f.write(struct.pack('<I', GGUF_MAGIC))
        # Version
        f.write(struct.pack('<I', GGUF_VERSION))
        # Tensor count
        f.write(struct.pack('<Q', len(self.tensors)))
        # Metadata count
        f.write(struct.pack('<Q', len(self.metadata)))
    
    def _write_metadata(self, f) -> None:
        """写入元数据"""
        for key, value in self.metadata.items():
            self._write_string(f, key)
            self._write_value(f, value)
    
    def _write_tensor_info(self, f) -> None:
        """写入张量信息"""
        offset = 0
        
        for name, tensor_info in self.tensors.items():
            # 名称
            self._write_string(f, name)
            
            # 维度数
            n_dims = len(tensor_info['shape'])
            f.write(struct.pack('<I', n_dims))
            
            # 形状
            for dim in tensor_info['shape']:
                f.write(struct.pack('<Q', dim))
            
            # 类型
            f.write(struct.pack('<I', tensor_info['type']))
            
            # 偏移
            f.write(struct.pack('<Q', offset))
            
            # 更新偏移
            if isinstance(tensor_info['data'], bytes):
                offset += len(tensor_info['data'])
            else:
                offset += tensor_info['data'].nbytes
    
    def _write_tensor_data(self, f) -> None:
        """写入张量数据"""
        for name, tensor_info in self.tensors.items():
            data = tensor_info['data']
            
            if isinstance(data, bytes):
                f.write(data)
            else:
                f.write(data.tobytes())
            
            # 对齐到32字节
            self._align_to(f, 32)
    
    def _write_string(self, f, s: str) -> None:
        """写入字符串"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value) -> None:
        """写入值"""
        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))  # GGUF_TYPE_INT32
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 6))  # GGUF_TYPE_FLOAT32
            f.write(struct.pack('<f', value))
        elif isinstance(value, bool):
            f.write(struct.pack('<I', 7))  # GGUF_TYPE_BOOL
            f.write(struct.pack('<?', value))
        else:
            # 默认作为字符串
            f.write(struct.pack('<I', 8))
            self._write_string(f, str(value))
    
    def _align_to(self, f, alignment: int) -> None:
        """对齐到指定字节"""
        pos = f.tell()
        padding = (alignment - (pos % alignment)) % alignment
        if padding > 0:
            f.write(b'\x00' * padding)
