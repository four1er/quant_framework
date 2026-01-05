"""
模型抽象层：提供统一的Block视图，屏蔽不同模型的结构差异
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..utils import get_logger
from ..utils.config import EasyDict

logger = get_logger(__name__)


# 线性层类型表：支持多次替换
_ALL_LINEAR_TYPES_ = (nn.Linear,)


def register_linear_type(linear_cls):
    """注册新的线性层类型"""
    global _ALL_LINEAR_TYPES_
    if linear_cls not in _ALL_LINEAR_TYPES_:
        _ALL_LINEAR_TYPES_ = _ALL_LINEAR_TYPES_ + (linear_cls,)


def get_all_linear_types():
    """获取所有线性层类型"""
    return _ALL_LINEAR_TYPES_


class CatcherException(Exception):
    """用于Catcher中断前向传播的自定义异常"""
    pass


class Catcher(nn.Module):
    """
    捕获首块输入的特殊wrapper
    通过抛出异常中断前向传播，避免不必要的计算
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.input_data: List[torch.Tensor] = []
        self.input_kwargs: Dict[str, Any] = {}
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> None:
        # 缓存输入
        self.input_data.append(hidden_states.detach().cpu())
        # 更新kwargs（只保留最后一次的）
        self.input_kwargs.update(kwargs)
        # 抛出异常中断
        raise CatcherException("Input caught!")


class BaseModel(ABC):
    """
    模型抽象基类
    
    提供统一接口:
    - get_blocks(): 获取所有Transformer Block
    - get_block_linears(block): 获取Block内的线性层
    - replace_module_subset(): 替换子模块
    - collect_first_block_input(): 捕获首块输入
    - set_modality(): 切换模态（用于VLM）
    """
    
    def __init__(self, config: EasyDict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.blocks = None
        self.current_modality = "language"
        self.first_block_input = None
        
        # 加载模型
        self._load_model()
        
        # 定位blocks
        self.find_blocks()
    
    def _load_model(self) -> None:
        """加载HuggingFace模型"""
        model_path = self.config.model.path
        torch_dtype = self._parse_dtype(self.config.model.get('torch_dtype', 'torch.float16'))
        device_map = self.config.model.get('device_map', 'auto')
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"  torch_dtype: {torch_dtype}")
        logger.info(f"  device_map: {device_map}")
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # 加载tokenizer
        tokenizer_mode = self.config.model.get('tokenizer_mode', 'slow')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=(tokenizer_mode == 'fast'),
            trust_remote_code=True,
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded successfully")
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """解析dtype字符串"""
        dtype_map = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'torch.float8_e4m3fn': getattr(torch, 'float8_e4m3fn', torch.float16),
            'torch.float8_e5m2': getattr(torch, 'float8_e5m2', torch.float16),
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    @abstractmethod
    def find_blocks(self) -> None:
        """
        定位所有Transformer Block
        子类必须实现此方法，设置 self.blocks
        """
        pass
    
    def get_blocks(self) -> nn.ModuleList:
        """获取所有Block"""
        return self.blocks
    
    def get_block_linears(self, block: nn.Module) -> Dict[str, nn.Module]:
        """
        获取Block内的所有线性层
        
        Args:
            block: Transformer Block
        
        Returns:
            {layer_name: module} 字典
        """
        linears = {}
        for name, module in block.named_modules():
            if isinstance(module, get_all_linear_types()):
                # 过滤掉某些特殊层
                if not self._should_skip_layer(name):
                    linears[name] = module
        return linears
    
    def _should_skip_layer(self, name: str) -> bool:
        """判断是否跳过某层"""
        skip_patterns = ['lm_head', 'embed', 'norm']
        return any(pattern in name.lower() for pattern in skip_patterns)
    
    def get_attn_in_block(self, block: nn.Module) -> Dict[str, nn.Module]:
        """获取Block内的注意力模块"""
        attns = {}
        for name, module in block.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'q_proj') or hasattr(module, 'qkv_proj'):
                    attns[name] = module
        return attns
    
    def set_modality(self, modality: str) -> None:
        """
        切换模态（用于VLM）
        
        Args:
            modality: 模态名称 ('language', 'vision', etc.)
        """
        self.current_modality = modality
        # 子类可以重写此方法来切换blocks
    
    def collect_first_block_input(
        self,
        calib_data: List[Dict[str, torch.Tensor]],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        捕获首块输入
        
        原理:
        1. 把blocks[0]替换成Catcher
        2. 遍历calib_data，Catcher缓存输入
        3. 恢复blocks[0]
        
        Args:
            calib_data: 校准数据列表
            padding_mask: padding掩码
        
        Returns:
            {'data': [tensor1, tensor2, ...], 'kwargs': {...}}
        """
        if self.blocks is None or len(self.blocks) == 0:
            raise RuntimeError("No blocks found. Call find_blocks() first.")
        
        # 保存原始block
        original_block = self.blocks[0]
        
        # 替换为Catcher
        catcher = Catcher(original_block)
        self._set_block(0, catcher)
        
        # 遍历校准数据
        self.model.eval()
        with torch.no_grad():
            for batch in calib_data:
                try:
                    # 移动到设备
                    batch = self._move_to_device(batch)
                    # 前向传播（会被Catcher中断）
                    self.model(**batch)
                except CatcherException:
                    # 正常中断
                    pass
                except Exception as e:
                    logger.warning(f"Error during input capture: {e}")
        
        # 恢复原始block
        self._set_block(0, original_block)
        
        # 构建结果
        self.first_block_input = {
            'data': catcher.input_data,
            'kwargs': catcher.input_kwargs,
        }
        
        logger.info(f"Captured {len(catcher.input_data)} input samples")
        return self.first_block_input
    
    def get_first_block_input(self) -> Optional[Dict[str, Any]]:
        """获取已捕获的首块输入"""
        return self.first_block_input
    
    def _set_block(self, idx: int, new_block: nn.Module) -> None:
        """设置指定位置的block"""
        self.blocks[idx] = new_block
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将batch移动到模型设备"""
        device = next(self.model.parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def replace_module_subset(
        self,
        module_cls: type,
        block: nn.Module,
        subset: Dict[str, nn.Module],
        block_idx: int,
        params_dict: Dict[str, Any],
    ) -> None:
        """
        替换Block内的子模块
        
        Args:
            module_cls: 新模块类（如FakeQuantLinear）
            block: 要操作的Block
            subset: {name: old_module} 要替换的模块
            block_idx: Block索引（用于日志）
            params_dict: 传给新模块的参数
        """
        for name, old_module in subset.items():
            # 用工厂方法创建新模块
            if hasattr(module_cls, 'new'):
                new_module = module_cls.new(old_module, **params_dict)
            else:
                new_module = module_cls(old_module, **params_dict)
            
            # 找到父模块并替换
            self._replace_module_by_name(block, name, new_module)
            
            logger.debug(
                f"Block {block_idx:3d} | {name:40s} | "
                f"{type(old_module).__name__} -> {type(new_module).__name__}"
            )
    
    def _replace_module_by_name(
        self,
        parent: nn.Module,
        name: str,
        new_module: nn.Module,
    ) -> None:
        """根据名称替换模块"""
        if '.' in name:
            # 嵌套路径
            parts = name.split('.')
            parent_name = '.'.join(parts[:-1])
            child_name = parts[-1]
            
            # 找到父模块
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            setattr(parent, child_name, new_module)
        else:
            setattr(parent, name, new_module)
    
    def get_model_type(self) -> str:
        """获取模型类型"""
        return self.model.config.model_type
    
    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return next(self.model.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """获取模型dtype"""
        return next(self.model.parameters()).dtype
