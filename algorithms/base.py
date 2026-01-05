"""
Blockwise调度引擎：按Block调度算法插件

核心类:
- BlockwiseOpt: 提供Block遍历骨架
- BaseBlockwiseQuantization: 量化专用基类，提供Quantizer和模块替换工具
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from tqdm import tqdm

from ..utils import get_logger
from ..utils.config import EasyDict
from ..models.base_model import BaseModel, register_linear_type
from ..quantization.quantizer import IntegerQuantizer, FloatQuantizer, QParams
from ..quantization.modules import (
    FakeQuantLinear,
    EfficientFakeQuantLinear,
    RealQuantLinear,
    OriginFloatLinear,
    RotateLinear,
    get_quant_linear_cls,
)

logger = get_logger(__name__)


class BlockwiseOpt(ABC):
    """
    Blockwise优化基类
    
    提供Block遍历骨架，子类实现block_opt方法
    """
    
    def __init__(
        self,
        model: BaseModel,
        compress_config: EasyDict,
        input: Optional[Dict[str, Any]],
        padding_mask: Optional[torch.Tensor],
        config: EasyDict,
    ):
        self.model = model
        self.compress_config = compress_config
        self.input = input  # first_block_input
        self.padding_mask = padding_mask
        self.config = config
        
        self.blocks = model.get_blocks()
        self.block_idx = 0
        
        # 是否为data-free模式
        self.data_free = (input is None)
        
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Blocks: {len(self.blocks)}")
        logger.info(f"  Data-free: {self.data_free}")
    
    def run_block_loop(self) -> None:
        """
        按Block遍历并优化
        
        遍历所有Block，对每个Block调用block_opt
        """
        logger.info(f"Starting block loop for {self.__class__.__name__}")
        
        for self.block_idx, block in enumerate(tqdm(self.blocks, desc="Processing blocks")):
            logger.debug(f"Processing block {self.block_idx}/{len(self.blocks)}")
            
            # 将block移到正确设备
            block = self._move_block_to_device(block)
            
            # 子类实现的优化逻辑
            self.block_opt(block)
            
            # 更新input为当前block的输出（用于下一个block）
            if not self.data_free and self.input is not None:
                self._update_input_for_next_block(block)
        
        logger.info(f"Block loop completed for {self.__class__.__name__}")
    
    @abstractmethod
    def block_opt(self, block: nn.Module) -> None:
        """
        单个Block的优化逻辑
        
        子类必须实现此方法
        
        Args:
            block: 当前处理的Transformer Block
        """
        pass
    
    def _move_block_to_device(self, block: nn.Module) -> nn.Module:
        """将block移到正确设备"""
        # 如果使用device_map='auto'，block可能已经在正确设备上
        return block
    
    def _update_input_for_next_block(self, block: nn.Module) -> None:
        """更新input为当前block的输出"""
        if self.input is None or 'data' not in self.input:
            return
        
        new_data = []
        block.eval()
        
        with torch.no_grad():
            for inp in self.input['data']:
                inp = inp.to(next(block.parameters()).device)
                kwargs = {k: v.to(inp.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in self.input.get('kwargs', {}).items()}
                
                output = block(inp, **kwargs)
                
                # 处理不同的输出格式
                if isinstance(output, tuple):
                    output = output[0]
                
                new_data.append(output.cpu())
        
        self.input['data'] = new_data
    
    def block_forward(
        self,
        block: nn.Module,
        input_data: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        用校准数据跑当前Block
        
        Args:
            block: 当前Block
            input_data: 输入数据列表
        
        Returns:
            输出数据列表
        """
        outputs = []
        block.eval()
        
        with torch.no_grad():
            for inp in input_data:
                device = next(block.parameters()).device
                inp = inp.to(device)
                
                kwargs = {}
                if self.input and 'kwargs' in self.input:
                    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in self.input['kwargs'].items()}
                
                output = block(inp, **kwargs)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                outputs.append(output)
        
        return outputs


class BaseBlockwiseQuantization(BlockwiseOpt):
    """
    量化专用的Blockwise基类
    
    在BlockwiseOpt基础上添加:
    - Quantizer初始化
    - w_qdq/a_qdq封装
    - get_replacement_params
    - 模块替换工具
    """
    
    def __init__(
        self,
        model: BaseModel,
        quant_config: EasyDict,
        input: Optional[Dict[str, Any]],
        padding_mask: Optional[torch.Tensor],
        config: EasyDict,
    ):
        super().__init__(model, quant_config, input, padding_mask, config)
        
        self.quant_config = quant_config
        
        # 初始化Quantizer
        self._init_quantizers()
        
        # 是否只量化权重
        self.w_only = not hasattr(quant_config, 'act') or quant_config.act is None
        
        # 激活是否静态量化
        self.act_static = False
        if not self.w_only and hasattr(quant_config.act, 'static'):
            self.act_static = quant_config.act.static
    
    def _init_quantizers(self) -> None:
        """初始化权重和激活Quantizer"""
        weight_cfg = self.quant_config.weight
        
        # 权重Quantizer
        self.wquantizer = IntegerQuantizer(
            bit=weight_cfg.get('bit', 4),
            symmetric=weight_cfg.get('symmetric', True),
            granularity=weight_cfg.get('granularity', 'per_group'),
            calib_algo=weight_cfg.get('calib_algo', 'minmax'),
            group_size=weight_cfg.get('group_size', 128),
            head_num=weight_cfg.get('head_num', 32),
        )
        
        # 激活Quantizer
        if hasattr(self.quant_config, 'act') and self.quant_config.act is not None:
            act_cfg = self.quant_config.act
            self.aquantizer = IntegerQuantizer(
                bit=act_cfg.get('bit', 8),
                symmetric=act_cfg.get('symmetric', True),
                granularity=act_cfg.get('granularity', 'per_tensor'),
                calib_algo=act_cfg.get('calib_algo', 'minmax'),
            )
        else:
            self.aquantizer = None
        
        logger.info(f"Weight quantizer: {self.wquantizer.bit}bit, {self.wquantizer.granularity}")
        if self.aquantizer:
            logger.info(f"Activation quantizer: {self.aquantizer.bit}bit, {self.aquantizer.granularity}")
    
    def w_qdq(self, module: nn.Module) -> torch.Tensor:
        """
        权重量化-反量化
        
        Args:
            module: 包含weight属性的模块
        
        Returns:
            量化-反量化后的权重
        """
        weight = module.weight if hasattr(module, 'weight') else module
        return self.wquantizer.fake_quant_weight_dynamic(weight, args={})
    
    def a_qdq(
        self,
        act: torch.Tensor,
        module: nn.Module,
        input_index: int = 0,
    ) -> torch.Tensor:
        """
        激活量化-反量化
        
        Args:
            act: 激活张量
            module: 当前模块（用于获取静态scale）
            input_index: 输入索引（多输入时使用）
        
        Returns:
            量化-反量化后的激活
        """
        if self.aquantizer is None:
            return act
        
        if self.act_static:
            # 静态量化：使用预计算的scale
            if hasattr(module, 'buf_act_scales'):
                scale = module.buf_act_scales
                zp = module.buf_act_zeros if hasattr(module, 'buf_act_zeros') else 0
                return self.aquantizer.fake_quant_act_static(act, scale, zp)
        
        # 动态量化
        return self.aquantizer.fake_quant_act_dynamic(act)
    
    def w_q(self, module: nn.Module) -> Dict[str, Any]:
        """
        权重真量化
        
        Args:
            module: 包含weight属性的模块
        
        Returns:
            量化结果字典
        """
        weight = module.weight if hasattr(module, 'weight') else module
        return self.wquantizer.real_quant_weight_dynamic(weight, args={})
    
    def get_replacement_params(
        self,
        mode: str,
        w_only: bool = False,
    ) -> Dict[str, Any]:
        """
        生成模块替换参数
        
        Args:
            mode: 替换模式 ('fake_quant', 'vllm_quant', 'origin_float', 'online_rotate')
            w_only: 是否只量化权重
        
        Returns:
            传给新模块的参数字典
        """
        if mode == 'fake_quant':
            return {
                'w_qdq': lambda m: self.w_qdq(m),
                'a_qdq': None if (w_only or self.w_only) else lambda x, m: self.a_qdq(x, m),
            }
        
        elif mode == 'efficient_fake_quant':
            return {
                'w_qdq': lambda m: self.w_qdq(m),
                'a_qdq': None if (w_only or self.w_only) else lambda x, m: self.a_qdq(x, m),
            }
        
        elif mode in ['vllm_quant', 'real_quant']:
            return {
                'w_q': lambda m: self.w_q(m),
                'quant_config': {
                    'bits': self.quant_config.weight.get('bit', 4),
                    'group_size': self.quant_config.weight.get('group_size', 128),
                    'symmetric': self.quant_config.weight.get('symmetric', True),
                },
            }
        
        elif mode == 'origin_float':
            return {}
        
        elif mode == 'online_rotate':
            had_K, K = self._get_hadamard_matrix()
            return {
                'had_K': had_K,
                'K': K,
                'online_full_had': True,
                'online_partial_had': False,
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _get_hadamard_matrix(self) -> Tuple[torch.Tensor, int]:
        """获取Hadamard矩阵"""
        # 默认使用128维
        K = 128
        H = RotateLinear._generate_hadamard(K)
        return H, K
    
    def replace_linears(
        self,
        block: nn.Module,
        mode: str = 'fake_quant',
    ) -> None:
        """
        替换Block内的所有线性层
        
        Args:
            block: 当前Block
            mode: 替换模式
        """
        linears = self.model.get_block_linears(block)
        params = self.get_replacement_params(mode)
        module_cls = get_quant_linear_cls(mode)
        
        self.model.replace_module_subset(
            module_cls=module_cls,
            block=block,
            subset=linears,
            block_idx=self.block_idx,
            params_dict=params,
        )
    
    def collect_block_qparams(self, block: nn.Module) -> Dict[str, QParams]:
        """
        收集Block内所有线性层的QParams
        
        用于静态量化
        
        Args:
            block: 当前Block
        
        Returns:
            {layer_name: QParams}
        """
        qparams_dict = {}
        linears = self.model.get_block_linears(block)
        
        for name, module in linears.items():
            weight = module.weight.data
            qparams = self.wquantizer.get_tensor_qparams(weight, args={})
            qparams_dict[name] = qparams
            
            # 缓存到module
            module.register_buffer('buf_scales', qparams.scale)
            module.register_buffer('buf_zeros', qparams.zero_point)
        
        return qparams_dict
    
    def get_subsets_in_block(self, block: nn.Module) -> Dict[str, Dict[str, nn.Module]]:
        """
        将Block内的线性层分组
        
        用于对不同类型的层使用不同策略
        
        Args:
            block: 当前Block
        
        Returns:
            {'attn': {...}, 'mlp': {...}}
        """
        linears = self.model.get_block_linears(block)
        subsets = {
            'attn': {},
            'mlp': {},
        }
        
        for name, module in linears.items():
            if 'attn' in name.lower() or 'attention' in name.lower():
                subsets['attn'][name] = module
            elif 'mlp' in name.lower() or 'ffn' in name.lower():
                subsets['mlp'][name] = module
            else:
                # 默认放到mlp
                subsets['mlp'][name] = module
        
        return subsets
    
    # 等效变换工具方法
    def scale_fc_fc(
        self,
        fc1: nn.Module,
        fc2: nn.Module,
        scales: torch.Tensor,
    ) -> None:
        """
        缩放变换：把激活的scale迁移到权重
        
        fc1.weight /= scales
        fc2.weight *= scales
        """
        # 确保scales在正确设备
        scales = scales.to(fc1.weight.device)
        
        # fc1: [out, in] -> 在out维度除以scales
        fc1.weight.data = fc1.weight.data / scales.view(-1, 1)
        if fc1.bias is not None:
            fc1.bias.data = fc1.bias.data / scales
        
        # fc2: [out, in] -> 在in维度乘以scales
        fc2.weight.data = fc2.weight.data * scales.view(1, -1)
    
    def shift_fc_fc(
        self,
        fc1: nn.Module,
        fc2: nn.Module,
        shifts: torch.Tensor,
    ) -> None:
        """
        平移变换：把bias或shift迁移到下游
        """
        shifts = shifts.to(fc1.weight.device)
        
        if fc1.bias is not None:
            fc1.bias.data = fc1.bias.data - shifts
        
        # 补偿到fc2
        fc2.bias.data = fc2.bias.data + fc2.weight.data @ shifts
    
    def scale_ln_fcs(
        self,
        ln: nn.Module,
        fcs: List[nn.Module],
        scales: torch.Tensor,
    ) -> None:
        """
        LayerNorm缩放：把scales应用到LN和下游FC
        """
        scales = scales.to(ln.weight.device)
        
        # 缩放LN
        ln.weight.data = ln.weight.data / scales
        if ln.bias is not None:
            ln.bias.data = ln.bias.data / scales
        
        # 缩放FC
        for fc in fcs:
            fc.weight.data = fc.weight.data * scales.view(1, -1)


# 注册量化模块类型
register_linear_type(FakeQuantLinear)
register_linear_type(EfficientFakeQuantLinear)
register_linear_type(RealQuantLinear)
register_linear_type(OriginFloatLinear)
register_linear_type(RotateLinear)
