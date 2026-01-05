"""
统一配置系统：YAML配置解析与校验
"""
import os
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class EasyDict(dict):
    """
    支持属性访问的字典，方便配置访问
    config.model.type 等价于 config['model']['type']
    """
    
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, EasyDict):
                value = EasyDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    
    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """支持嵌套key访问，如 config.get('model.type', 'llama')"""
        keys = key.split('.')
        value = self
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'EasyDict':
        """递归转换嵌套字典"""
        result = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = cls.from_dict(v)
            elif isinstance(v, list):
                result[k] = [cls.from_dict(i) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result


def load_config(config_path: Union[str, Path]) -> EasyDict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        EasyDict格式的配置对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = EasyDict.from_dict(config_dict)
    
    # 设置默认值
    _set_defaults(config)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def _set_defaults(config: EasyDict) -> None:
    """设置配置默认值"""
    # base defaults
    if 'base' not in config:
        config.base = EasyDict()
    config.base.setdefault('seed', 42)
    
    # model defaults
    if 'model' not in config:
        config.model = EasyDict()
    config.model.setdefault('torch_dtype', 'torch.float16')
    config.model.setdefault('tokenizer_mode', 'slow')
    config.model.setdefault('device_map', 'auto')
    
    # calib defaults
    if 'calib' not in config:
        config.calib = EasyDict()
    config.calib.setdefault('n_samples', 128)
    config.calib.setdefault('bs', 1)
    config.calib.setdefault('seq_len', 512)
    
    # eval defaults
    if 'eval' not in config:
        config.eval = EasyDict()
    config.eval.setdefault('eval_pos', ['pretrain', 'fake_quant'])
    
    # save defaults
    if 'save' not in config:
        config.save = EasyDict()
    config.save.setdefault('save_path', './output')


def check_config(config: EasyDict) -> None:
    """
    配置校验：在启动阶段拦截不合法配置
    
    检查项:
    - granularity='per_group' 时 group_size > 0
    - static 激活量化只允许 per_tensor
    - 导出目录检查
    - 后端兼容性检查
    """
    errors = []
    warnings = []
    
    # 检查模型配置
    if 'model' not in config or 'type' not in config.model:
        errors.append("config.model.type is required")
    
    if 'model' in config and 'path' not in config.model:
        errors.append("config.model.path is required")
    
    # 检查量化配置
    if 'quant' in config:
        for modality_name, modality_config in config.quant.items():
            if not isinstance(modality_config, dict):
                continue
                
            # 检查method
            if 'method' not in modality_config:
                errors.append(f"config.quant.{modality_name}.method is required")
            
            # 检查weight配置
            if 'weight' in modality_config:
                weight_cfg = modality_config.weight
                
                # per_group需要group_size
                if weight_cfg.get('granularity') == 'per_group':
                    group_size = weight_cfg.get('group_size', 0)
                    if group_size <= 0:
                        errors.append(
                            f"config.quant.{modality_name}.weight: "
                            f"per_group granularity requires group_size > 0, got {group_size}"
                        )
                
                # per_head需要head_num
                if weight_cfg.get('granularity') == 'per_head':
                    if 'head_num' not in weight_cfg or weight_cfg.head_num <= 0:
                        errors.append(
                            f"config.quant.{modality_name}.weight: "
                            f"per_head granularity requires head_num > 0"
                        )
                
                # 检查bit范围
                bit = weight_cfg.get('bit', 4)
                if isinstance(bit, int) and bit not in [2, 3, 4, 8]:
                    warnings.append(
                        f"config.quant.{modality_name}.weight.bit={bit} "
                        f"is unusual, common values are 2, 3, 4, 8"
                    )
            
            # 检查act配置
            if 'act' in modality_config:
                act_cfg = modality_config.act
                
                # 静态激活量化只支持per_tensor
                if act_cfg.get('static', False):
                    if act_cfg.get('granularity', 'per_tensor') != 'per_tensor':
                        errors.append(
                            f"config.quant.{modality_name}.act: "
                            f"static activation quantization only supports per_tensor granularity"
                        )
    
    # 检查tokenizer_mode
    if config.get('model.tokenizer_mode') not in ['slow', 'fast', None]:
        warnings.append(
            f"tokenizer_mode '{config.model.tokenizer_mode}' not recognized, "
            f"using 'slow'"
        )
        config.model.tokenizer_mode = 'slow'
    
    # 检查后端导出兼容性
    if 'save' in config:
        save_cfg = config.save
        
        # AutoAWQ约束
        if save_cfg.get('save_autoawq', False) and 'quant' in config:
            for modality_name, modality_config in config.quant.items():
                if not isinstance(modality_config, dict):
                    continue
                weight_cfg = modality_config.get('weight', {})
                act_cfg = modality_config.get('act', {})
                
                if weight_cfg.get('bit') != 4:
                    errors.append("AutoAWQ only supports 4-bit weight quantization")
                if weight_cfg.get('symmetric', True):
                    errors.append("AutoAWQ requires asymmetric weight quantization")
                if act_cfg.get('bit', 16) != 16:
                    errors.append("AutoAWQ only supports A16 (no activation quantization)")
    
    # 输出警告
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")
    
    # 如果有错误，抛出异常
    if errors:
        error_msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    logger.info("Config validation passed")


def save_config(config: EasyDict, save_path: Union[str, Path]) -> None:
    """保存配置到YAML文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换回普通dict
    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        else:
            return obj
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Saved config to {save_path}")
