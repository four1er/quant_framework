"""
插件注册机制：从字符串配置映射到真正的类和构造函数
"""
from typing import Dict, Type, Any, Optional, Callable


class Registry(dict):
    """
    通用注册器，支持装饰器和显式注册两种方式
    
    使用方式1 - 装饰器（使用类名作为key）:
        @ALGO_REGISTRY
        class GPTQ(BaseBlockwiseQuantization):
            pass
    
    使用方式2 - 显式指定名称:
        @ALGO_REGISTRY.register("my_gptq")
        class GPTQ(BaseBlockwiseQuantization):
            pass
    
    使用方式3 - 函数式注册:
        ALGO_REGISTRY.register_class("GPTQ", GPTQClass)
    """
    
    def __init__(self, name: str = "Registry"):
        super().__init__()
        self._name = name
    
    def __call__(self, cls: Type) -> Type:
        """使用类名作为key的装饰器方式"""
        key = cls.__name__
        if key in self:
            raise KeyError(f"[{self._name}] Key '{key}' already registered")
        self[key] = cls
        return cls
    
    def register(self, name: str) -> Callable:
        """显式指定名称的装饰器方式"""
        def decorator(cls: Type) -> Type:
            if name in self:
                raise KeyError(f"[{self._name}] Key '{name}' already registered")
            self[name] = cls
            return cls
        return decorator
    
    def register_class(self, name: str, cls: Type) -> None:
        """函数式注册"""
        if name in self:
            raise KeyError(f"[{self._name}] Key '{name}' already registered")
        self[name] = cls
    
    def get(self, key: str, default: Optional[Type] = None) -> Optional[Type]:
        """安全获取，不存在时返回默认值"""
        return super().get(key, default)
    
    def build(self, key: str, *args, **kwargs) -> Any:
        """根据key构建实例"""
        if key not in self:
            available = list(self.keys())
            raise KeyError(
                f"[{self._name}] Key '{key}' not found. "
                f"Available: {available}"
            )
        return self[key](*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self._name}({list(self.keys())})"


# 创建全局注册器实例
ALGO_REGISTRY = Registry("ALGO_REGISTRY")      # 量化算法注册表
MODEL_REGISTRY = Registry("MODEL_REGISTRY")    # 模型构造函数注册表
KV_REGISTRY = Registry("KV_REGISTRY")          # KV-Cache实现注册表
EXPORT_REGISTRY = Registry("EXPORT_REGISTRY")  # 后端导出器注册表
