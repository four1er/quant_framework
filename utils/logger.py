"""
日志系统：统一的日志管理
"""
import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


# 全局日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 日志级别映射
LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 全局logger缓存
_loggers = {}
_initialized = False


def setup_logger(
    name: str = "quant_framework",
    level: str = "info",
    log_file: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """
    设置并返回logger
    
    Args:
        name: logger名称
        level: 日志级别 (debug/info/warning/error/critical)
        log_file: 日志文件路径，None则只输出到控制台
        rank: 分布式训练的rank，非0时降低日志级别
    
    Returns:
        配置好的logger实例
    """
    global _initialized
    
    logger = logging.getLogger(name)
    
    # 避免重复初始化
    if name in _loggers:
        return _loggers[name]
    
    # 设置日志级别
    log_level = LEVEL_MAP.get(level.lower(), logging.INFO)
    
    # 非主进程降低日志级别
    if rank != 0:
        log_level = logging.WARNING
    
    logger.setLevel(log_level)
    
    # 创建formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定）
    if log_file is not None and rank == 0:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传播到root logger
    logger.propagate = False
    
    _loggers[name] = logger
    _initialized = True
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称，None则返回root logger
    
    Returns:
        logger实例
    """
    if name is None:
        name = "quant_framework"
    
    if name in _loggers:
        return _loggers[name]
    
    # 如果还没初始化，创建一个基础logger
    if not _initialized:
        return setup_logger(name)
    
    # 创建子logger
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


class LoggerContext:
    """日志上下文管理器，用于临时改变日志级别"""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = LEVEL_MAP.get(level.lower(), logging.INFO)
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


def log_block_replace(logger: logging.Logger, block_idx: int, layer_name: str, 
                      old_type: str, new_type: str) -> None:
    """记录模块替换日志"""
    logger.info(
        f"Block {block_idx:3d} | {layer_name:40s} | {old_type} -> {new_type}"
    )


def log_quant_params(logger: logging.Logger, layer_name: str, 
                     scale_shape: tuple, bit: int, granularity: str) -> None:
    """记录量化参数日志"""
    logger.debug(
        f"Quant params | {layer_name:40s} | "
        f"scale_shape={scale_shape}, bit={bit}, granularity={granularity}"
    )
