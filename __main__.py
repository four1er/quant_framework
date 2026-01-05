"""
入口层：解析配置、分布式初始化、驱动整体流程
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

from .utils import load_config, check_config, setup_logger, get_logger
from .utils import ALGO_REGISTRY, MODEL_REGISTRY
from .models import *  # 注册所有模型
from .algorithms import *  # 注册所有算法
from .data import get_calib_dataset
from .eval import PPLEvaluator, AccuracyEvaluator, SpeedEvaluator
from .export import VLLMExporter, HuggingFaceExporter, GGUFExporter


def parse_args():
    parser = argparse.ArgumentParser(description="LightQuant: 工业级大模型量化框架")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="配置文件路径 (YAML格式)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="日志级别"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径"
    )
    return parser.parse_args()


def init_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def set_seed(seed: int, rank: int = 0):
    """设置随机种子"""
    import random
    import numpy as np
    
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_modality_configs(config):
    """获取各模态的量化配置"""
    modalities = []
    modality_configs = []
    
    if 'quant' in config:
        for modality_name, modality_config in config.quant.items():
            if isinstance(modality_config, dict) and 'method' in modality_config:
                modalities.append(modality_name)
                modality_configs.append(modality_config)
    
    return modalities, modality_configs


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化分布式
    rank, local_rank, world_size = init_distributed()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger(
        name="quant_framework",
        level=args.log_level,
        log_file=args.log_file,
        rank=rank,
    )
    
    logger.info("=" * 60)
    logger.info("LightQuant: 工业级大模型量化框架")
    logger.info("=" * 60)
    
    # 配置校验
    check_config(config)
    
    # 设置随机种子
    set_seed(config.base.seed, rank)
    
    # ========== 阶段1：模型构建 ==========
    logger.info("Stage 1: Building model...")
    model_type = config.model.type
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model = model_cls(config)
    logger.info(f"Model loaded: {model_type}")
    
    # ========== 阶段2：数据准备与首块捕获 ==========
    logger.info("Stage 2: Preparing calibration data...")
    
    calib_data = None
    padding_mask = None
    first_block_input = None
    
    if 'calib' in config and config.calib.get('name'):
        calib_data, padding_mask = get_calib_dataset(
            tokenizer=model.tokenizer,
            config=config.calib,
        )
        logger.info(f"Calibration data prepared: {len(calib_data)} samples")
        
        # 捕获首块输入
        first_block_input = model.collect_first_block_input(calib_data, padding_mask)
        logger.info("First block input captured")
    else:
        logger.info("No calibration data, running in data-free mode")
    
    # ========== 阶段3：Blockwise量化 ==========
    logger.info("Stage 3: Blockwise quantization...")
    
    modalities, modality_configs = get_modality_configs(config)
    blockwise_opts = []
    
    for modality, modality_config in zip(modalities, modality_configs):
        logger.info(f"Processing modality: {modality}")
        
        # 切换模态
        model.set_modality(modality)
        
        # 获取算法类
        method = modality_config.method
        algo_cls = ALGO_REGISTRY.get(method)
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm: {method}. Available: {list(ALGO_REGISTRY.keys())}")
        
        # 实例化算法
        blockwise_opt = algo_cls(
            model=model,
            quant_config=modality_config,
            input=first_block_input,
            padding_mask=padding_mask,
            config=config,
        )
        
        # 运行Blockwise循环
        blockwise_opt.run_block_loop()
        blockwise_opts.append(blockwise_opt)
        
        logger.info(f"Modality {modality} quantization completed")
    
    # ========== 阶段4：评测 ==========
    logger.info("Stage 4: Evaluation...")
    
    if 'eval' in config:
        eval_config = config.eval
        
        # PPL评估
        if eval_config.get('ppl', False):
            logger.info("Running PPL evaluation...")
            ppl_evaluator = PPLEvaluator(model.model, model.tokenizer)
            ppl_results = ppl_evaluator.evaluate(
                dataset_name=eval_config.get('dataset', 'wikitext2'),
                max_samples=eval_config.get('max_samples', -1),
            )
            ppl_evaluator.print_results()
        
        # 速度评估
        if eval_config.get('speed', False):
            logger.info("Running speed evaluation...")
            speed_evaluator = SpeedEvaluator(model.model)
            speed_results = speed_evaluator.evaluate(
                input_shape=(1, eval_config.get('seq_len', 2048)),
            )
            speed_evaluator.print_results()
    
    # ========== 阶段5：导出 ==========
    logger.info("Stage 5: Export...")
    
    if 'save' in config:
        save_path = Path(config.save.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 获取量化配置
        quant_config = modality_configs[0] if modality_configs else config.get('quant', {})
        
        # 确定导出格式
        export_format = config.save.get('format', 'huggingface')
        
        if export_format == 'vllm' or config.save.get('save_vllm', False):
            # 替换为RealQuant模块
            for opt in blockwise_opts:
                for block in model.get_blocks():
                    opt.replace_linears(block, mode='vllm_quant')
            
            exporter = VLLMExporter(model.model, quant_config, str(save_path))
            exporter.export(tokenizer=model.tokenizer)
            
        elif export_format == 'gguf':
            exporter = GGUFExporter(model.model, quant_config, str(save_path))
            exporter.export(
                tokenizer=model.tokenizer,
                quantization_type=config.save.get('gguf_type', 'q4_k'),
            )
            
        else:
            # 默认HuggingFace格式
            for opt in blockwise_opts:
                for block in model.get_blocks():
                    opt.replace_linears(block, mode='fake_quant')
            
            exporter = HuggingFaceExporter(model.model, quant_config, str(save_path))
            exporter.export(tokenizer=model.tokenizer)
        
        logger.info(f"Model saved to {save_path}")
    
    logger.info("=" * 60)
    logger.info("Quantization completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
