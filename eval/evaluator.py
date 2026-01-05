"""
评估器：提供统一的评估接口
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from tqdm import tqdm
import json
import os

from ..utils import get_logger
from ..utils.config import EasyDict
from .metrics import (
    compute_ppl,
    compute_accuracy,
    compute_mse,
    compute_cosine_similarity,
    compute_snr,
    compute_quantization_error_stats,
)

logger = get_logger(__name__)


class Evaluator(ABC):
    """
    评估器基类
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EasyDict] = None,
    ):
        self.model = model
        self.config = config or EasyDict()
        self.results = {}
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        执行评估
        
        Returns:
            评估结果字典
        """
        pass
    
    def save_results(self, save_path: str) -> None:
        """保存评估结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    
    def print_results(self) -> None:
        """打印评估结果"""
        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        for key, value in self.results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 50)


class PPLEvaluator(Evaluator):
    """
    困惑度评估器
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[EasyDict] = None,
    ):
        super().__init__(model, config)
        self.tokenizer = tokenizer
    
    def evaluate(
        self,
        dataloader=None,
        dataset_name: str = 'wikitext2',
        max_samples: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        评估困惑度
        
        Args:
            dataloader: 数据加载器
            dataset_name: 数据集名称
            max_samples: 最大样本数
        
        Returns:
            {'ppl': float, 'dataset': str}
        """
        logger.info(f"Evaluating PPL on {dataset_name}...")
        
        if dataloader is None:
            dataloader = self._get_default_dataloader(dataset_name)
        
        ppl = compute_ppl(
            self.model,
            dataloader,
            max_samples=max_samples,
        )
        
        self.results = {
            'ppl': ppl,
            'dataset': dataset_name,
            'max_samples': max_samples,
        }
        
        logger.info(f"PPL on {dataset_name}: {ppl:.2f}")
        return self.results
    
    def _get_default_dataloader(self, dataset_name: str):
        """获取默认数据加载器"""
        # 简化实现，实际应该加载真实数据集
        logger.warning(f"Using dummy dataloader for {dataset_name}")
        
        # 创建dummy数据
        batch_size = self.config.get('batch_size', 1)
        seq_len = self.config.get('seq_len', 2048)
        num_samples = self.config.get('num_samples', 10)
        
        class DummyDataloader:
            def __init__(self, tokenizer, batch_size, seq_len, num_samples):
                self.tokenizer = tokenizer
                self.batch_size = batch_size
                self.seq_len = seq_len
                self.num_samples = num_samples
            
            def __iter__(self):
                for _ in range(self.num_samples):
                    # 随机生成token ids
                    input_ids = torch.randint(
                        0, self.tokenizer.vocab_size,
                        (self.batch_size, self.seq_len)
                    )
                    yield {'input_ids': input_ids, 'labels': input_ids}
            
            def __len__(self):
                return self.num_samples
        
        return DummyDataloader(self.tokenizer, batch_size, seq_len, num_samples)


class AccuracyEvaluator(Evaluator):
    """
    精度评估器：比较量化前后的输出差异
    """
    
    def __init__(
        self,
        model_fp: nn.Module,
        model_quant: nn.Module,
        config: Optional[EasyDict] = None,
    ):
        super().__init__(model_quant, config)
        self.model_fp = model_fp
        self.model_quant = model_quant
    
    def evaluate(
        self,
        dataloader=None,
        max_samples: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        评估量化精度
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数
        
        Returns:
            精度指标字典
        """
        logger.info("Evaluating quantization accuracy...")
        
        self.model_fp.eval()
        self.model_quant.eval()
        
        total_mse = 0.0
        total_cos_sim = 0.0
        total_snr = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_samples > 0 and i >= max_samples:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                else:
                    input_ids = batch[0]
                
                device_fp = next(self.model_fp.parameters()).device
                device_quant = next(self.model_quant.parameters()).device
                
                # FP模型输出
                output_fp = self.model_fp(input_ids.to(device_fp))
                if hasattr(output_fp, 'logits'):
                    output_fp = output_fp.logits
                
                # 量化模型输出
                output_quant = self.model_quant(input_ids.to(device_quant))
                if hasattr(output_quant, 'logits'):
                    output_quant = output_quant.logits
                
                # 移到同一设备比较
                output_fp = output_fp.cpu().float()
                output_quant = output_quant.cpu().float()
                
                total_mse += compute_mse(output_fp, output_quant)
                total_cos_sim += compute_cosine_similarity(output_fp, output_quant)
                total_snr += compute_snr(output_fp, output_quant)
                num_samples += 1
        
        self.results = {
            'mse': total_mse / num_samples,
            'cosine_similarity': total_cos_sim / num_samples,
            'snr_db': total_snr / num_samples,
            'num_samples': num_samples,
        }
        
        logger.info(f"MSE: {self.results['mse']:.6f}")
        logger.info(f"Cosine Similarity: {self.results['cosine_similarity']:.6f}")
        logger.info(f"SNR: {self.results['snr_db']:.2f} dB")
        
        return self.results


class LayerwiseEvaluator(Evaluator):
    """
    逐层评估器：评估每一层的量化误差
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EasyDict] = None,
    ):
        super().__init__(model, config)
        self.layer_stats = {}
    
    def evaluate(
        self,
        fp_weights: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        评估每层的量化误差
        
        Args:
            fp_weights: FP权重字典，如果为None则从模型中获取
        
        Returns:
            逐层误差统计
        """
        logger.info("Evaluating layer-wise quantization error...")
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight_quant = module.weight.data
                
                # 获取FP权重
                if fp_weights and name in fp_weights:
                    weight_fp = fp_weights[name]
                elif hasattr(module, 'weight_fp'):
                    weight_fp = module.weight_fp
                else:
                    continue
                
                stats = compute_quantization_error_stats(weight_fp, weight_quant)
                self.layer_stats[name] = stats
        
        # 汇总统计
        if self.layer_stats:
            avg_mse = sum(s['mse'] for s in self.layer_stats.values()) / len(self.layer_stats)
            avg_cos_sim = sum(s['cosine_sim'] for s in self.layer_stats.values()) / len(self.layer_stats)
            max_error_layer = max(self.layer_stats.items(), key=lambda x: x[1]['max_error'])
            
            self.results = {
                'num_layers': len(self.layer_stats),
                'avg_mse': avg_mse,
                'avg_cosine_sim': avg_cos_sim,
                'max_error_layer': max_error_layer[0],
                'max_error_value': max_error_layer[1]['max_error'],
                'layer_stats': self.layer_stats,
            }
        else:
            self.results = {'num_layers': 0}
        
        return self.results


class SpeedEvaluator(Evaluator):
    """
    速度评估器
    """
    
    def evaluate(
        self,
        input_shape: tuple = (1, 2048),
        warmup_steps: int = 5,
        test_steps: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        评估推理速度
        
        Args:
            input_shape: 输入形状 (batch_size, seq_len)
            warmup_steps: 预热步数
            test_steps: 测试步数
        
        Returns:
            速度指标
        """
        import time
        
        logger.info("Evaluating inference speed...")
        
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # 创建dummy输入
        input_ids = torch.randint(0, 32000, input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = self.model(input_ids)
        
        # 同步
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_steps):
                _ = self.model(input_ids)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / test_steps
        tokens_per_second = input_shape[0] * input_shape[1] / avg_time
        
        self.results = {
            'total_time_s': total_time,
            'avg_time_ms': avg_time * 1000,
            'tokens_per_second': tokens_per_second,
            'batch_size': input_shape[0],
            'seq_len': input_shape[1],
            'test_steps': test_steps,
        }
        
        logger.info(f"Average inference time: {self.results['avg_time_ms']:.2f} ms")
        logger.info(f"Tokens per second: {self.results['tokens_per_second']:.0f}")
        
        return self.results
