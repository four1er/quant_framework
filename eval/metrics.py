"""
评估指标计算
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union
import numpy as np


def compute_ppl(
    model: torch.nn.Module,
    dataloader,
    device: Optional[torch.device] = None,
    max_samples: int = -1,
) -> float:
    """
    计算困惑度(Perplexity)
    
    Args:
        model: 语言模型
        dataloader: 数据加载器，返回(input_ids, labels)
        device: 计算设备
        max_samples: 最大样本数，-1表示使用全部
    
    Returns:
        困惑度值
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_samples > 0 and i >= max_samples:
                break
            
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else input_ids
                attention_mask = None
            
            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # 计算有效token数
            if attention_mask is not None:
                num_tokens = attention_mask.sum().item()
            else:
                num_tokens = input_ids.numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = np.exp(avg_loss)
    
    return ppl


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测logits [batch, seq, vocab] 或 [batch, seq]
        labels: 标签 [batch, seq]
        ignore_index: 忽略的标签值
    
    Returns:
        准确率
    """
    if predictions.dim() == 3:
        # logits -> predictions
        predictions = predictions.argmax(dim=-1)
    
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_mse(
    output_fp: torch.Tensor,
    output_quant: torch.Tensor,
    reduction: str = 'mean',
) -> float:
    """
    计算量化前后输出的MSE
    
    Args:
        output_fp: FP模型输出
        output_quant: 量化模型输出
        reduction: 'mean', 'sum', 'none'
    
    Returns:
        MSE值
    """
    mse = F.mse_loss(output_fp, output_quant, reduction=reduction)
    
    if reduction == 'none':
        return mse
    return mse.item()


def compute_cosine_similarity(
    output_fp: torch.Tensor,
    output_quant: torch.Tensor,
) -> float:
    """
    计算量化前后输出的余弦相似度
    
    Args:
        output_fp: FP模型输出
        output_quant: 量化模型输出
    
    Returns:
        余弦相似度
    """
    # Flatten
    fp_flat = output_fp.flatten().float()
    quant_flat = output_quant.flatten().float()
    
    cos_sim = F.cosine_similarity(fp_flat.unsqueeze(0), quant_flat.unsqueeze(0))
    return cos_sim.item()


def compute_snr(
    output_fp: torch.Tensor,
    output_quant: torch.Tensor,
) -> float:
    """
    计算信噪比(Signal-to-Noise Ratio)
    
    Args:
        output_fp: FP模型输出（信号）
        output_quant: 量化模型输出
    
    Returns:
        SNR (dB)
    """
    signal = output_fp.float()
    noise = (output_fp - output_quant).float()
    
    signal_power = (signal ** 2).mean()
    noise_power = (noise ** 2).mean()
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_quantization_error_stats(
    weight_fp: torch.Tensor,
    weight_quant: torch.Tensor,
) -> dict:
    """
    计算量化误差统计
    
    Args:
        weight_fp: FP权重
        weight_quant: 量化后权重
    
    Returns:
        统计信息字典
    """
    error = (weight_fp - weight_quant).abs()
    
    return {
        'mean_error': error.mean().item(),
        'max_error': error.max().item(),
        'std_error': error.std().item(),
        'relative_error': (error / (weight_fp.abs() + 1e-8)).mean().item(),
        'mse': compute_mse(weight_fp, weight_quant),
        'cosine_sim': compute_cosine_similarity(weight_fp, weight_quant),
        'snr_db': compute_snr(weight_fp, weight_quant),
    }
