"""
校准数据集

支持多种校准数据源:
- wikitext2
- c4
- pile
- 自定义数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any, Union
import random

from ..utils import get_logger
from ..utils.config import EasyDict

logger = get_logger(__name__)


class CalibrationDataset(Dataset):
    """
    校准数据集基类
    """
    
    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        num_samples: int = 128,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
    
    def load_data(self) -> None:
        """加载数据，子类实现"""
        raise NotImplementedError


class Wikitext2Dataset(CalibrationDataset):
    """
    Wikitext-2校准数据集
    """
    
    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        num_samples: int = 128,
        split: str = 'train',
    ):
        super().__init__(tokenizer, seq_len, num_samples)
        self.split = split
        self.load_data()
    
    def load_data(self) -> None:
        """加载Wikitext-2数据"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=self.split)
            
            # 合并所有文本
            text = '\n\n'.join([item['text'] for item in dataset if item['text'].strip()])
            
            # Tokenize
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,
            )
            
            input_ids = encodings['input_ids'][0]
            
            # 切分为固定长度的样本
            for i in range(0, len(input_ids) - self.seq_len, self.seq_len):
                if len(self.data) >= self.num_samples:
                    break
                
                sample_ids = input_ids[i:i + self.seq_len]
                self.data.append({
                    'input_ids': sample_ids,
                    'attention_mask': torch.ones_like(sample_ids),
                })
            
            logger.info(f"Loaded {len(self.data)} samples from Wikitext-2")
            
        except Exception as e:
            logger.warning(f"Failed to load Wikitext-2: {e}")
            logger.info("Using random data as fallback")
            self._generate_random_data()
    
    def _generate_random_data(self) -> None:
        """生成随机数据作为后备"""
        vocab_size = self.tokenizer.vocab_size
        
        for _ in range(self.num_samples):
            input_ids = torch.randint(0, vocab_size, (self.seq_len,))
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
            })


class C4Dataset(CalibrationDataset):
    """
    C4校准数据集
    """
    
    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        num_samples: int = 128,
    ):
        super().__init__(tokenizer, seq_len, num_samples)
        self.load_data()
    
    def load_data(self) -> None:
        """加载C4数据"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                'allenai/c4',
                'en',
                split='train',
                streaming=True,
            )
            
            collected = 0
            for item in dataset:
                if collected >= self.num_samples:
                    break
                
                text = item['text']
                
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.seq_len,
                    padding='max_length',
                )
                
                if encodings['input_ids'].shape[1] >= self.seq_len // 2:
                    self.data.append({
                        'input_ids': encodings['input_ids'][0],
                        'attention_mask': encodings['attention_mask'][0],
                    })
                    collected += 1
            
            logger.info(f"Loaded {len(self.data)} samples from C4")
            
        except Exception as e:
            logger.warning(f"Failed to load C4: {e}")
            logger.info("Using random data as fallback")
            self._generate_random_data()
    
    def _generate_random_data(self) -> None:
        """生成随机数据"""
        vocab_size = self.tokenizer.vocab_size
        
        for _ in range(self.num_samples):
            input_ids = torch.randint(0, vocab_size, (self.seq_len,))
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
            })


class CustomDataset(CalibrationDataset):
    """
    自定义校准数据集
    """
    
    def __init__(
        self,
        tokenizer,
        data_path: str,
        seq_len: int = 2048,
        num_samples: int = 128,
    ):
        super().__init__(tokenizer, seq_len, num_samples)
        self.data_path = data_path
        self.load_data()
    
    def load_data(self) -> None:
        """加载自定义数据"""
        import json
        import os
        
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path not found: {self.data_path}")
            self._generate_random_data()
            return
        
        try:
            # 支持多种格式
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r') as f:
                    raw_data = json.load(f)
            elif self.data_path.endswith('.jsonl'):
                raw_data = []
                with open(self.data_path, 'r') as f:
                    for line in f:
                        raw_data.append(json.loads(line))
            elif self.data_path.endswith('.txt'):
                with open(self.data_path, 'r') as f:
                    raw_data = [{'text': f.read()}]
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
            
            # 处理数据
            for item in raw_data[:self.num_samples]:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get('text', item.get('content', str(item)))
                else:
                    continue
                
                encodings = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.seq_len,
                    padding='max_length',
                )
                
                self.data.append({
                    'input_ids': encodings['input_ids'][0],
                    'attention_mask': encodings['attention_mask'][0],
                })
            
            logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load custom data: {e}")
            self._generate_random_data()
    
    def _generate_random_data(self) -> None:
        """生成随机数据"""
        vocab_size = self.tokenizer.vocab_size
        
        for _ in range(self.num_samples):
            input_ids = torch.randint(0, vocab_size, (self.seq_len,))
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
            })


def get_calib_dataset(
    tokenizer,
    config: EasyDict,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    获取校准数据集
    
    Args:
        tokenizer: tokenizer对象
        config: 校准配置
            - name: 数据集名称 ('wikitext2', 'c4', 'custom')
            - path: 自定义数据集路径
            - n_samples: 样本数
            - seq_len: 序列长度
    
    Returns:
        (input_data_list, padding_mask)
    """
    dataset_name = config.get('name', 'wikitext2')
    seq_len = config.get('seq_len', 2048)
    n_samples = config.get('n_samples', 128)
    data_path = config.get('path', None)
    
    logger.info(f"Loading calibration dataset: {dataset_name}")
    logger.info(f"  Samples: {n_samples}, Seq length: {seq_len}")
    
    # 创建数据集
    if dataset_name == 'wikitext2':
        dataset = Wikitext2Dataset(tokenizer, seq_len, n_samples)
    elif dataset_name == 'c4':
        dataset = C4Dataset(tokenizer, seq_len, n_samples)
    elif dataset_name == 'custom':
        if data_path is None:
            raise ValueError("Custom dataset requires 'path' in config")
        dataset = CustomDataset(tokenizer, data_path, seq_len, n_samples)
    else:
        logger.warning(f"Unknown dataset: {dataset_name}, using wikitext2")
        dataset = Wikitext2Dataset(tokenizer, seq_len, n_samples)
    
    # 转换为列表格式
    input_data = []
    attention_masks = []
    
    for item in dataset:
        input_data.append(item['input_ids'].unsqueeze(0))  # [1, seq_len]
        attention_masks.append(item['attention_mask'].unsqueeze(0))
    
    # 合并attention_mask
    if attention_masks:
        padding_mask = torch.cat(attention_masks, dim=0)  # [n_samples, seq_len]
    else:
        padding_mask = None
    
    logger.info(f"Calibration data ready: {len(input_data)} samples")
    
    return input_data, padding_mask


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    DataLoader的collate函数
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }


def get_calib_dataloader(
    tokenizer,
    config: EasyDict,
    batch_size: int = 1,
) -> DataLoader:
    """
    获取校准数据的DataLoader
    """
    dataset_name = config.get('name', 'wikitext2')
    seq_len = config.get('seq_len', 2048)
    n_samples = config.get('n_samples', 128)
    data_path = config.get('path', None)
    
    if dataset_name == 'wikitext2':
        dataset = Wikitext2Dataset(tokenizer, seq_len, n_samples)
    elif dataset_name == 'c4':
        dataset = C4Dataset(tokenizer, seq_len, n_samples)
    elif dataset_name == 'custom':
        dataset = CustomDataset(tokenizer, data_path, seq_len, n_samples)
    else:
        dataset = Wikitext2Dataset(tokenizer, seq_len, n_samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
