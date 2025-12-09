"""
CTC Model with Self-Supervised Pretraining

This module provides a CTC model variant that supports:
1. Masked spectrogram pretraining (self-supervised)
2. Loading pretrained encoder weights for fine-tuning

Key advantages:
- Better generalization with limited labeled data
- Faster convergence during fine-tuning
- Domain adaptation (synthetic → real-world)

Usage:
    # Pretraining
    python pretrain_masked_ctc_w_pretrain.py --data-dirs audio_dir/ --epochs 50
    
    # Fine-tuning
    python train.py --config config.yaml --pretrained-encoder checkpoints/pretrained_encoder.pt
    
    # Inference
    python inference.py --checkpoint checkpoints/best_model.pt --audio test.wav
"""

from .model import (
    CTCModel,
    CTCEncoder,
    ConvFrontend,
    PositionalEncoding,
    TinyEncoderConfig,
    create_ctc_model,
    create_tiny_encoder
)
from .dataset import (
    CTCAudioTextDataset,
    RadioUnlabeledDataset,
    load_ctc_dataset,
    load_unlabeled_dataset,
    create_ctc_vocab
)
from .inference import CTCDecoder

__all__ = [
    # Model
    'CTCModel',
    'CTCEncoder',
    'ConvFrontend',
    'PositionalEncoding',
    'TinyEncoderConfig',
    'create_ctc_model',
    'create_tiny_encoder',
    # Dataset
    'CTCAudioTextDataset',
    'RadioUnlabeledDataset',
    'load_ctc_dataset',
    'load_unlabeled_dataset',
    'create_ctc_vocab',
    # Inference
    'CTCDecoder'
]
