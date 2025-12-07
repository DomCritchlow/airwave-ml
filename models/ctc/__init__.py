"""
CTC-based Audio-to-Text Model

This module provides an alternative to the seq2seq Transformer model,
using CTC (Connectionist Temporal Classification) loss for training.

Key advantages over seq2seq:
- No language model hallucination (better for call signs like W1ABC)
- Faster inference (single forward pass, no autoregressive decoding)
- Simpler architecture (encoder only)
- Better for arbitrary character sequences

Usage:
    # Training
    python src_ctc/train.py --config config_ctc.yaml
    
    # Inference
    python src_ctc/inference.py --checkpoint checkpoints_ctc/best_model_ctc.pt --audio test.wav
"""

from .model import CTCModel, create_ctc_model
from .dataset import CTCAudioTextDataset, load_ctc_dataset, create_ctc_vocab
from .inference import CTCDecoder

__all__ = [
    'CTCModel',
    'create_ctc_model',
    'CTCAudioTextDataset',
    'load_ctc_dataset',
    'create_ctc_vocab',
    'CTCDecoder'
]

