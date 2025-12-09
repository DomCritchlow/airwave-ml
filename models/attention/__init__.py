"""
Attention-based Seq2Seq Audio-to-Text Model

Full Transformer encoder-decoder with cross-attention for transcription.

Key characteristics:
- Teacher forcing during training
- Autoregressive decoding (beam search or greedy)
- Implicit language model (can correct errors, may hallucinate)

Usage:
    # Training
    python train.py --config config.yaml
    
    # Inference
    python inference.py --checkpoint checkpoints/best_model.pt --audio test.wav
"""

__version__ = "0.1.0"

from .model import AudioToTextModel, count_parameters
from .dataset import AudioTextDataset, load_dataset, get_dataloaders
from .utils import load_config, save_checkpoint, load_checkpoint
from .inference import load_audio, transcribe_audio, batch_transcribe

__all__ = [
    # Model
    'AudioToTextModel',
    'count_parameters',
    # Dataset
    'AudioTextDataset',
    'load_dataset',
    'get_dataloaders',
    # Inference
    'load_audio',
    'transcribe_audio',
    'batch_transcribe',
    # Utils
    'load_config',
    'save_checkpoint',
    'load_checkpoint',
]
