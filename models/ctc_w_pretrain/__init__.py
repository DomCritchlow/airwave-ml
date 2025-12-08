"""
CTC model with pretraining support.

This variant supports masked spectrogram pretraining before CTC fine-tuning.
"""

from .model import (
    ConvFrontend,
    PositionalEncoding,
    CTCEncoder,
    CTCModel,
    create_ctc_model,
    create_tiny_encoder
)

