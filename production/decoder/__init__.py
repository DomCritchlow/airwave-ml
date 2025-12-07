"""
Production Morse Code Decoder

Components:
- MorseDecoder: Model wrapper for inference
- StreamingBuffer: Handle continuous audio with overlap
- TextMerger: Merge overlapping text outputs
- AudioSource: Input from mic, file, or callback
"""

from .model_wrapper import MorseDecoder
from .buffer import StreamingBuffer, TextMerger, VADBuffer
from .audio_input import AudioSource, MicrophoneInput, FileInput, create_audio_source

__all__ = [
    'MorseDecoder',
    'StreamingBuffer',
    'TextMerger',
    'VADBuffer',
    'AudioSource',
    'MicrophoneInput',
    'FileInput',
    'create_audio_source'
]

