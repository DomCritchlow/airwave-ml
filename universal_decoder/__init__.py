"""
Universal Radio Decoder

A modular system for detecting and decoding radio signals.

Components:
- detector: CNN-based signal detection on spectrograms
- extractor: Bandpass filtering and signal isolation
- decoders: Mode-specific decoders (CW, PSK31, etc.)
- pipeline: Main orchestrator

Usage:
    from universal_decoder import UniversalRadioDecoder
    
    decoder = UniversalRadioDecoder()
    decoder.load_models(cw_checkpoint='checkpoints_ctc/best_model_ctc.pt')
    
    results = decoder.decode_audio(audio, sample_rate=48000)
    
    for r in results:
        print(f"{r.center_freq_hz/1000:.1f} kHz [{r.mode}]: {r.text}")
"""

from .pipeline.radio_decoder import (
    UniversalRadioDecoder,
    StreamingRadioDecoder,
    RadioDecoderConfig,
    create_decoder
)

from .detector.signal_detector import (
    SignalDetectorCNN,
    SignalMode,
    DetectedSignal
)

from .extractor.signal_extractor import (
    SignalExtractor,
    SpectrogramGenerator,
    ExtractedSignal
)

from .decoders.decoder_router import (
    DecoderRouter,
    DecodedResult,
    BaseDecoder
)

__all__ = [
    # Pipeline
    'UniversalRadioDecoder',
    'StreamingRadioDecoder',
    'RadioDecoderConfig',
    'create_decoder',
    
    # Detector
    'SignalDetectorCNN',
    'SignalMode',
    'DetectedSignal',
    
    # Extractor
    'SignalExtractor',
    'SpectrogramGenerator',
    'ExtractedSignal',
    
    # Decoders
    'DecoderRouter',
    'DecodedResult',
    'BaseDecoder',
]

