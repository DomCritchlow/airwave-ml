"""
Universal Radio Decoder Pipeline

Main orchestrator that ties together:
1. Signal Detection (CNN on spectrograms)
2. Signal Extraction (bandpass filtering)
3. Mode-specific Decoding (CW, PSK31, etc.)

Usage:
    from universal_decoder.pipeline.radio_decoder import UniversalRadioDecoder
    
    decoder = UniversalRadioDecoder()
    decoder.load_models()
    
    # Decode wideband audio
    results = decoder.decode_audio(audio, sample_rate)
    
    for result in results:
        print(f"{result.center_freq_hz/1000:.1f} kHz [{result.mode}]: {result.text}")
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detector.signal_detector import SignalDetectorCNN, SignalMode, DetectedSignal
from extractor.signal_extractor import SignalExtractor, SpectrogramGenerator, get_typical_bandwidth
from decoders.decoder_router import DecoderRouter, DecodedResult


@dataclass
class RadioDecoderConfig:
    """Configuration for the universal decoder."""
    # Audio settings
    input_sample_rate: int = 48000
    decoder_sample_rate: int = 16000
    
    # Spectrogram settings
    fft_size: int = 2048
    hop_size: int = 512
    
    # Detector settings
    detector_input_height: int = 256
    detector_input_width: int = 512
    detection_threshold: float = 0.5
    nms_threshold: float = 0.3
    
    # Frequency range (relative to audio baseband)
    freq_min: float = 0
    freq_max: float = 24000  # Nyquist of 48kHz
    
    # Paths
    detector_checkpoint: Optional[str] = None
    cw_decoder_checkpoint: Optional[str] = None
    psk31_decoder_checkpoint: Optional[str] = None


class UniversalRadioDecoder:
    """
    Universal radio decoder pipeline.
    
    Flow:
        Wideband Audio
              ↓
        Spectrogram Generation
              ↓
        Signal Detection (CNN)
              ↓
        For each detected signal:
            ↓
        Signal Extraction (Bandpass)
              ↓
        Mode-specific Decoding
              ↓
        Aggregated Results
    """
    
    def __init__(self, config: Optional[RadioDecoderConfig] = None):
        """
        Initialize the universal decoder.
        
        Args:
            config: Configuration object
        """
        self.config = config or RadioDecoderConfig()
        
        # Components (created on load)
        self.detector: Optional[SignalDetectorCNN] = None
        self.extractor: Optional[SignalExtractor] = None
        self.spectrogram_gen: Optional[SpectrogramGenerator] = None
        self.decoder_router: Optional[DecoderRouter] = None
        
        self._loaded = False
    
    def load_models(
        self,
        detector_checkpoint: Optional[str] = None,
        cw_checkpoint: Optional[str] = None
    ):
        """
        Load all model components.
        
        Args:
            detector_checkpoint: Path to signal detector checkpoint
            cw_checkpoint: Path to CW decoder checkpoint
        """
        print("Loading Universal Radio Decoder...")
        
        # Signal detector
        self.detector = SignalDetectorCNN(
            input_height=self.config.detector_input_height,
            input_width=self.config.detector_input_width
        )
        
        if detector_checkpoint or self.config.detector_checkpoint:
            path = detector_checkpoint or self.config.detector_checkpoint
            if Path(path).exists():
                checkpoint = torch.load(path, map_location='cpu')
                self.detector.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded detector from {path}")
            else:
                print(f"Detector checkpoint not found: {path}")
                print("Using untrained detector (will not detect signals)")
        else:
            print("No detector checkpoint - using untrained model")
        
        self.detector.eval()
        
        # Signal extractor
        self.extractor = SignalExtractor(
            sample_rate=self.config.input_sample_rate,
            output_sample_rate=self.config.decoder_sample_rate
        )
        
        # Spectrogram generator
        self.spectrogram_gen = SpectrogramGenerator(
            sample_rate=self.config.input_sample_rate,
            fft_size=self.config.fft_size,
            hop_size=self.config.hop_size,
            freq_min=self.config.freq_min,
            freq_max=self.config.freq_max
        )
        
        # Decoder router
        self.decoder_router = DecoderRouter(lazy_load=True)
        self.decoder_router.register_default_decoders(
            cw_checkpoint=cw_checkpoint or self.config.cw_decoder_checkpoint,
            psk31_checkpoint=self.config.psk31_decoder_checkpoint
        )
        
        self._loaded = True
        print("Universal Radio Decoder ready")
    
    def decode_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        freq_offset: float = 0,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[DecodedResult]:
        """
        Decode all signals in wideband audio.
        
        Args:
            audio: Wideband audio samples
            sample_rate: Sample rate (default: config.input_sample_rate)
            freq_offset: Frequency offset if audio is from SDR
                        (e.g., if 7 MHz band is mixed to 0-48kHz, offset=7000000)
            time_range: Optional (start, end) time slice
            
        Returns:
            List of DecodedResults for each detected signal
        """
        if not self._loaded:
            self.load_models()
        
        sample_rate = sample_rate or self.config.input_sample_rate
        
        # Generate spectrogram for detection
        spectrogram = self.spectrogram_gen.generate_for_detector(
            audio,
            target_height=self.config.detector_input_height,
            target_width=self.config.detector_input_width
        )
        
        # Convert to tensor
        spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)
        
        # Detect signals
        freq_range = (
            self.config.freq_min + freq_offset,
            self.config.freq_max + freq_offset
        )
        time_range = time_range or (0, len(audio) / sample_rate)
        
        detected_signals = self.detector.detect(
            spectrogram_tensor,
            freq_range=freq_range,
            time_range=time_range,
            threshold=self.config.detection_threshold,
            nms_threshold=self.config.nms_threshold
        )
        
        print(f"Detected {len(detected_signals)} signals")
        
        # Decode each signal
        results = []
        
        for signal in detected_signals:
            result = self._decode_signal(audio, sample_rate, signal, freq_offset)
            results.append(result)
        
        return results
    
    def _decode_signal(
        self,
        audio: np.ndarray,
        sample_rate: int,
        signal: DetectedSignal,
        freq_offset: float
    ) -> DecodedResult:
        """
        Extract and decode a single detected signal.
        """
        # Get mode name
        mode_name = signal.mode.name
        
        # Get appropriate bandwidth for mode
        bandwidth = max(signal.bandwidth_hz, get_typical_bandwidth(mode_name))
        
        # Extract the signal
        extracted = self.extractor.extract(
            audio=audio,
            center_freq=signal.center_freq_hz,
            bandwidth=bandwidth,
            freq_offset=freq_offset
        )
        
        print(f"Extracting {signal.center_freq_hz/1000:.1f} kHz, "
              f"BW={bandwidth:.0f} Hz, mode={mode_name}")
        
        # Decode
        result = self.decoder_router.decode(
            audio=extracted.audio,
            sample_rate=extracted.sample_rate,
            mode=mode_name,
            center_freq_hz=signal.center_freq_hz
        )
        
        # Update confidence from detection
        result.confidence = signal.confidence * result.confidence
        
        return result
    
    def decode_known_signal(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str,
        center_freq: float = 0,
        bandwidth: Optional[float] = None
    ) -> DecodedResult:
        """
        Decode a signal when mode is already known (skip detection).
        
        Useful for:
        - Testing decoders
        - Narrowband input where detection isn't needed
        - Manual mode selection
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            mode: Signal mode (CW, PSK31, etc.)
            center_freq: Center frequency (for metadata)
            bandwidth: Signal bandwidth (uses default for mode if not specified)
        """
        if not self._loaded:
            self.load_models()
        
        # If audio needs extraction (wideband), extract first
        if bandwidth and center_freq:
            extracted = self.extractor.extract(
                audio=audio,
                center_freq=center_freq,
                bandwidth=bandwidth
            )
            audio = extracted.audio
            sample_rate = extracted.sample_rate
        
        # Decode directly
        return self.decoder_router.decode(
            audio=audio,
            sample_rate=sample_rate,
            mode=mode,
            center_freq_hz=center_freq
        )
    
    def get_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spectrogram for visualization.
        
        Returns:
            (spectrogram, frequencies, times)
        """
        if self.spectrogram_gen is None:
            self.spectrogram_gen = SpectrogramGenerator(
                sample_rate=sample_rate or self.config.input_sample_rate,
                fft_size=self.config.fft_size,
                hop_size=self.config.hop_size
            )
        
        return self.spectrogram_gen.generate(audio)
    
    def list_available_modes(self) -> List[str]:
        """List all available decoder modes."""
        if self.decoder_router:
            return self.decoder_router.list_modes()
        return []


class StreamingRadioDecoder:
    """
    Streaming version of the universal decoder for real-time use.
    
    Processes audio in chunks with overlap for continuous decoding.
    """
    
    def __init__(
        self,
        config: Optional[RadioDecoderConfig] = None,
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0
    ):
        self.decoder = UniversalRadioDecoder(config)
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        self.buffer = np.array([])
        self.sample_rate = config.input_sample_rate if config else 48000
        
        # Track recent detections to avoid duplicates
        self.recent_results: List[DecodedResult] = []
    
    def load_models(self, **kwargs):
        """Load decoder models."""
        self.decoder.load_models(**kwargs)
    
    def process_chunk(self, audio_chunk: np.ndarray) -> List[DecodedResult]:
        """
        Process an audio chunk from the stream.
        
        Args:
            audio_chunk: New audio samples
            
        Returns:
            List of decoded results (if any complete signals)
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Check if we have enough for a full chunk
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        
        results = []
        
        while len(self.buffer) >= chunk_samples:
            # Process chunk
            chunk = self.buffer[:chunk_samples]
            chunk_results = self.decoder.decode_audio(chunk, self.sample_rate)
            
            # Filter out duplicates based on frequency and recent results
            new_results = self._filter_duplicates(chunk_results)
            results.extend(new_results)
            
            # Slide buffer
            self.buffer = self.buffer[chunk_samples - overlap_samples:]
        
        return results
    
    def _filter_duplicates(
        self,
        results: List[DecodedResult]
    ) -> List[DecodedResult]:
        """Filter out duplicate detections from overlapping windows."""
        new_results = []
        
        for result in results:
            is_duplicate = False
            
            for recent in self.recent_results[-10:]:  # Check last 10
                # Same frequency and similar text = duplicate
                freq_match = abs(result.center_freq_hz - recent.center_freq_hz) < 100
                text_overlap = self._text_similarity(result.text, recent.text) > 0.8
                
                if freq_match and text_overlap:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_results.append(result)
                self.recent_results.append(result)
        
        # Trim recent results
        if len(self.recent_results) > 50:
            self.recent_results = self.recent_results[-50:]
        
        return new_results
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Simple text similarity (overlap ratio)."""
        if not a or not b:
            return 0.0
        
        # Check if one contains the other
        if a in b or b in a:
            return 1.0
        
        # Count common characters
        common = sum(1 for c in a if c in b)
        return common / max(len(a), len(b))


# Convenience function
def create_decoder(
    detector_checkpoint: Optional[str] = None,
    cw_checkpoint: Optional[str] = None,
    config: Optional[RadioDecoderConfig] = None
) -> UniversalRadioDecoder:
    """
    Create and initialize a universal radio decoder.
    
    Args:
        detector_checkpoint: Path to signal detector model
        cw_checkpoint: Path to CW decoder model
        config: Decoder configuration
        
    Returns:
        Initialized UniversalRadioDecoder
    """
    decoder = UniversalRadioDecoder(config)
    decoder.load_models(
        detector_checkpoint=detector_checkpoint,
        cw_checkpoint=cw_checkpoint
    )
    return decoder

