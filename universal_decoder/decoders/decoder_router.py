"""
Decoder Router

Routes detected signals to appropriate decoders based on mode.
Supports pluggable decoders for different modes.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# Add paths for our decoder modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src_ctc'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'production'))


@dataclass
class DecodedResult:
    """Result from a decoder."""
    text: str                    # Decoded text
    mode: str                    # Signal mode
    confidence: float            # Decoder confidence
    center_freq_hz: float        # Original frequency
    raw_output: Optional[any] = None  # Mode-specific raw output


class BaseDecoder(ABC):
    """Abstract base class for signal decoders."""
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the mode this decoder handles."""
        pass
    
    @abstractmethod
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        """
        Decode audio to text.
        
        Args:
            audio: Audio samples (float32, normalized)
            sample_rate: Sample rate of audio
            
        Returns:
            DecodedResult with text and metadata
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if decoder is loaded and ready."""
        pass


class CWDecoder(BaseDecoder):
    """
    Morse code (CW) decoder using our CTC model.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self._decoder = None
        self._loaded = False
    
    @property
    def mode_name(self) -> str:
        return "CW"
    
    def load(self):
        """Load the CTC model."""
        if self._loaded:
            return
        
        try:
            from src_ctc.inference import CTCDecoder
            
            if self.checkpoint_path:
                self._decoder = CTCDecoder(self.checkpoint_path)
            else:
                # Try default path
                default_path = Path(__file__).parent.parent.parent / 'checkpoints_ctc' / 'best_model_ctc.pt'
                if default_path.exists():
                    self._decoder = CTCDecoder(str(default_path))
                else:
                    print(f"CW decoder: checkpoint not found at {default_path}")
                    return
            
            self._loaded = True
            print("CW decoder loaded")
        except Exception as e:
            print(f"Failed to load CW decoder: {e}")
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            return DecodedResult(
                text="[CW DECODER NOT LOADED]",
                mode=self.mode_name,
                confidence=0.0,
                center_freq_hz=0
            )
        
        # Use our CTC decoder
        text = self._decoder.transcribe_audio(audio, sample_rate)
        
        return DecodedResult(
            text=text,
            mode=self.mode_name,
            confidence=0.9,  # TODO: Get actual confidence from model
            center_freq_hz=0
        )


class FT8Decoder(BaseDecoder):
    """
    FT8 decoder (placeholder - would use WSJT-X library).
    
    FT8 is a deterministic protocol, so we could either:
    1. Use the WSJT-X decoding library
    2. Train an ML model (overkill for FT8)
    """
    
    @property
    def mode_name(self) -> str:
        return "FT8"
    
    def is_loaded(self) -> bool:
        return True  # Placeholder
    
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        # Placeholder - would integrate with WSJT-X
        return DecodedResult(
            text="[FT8 decoding requires WSJT-X integration]",
            mode=self.mode_name,
            confidence=0.0,
            center_freq_hz=0
        )


class PSK31Decoder(BaseDecoder):
    """
    PSK31 decoder (placeholder - would train a CTC model).
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self._loaded = False
    
    @property
    def mode_name(self) -> str:
        return "PSK31"
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self):
        """Load PSK31 model (when trained)."""
        # Would load a PSK31-specific CTC model
        self._loaded = False  # Not yet trained
    
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        return DecodedResult(
            text="[PSK31 decoder not yet trained]",
            mode=self.mode_name,
            confidence=0.0,
            center_freq_hz=0
        )


class RTTYDecoder(BaseDecoder):
    """RTTY decoder (placeholder)."""
    
    @property
    def mode_name(self) -> str:
        return "RTTY"
    
    def is_loaded(self) -> bool:
        return False
    
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        return DecodedResult(
            text="[RTTY decoder not yet trained]",
            mode=self.mode_name,
            confidence=0.0,
            center_freq_hz=0
        )


class VoiceDecoder(BaseDecoder):
    """
    Voice decoder (SSB) using Whisper or similar.
    """
    
    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._model = None
        self._loaded = False
    
    @property
    def mode_name(self) -> str:
        return "SSB"
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self):
        """Load Whisper model."""
        try:
            import whisper
            self._model = whisper.load_model(self.model_name)
            self._loaded = True
            print(f"Voice decoder loaded (Whisper {self.model_name})")
        except ImportError:
            print("Whisper not installed. Install with: pip install openai-whisper")
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
    
    def decode(self, audio: np.ndarray, sample_rate: int) -> DecodedResult:
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            return DecodedResult(
                text="[VOICE DECODER NOT LOADED]",
                mode=self.mode_name,
                confidence=0.0,
                center_freq_hz=0
            )
        
        # Whisper expects 16kHz
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, num_samples)
        
        # Transcribe
        result = self._model.transcribe(audio.astype(np.float32))
        
        return DecodedResult(
            text=result['text'],
            mode=self.mode_name,
            confidence=0.9,
            center_freq_hz=0,
            raw_output=result
        )


class DecoderRouter:
    """
    Routes signals to appropriate decoders based on mode.
    
    Usage:
        router = DecoderRouter()
        router.register_decoder('CW', CWDecoder('/path/to/checkpoint.pt'))
        
        result = router.decode(audio, sample_rate, mode='CW')
    """
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialize router.
        
        Args:
            lazy_load: Load decoders only when first used
        """
        self.decoders: Dict[str, BaseDecoder] = {}
        self.lazy_load = lazy_load
    
    def register_decoder(self, mode: str, decoder: BaseDecoder):
        """
        Register a decoder for a mode.
        
        Args:
            mode: Mode name (CW, PSK31, etc.)
            decoder: Decoder instance
        """
        self.decoders[mode.upper()] = decoder
        print(f"Registered decoder for {mode}")
    
    def register_default_decoders(
        self,
        cw_checkpoint: Optional[str] = None,
        psk31_checkpoint: Optional[str] = None
    ):
        """
        Register all available decoders with default configs.
        """
        self.register_decoder('CW', CWDecoder(cw_checkpoint))
        self.register_decoder('PSK31', PSK31Decoder(psk31_checkpoint))
        self.register_decoder('RTTY', RTTYDecoder())
        self.register_decoder('FT8', FT8Decoder())
        self.register_decoder('SSB', VoiceDecoder())
    
    def get_decoder(self, mode: str) -> Optional[BaseDecoder]:
        """Get decoder for a mode."""
        return self.decoders.get(mode.upper())
    
    def list_modes(self) -> List[str]:
        """List all registered modes."""
        return list(self.decoders.keys())
    
    def decode(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str,
        center_freq_hz: float = 0
    ) -> DecodedResult:
        """
        Decode audio using the appropriate decoder.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            mode: Signal mode
            center_freq_hz: Original center frequency
            
        Returns:
            DecodedResult
        """
        mode = mode.upper()
        
        decoder = self.decoders.get(mode)
        if decoder is None:
            return DecodedResult(
                text=f"[No decoder for mode: {mode}]",
                mode=mode,
                confidence=0.0,
                center_freq_hz=center_freq_hz
            )
        
        # Lazy load if needed
        if not decoder.is_loaded() and hasattr(decoder, 'load'):
            decoder.load()
        
        # Decode
        result = decoder.decode(audio, sample_rate)
        result.center_freq_hz = center_freq_hz
        
        return result
    
    def decode_multiple(
        self,
        signals: List[Dict],
        audio_getter: Callable
    ) -> List[DecodedResult]:
        """
        Decode multiple detected signals.
        
        Args:
            signals: List of dicts with 'mode', 'center_freq_hz', etc.
            audio_getter: Function that returns (audio, sample_rate) for a signal
            
        Returns:
            List of DecodedResults
        """
        results = []
        
        for sig in signals:
            mode = sig.get('mode', 'UNKNOWN')
            center_freq = sig.get('center_freq_hz', 0)
            
            try:
                audio, sr = audio_getter(sig)
                result = self.decode(audio, sr, mode, center_freq)
            except Exception as e:
                result = DecodedResult(
                    text=f"[Decode error: {e}]",
                    mode=mode,
                    confidence=0.0,
                    center_freq_hz=center_freq
                )
            
            results.append(result)
        
        return results

