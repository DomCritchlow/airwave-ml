"""
Signal Extractor

Extracts individual signals from wideband audio by frequency.
Uses bandpass filtering and frequency mixing to isolate signals.

Key operations:
1. Bandpass filter around detected signal
2. Mix down to baseband if needed
3. Resample to standard rate for decoder
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal
from dataclasses import dataclass


@dataclass
class ExtractedSignal:
    """An extracted narrowband signal."""
    audio: np.ndarray           # Extracted audio samples
    sample_rate: int            # Sample rate
    center_freq_hz: float       # Original center frequency
    bandwidth_hz: float         # Signal bandwidth
    duration_s: float           # Duration in seconds


class SignalExtractor:
    """
    Extracts narrowband signals from wideband audio.
    
    Usage:
        extractor = SignalExtractor(sample_rate=48000)
        extracted = extractor.extract(
            audio=wideband_audio,
            center_freq=7040,
            bandwidth=500,
            freq_offset=0  # If audio is already baseband
        )
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        output_sample_rate: int = 16000
    ):
        """
        Initialize extractor.
        
        Args:
            sample_rate: Input audio sample rate
            output_sample_rate: Output sample rate for decoders
        """
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
    
    def extract(
        self,
        audio: np.ndarray,
        center_freq: float,
        bandwidth: float,
        freq_offset: float = 0,
        normalize: bool = True
    ) -> ExtractedSignal:
        """
        Extract a narrowband signal.
        
        Args:
            audio: Wideband audio samples
            center_freq: Center frequency of desired signal (Hz)
            bandwidth: Signal bandwidth (Hz)
            freq_offset: Frequency offset if audio is mixed (Hz)
                        (e.g., if 7040 kHz is at 1000 Hz in audio, offset=7039000)
            normalize: Normalize output amplitude
            
        Returns:
            ExtractedSignal with isolated audio
        """
        # Calculate actual frequency in audio
        actual_center = center_freq - freq_offset
        
        # Bandpass filter
        filtered = self.bandpass_filter(
            audio,
            center_freq=actual_center,
            bandwidth=bandwidth
        )
        
        # Mix down to baseband (center at 0 Hz)
        baseband = self.mix_to_baseband(filtered, actual_center)
        
        # Low-pass filter at bandwidth/2
        baseband = self.lowpass_filter(baseband, bandwidth / 2)
        
        # Resample to output rate
        resampled = self.resample(baseband, self.output_sample_rate)
        
        # Normalize
        if normalize and np.max(np.abs(resampled)) > 0:
            resampled = resampled / np.max(np.abs(resampled))
        
        return ExtractedSignal(
            audio=resampled.astype(np.float32),
            sample_rate=self.output_sample_rate,
            center_freq_hz=center_freq,
            bandwidth_hz=bandwidth,
            duration_s=len(resampled) / self.output_sample_rate
        )
    
    def bandpass_filter(
        self,
        audio: np.ndarray,
        center_freq: float,
        bandwidth: float,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply bandpass filter around center frequency.
        
        Args:
            audio: Input samples
            center_freq: Center frequency (Hz)
            bandwidth: Filter bandwidth (Hz)
            order: Filter order
            
        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        
        low = (center_freq - bandwidth / 2) / nyquist
        high = (center_freq + bandwidth / 2) / nyquist
        
        # Clamp to valid range
        low = max(0.01, min(0.99, low))
        high = max(0.01, min(0.99, high))
        
        if low >= high:
            return audio
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        return signal.filtfilt(b, a, audio)
    
    def lowpass_filter(
        self,
        audio: np.ndarray,
        cutoff: float,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply low-pass filter.
        
        Args:
            audio: Input samples
            cutoff: Cutoff frequency (Hz)
            order: Filter order
            
        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = min(0.99, cutoff / nyquist)
        
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        return signal.filtfilt(b, a, audio)
    
    def mix_to_baseband(
        self,
        audio: np.ndarray,
        freq: float
    ) -> np.ndarray:
        """
        Mix signal down to baseband (0 Hz center).
        
        This shifts the signal's center frequency to 0 Hz,
        converting it to a complex baseband representation.
        
        Args:
            audio: Input audio (can be real or complex)
            freq: Frequency to mix down (Hz)
            
        Returns:
            Complex baseband signal (we take real part)
        """
        t = np.arange(len(audio)) / self.sample_rate
        
        # Complex exponential for mixing
        mixer = np.exp(-2j * np.pi * freq * t)
        
        # Mix down
        mixed = audio * mixer
        
        # Return real part (assuming real input signal)
        return np.real(mixed)
    
    def resample(
        self,
        audio: np.ndarray,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input samples
            target_rate: Target sample rate
            
        Returns:
            Resampled audio
        """
        if target_rate == self.sample_rate:
            return audio
        
        num_samples = int(len(audio) * target_rate / self.sample_rate)
        return signal.resample(audio, num_samples)
    
    def extract_time_slice(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """
        Extract a time slice from audio.
        
        Args:
            audio: Input samples
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio slice
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        return audio[start_sample:end_sample]


class SpectrogramGenerator:
    """
    Generates spectrograms for signal detection.
    
    Creates waterfall-style spectrograms suitable for the detector CNN.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 2048,
        hop_size: int = 512,
        freq_min: float = 0,
        freq_max: float = None
    ):
        """
        Initialize spectrogram generator.
        
        Args:
            sample_rate: Audio sample rate
            fft_size: FFT window size
            hop_size: Hop between windows
            freq_min: Minimum frequency to include
            freq_max: Maximum frequency (default: Nyquist)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.freq_min = freq_min
        self.freq_max = freq_max or sample_rate / 2
    
    def generate(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spectrogram from audio.
        
        Args:
            audio: Audio samples
            log_scale: Apply log scaling to magnitude
            normalize: Normalize to 0-1 range
            
        Returns:
            Tuple of:
                - spectrogram: (freq_bins, time_steps)
                - frequencies: Array of frequency values
                - times: Array of time values
        """
        # Compute STFT
        freqs, times, Sxx = signal.spectrogram(
            audio,
            fs=self.sample_rate,
            nperseg=self.fft_size,
            noverlap=self.fft_size - self.hop_size,
            mode='magnitude'
        )
        
        # Filter frequency range
        freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        Sxx = Sxx[freq_mask, :]
        freqs = freqs[freq_mask]
        
        # Log scale
        if log_scale:
            Sxx = np.log10(Sxx + 1e-10)
        
        # Normalize
        if normalize:
            Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-10)
        
        return Sxx, freqs, times
    
    def generate_for_detector(
        self,
        audio: np.ndarray,
        target_height: int = 256,
        target_width: int = 512
    ) -> np.ndarray:
        """
        Generate spectrogram resized for detector CNN.
        
        Args:
            audio: Audio samples
            target_height: Target frequency bins
            target_width: Target time steps
            
        Returns:
            Spectrogram of shape (1, target_height, target_width)
        """
        spectrogram, _, _ = self.generate(audio)
        
        # Resize to target dimensions
        from scipy.ndimage import zoom
        
        h, w = spectrogram.shape
        zoom_h = target_height / h
        zoom_w = target_width / w
        
        resized = zoom(spectrogram, (zoom_h, zoom_w), order=1)
        
        # Add channel dimension
        return resized[np.newaxis, :, :]


def get_typical_bandwidth(mode: str) -> float:
    """
    Get typical bandwidth for a signal mode.
    
    Args:
        mode: Signal mode name
        
    Returns:
        Bandwidth in Hz
    """
    bandwidths = {
        'CW': 200,      # Morse: ~100-300 Hz
        'PSK31': 100,   # PSK31: ~31 Hz but use margin
        'RTTY': 500,    # RTTY: ~300-500 Hz
        'FT8': 100,     # FT8: ~50 Hz
        'SSB': 3000,    # Voice: ~2400-3000 Hz
        'AM': 10000,    # AM broadcast
        'FM': 15000,    # FM broadcast (narrowband)
    }
    return bandwidths.get(mode.upper(), 500)

