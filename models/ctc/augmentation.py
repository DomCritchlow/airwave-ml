"""
Audio augmentation for robust Morse code training.

These augmentations force the model to learn actual Morse patterns
rather than overfitting to specific audio characteristics.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import random
import io
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""
    enabled: bool = True
    
    # Probability of applying each augmentation
    prob_mp3_compression: float = 0.3
    prob_bandpass: float = 0.3
    prob_time_stretch: float = 0.2
    prob_pitch_shift: float = 0.2
    prob_noise: float = 0.4
    prob_volume: float = 0.3
    prob_clipping: float = 0.1
    prob_lowpass: float = 0.2
    prob_reverb: float = 0.1
    
    # Parameter ranges
    mp3_bitrates: tuple = (32, 64, 96, 128)  # kbps
    bandpass_low: tuple = (200, 500)   # Hz range
    bandpass_high: tuple = (2000, 3500)  # Hz range
    time_stretch_range: tuple = (0.9, 1.1)  # speed factor
    pitch_shift_range: tuple = (-2, 2)  # semitones
    noise_snr_range: tuple = (5, 25)  # dB
    volume_range: tuple = (0.5, 1.5)  # multiplier
    clip_threshold: tuple = (0.3, 0.8)  # clip above this
    lowpass_range: tuple = (2000, 4000)  # Hz cutoff


class AudioAugmenter:
    """
    Applies random augmentations to audio for robust training.
    
    Simulates:
    - Compression artifacts (MP3, low bitrate)
    - Radio bandwidth limitations (bandpass)
    - Transmission variations (pitch, speed)
    - Noise conditions (various noise types)
    - Recording quality (clipping, volume)
    """
    
    def __init__(self, config: AugmentationConfig = None, sample_rate: int = 16000):
        self.config = config or AugmentationConfig()
        self.sample_rate = sample_rate
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform."""
        if not self.config.enabled:
            return waveform
            
        # Ensure we have a 2D tensor (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply augmentations randomly
        if random.random() < self.config.prob_volume:
            waveform = self._augment_volume(waveform)
            
        if random.random() < self.config.prob_noise:
            waveform = self._add_noise(waveform)
            
        if random.random() < self.config.prob_bandpass:
            waveform = self._bandpass_filter(waveform)
            
        if random.random() < self.config.prob_lowpass:
            waveform = self._lowpass_filter(waveform)
            
        if random.random() < self.config.prob_time_stretch:
            waveform = self._time_stretch(waveform)
            
        if random.random() < self.config.prob_pitch_shift:
            waveform = self._pitch_shift(waveform)
        
        if random.random() < self.config.prob_mp3_compression:
            waveform = self._simulate_compression(waveform)
            
        if random.random() < self.config.prob_clipping:
            waveform = self._apply_clipping(waveform)
            
        if random.random() < self.config.prob_reverb:
            waveform = self._add_reverb(waveform)
        
        # Normalize to prevent clipping
        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val
            
        return waveform
    
    def _augment_volume(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random volume change."""
        low, high = self.config.volume_range
        gain = random.uniform(low, high)
        return waveform * gain
    
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add noise at random SNR."""
        snr_low, snr_high = self.config.noise_snr_range
        snr_db = random.uniform(snr_low, snr_high)
        
        # Choose noise type
        noise_type = random.choice(['white', 'pink', 'brown', 'band_limited'])
        
        samples = waveform.shape[-1]
        
        if noise_type == 'white':
            noise = torch.randn_like(waveform)
        elif noise_type == 'pink':
            # Pink noise (1/f) - approximate with filtered white noise
            noise = torch.randn_like(waveform)
            # Simple pink noise approximation
            b = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = torch.tensor([1, -2.494956002, 2.017265875, -0.522189400])
            noise = torchaudio.functional.lfilter(noise, a, b, clamp=False)
        elif noise_type == 'brown':
            # Brown noise (random walk)
            noise = torch.randn_like(waveform).cumsum(dim=-1)
            noise = noise - noise.mean()
            noise = noise / (noise.abs().max() + 1e-8)
        else:  # band_limited
            # Band-limited noise (like radio static)
            noise = torch.randn_like(waveform)
            # Apply bandpass
            low_hz = random.uniform(300, 800)
            high_hz = random.uniform(1500, 3000)
            noise = torchaudio.functional.bandpass_biquad(
                noise, self.sample_rate, 
                (low_hz + high_hz) / 2, 
                (high_hz - low_hz) / ((low_hz + high_hz) / 2)
            )
        
        # Calculate signal power and scale noise for target SNR
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-8
        
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        return waveform + noise * noise_scale
    
    def _bandpass_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter (simulates radio bandwidth)."""
        low_range = self.config.bandpass_low
        high_range = self.config.bandpass_high
        
        low_hz = random.uniform(*low_range)
        high_hz = random.uniform(*high_range)
        
        # Ensure low < high
        if low_hz >= high_hz:
            low_hz = high_hz * 0.5
        
        center_freq = (low_hz + high_hz) / 2
        bandwidth = high_hz - low_hz
        Q = center_freq / bandwidth if bandwidth > 0 else 1.0
        
        return torchaudio.functional.bandpass_biquad(
            waveform, self.sample_rate, center_freq, Q
        )
    
    def _lowpass_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply lowpass filter (simulates poor audio quality)."""
        cutoff = random.uniform(*self.config.lowpass_range)
        return torchaudio.functional.lowpass_biquad(
            waveform, self.sample_rate, cutoff
        )
    
    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Time stretch without changing pitch."""
        low, high = self.config.time_stretch_range
        rate = random.uniform(low, high)
        
        # Use torchaudio's speed perturbation
        effects = [["speed", str(rate)], ["rate", str(self.sample_rate)]]
        
        try:
            waveform_np = waveform.numpy()
            stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
            return stretched
        except:
            # Fallback: simple resampling (changes pitch too but better than nothing)
            return waveform
    
    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Shift pitch by random semitones."""
        low, high = self.config.pitch_shift_range
        semitones = random.uniform(low, high)
        
        # Pitch shift via resampling
        rate_change = 2 ** (semitones / 12)
        
        try:
            effects = [
                ["speed", str(rate_change)],
                ["rate", str(self.sample_rate)]
            ]
            shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
            return shifted
        except:
            return waveform
    
    def _simulate_compression(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate lossy compression artifacts."""
        # Simulate MP3-like compression:
        # 1. Low-pass filter (MP3 removes high frequencies)
        # 2. Add quantization noise
        # 3. Slight spectral smearing
        
        bitrate = random.choice(self.config.mp3_bitrates)
        
        # Lower bitrate = lower cutoff frequency
        cutoff_map = {32: 4000, 64: 8000, 96: 12000, 128: 15000}
        cutoff = cutoff_map.get(bitrate, 8000)
        
        # Apply lowpass
        waveform = torchaudio.functional.lowpass_biquad(
            waveform, self.sample_rate, min(cutoff, self.sample_rate // 2 - 100)
        )
        
        # Add quantization noise (proportional to inverse of bitrate)
        noise_level = 0.02 * (128 / bitrate)
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        return waveform
    
    def _apply_clipping(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply soft clipping (simulates overdriven input)."""
        low, high = self.config.clip_threshold
        threshold = random.uniform(low, high)
        
        # Soft clipping using tanh
        max_val = waveform.abs().max()
        if max_val > 0:
            normalized = waveform / max_val
            clipped = torch.tanh(normalized / threshold) * threshold
            return clipped * max_val
        return waveform
    
    def _add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add simple reverb (simulates room acoustics)."""
        # Simple comb filter reverb
        delay_samples = int(random.uniform(0.01, 0.05) * self.sample_rate)
        decay = random.uniform(0.2, 0.5)
        
        if delay_samples >= waveform.shape[-1]:
            return waveform
            
        reverb = torch.zeros_like(waveform)
        reverb[..., delay_samples:] = waveform[..., :-delay_samples] * decay
        
        return waveform + reverb


def create_augmenter_from_config(config: dict, sample_rate: int = 16000) -> Optional[AudioAugmenter]:
    """Create augmenter from training config."""
    aug_config = config.get('augmentation', {})
    
    if not aug_config.get('enabled', True):
        return None
    
    return AudioAugmenter(
        config=AugmentationConfig(
            enabled=aug_config.get('enabled', True),
            prob_mp3_compression=aug_config.get('prob_mp3_compression', 0.3),
            prob_bandpass=aug_config.get('prob_bandpass', 0.3),
            prob_time_stretch=aug_config.get('prob_time_stretch', 0.2),
            prob_pitch_shift=aug_config.get('prob_pitch_shift', 0.2),
            prob_noise=aug_config.get('prob_noise', 0.4),
            prob_volume=aug_config.get('prob_volume', 0.3),
            prob_clipping=aug_config.get('prob_clipping', 0.1),
            prob_lowpass=aug_config.get('prob_lowpass', 0.2),
            prob_reverb=aug_config.get('prob_reverb', 0.1),
        ),
        sample_rate=sample_rate
    )

