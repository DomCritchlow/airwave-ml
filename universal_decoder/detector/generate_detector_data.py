#!/usr/bin/env python3
"""
Synthetic Data Generator for Signal Detector Training

Generates spectrograms with labeled signals for training the CNN detector.

Each sample contains:
- Spectrogram image with 0-N signals
- Ground truth: signal locations, modes, bounding boxes

Signal types simulated:
- CW (Morse): Narrow, on-off keyed tones
- PSK31: Narrow warbling phase-shift
- RTTY: Two-tone frequency shift
- FT8: Wide structured digital
- SSB: Wide voice-like spectrum
- Noise: Random static
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import signal as scipy_signal
import random


class SignalType(Enum):
    NOISE = 0
    CW = 1
    PSK31 = 2
    RTTY = 3
    FT8 = 4
    SSB = 5


@dataclass
class GeneratedSignal:
    """A generated signal for training."""
    signal_type: SignalType
    center_freq: float      # Hz
    bandwidth: float        # Hz
    start_time: float       # Seconds
    duration: float         # Seconds
    snr_db: float           # Signal-to-noise ratio


def generate_cw_signal(
    duration: float,
    sample_rate: int,
    tone_freq: float = 700,
    wpm: int = 20
) -> np.ndarray:
    """
    Generate Morse code-like CW signal.
    Random on-off keying at specified WPM.
    """
    samples = int(duration * sample_rate)
    
    # Element timing
    dot_duration = 1.2 / wpm  # Standard PARIS timing
    
    # Generate random dit-dah pattern
    pattern = []
    t = 0
    while t < duration:
        is_on = random.random() > 0.4  # 60% duty cycle average
        element_len = dot_duration * random.choice([1, 3])  # Dit or dah
        
        if is_on:
            pattern.append((t, t + element_len))
        
        t += element_len + dot_duration * random.choice([1, 3, 7])  # Gap
    
    # Generate audio
    t_array = np.linspace(0, duration, samples)
    audio = np.zeros(samples)
    
    for start, end in pattern:
        if end > duration:
            break
        mask = (t_array >= start) & (t_array <= end)
        audio[mask] = np.sin(2 * np.pi * tone_freq * t_array[mask])
    
    return audio


def generate_psk31_signal(
    duration: float,
    sample_rate: int,
    center_freq: float = 1000
) -> np.ndarray:
    """
    Generate PSK31-like signal.
    Phase-shifting warble pattern.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # PSK31 symbol rate is ~31.25 baud
    symbol_rate = 31.25
    num_symbols = int(duration * symbol_rate)
    
    # Random phase changes
    phases = np.random.choice([0, np.pi], num_symbols)
    
    # Expand to full sample rate
    samples_per_symbol = int(sample_rate / symbol_rate)
    phase_signal = np.repeat(phases, samples_per_symbol)[:samples]
    
    # Pad if needed
    if len(phase_signal) < samples:
        phase_signal = np.pad(phase_signal, (0, samples - len(phase_signal)))
    
    # Smooth phase transitions
    from scipy.ndimage import gaussian_filter1d
    phase_signal = gaussian_filter1d(phase_signal, sigma=samples_per_symbol/4)
    
    # Generate signal
    audio = np.sin(2 * np.pi * center_freq * t + phase_signal)
    
    return audio


def generate_rtty_signal(
    duration: float,
    sample_rate: int,
    mark_freq: float = 2125,
    shift: float = 170
) -> np.ndarray:
    """
    Generate RTTY-like signal.
    Two-tone frequency shift keying.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # RTTY is typically 45.45 baud
    baud_rate = 45.45
    num_bits = int(duration * baud_rate)
    
    # Random bits
    bits = np.random.randint(0, 2, num_bits)
    
    # Expand to samples
    samples_per_bit = int(sample_rate / baud_rate)
    bit_signal = np.repeat(bits, samples_per_bit)[:samples]
    
    if len(bit_signal) < samples:
        bit_signal = np.pad(bit_signal, (0, samples - len(bit_signal)))
    
    # Frequency based on bit value
    freq = mark_freq - shift/2 + bit_signal * shift
    
    # Generate FSK
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    audio = np.sin(phase)
    
    return audio


def generate_ft8_signal(
    duration: float,
    sample_rate: int,
    base_freq: float = 1000
) -> np.ndarray:
    """
    Generate FT8-like signal.
    8-FSK with characteristic tone structure.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # FT8: 79 symbols, 12.64 second cycle, 6.25 baud
    symbol_rate = 6.25
    num_symbols = min(79, int(duration * symbol_rate))
    
    # 8 tones, 6.25 Hz apart
    tones = np.random.randint(0, 8, num_symbols)
    
    # Expand
    samples_per_symbol = int(sample_rate / symbol_rate)
    tone_signal = np.repeat(tones, samples_per_symbol)[:samples]
    
    if len(tone_signal) < samples:
        tone_signal = np.pad(tone_signal, (0, samples - len(tone_signal)))
    
    freq = base_freq + tone_signal * 6.25
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    audio = np.sin(phase)
    
    return audio


def generate_ssb_signal(
    duration: float,
    sample_rate: int,
    center_freq: float = 1500
) -> np.ndarray:
    """
    Generate SSB voice-like signal.
    Filtered noise with speech-like characteristics.
    """
    samples = int(duration * sample_rate)
    
    # Generate noise
    audio = np.random.randn(samples)
    
    # Apply speech-like envelope (varies in amplitude)
    t = np.linspace(0, duration, samples)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    envelope *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    
    audio = audio * envelope
    
    # Bandpass filter to voice range (300-3000 Hz offset from center)
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = min(0.99, 3000 / nyquist)
    
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    audio = scipy_signal.filtfilt(b, a, audio)
    
    # Shift to center frequency
    t = np.arange(samples) / sample_rate
    audio = audio * np.sin(2 * np.pi * center_freq * t)
    
    return audio


def generate_spectrogram_sample(
    duration: float = 10.0,
    sample_rate: int = 48000,
    fft_size: int = 2048,
    hop_size: int = 512,
    num_signals: int = None,
    signal_types: List[SignalType] = None,
    freq_range: Tuple[float, float] = (0, 24000),
    noise_floor_db: float = -40
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate a spectrogram with random signals for training.
    
    Returns:
        Tuple of:
            - Spectrogram (normalized 0-1)
            - List of signal annotations
    """
    samples = int(duration * sample_rate)
    
    # Start with noise floor
    audio = 10 ** (noise_floor_db / 20) * np.random.randn(samples)
    
    # Random number of signals if not specified
    if num_signals is None:
        num_signals = random.randint(0, 5)
    
    # Available signal types
    if signal_types is None:
        signal_types = [SignalType.CW, SignalType.PSK31, SignalType.RTTY, 
                       SignalType.FT8, SignalType.SSB]
    
    annotations = []
    
    for _ in range(num_signals):
        # Random signal type
        sig_type = random.choice(signal_types)
        
        # Random parameters
        if sig_type == SignalType.CW:
            bandwidth = random.uniform(100, 300)
            generator = lambda d, sr, cf: generate_cw_signal(d, sr, cf)
        elif sig_type == SignalType.PSK31:
            bandwidth = random.uniform(50, 150)
            generator = lambda d, sr, cf: generate_psk31_signal(d, sr, cf)
        elif sig_type == SignalType.RTTY:
            bandwidth = random.uniform(200, 500)
            generator = lambda d, sr, cf: generate_rtty_signal(d, sr, cf)
        elif sig_type == SignalType.FT8:
            bandwidth = random.uniform(50, 100)
            generator = lambda d, sr, cf: generate_ft8_signal(d, sr, cf)
        else:  # SSB
            bandwidth = random.uniform(2000, 3500)
            generator = lambda d, sr, cf: generate_ssb_signal(d, sr, cf)
        
        # Random frequency (avoiding edges)
        center_freq = random.uniform(
            freq_range[0] + bandwidth,
            freq_range[1] - bandwidth
        )
        
        # Random timing
        start_time = random.uniform(0, duration * 0.8)
        sig_duration = random.uniform(1.0, duration - start_time)
        
        # Random SNR
        snr_db = random.uniform(5, 25)
        
        # Generate signal
        sig_samples = int(sig_duration * sample_rate)
        sig_audio = generator(sig_duration, sample_rate, center_freq)
        
        # Scale for SNR
        signal_power = np.mean(sig_audio ** 2)
        noise_power = 10 ** (noise_floor_db / 10)
        target_power = noise_power * 10 ** (snr_db / 10)
        
        if signal_power > 0:
            scale = np.sqrt(target_power / signal_power)
            sig_audio = sig_audio * scale
        
        # Add to main audio
        start_sample = int(start_time * sample_rate)
        end_sample = min(start_sample + len(sig_audio), samples)
        audio[start_sample:end_sample] += sig_audio[:end_sample - start_sample]
        
        # Record annotation
        annotations.append({
            'signal_type': sig_type.value,
            'signal_name': sig_type.name,
            'center_freq': center_freq,
            'bandwidth': bandwidth,
            'start_time': start_time,
            'duration': sig_duration,
            'snr_db': snr_db
        })
    
    # Generate spectrogram
    freqs, times, Sxx = scipy_signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=fft_size,
        noverlap=fft_size - hop_size,
        mode='magnitude'
    )
    
    # Log scale and normalize
    Sxx = np.log10(Sxx + 1e-10)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-10)
    
    return Sxx, annotations


def generate_training_dataset(
    output_dir: str,
    num_samples: int = 1000,
    duration: float = 10.0,
    sample_rate: int = 48000,
    target_height: int = 256,
    target_width: int = 512
):
    """
    Generate a complete training dataset for the signal detector.
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Number of samples to generate
        duration: Audio duration per sample
        sample_rate: Audio sample rate
        target_height: Spectrogram height (freq bins)
        target_width: Spectrogram width (time steps)
    """
    from scipy.ndimage import zoom
    
    output_path = Path(output_dir)
    (output_path / 'spectrograms').mkdir(parents=True, exist_ok=True)
    (output_path / 'annotations').mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} training samples...")
    
    all_annotations = []
    
    for i in range(num_samples):
        # Generate sample
        spectrogram, annotations = generate_spectrogram_sample(
            duration=duration,
            sample_rate=sample_rate
        )
        
        # Resize to target dimensions
        h, w = spectrogram.shape
        zoom_h = target_height / h
        zoom_w = target_width / w
        spectrogram = zoom(spectrogram, (zoom_h, zoom_w), order=1)
        
        # Save spectrogram
        np.save(
            output_path / 'spectrograms' / f'spec_{i:05d}.npy',
            spectrogram.astype(np.float32)
        )
        
        # Save annotation
        annotation = {
            'id': i,
            'signals': annotations,
            'freq_range': [0, sample_rate / 2],
            'time_range': [0, duration]
        }
        
        with open(output_path / 'annotations' / f'ann_{i:05d}.json', 'w') as f:
            json.dump(annotation, f, indent=2)
        
        all_annotations.append(annotation)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'duration': duration,
        'sample_rate': sample_rate,
        'spectrogram_shape': [target_height, target_width],
        'signal_types': [s.name for s in SignalType],
        'annotations': all_annotations
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"  Spectrograms: {output_path}/spectrograms/")
    print(f"  Annotations: {output_path}/annotations/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate detector training data')
    parser.add_argument('--output', '-o', type=str, default='detector_data',
                       help='Output directory')
    parser.add_argument('--num-samples', '-n', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                       help='Audio duration per sample (seconds)')
    
    args = parser.parse_args()
    
    generate_training_dataset(
        output_dir=args.output,
        num_samples=args.num_samples,
        duration=args.duration
    )

