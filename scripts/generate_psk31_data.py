"""
Generate PSK31 audio + text training data.

PSK31 (Phase Shift Keying, 31.25 baud) is a popular digital mode for
keyboard-to-keyboard communication on amateur radio.

Key characteristics:
- Varicode encoding (variable-length bit patterns per character)
- BPSK modulation (phase shifts between 0° and 180°)
- 31.25 baud rate
- ~31 Hz bandwidth (very narrow)
- Typical audio frequency: 500-2000 Hz

Usage:
    python scripts/generate_psk31_data.py --output_dir data/synthetic/psk31_v1 --num_samples 2000
"""

import argparse
import numpy as np
import random
import json
import wave
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import text generator
try:
    from generate_training_text import generate_texts_for_mode
    HAS_TEXT_GENERATOR = True
except ImportError:
    HAS_TEXT_GENERATOR = False


# PSK31 Varicode table
# Each character maps to a variable-length bit pattern
# Pattern ends with "00" (which is never in the middle of a code)
VARICODE = {
    '\x00': '1010101011',  # NUL
    '\n': '11101111',      # LF
    '\r': '11101',         # CR
    ' ': '1',              # Space (shortest - most common)
    '!': '111111111',
    '"': '101011111',
    '#': '111110101',
    '$': '111011011',
    '%': '1011010101',
    '&': '1010111011',
    "'": '101111111',
    '(': '11111011',
    ')': '11110111',
    '*': '101101111',
    '+': '111011111',
    ',': '1110101',
    '-': '110101',
    '.': '1010111',
    '/': '110101111',
    '0': '10110111',
    '1': '10111101',
    '2': '11101101',
    '3': '11111111',
    '4': '101110111',
    '5': '101011011',
    '6': '101101011',
    '7': '110101101',
    '8': '110101011',
    '9': '110110111',
    ':': '11110101',
    ';': '110111101',
    '<': '111101101',
    '=': '1010101',
    '>': '111010111',
    '?': '1010101111',
    '@': '1010111101',
    'A': '1111101',
    'B': '11101011',
    'C': '10101101',
    'D': '10110101',
    'E': '1110111',
    'F': '11011011',
    'G': '11111101',
    'H': '101010101',
    'I': '1111111',
    'J': '111111101',
    'K': '101111101',
    'L': '11010111',
    'M': '10111011',
    'N': '11011101',
    'O': '10101011',
    'P': '11010101',
    'Q': '111011101',
    'R': '10101111',
    'S': '1101111',
    'T': '1101101',
    'U': '101010111',
    'V': '110110101',
    'W': '101011101',
    'X': '101110101',
    'Y': '101111011',
    'Z': '1010101101',
    '[': '111110111',
    '\\': '111101111',
    ']': '111111011',
    '^': '1010111111',
    '_': '101101101',
    '`': '1011011111',
    'a': '1011',
    'b': '1011111',
    'c': '101111',
    'd': '101101',
    'e': '11',
    'f': '111101',
    'g': '1011011',
    'h': '101011',
    'i': '1101',
    'j': '111101011',
    'k': '10111111',
    'l': '11011',
    'm': '111011',
    'n': '1111',
    'o': '111',
    'p': '111111',
    'q': '110111111',
    'r': '10101',
    's': '10111',
    't': '101',
    'u': '110111',
    'v': '1111011',
    'w': '1101011',
    'x': '11011111',
    'y': '1011101',
    'z': '111010101',
    '{': '1010110111',
    '|': '110111011',
    '}': '1010110101',
    '~': '1011010111',
}

# Add uppercase mappings (same as lowercase for simplicity)
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    if c not in VARICODE:
        VARICODE[c] = VARICODE[c.lower()]


@dataclass
class PSK31Config:
    """Configuration for PSK31 generation."""
    baud_rate: float = 31.25        # PSK31 standard baud rate
    carrier_freq: float = 1000.0    # Audio carrier frequency (Hz)
    sample_rate: int = 16000        # Audio sample rate
    
    # Variability settings
    freq_drift: float = 0.02        # ±2% frequency drift
    snr_min: float = 5.0            # Minimum SNR in dB
    snr_max: float = 20.0           # Maximum SNR in dB
    
    # Fading simulation
    add_fading: bool = True
    fade_rate: float = 0.5          # Fades per second (QSB)
    fade_depth: float = 0.5         # How deep fades go (0-1)
    
    # Phase jitter (simulates imperfect transmitter)
    phase_jitter: float = 0.05      # Radians of phase noise


class PSK31Generator:
    """Generate PSK31 audio from text."""
    
    def __init__(self, config: PSK31Config):
        self.config = config
        self.samples_per_symbol = int(config.sample_rate / config.baud_rate)
        
    def text_to_varicode(self, text: str) -> str:
        """Convert text to Varicode bit string."""
        bits = []
        for char in text:
            if char.upper() in VARICODE:
                # Add character bits
                bits.append(VARICODE[char.upper()])
                # Add "00" separator between characters
                bits.append('00')
            # Skip unknown characters
        return ''.join(bits)
    
    def generate_bpsk_symbol(self, phase: float, duration_samples: int) -> np.ndarray:
        """Generate a single BPSK symbol with raised cosine shaping."""
        t = np.arange(duration_samples) / self.config.sample_rate
        
        # Carrier with phase
        carrier = np.cos(2 * np.pi * self.config.carrier_freq * t + phase)
        
        # Raised cosine envelope for smooth transitions
        # Shape the first and last 25% of symbol
        envelope = np.ones(duration_samples)
        ramp_len = duration_samples // 4
        
        if ramp_len > 0:
            # Raised cosine ramp up
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            envelope[:ramp_len] = ramp
            # Raised cosine ramp down
            envelope[-ramp_len:] = ramp[::-1]
        
        return carrier * envelope
    
    def generate_audio(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Generate PSK31 audio from text.
        
        Returns:
            audio: numpy array of audio samples
            duration: total duration in seconds
        """
        # Convert to Varicode
        bits = self.text_to_varicode(text)
        
        if not bits:
            return np.array([]), 0.0
        
        # Generate carrier frequency with drift
        freq_drift = 1.0 + random.uniform(-self.config.freq_drift, self.config.freq_drift)
        actual_freq = self.config.carrier_freq * freq_drift
        
        # Generate audio
        audio_parts = []
        current_phase = 0.0
        
        for bit in bits:
            # Add phase jitter
            phase_noise = random.gauss(0, self.config.phase_jitter)
            
            if bit == '0':
                # Phase reversal (180° shift)
                current_phase += np.pi
            # bit == '1' means no phase change
            
            # Generate symbol
            t = np.arange(self.samples_per_symbol) / self.config.sample_rate
            symbol = np.cos(2 * np.pi * actual_freq * t + current_phase + phase_noise)
            
            audio_parts.append(symbol)
        
        audio = np.concatenate(audio_parts)
        
        # Apply raised cosine shaping at symbol boundaries
        audio = self._apply_symbol_shaping(audio)
        
        # Apply fading if enabled
        if self.config.add_fading:
            audio = self._apply_fading(audio)
        
        # Add noise based on SNR
        snr_db = random.uniform(self.config.snr_min, self.config.snr_max)
        audio = self._add_noise(audio, snr_db)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        duration = len(audio) / self.config.sample_rate
        return audio, duration
    
    def _apply_symbol_shaping(self, audio: np.ndarray) -> np.ndarray:
        """Apply smooth transitions between symbols."""
        # Apply a gentle lowpass filter to smooth phase transitions
        # This simulates the bandwidth limiting of real PSK31
        
        # Simple moving average for smoothing
        window_size = self.samples_per_symbol // 8
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            audio = np.convolve(audio, kernel, mode='same')
        
        return audio
    
    def _apply_fading(self, audio: np.ndarray) -> np.ndarray:
        """Apply realistic fading (QSB) to simulate propagation."""
        duration = len(audio) / self.config.sample_rate
        t = np.arange(len(audio)) / self.config.sample_rate
        
        # Multiple fade components at different rates
        fade1 = np.sin(2 * np.pi * self.config.fade_rate * t + random.uniform(0, 2*np.pi))
        fade2 = np.sin(2 * np.pi * self.config.fade_rate * 0.3 * t + random.uniform(0, 2*np.pi))
        
        # Combine and scale
        fade = 0.5 * (fade1 + fade2)
        fade = 1.0 - self.config.fade_depth * (fade + 1) / 2  # Scale to fade_depth
        
        return audio * fade
    
    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white noise at specified SNR."""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
        return audio + noise
    
    def save_wav(self, audio: np.ndarray, filepath: str):
        """Save audio as WAV file."""
        audio_normalized = audio / np.max(np.abs(audio)) * 0.8
        audio_int = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_int.tobytes())


def generate_psk31_data(
    output_dir: str,
    num_samples: int = 1000,
    carrier_freq: float = 1000.0,
    snr_min: float = 5.0,
    snr_max: float = 20.0,
    add_fading: bool = True
):
    """
    Generate PSK31 training data.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        carrier_freq: Base carrier frequency
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        add_fading: Whether to add QSB fading
    """
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    text_dir = output_path / 'text'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} PSK31 samples...")
    print(f"Carrier frequency: {carrier_freq} Hz")
    print(f"SNR range: {snr_min} - {snr_max} dB")
    print(f"Output directory: {output_path}")
    
    # Get text samples
    if HAS_TEXT_GENERATOR:
        print("Using advanced text generator (callsigns, Q-codes, varied patterns)")
        text_samples = generate_texts_for_mode('PSK31', num_samples * 2)
        print(f"  Pre-generated {len(text_samples)} text samples")
    else:
        print("Warning: Text generator not found, using basic fallback")
        text_samples = None
    
    print("=" * 60)
    
    metadata = []
    actual_count = 0
    text_index = 0
    
    for i in range(num_samples * 3):  # Generate extra to account for skipped
        if actual_count >= num_samples:
            break
        
        # Create config with some randomization
        config = PSK31Config(
            carrier_freq=carrier_freq + random.uniform(-100, 100),
            snr_min=snr_min,
            snr_max=snr_max,
            add_fading=add_fading,
            fade_rate=random.uniform(0.2, 1.0),
            fade_depth=random.uniform(0.2, 0.6),
            freq_drift=random.uniform(0.01, 0.03),
            phase_jitter=random.uniform(0.02, 0.08)
        )
        
        generator = PSK31Generator(config)
        
        # Get text
        if text_samples and text_index < len(text_samples):
            text = text_samples[text_index]
            text_index += 1
        else:
            # Basic fallback
            text = "CQ CQ CQ DE W1ABC K"
        
        # Generate audio
        audio, duration = generator.generate_audio(text)
        
        # Skip if too short or too long
        if duration < 1.0 or duration > 30.0 or len(audio) < 100:
            continue
        
        # Save
        sample_id = f"psk31_{actual_count:05d}"
        
        audio_path = audio_dir / f"{sample_id}.wav"
        generator.save_wav(audio, str(audio_path))
        
        text_path = text_dir / f"{sample_id}.txt"
        with open(text_path, 'w') as f:
            f.write(text)
        
        metadata.append({
            'id': sample_id,
            'text': text,
            'duration': round(duration, 2),
            'carrier_freq': round(config.carrier_freq, 1),
            'snr_db': round(random.uniform(snr_min, snr_max), 1),
            'has_fading': add_fading,
            'text_length': len(text)
        })
        
        actual_count += 1
        
        if actual_count % 100 == 0:
            print(f"  Generated {actual_count}/{num_samples} samples...")
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Statistics
    durations = [m['duration'] for m in metadata]
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Samples created: {len(metadata)}")
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Avg duration: {sum(durations)/len(durations):.1f}s")
    print(f"  Total audio: {sum(durations)/60:.1f} minutes")
    
    print(f"\nSample entries:")
    for m in metadata[:5]:
        print(f"  {m['id']}: \"{m['text'][:40]}...\" ({m['duration']}s, {m['snr_db']}dB)")
    
    # Vocabulary analysis
    all_text = ' '.join(m['text'] for m in metadata)
    unique_chars = sorted(set(all_text))
    print(f"\nVocabulary: {''.join(unique_chars)}")
    print(f"Unique characters: {len(unique_chars)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate PSK31 training data')
    parser.add_argument('--output_dir', type=str, default='psk31_synthetic',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--carrier_freq', type=float, default=1000.0,
                       help='Base carrier frequency in Hz')
    parser.add_argument('--snr_min', type=float, default=5.0,
                       help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=20.0,
                       help='Maximum SNR in dB')
    parser.add_argument('--no_fading', action='store_true',
                       help='Disable QSB fading simulation')
    
    args = parser.parse_args()
    
    generate_psk31_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        carrier_freq=args.carrier_freq,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        add_fading=not args.no_fading
    )


if __name__ == '__main__':
    main()

