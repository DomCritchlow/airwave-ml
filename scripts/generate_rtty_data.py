"""
Generate RTTY (Radio Teletype) audio + text training data.

RTTY is a digital mode using Frequency Shift Keying (FSK) to transmit
Baudot-encoded text. It's one of the oldest digital modes, still widely
used in amateur radio contesting.

Key characteristics:
- Baudot encoding (5-bit code with LETTERS/FIGURES shift)
- FSK modulation (two tones: mark and space)
- Common baud rates: 45.45 (standard), 50, 75, 100
- Standard shift: 170 Hz between mark and space
- Typical mark frequency: 2125 Hz, space: 2295 Hz

Usage:
    python scripts/generate_rtty_data.py --output_dir data/synthetic/rtty_v1 --num_samples 2000
"""

import argparse
import numpy as np
import random
import json
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Import text generator
try:
    from generate_training_text import generate_texts_for_mode
    HAS_TEXT_GENERATOR = True
except ImportError:
    HAS_TEXT_GENERATOR = False


class BaudotShift(Enum):
    """Baudot shift state."""
    LETTERS = 0
    FIGURES = 1


# Baudot code tables (ITA2 standard)
# 5-bit codes mapped to characters
BAUDOT_LETTERS = {
    0b00000: '\x00',  # NULL
    0b00100: ' ',     # Space
    0b00010: '\n',    # LF
    0b01000: '\r',    # CR
    0b11111: 'LTRS',  # Letters shift
    0b11011: 'FIGS',  # Figures shift
    0b00011: 'A',
    0b11001: 'B',
    0b01110: 'C',
    0b01001: 'D',
    0b00001: 'E',
    0b01101: 'F',
    0b11010: 'G',
    0b10100: 'H',
    0b00110: 'I',
    0b01011: 'J',
    0b01111: 'K',
    0b10010: 'L',
    0b11100: 'M',
    0b01100: 'N',
    0b11000: 'O',
    0b10110: 'P',
    0b10111: 'Q',
    0b01010: 'R',
    0b00101: 'S',
    0b10000: 'T',
    0b00111: 'U',
    0b11110: 'V',
    0b10011: 'W',
    0b11101: 'X',
    0b10101: 'Y',
    0b10001: 'Z',
}

BAUDOT_FIGURES = {
    0b00000: '\x00',  # NULL
    0b00100: ' ',     # Space
    0b00010: '\n',    # LF
    0b01000: '\r',    # CR
    0b11111: 'LTRS',  # Letters shift
    0b11011: 'FIGS',  # Figures shift
    0b00011: '-',
    0b11001: '?',
    0b01110: ':',
    0b01001: '$',
    0b00001: '3',
    0b01101: '!',
    0b11010: '&',
    0b10100: '#',
    0b00110: '8',
    0b01011: "'",
    0b01111: '(',
    0b10010: ')',
    0b11100: '.',
    0b01100: ',',
    0b11000: '9',
    0b10110: '0',
    0b10111: '1',
    0b01010: '4',
    0b00101: "'",     # Bell in original, using apostrophe
    0b10000: '5',
    0b00111: '7',
    0b11110: ';',
    0b10011: '2',
    0b11101: '/',
    0b10101: '6',
    0b10001: '+',
}

# Reverse lookup: character to Baudot code
CHAR_TO_BAUDOT = {}
for code, char in BAUDOT_LETTERS.items():
    if char not in ['LTRS', 'FIGS', '\x00']:
        CHAR_TO_BAUDOT[char] = (code, BaudotShift.LETTERS)

for code, char in BAUDOT_FIGURES.items():
    if char not in ['LTRS', 'FIGS', '\x00', ' ', '\n', '\r']:
        CHAR_TO_BAUDOT[char] = (code, BaudotShift.FIGURES)

# Shift codes
LTRS_CODE = 0b11111
FIGS_CODE = 0b11011


@dataclass
class RTTYConfig:
    """Configuration for RTTY generation."""
    baud_rate: float = 45.45       # Standard RTTY baud rate
    mark_freq: float = 2125.0      # Mark tone frequency (Hz)
    space_freq: float = 2295.0     # Space tone frequency (Hz) 
    sample_rate: int = 16000       # Audio sample rate
    
    # Standard shifts
    shift: float = 170.0           # Frequency shift (Hz)
    
    # Start and stop bits
    start_bits: int = 1            # Number of start bits (space)
    stop_bits: float = 1.5         # Number of stop bits (mark)
    
    # Variability settings
    freq_drift: float = 0.02       # ±2% frequency drift
    snr_min: float = 5.0           # Minimum SNR in dB
    snr_max: float = 25.0          # Maximum SNR in dB
    
    # Fading simulation
    add_fading: bool = True
    fade_rate: float = 0.3         # Fades per second
    fade_depth: float = 0.4        # How deep fades go (0-1)
    
    # Timing jitter
    timing_jitter: float = 0.05    # ±5% timing variation


class RTTYGenerator:
    """Generate RTTY audio from text."""
    
    def __init__(self, config: RTTYConfig):
        self.config = config
        self.samples_per_bit = int(config.sample_rate / config.baud_rate)
        self.current_shift = BaudotShift.LETTERS
        
    def text_to_baudot(self, text: str) -> List[Tuple[int, int]]:
        """
        Convert text to Baudot codes.
        
        Returns list of (code, num_bits) tuples.
        Each character is: start_bit + 5 data bits + stop_bits
        """
        codes = []
        current_shift = BaudotShift.LETTERS
        
        for char in text.upper():
            if char in CHAR_TO_BAUDOT:
                code, required_shift = CHAR_TO_BAUDOT[char]
                
                # Add shift if needed
                if required_shift != current_shift:
                    if required_shift == BaudotShift.LETTERS:
                        codes.append((LTRS_CODE, 5))
                    else:
                        codes.append((FIGS_CODE, 5))
                    current_shift = required_shift
                
                codes.append((code, 5))
            elif char == ' ':
                # Space is same in both shifts
                codes.append((0b00100, 5))
        
        return codes
    
    def generate_tone(self, frequency: float, duration_samples: int) -> np.ndarray:
        """Generate a pure tone."""
        t = np.arange(duration_samples) / self.config.sample_rate
        return np.sin(2 * np.pi * frequency * t)
    
    def generate_bit(self, bit: int, jitter: float = 0.0) -> np.ndarray:
        """
        Generate audio for a single bit.
        
        bit=1 (mark): higher frequency
        bit=0 (space): lower frequency
        """
        # Apply timing jitter
        jitter_factor = 1.0 + random.uniform(-jitter, jitter)
        duration_samples = int(self.samples_per_bit * jitter_factor)
        
        if bit == 1:
            freq = self.config.mark_freq
        else:
            freq = self.config.space_freq
        
        return self.generate_tone(freq, duration_samples)
    
    def generate_character(self, code: int, num_bits: int = 5) -> np.ndarray:
        """Generate audio for a Baudot character with start/stop bits."""
        audio_parts = []
        jitter = self.config.timing_jitter
        
        # Start bit (space = 0)
        for _ in range(self.config.start_bits):
            audio_parts.append(self.generate_bit(0, jitter))
        
        # Data bits (LSB first)
        for i in range(num_bits):
            bit = (code >> i) & 1
            audio_parts.append(self.generate_bit(bit, jitter))
        
        # Stop bits (mark = 1)
        # Handle fractional stop bits (1.5 is common)
        full_stops = int(self.config.stop_bits)
        partial_stop = self.config.stop_bits - full_stops
        
        for _ in range(full_stops):
            audio_parts.append(self.generate_bit(1, jitter))
        
        if partial_stop > 0:
            partial_samples = int(self.samples_per_bit * partial_stop)
            audio_parts.append(self.generate_tone(self.config.mark_freq, partial_samples))
        
        return np.concatenate(audio_parts)
    
    def generate_audio(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Generate RTTY audio from text.
        
        Returns:
            audio: numpy array of audio samples
            duration: total duration in seconds
        """
        # Convert to Baudot
        codes = self.text_to_baudot(text)
        
        if not codes:
            return np.array([]), 0.0
        
        # Apply frequency drift
        drift = random.uniform(-self.config.freq_drift, self.config.freq_drift)
        original_mark = self.config.mark_freq
        original_space = self.config.space_freq
        self.config.mark_freq *= (1 + drift)
        self.config.space_freq *= (1 + drift)
        
        # Generate audio for each character
        audio_parts = []
        
        # Add some idle (mark) at the start
        idle_samples = int(self.samples_per_bit * 5)
        audio_parts.append(self.generate_tone(self.config.mark_freq, idle_samples))
        
        for code, num_bits in codes:
            char_audio = self.generate_character(code, num_bits)
            audio_parts.append(char_audio)
        
        # Add idle at the end
        audio_parts.append(self.generate_tone(self.config.mark_freq, idle_samples))
        
        # Restore original frequencies
        self.config.mark_freq = original_mark
        self.config.space_freq = original_space
        
        audio = np.concatenate(audio_parts)
        
        # Apply fading if enabled
        if self.config.add_fading:
            audio = self._apply_fading(audio)
        
        # Add noise
        snr_db = random.uniform(self.config.snr_min, self.config.snr_max)
        audio = self._add_noise(audio, snr_db)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        duration = len(audio) / self.config.sample_rate
        return audio, duration
    
    def _apply_fading(self, audio: np.ndarray) -> np.ndarray:
        """Apply realistic fading (QSB)."""
        t = np.arange(len(audio)) / self.config.sample_rate
        
        # Multiple fade components
        fade1 = np.sin(2 * np.pi * self.config.fade_rate * t + random.uniform(0, 2*np.pi))
        fade2 = np.sin(2 * np.pi * self.config.fade_rate * 0.3 * t + random.uniform(0, 2*np.pi))
        
        fade = 0.5 * (fade1 + fade2)
        fade = 1.0 - self.config.fade_depth * (fade + 1) / 2
        
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


def generate_rtty_data(
    output_dir: str,
    num_samples: int = 1000,
    baud_rate: float = 45.45,
    snr_min: float = 5.0,
    snr_max: float = 25.0,
    add_fading: bool = True
):
    """
    Generate RTTY training data.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        baud_rate: Baud rate (45.45 standard)
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        add_fading: Whether to add fading
    """
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    text_dir = output_path / 'text'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} RTTY samples...")
    print(f"Baud rate: {baud_rate}")
    print(f"SNR range: {snr_min} - {snr_max} dB")
    print(f"Output directory: {output_path}")
    
    # Get text samples
    if HAS_TEXT_GENERATOR:
        print("Using advanced text generator (callsigns, headlines, varied patterns)")
        text_samples = generate_texts_for_mode('RTTY', num_samples * 2)
        print(f"  Pre-generated {len(text_samples)} text samples")
    else:
        print("Warning: Text generator not found, using basic fallback")
        text_samples = None
    
    print("=" * 60)
    
    metadata = []
    actual_count = 0
    text_index = 0
    
    for i in range(num_samples * 3):
        if actual_count >= num_samples:
            break
        
        # Create config with randomization
        # Vary mark frequency (common range 1000-2500 Hz)
        mark_freq = random.uniform(1500, 2300)
        
        config = RTTYConfig(
            baud_rate=baud_rate,
            mark_freq=mark_freq,
            space_freq=mark_freq + 170,  # Standard 170 Hz shift
            snr_min=snr_min,
            snr_max=snr_max,
            add_fading=add_fading,
            fade_rate=random.uniform(0.1, 0.5),
            fade_depth=random.uniform(0.2, 0.5),
            freq_drift=random.uniform(0.01, 0.03),
            timing_jitter=random.uniform(0.02, 0.08)
        )
        
        generator = RTTYGenerator(config)
        
        # Get text
        if text_samples and text_index < len(text_samples):
            text = text_samples[text_index]
            text_index += 1
        else:
            text = "CQ CQ CQ DE W1ABC W1ABC K"
        
        # RTTY typically uses uppercase and has limited character set
        # Filter to valid Baudot characters
        valid_text = ''.join(c for c in text.upper() if c.upper() in CHAR_TO_BAUDOT or c == ' ')
        
        if len(valid_text) < 5:
            continue
        
        # Generate audio
        audio, duration = generator.generate_audio(valid_text)
        
        # Skip if too short or too long
        if duration < 1.0 or duration > 30.0 or len(audio) < 100:
            continue
        
        # Save
        sample_id = f"rtty_{actual_count:05d}"
        
        audio_path = audio_dir / f"{sample_id}.wav"
        generator.save_wav(audio, str(audio_path))
        
        text_path = text_dir / f"{sample_id}.txt"
        with open(text_path, 'w') as f:
            f.write(valid_text)
        
        metadata.append({
            'id': sample_id,
            'text': valid_text,
            'duration': round(duration, 2),
            'mark_freq': round(config.mark_freq, 1),
            'baud_rate': baud_rate,
            'snr_db': round(random.uniform(snr_min, snr_max), 1),
            'has_fading': add_fading,
            'text_length': len(valid_text)
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
        print(f"  {m['id']}: \"{m['text'][:40]}...\" ({m['duration']}s)")
    
    # Vocabulary analysis
    all_text = ' '.join(m['text'] for m in metadata)
    unique_chars = sorted(set(all_text))
    print(f"\nVocabulary: {''.join(unique_chars)}")
    print(f"Unique characters: {len(unique_chars)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate RTTY training data')
    parser.add_argument('--output_dir', type=str, default='rtty_synthetic',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--baud_rate', type=float, default=45.45,
                       help='Baud rate (45.45 standard)')
    parser.add_argument('--snr_min', type=float, default=5.0,
                       help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=25.0,
                       help='Maximum SNR in dB')
    parser.add_argument('--no_fading', action='store_true',
                       help='Disable fading simulation')
    
    args = parser.parse_args()
    
    generate_rtty_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        baud_rate=args.baud_rate,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        add_fading=not args.no_fading
    )


if __name__ == '__main__':
    main()

