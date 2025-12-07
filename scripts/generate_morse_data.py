"""
Generate perfectly aligned Morse code audio + text training data.

This creates synthetic Morse code audio from text, giving us:
- Perfect audio-text alignment (we control exactly what's generated)
- Configurable WPM (words per minute)
- Configurable sample length
- Any text we want (sentences, words, callsigns, etc.)

Usage:
    python scripts/generate_morse_data.py --output_dir morse_synthetic --num_samples 500 --wpm 25
"""

import argparse
import numpy as np
import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import wave
import struct

# Morse code dictionary
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', 
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '!': '-.-.--', '/': '-..-.',
    '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.',
    '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
    '$': '...-..-', '@': '.--.-.', "'": '.----.',
    # Prosigns (sent as single units)
    '<AR>': '.-.-.', '<SK>': '...-.-', '<BT>': '-...-', '<KN>': '-.--.',
}

# Import the comprehensive text generator
try:
    from generate_training_text import TextGenerator, generate_texts_for_mode, TextGeneratorConfig
    HAS_TEXT_GENERATOR = True
except ImportError:
    HAS_TEXT_GENERATOR = False

# Fallback sample text sources (used if text generator not available)
SAMPLE_SENTENCES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "HELLO WORLD THIS IS A TEST",
    "CQ CQ CQ DE W1ABC W1ABC K",
    "WEATHER REPORT SUNNY AND WARM",
    "BREAKING NEWS TODAY",
]

# Fallback common words
COMMON_WORDS = [
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
]


@dataclass
class MorseConfig:
    """Configuration for Morse code generation."""
    wpm: int = 25              # Words per minute (base speed)
    frequency: int = 600       # Tone frequency in Hz (base)
    sample_rate: int = 16000   # Audio sample rate
    rise_time: float = 0.005   # Attack/decay time in seconds (prevents clicks)
    
    # Variability settings (simulates human "fist")
    timing_variance: float = 0.15    # ±15% timing variation per element
    wpm_drift: float = 0.10          # ±10% speed drift over transmission
    frequency_drift: float = 0.02    # ±2% frequency drift
    
    # Noise and imperfections
    add_noise: bool = True           # Add background noise
    noise_level: float = 0.05        # Noise amplitude (0-1)
    click_probability: float = 0.02  # Probability of key click artifact
    
    # Realistic fist patterns
    rushed_dits: float = 0.85        # Dits often slightly short
    lazy_dahs: float = 1.05          # Dahs often slightly long
    word_gap_variance: float = 0.30  # Word gaps vary a lot (±30%)


class MorseGenerator:
    """Generate Morse code audio from text with realistic human variability."""
    
    def __init__(self, config: MorseConfig):
        self.config = config
        self._reset_transmission()
    
    def _reset_transmission(self):
        """Reset per-transmission state (for speed drift, etc.)."""
        # Calculate base timing based on WPM
        # "PARIS" is the standard word (50 units)
        self.base_unit_duration = 60.0 / (self.config.wpm * 50)
        
        # Current drift (changes gradually during transmission)
        self.current_speed_factor = 1.0
        self.current_freq_offset = 0
        
        # Track position for drift
        self.element_count = 0
    
    def _apply_timing_variance(self, duration: float, element_type: str = 'normal') -> float:
        """Apply realistic timing variance to an element."""
        # Base variance
        variance = 1.0 + random.gauss(0, self.config.timing_variance)
        
        # Specific patterns for different elements
        if element_type == 'dit':
            # Dits tend to be slightly rushed
            variance *= self.config.rushed_dits + random.gauss(0, 0.05)
        elif element_type == 'dah':
            # Dahs tend to be slightly lazy/long
            variance *= self.config.lazy_dahs + random.gauss(0, 0.05)
        elif element_type == 'word_gap':
            # Word gaps have high variance (operator thinking)
            variance *= 1.0 + random.gauss(0, self.config.word_gap_variance)
        
        # Apply speed drift (gradual change over transmission)
        self.element_count += 1
        drift_change = random.gauss(0, 0.002)  # Small random walk
        self.current_speed_factor += drift_change
        self.current_speed_factor = np.clip(
            self.current_speed_factor, 
            1 - self.config.wpm_drift, 
            1 + self.config.wpm_drift
        )
        
        return duration * variance * self.current_speed_factor
    
    def _get_current_frequency(self) -> float:
        """Get current frequency with drift."""
        # Random walk for frequency
        drift_change = random.gauss(0, 0.5)  # Small Hz change
        self.current_freq_offset += drift_change
        max_drift = self.config.frequency * self.config.frequency_drift
        self.current_freq_offset = np.clip(self.current_freq_offset, -max_drift, max_drift)
        
        return self.config.frequency + self.current_freq_offset
    
    def _generate_tone(self, duration: float, element_type: str = 'normal') -> np.ndarray:
        """Generate a sine wave tone with envelope and realistic imperfections."""
        # Apply timing variance
        actual_duration = self._apply_timing_variance(duration, element_type)
        actual_duration = max(actual_duration, 0.01)  # Minimum duration
        
        num_samples = int(actual_duration * self.config.sample_rate)
        if num_samples < 10:
            return np.zeros(10)
        
        t = np.linspace(0, actual_duration, num_samples, endpoint=False)
        
        # Get frequency with drift
        freq = self._get_current_frequency()
        
        # Generate sine wave
        tone = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope to prevent clicks
        rise_time = self.config.rise_time * (1 + random.gauss(0, 0.2))  # Variable rise time
        rise_samples = int(rise_time * self.config.sample_rate)
        
        if rise_samples > 0 and num_samples > 2 * rise_samples:
            envelope = np.ones(num_samples)
            # Attack (sometimes sharper, sometimes softer)
            attack_curve = random.uniform(0.5, 2.0)  # Vary the curve shape
            envelope[:rise_samples] = np.power(np.linspace(0, 1, rise_samples), attack_curve)
            # Decay
            decay_curve = random.uniform(0.5, 2.0)
            envelope[-rise_samples:] = np.power(np.linspace(1, 0, rise_samples), decay_curve)
            tone = tone * envelope
        
        # Occasionally add key click artifacts
        if self.config.click_probability > 0 and random.random() < self.config.click_probability:
            click = np.zeros(num_samples)
            click_pos = random.randint(0, min(50, num_samples - 1))
            click_len = random.randint(5, 20)
            click[click_pos:click_pos + click_len] = random.uniform(0.1, 0.3)
            tone = tone + click
        
        return tone
    
    def _generate_silence(self, duration: float, element_type: str = 'normal') -> np.ndarray:
        """Generate silence with realistic timing variance."""
        actual_duration = self._apply_timing_variance(duration, element_type)
        actual_duration = max(actual_duration, 0.005)  # Minimum gap
        
        num_samples = int(actual_duration * self.config.sample_rate)
        silence = np.zeros(num_samples)
        
        # Add background noise during silence
        if self.config.add_noise and self.config.noise_level > 0:
            noise = np.random.randn(num_samples) * self.config.noise_level * 0.3
            silence = silence + noise
        
        return silence
    
    def text_to_morse(self, text: str) -> str:
        """Convert text to Morse code string."""
        text = text.upper()
        morse_parts = []
        
        for char in text:
            if char == ' ':
                morse_parts.append(' ')  # Word separator
            elif char in MORSE_CODE:
                morse_parts.append(MORSE_CODE[char])
            # Skip unknown characters
        
        return ' '.join(morse_parts)
    
    def generate_audio(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Generate Morse code audio from text with realistic human variability.
        
        Returns:
            audio: numpy array of audio samples
            duration: total duration in seconds
        """
        # Reset state for new transmission
        self._reset_transmission()
        
        text = text.upper()
        audio_parts = []
        
        # Base timing (will be varied per element)
        unit = self.base_unit_duration
        dit_base = 1 * unit
        dah_base = 3 * unit
        intra_gap_base = 1 * unit
        inter_gap_base = 3 * unit
        word_gap_base = 7 * unit
        
        i = 0
        while i < len(text):
            # Check for prosigns like <AR>
            if text[i] == '<':
                end = text.find('>', i)
                if end != -1:
                    prosign = text[i:end+1]
                    if prosign in MORSE_CODE:
                        morse = MORSE_CODE[prosign]
                        audio_parts.append(self._encode_morse(morse, dit_base, dah_base, intra_gap_base))
                        audio_parts.append(self._generate_silence(inter_gap_base, 'inter_char'))
                        i = end + 1
                        continue
            
            char = text[i]
            
            if char == ' ':
                # Word gap (minus the inter-char gap already added)
                gap_duration = word_gap_base - inter_gap_base
                audio_parts.append(self._generate_silence(gap_duration, 'word_gap'))
            elif char in MORSE_CODE:
                morse = MORSE_CODE[char]
                audio_parts.append(self._encode_morse(morse, dit_base, dah_base, intra_gap_base))
                audio_parts.append(self._generate_silence(inter_gap_base, 'inter_char'))
            
            i += 1
        
        if audio_parts:
            audio = np.concatenate(audio_parts)
            
            # Add overall background noise
            if self.config.add_noise and self.config.noise_level > 0:
                noise = np.random.randn(len(audio)) * self.config.noise_level
                audio = audio + noise
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
        else:
            audio = np.array([])
        
        duration = len(audio) / self.config.sample_rate
        return audio, duration
    
    def _encode_morse(self, morse: str, dit_base: float, dah_base: float, intra_gap_base: float) -> np.ndarray:
        """Encode a single Morse code character with realistic timing variation."""
        parts = []
        
        for i, symbol in enumerate(morse):
            if symbol == '.':
                parts.append(self._generate_tone(dit_base, 'dit'))
            elif symbol == '-':
                parts.append(self._generate_tone(dah_base, 'dah'))
            
            # Add intra-character gap (except after last symbol)
            if i < len(morse) - 1:
                parts.append(self._generate_silence(intra_gap_base, 'intra_char'))
        
        return np.concatenate(parts) if parts else np.array([])
    
    def save_wav(self, audio: np.ndarray, filepath: str):
        """Save audio as WAV file."""
        # Normalize to 16-bit range
        audio = audio / np.max(np.abs(audio)) * 0.8  # Leave headroom
        audio_int = (audio * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_int.tobytes())


def generate_random_sentence(min_words: int = 3, max_words: int = 8) -> str:
    """Generate a random sentence from common words."""
    num_words = random.randint(min_words, max_words)
    words = random.choices(COMMON_WORDS, k=num_words)
    return ' '.join(words)


def create_operator_style() -> MorseConfig:
    """Create a random operator 'fist' style."""
    base_wpm = random.randint(18, 32)  # Vary speed between operators
    
    return MorseConfig(
        wpm=base_wpm,
        frequency=random.randint(500, 800),  # Different operators use different tones
        timing_variance=random.uniform(0.08, 0.25),  # Some more consistent than others
        wpm_drift=random.uniform(0.05, 0.15),  # Some drift more
        frequency_drift=random.uniform(0.01, 0.04),
        add_noise=True,
        noise_level=random.uniform(0.02, 0.10),  # Varying conditions
        click_probability=random.uniform(0.0, 0.05),
        rushed_dits=random.uniform(0.80, 0.95),  # Operator habits
        lazy_dahs=random.uniform(1.0, 1.15),
        word_gap_variance=random.uniform(0.15, 0.40),  # Thinking time varies
    )


def generate_training_data(
    output_dir: str,
    num_samples: int = 500,
    wpm: int = 25,
    min_words: int = 3,
    max_words: int = 8,
    include_sentences: bool = True,
    include_prosigns: bool = True,
    vary_operators: bool = True,
    use_advanced_text: bool = True  # Use comprehensive text generator
):
    """
    Generate Morse code training data with realistic human variability.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        wpm: Base words per minute (will vary if vary_operators=True)
        min_words: Minimum words per sample
        max_words: Maximum words per sample
        include_sentences: Include full sentences from SAMPLE_SENTENCES
        include_prosigns: Add AR prosigns between some phrases
        vary_operators: Simulate different operator styles
        use_advanced_text: Use comprehensive text generator with callsigns, Q-codes, etc.
    """
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    text_dir = output_path / 'text'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} Morse code samples...")
    print(f"Base WPM: {wpm} (will vary per operator)" if vary_operators else f"WPM: {wpm}")
    print(f"Output directory: {output_path}")
    
    # Use advanced text generator if available
    if use_advanced_text and HAS_TEXT_GENERATOR:
        print("Using advanced text generator (callsigns, Q-codes, varied patterns)")
        text_samples = generate_texts_for_mode('CW', num_samples * 2)  # Generate extra
        print(f"  Pre-generated {len(text_samples)} text samples")
    else:
        if use_advanced_text and not HAS_TEXT_GENERATOR:
            print("Warning: Advanced text generator not found, using basic fallback")
        text_samples = None
    
    print("=" * 60)
    
    metadata = []
    actual_count = 0
    text_index = 0
    
    # Create a pool of "operators" (different fists)
    num_operators = max(10, num_samples // 50)  # ~50 samples per operator
    operator_styles = [create_operator_style() for _ in range(num_operators)]
    
    for i in range(num_samples * 3):  # Generate extra to account for skipped samples
        if actual_count >= num_samples:
            break
            
        # Pick an operator style
        if vary_operators:
            config = random.choice(operator_styles)
        else:
            config = MorseConfig(wpm=wpm)
        
        generator = MorseGenerator(config)
        
        # Get text sample
        if text_samples and text_index < len(text_samples):
            # Use pre-generated diverse text
            text = text_samples[text_index]
            text_index += 1
        else:
            # Fallback to old method
            sample_type = random.choice(['random', 'sentence', 'mixed'] if include_sentences else ['random'])
            
            if sample_type == 'sentence':
                text = random.choice(SAMPLE_SENTENCES)
            elif sample_type == 'mixed':
                sentence = random.choice(SAMPLE_SENTENCES)
                words = sentence.split()[:random.randint(2, 5)]
                extra_words = random.choices(COMMON_WORDS, k=random.randint(1, 3))
                text = ' '.join(words + extra_words)
            else:
                text = generate_random_sentence(min_words, max_words)
        
        # Optionally add prosigns (less often since text generator already includes them)
        if include_prosigns and not text_samples and random.random() < 0.3:
            if random.random() < 0.5:
                text = "VVV " + text
            if random.random() < 0.5:
                text = text + " AR"
        
        # Generate audio
        audio, duration = generator.generate_audio(text)
        
        # Skip if too short or too long
        if duration < 2.0 or duration > 20.0 or len(audio) < 100:
            continue
        
        # Save
        sample_id = f"morse_{actual_count:05d}"
        
        audio_path = audio_dir / f"{sample_id}.wav"
        generator.save_wav(audio, str(audio_path))
        
        text_path = text_dir / f"{sample_id}.txt"
        with open(text_path, 'w') as f:
            f.write(text)
        
        metadata.append({
            'id': sample_id,
            'text': text,
            'duration': round(duration, 2),
            'wpm': config.wpm,
            'frequency': config.frequency,
            'noise_level': round(config.noise_level, 3),
            'num_words': len(text.split())
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
    print(f"  WPM: {wpm}")
    print(f"\nSample entries:")
    for m in metadata[:5]:
        print(f"  {m['id']}: \"{m['text'][:40]}...\" ({m['duration']}s)")
    print(f"{'='*60}")
    
    # Vocabulary analysis
    all_text = ' '.join(m['text'] for m in metadata)
    unique_chars = sorted(set(all_text))
    print(f"\nVocabulary: {''.join(unique_chars)}")
    print(f"Unique characters: {len(unique_chars)}")


def main():
    parser = argparse.ArgumentParser(description='Generate Morse code training data with realistic human variability')
    parser.add_argument('--output_dir', type=str, default='morse_synthetic',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--wpm', type=int, default=25,
                       help='Base words per minute (default: 25)')
    parser.add_argument('--min_words', type=int, default=3,
                       help='Minimum words per sample')
    parser.add_argument('--max_words', type=int, default=8,
                       help='Maximum words per sample')
    parser.add_argument('--no_sentences', action='store_true',
                       help='Disable full sentences')
    parser.add_argument('--no_prosigns', action='store_true',
                       help='Disable prosigns (VVV, AR)')
    parser.add_argument('--no_vary_operators', action='store_true',
                       help='Disable operator variability (consistent timing)')
    parser.add_argument('--basic_text', action='store_true',
                       help='Use basic text (disable advanced callsigns/Q-codes)')
    
    args = parser.parse_args()
    
    generate_training_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        wpm=args.wpm,
        min_words=args.min_words,
        max_words=args.max_words,
        include_sentences=not args.no_sentences,
        include_prosigns=not args.no_prosigns,
        vary_operators=not args.no_vary_operators,
        use_advanced_text=not args.basic_text
    )


if __name__ == '__main__':
    main()

