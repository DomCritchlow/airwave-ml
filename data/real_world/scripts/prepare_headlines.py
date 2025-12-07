"""
Prepare headline-based training data.
Instead of arbitrary time-based chunks, split by AR markers (headline boundaries).
This provides better audio-text alignment for Morse code.
"""

import json
import argparse
from pathlib import Path
import torchaudio
import torch


def estimate_headline_timing(segments, total_duration):
    """
    Estimate timing for each headline based on character count.
    Morse code timing roughly correlates with character count.
    
    Returns list of (start_time, end_time) for each segment.
    """
    # Calculate total characters (excluding spaces which are pauses)
    char_counts = []
    for seg in segments:
        # Weight: letters = 1, spaces = 0.5 (word gaps)
        count = sum(1 if c != ' ' else 0.5 for c in seg)
        char_counts.append(count)
    
    total_chars = sum(char_counts)
    
    # Estimate timing proportionally
    timings = []
    current_time = 0.0
    
    # Leave small gaps for AR prosigns between headlines (~2% of total)
    ar_gap = total_duration * 0.02 / max(len(segments) - 1, 1)
    usable_duration = total_duration - ar_gap * (len(segments) - 1)
    
    for i, count in enumerate(char_counts):
        duration = (count / total_chars) * usable_duration
        end_time = current_time + duration
        timings.append((current_time, end_time))
        current_time = end_time + ar_gap
    
    return timings


def extract_headline_samples(raw_dir, output_dir, min_duration=3.0, max_duration=15.0):
    """
    Extract individual headline samples from raw episodes.
    
    Args:
        raw_dir: Directory with raw episodes (audio/ and text/ subdirs + metadata.json)
        output_dir: Output directory for headline samples
        min_duration: Minimum sample duration (skip shorter)
        max_duration: Maximum sample duration (skip longer)
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    audio_out = output_path / 'audio'
    text_out = output_path / 'text'
    audio_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = raw_path / 'metadata.json'
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {raw_path}")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"Processing {len(metadata)} episodes...")
    print("=" * 60)
    
    all_samples = []
    sample_idx = 0
    
    for episode in metadata:
        episode_id = episode['id']
        audio_path = raw_path / 'audio' / episode['audio_file']
        segments = episode['segments']
        
        if not audio_path.exists():
            print(f"  Warning: Audio not found for {episode_id}")
            continue
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        total_duration = waveform.shape[1] / sr
        
        # Estimate timing for each headline
        timings = estimate_headline_timing(segments, total_duration)
        
        # Extract each headline as a sample
        for seg_idx, (segment, (start_time, end_time)) in enumerate(zip(segments, timings)):
            duration = end_time - start_time
            
            # Skip too short or too long
            if duration < min_duration or duration > max_duration:
                continue
            
            # Skip very short text
            if len(segment) < 10:
                continue
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = waveform[:, start_sample:end_sample]
            
            # Save
            sample_id = f"headline_{sample_idx:04d}"
            
            audio_file = audio_out / f"{sample_id}.wav"
            torchaudio.save(str(audio_file), audio_segment, sr)
            
            text_file = text_out / f"{sample_id}.txt"
            with open(text_file, 'w') as f:
                f.write(segment)
            
            all_samples.append({
                'id': sample_id,
                'source_episode': episode_id,
                'segment_idx': seg_idx,
                'text': segment,
                'duration': duration,
                'text_length': len(segment)
            })
            
            sample_idx += 1
    
    # Save metadata
    with open(output_path / 'samples.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Statistics
    durations = [s['duration'] for s in all_samples]
    text_lengths = [s['text_length'] for s in all_samples]
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Avg duration: {sum(durations)/len(durations):.1f}s")
    print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)} chars")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Show sample
    print("\nSample entries:")
    for s in all_samples[:3]:
        print(f"  {s['id']}: \"{s['text'][:40]}...\" ({s['duration']:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description='Extract headline samples from raw episodes')
    parser.add_argument('--raw_dir', type=str, default='morse_data/raw',
                       help='Directory with raw episodes')
    parser.add_argument('--output_dir', type=str, default='morse_data/headlines',
                       help='Output directory for headline samples')
    parser.add_argument('--min_duration', type=float, default=3.0,
                       help='Minimum sample duration in seconds')
    parser.add_argument('--max_duration', type=float, default=15.0,
                       help='Maximum sample duration in seconds')
    
    args = parser.parse_args()
    extract_headline_samples(args.raw_dir, args.output_dir, 
                            args.min_duration, args.max_duration)


if __name__ == '__main__':
    main()

