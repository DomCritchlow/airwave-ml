"""
Download and prepare Morse code training data from RSS feed.

Usage:
    python scripts/download_morse_data.py download --output_dir morse_data
    python scripts/download_morse_data.py chunk --input_dir morse_data --output_dir morse_chunked --chunk_duration 10
    python scripts/download_morse_data.py full-pipeline --output_dir morse_ready --chunk_duration 10
"""

import argparse
import html
import xml.etree.ElementTree as ET
import urllib.request
import ssl
import re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json

# Create SSL context that doesn't verify certificates (for macOS compatibility)
# This is acceptable for downloading public podcast data
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Check for optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def fetch_rss(url):
    """Fetch and parse RSS feed."""
    print(f"Fetching RSS feed: {url}")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
        content = response.read()
    
    # Parse XML
    root = ET.fromstring(content)
    
    # Handle namespaces
    namespaces = {
        'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
        'content': 'http://purl.org/rss/1.0/modules/content/'
    }
    
    items = []
    for item in root.findall('.//item'):
        # Get enclosure (audio URL)
        enclosure = item.find('enclosure')
        if enclosure is None:
            continue
        
        audio_url = enclosure.get('url')
        
        # Get description (contains transcription)
        desc_elem = item.find('description')
        if desc_elem is None or desc_elem.text is None:
            continue
        
        description = desc_elem.text
        
        # Get title and date for naming
        title = item.find('title')
        title_text = title.text if title is not None else 'unknown'
        
        pub_date = item.find('pubDate')
        pub_date_text = pub_date.text if pub_date is not None else ''
        
        # Get guid for unique naming
        guid = item.find('guid')
        guid_text = guid.text if guid is not None else ''
        
        items.append({
            'audio_url': audio_url,
            'description': description,
            'title': title_text,
            'pub_date': pub_date_text,
            'guid': guid_text
        })
    
    print(f"Found {len(items)} episodes")
    return items


def clean_transcription(description, keep_prosigns=True):
    """
    Clean the morse code transcription from RSS description.
    
    The format is:
    - Starts with "vvv vvv" (attention signal)
    - Headlines separated by "<AR>" (end of message)
    - Ends with "<SK>" (end of transmission)
    
    Args:
        description: Raw RSS description text
        keep_prosigns: If True, keep VVV/AR/SK markers (they're in the audio!)
    """
    # Remove HTML tags (but not prosign markers like <AR>)
    text = re.sub(r'<br\s*/?>', ' ', description)
    
    # Decode HTML entities first (converts &lt;AR&gt; to <AR>)
    text = html.unescape(text)
    
    # Remove the "Morse code transcription:" prefix if present
    text = re.sub(r'Morse code transcription:\s*', '', text, flags=re.IGNORECASE)
    
    if keep_prosigns:
        # Convert prosigns to readable tokens that won't confuse the model
        # These ARE transmitted in the audio, so we must keep them!
        text = re.sub(r'vvv\s+vvv', ' VVV ', text, flags=re.IGNORECASE)
        text = re.sub(r'<AR>', ' AR ', text, flags=re.IGNORECASE)
        text = re.sub(r'<SK>', ' SK ', text, flags=re.IGNORECASE)
    else:
        # Remove prosigns (NOT recommended - creates audio/text mismatch)
        text = re.sub(r'vvv\s+vvv', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<AR>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<SK>', '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text.strip().upper()


def extract_segments(transcription):
    """
    Extract individual segments from the transcription.
    Uses <AR> as delimiter between headlines.
    
    Returns list of clean text segments.
    """
    # Remove leading vvv vvv
    text = re.sub(r'^vvv\s+vvv\s*', '', transcription, flags=re.IGNORECASE)
    
    # Remove trailing SK markers
    text = re.sub(r'\s*<SK>.*$', '', text, flags=re.IGNORECASE)
    
    # Split by <AR> markers
    segments = re.split(r'\s*<AR>\s*', text, flags=re.IGNORECASE)
    
    # Clean each segment
    clean_segments = []
    for seg in segments:
        seg = seg.strip()
        if seg and len(seg) > 5:  # Skip very short segments
            # Convert to uppercase (standard for morse)
            seg = seg.upper()
            clean_segments.append(seg)
    
    return clean_segments


def download_audio(url, output_path, convert_to_wav=True):
    """Download audio file and optionally convert to WAV."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
            content = response.read()
        
        # Save as MP3 first
        mp3_path = output_path.with_suffix('.mp3')
        with open(mp3_path, 'wb') as f:
            f.write(content)
        
        if convert_to_wav:
            wav_path = output_path.with_suffix('.wav')
            # Try to convert using ffmpeg
            try:
                result = subprocess.run(
                    ['ffmpeg', '-i', str(mp3_path), '-ar', '16000', '-ac', '1', 
                     '-y', str(wav_path)],
                    capture_output=True,
                    timeout=120
                )
                if result.returncode == 0:
                    # Remove MP3 after successful conversion
                    mp3_path.unlink()
                    return wav_path, None
                else:
                    return mp3_path, f"FFmpeg conversion failed: {result.stderr.decode()[:200]}"
            except FileNotFoundError:
                return mp3_path, "FFmpeg not found - keeping MP3 format"
            except subprocess.TimeoutExpired:
                return mp3_path, "FFmpeg timeout - keeping MP3 format"
        
        return mp3_path, None
        
    except Exception as e:
        return None, str(e)


def download_dataset(rss_url, output_dir, max_episodes=None):
    """
    Download full dataset from RSS feed.
    
    Creates:
        output_dir/
        ├── audio/
        │   ├── episode_001.wav
        │   └── ...
        ├── text/
        │   ├── episode_001.txt  (full transcription)
        │   └── ...
        └── metadata.json
    """
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    text_dir = output_path / 'text'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch RSS
    items = fetch_rss(rss_url)
    
    if max_episodes:
        items = items[:max_episodes]
    
    print(f"\nDownloading {len(items)} episodes...")
    print("=" * 60)
    
    metadata = []
    errors = []
    
    for idx, item in enumerate(tqdm(items, desc="Downloading")):
        episode_id = f"episode_{idx:03d}"
        
        # Clean transcription - KEEP PROSIGNS since they're in the audio!
        full_transcription = clean_transcription(item['description'], keep_prosigns=True)
        
        # Also extract segments (without prosigns) for metadata reference
        transcription_no_prosigns = clean_transcription(item['description'], keep_prosigns=False)
        segments = extract_segments(transcription_no_prosigns)
        
        if not full_transcription or len(full_transcription) < 10:
            errors.append(f"{episode_id}: No valid transcription found")
            continue
        
        # Download audio
        audio_path = audio_dir / episode_id
        result_path, error = download_audio(item['audio_url'], audio_path)
        
        if error:
            print(f"\n  Warning for {episode_id}: {error}")
        
        if result_path is None:
            errors.append(f"{episode_id}: Download failed")
            continue
        
        # Save full transcription WITH PROSIGNS
        text_path = text_dir / f"{episode_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(full_transcription)
        
        # Store metadata
        metadata.append({
            'id': episode_id,
            'audio_file': result_path.name,
            'text_file': text_path.name,
            'segments': segments,  # Individual headlines (for reference)
            'num_segments': len(segments),
            'full_text_length': len(full_transcription),
            'has_prosigns': True,  # VVV, AR, SK are included
            'pub_date': item['pub_date'],
            'original_url': item['audio_url']
        })
    
    # Save metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Episodes downloaded: {len(metadata)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output directory: {output_path}")
    print(f"  Metadata saved to: {metadata_path}")
    
    if errors:
        print(f"\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")
    
    return metadata


def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe or torchaudio."""
    try:
        # Try ffprobe first
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.decode().strip())
    except:
        pass
    
    # Fallback to torchaudio
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except:
        pass
    
    return None


def chunk_episode(audio_path, text, output_dir, episode_id, chunk_duration=10.0, 
                  overlap=2.0, sample_rate=16000):
    """
    Chunk a single episode into smaller segments.
    
    Since we don't have word-level timestamps, we estimate based on:
    - Morse code at 30 WPM ≈ 6 characters per second (including spaces)
    - We chunk audio and proportionally chunk text
    
    Args:
        audio_path: Path to audio file
        text: Full transcription
        output_dir: Output directory
        episode_id: Episode identifier
        chunk_duration: Target chunk duration in seconds
        overlap: Overlap between chunks in seconds
        sample_rate: Target sample rate
    
    Returns:
        List of created chunk info dicts
    """
    try:
        import torchaudio
        import torch
    except ImportError:
        print("Error: torchaudio required for chunking. Install with: pip install torchaudio")
        return []
    
    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    total_duration = waveform.shape[1] / sr
    total_chars = len(text)
    
    # Calculate chars per second (for proportional text chunking)
    chars_per_second = total_chars / total_duration if total_duration > 0 else 6
    
    # Create chunks
    chunks = []
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    audio_dir = Path(output_dir) / 'audio'
    text_dir = Path(output_dir) / 'text'
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_idx = 0
    start_sample = 0
    
    while start_sample < waveform.shape[1]:
        end_sample = min(start_sample + chunk_samples, waveform.shape[1])
        
        # Extract audio chunk
        audio_chunk = waveform[:, start_sample:end_sample]
        
        # Skip very short chunks at the end
        chunk_dur = audio_chunk.shape[1] / sr
        if chunk_dur < 2.0:  # Skip chunks shorter than 2 seconds
            break
        
        # Calculate corresponding text positions
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        # Proportionally extract text
        text_start = int(start_time * chars_per_second)
        text_end = int(end_time * chars_per_second)
        
        # Ensure we're within bounds
        text_start = max(0, min(text_start, total_chars))
        text_end = max(text_start, min(text_end, total_chars))
        
        # Try to break at word boundaries
        text_chunk = text[text_start:text_end]
        
        # Extend to word boundary at start (go back to find space)
        if text_start > 0 and text_start < total_chars and text[text_start] != ' ':
            for i in range(min(20, text_start)):
                if text[text_start - i - 1] == ' ':
                    text_chunk = text[text_start - i:text_end]
                    break
        
        # Extend to word boundary at end
        if text_end < total_chars and text_end > text_start and text[text_end - 1] != ' ':
            for i in range(min(20, total_chars - text_end)):
                if text[text_end + i] == ' ':
                    text_chunk = text[text_start:text_end + i]
                    break
        
        text_chunk = text_chunk.strip()
        
        # Skip empty text chunks
        if not text_chunk:
            start_sample += step_samples
            continue
        
        # Save chunk
        chunk_id = f"{episode_id}_chunk_{chunk_idx:03d}"
        
        audio_out_path = audio_dir / f"{chunk_id}.wav"
        torchaudio.save(str(audio_out_path), audio_chunk, sr)
        
        text_out_path = text_dir / f"{chunk_id}.txt"
        with open(text_out_path, 'w', encoding='utf-8') as f:
            f.write(text_chunk)
        
        chunks.append({
            'id': chunk_id,
            'audio_file': audio_out_path.name,
            'text_file': text_out_path.name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': chunk_dur,
            'text_length': len(text_chunk),
            'text_preview': text_chunk[:50] + '...' if len(text_chunk) > 50 else text_chunk
        })
        
        chunk_idx += 1
        start_sample += step_samples
    
    return chunks


def chunk_dataset(input_dir, output_dir, chunk_duration=10.0, overlap=2.0):
    """
    Chunk all episodes in a dataset into smaller training segments.
    
    Args:
        input_dir: Directory with downloaded episodes (audio/ and text/ subdirs)
        output_dir: Output directory for chunked data
        chunk_duration: Target duration per chunk in seconds
        overlap: Overlap between consecutive chunks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    audio_dir = input_path / 'audio'
    text_dir = input_path / 'text'
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    # Find all audio files
    audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
    
    if not audio_files:
        print(f"Error: No audio files found in {audio_dir}")
        return
    
    print(f"Chunking {len(audio_files)} episodes...")
    print(f"  Chunk duration: {chunk_duration}s")
    print(f"  Overlap: {overlap}s")
    print("=" * 60)
    
    all_chunks = []
    
    for audio_path in tqdm(audio_files, desc="Chunking"):
        episode_id = audio_path.stem
        text_path = text_dir / f"{episode_id}.txt"
        
        if not text_path.exists():
            print(f"\n  Warning: No text file for {episode_id}")
            continue
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        chunks = chunk_episode(
            audio_path, text, output_dir, episode_id,
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("\nNo chunks created. Make sure torchaudio is installed.")
        return []
    
    # Save chunk metadata
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / 'chunk_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Calculate statistics
    durations = [c['duration'] for c in all_chunks]
    text_lengths = [c['text_length'] for c in all_chunks]
    
    print(f"\n{'='*60}")
    print(f"Chunking complete!")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Avg duration: {sum(durations)/len(durations):.1f}s")
    print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)} chars")
    print(f"  Output: {output_path}")
    
    return all_chunks


def full_pipeline(rss_url, output_dir, chunk_duration=10.0, overlap=2.0, max_episodes=None):
    """
    Full pipeline: download, chunk, and prepare for training.
    """
    output_path = Path(output_dir)
    raw_dir = output_path / 'raw'
    chunked_dir = output_path / 'chunked'
    
    print("=" * 60)
    print("MORSE CODE DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Download
    print("\n[Step 1/3] Downloading episodes...")
    metadata = download_dataset(rss_url, raw_dir, max_episodes=max_episodes)
    
    if not metadata:
        print("Error: No episodes downloaded")
        return
    
    # Step 2: Chunk
    print("\n[Step 2/3] Chunking into training segments...")
    chunks = chunk_dataset(raw_dir, chunked_dir, chunk_duration=chunk_duration, overlap=overlap)
    
    if not chunks:
        print("Error: No chunks created")
        return
    
    # Step 3: Create final training structure
    print("\n[Step 3/3] Creating training dataset structure...")
    
    # The chunked dir already has audio/ and text/ subdirs, so we're ready!
    
    # Analyze vocabulary
    all_text = []
    text_dir = chunked_dir / 'text'
    for txt_file in text_dir.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            all_text.append(f.read())
    
    full_text = ' '.join(all_text)
    unique_chars = sorted(set(full_text))
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDataset ready for training:")
    print(f"  Location: {chunked_dir}")
    print(f"  Total samples: {len(chunks)}")
    print(f"  Unique characters: {len(unique_chars)}")
    print(f"  Vocabulary: {''.join(unique_chars)}")
    
    print(f"\nSuggested config.yaml updates:")
    print(f'  vocab: "{" ".join(unique_chars)}"')
    print(f'  data_dir: "{chunked_dir}"')
    print(f'  max_audio_len: {chunk_duration}')
    
    print(f"\nTo start training:")
    print(f"  python src/train.py --config config.yaml --data_dir {chunked_dir}")
    
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare Morse code training data from RSS feed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only
  python scripts/download_morse_data.py download --output_dir morse_data
  
  # Chunk existing data
  python scripts/download_morse_data.py chunk --input_dir morse_data --output_dir morse_chunked
  
  # Full pipeline (download + chunk)
  python scripts/download_morse_data.py full-pipeline --output_dir morse_ready --chunk_duration 10
  
  # Download limited episodes for testing
  python scripts/download_morse_data.py full-pipeline --output_dir morse_test --max_episodes 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    dl_parser = subparsers.add_parser('download', help='Download episodes from RSS feed')
    dl_parser.add_argument('--rss_url', type=str, 
                          default='https://morse.mdp.im/podcast/rss-30.xml',
                          help='RSS feed URL')
    dl_parser.add_argument('--output_dir', type=str, required=True,
                          help='Output directory')
    dl_parser.add_argument('--max_episodes', type=int, default=None,
                          help='Maximum episodes to download')
    
    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk downloaded data')
    chunk_parser.add_argument('--input_dir', type=str, required=True,
                             help='Input directory with raw data')
    chunk_parser.add_argument('--output_dir', type=str, required=True,
                             help='Output directory for chunks')
    chunk_parser.add_argument('--chunk_duration', type=float, default=10.0,
                             help='Target chunk duration in seconds')
    chunk_parser.add_argument('--overlap', type=float, default=2.0,
                             help='Overlap between chunks in seconds')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', 
                                        help='Download, chunk, and prepare for training')
    full_parser.add_argument('--rss_url', type=str,
                            default='https://morse.mdp.im/podcast/rss-30.xml',
                            help='RSS feed URL')
    full_parser.add_argument('--output_dir', type=str, required=True,
                            help='Output directory')
    full_parser.add_argument('--chunk_duration', type=float, default=10.0,
                            help='Target chunk duration in seconds')
    full_parser.add_argument('--overlap', type=float, default=2.0,
                            help='Overlap between chunks in seconds')
    full_parser.add_argument('--max_episodes', type=int, default=None,
                            help='Maximum episodes to download')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        download_dataset(args.rss_url, args.output_dir, args.max_episodes)
    elif args.command == 'chunk':
        chunk_dataset(args.input_dir, args.output_dir, args.chunk_duration, args.overlap)
    elif args.command == 'full-pipeline':
        full_pipeline(args.rss_url, args.output_dir, args.chunk_duration, 
                     args.overlap, args.max_episodes)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

