#!/usr/bin/env python3
"""
Live Morse Code Decoder

Usage:
    # Decode from microphone
    python live_decode.py --checkpoint ../checkpoints/best_model.pt --source mic
    
    # Decode from file
    python live_decode.py --checkpoint ../checkpoints/best_model.pt --source file --input test.wav
    
    # Decode with beam search (slower but more accurate)
    python live_decode.py --checkpoint ../checkpoints/best_model.pt --beam-search
"""

import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from decoder import MorseDecoder, StreamingBuffer, TextMerger, create_audio_source


def decode_live(
    checkpoint_path: str,
    source_type: str = 'mic',
    input_path: str = None,
    window_duration: float = 10.0,
    overlap_duration: float = 2.0,
    use_beam_search: bool = False,
    device: str = None
):
    """
    Run live decoding.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        source_type: 'mic' or 'file'
        input_path: Path to audio file (if source_type='file')
        window_duration: Processing window size in seconds
        overlap_duration: Overlap between windows
        use_beam_search: Use beam search decoding
        device: 'cpu', 'cuda', or 'mps'
    """
    print("=" * 60)
    print("LIVE MORSE CODE DECODER")
    print("=" * 60)
    
    # Load model
    decoder = MorseDecoder(checkpoint_path, device=device)
    
    # Create audio source
    if source_type == 'file':
        if not input_path:
            raise ValueError("--input required for file source")
        source = create_audio_source('file', file_path=input_path, realtime=True)
    else:
        source = create_audio_source('mic')
    
    # Create streaming components
    buffer = StreamingBuffer(
        window_duration=window_duration,
        overlap_duration=overlap_duration
    )
    merger = TextMerger()
    
    print(f"\nSettings:")
    print(f"  Window: {window_duration}s, Overlap: {overlap_duration}s")
    print(f"  Decoding: {'beam search' if use_beam_search else 'greedy'}")
    print()
    print("Decoded text will appear below:")
    print("-" * 60)
    
    try:
        # Process audio stream
        for chunk in source.stream():
            # Add chunk to buffer
            for window in buffer.add(chunk):
                # Decode window
                text = decoder.decode(window, use_beam_search=use_beam_search)
                
                # Merge with previous
                new_text = merger.merge(text)
                
                # Display new text
                if new_text.strip():
                    print(new_text, end='', flush=True)
        
        # Flush remaining buffer
        final_window = buffer.flush()
        if final_window is not None:
            text = decoder.decode(final_window, use_beam_search=use_beam_search)
            new_text = merger.merge(text)
            if new_text.strip():
                print(new_text, end='', flush=True)
                
    except KeyboardInterrupt:
        print("\n")
    
    # Final output
    print()
    print("-" * 60)
    print("\nFull transcript:")
    print(merger.get_full_text())
    print()


def decode_file_batch(
    checkpoint_path: str,
    input_path: str,
    use_beam_search: bool = False,
    device: str = None
):
    """
    Decode entire file at once (non-streaming).
    Faster for short files.
    """
    print("=" * 60)
    print("BATCH MORSE CODE DECODER")
    print("=" * 60)
    
    decoder = MorseDecoder(checkpoint_path, device=device)
    
    print(f"\nDecoding: {input_path}")
    text = decoder.decode_file(input_path, use_beam_search=use_beam_search)
    
    print("-" * 60)
    print(text)
    print("-" * 60)
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Live Morse Code Decoder')
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='../checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        choices=['mic', 'file'],
        default='mic',
        help='Audio source type'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input audio file (required if source=file)'
    )
    
    parser.add_argument(
        '--window',
        type=float,
        default=10.0,
        help='Processing window duration in seconds'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=2.0,
        help='Window overlap in seconds'
    )
    
    parser.add_argument(
        '--beam-search',
        action='store_true',
        help='Use beam search decoding (slower but more accurate)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Decode entire file at once (non-streaming)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to script location
        alt_path = Path(__file__).parent / args.checkpoint
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            print("Make sure training has completed and checkpoint exists.")
            sys.exit(1)
    
    # Run decoder
    if args.batch and args.input:
        decode_file_batch(
            str(checkpoint_path),
            args.input,
            use_beam_search=args.beam_search,
            device=args.device
        )
    else:
        decode_live(
            str(checkpoint_path),
            source_type=args.source,
            input_path=args.input,
            window_duration=args.window,
            overlap_duration=args.overlap,
            use_beam_search=args.beam_search,
            device=args.device
        )


if __name__ == '__main__':
    main()

