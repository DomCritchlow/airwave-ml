"""
Inference script for audio-to-text model.
Decode audio files to text using a trained model.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import argparse
from pathlib import Path

from model import AudioToTextModel
from utils import load_checkpoint, decode_sequence


def load_audio(audio_path, config):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        
    Returns:
        features: (1, time, freq) tensor
        length: tensor with audio length
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    target_sr = config['audio']['sample_rate']
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract features
    if config['audio']['feature_type'] == 'mel':
        feature_extractor = T.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=config['audio']['n_mels'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length']
        )
        features = feature_extractor(waveform)
        features = torch.log(features + 1e-9)
        features = features.squeeze(0).T  # (time, freq)
    elif config['audio']['feature_type'] == 'mfcc':
        feature_extractor = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=config['audio']['n_mels'],
            melkwargs={
                'n_fft': config['audio']['n_fft'],
                'hop_length': config['audio']['hop_length']
            }
        )
        features = feature_extractor(waveform)
        features = features.squeeze(0).T
    else:
        features = waveform.squeeze(0).unsqueeze(-1)
    
    # Add batch dimension
    features = features.unsqueeze(0)  # (1, time, freq)
    length = torch.tensor([features.size(1)])
    
    return features, length


def transcribe_audio(model, audio_path, config, idx_to_char, device, use_beam_search=False, beam_width=5):
    """
    Transcribe a single audio file.
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        config: Configuration
        idx_to_char: Index to character mapping
        device: Device
        use_beam_search: Whether to use beam search
        beam_width: Beam width for beam search
        
    Returns:
        Transcribed text
    """
    model.eval()
    
    # Load audio
    features, length = load_audio(audio_path, config)
    features = features.to(device)
    length = length.to(device)
    
    # Decode
    with torch.no_grad():
        if use_beam_search:
            pred_seq = model.beam_search_decode(
                features,
                length,
                beam_width=beam_width,
                max_len=config['data']['max_text_len'],
                sos_token=1,
                eos_token=2
            )
        else:
            pred_seq = model.greedy_decode(
                features,
                length,
                max_len=config['data']['max_text_len'],
                sos_token=1,
                eos_token=2
            )
    
    # Decode to text
    text = decode_sequence(pred_seq, idx_to_char)
    
    return text


def batch_transcribe(model, audio_dir, config, idx_to_char, device, output_file=None, use_beam_search=False):
    """
    Transcribe all audio files in a directory.
    
    Args:
        model: Trained model
        audio_dir: Directory containing audio files
        config: Configuration
        idx_to_char: Index to character mapping
        device: Device
        output_file: Optional file to save results
        use_beam_search: Whether to use beam search
    """
    audio_dir = Path(audio_dir)
    audio_files = sorted(list(audio_dir.glob('*.wav')))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print("=" * 80)
    
    results = []
    
    for audio_path in audio_files:
        print(f"\nTranscribing: {audio_path.name}")
        
        text = transcribe_audio(
            model, audio_path, config, idx_to_char, device, 
            use_beam_search=use_beam_search
        )
        
        print(f"Result: {text}")
        results.append((audio_path.name, text))
    
    print("\n" + "=" * 80)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename, text in results:
                f.write(f"{filename}\t{text}\n")
        print(f"\nResults saved to {output_file}")


def main(args):
    """Main inference function."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    idx_to_char = checkpoint['idx_to_char']
    
    print(f"Model from epoch {checkpoint['epoch']}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Set device (auto-detect best available, or use specified)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Get input dimension from config
    if config['audio']['feature_type'] in ['mel', 'mfcc']:
        input_dim = config['audio']['n_mels']
    else:
        input_dim = 1
    
    # Create model
    print("\nCreating model...")
    model = AudioToTextModel(
        vocab_size=len(vocab),
        input_dim=input_dim,
        config=config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Transcribe
    if args.audio:
        # Single file
        print(f"\nTranscribing: {args.audio}")
        print("=" * 80)
        
        text = transcribe_audio(
            model, args.audio, config, idx_to_char, device,
            use_beam_search=args.beam_search,
            beam_width=args.beam_width
        )
        
        print(f"\nResult: {text}")
        print("=" * 80)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\nSaved to {args.output}")
    
    elif args.audio_dir:
        # Batch transcription
        batch_transcribe(
            model, args.audio_dir, config, idx_to_char, device,
            output_file=args.output,
            use_beam_search=args.beam_search
        )
    
    else:
        print("Error: Please specify either --audio or --audio_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio to text')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Path to directory with audio files')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output text')
    parser.add_argument('--beam_search', action='store_true',
                        help='Use beam search instead of greedy decoding')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Beam width for beam search')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    main(args)
