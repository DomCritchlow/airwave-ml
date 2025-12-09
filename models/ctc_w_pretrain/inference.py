#!/usr/bin/env python3
"""
Inference script for CTC model with pretraining support.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --audio test.wav
    python inference.py --checkpoint checkpoints/best_model.pt --audio audio_dir/
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from model import CTCModel, create_ctc_model


class CTCDecoder:
    """Wrapper for CTC model inference."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """Load model from checkpoint."""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.vocab = checkpoint['vocab']
        self.idx_to_char = checkpoint['idx_to_char']
        
        # Create model
        vocab_size = len(self.vocab)
        self.model = create_ctc_model(self.config, vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Audio config
        audio_cfg = self.config.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 16000)
        self.n_mels = audio_cfg.get('n_mels', 80)
        
        # Feature extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=audio_cfg.get('n_fft', 400),
            hop_length=audio_cfg.get('hop_length', 160)
        )
        
        print(f"Model loaded. Vocab size: {vocab_size}")
        print(f"Model type: {checkpoint.get('model_type', 'unknown')}")
    
    def preprocess(self, audio_path: str) -> tuple:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Mel spectrogram
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = mel.squeeze(0).T  # (time, n_mels)
        
        # Add batch dimension
        mel = mel.unsqueeze(0)  # (1, time, n_mels)
        length = torch.tensor([mel.size(1)])
        
        return mel.to(self.device), length.to(self.device)
    
    def decode_tokens(self, tokens: list) -> str:
        """Convert token list to string."""
        chars = []
        for idx in tokens:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char != '<BLANK>':
                    chars.append(char)
        return ''.join(chars)
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        mel, length = self.preprocess(audio_path)
        
        with torch.no_grad():
            pred_seqs = self.model.decode_greedy(mel, length)
        
        text = self.decode_tokens(pred_seqs[0])
        return text
    
    def transcribe_batch(self, audio_paths: list) -> list:
        """Transcribe multiple audio files."""
        results = []
        for path in audio_paths:
            text = self.transcribe(path)
            results.append({'path': path, 'text': text})
        return results


def main():
    parser = argparse.ArgumentParser(description='CTC Model Inference')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio', '-a', type=str, required=True,
                       help='Path to audio file or directory')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    decoder = CTCDecoder(args.checkpoint, device=args.device)
    
    # Process audio
    audio_path = Path(args.audio)
    
    if audio_path.is_file():
        # Single file
        print(f"\nTranscribing: {audio_path}")
        text = decoder.transcribe(str(audio_path))
        print(f"\nResult: {text}")
        
    elif audio_path.is_dir():
        # Directory of files
        files = list(audio_path.glob('*.wav'))
        print(f"\nTranscribing {len(files)} files...")
        
        for f in files[:20]:  # Limit to 20 for display
            text = decoder.transcribe(str(f))
            print(f"{f.name}: {text}")
    else:
        print(f"Error: {audio_path} not found")
        sys.exit(1)


if __name__ == '__main__':
    main()

