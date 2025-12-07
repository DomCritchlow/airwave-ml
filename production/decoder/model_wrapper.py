"""
Model wrapper for inference.
Loads a trained checkpoint and provides a simple decode() interface.
Supports both CTC and seq2seq models (auto-detected from checkpoint).
"""

import sys
from pathlib import Path

# Add model directories to path for imports
# New structure: models/attention/ and models/ctc/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models' / 'attention'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models' / 'ctc'))
# Legacy paths (for backwards compatibility)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src_ctc'))

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, List


class MorseDecoder:
    """
    Wrapper around the trained model for real-time inference.
    Automatically detects model type (CTC or seq2seq) from checkpoint.
    
    Usage:
        decoder = MorseDecoder('checkpoints/best_model.pt')
        text = decoder.decode(audio_array)  # numpy array, 16kHz mono
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        """
        # Determine device
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
        
        # Detect model type
        self.model_type = checkpoint.get('model_type', 'seq2seq')
        print(f"Model type: {self.model_type}")
        
        if self.model_type == 'ctc':
            self._load_ctc_model(checkpoint)
        else:
            self._load_seq2seq_model(checkpoint)
        
        # Create feature extractor
        self.sample_rate = self.config['audio']['sample_rate']
        self._setup_feature_extractor()
        
        print(f"Model loaded. Vocab size: {len(self.vocab)}")
    
    def _load_ctc_model(self, checkpoint):
        """Load CTC model."""
        from model import create_ctc_model
        
        vocab_size = len(self.vocab)
        self.model = create_ctc_model(self.config, vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_seq2seq_model(self, checkpoint):
        """Load seq2seq Transformer model."""
        from model import AudioToTextModel
        
        # Determine input dimension
        feature_type = self.config.get('audio', {}).get('feature_type', 'mel')
        if feature_type in ['mel', 'mfcc']:
            input_dim = self.config['audio']['n_mels']
        else:
            input_dim = 1
        
        self.model = AudioToTextModel(
            vocab_size=len(self.vocab),
            input_dim=input_dim,
            config=self.config
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _setup_feature_extractor(self):
        """Setup mel spectrogram or MFCC extractor."""
        feature_type = self.config.get('audio', {}).get('feature_type', 'mel')
        
        if feature_type == 'mel':
            self.feature_extractor = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.config['audio']['n_mels'],
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length']
            )
        elif feature_type == 'mfcc':
            self.feature_extractor = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.config['audio']['n_mels'],
                melkwargs={
                    'n_fft': self.config['audio']['n_fft'],
                    'hop_length': self.config['audio']['hop_length']
                }
            )
        else:
            self.feature_extractor = None
    
    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio array to model input features.
        
        Args:
            audio: numpy array, expected 16kHz mono
            
        Returns:
            features: (1, time, freq) tensor
        """
        # Convert to tensor
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        waveform = torch.tensor(audio).unsqueeze(0)  # (1, samples)
        
        # Extract features
        if self.feature_extractor is not None:
            features = self.feature_extractor(waveform)
            features = torch.log(features + 1e-9)
            features = features.squeeze(0).T  # (time, freq)
        else:
            features = waveform.squeeze(0).unsqueeze(-1)
        
        # Add batch dimension
        features = features.unsqueeze(0)  # (1, time, freq)
        
        return features.to(self.device)
    
    def decode(
        self, 
        audio: np.ndarray, 
        use_beam_search: bool = False,
        beam_width: int = 5
    ) -> str:
        """
        Decode audio to text.
        
        Args:
            audio: numpy array, 16kHz mono
            use_beam_search: Use beam search for better quality (slower)
            beam_width: Beam width if using beam search
            
        Returns:
            Decoded text string
        """
        # Preprocess
        features = self.preprocess(audio)
        length = torch.tensor([features.size(1)], device=self.device)
        
        # Inference (different for CTC vs seq2seq)
        with torch.no_grad():
            if self.model_type == 'ctc':
                pred_seq = self._decode_ctc(features, length, use_beam_search, beam_width)
            else:
                pred_seq = self._decode_seq2seq(features, length, use_beam_search, beam_width)
        
        # Convert to text
        text = self._decode_sequence(pred_seq)
        
        return text
    
    def _decode_ctc(self, features, length, use_beam, beam_width):
        """CTC decoding."""
        if use_beam:
            pred_seqs = self.model.decode_beam(features, length, beam_width=beam_width)
        else:
            pred_seqs = self.model.decode_greedy(features, length)
        return pred_seqs[0]  # Return first (only) sequence in batch
    
    def _decode_seq2seq(self, features, length, use_beam, beam_width):
        """Seq2seq decoding."""
        max_len = self.config.get('data', {}).get('max_text_len', 200)
        
        if use_beam:
            pred_seq = self.model.beam_search_decode(
                features, length,
                beam_width=beam_width,
                max_len=max_len,
                sos_token=1,
                eos_token=2
            )
        else:
            pred_seq = self.model.greedy_decode(
                features, length,
                max_len=max_len,
                sos_token=1,
                eos_token=2
            )
        return pred_seq
    
    def _decode_sequence(self, sequence: List[int]) -> str:
        """Convert token sequence to text."""
        chars = []
        for idx in sequence:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                # Skip special tokens
                if char in ['<PAD>', '<SOS>', '<EOS>', '<BLANK>']:
                    if char == '<EOS>':
                        break
                    continue
                chars.append(char)
        return ''.join(chars)
    
    def decode_file(self, audio_path: str, **kwargs) -> str:
        """
        Decode an audio file.
        
        Args:
            audio_path: Path to WAV file
            **kwargs: Passed to decode()
            
        Returns:
            Decoded text
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio = waveform.squeeze().numpy()
        
        return self.decode(audio, **kwargs)

