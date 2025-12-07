"""
Dataset module for audio-to-text training.
Handles audio loading, feature extraction, and text encoding.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T
from pathlib import Path


class AudioTextDataset(Dataset):
    """Dataset for audio-to-text pairs."""
    
    def __init__(self, audio_paths, text_paths, vocab, config):
        """
        Args:
            audio_paths: List of paths to audio files
            text_paths: List of paths to corresponding text files
            vocab: Vocabulary dictionary (char -> idx)
            config: Config dict with audio parameters
        """
        self.audio_paths = audio_paths
        self.text_paths = text_paths
        self.vocab = vocab
        self.config = config
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2
        
        # Audio processing
        self.sample_rate = config['audio']['sample_rate']
        
        # Choose feature extractor based on config
        if config['audio']['feature_type'] == 'mel':
            self.feature_extractor = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=config['audio']['n_mels'],
                n_fft=config['audio']['n_fft'],
                hop_length=config['audio']['hop_length']
            )
        elif config['audio']['feature_type'] == 'mfcc':
            self.feature_extractor = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=config['audio']['n_mels'],
                melkwargs={
                    'n_fft': config['audio']['n_fft'],
                    'hop_length': config['audio']['hop_length']
                }
            )
        else:
            self.feature_extractor = None  # Raw waveform
            
        # Load texts
        self.texts = []
        for text_path in text_paths:
            with open(text_path, 'r', encoding='utf-8') as f:
                self.texts.append(f.read().strip())
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Truncate audio if too long
        max_audio_samples = int(self.config['data']['max_audio_len'] * self.sample_rate)
        if waveform.shape[1] > max_audio_samples:
            waveform = waveform[:, :max_audio_samples]
        
        # Extract features
        if self.feature_extractor is not None:
            features = self.feature_extractor(waveform)
            features = torch.log(features + 1e-9)  # Log scale
            features = features.squeeze(0).T  # (time, freq)
        else:
            features = waveform.squeeze(0).unsqueeze(-1)  # (time, 1)
        
        # Encode text (also truncate if needed)
        text = self.texts[idx]
        max_text_len = self.config['data']['max_text_len']
        if len(text) > max_text_len - 2:  # Reserve space for SOS/EOS
            text = text[:max_text_len - 2]
        text_encoded = self._encode_text(text)
        
        return features, text_encoded
    
    def _encode_text(self, text):
        """Encode text to token indices with SOS and EOS."""
        encoded = [self.SOS_TOKEN]
        for char in text:
            if char in self.vocab:
                encoded.append(self.vocab[char])
            else:
                # Unknown character, could add <UNK> token
                pass
        encoded.append(self.EOS_TOKEN)
        return torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function to handle variable-length sequences.
    
    Args:
        batch: List of (audio_features, text_encoded) tuples
        
    Returns:
        audio_padded: (batch, max_audio_len, feat_dim)
        audio_lengths: (batch,)
        text_padded: (batch, max_text_len)
        text_lengths: (batch,)
    """
    audios, texts = zip(*batch)
    
    # Get lengths before padding
    audio_lengths = torch.tensor([a.shape[0] for a in audios])
    text_lengths = torch.tensor([len(t) for t in texts])
    
    # Pad sequences
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return audio_padded, audio_lengths, text_padded, text_lengths


def create_vocab(texts, config):
    """
    Create vocabulary from text data or config.
    
    Args:
        texts: List of text strings (can be empty)
        config: Config dict with vocab string
        
    Returns:
        vocab: Dict mapping char -> idx
        idx_to_char: Dict mapping idx -> char
    """
    # Use vocabulary from config
    vocab_str = config.get('vocab', ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Reserve indices for special tokens
    # 0: PAD, 1: SOS, 2: EOS
    vocab = {char: idx + 3 for idx, char in enumerate(vocab_str)}
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = 1
    vocab['<EOS>'] = 2
    
    # Create reverse mapping
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    return vocab, idx_to_char


def load_dataset(data_dir, config, split_ratio=0.9):
    """
    Load dataset from directory.
    
    Args:
        data_dir: Path to data directory
        config: Config dict
        split_ratio: Train/val split ratio
        
    Returns:
        train_dataset, val_dataset, vocab, idx_to_char
    """
    data_path = Path(data_dir)
    audio_dir = data_path / 'audio'
    text_dir = data_path / 'text'
    
    # Get all audio files
    audio_paths = sorted(list(audio_dir.glob('*.wav')))
    
    # Find corresponding text files
    text_paths = []
    valid_audio_paths = []
    
    for audio_path in audio_paths:
        text_path = text_dir / f"{audio_path.stem}.txt"
        if text_path.exists():
            valid_audio_paths.append(str(audio_path))
            text_paths.append(str(text_path))
    
    if len(valid_audio_paths) == 0:
        raise ValueError(f"No matching audio-text pairs found in {data_dir}")
    
    print(f"Found {len(valid_audio_paths)} audio-text pairs")
    
    # Warn about audio length limits
    max_audio_len = config['data']['max_audio_len']
    print(f"Note: Audio will be truncated to {max_audio_len}s. Ensure your text matches this duration!")
    
    # Load texts for vocabulary creation
    texts = []
    for text_path in text_paths:
        with open(text_path, 'r', encoding='utf-8') as f:
            texts.append(f.read().strip())
    
    # Create vocabulary
    vocab, idx_to_char = create_vocab(texts, config)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Shuffle data before split (with fixed seed for reproducibility)
    combined = list(zip(valid_audio_paths, text_paths))
    random.seed(42)
    random.shuffle(combined)
    valid_audio_paths, text_paths = zip(*combined)
    valid_audio_paths = list(valid_audio_paths)
    text_paths = list(text_paths)
    
    # Split into train/val
    split_idx = int(len(valid_audio_paths) * split_ratio)
    
    train_audio = valid_audio_paths[:split_idx]
    train_text = text_paths[:split_idx]
    val_audio = valid_audio_paths[split_idx:]
    val_text = text_paths[split_idx:]
    
    # Create datasets
    train_dataset = AudioTextDataset(train_audio, train_text, vocab, config)
    val_dataset = AudioTextDataset(val_audio, val_text, vocab, config)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, vocab, idx_to_char


def get_dataloaders(train_dataset, val_dataset, config):
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle'],
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader
