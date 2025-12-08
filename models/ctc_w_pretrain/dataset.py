"""
Dataset classes for CTC model with pretraining support.

Includes:
- RadioUnlabeledDataset: For masked spectrogram pretraining (no labels needed)
- Reuses existing audio pipeline (mel spectrogram extraction)
"""

import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Reuse augmentation from main CTC model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'ctc'))
try:
    from augmentation import AudioAugmenter, create_augmenter_from_config
except ImportError:
    AudioAugmenter = None
    create_augmenter_from_config = None


class RadioUnlabeledDataset(Dataset):
    """
    Dataset for unlabeled radio audio - used for masked spectrogram pretraining.
    
    Loads audio files and returns mel spectrograms without any labels.
    Can combine synthetic and real-world audio.
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        config: dict,
        augmenter: Optional['AudioAugmenter'] = None
    ):
        self.audio_paths = audio_paths
        self.config = config
        self.augmenter = augmenter
        
        # Audio config
        audio_cfg = config.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 16000)
        self.n_mels = audio_cfg.get('n_mels', 80)
        self.n_fft = audio_cfg.get('n_fft', 400)
        self.hop_length = audio_cfg.get('hop_length', 160)
        
        # Data config
        data_cfg = config.get('data', {})
        self.max_audio_len = data_cfg.get('max_audio_len', 2000)
        
        # Feature extractor (reuses same pipeline as CTC model)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            mel: (time, n_mels) mel spectrogram
        """
        # Load audio
        audio_path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply augmentation
        if self.augmenter is not None:
            waveform = self.augmenter(waveform)
        
        # Extract mel spectrogram
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = mel.squeeze(0).T  # (time, n_mels)
        
        # Truncate if too long
        if mel.size(0) > self.max_audio_len:
            mel = mel[:self.max_audio_len]
        
        return mel


def unlabeled_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for unlabeled data.
    
    Returns:
        mel_padded: (batch, max_time, n_mels)
        lengths: (batch,)
    """
    lengths = torch.tensor([mel.size(0) for mel in batch], dtype=torch.long)
    mel_padded = pad_sequence(batch, batch_first=True)
    return mel_padded, lengths


def find_audio_files(directories: List[str], extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[str]:
    """
    Find all audio files in given directories.
    
    Args:
        directories: List of directory paths to search
        extensions: Audio file extensions to include
        
    Returns:
        List of audio file paths
    """
    audio_paths = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {directory}")
            continue
            
        for ext in extensions:
            audio_paths.extend([str(p) for p in dir_path.rglob(f'*{ext}')])
    
    return sorted(audio_paths)


def load_unlabeled_dataset(
    data_dirs: List[str],
    config: dict,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Load unlabeled audio dataset for pretraining.
    
    Args:
        data_dirs: List of directories containing audio files
        config: Configuration dict
        val_ratio: Validation set ratio
        
    Returns:
        train_loader, val_loader
    """
    # Find all audio files
    audio_paths = find_audio_files(data_dirs)
    print(f"Found {len(audio_paths)} audio files for pretraining")
    
    if len(audio_paths) == 0:
        raise ValueError(f"No audio files found in {data_dirs}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(audio_paths)
    
    split_idx = int(len(audio_paths) * (1 - val_ratio))
    train_paths = audio_paths[:split_idx]
    val_paths = audio_paths[split_idx:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create augmenter for training (if available)
    augmenter = None
    if create_augmenter_from_config is not None:
        audio_cfg = config.get('audio', {})
        sample_rate = audio_cfg.get('sample_rate', 16000)
        augmenter = create_augmenter_from_config(config, sample_rate)
        if augmenter:
            print("Audio augmentation: ENABLED for pretraining")
    
    # Create datasets
    train_dataset = RadioUnlabeledDataset(train_paths, config, augmenter=augmenter)
    val_dataset = RadioUnlabeledDataset(val_paths, config, augmenter=None)
    
    # Create dataloaders
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=unlabeled_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=unlabeled_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Also include the labeled dataset loader for CTC fine-tuning
# (reuses logic from models/ctc/dataset.py)

def create_ctc_vocab(config: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create vocabulary for CTC model."""
    vocab_chars = config.get('vocab', ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    char_to_idx = {'<BLANK>': 0}
    idx_to_char = {0: '<BLANK>'}
    
    for i, char in enumerate(vocab_chars):
        char_to_idx[char] = i + 1
        idx_to_char[i + 1] = char
    
    return char_to_idx, idx_to_char


class CTCAudioTextDataset(Dataset):
    """Dataset for CTC training with labels."""
    
    def __init__(
        self,
        audio_paths: List[str],
        text_paths: List[str],
        char_to_idx: Dict[str, int],
        config: dict,
        augmenter: Optional['AudioAugmenter'] = None,
        is_training: bool = True
    ):
        self.audio_paths = audio_paths
        self.text_paths = text_paths
        self.char_to_idx = char_to_idx
        self.config = config
        self.augmenter = augmenter
        self.is_training = is_training
        
        audio_cfg = config.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 16000)
        self.n_mels = audio_cfg.get('n_mels', 80)
        self.n_fft = audio_cfg.get('n_fft', 400)
        self.hop_length = audio_cfg.get('hop_length', 160)
        
        data_cfg = config.get('data', {})
        self.max_audio_len = data_cfg.get('max_audio_len', 2000)
        self.max_text_len = data_cfg.get('max_text_len', 200)
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if self.is_training and self.augmenter is not None:
            waveform = self.augmenter(waveform)
        
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = mel.squeeze(0).T
        
        if mel.size(0) > self.max_audio_len:
            mel = mel[:self.max_audio_len]
        
        text_path = self.text_paths[idx]
        with open(text_path, 'r') as f:
            text = f.read().strip().upper()
        
        if len(text) > self.max_text_len:
            text = text[:self.max_text_len]
        
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        
        text_tensor = torch.tensor(encoded, dtype=torch.long)
        
        return mel, text_tensor


def ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for CTC training."""
    audios, texts = zip(*batch)
    
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
    
    audio_padded = pad_sequence(audios, batch_first=True)
    text_concat = torch.cat(texts)
    
    return audio_padded, audio_lengths, text_concat, text_lengths


def load_ctc_dataset(
    data_dir: str,
    config: dict,
    split_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Load labeled dataset for CTC fine-tuning."""
    data_dir = Path(data_dir)
    audio_dir = data_dir / 'audio'
    text_dir = data_dir / 'text'
    
    audio_paths = []
    text_paths = []
    
    for audio_file in sorted(audio_dir.glob('*.wav')):
        text_file = text_dir / (audio_file.stem + '.txt')
        if text_file.exists():
            audio_paths.append(str(audio_file))
            text_paths.append(str(text_file))
    
    print(f"Found {len(audio_paths)} audio/text pairs")
    
    if len(audio_paths) == 0:
        raise ValueError(f"No data found in {data_dir}")
    
    char_to_idx, idx_to_char = create_ctc_vocab(config)
    print(f"Vocabulary size: {len(char_to_idx)}")
    
    indices = list(range(len(audio_paths)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * split_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_audio = [audio_paths[i] for i in train_indices]
    train_text = [text_paths[i] for i in train_indices]
    val_audio = [audio_paths[i] for i in val_indices]
    val_text = [text_paths[i] for i in val_indices]
    
    print(f"Train: {len(train_audio)}, Val: {len(val_audio)}")
    
    # Create augmenter
    augmenter = None
    if create_augmenter_from_config is not None:
        audio_cfg = config.get('audio', {})
        sample_rate = audio_cfg.get('sample_rate', 16000)
        augmenter = create_augmenter_from_config(config, sample_rate)
    
    train_dataset = CTCAudioTextDataset(
        train_audio, train_text, char_to_idx, config,
        augmenter=augmenter, is_training=True
    )
    val_dataset = CTCAudioTextDataset(
        val_audio, val_text, char_to_idx, config,
        augmenter=None, is_training=False
    )
    
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 16)
    num_workers = train_cfg.get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, char_to_idx, idx_to_char

