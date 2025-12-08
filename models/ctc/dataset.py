"""
Dataset for CTC-based model.

Key differences from seq2seq dataset:
- Vocab has BLANK at index 0 (required by CTC)
- No SOS/EOS tokens (CTC handles alignment)
- Different collate function for CTC loss input format
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

from augmentation import AudioAugmenter, create_augmenter_from_config


def create_ctc_vocab(config: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary for CTC model.
    
    CTC requires blank token at index 0.
    
    Args:
        config: Config with vocab string
        
    Returns:
        char_to_idx: {char: index}
        idx_to_char: {index: char}
    """
    vocab_chars = config.get('vocab', ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Index 0 is reserved for CTC blank
    char_to_idx = {'<BLANK>': 0}
    idx_to_char = {0: '<BLANK>'}
    
    # Add vocab characters starting at index 1
    for i, char in enumerate(vocab_chars):
        char_to_idx[char] = i + 1
        idx_to_char[i + 1] = char
    
    return char_to_idx, idx_to_char


class CTCAudioTextDataset(Dataset):
    """
    Dataset for CTC training.
    
    Loads audio files and corresponding text files.
    Returns mel spectrograms and encoded text (without SOS/EOS).
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        text_paths: List[str],
        char_to_idx: Dict[str, int],
        config: dict,
        augmenter: Optional[AudioAugmenter] = None,
        is_training: bool = True
    ):
        self.audio_paths = audio_paths
        self.text_paths = text_paths
        self.char_to_idx = char_to_idx
        self.config = config
        self.augmenter = augmenter
        self.is_training = is_training
        
        # Audio config
        audio_cfg = config.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 16000)
        self.n_mels = audio_cfg.get('n_mels', 80)
        self.n_fft = audio_cfg.get('n_fft', 400)
        self.hop_length = audio_cfg.get('hop_length', 160)
        
        # Data config
        data_cfg = config.get('data', {})
        self.max_audio_len = data_cfg.get('max_audio_len', 2000)
        self.max_text_len = data_cfg.get('max_text_len', 200)
        
        # Feature extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Apply augmentation (only during training)
        if self.is_training and self.augmenter is not None:
            waveform = self.augmenter(waveform)
        
        # Extract mel spectrogram
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = mel.squeeze(0).T  # (time, n_mels)
        
        # Truncate if too long
        if mel.size(0) > self.max_audio_len:
            mel = mel[:self.max_audio_len]
        
        # Load and encode text
        text_path = self.text_paths[idx]
        with open(text_path, 'r') as f:
            text = f.read().strip().upper()
        
        # Truncate text if needed
        if len(text) > self.max_text_len:
            text = text[:self.max_text_len]
        
        # Encode text (no SOS/EOS for CTC)
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            # Skip unknown characters
        
        text_tensor = torch.tensor(encoded, dtype=torch.long)
        
        return mel, text_tensor


def ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for CTC training.
    
    Returns format expected by torch.nn.CTCLoss:
        - audio_padded: (batch, max_time, n_mels)
        - audio_lengths: (batch,)
        - text_concat: (sum of all text lengths,) - concatenated targets
        - text_lengths: (batch,)
    """
    audios, texts = zip(*batch)
    
    # Get lengths
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
    
    # Pad audio
    audio_padded = pad_sequence(audios, batch_first=True)
    
    # Concatenate text (CTC loss expects this format)
    text_concat = torch.cat(texts)
    
    return audio_padded, audio_lengths, text_concat, text_lengths


def load_ctc_dataset(
    data_dir: str,
    config: dict,
    split_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Load dataset and create train/val dataloaders for CTC training.
    
    Args:
        data_dir: Directory with audio/ and text/ subdirectories
        config: Configuration dict
        split_ratio: Train/val split ratio
        
    Returns:
        train_loader, val_loader, char_to_idx, idx_to_char
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / 'audio'
    text_dir = data_dir / 'text'
    
    # Find paired audio/text files
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
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_ctc_vocab(config)
    print(f"Vocabulary size: {len(char_to_idx)} (including blank)")
    
    # Shuffle and split
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
    
    # Create augmenter for training
    audio_cfg = config.get('audio', {})
    sample_rate = audio_cfg.get('sample_rate', 16000)
    augmenter = create_augmenter_from_config(config, sample_rate)
    
    if augmenter:
        print("Audio augmentation: ENABLED")
    else:
        print("Audio augmentation: DISABLED")
    
    # Create datasets (augmentation only for training)
    train_dataset = CTCAudioTextDataset(
        train_audio, train_text, char_to_idx, config,
        augmenter=augmenter, is_training=True
    )
    val_dataset = CTCAudioTextDataset(
        val_audio, val_text, char_to_idx, config,
        augmenter=None, is_training=False  # No augmentation for validation
    )
    
    # Create dataloaders
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

