#!/usr/bin/env python3
"""
Masked Spectrogram Pretraining for CTC Encoder

This script implements self-supervised pretraining using masked spectrogram prediction,
similar to BERT/wav2vec2 but for mel spectrograms.

The model learns to predict masked time-frames of the spectrogram, forcing it to
learn useful audio representations before fine-tuning on labeled Morse code data.

Usage:
    python pretrain_masked_ctc_w_pretrain.py \
        --data-dirs data/synthetic/morse_v2/audio data/real_world/morse_data/chunked/audio \
        --epochs 50 \
        --save-path models/ctc_w_pretrain/checkpoints/pretrained_encoder.pt
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import yaml

# Add model path
sys.path.insert(0, str(Path(__file__).parent / 'models' / 'ctc_w_pretrain'))

from models.ctc_w_pretrain.model import (
    ConvFrontend, 
    CTCEncoder, 
    TinyEncoderConfig,
    create_tiny_encoder
)
from models.ctc_w_pretrain.dataset import load_unlabeled_dataset, find_audio_files

# Try to import rich for nice output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class MaskedSpecModel(nn.Module):
    """
    Masked Spectrogram Prediction Model for pretraining.
    
    Architecture:
        Mel Spectrogram → Mask → ConvFrontend → Encoder → Decoder → Reconstruct masked frames
    
    The model learns by predicting the original mel values at masked positions.
    """
    
    def __init__(
        self,
        frontend: ConvFrontend,
        encoder: CTCEncoder,
        n_mels: int = 80,
        mask_prob: float = 0.15,
        mask_length: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.frontend = frontend
        self.encoder = encoder
        self.n_mels = n_mels
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.hidden_dim = hidden_dim
        
        # Learnable mask token (replaces masked positions in input)
        self.mask_token = nn.Parameter(torch.randn(n_mels))
        
        # Decoder to predict original mel values
        # Maps from encoder output back to mel-spectrogram space
        # Note: Frontend reduces time by 4x, so decoder needs to upsample
        self.decoder = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_mels * 4),  # 4x for upsampling
        )
    
    def create_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Create contiguous span masks along time axis.
        
        Args:
            mel: (batch, time, n_mels)
            
        Returns:
            mask: (batch, time) boolean tensor, True = masked
        """
        batch_size, time_steps, _ = mel.shape
        mask = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=mel.device)
        
        for b in range(batch_size):
            # Calculate number of spans to mask
            num_mask = int(time_steps * self.mask_prob / self.mask_length)
            num_mask = max(1, num_mask)
            
            # Random starting positions for spans
            for _ in range(num_mask):
                start = random.randint(0, max(0, time_steps - self.mask_length))
                end = min(start + self.mask_length, time_steps)
                mask[b, start:end] = True
        
        return mask
    
    def apply_mask(self, mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Replace masked positions with learnable mask token.
        
        Args:
            mel: (batch, time, n_mels)
            mask: (batch, time) boolean
            
        Returns:
            masked_mel: (batch, time, n_mels)
        """
        masked_mel = mel.clone()
        # Expand mask to match mel dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(mel)
        masked_mel[mask_expanded] = self.mask_token.expand_as(mel)[mask_expanded]
        return masked_mel
    
    def forward(
        self, 
        mel: torch.Tensor, 
        lengths: torch.Tensor
    ) -> tuple:
        """
        Forward pass with masking.
        
        Args:
            mel: (batch, time, n_mels) input spectrogram
            lengths: (batch,) sequence lengths
            
        Returns:
            loss: MSE loss on masked positions
            predictions: (batch, time, n_mels) reconstructed spectrogram
            mask: (batch, time) mask used
        """
        batch_size, time_steps, n_mels = mel.shape
        
        # Create and apply mask
        mask = self.create_mask(mel)
        masked_mel = self.apply_mask(mel, mask)
        
        # Encode masked input
        frontend_out = self.frontend(masked_mel)  # (batch, time//4, hidden)
        encoder_out = self.encoder(frontend_out)   # (batch, time//4, hidden)
        
        # Decode to predict original mel values
        decoded = self.decoder(encoder_out)  # (batch, time//4, n_mels*4)
        
        # Reshape to match original time dimension
        decoded = decoded.view(batch_size, -1, n_mels)  # (batch, time, n_mels)
        
        # Handle length mismatch (due to conv strides)
        if decoded.size(1) < time_steps:
            # Pad decoded output
            pad_size = time_steps - decoded.size(1)
            decoded = torch.nn.functional.pad(decoded, (0, 0, 0, pad_size))
        elif decoded.size(1) > time_steps:
            # Truncate decoded output
            decoded = decoded[:, :time_steps, :]
        
        # Calculate loss only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(mel)
        
        if mask_expanded.any():
            loss = torch.nn.functional.mse_loss(
                decoded[mask_expanded],
                mel[mask_expanded]
            )
        else:
            loss = torch.tensor(0.0, device=mel.device)
        
        return loss, decoded, mask
    
    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for loading into CTC model."""
        return {
            'frontend': self.frontend.state_dict(),
            'encoder': self.encoder.state_dict()
        }


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, use_rich=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if use_rich and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• loss: {task.fields[loss]}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                "Training",
                total=len(train_loader),
                epoch=epoch,
                loss="--"
            )
            
            for batch_idx, (mel, lengths) in enumerate(train_loader):
                mel = mel.to(device)
                lengths = lengths.to(device)
                
                optimizer.zero_grad()
                loss, _, _ = model(mel, lengths)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress.update(task, advance=1, loss=f"{loss.item():.4f}")
    else:
        for batch_idx, (mel, lengths) in enumerate(train_loader):
            mel = mel.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model(mel, lengths)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for mel, lengths in val_loader:
            mel = mel.to(device)
            lengths = lengths.to(device)
            
            loss, _, _ = model(mel, lengths)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Masked Spectrogram Pretraining')
    parser.add_argument('--data-dirs', type=str, nargs='+', required=True,
                       help='Directories containing audio files')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, 
                       default='models/ctc_w_pretrain/checkpoints/pretrained_encoder.pt',
                       help='Path to save pretrained encoder')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension of encoder')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of encoder layers')
    parser.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of masking each timestep')
    parser.add_argument('--mask-length', type=int, default=10,
                       help='Length of masked spans')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Config
    config = {
        'audio': {
            'sample_rate': 16000,
            'n_mels': 80,
            'n_fft': 400,
            'hop_length': 160
        },
        'training': {
            'batch_size': args.batch_size,
            'num_workers': 0
        },
        'data': {
            'max_audio_len': 2000
        },
        'augmentation': {
            'enabled': True,
            'prob_noise': 0.3,
            'prob_volume': 0.2,
        }
    }
    
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading audio from: {args.data_dirs}")
    train_loader, val_loader = load_unlabeled_dataset(args.data_dirs, config)
    
    # Create model
    print("\nCreating masked spectrogram model...")
    
    encoder_config = TinyEncoderConfig(
        input_dim=80,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
        encoder_type="transformer",
        nhead=4
    )
    
    frontend, encoder = create_tiny_encoder(encoder_config)
    
    model = MaskedSpecModel(
        frontend=frontend,
        encoder=encoder,
        n_mels=80,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        hidden_dim=args.hidden_dim
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Training
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold]Masked Spectrogram Pretraining[/bold]\n\n"
            f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val\n"
            f"Epochs: {args.epochs}\n"
            f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}\n"
            f"Mask prob: {args.mask_prob}, Mask length: {args.mask_length}",
            border_style="blue"
        ))
    else:
        print("\n" + "=" * 50)
        print("MASKED SPECTROGRAM PRETRAINING")
        print("=" * 50)
        print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
        print(f"Epochs: {args.epochs}")
        print(f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    
    best_val_loss = float('inf')
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, epoch, use_rich=RICH_AVAILABLE
        )
        val_loss = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'time': epoch_time,
            'is_best': is_best
        })
        
        # Print progress
        if RICH_AVAILABLE:
            table = Table(box=box.SIMPLE)
            table.add_column("Epoch", justify="right")
            table.add_column("Train Loss", justify="right")
            table.add_column("Val Loss", justify="right")
            table.add_column("LR", justify="right")
            table.add_column("Time", justify="right")
            table.add_column("", justify="center")
            
            for h in history[-5:]:
                style = "green" if h['is_best'] else ""
                table.add_row(
                    str(h['epoch']),
                    f"{h['train_loss']:.4f}",
                    f"{h['val_loss']:.4f}",
                    f"{h['lr']:.1e}",
                    f"{h['time']:.0f}s",
                    "✓" if h['is_best'] else ""
                )
            
            console.print(table)
        else:
            mark = "✓ BEST" if is_best else ""
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"lr={current_lr:.1e}, time={epoch_time:.0f}s {mark}")
        
        # Save best model
        if is_best:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save only encoder weights
            torch.save({
                'frontend': model.frontend.state_dict(),
                'encoder': model.encoder.state_dict(),
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'n_mels': 80,
                },
                'epoch': epoch,
                'val_loss': val_loss,
            }, save_path)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("PRETRAINING COMPLETE")
    print("=" * 50)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Encoder saved to: {args.save_path}")
    print("\nTo use for CTC fine-tuning:")
    print(f"  python models/ctc_w_pretrain/train.py --pretrained-encoder {args.save_path}")


if __name__ == '__main__':
    main()

