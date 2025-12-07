#!/usr/bin/env python3
"""
Training script for CTC-based audio-to-text model.

Key differences from seq2seq training:
- Uses CTC loss (torch.nn.CTCLoss)
- No teacher forcing (encoder-only architecture)
- Different output format handling
- Simpler training loop

Usage:
    python src_ctc/train.py --config config_ctc.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from model import CTCModel, create_ctc_model
from dataset import load_ctc_dataset, create_ctc_vocab


def decode_prediction(pred_tokens: list, idx_to_char: dict) -> str:
    """Convert token list to string."""
    chars = []
    for idx in pred_tokens:
        if idx in idx_to_char and idx_to_char[idx] != '<BLANK>':
            chars.append(idx_to_char[idx])
    return ''.join(chars)


def decode_target(target_tokens: torch.Tensor, idx_to_char: dict) -> str:
    """Convert target tensor to string."""
    chars = []
    for idx in target_tokens.tolist():
        if idx in idx_to_char and idx_to_char[idx] != '<BLANK>':
            chars.append(idx_to_char[idx])
    return ''.join(chars)


def calculate_cer(pred: str, target: str) -> float:
    """Calculate Character Error Rate using edit distance."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Simple Levenshtein distance
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(target)


def train_epoch(
    model: CTCModel,
    train_loader,
    optimizer,
    ctc_loss,
    device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (audio, audio_lengths, targets, target_lengths) in enumerate(train_loader):
        audio = audio.to(device)
        audio_lengths = audio_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        log_probs, output_lengths = model(audio, audio_lengths)
        
        # CTC loss expects: (T, N, C) for log_probs
        log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab)
        
        # Compute loss
        loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress update
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


def validate(
    model: CTCModel,
    val_loader,
    ctc_loss,
    device,
    idx_to_char: dict
) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_batches = 0
    num_samples = 0
    
    sample_preds = []
    sample_targets = []
    
    with torch.no_grad():
        for audio, audio_lengths, targets, target_lengths in val_loader:
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs, output_lengths = model(audio, audio_lengths)
            
            # CTC loss
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss(log_probs_t, targets, output_lengths, target_lengths)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            # Decode predictions for CER
            pred_seqs = model.decode_greedy(audio, audio_lengths)
            
            # Split targets back into individual sequences
            target_offset = 0
            for i, length in enumerate(target_lengths):
                target_seq = targets[target_offset:target_offset + length]
                target_offset += length
                
                pred_text = decode_prediction(pred_seqs[i], idx_to_char)
                target_text = decode_target(target_seq, idx_to_char)
                
                cer = calculate_cer(pred_text, target_text)
                total_cer += cer
                num_samples += 1
                
                # Save samples for display
                if len(sample_preds) < 5:
                    sample_preds.append(pred_text)
                    sample_targets.append(target_text)
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_cer = total_cer / max(num_samples, 1)
    
    return avg_loss, avg_cer, sample_preds, sample_targets


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    cer,
    config,
    vocab,
    idx_to_char,
    path
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'cer': cer,
        'config': config,
        'vocab': vocab,
        'idx_to_char': idx_to_char,
        'model_type': 'ctc'  # Important: identifies this as CTC model
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train CTC audio-to-text model')
    parser.add_argument('--config', '-c', type=str, default='config_ctc.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Creating default CTC config...")
        # Will be created below
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1,
                'encoder_type': 'transformer',
                'nhead': 4
            },
            'audio': {
                'sample_rate': 16000,
                'n_mels': 80,
                'n_fft': 400,
                'hop_length': 160
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.0003,
                'num_workers': 0
            },
            'data': {
                'max_audio_len': 2000,
                'max_text_len': 200
            },
            'paths': {
                'data_dir': 'morse_synthetic',
                'checkpoint_dir': 'checkpoints_ctc'
            },
            'vocab': ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        }
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    data_dir = config.get('paths', {}).get('data_dir', 'morse_synthetic')
    print(f"\nLoading data from: {data_dir}")
    
    train_loader, val_loader, char_to_idx, idx_to_char = load_ctc_dataset(
        data_dir, config
    )
    
    vocab_size = len(char_to_idx)
    print(f"Vocab size: {vocab_size}")
    
    # Create model
    print("\nCreating CTC model...")
    model = create_ctc_model(config, vocab_size)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Training setup
    train_cfg = config.get('training', {})
    learning_rate = train_cfg.get('learning_rate', 0.0003)
    epochs = train_cfg.get('epochs', 100)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # CTC loss (blank_idx=0)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Checkpoint directory
    ckpt_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints_ctc'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_cer = float('inf')
    
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cer = checkpoint.get('cer', float('inf'))
    
    # Early stopping
    patience = config.get('early_stopping', {}).get('patience', 10)
    min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
    epochs_no_improve = 0
    
    print("\n" + "=" * 60)
    print("STARTING CTC TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {train_cfg.get('batch_size', 16)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device, epoch)
        
        # Validate
        val_loss, val_cer, sample_preds, sample_targets = validate(
            model, val_loader, ctc_loss, device, idx_to_char
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val CER: {val_cer:.4f} ({(1 - val_cer) * 100:.1f}% accuracy)")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Sample predictions
        print("\nSample Predictions:")
        for i in range(min(3, len(sample_preds))):
            print(f"  Target: {sample_targets[i][:50]}")
            print(f"  Pred:   {sample_preds[i][:50]}")
            print()
        
        # Save checkpoint
        is_best = val_cer < best_cer - min_delta
        
        if is_best:
            best_cer = val_cer
            epochs_no_improve = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / 'best_model_ctc.pt'
            )
            print(f"✓ New best model saved (CER: {val_cer:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            )
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        print()
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Best CER: {best_cer:.4f} ({(1 - best_cer) * 100:.1f}% accuracy)")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

