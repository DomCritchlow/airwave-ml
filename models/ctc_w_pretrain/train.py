#!/usr/bin/env python3
"""
CTC Training with Pretrained Encoder Support

This training script is for the ctc_w_pretrain variant that supports
loading pretrained encoder weights from masked spectrogram pretraining.

Usage:
    # Train from scratch
    python train.py --config config.yaml
    
    # Train with pretrained encoder
    python train.py --config config.yaml --pretrained-encoder checkpoints/pretrained_encoder.pt
    
    # Freeze encoder for first N epochs
    python train.py --config config.yaml --pretrained-encoder checkpoints/pretrained_encoder.pt --freeze-epochs 5
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Optimize CPU threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import yaml

from model import CTCModel, create_ctc_model
from dataset import load_ctc_dataset, create_ctc_vocab

# Try to import rich
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


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
    """Calculate Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
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


def train_epoch(model, train_loader, optimizer, scheduler, ctc_loss, device, epoch, progress=None, task=None):
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
        
        log_probs, output_lengths = model(audio, audio_lengths)
        log_probs = log_probs.transpose(0, 1)
        
        loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if progress and task is not None:
            progress.update(task, advance=1, loss=f"{loss.item():.3f}")
    
    return total_loss / max(num_batches, 1)


def validate(model, val_loader, ctc_loss, device, idx_to_char):
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
            
            log_probs, output_lengths = model(audio, audio_lengths)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss(log_probs_t, targets, output_lengths, target_lengths)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            pred_seqs = model.decode_greedy(audio, audio_lengths)
            
            target_offset = 0
            for i, length in enumerate(target_lengths):
                target_seq = targets[target_offset:target_offset + length]
                target_offset += length
                
                pred_text = decode_prediction(pred_seqs[i], idx_to_char)
                target_text = decode_target(target_seq, idx_to_char)
                
                cer = calculate_cer(pred_text, target_text)
                total_cer += cer
                num_samples += 1
                
                if len(sample_preds) < 5:
                    sample_preds.append(pred_text)
                    sample_targets.append(target_text)
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_cer = total_cer / max(num_samples, 1)
    
    return avg_loss, avg_cer, sample_preds, sample_targets


def save_checkpoint(model, optimizer, scheduler, epoch, loss, cer, config, vocab, idx_to_char, path):
    """Save checkpoint."""
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
        'model_type': 'ctc_w_pretrain'
    }, path)


def main():
    parser = argparse.ArgumentParser(description='CTC Training with Pretrained Encoder')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained-encoder', type=str, default=None,
                       help='Path to pretrained encoder checkpoint')
    parser.add_argument('--freeze-epochs', type=int, default=0,
                       help='Number of epochs to freeze encoder')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this run (used in checkpoint filenames)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config for tiny model
        config = {
            'model': {
                'hidden_dim': 128,
                'num_layers': 2,
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
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.0005,
                'num_workers': 0
            },
            'data': {
                'max_audio_len': 2000,
                'max_text_len': 200
            },
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.002
            },
            'paths': {
                'data_dir': '../../data/synthetic/morse_v2',
                'checkpoint_dir': 'checkpoints'
            },
            'vocab': ' -0123456789?ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        }
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data (allow command-line override)
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config.get('paths', {}).get('data_dir', '../../data/synthetic/morse_v2')
    print(f"\nLoading data from: {data_dir}")
    
    train_loader, val_loader, char_to_idx, idx_to_char = load_ctc_dataset(data_dir, config)
    vocab_size = len(char_to_idx)
    
    # Create model
    print("\nCreating CTC model (tiny variant)...")
    model = create_ctc_model(config, vocab_size)
    
    # Load pretrained encoder if provided
    if args.pretrained_encoder and Path(args.pretrained_encoder).exists():
        print(f"\nLoading pretrained encoder from: {args.pretrained_encoder}")
        pretrained = torch.load(args.pretrained_encoder, map_location='cpu')
        
        # Load frontend and encoder weights
        if 'frontend' in pretrained:
            model.frontend.load_state_dict(pretrained['frontend'], strict=False)
            print("  ✓ Frontend weights loaded")
        if 'encoder' in pretrained:
            model.encoder.load_state_dict(pretrained['encoder'], strict=False)
            print("  ✓ Encoder weights loaded")
        
        pretrain_loss = pretrained.get('val_loss', 'N/A')
        print(f"  Pretrain validation loss: {pretrain_loss}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Training setup
    train_cfg = config.get('training', {})
    learning_rate = train_cfg.get('learning_rate', 0.0005)
    epochs = train_cfg.get('epochs', 50)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Checkpoint directory
    ckpt_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Run name for checkpoints
    if args.run_name:
        run_name = args.run_name
        model_name = f'best_model_{run_name}.pt'
    else:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = 'best_model.pt'
    
    # TensorBoard
    log_dir = Path('runs') / f'ctc_{run_name}'
    writer = SummaryWriter(log_dir)
    
    # Log config and run info
    writer.add_text('Config', f"```yaml\n{yaml.dump(config, default_flow_style=False)}```")
    writer.add_text('RunInfo', f"""
**Run Name:** {run_name}
**Data Dir:** {data_dir}
**Pretrained Encoder:** {args.pretrained_encoder or 'None'}
**Resume From:** {args.resume or 'None'}
**Freeze Epochs:** {args.freeze_epochs}
""")
    
    # Resume
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
    min_delta = config.get('early_stopping', {}).get('min_delta', 0.002)
    epochs_no_improve = 0
    
    # Freeze encoder if specified
    encoder_frozen = False
    if args.freeze_epochs > 0 and args.pretrained_encoder:
        print(f"\nFreezing encoder for first {args.freeze_epochs} epochs")
        for param in model.frontend.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
    
    print("\n" + "=" * 60)
    print("STARTING CTC TRAINING (with pretrain support)")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Pretrained encoder: {'Yes' if args.pretrained_encoder else 'No'}")
    print(f"Freeze epochs: {args.freeze_epochs}")
    print("=" * 60 + "\n")
    
    history = []
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Unfreeze encoder after freeze_epochs
        if encoder_frozen and epoch >= args.freeze_epochs:
            print(f"\nUnfreezing encoder at epoch {epoch + 1}")
            for param in model.frontend.parameters():
                param.requires_grad = True
            for param in model.encoder.parameters():
                param.requires_grad = True
            encoder_frozen = False
        
        # Train
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]Epoch {epoch + 1}/{epochs}"),
                BarColumn(bar_width=30),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("• loss: {task.fields[loss]}"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Training", total=len(train_loader), loss="--")
                train_loss = train_epoch(
                    model, train_loader, optimizer, scheduler, ctc_loss, 
                    device, epoch, progress, task
                )
        else:
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, ctc_loss, 
                device, epoch, None, None
            )
        
        # Validate
        val_loss, val_cer, sample_preds, sample_targets = validate(
            model, val_loader, ctc_loss, device, idx_to_char
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('Accuracy/val', (1 - val_cer) * 100, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)
        
        # Log sample predictions to TensorBoard
        if sample_preds and sample_targets:
            samples_text = [
                f"**Target:** `{t[:60]}`  \n**Pred:** `{p[:60]}`"
                for t, p in zip(sample_targets[:3], sample_preds[:3])
            ]
            writer.add_text('Predictions', '\n\n---\n\n'.join(samples_text), epoch)
        
        # Check best
        is_best = val_cer < best_cer - min_delta
        
        if is_best:
            best_cer = val_cer
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / model_name
            )
        else:
            epochs_no_improve += 1
        
        # History
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'cer': val_cer,
            'acc': (1 - val_cer) * 100,
            'lr': current_lr,
            'time': epoch_time,
            'is_best': is_best,
            'frozen': encoder_frozen
        })
        
        # Print progress
        if RICH_AVAILABLE:
            table = Table(box=box.SIMPLE)
            table.add_column("Epoch", justify="right")
            table.add_column("Train", justify="right")
            table.add_column("Val", justify="right")
            table.add_column("CER", justify="right")
            table.add_column("Acc", justify="right")
            table.add_column("LR", justify="right")
            table.add_column("", justify="center")
            
            for h in history[-5:]:
                cer_style = "green" if h['cer'] < 0.2 else "yellow" if h['cer'] < 0.5 else "red"
                frozen_mark = "❄" if h['frozen'] else ""
                table.add_row(
                    str(h['epoch']),
                    f"{h['train_loss']:.4f}",
                    f"{h['val_loss']:.4f}",
                    f"[{cer_style}]{h['cer']:.2%}[/{cer_style}]",
                    f"{h['acc']:.1f}%",
                    f"{h['lr']:.1e}",
                    f"{'✓' if h['is_best'] else ''}{frozen_mark}"
                )
            
            console.print(table)
            
            # Sample predictions
            if sample_preds:
                console.print(Panel(
                    f"[dim]Target:[/dim] {sample_targets[0][:50]}\n"
                    f"[dim]Pred:[/dim]   {sample_preds[0][:50]}",
                    title="Sample Prediction",
                    border_style="dim"
                ))
        else:
            mark = "✓ BEST" if is_best else ""
            frozen = " (frozen)" if encoder_frozen else ""
            print(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"cer={val_cer:.2%}, acc={(1-val_cer)*100:.1f}% {mark}{frozen}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            )
    
    total_time = time.time() - start_time
    
    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            'hidden_dim': config['model']['hidden_dim'],
            'num_layers': config['model']['num_layers'],
            'lr': train_cfg.get('learning_rate', 0.0005),
            'batch_size': train_cfg.get('batch_size', 32),
            'pretrained': args.pretrained_encoder is not None,
            'data_dir': data_dir,
        },
        {
            'hparam/best_cer': best_cer,
            'hparam/best_accuracy': (1 - best_cer) * 100,
            'hparam/total_epochs': epoch + 1,
            'hparam/total_time_min': total_time / 60,
        }
    )
    writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best CER: {best_cer:.2%} ({(1 - best_cer) * 100:.1f}% accuracy)")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"TensorBoard: {log_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

