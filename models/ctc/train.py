#!/usr/bin/env python3
"""
Training script for CTC-based audio-to-text model.

Key differences from seq2seq training:
- Uses CTC loss (torch.nn.CTCLoss)
- No teacher forcing (encoder-only architecture)
- Different output format handling
- Simpler training loop

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Optimize CPU threading for multi-core
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)  # Use multiple cores
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import yaml

# Rich for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich import box

from model import CTCModel, create_ctc_model
from dataset import load_ctc_dataset, create_ctc_vocab

console = Console()


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
    scheduler,
    ctc_loss,
    device,
    epoch: int,
    progress,
    task_id
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
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        scheduler.step()  # OneCycleLR steps per batch
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress.update(task_id, advance=1, loss=f"{loss.item():.3f}")
    
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


def create_header_panel(config, num_params, device, epochs, learning_rate, patience):
    """Create the header panel with training info."""
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    
    info_table.add_row(
        "Device:", str(device),
        "Parameters:", f"{num_params:,}"
    )
    info_table.add_row(
        "Epochs:", str(epochs),
        "Learning Rate:", f"{learning_rate:.1e}"
    )
    info_table.add_row(
        "Batch Size:", str(config.get('training', {}).get('batch_size', 16)),
        "Early Stop:", f"{patience} epochs"
    )
    
    return Panel(
        info_table,
        title="[bold white]CTC Morse Decoder Training[/bold white]",
        border_style="blue",
        box=box.ROUNDED
    )


def create_metrics_table(history):
    """Create the metrics table."""
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Epoch", justify="right", style="dim")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("LR", justify="right", style="dim")
    table.add_column("Time", justify="right", style="dim")
    table.add_column("", justify="center")
    
    # Show last 10 epochs
    for h in history[-10:]:
        cer_style = "green" if h['cer'] < 0.3 else "yellow" if h['cer'] < 0.6 else "red"
        acc_style = "green" if h['acc'] > 70 else "yellow" if h['acc'] > 40 else "red"
        
        table.add_row(
            f"{h['epoch']}/{h['total']}",
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"[{cer_style}]{h['cer']:.2%}[/{cer_style}]",
            f"[{acc_style}]{h['acc']:.1f}%[/{acc_style}]",
            f"{h['lr']:.1e}",
            f"{h['time']:.0f}s",
            "✓" if h['is_best'] else ""
        )
    
    return table


def create_samples_panel(sample_preds, sample_targets):
    """Create the sample predictions panel."""
    samples = []
    for i in range(min(2, len(sample_preds))):
        target = sample_targets[i][:50]
        pred = sample_preds[i][:50]
        match = "✓" if target == pred else "✗"
        samples.append(f"[dim]Target:[/dim] [white]{target}[/white]")
        samples.append(f"[dim]Pred:[/dim]   [yellow]{pred}[/yellow] {match}")
        if i < min(2, len(sample_preds)) - 1:
            samples.append("")
    
    return Panel(
        "\n".join(samples) if samples else "[dim]No samples yet[/dim]",
        title="[bold]Sample Predictions[/bold]",
        border_style="dim",
        box=box.ROUNDED
    )


def main():
    parser = argparse.ArgumentParser(description='Train CTC audio-to-text model')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this run (used in logs and checkpoints)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.data_dir:
        config.setdefault('paths', {})['data_dir'] = args.data_dir
    
    # Run name for logging
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load data
    data_dir = config.get('paths', {}).get('data_dir', 'morse_synthetic')
    
    with console.status("[bold blue]Loading dataset...", spinner="dots"):
        train_loader, val_loader, char_to_idx, idx_to_char = load_ctc_dataset(
            data_dir, config
        )
    
    vocab_size = len(char_to_idx)
    
    # Create model
    with console.status("[bold blue]Creating model...", spinner="dots"):
        model = create_ctc_model(config, vocab_size)
        model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Training setup
    train_cfg = config.get('training', {})
    learning_rate = train_cfg.get('learning_rate', 0.0003)
    epochs = train_cfg.get('epochs', 100)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # OneCycleLR: warmup -> peak -> gradual decay (better for CTC)
    # pct_start=0.1 means 10% warmup, div_factor=25 means start at LR/25
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,           # 10% warmup
        div_factor=25,           # Start LR = max_lr/25
        final_div_factor=1000,   # End LR = max_lr/1000 (not too low)
        anneal_strategy='cos'
    )
    
    # CTC loss (blank_idx=0)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Checkpoint directory
    ckpt_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard setup
    log_dir = Path('runs') / f'ctc_{run_name}'
    writer = SummaryWriter(log_dir)
    writer.add_text('Config', f"```yaml\n{yaml.dump(config, default_flow_style=False)}```")
    
    # Resume from checkpoint
    start_epoch = 0
    best_cer = float('inf')
    
    if args.resume and Path(args.resume).exists():
        console.print(f"[yellow]Resuming from: {args.resume}[/yellow]")
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
    
    # Print header
    console.print()
    console.print(create_header_panel(config, num_params, device, epochs, learning_rate, patience))
    console.print(f"[dim]TensorBoard: tensorboard --logdir {log_dir.parent}[/dim]")
    console.print()
    
    # Training history
    history = []
    sample_preds = []
    sample_targets = []
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Training with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("loss: {task.fields[loss]}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                "Training",
                total=len(train_loader),
                epoch=epoch + 1,
                total_epochs=epochs,
                loss="--"
            )
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, ctc_loss, device, epoch, progress, task)
        
        # Validate
        val_loss, val_cer, sample_preds, sample_targets = validate(
            model, val_loader, ctc_loss, device, idx_to_char
        )
        
        # Get current learning rate (scheduler steps per batch now)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('Accuracy/val', (1 - val_cer) * 100, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Log sample predictions to TensorBoard
        samples_text = [f"**Target:** `{t[:60]}`  \n**Pred:** `{p[:60]}`" 
                       for t, p in zip(sample_targets[:3], sample_preds[:3])]
        if samples_text:
            writer.add_text('Predictions', '\n\n---\n\n'.join(samples_text), epoch)
        
        # Check if best
        is_best = val_cer < best_cer - min_delta
        
        # Save checkpoint
        if is_best:
            best_cer = val_cer
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / 'best_model_ctc.pt'
            )
        else:
            epochs_no_improve += 1
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_loss, val_cer, config, char_to_idx, idx_to_char,
                ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            )
        
        # Add to history
        history.append({
            'epoch': epoch + 1,
            'total': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'cer': val_cer,
            'acc': (1 - val_cer) * 100,
            'lr': current_lr,
            'time': epoch_time,
            'is_best': is_best
        })
        
        # Display metrics table and samples
        console.print(create_metrics_table(history))
        console.print(create_samples_panel(sample_preds, sample_targets))
        console.print()
        
        # Early stopping
        if epochs_no_improve >= patience:
            console.print(f"[yellow]Early stopping after {epoch + 1} epochs[/yellow]")
            break
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Close TensorBoard writer
    writer.add_hparams(
        {
            'hidden_dim': config['model']['hidden_dim'],
            'num_layers': config['model']['num_layers'],
            'lr': learning_rate,
            'batch_size': train_cfg.get('batch_size', 16),
        },
        {
            'hparam/best_cer': best_cer,
            'hparam/best_accuracy': (1 - best_cer) * 100,
        }
    )
    writer.close()
    
    # Final summary
    summary = Table(show_header=False, box=None)
    summary.add_column(style="cyan")
    summary.add_column(style="white")
    summary.add_row("Total Time", f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    summary.add_row("Best CER", f"{best_cer:.2%}")
    summary.add_row("Best Accuracy", f"{(1 - best_cer) * 100:.1f}%")
    summary.add_row("Checkpoints", str(ckpt_dir))
    summary.add_row("TensorBoard", str(log_dir))
    
    console.print(Panel(
        summary,
        title="[bold green]Training Complete[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))


if __name__ == '__main__':
    main()
