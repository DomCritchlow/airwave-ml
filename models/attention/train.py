"""
Training script for audio-to-text model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
from pathlib import Path
from datetime import datetime
import time
import os
from tqdm import tqdm

# Optimize CPU threading for Intel Macs
torch.set_num_threads(4)  # Use all 4 cores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from dataset import load_dataset, get_dataloaders
from model import AudioToTextModel, count_parameters
from utils import (
    load_config, save_config, save_checkpoint, load_checkpoint,
    AverageMeter, get_lr, create_directories, setup_logging,
    log_metrics, log_hparams, print_sample_predictions, format_time,
    decode_sequence, calculate_wer
)


def train_epoch(model, train_loader, optimizer, criterion, device, config, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (audio, audio_lengths, text, text_lengths) in enumerate(pbar):
        # Move to device
        audio = audio.to(device)
        audio_lengths = audio_lengths.to(device)
        text = text.to(device)
        text_lengths = text_lengths.to(device)
        
        # Prepare input and target for teacher forcing
        # Input: <SOS> + text[:-1]
        # Target: text[1:] (shifted by one)
        tgt_input = text[:, :-1]
        tgt_output = text[:, 1:]
        tgt_input_lengths = text_lengths - 1
        
        # Forward pass
        optimizer.zero_grad()
        output = model(audio, audio_lengths, tgt_input, tgt_input_lengths)
        
        # Calculate loss (ignore padding)
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = output.argmax(dim=-1)
        mask = tgt_output != 0  # Ignore padding
        correct = (predictions == tgt_output) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        
        # Update meters
        losses.update(loss.item(), audio.size(0))
        accuracies.update(accuracy, audio.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, idx_to_char):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for audio, audio_lengths, text, text_lengths in tqdm(val_loader, desc="Validating"):
            # Move to device
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            
            # Prepare input and target
            tgt_input = text[:, :-1]
            tgt_output = text[:, 1:]
            tgt_input_lengths = text_lengths - 1
            
            # Forward pass
            output = model(audio, audio_lengths, tgt_input, tgt_input_lengths)
            
            # Calculate loss
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            mask = tgt_output != 0
            correct = (predictions == tgt_output) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            
            # Store for WER calculation
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(tgt_output.cpu().tolist())
            
            # Update meters
            losses.update(loss.item(), audio.size(0))
            accuracies.update(accuracy, audio.size(0))
    
    # Calculate WER
    wer = calculate_wer(all_predictions, all_targets, idx_to_char)
    
    return losses.avg, accuracies.avg, wer


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create directories
    create_directories(config)
    
    # Save config
    save_config(config, Path(config['paths']['checkpoint_dir']) / 'config.yaml')
    
    # Set device (auto-detect best available)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset, val_dataset, vocab, idx_to_char = load_dataset(
        config['paths']['data_dir'],
        config,
        split_ratio=config['data']['train_split']
    )
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, config)
    
    # Get input dimension from first sample
    sample_audio, _ = train_dataset[0]
    input_dim = sample_audio.shape[1]
    print(f"Audio feature dimension: {input_dim}")
    
    # Create model
    print("\nCreating model...")
    model = AudioToTextModel(
        vocab_size=len(vocab),
        input_dim=input_dim,
        config=config
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer (label smoothing helps prevent mode collapse)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Early stopping config
    early_stop_config = config.get('early_stopping', {})
    early_stop_enabled = early_stop_config.get('enabled', False)
    early_stop_patience = early_stop_config.get('patience', 10)
    early_stop_min_delta = early_stop_config.get('min_delta', 0.001)
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model, optimizer, scheduler, start_epoch, best_val_loss, _, _, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1
    
    # Setup logging
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    writer, log_dir = setup_logging(config, run_name)
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch
        )
        
        # Validate
        val_loss, val_acc, val_wer = validate(model, val_loader, criterion, device, idx_to_char)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['training']['epochs']-1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val WER: {val_wer:.4f}")
        print(f"LR: {get_lr(optimizer):.6f} | Time: {format_time(epoch_time)}")
        
        # Log to tensorboard
        if writer:
            log_metrics(writer, {
                'loss': train_loss,
                'accuracy': train_acc,
                'lr': get_lr(optimizer)
            }, epoch, prefix='train')
            
            log_metrics(writer, {
                'loss': val_loss,
                'accuracy': val_acc,
                'wer': val_wer
            }, epoch, prefix='val')
        
        # Print sample predictions
        if (epoch + 1) % 5 == 0:
            print_sample_predictions(model, val_loader, idx_to_char, device, num_samples=3)
        
        # Save checkpoint
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                save_path, vocab, idx_to_char, config
            )
        
        # Save best model and check for early stopping
        if config['checkpoint']['save_best'] and val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                save_path, vocab, idx_to_char, config
            )
            print(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if early_stop_enabled and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
        
        print("=" * 80)
    
    total_time = time.time() - start_time
    
    # Log hyperparameters
    log_hparams(writer, config, best_val_loss, None, epoch + 1, total_time)
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {Path(config['paths']['checkpoint_dir']).absolute()}")
    if log_dir:
        print(f"TensorBoard: {log_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train attention-based audio-to-text model')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this run (used in logs)')
    
    args = parser.parse_args()
    main(args)
