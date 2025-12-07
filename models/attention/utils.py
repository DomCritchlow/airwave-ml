"""
Utility functions for training and evaluation.
"""

import torch
import yaml
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, vocab, idx_to_char, config):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'vocab': vocab,
        'idx_to_char': idx_to_char,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    vocab = checkpoint.get('vocab', {})
    idx_to_char = checkpoint.get('idx_to_char', {})
    config = checkpoint.get('config', {})
    
    print(f"Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    
    return model, optimizer, scheduler, epoch, loss, vocab, idx_to_char, config


def decode_sequence(sequence, idx_to_char, remove_special=True):
    """
    Decode a sequence of token indices to text.
    
    Args:
        sequence: List or tensor of token indices
        idx_to_char: Dictionary mapping indices to characters
        remove_special: Whether to remove special tokens (<PAD>, <SOS>, <EOS>)
        
    Returns:
        Decoded text string
    """
    if torch.is_tensor(sequence):
        sequence = sequence.tolist()
    
    chars = []
    for idx in sequence:
        if idx in idx_to_char:
            char = idx_to_char[idx]
            if remove_special and char in ['<PAD>', '<SOS>', '<EOS>']:
                if char == '<EOS>':
                    break
                continue
            chars.append(char)
    
    return ''.join(chars)


def calculate_metrics(predictions, targets, idx_to_char):
    """
    Calculate accuracy metrics.
    
    Args:
        predictions: (batch, seq_len) predicted token indices
        targets: (batch, seq_len) target token indices
        idx_to_char: Index to character mapping
        
    Returns:
        Character-level accuracy
    """
    correct = 0
    total = 0
    
    for pred, tgt in zip(predictions, targets):
        pred_text = decode_sequence(pred, idx_to_char)
        tgt_text = decode_sequence(tgt, idx_to_char)
        
        # Character-level accuracy
        for p, t in zip(pred_text, tgt_text):
            if p == t:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def calculate_wer(predictions, targets, idx_to_char):
    """
    Calculate Word Error Rate.
    
    Args:
        predictions: (batch, seq_len)
        targets: (batch, seq_len)
        idx_to_char: Mapping
        
    Returns:
        WER (Word Error Rate)
    """
    total_words = 0
    total_errors = 0
    
    for pred, tgt in zip(predictions, targets):
        pred_text = decode_sequence(pred, idx_to_char)
        tgt_text = decode_sequence(tgt, idx_to_char)
        
        pred_words = pred_text.split()
        tgt_words = tgt_text.split()
        
        # Simple word-level comparison
        errors = sum(1 for p, t in zip(pred_words, tgt_words) if p != t)
        errors += abs(len(pred_words) - len(tgt_words))
        
        total_errors += errors
        total_words += len(tgt_words)
    
    wer = total_errors / total_words if total_words > 0 else 0.0
    return wer


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_directories(config):
    """Create necessary directories for checkpoints and logs."""
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {config['paths']['checkpoint_dir']}, {config['paths']['log_dir']}")


def setup_logging(config):
    """Setup tensorboard logging if enabled."""
    if config['logging']['use_tensorboard']:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(config['paths']['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
            return writer
        except ImportError:
            print("TensorBoard not available, logging disabled")
            return None
    return None


def log_metrics(writer, metrics, step, prefix='train'):
    """Log metrics to tensorboard."""
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'{prefix}/{key}', value, step)


def print_sample_predictions(model, val_loader, idx_to_char, device, num_samples=3):
    """
    Print sample predictions during validation.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        idx_to_char: Index to character mapping
        device: Device
        num_samples: Number of samples to print
    """
    model.eval()
    samples_printed = 0
    
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    with torch.no_grad():
        for audio, audio_lengths, text, text_lengths in val_loader:
            if samples_printed >= num_samples:
                break
            
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)
            
            # Get predictions for first few samples in batch
            batch_size = min(num_samples - samples_printed, audio.size(0))
            
            for i in range(batch_size):
                # Greedy decode
                pred_seq = model.greedy_decode(
                    audio[i:i+1],
                    audio_lengths[i:i+1],
                    max_len=200,
                    sos_token=1,
                    eos_token=2
                )
                
                # Decode
                pred_text = decode_sequence(pred_seq, idx_to_char)
                target_text = decode_sequence(text[i], idx_to_char)
                
                print(f"\nSample {samples_printed + 1}:")
                print(f"Target:     {target_text}")
                print(f"Prediction: {pred_text}")
                print("-" * 80)
                
                samples_printed += 1
                
                if samples_printed >= num_samples:
                    break
    
    print("="*80 + "\n")
    model.train()


def format_time(seconds):
    """Format seconds to human readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
