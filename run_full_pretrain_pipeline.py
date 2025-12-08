#!/usr/bin/env python3
"""
Full Pretraining + Fine-tuning Pipeline

Runs the complete pipeline:
1. Masked spectrogram pretraining (self-supervised)
2. CTC fine-tuning (supervised)

Usage:
    python run_full_pretrain_pipeline.py
    
    # Or with custom settings:
    python run_full_pretrain_pipeline.py --pretrain-epochs 50 --finetune-epochs 50
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and stream output."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f"\n❌ {description} FAILED (exit code {process.returncode})")
        return False
    
    print(f"\n✓ {description} completed in {elapsed/60:.1f} minutes")
    return True


def main():
    parser = argparse.ArgumentParser(description='Full Pretraining Pipeline')
    parser.add_argument('--pretrain-epochs', type=int, default=50,
                       help='Number of pretraining epochs')
    parser.add_argument('--finetune-epochs', type=int, default=50,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                       help='Epochs to freeze encoder during fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pretraining (use existing checkpoint)')
    parser.add_argument('--test-only', action='store_true',
                       help='Quick test run (1 epoch each)')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).parent
    pretrain_script = repo_root / "pretrain_masked_ctc_w_pretrain.py"
    finetune_script = repo_root / "models" / "ctc_w_pretrain" / "train.py"
    config_file = repo_root / "models" / "ctc_w_pretrain" / "config.yaml"
    checkpoint_dir = repo_root / "models" / "ctc_w_pretrain" / "checkpoints"
    pretrained_path = checkpoint_dir / "pretrained_encoder.pt"
    
    # Data directories
    data_dirs = [
        str(repo_root / "data" / "synthetic" / "morse_v2" / "audio"),
        str(repo_root / "data" / "real_world" / "morse_data" / "chunked" / "audio"),
    ]
    
    # Filter to existing directories
    data_dirs = [d for d in data_dirs if Path(d).exists()]
    
    if not data_dirs:
        print("❌ No data directories found!")
        sys.exit(1)
    
    # Test mode: 1 epoch each
    if args.test_only:
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.freeze_epochs = 0
        print("🧪 TEST MODE: Running 1 epoch each")
    
    print("\n" + "=" * 70)
    print("  FULL PRETRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Fine-tune epochs: {args.finetune_epochs}")
    print(f"Data dirs: {data_dirs}")
    print(f"Checkpoint: {pretrained_path}")
    
    total_start = time.time()
    
    # Step 1: Pretraining
    if not args.skip_pretrain:
        pretrain_cmd = [
            sys.executable, str(pretrain_script),
            "--data-dirs", *data_dirs,
            "--epochs", str(args.pretrain_epochs),
            "--batch-size", str(args.batch_size),
            "--save-path", str(pretrained_path),
        ]
        
        if not run_command(pretrain_cmd, "STEP 1: Masked Spectrogram Pretraining"):
            print("\n❌ Pipeline failed at pretraining step")
            sys.exit(1)
    else:
        print(f"\n⏭ Skipping pretraining, using existing: {pretrained_path}")
    
    # Check pretrained checkpoint exists
    if not pretrained_path.exists():
        print(f"\n❌ Pretrained checkpoint not found: {pretrained_path}")
        sys.exit(1)
    
    # Step 2: Fine-tuning
    finetune_cmd = [
        sys.executable, str(finetune_script),
        "--config", str(config_file),
        "--pretrained-encoder", str(pretrained_path),
        "--freeze-epochs", str(args.freeze_epochs),
    ]
    
    if not run_command(finetune_cmd, "STEP 2: CTC Fine-tuning"):
        print("\n❌ Pipeline failed at fine-tuning step")
        sys.exit(1)
    
    total_time = time.time() - total_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Pretrained encoder: {pretrained_path}")
    print(f"Final model: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 70)


if __name__ == '__main__':
    main()

