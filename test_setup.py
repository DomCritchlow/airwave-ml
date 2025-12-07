"""
Simple test script to verify the installation and setup.
Tests both attention and CTC model architectures.
"""

import torch
import sys
from pathlib import Path


def test_dependencies():
    """Test that all dependencies are installed."""
    print("Testing dependencies...")
    
    required = [
        'torch',
        'torchaudio',
        'numpy',
        'yaml',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies installed!\n")
    return True


def test_attention_model():
    """Test attention-based seq2seq model creation."""
    print("Testing Attention model (models/attention/)...")
    
    try:
        # Add path
        sys.path.insert(0, str(Path(__file__).parent / 'models' / 'attention'))
        
        from model import AudioToTextModel
        from utils import load_config
        
        # Load config
        config_path = Path(__file__).parent / 'models' / 'attention' / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            # Default config
            config = {
                'model': {'d_model': 192, 'nhead': 6, 'num_encoder_layers': 3, 
                         'num_decoder_layers': 3, 'dim_feedforward': 768, 'dropout': 0.15},
                'audio': {'n_mels': 80}
            }
        
        # Create model
        model = AudioToTextModel(vocab_size=40, input_dim=80, config=config)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ Model created: {num_params:,} parameters")
        
        # Test forward pass
        src = torch.randn(2, 100, 80)
        src_lengths = torch.tensor([100, 100])
        tgt = torch.randint(0, 40, (2, 20))
        
        output = model(src, src_lengths, tgt)
        print(f"  ✓ Forward pass: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_ctc_model():
    """Test CTC model creation."""
    print("\nTesting CTC model (models/ctc/)...")
    
    try:
        # Add path
        sys.path.insert(0, str(Path(__file__).parent / 'models' / 'ctc'))
        
        from model import CTCModel
        
        # Create model
        model = CTCModel(vocab_size=38, input_dim=80, hidden_dim=192, num_layers=3, nhead=6)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ Model created: {num_params:,} parameters")
        
        # Test forward pass
        src = torch.randn(2, 100, 80)
        src_lengths = torch.tensor([100, 100])
        
        log_probs, out_lengths = model(src, src_lengths)
        print(f"  ✓ Forward pass: {log_probs.shape}")
        
        # Test decoding
        decoded = model.decode_greedy(src, src_lengths)
        print(f"  ✓ Greedy decode: {len(decoded)} sequences")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_data_exists():
    """Check if training data exists."""
    print("\nChecking training data...")
    
    data_paths = [
        ('Synthetic v1', Path('data/synthetic/morse_v1')),
        ('Real world', Path('data/real_world/morse_data')),
        ('Detector', Path('data/detector')),
    ]
    
    found_any = False
    for name, path in data_paths:
        if path.exists():
            # Count files
            audio_count = len(list((path / 'audio').glob('*.wav'))) if (path / 'audio').exists() else 0
            text_count = len(list((path / 'text').glob('*.txt'))) if (path / 'text').exists() else 0
            
            if audio_count > 0:
                print(f"  ✓ {name}: {audio_count} audio, {text_count} text files")
                found_any = True
            else:
                print(f"  ⓘ {name}: exists but empty")
        else:
            print(f"  - {name}: not found")
    
    if not found_any:
        print("\n  To generate data:")
        print("    python scripts/generate_morse_data.py --output_dir data/synthetic/morse_v2 --num_samples 2000")
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Audio-to-Text Decoder - Setup Test")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("Dependencies", test_dependencies()))
    results.append(("Attention Model", test_attention_model()))
    results.append(("CTC Model", test_ctc_model()))
    results.append(("Training Data", test_data_exists()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("=" * 70)
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Generate data:  python scripts/generate_morse_data.py --output_dir data/synthetic/morse_v2 --num_samples 2000")
        print("  2. Train CTC:      cd models/ctc && python train.py --config config.yaml")
        print("  3. Train Attention: cd models/attention && python train.py --config config.yaml")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    print()


if __name__ == '__main__':
    main()
