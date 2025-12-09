<p align="center">
  <img src="assets/icon.svg" alt="airwave-ml logo" width="80">
</p>

<p align="center">
  <img src="assets/logo.svg" alt="airwave-ml" width="600">
</p>

<p align="center">
  <strong>Machine learning for decoding radio audio — Morse code, digital modes, and beyond.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Features

- **Multiple model architectures**: CTC, CTC with pretraining, Attention-based seq2seq
- **Universal signal decoder**: Detect and decode multiple radio modes
- **Synthetic data generation**: Realistic training data with operator variability
- **Audio augmentation**: MP3 compression, noise, bandpass, time-stretch for robust models
- **Production deployment**: CLI, API, and streaming support

## Repository Structure

```
airwave-ml/
├── models/                      # Model architectures
│   ├── ctc/                    # CTC model (encoder-only, recommended)
│   ├── ctc_w_pretrain/         # CTC with masked spectrogram pretraining
│   └── attention/              # Seq2Seq Transformer (encoder-decoder)
│
├── data/                        # Training data
│   ├── synthetic/
│   │   ├── morse_v2/           # 20,000 Morse samples
│   │   ├── psk31_v1/           # 2,000 PSK31 samples
│   │   └── rtty_v1/            # 2,000 RTTY samples
│   └── real_world/
│       └── morse_data/         # Real radio recordings
│
├── scripts/                     # Data generation
│   ├── generate_morse_data.py  # Morse code generator
│   ├── generate_psk31_data.py  # PSK31 generator
│   ├── generate_rtty_data.py   # RTTY generator
│   └── generate_training_text.py
│
├── universal_decoder/           # Multi-mode radio decoder
│   ├── detector/               # Signal detection CNN
│   ├── extractor/              # Bandpass filtering
│   ├── decoders/               # Mode-specific decoders
│   └── pipeline/               # Orchestration
│
├── production/                  # Deployment code
│   ├── live_decode.py          # CLI decoder
│   └── api_server.py           # REST/WebSocket API
│
└── pretrain_masked_ctc_w_pretrain.py  # Self-supervised pretraining
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Morse code (20,000 samples with operator variability)
python scripts/generate_morse_data.py \
    --output_dir data/synthetic/morse_v2 \
    --num_samples 20000

# PSK31
python scripts/generate_psk31_data.py \
    --output_dir data/synthetic/psk31_v1 \
    --num_samples 2000

# RTTY
python scripts/generate_rtty_data.py \
    --output_dir data/synthetic/rtty_v1 \
    --num_samples 2000
```

### 3. Train a Model

**CTC (recommended for Morse/radio):**
```bash
cd models/ctc
python train.py --config config.yaml
```

**CTC with Pretraining (best for limited labeled data):**
```bash
# Step 1: Pretrain encoder on unlabeled audio
python pretrain_masked_ctc_w_pretrain.py \
    --data-dirs data/synthetic/morse_v2/audio \
    --epochs 50 \
    --save-path models/ctc_w_pretrain/checkpoints/pretrained_encoder.pt

# Step 2: Fine-tune with CTC loss
cd models/ctc_w_pretrain
python train.py --config config.yaml \
    --pretrained-encoder checkpoints/pretrained_encoder.pt
```

### 4. Decode Audio

```bash
cd models/ctc
python inference.py \
    --checkpoint checkpoints/best_model_ctc.pt \
    --audio path/to/audio.wav
```

### 5. Monitor Training

```bash
tensorboard --logdir models/ctc/runs
# Open http://localhost:6006
```

## Model Comparison

| Model | Params | Best For | Training |
|-------|--------|----------|----------|
| **CTC** | ~2M | Morse, call signs, exact transcription | Supervised |
| **CTC + Pretrain** | ~300K | Limited labeled data, domain adaptation | Self-supervised → Supervised |
| **Attention** | ~3.4M | Natural language, context-aware | Supervised |

## Training Features

### Audio Augmentation

The training pipeline includes robust augmentation:
- MP3 compression simulation
- Bandpass filtering (radio bandwidth)
- Noise injection (white, pink, band-limited)
- Time stretching and pitch shifting
- Volume variations and clipping

### Logging

All models log to TensorBoard:
- Loss curves (train/val)
- CER and accuracy
- Learning rate
- Sample predictions
- Hyperparameters

### Early Stopping

Training automatically stops when validation CER plateaus (configurable patience).

## Results

### CTC Model on Morse Code

| Dataset | Samples | Accuracy |
|---------|---------|----------|
| Synthetic (morse_v2) | 20,000 | 97.8% |
| Real-world (fine-tuned) | 901 | ~65% |

### With Pretraining

Pretraining reduces required epochs by ~3x and improves generalization to real-world audio.

## Documentation

- `models/ctc/README.md` - CTC model details
- `models/ctc_w_pretrain/README.md` - Pretraining approach
- `models/attention/README.md` - Attention model
- `data/README.md` - Data organization
- `universal_decoder/README.md` - Multi-mode decoder
- `production/README.md` - Deployment

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchaudio >= 2.0
- NumPy, SciPy, librosa

See `requirements.txt` for full list.

## License

MIT
