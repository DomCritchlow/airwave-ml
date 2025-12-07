# airwave-ml

Machine learning for decoding radio audio — Morse, digital modes, and beyond.

## Features

- **Two model architectures**: Attention-based seq2seq and CTC
- **Universal signal decoder**: Detect and decode multiple radio modes
- **Synthetic data generation**: Realistic training data with operator variability
- **Production deployment**: CLI, API, and streaming support

## Repository Structure

```
airwave-ml/
├── models/                  # Model architectures
│   ├── attention/          # Seq2Seq Transformer (encoder-decoder)
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── config.yaml
│   │   └── checkpoints/
│   └── ctc/                # CTC model (encoder-only)
│       ├── model.py
│       ├── train.py
│       ├── config.yaml
│       └── checkpoints/
│
├── data/                    # Training data
│   ├── synthetic/          # Generated audio/text pairs
│   │   ├── morse_v1/      # 1000 samples
│   │   └── morse_v2/      # Improved diversity
│   ├── real_world/         # Real radio recordings
│   └── detector/           # Signal detector data
│
├── universal_decoder/       # Multi-mode radio decoder
│   ├── detector/           # Signal detection CNN
│   ├── extractor/          # Bandpass filtering
│   ├── decoders/           # Mode-specific decoders
│   └── pipeline/           # Orchestration
│
├── production/              # Deployment code
│   ├── live_decode.py      # CLI decoder
│   └── api_server.py       # REST/WebSocket API
│
└── scripts/                 # Data generation & utilities
    ├── generate_morse_data.py
    └── generate_training_text.py
```

## Quick Start

### 1. Setup Environment

```bash
./setup_env.sh
source .venv/bin/activate
```

### 2. Generate Training Data

```bash
python scripts/generate_morse_data.py \
    --output_dir data/synthetic/morse_v2 \
    --num_samples 2000
```

### 3. Train a Model

**CTC (recommended for Morse/radio):**
```bash
cd models/ctc
python train.py --config config.yaml
```

**Attention (for natural language):**
```bash
cd models/attention
python train.py --config config.yaml
```

### 4. Decode Audio

```bash
python production/live_decode.py \
    --checkpoint models/ctc/checkpoints/best_model_ctc.pt \
    --source mic
```

## Model Comparison

| Feature | Attention | CTC |
|---------|-----------|-----|
| Parameters | 3.4M | 1.6M |
| Training speed | 1x | 1.5-2x |
| Call signs | May hallucinate | ✅ Exact |
| Language context | ✅ Yes | No |
| Best for | Natural language | Radio/technical |

## Architecture

### Attention Model (Seq2Seq)
```
Audio → CNN → Transformer Encoder → Cross-Attention → Decoder → Text
```

### CTC Model
```
Audio → CNN → Transformer Encoder → Linear → CTC Loss → Text
```

### Universal Decoder
```
Wideband Audio → Signal Detector → Extractor → Mode Router → Decoders → Text
```

## Documentation

- `models/README.md` - Model architectures
- `data/README.md` - Data organization
- `universal_decoder/README.md` - Multi-mode decoder
- `production/README.md` - Deployment

## Requirements

- Python 3.9+
- PyTorch
- torchaudio
- NumPy, SciPy

See `requirements.txt` for full list.

## License

MIT
