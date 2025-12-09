# Models

Three audio-to-text model architectures for different use cases.

## Model Comparison

| Model | Params | Architecture | Best For |
|-------|--------|--------------|----------|
| `ctc/` | ~2M | CNN + Transformer Encoder | Morse, exact transcription |
| `ctc_w_pretrain/` | ~300K | Same, with pretraining | Limited labeled data |
| `attention/` | ~3.4M | Encoder-Decoder Transformer | Natural language |

## Directory Structure

Each model directory follows the same structure:

```
models/{model_name}/
├── model.py          # Model architecture
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training script
├── inference.py      # Inference script
├── config.yaml       # Configuration
├── checkpoints/      # Saved models
├── runs/             # TensorBoard logs
└── README.md         # Model-specific documentation
```

---

## `ctc/` - CTC Encoder (Recommended)

Encoder-only model with CTC loss for direct alignment. No language model means no hallucination.

```bash
cd models/ctc
python train.py --config config.yaml
```

**Features:**
- 2M parameters (256 hidden, 4 layers)
- Audio augmentation (noise, compression, bandpass)
- OneCycleLR scheduler with warmup
- TensorBoard logging
- Early stopping

**Best for:**
- Call signs (W1ABC, K3XYZ)
- Exact character-by-character transcription
- Fast inference

**Training arguments:**
```
--config, -c     Config file (default: config.yaml)
--resume, -r     Resume from checkpoint
```

---

## `ctc_w_pretrain/` - CTC with Self-Supervised Pretraining

Smaller encoder that can be pretrained on unlabeled audio using masked spectrogram prediction.

```bash
# Step 1: Pretrain
python pretrain_masked_ctc_w_pretrain.py \
    --data-dirs data/synthetic/morse_v2/audio \
    --epochs 50 \
    --save-path models/ctc_w_pretrain/checkpoints/pretrained_encoder.pt

# Step 2: Fine-tune
cd models/ctc_w_pretrain
python train.py --config config.yaml \
    --pretrained-encoder checkpoints/pretrained_encoder.pt
```

**Features:**
- 300K parameters (128 hidden, 2 layers)
- Masked spectrogram pretraining
- Can load pretrained encoder weights
- Optional encoder freezing

**Best for:**
- Limited labeled data
- Domain adaptation (synthetic → real-world)
- Faster experimentation

**Training arguments:**
```
--config, -c           Config file
--resume, -r           Resume from full checkpoint
--pretrained-encoder   Load encoder weights only (resets LR)
--freeze-epochs        Freeze encoder for N epochs
--data-dir             Override data directory
--run-name             Name for this training run
```

---

## `attention/` - Seq2Seq Transformer

Full encoder-decoder Transformer with cross-attention.

```bash
cd models/attention
python train.py --config config.yaml
```

**Features:**
- 3.4M parameters
- Autoregressive decoding
- Implicit language model

**Best for:**
- Natural language where context helps
- Error correction is desired
- You have lots of training data

⚠️ **Warning:** May hallucinate plausible but incorrect text. Not recommended for call signs.

---

## Common Configuration (config.yaml)

All models use similar config structure:

```yaml
model:
  hidden_dim: 256
  num_layers: 4
  dropout: 0.1
  nhead: 8

audio:
  sample_rate: 16000
  n_mels: 80
  n_fft: 400
  hop_length: 160

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0005
  num_workers: 4

augmentation:
  enabled: true
  prob_noise: 0.4
  prob_mp3_compression: 0.3

early_stopping:
  patience: 10
  min_delta: 0.002

paths:
  data_dir: ../../data/synthetic/morse_v2
  checkpoint_dir: checkpoints

vocab: " -0123456789?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---

## Training Data

All models use data from:
- `../data/synthetic/` - Generated audio/text pairs
- `../data/real_world/` - Real radio recordings

See `../data/README.md` for data organization.

---

## Logging

All models log to TensorBoard in `runs/`:

```bash
tensorboard --logdir models/ctc/runs
```

Logged metrics:
- Loss/train, Loss/val
- CER/val, Accuracy/val
- LearningRate
- Sample predictions (text)
- Hyperparameters (at end)
