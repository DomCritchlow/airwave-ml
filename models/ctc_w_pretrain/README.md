# CTC Model with Self-Supervised Pretraining

This variant of the CTC model supports **masked spectrogram pretraining** before supervised fine-tuning, similar to wav2vec2/BERT approaches.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Pretraining Phase                      │
│  Mel Spectrogram → [MASK] → CNN → Encoder → Reconstruct │
│                    (Self-supervised, no labels)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Fine-tuning Phase                      │
│  Mel Spectrogram → CNN → Encoder → Linear → CTC Loss    │
│                    (Supervised, with labels)             │
└─────────────────────────────────────────────────────────┘
```

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 128 | Smaller than main CTC (256) |
| num_layers | 2 | Fewer transformer layers |
| nhead | 4 | Attention heads |
| Parameters | ~300K | Efficient for pretraining |

## Usage

### 1. Pretrain the Encoder (Self-Supervised)

Uses unlabeled audio to learn representations via masked prediction:

```bash
python pretrain_masked_ctc_w_pretrain.py \
    --data-dirs data/synthetic/morse_v2/audio data/real_world/morse_data/chunked/audio \
    --epochs 50 \
    --save-path models/ctc_w_pretrain/checkpoints/pretrained_encoder.pt
```

### 2. Fine-tune with CTC Loss (Supervised)

Load pretrained weights and train on labeled data:

```bash
cd models/ctc_w_pretrain
python train.py \
    --config config.yaml \
    --pretrained-encoder checkpoints/pretrained_encoder.pt \
    --data-dir ../../data/synthetic/morse_v2
```

### 3. Inference

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav
```

## Training Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to config.yaml |
| `--resume` | Resume from full checkpoint |
| `--pretrained-encoder` | Load only encoder weights (resets LR) |
| `--freeze-epochs` | Freeze encoder for N epochs |
| `--data-dir` | Override data directory |
| `--run-name` | Name for checkpoints and logs |

## Pretraining Details

The masked spectrogram prediction task:
1. Randomly masks 15% of time-steps (contiguous spans of 10 frames)
2. Passes masked spectrogram through encoder
3. Predicts original mel values at masked positions
4. Loss: MSE between predicted and original values

This forces the encoder to learn useful audio representations before seeing any labels.

## Files

```
models/ctc_w_pretrain/
├── model.py         # CTCModel, ConvFrontend, CTCEncoder
├── dataset.py       # Labeled and unlabeled dataset loaders
├── train.py         # CTC fine-tuning with pretrain support
├── inference.py     # Decode audio files
├── config.yaml      # Model and training configuration
├── checkpoints/     # Saved models
└── runs/            # TensorBoard logs
```

## Results

| Training | Synthetic Acc | Real-World Acc |
|----------|--------------|----------------|
| From scratch | ~30% (90 epochs) | ~0% |
| With pretraining | ~92% (30 epochs) | ~65% |

Pretraining significantly improves:
- Convergence speed (3x fewer epochs)
- Final accuracy on synthetic data
- Generalization to real-world audio

