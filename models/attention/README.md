# Attention-based Seq2Seq Model

Full Transformer encoder-decoder with cross-attention for audio-to-text transcription.

## Architecture

```
Audio → Mel Spectrogram → CNN Frontend → Encoder → Cross-Attention → Decoder → Text
                                              ↑
                                         <SOS> + text
```

**Key features:**
- Teacher forcing during training
- Autoregressive decoding (beam search or greedy)
- Implicit language model (can correct errors, but may hallucinate)

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 192 | Embedding dimension |
| nhead | 6 | Attention heads |
| encoder_layers | 3 | Transformer encoder depth |
| decoder_layers | 3 | Transformer decoder depth |
| dim_feedforward | 768 | FFN hidden dimension |
| Parameters | ~3.4M | Total learnable parameters |

## Usage

### Training

```bash
cd models/attention
python train.py --config config.yaml
```

**With checkpoint resume:**
```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

### Inference

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav
```

**Beam search (better quality):**
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav \
    --beam-width 10
```

## When to Use

**✅ Good for:**
- Natural language with context
- When error correction is helpful
- Abundant training data

**⚠️ Not recommended for:**
- Call signs (W1ABC) - may hallucinate
- Exact character sequences required
- Real-time decoding (slower inference)

## Comparison with CTC

| Feature | Attention | CTC |
|---------|-----------|-----|
| Decoding | Autoregressive | Single-pass |
| Language model | Yes (implicit) | No |
| Call signs | May hallucinate | ✅ Exact |
| Inference speed | Slower | ✅ Faster |
| Training speed | 1x | ~1.5x faster |

## Files

```
models/attention/
├── model.py          # AudioToTextModel architecture
├── dataset.py        # Data loading with collation
├── train.py          # Training loop with teacher forcing
├── inference.py      # Beam search / greedy decoding
├── utils.py          # Helpers and metrics
├── config.yaml       # Configuration
├── checkpoints/      # Saved models
└── runs/             # TensorBoard logs
```

## Logging

Training logs to TensorBoard:

```bash
tensorboard --logdir models/attention/runs
```

Metrics logged:
- train/loss, train/accuracy
- val/loss, val/accuracy, val/wer
- Learning rate
- Sample predictions

