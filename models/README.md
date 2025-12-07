# Models

Two audio-to-text model architectures for different use cases.

## Model Comparison

| Feature | `attention/` | `ctc/` |
|---------|--------------|--------|
| **Architecture** | Encoder-Decoder | Encoder only |
| **Parameters** | ~3.4M | ~1.6M |
| **Decoding** | Autoregressive | Single-pass |
| **Language model** | Yes (implicit) | No |
| **Call signs** | May hallucinate | ✅ Exact |
| **Training speed** | 1x | ~1.5-2x faster |
| **Inference speed** | Slower | ✅ Faster |

## `attention/` - Seq2Seq Transformer

Full encoder-decoder Transformer with cross-attention.

```bash
cd models/attention
python train.py --config config.yaml
```

**Best for:**
- Natural language where context helps
- Error correction is desired
- You have lots of training data

**Checkpoints:** `models/attention/checkpoints/`

## `ctc/` - CTC Encoder

Encoder-only model with CTC loss for direct alignment.

```bash
cd models/ctc
python train.py --config config.yaml
```

**Best for:**
- Call signs (W1ABC, K3XYZ)
- Exact character-by-character transcription
- Fast inference
- Smaller memory footprint

**Checkpoints:** `models/ctc/checkpoints/`

## Training Data

Both models use data from `../data/synthetic/` or `../data/real_world/`.

See `../data/README.md` for data organization.

