# CTC-based Morse Code Decoder

Alternative architecture using CTC (Connectionist Temporal Classification) instead of seq2seq with attention.

## Why CTC?

| Feature | Seq2Seq (Transformer) | CTC |
|---------|----------------------|-----|
| **Call signs** (W1ABC) | May hallucinate common patterns | ✅ Exact character decoding |
| **Language priors** | Yes (can correct/hallucinate) | No (pure pattern matching) |
| **Inference speed** | Slow (autoregressive) | ✅ Fast (single pass) |
| **Architecture** | Encoder + Decoder | ✅ Encoder only |
| **Training** | Teacher forcing | CTC alignment |

## Architecture

```
Audio → Mel Spectrogram → CNN Frontend → Transformer Encoder → Linear → CTC Loss
                           (stride 4)        (4 layers)         (vocab)
```

### Key Components

1. **CNN Frontend**: Reduces time dimension by 4x, extracts local features
2. **Transformer Encoder**: Self-attention over audio frames (no decoder!)
3. **Linear Output**: Projects to vocab size (including blank token)
4. **CTC Loss**: Learns alignment between audio and text automatically

## Files

```
src_ctc/
├── model.py      # CTCModel with greedy and beam decoding
├── dataset.py    # CTC-specific dataset (blank token at index 0)
├── train.py      # Training loop with CTC loss
├── inference.py  # CTCDecoder for inference
└── README.md
```

## Training

```bash
# Uses the same synthetic data as seq2seq model
python src_ctc/train.py --config config_ctc.yaml
```

Output:
- `checkpoints_ctc/best_model_ctc.pt` - Best model by CER
- `checkpoints_ctc/checkpoint_epoch_*.pt` - Periodic checkpoints

## Inference

```bash
# Single file
python src_ctc/inference.py \
    --checkpoint checkpoints_ctc/best_model_ctc.pt \
    --audio morse_synthetic/audio/morse_00001.wav

# With beam search (more accurate)
python src_ctc/inference.py \
    --checkpoint checkpoints_ctc/best_model_ctc.pt \
    --audio morse_synthetic/audio/morse_00001.wav \
    --beam --beam-width 10

# Directory of files
python src_ctc/inference.py \
    --checkpoint checkpoints_ctc/best_model_ctc.pt \
    --audio morse_synthetic/audio/
```

## CTC Decoding

### Greedy Decoding
1. Take argmax at each timestep
2. Collapse repeated tokens
3. Remove blank tokens

Example:
```
Model output:  [A, A, BLANK, B, B, B, BLANK, BLANK, C]
After collapse: [A, B, C]
Final text:     "ABC"
```

### Beam Search
More accurate but slower. Considers multiple hypotheses at each step.

## Vocabulary

CTC requires a special blank token (index 0) for alignment:

```
Index 0:  <BLANK>  (CTC alignment token)
Index 1:  ' '      (space)
Index 2:  '0'
...
Index 37: 'Z'
```

## Comparison with Seq2Seq

### When to use CTC:
- Decoding call signs (W1ABC, K3XYZ)
- Exact character-for-character transcription
- Speed is important
- You DON'T want the model to "correct" errors

### When to use Seq2Seq:
- Natural language text where context helps
- Error correction is desired
- You have a lot of training data

## Integration with Production Decoder

The `production/` decoder can use either model. Just point to the appropriate checkpoint:

```bash
# CTC model
python production/live_decode.py \
    --checkpoint checkpoints_ctc/best_model_ctc.pt \
    --source mic

# Seq2Seq model  
python production/live_decode.py \
    --checkpoint checkpoints/best_model.pt \
    --source mic
```

Note: The production decoder auto-detects model type from checkpoint.

