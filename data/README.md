# Training Data

Organized training data for audio-to-text models.

## Directory Structure

```
data/
├── synthetic/           # Machine-generated audio/text pairs
│   ├── morse_v1/       # Original synthetic Morse (1000 samples)
│   └── morse_v2/       # Improved text diversity (to be generated)
│
├── real_world/          # Real radio recordings
│   ├── morse_data/
│   │   ├── raw/        # Full episode downloads
│   │   └── chunked/    # Chunked into training samples
│   └── scripts/        # Download/processing scripts
│
└── detector/            # Signal detector training data
    └── spectrograms/    # Labeled waterfall images
```

## Synthetic Data

Generated with `scripts/generate_morse_data.py`

### `morse_v1/` - Original
- 1000 samples
- Basic word list
- Variable operator timing

### `morse_v2/` - Improved (to be generated)
- 2000+ samples
- Rich text diversity:
  - Call signs (W1ABC, K3XYZ)
  - Q-codes (QTH, QSL)
  - Numbers, frequencies
  - General language
  - Adversarial patterns

To generate:
```bash
python scripts/generate_morse_data.py \
    --output_dir data/synthetic/morse_v2 \
    --num_samples 2000
```

## Real World Data

From RSS podcast feeds of Morse code news.

### Collection
```bash
python data/real_world/scripts/download_morse_data.py full-pipeline \
    --output_dir data/real_world/morse_data
```

### Chunking
Audio is chunked into 10-second segments with 2-second overlap for training.

## Detector Data

For training the signal detector CNN.

Generated with `universal_decoder/detector/generate_detector_data.py`

Contains labeled spectrograms with:
- Signal bounding boxes
- Mode classifications (CW, PSK31, RTTY, FT8, SSB)

To generate:
```bash
python universal_decoder/detector/generate_detector_data.py \
    --output data/detector \
    --num-samples 5000
```

## Data Format

All datasets follow the same format:

```
dataset_name/
├── audio/           # WAV files (16kHz mono)
│   ├── sample_00000.wav
│   ├── sample_00001.wav
│   └── ...
├── text/            # Matching text files
│   ├── sample_00000.txt
│   └── ...
└── metadata.json    # Sample info (duration, WPM, etc.)
```

