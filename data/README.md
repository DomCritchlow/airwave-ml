# Training Data

Organized training data for audio-to-text models.

## Directory Structure

```
data/
├── synthetic/              # Machine-generated audio/text pairs
│   ├── morse_v1/          # Original Morse (2,000 samples)
│   ├── morse_v2/          # Large-scale Morse (20,000 samples)
│   ├── psk31_v1/          # PSK31 digital mode (2,000 samples)
│   └── rtty_v1/           # RTTY digital mode (2,000 samples)
│
├── real_world/             # Real radio recordings
│   └── morse_data/
│       ├── raw/           # Full episode downloads
│       ├── chunked/       # 10-second training samples (901 chunks)
│       └── scripts/       # Download/processing scripts
│
└── detector/               # Signal detector training data
    └── spectrograms/       # Labeled waterfall images
```

## Synthetic Data

### Morse Code

| Dataset | Samples | Characters | Description |
|---------|---------|------------|-------------|
| `morse_v1/` | 2,000 | 39 | Basic operator variability |
| `morse_v2/` | 20,000 | 39 | Full variability, augmentation-ready |

**Features:**
- Variable WPM (18-32)
- Operator "fist" simulation (timing variance, drift)
- Background noise
- Key-click artifacts

**Generate:**
```bash
python scripts/generate_morse_data.py \
    --output_dir data/synthetic/morse_v2 \
    --num_samples 20000
```

### PSK31

| Dataset | Samples | Characters | Description |
|---------|---------|------------|-------------|
| `psk31_v1/` | 2,000 | 95 | Full ASCII (Varicode) |

**Features:**
- BPSK at 31.25 baud
- Varicode encoding
- QSB fading simulation
- SNR variation

**Generate:**
```bash
python scripts/generate_psk31_data.py \
    --output_dir data/synthetic/psk31_v1 \
    --num_samples 2000
```

### RTTY

| Dataset | Samples | Characters | Description |
|---------|---------|------------|-------------|
| `rtty_v1/` | 2,000 | 52 | Baudot (letters + figures) |

**Features:**
- FSK at 45.45 baud (170 Hz shift)
- Baudot encoding with LTRS/FIGS shifts
- Timing jitter
- Noise injection

**Generate:**
```bash
python scripts/generate_rtty_data.py \
    --output_dir data/synthetic/rtty_v1 \
    --num_samples 2000
```

## Real-World Data

From RSS podcast feeds of Morse code news broadcasts.

### Collection

```bash
python data/real_world/scripts/download_morse_data.py full-pipeline \
    --output_dir data/real_world/morse_data
```

### Chunking

Audio is automatically chunked into:
- 10-second segments
- 2-second overlap
- 16kHz mono WAV

**Current stats:**
- 901 labeled chunks
- Transcripts from broadcast scripts

## Data Format

All datasets follow the same format:

```
dataset_name/
├── audio/              # WAV files (16kHz mono)
│   ├── sample_00000.wav
│   ├── sample_00001.wav
│   └── ...
├── text/               # Matching text files
│   ├── sample_00000.txt
│   └── ...
└── metadata.json       # Sample info (duration, WPM, etc.)
```

### Metadata Schema

```json
{
  "id": "morse_00000",
  "text": "CQ CQ DE W1ABC",
  "duration": 4.5,
  "wpm": 25,
  "frequency": 600,
  "noise_level": 0.05
}
```

## Character Sets

| Mode | Characters | Count |
|------|------------|-------|
| Morse | `SPACE - 0-9 ? A-Z` | 39 |
| PSK31 | Full ASCII (printable) | 95 |
| RTTY | `A-Z 0-9 SPACE` + Baudot figures | 52 |

## Using Custom Data

To add your own data:

1. Create directory structure:
   ```
   data/synthetic/my_dataset/
   ├── audio/
   └── text/
   ```

2. Add paired files:
   - `audio/sample_00000.wav` (16kHz mono)
   - `text/sample_00000.txt` (uppercase text)

3. Update model config:
   ```yaml
   paths:
     data_dir: ../../data/synthetic/my_dataset
   ```
