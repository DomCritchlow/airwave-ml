# Real World Test Data

This directory contains real Morse code data from the [morse.mdp.im podcast](https://morse.mdp.im) for testing trained models against real-world audio.

## Contents

- `morse_data/` - Downloaded podcast episodes and chunked training data
- `scripts/download_morse_data.py` - Script to download more podcast data
- `scripts/prepare_headlines.py` - Script to extract headlines from episodes

## Usage

### Download More Episodes

```bash
# From project root
python data/real_world/scripts/download_morse_data.py full-pipeline \
    --output_dir data/real_world/morse_data \
    --max_episodes 30
```

### Test a Trained Model

```bash
# From project root
python models/ctc/inference.py \
    --checkpoint models/ctc/checkpoints/best_model_ctc.pt \
    --audio data/real_world/morse_data/raw/audio/episode_000.wav
```

## Note

This data has inherent audio-text alignment challenges since we don't have exact timestamps for each word. It's best used for:
- Final model evaluation
- Testing generalization to real-world conditions
- Comparing synthetic vs real performance

For training, use the synthetic data generator (`scripts/generate_morse_data.py`) which provides perfect alignment.

