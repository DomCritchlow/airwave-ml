# Universal Radio Decoder

A modular system for detecting and decoding multiple radio signal types from wideband audio.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL RADIO DECODER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Wideband Audio (48kHz)                                        │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │  Spectrogram    │  FFT → Waterfall image                    │
│  │  Generator      │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Signal Detector │  CNN detects signals in spectrogram       │
│  │ (CNN)           │  Outputs: [(freq, mode, confidence), ...]  │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Signal Extractor│  Bandpass filter → isolate signal         │
│  └────────┬────────┘                                           │
│           │                                                     │
│     ┌─────┴─────┬─────────┬─────────┐                          │
│     ▼           ▼         ▼         ▼                          │
│  ┌─────┐    ┌─────┐   ┌─────┐   ┌─────┐                       │
│  │ CW  │    │PSK31│   │ FT8 │   │Voice│                       │
│  │(CTC)│    │(CTC)│   │(lib)│   │(STT)│                       │
│  └──┬──┘    └──┬──┘   └──┬──┘   └──┬──┘                       │
│     └──────────┴─────────┴─────────┘                           │
│                    │                                            │
│                    ▼                                            │
│         Decoded Text + Metadata                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Signal Detector (`detector/`)
CNN-based detector that identifies signals in spectrograms.

- **Input**: Spectrogram image (256×512)
- **Output**: List of (frequency, mode, confidence, bounding box)
- **Modes**: CW, PSK31, RTTY, FT8, SSB, NOISE

```python
from universal_decoder.detector import SignalDetectorCNN, SignalMode

detector = SignalDetectorCNN()
signals = detector.detect(spectrogram, freq_range=(0, 24000), time_range=(0, 10))
```

### 2. Signal Extractor (`extractor/`)
Isolates individual signals from wideband audio.

- Bandpass filtering
- Frequency mixing to baseband
- Resampling to decoder sample rate

```python
from universal_decoder.extractor import SignalExtractor

extractor = SignalExtractor(sample_rate=48000, output_sample_rate=16000)
extracted = extractor.extract(audio, center_freq=7040, bandwidth=200)
```

### 3. Decoder Router (`decoders/`)
Routes signals to appropriate mode-specific decoders.

- **CW**: Uses our CTC-based Morse decoder
- **PSK31**: CTC decoder (when trained)
- **FT8**: Would integrate with WSJT-X
- **SSB**: Uses Whisper for speech-to-text

```python
from universal_decoder.decoders import DecoderRouter, CWDecoder

router = DecoderRouter()
router.register_decoder('CW', CWDecoder('checkpoints_ctc/best_model_ctc.pt'))
result = router.decode(audio, sample_rate=16000, mode='CW')
```

### 4. Pipeline (`pipeline/`)
Main orchestrator that ties everything together.

```python
from universal_decoder import UniversalRadioDecoder

decoder = UniversalRadioDecoder()
decoder.load_models(cw_checkpoint='checkpoints_ctc/best_model_ctc.pt')

results = decoder.decode_audio(wideband_audio, sample_rate=48000)

for r in results:
    print(f"{r.center_freq_hz/1000:.1f} kHz [{r.mode}]: {r.text}")
```

## Training

### 1. Train Signal Detector

```bash
# Generate synthetic training data
python universal_decoder/detector/generate_detector_data.py \
    --output detector_data \
    --num-samples 5000

# Train detector (TODO: training script)
python universal_decoder/detector/train_detector.py \
    --data detector_data \
    --epochs 50
```

### 2. Train CW Decoder

```bash
# Generate Morse code training data
python scripts/generate_morse_data.py \
    --output morse_synthetic \
    --num-samples 1000

# Train CTC model
python src_ctc/train.py --config config_ctc.yaml
```

### 3. Train PSK31 Decoder

```bash
# Generate PSK31 training data (TODO)
python scripts/generate_psk31_data.py \
    --output psk31_synthetic \
    --num-samples 1000

# Train CTC model with PSK31 config
python src_ctc/train.py --config config_psk31.yaml
```

## Usage Examples

### Basic Decoding

```python
from universal_decoder import UniversalRadioDecoder

# Initialize
decoder = UniversalRadioDecoder()
decoder.load_models(cw_checkpoint='checkpoints_ctc/best_model_ctc.pt')

# Decode from SDR audio
results = decoder.decode_audio(audio, sample_rate=48000)
```

### Known Signal (Skip Detection)

```python
# If you know the mode, skip detection
result = decoder.decode_known_signal(
    audio=audio,
    sample_rate=16000,
    mode='CW'
)
print(result.text)
```

### Streaming Decoding

```python
from universal_decoder import StreamingRadioDecoder

streamer = StreamingRadioDecoder(chunk_duration=5.0)
streamer.load_models()

# Feed audio chunks from SDR
for chunk in sdr_stream:
    results = streamer.process_chunk(chunk)
    for r in results:
        print(f"[{r.mode}] {r.text}")
```

### With SDR

```python
import rtlsdr  # RTL-SDR library

from universal_decoder import UniversalRadioDecoder, RadioDecoderConfig

config = RadioDecoderConfig(
    input_sample_rate=240000,  # SDR sample rate
    freq_min=0,
    freq_max=12000  # Visible bandwidth
)

decoder = UniversalRadioDecoder(config)
decoder.load_models()

# Tune to 7.040 MHz and decode
sdr = rtlsdr.RtlSdr()
sdr.sample_rate = 240000
sdr.center_freq = 7040000

samples = sdr.read_samples(240000 * 10)  # 10 seconds
results = decoder.decode_audio(samples)
```

## Directory Structure

```
universal_decoder/
├── detector/
│   ├── signal_detector.py      # CNN detector model
│   └── generate_detector_data.py  # Training data generator
├── extractor/
│   └── signal_extractor.py     # Bandpass filter & extraction
├── decoders/
│   └── decoder_router.py       # Mode-specific decoder routing
├── pipeline/
│   └── radio_decoder.py        # Main orchestrator
├── __init__.py
└── README.md
```

## Supported Modes

| Mode | Status | Decoder |
|------|--------|---------|
| CW (Morse) | ✅ Ready | CTC model |
| PSK31 | 🔄 Pending | CTC model (needs training) |
| RTTY | 🔄 Pending | CTC model (needs training) |
| FT8 | 📋 Planned | WSJT-X integration |
| SSB (Voice) | 📋 Planned | Whisper |

## Next Steps

1. **Train CW decoder** on synthetic data
2. **Train signal detector** on synthetic spectrograms
3. **Generate PSK31 data** and train decoder
4. **Integrate with SDR** for live decoding
5. **Build web UI** for visualization

