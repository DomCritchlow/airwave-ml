# Production Morse Code Decoder

This directory contains the production deployment code for the trained Morse code model.

## Quick Start

```bash
cd production

# Decode from microphone (CTC model - recommended)
python live_decode.py \
    --checkpoint ../models/ctc/checkpoints/best_model_ctc.pt \
    --source mic

# Decode from microphone (Attention model)
python live_decode.py \
    --checkpoint ../models/attention/checkpoints/best_model.pt \
    --source mic

# Decode a WAV file
python live_decode.py \
    --checkpoint ../models/ctc/checkpoints/best_model_ctc.pt \
    --source file --input path/to/audio.wav

# Use beam search for better accuracy
python live_decode.py --beam-search --source mic
```

## Components

### `decoder/model_wrapper.py`
Loads the trained model and provides a simple `decode(audio)` interface.

### `decoder/audio_input.py`
Audio input handlers:
- `MicrophoneInput`: Live microphone capture (requires `sounddevice`)
- `FileInput`: Read from WAV files
- `CallbackInput`: For integration with other systems

### `decoder/buffer.py`
Streaming components:
- `StreamingBuffer`: Manages sliding window processing
- `TextMerger`: Merges overlapping decoded text
- `VADBuffer`: Voice activity detection variant

### `live_decode.py`
Main CLI for live decoding.

### `api_server.py`
REST API and WebSocket server for web integration.

## Requirements

Additional production dependencies:
```bash
pip install sounddevice  # For microphone input
pip install fastapi uvicorn websockets  # For API server (optional)
```

## Usage Examples

### CLI - Microphone

```bash
python live_decode.py \
    --checkpoint ../checkpoints/best_model.pt \
    --source mic \
    --window 10 \
    --overlap 2
```

### CLI - File (Streaming)

```bash
python live_decode.py \
    --checkpoint ../checkpoints/best_model.pt \
    --source file \
    --input recording.wav
```

### CLI - File (Batch)

For short files, process all at once (faster):

```bash
python live_decode.py \
    --checkpoint ../checkpoints/best_model.pt \
    --batch \
    --input recording.wav
```

### Python API

```python
from decoder import MorseDecoder, StreamingBuffer, TextMerger

# Load model
decoder = MorseDecoder('checkpoints/best_model.pt')

# Decode audio array
text = decoder.decode(audio_array)

# Or decode a file
text = decoder.decode_file('recording.wav')
```

### Web API

```bash
# Start server
python api_server.py --checkpoint ../checkpoints/best_model.pt

# POST audio for decoding
curl -X POST http://localhost:8000/decode \
    -F "audio=@recording.wav"

# Or use WebSocket for real-time streaming
# Connect to ws://localhost:8000/ws
```

## Configuration

Adjust these parameters based on your use case:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window` | 10.0 | Processing window in seconds |
| `--overlap` | 2.0 | Overlap between windows |
| `--beam-search` | False | Use beam search (slower, more accurate) |
| `--device` | auto | Force CPU/CUDA/MPS |

## Performance Notes

- **CPU**: Expect ~2-5x realtime speed (10s audio processes in 2-5s)
- **GPU**: Near realtime or faster
- **Beam search**: 3-5x slower than greedy but more accurate

## Troubleshooting

### No microphone detected
```bash
# List available devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Specify device index
python live_decode.py --source mic --device 1
```

### Model not loading
- Ensure training completed successfully
- Check `../checkpoints/best_model.pt` exists
- Verify PyTorch version matches training environment

