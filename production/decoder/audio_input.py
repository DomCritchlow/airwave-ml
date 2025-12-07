"""
Audio input handlers for live decoding.
Supports microphone, audio files, and streaming.
"""

import numpy as np
from pathlib import Path
from typing import Generator, Optional
import wave
import time


class AudioSource:
    """Base class for audio sources."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield audio chunks."""
        raise NotImplementedError


class MicrophoneInput(AudioSource):
    """
    Capture audio from microphone using sounddevice.
    
    Requires: pip install sounddevice
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        device: Optional[int] = None
    ):
        super().__init__(sample_rate)
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        
        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            raise ImportError(
                "sounddevice required for microphone input. "
                "Install with: pip install sounddevice"
            )
    
    def list_devices(self):
        """List available audio input devices."""
        print(self.sd.query_devices())
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks from microphone."""
        print(f"Starting microphone capture at {self.sample_rate}Hz...")
        print("Press Ctrl+C to stop")
        
        with self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device,
            blocksize=self.chunk_samples
        ) as stream:
            while True:
                chunk, overflowed = stream.read(self.chunk_samples)
                if overflowed:
                    print("Warning: Audio buffer overflow")
                yield chunk.flatten()


class FileInput(AudioSource):
    """Read audio from a WAV file, simulating real-time streaming."""
    
    def __init__(
        self, 
        file_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        realtime: bool = False
    ):
        super().__init__(sample_rate)
        self.file_path = Path(file_path)
        self.chunk_duration = chunk_duration
        self.realtime = realtime
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks from file."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(self.file_path))
            
            if sr != self.sample_rate:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            audio = waveform.squeeze().numpy()
            
        except ImportError:
            with wave.open(str(self.file_path), 'rb') as wf:
                if wf.getframerate() != self.sample_rate:
                    raise ValueError(
                        f"Sample rate mismatch. Install torchaudio for resampling."
                    )
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0
        
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            if self.realtime:
                time.sleep(self.chunk_duration)
            yield chunk


class CallbackInput(AudioSource):
    """Receive audio via callbacks for integration with other systems."""
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self._buffer = []
        self._running = True
    
    def push(self, audio: np.ndarray):
        """Push audio chunk to buffer."""
        self._buffer.append(audio)
    
    def stop(self):
        """Stop the stream."""
        self._running = False
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield audio as it arrives."""
        while self._running or self._buffer:
            if self._buffer:
                yield self._buffer.pop(0)
            else:
                time.sleep(0.01)


def create_audio_source(source_type: str, sample_rate: int = 16000, **kwargs) -> AudioSource:
    """Create an audio source by type."""
    sources = {
        'microphone': MicrophoneInput,
        'mic': MicrophoneInput,
        'file': FileInput,
        'callback': CallbackInput
    }
    
    if source_type not in sources:
        raise ValueError(f"Unknown source: {source_type}. Options: {list(sources.keys())}")
    
    return sources[source_type](sample_rate=sample_rate, **kwargs)

