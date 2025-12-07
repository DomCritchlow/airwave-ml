"""
Streaming buffer for handling continuous audio with overlapping windows.
"""

import numpy as np
from typing import Generator, Tuple, Optional
from collections import deque


class StreamingBuffer:
    """
    Manages a sliding window buffer for continuous audio processing.
    
    Audio comes in small chunks (e.g., 100ms) and we need to process
    larger windows (e.g., 10s) with overlap for context continuity.
    """
    
    def __init__(
        self,
        window_duration: float = 10.0,
        overlap_duration: float = 2.0,
        sample_rate: int = 16000
    ):
        """
        Initialize the streaming buffer.
        
        Args:
            window_duration: Size of processing window in seconds
            overlap_duration: Overlap between consecutive windows
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Buffer to accumulate audio
        self.buffer = np.zeros(0, dtype=np.float32)
        
        # Track windows processed
        self.windows_processed = 0
    
    def add(self, chunk: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Add audio chunk to buffer and yield complete windows.
        
        Args:
            chunk: Audio samples to add
            
        Yields:
            Complete windows when buffer is full enough
        """
        # Append to buffer
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])
        
        # Yield windows while we have enough data
        while len(self.buffer) >= self.window_samples:
            # Extract window
            window = self.buffer[:self.window_samples].copy()
            
            # Slide buffer by step size (keeping overlap)
            self.buffer = self.buffer[self.step_samples:]
            
            self.windows_processed += 1
            yield window
    
    def flush(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio in the buffer (padded if needed).
        Call this when stream ends.
        
        Returns:
            Final window (possibly padded) or None if buffer empty
        """
        if len(self.buffer) == 0:
            return None
        
        # Pad to window size
        if len(self.buffer) < self.window_samples:
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[:len(self.buffer)] = self.buffer
            window = padded
        else:
            window = self.buffer[:self.window_samples]
        
        self.buffer = np.zeros(0, dtype=np.float32)
        return window
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.windows_processed = 0
    
    @property
    def buffered_duration(self) -> float:
        """Current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate


class TextMerger:
    """
    Merges overlapping text outputs from consecutive windows.
    Handles deduplication of overlapping regions.
    """
    
    def __init__(self, min_overlap_chars: int = 3):
        """
        Initialize the text merger.
        
        Args:
            min_overlap_chars: Minimum overlap to detect for merging
        """
        self.min_overlap = min_overlap_chars
        self.full_text = ""
        self.last_text = ""
    
    def merge(self, new_text: str) -> str:
        """
        Merge new text with accumulated text, handling overlaps.
        
        Args:
            new_text: Text from latest window
            
        Returns:
            Only the new (non-overlapping) portion of text
        """
        if not self.last_text:
            self.full_text = new_text
            self.last_text = new_text
            return new_text
        
        # Find overlap between end of last text and start of new text
        overlap = self._find_overlap(self.last_text, new_text)
        
        # Get new portion
        if overlap:
            new_portion = new_text[len(overlap):]
        else:
            # No overlap detected - might be a gap or different content
            new_portion = " " + new_text if new_text else ""
        
        # Append new portion
        self.full_text += new_portion
        self.last_text = new_text
        
        return new_portion
    
    def _find_overlap(self, prev: str, curr: str) -> str:
        """Find overlapping substring between end of prev and start of curr."""
        if not prev or not curr:
            return ""
        
        # Check for overlaps of decreasing length
        max_check = min(len(prev), len(curr), 50)  # Limit search
        
        for length in range(max_check, self.min_overlap - 1, -1):
            if prev[-length:] == curr[:length]:
                return curr[:length]
        
        return ""
    
    def get_full_text(self) -> str:
        """Get the complete merged text."""
        return self.full_text
    
    def reset(self):
        """Clear accumulated text."""
        self.full_text = ""
        self.last_text = ""


class VADBuffer(StreamingBuffer):
    """
    Voice Activity Detection buffer.
    Only processes windows when signal is detected.
    """
    
    def __init__(
        self,
        window_duration: float = 10.0,
        overlap_duration: float = 2.0,
        sample_rate: int = 16000,
        energy_threshold: float = 0.01,
        min_signal_duration: float = 0.5
    ):
        super().__init__(window_duration, overlap_duration, sample_rate)
        self.energy_threshold = energy_threshold
        self.min_signal_samples = int(min_signal_duration * sample_rate)
        self._signal_detected = False
        self._signal_samples = 0
    
    def _has_signal(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains signal above threshold."""
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.energy_threshold
    
    def add(self, chunk: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Add chunk, only yield windows when signal present."""
        if self._has_signal(chunk):
            self._signal_detected = True
            self._signal_samples += len(chunk)
        else:
            # Signal ended - check if we should process
            if self._signal_detected and self._signal_samples >= self.min_signal_samples:
                # Process buffered audio
                for window in super().add(chunk):
                    yield window
                
                # Also flush remaining
                final = self.flush()
                if final is not None:
                    yield final
            
            self._signal_detected = False
            self._signal_samples = 0
            return
        
        # Signal active - add to buffer
        for window in super().add(chunk):
            yield window

