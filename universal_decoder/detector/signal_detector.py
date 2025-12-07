"""
Signal Detector

Detects and classifies radio signals from a wideband spectrogram.
Uses a CNN to identify signal presence, location (frequency), and mode.

Architecture:
    Spectrogram Image → CNN → [(freq_start, freq_end, mode, confidence), ...]

Supported modes:
    - CW (Morse code)
    - PSK31
    - RTTY
    - FT8
    - SSB (voice)
    - NOISE (no signal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SignalMode(Enum):
    """Radio signal modes."""
    NOISE = 0
    CW = 1       # Morse code
    PSK31 = 2    # Phase Shift Keying
    RTTY = 3     # Radio Teletype
    FT8 = 4      # Digital weak signal
    SSB = 5      # Voice (Single Sideband)
    AM = 6       # AM broadcast
    FM = 7       # FM broadcast
    UNKNOWN = 8


@dataclass
class DetectedSignal:
    """A detected signal in the spectrum."""
    freq_start_hz: float      # Start frequency in Hz
    freq_end_hz: float        # End frequency in Hz
    center_freq_hz: float     # Center frequency
    bandwidth_hz: float       # Signal bandwidth
    mode: SignalMode          # Detected mode
    confidence: float         # Detection confidence (0-1)
    time_start: float         # Start time in spectrogram
    time_end: float           # End time in spectrogram
    
    def __repr__(self):
        return (f"Signal({self.center_freq_hz/1000:.1f}kHz, "
                f"{self.mode.name}, {self.confidence:.0%})")


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SignalDetectorCNN(nn.Module):
    """
    CNN-based signal detector.
    
    Takes a spectrogram image and outputs:
    - Detection grid: presence of signal at each (time, freq) cell
    - Mode classification for each detected cell
    - Confidence scores
    
    Similar to YOLO but for radio signals.
    """
    
    def __init__(
        self,
        num_modes: int = len(SignalMode),
        input_height: int = 256,   # Frequency bins
        input_width: int = 512,    # Time steps
        grid_size: int = 16        # Detection grid resolution
    ):
        super().__init__()
        
        self.num_modes = num_modes
        self.grid_size = grid_size
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # /2
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # /4
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # /8
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),           # /16
            ConvBlock(256, 256, 3, 1, 1),
        )
        
        # Detection head
        # For each grid cell: [objectness, mode_probs..., bbox_offsets]
        # bbox = (freq_offset, time_offset, freq_width, time_width)
        output_channels = 1 + num_modes + 4
        
        self.detection_head = nn.Sequential(
            ConvBlock(256, 128, 3, 1, 1),
            nn.Conv2d(128, output_channels, 1, 1, 0)
        )
        
        # Adaptive pooling to fixed grid size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, height, width) spectrogram image
            
        Returns:
            dict with:
                - objectness: (batch, grid, grid) - signal presence probability
                - mode_probs: (batch, num_modes, grid, grid) - mode probabilities
                - bbox: (batch, 4, grid, grid) - bounding box offsets
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Pool to grid size
        features = self.adaptive_pool(features)
        
        # Detection head
        output = self.detection_head(features)
        
        # Split outputs
        objectness = torch.sigmoid(output[:, 0:1, :, :])
        mode_logits = output[:, 1:1+self.num_modes, :, :]
        mode_probs = F.softmax(mode_logits, dim=1)
        bbox = torch.sigmoid(output[:, 1+self.num_modes:, :, :])  # Normalized 0-1
        
        return {
            'objectness': objectness.squeeze(1),  # (batch, grid, grid)
            'mode_probs': mode_probs,              # (batch, num_modes, grid, grid)
            'mode_logits': mode_logits,            # For loss computation
            'bbox': bbox                           # (batch, 4, grid, grid)
        }
    
    def detect(
        self,
        spectrogram: torch.Tensor,
        freq_range: Tuple[float, float],
        time_range: Tuple[float, float],
        threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> List[DetectedSignal]:
        """
        Detect signals in a spectrogram.
        
        Args:
            spectrogram: (1, height, width) or (height, width) spectrogram
            freq_range: (min_hz, max_hz) frequency range
            time_range: (start_s, end_s) time range
            threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            List of DetectedSignal objects
        """
        self.eval()
        
        # Ensure correct shape
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
        elif spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward(spectrogram)
        
        objectness = output['objectness'][0]  # (grid, grid)
        mode_probs = output['mode_probs'][0]  # (num_modes, grid, grid)
        bbox = output['bbox'][0]              # (4, grid, grid)
        
        # Convert to numpy for processing
        objectness = objectness.cpu().numpy()
        mode_probs = mode_probs.cpu().numpy()
        bbox = bbox.cpu().numpy()
        
        # Find detections above threshold
        detections = []
        grid_h, grid_w = objectness.shape
        
        freq_min, freq_max = freq_range
        freq_span = freq_max - freq_min
        
        time_min, time_max = time_range
        time_span = time_max - time_min
        
        for i in range(grid_h):
            for j in range(grid_w):
                conf = objectness[i, j]
                if conf < threshold:
                    continue
                
                # Get mode
                mode_idx = mode_probs[:, i, j].argmax()
                mode = SignalMode(mode_idx)
                mode_conf = mode_probs[mode_idx, i, j]
                
                # Skip if classified as noise
                if mode == SignalMode.NOISE:
                    continue
                
                # Calculate bounding box in real units
                # bbox = (freq_offset, time_offset, freq_width, time_width)
                freq_offset = bbox[0, i, j]
                time_offset = bbox[1, i, j]
                freq_width = bbox[2, i, j]
                time_width = bbox[3, i, j]
                
                # Convert grid cell to frequency/time
                cell_freq_center = freq_min + (i + 0.5) / grid_h * freq_span
                cell_time_center = time_min + (j + 0.5) / grid_w * time_span
                
                # Apply offsets
                freq_center = cell_freq_center + (freq_offset - 0.5) * freq_span / grid_h
                time_center = cell_time_center + (time_offset - 0.5) * time_span / grid_w
                
                # Calculate width
                bandwidth = freq_width * freq_span / 4  # Scale factor
                duration = time_width * time_span / 4
                
                detection = DetectedSignal(
                    freq_start_hz=freq_center - bandwidth/2,
                    freq_end_hz=freq_center + bandwidth/2,
                    center_freq_hz=freq_center,
                    bandwidth_hz=bandwidth,
                    mode=mode,
                    confidence=float(conf * mode_conf),
                    time_start=time_center - duration/2,
                    time_end=time_center + duration/2
                )
                detections.append(detection)
        
        # Apply NMS to remove overlapping detections
        detections = self._nms(detections, nms_threshold)
        
        return detections
    
    def _nms(
        self, 
        detections: List[DetectedSignal], 
        threshold: float
    ) -> List[DetectedSignal]:
        """Non-maximum suppression for overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [
                d for d in detections
                if not self._overlaps(best, d, threshold)
            ]
        
        return keep
    
    def _overlaps(self, d1: DetectedSignal, d2: DetectedSignal, threshold: float) -> bool:
        """Check if two detections overlap significantly in frequency."""
        # Calculate frequency overlap
        overlap_start = max(d1.freq_start_hz, d2.freq_start_hz)
        overlap_end = min(d1.freq_end_hz, d2.freq_end_hz)
        
        if overlap_end <= overlap_start:
            return False
        
        overlap = overlap_end - overlap_start
        union = max(d1.freq_end_hz, d2.freq_end_hz) - min(d1.freq_start_hz, d2.freq_start_hz)
        
        iou = overlap / union
        return iou > threshold


class SignalDetectorLoss(nn.Module):
    """
    Loss function for signal detector training.
    
    Combines:
    - Binary cross-entropy for objectness
    - Cross-entropy for mode classification
    - Smooth L1 for bounding box regression
    """
    
    def __init__(self, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model output dict
            targets: dict with:
                - objectness: (batch, grid, grid) binary
                - mode: (batch, grid, grid) class indices
                - bbox: (batch, 4, grid, grid) normalized coords
        """
        pred_obj = predictions['objectness']
        pred_mode = predictions['mode_logits']
        pred_bbox = predictions['bbox']
        
        tgt_obj = targets['objectness']
        tgt_mode = targets['mode']
        tgt_bbox = targets['bbox']
        
        # Objectness loss (BCE)
        obj_mask = tgt_obj.float()
        noobj_mask = 1 - obj_mask
        
        obj_loss = F.binary_cross_entropy(
            pred_obj * obj_mask,
            tgt_obj * obj_mask,
            reduction='sum'
        )
        
        noobj_loss = F.binary_cross_entropy(
            pred_obj * noobj_mask,
            tgt_obj * noobj_mask,
            reduction='sum'
        )
        
        # Mode classification loss (only for cells with objects)
        mode_loss = 0
        if obj_mask.sum() > 0:
            # Reshape for cross entropy
            B, C, H, W = pred_mode.shape
            pred_mode_flat = pred_mode.permute(0, 2, 3, 1).reshape(-1, C)
            tgt_mode_flat = tgt_mode.reshape(-1)
            obj_mask_flat = obj_mask.reshape(-1).bool()
            
            mode_loss = F.cross_entropy(
                pred_mode_flat[obj_mask_flat],
                tgt_mode_flat[obj_mask_flat],
                reduction='sum'
            )
        
        # Bounding box loss (only for cells with objects)
        bbox_loss = 0
        if obj_mask.sum() > 0:
            obj_mask_4d = obj_mask.unsqueeze(1).expand_as(pred_bbox)
            bbox_loss = F.smooth_l1_loss(
                pred_bbox * obj_mask_4d,
                tgt_bbox * obj_mask_4d,
                reduction='sum'
            )
        
        # Combine losses
        total_loss = (
            obj_loss +
            self.lambda_noobj * noobj_loss +
            mode_loss +
            self.lambda_coord * bbox_loss
        )
        
        return total_loss / pred_obj.size(0)  # Average over batch

