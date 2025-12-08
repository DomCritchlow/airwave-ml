"""
CTC-based Audio-to-Text Model with Pretraining Support

This is a variant of the CTC model that supports:
1. Masked spectrogram pretraining (self-supervised)
2. Loading pretrained encoder weights for fine-tuning

The model architecture matches the original CTC model for compatibility,
but includes a "tiny" configuration for efficient pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TinyEncoderConfig:
    """Configuration for the tiny encoder variant."""
    input_dim: int = 80          # Mel spectrogram bins
    hidden_dim: int = 128        # Smaller than default 256
    num_layers: int = 2          # Fewer layers
    dropout: float = 0.1
    encoder_type: str = "transformer"
    nhead: int = 4


class ConvFrontend(nn.Module):
    """
    CNN frontend to reduce time dimension and extract local features.
    Matches the original CTC model frontend for weight compatibility.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Conv layers with stride to reduce time dimension
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, freq) mel spectrogram
            
        Returns:
            output: (batch, time//4, output_dim) features
        """
        # Transpose for conv: (batch, freq, time)
        x = x.transpose(1, 2)
        
        # Conv layers
        x = self.activation(self.conv1(x))
        x = x.transpose(1, 2)  # (batch, time, hidden)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        x = self.activation(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        x = self.activation(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.norm3(x)
        x = self.dropout(x)
        
        return x  # (batch, time//4, output_dim)
    
    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate output lengths after striding."""
        lengths = input_lengths
        lengths = (lengths + 1) // 2  # First stride-2
        lengths = (lengths + 1) // 2  # Second stride-2
        return lengths


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CTCEncoder(nn.Module):
    """
    Encoder for CTC-based ASR.
    Can use either BiLSTM or Transformer encoder.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
        nhead: int = 4
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if encoder_type == "transformer":
            self.pos_encoder = PositionalEncoding(input_dim, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=input_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = input_dim
            
        else:  # BiLSTM
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, time, input_dim)
            lengths: (batch,) actual lengths
            
        Returns:
            (batch, time, output_dim)
        """
        if self.encoder_type == "transformer":
            x = self.pos_encoder(x)
            
            # Create padding mask if lengths provided
            if lengths is not None:
                max_len = x.size(1)
                mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len)
                mask = mask >= lengths.unsqueeze(1)
            else:
                mask = None
            
            x = self.encoder(x, src_key_padding_mask=mask)
            
        else:  # LSTM
            if lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            
            x, _ = self.encoder(x)
            
            if lengths is not None:
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        return x


class CTCModel(nn.Module):
    """
    Complete CTC-based model for audio-to-text.
    
    Architecture:
        Audio (mel) → CNN Frontend → Encoder → Linear → CTC output
    """
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
        nhead: int = 4
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank_idx = 0
        
        # CNN frontend
        self.frontend = ConvFrontend(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Main encoder
        self.encoder = CTCEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type,
            nhead=nhead
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.encoder.output_dim, vocab_size)
    
    def forward(
        self, 
        audio: torch.Tensor, 
        audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.frontend(audio)
        output_lengths = self.frontend.get_output_lengths(audio_lengths)
        x = self.encoder(x, output_lengths)
        logits = self.output_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, output_lengths
    
    def get_encoder_state_dict(self) -> dict:
        """Get only the encoder weights (for pretraining)."""
        return {
            'frontend': self.frontend.state_dict(),
            'encoder': self.encoder.state_dict()
        }
    
    def load_pretrained_encoder(self, state_dict: dict, strict: bool = False):
        """Load pretrained encoder weights."""
        if 'frontend' in state_dict:
            self.frontend.load_state_dict(state_dict['frontend'], strict=strict)
        if 'encoder' in state_dict:
            self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
    
    def decode_greedy(
        self, 
        audio: torch.Tensor, 
        audio_lengths: torch.Tensor
    ) -> List[List[int]]:
        """Greedy CTC decoding."""
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(audio, audio_lengths)
            predictions = log_probs.argmax(dim=-1)
            
            batch_decoded = []
            for i in range(predictions.size(0)):
                length = output_lengths[i].item()
                pred_seq = predictions[i, :length].tolist()
                
                decoded = []
                prev_token = None
                for token in pred_seq:
                    if token != self.blank_idx and token != prev_token:
                        decoded.append(token)
                    prev_token = token
                
                batch_decoded.append(decoded)
            
            return batch_decoded


def create_ctc_model(config: dict, vocab_size: int) -> CTCModel:
    """Create CTC model from config."""
    model_cfg = config.get('model', {})
    audio_cfg = config.get('audio', {})
    
    return CTCModel(
        vocab_size=vocab_size,
        input_dim=audio_cfg.get('n_mels', 80),
        hidden_dim=model_cfg.get('hidden_dim', 128),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.1),
        encoder_type=model_cfg.get('encoder_type', 'transformer'),
        nhead=model_cfg.get('nhead', 4)
    )


def create_tiny_encoder(config: TinyEncoderConfig = None) -> Tuple[ConvFrontend, CTCEncoder]:
    """
    Create a tiny encoder for pretraining.
    
    Returns frontend and encoder separately for the masked prediction task.
    """
    if config is None:
        config = TinyEncoderConfig()
    
    frontend = ConvFrontend(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.hidden_dim,
        dropout=config.dropout
    )
    
    encoder = CTCEncoder(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        encoder_type=config.encoder_type,
        nhead=config.nhead
    )
    
    return frontend, encoder

