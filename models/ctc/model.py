"""
CTC-based Audio-to-Text Model

Key differences from seq2seq Transformer:
- Encoder only (no decoder)
- CTC loss handles alignment automatically
- No autoregressive decoding = no language model hallucination
- Better for arbitrary sequences like call signs (W1ABC)
- Faster inference (single forward pass)

Architecture:
    Audio → CNN → BiLSTM/Transformer Encoder → Linear → CTC Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


class ConvFrontend(nn.Module):
    """
    CNN frontend to reduce time dimension and extract local features.
    Similar to Wav2Vec2/Whisper front-end.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Conv layers with stride to reduce time dimension
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, freq) mel spectrogram
            
        Returns:
            output: (batch, time//4, output_dim) features
            lengths: adjusted lengths after striding
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
        # Two stride-2 convs
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
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "transformer",  # "transformer" or "lstm"
        nhead: int = 4
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
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
    
    Output is log probabilities over vocab + blank token at each timestep.
    CTC loss handles alignment during training.
    """
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
        nhead: int = 4
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank_idx = 0  # CTC blank token (index 0)
        
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
        
        # Output projection (vocab_size includes blank at index 0)
        self.output_proj = nn.Linear(self.encoder.output_dim, vocab_size)
    
    def forward(
        self, 
        audio: torch.Tensor, 
        audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio: (batch, time, freq) mel spectrogram
            audio_lengths: (batch,) frame counts
            
        Returns:
            log_probs: (batch, time', vocab_size) log probabilities
            output_lengths: (batch,) output time lengths
        """
        # CNN frontend (reduces time dimension)
        x = self.frontend(audio)
        output_lengths = self.frontend.get_output_lengths(audio_lengths)
        
        # Encoder
        x = self.encoder(x, output_lengths)
        
        # Project to vocab
        logits = self.output_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, output_lengths
    
    def decode_greedy(
        self, 
        audio: torch.Tensor, 
        audio_lengths: torch.Tensor
    ) -> List[List[int]]:
        """
        Greedy CTC decoding (argmax + collapse).
        
        Args:
            audio: (batch, time, freq)
            audio_lengths: (batch,)
            
        Returns:
            List of decoded token sequences (without blanks/repeats)
        """
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(audio, audio_lengths)
            
            # Argmax at each timestep
            predictions = log_probs.argmax(dim=-1)  # (batch, time)
            
            # Collapse: remove blanks and repeated tokens
            batch_decoded = []
            for i in range(predictions.size(0)):
                length = output_lengths[i].item()
                pred_seq = predictions[i, :length].tolist()
                
                # Collapse repeats and remove blanks
                decoded = []
                prev_token = None
                for token in pred_seq:
                    if token != self.blank_idx and token != prev_token:
                        decoded.append(token)
                    prev_token = token
                
                batch_decoded.append(decoded)
            
            return batch_decoded
    
    def decode_beam(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        beam_width: int = 10
    ) -> List[List[int]]:
        """
        Beam search CTC decoding (prefix beam search).
        
        More accurate than greedy but slower.
        """
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(audio, audio_lengths)
            
            batch_decoded = []
            for i in range(log_probs.size(0)):
                length = output_lengths[i].item()
                probs = log_probs[i, :length].exp()  # (time, vocab)
                
                decoded = self._prefix_beam_search(probs, beam_width)
                batch_decoded.append(decoded)
            
            return batch_decoded
    
    def _prefix_beam_search(
        self, 
        probs: torch.Tensor, 
        beam_width: int
    ) -> List[int]:
        """
        Prefix beam search for single sequence.
        
        Based on: https://distill.pub/2017/ctc/
        """
        T, V = probs.shape
        
        # Beams: (prefix_tuple, (prob_blank, prob_non_blank))
        beams = {(): (1.0, 0.0)}
        
        for t in range(T):
            new_beams = {}
            
            for prefix, (p_b, p_nb) in beams.items():
                # Probability of this prefix so far
                p_prefix = p_b + p_nb
                
                # Extend with blank
                new_p_b = p_prefix * probs[t, self.blank_idx].item()
                key = prefix
                if key in new_beams:
                    new_beams[key] = (new_beams[key][0] + new_p_b, new_beams[key][1])
                else:
                    new_beams[key] = (new_p_b, 0.0)
                
                # Extend with each non-blank token
                for c in range(1, V):
                    p_c = probs[t, c].item()
                    
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same as last char: only extend if previous was blank
                        new_p_nb = p_b * p_c
                    else:
                        # Different char: can extend from any
                        new_p_nb = p_prefix * p_c
                    
                    new_prefix = prefix + (c,)
                    if new_prefix in new_beams:
                        new_beams[new_prefix] = (
                            new_beams[new_prefix][0],
                            new_beams[new_prefix][1] + new_p_nb
                        )
                    else:
                        new_beams[new_prefix] = (0.0, new_p_nb)
            
            # Prune to beam width
            beams = dict(sorted(
                new_beams.items(),
                key=lambda x: x[1][0] + x[1][1],
                reverse=True
            )[:beam_width])
        
        # Return best beam
        best_prefix = max(beams.keys(), key=lambda x: beams[x][0] + beams[x][1])
        return list(best_prefix)


def create_ctc_model(config: dict, vocab_size: int) -> CTCModel:
    """
    Create CTC model from config.
    
    Config structure:
        model:
            hidden_dim: 256
            num_layers: 4
            dropout: 0.1
            encoder_type: "transformer"  # or "lstm"
            nhead: 4
        audio:
            n_mels: 80
    """
    model_cfg = config.get('model', {})
    audio_cfg = config.get('audio', {})
    
    return CTCModel(
        vocab_size=vocab_size,
        input_dim=audio_cfg.get('n_mels', 80),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_layers=model_cfg.get('num_layers', 4),
        dropout=model_cfg.get('dropout', 0.1),
        encoder_type=model_cfg.get('encoder_type', 'transformer'),
        nhead=model_cfg.get('nhead', 4)
    )

